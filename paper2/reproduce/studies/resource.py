"""Whole-update resource measurement with streamed or resident inputs.

Time and CUDA allocator peaks for one complete update (forward, credit, clipping,
Adam) for online credit, TBPTT and BPTT. For online credit, the allocated E bytes
must equal the B-channel formula B * (2*S*I + 2*S^2 + 60*S) * 4 exactly.
"""
from __future__ import annotations

import itertools
import math
import statistics
import time

import torch
from torch.nn import functional as F
from dabsn.distributed import clip_grad_norm
from dabsn.temporal_credit import OnlineCoreFrame, initial_core_eligibility

import harness
import models

NEEDS_NATIVE = True
PROTOCOL = None


def jobs(protocol):
    return [{"input":shape[0],"state":shape[1],"length":length,"method":method,"input_policy":policy}
            for shape,length,method,policy in itertools.product(
                protocol["shapes"],protocol["lengths"],protocol["methods"],protocol["input_policies"])]


def identifier(job):
    return f"i{job['input']}-s{job['state']}-l{job['length']}-{job['method']}-{job['input_policy']}"


def cpu_chunk(batch,steps,inputs,seed,index):
    rng=torch.Generator(device="cpu").manual_seed(8000+10007*seed+1000003*index)
    return torch.randn(batch,steps,inputs,generator=rng)*.1


def chunks(job,device,repetition,resident=None):
    batch=PROTOCOL["batch"]; width=PROTOCOL["chunk"]
    for index,start in enumerate(range(0,job["length"],width)):
        stop=min(job["length"],start+width)
        if resident is None:
            yield cpu_chunk(batch,stop-start,job["input"],
                            PROTOCOL["seed"]+1000*repetition,index).to(device)
        else:
            yield resident[:,start:stop]


def resident_inputs(job,device,repetition):
    parts=[cpu_chunk(PROTOCOL["batch"],min(PROTOCOL["chunk"],job["length"]-start),
                     job["input"],PROTOCOL["seed"]+1000*repetition,index)
           for index,start in enumerate(range(0,job["length"],PROTOCOL["chunk"]))]
    return torch.cat(parts,dim=1).to(device)


def one_update(model,optimizer,job,device,repetition,resident):
    method=job["method"]
    state=model.core.initial_state(PROTOCOL["batch"],device=device,dtype=torch.float32)
    eligibility=(initial_core_eligibility(model.core,PROTOCOL["batch"],
        device=device,dtype=torch.float32) if method=="online" else None)
    optimizer.zero_grad(set_to_none=True)
    labels=torch.arange(PROTOCOL["batch"],device=device)%4
    last=None
    steps=math.ceil(job["length"]/PROTOCOL["chunk"])
    for index,x in enumerate(chunks(job,device,repetition,resident)):
        if not hasattr(model,"_paper2_projection_precision"):
            model._paper2_projection_precision=harness.assert_projection_precision(model,x)
        frame=OnlineCoreFrame(eligibility) if eligibility is not None else None
        with harness.autocast(device):
            output,state=model.core.forward_from_state(x,initial_state=state,
                return_final_state=True,temporal_credit="online" if frame is not None else "bptt",
                eligibility_state=eligibility,online_frame=frame)
        state=tuple(value.float() for value in state)
        last=index==steps-1
        if last:
            with harness.autocast(device):
                logits=model.head(torch.tanh(output[0][:,-1,model.core.hidden_dim:]))
            loss=F.cross_entropy(logits.float(),labels)
            harness.finite(loss,"nonfinite resource-study loss")
            loss.backward()
        elif frame is not None:
            (output[0].sum()*0).backward()
        if frame is not None:
            if not frame.replayed or frame.resulting_eligibility is None:
                raise RuntimeError("Online eligibility failed to advance")
            eligibility=frame.resulting_eligibility
        if method!="bptt": state=tuple(v.detach() for v in state)
    if not last: raise AssertionError("No terminal chunk")
    preclip=float(clip_grad_norm(model,PROTOCOL["clip_norm"]))
    if not math.isfinite(preclip): raise FloatingPointError("Nonfinite preclip norm")
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite updated parameter")
    return {"loss":float(loss.detach()),"pre_clip_norm":preclip,"precision":harness.precision_record(device),
            "projection_precision":model._paper2_projection_precision,
            "eligibility_allocated_bytes":eligibility.nbytes() if eligibility is not None else 0,
            "backend":getattr(model.core,"_last_core_backend",None)}


def optimizer_bytes(optimizer):
    return sum(value.numel()*value.element_size() for state in optimizer.state.values()
               for value in state.values() if isinstance(value,torch.Tensor))


def run(job, ctx):
    global PROTOCOL
    PROTOCOL=ctx.protocol; device=ctx.device
    runtime=ctx.runtime
    model=models.Model(job["input"],job["state"],4,PROTOCOL["seed"],device)
    optimizer=torch.optim.Adam(model.parameters(),lr=PROTOCOL["lr"],
        betas=tuple(PROTOCOL["betas"]),weight_decay=PROTOCOL["weight_decay"])
    proof_x=cpu_chunk(PROTOCOL["batch"],min(PROTOCOL["chunk"],job["length"]),
                     job["input"],PROTOCOL["seed"],0).to(device)
    model._paper2_projection_precision=harness.assert_projection_precision(model,proof_x)
    del proof_x
    runtime.synchronize()
    rows=[]; first=None
    for repetition in range(PROTOCOL["warmup"]+PROTOCOL["measurements"]):
        runtime.synchronize(); runtime.reset_peak()
        started=time.perf_counter()
        resident=(resident_inputs(job,device,repetition)
                  if job["input_policy"]=="resident" else None)
        measured=one_update(model,optimizer,job,device,repetition,resident)
        runtime.synchronize()
        elapsed=time.perf_counter()-started
        allocated,reserved=runtime.peaks()
        if first is None: first=elapsed
        if repetition>=PROTOCOL["warmup"]:
            rows.append({"seconds":elapsed,"peak_allocated_bytes":allocated,
                         "peak_reserved_bytes":reserved,**measured})
        del resident
    formula=PROTOCOL["batch"]*(2*job["state"]*job["input"]+
            2*job["state"]**2+60*job["state"])*4
    if job["method"]=="online" and any(
        row["eligibility_allocated_bytes"]!=formula for row in rows):
        raise AssertionError("Actual eligibility allocation differs from formula")
    return {"first_update_seconds":first,"replicates":rows,"precision":harness.precision_record(device),
            "projection_precision":model._paper2_projection_precision,
            "median_update_seconds":statistics.median(r["seconds"] for r in rows),
            "parameters_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
            "optimizer_state_bytes":optimizer_bytes(optimizer),
            "eligibility_bytes_formula":formula if job["method"]=="online" else 0,
            "resident_input_bytes_formula":(PROTOCOL["batch"]*job["length"]*job["input"]*4
                if job["input_policy"]=="resident" else 0),
            "one_streamed_chunk_input_bytes_formula":PROTOCOL["batch"]*PROTOCOL["chunk"]*job["input"]*4,
            "input_policy":job["input_policy"],"device":device,
            "memory_scope":"CUDA allocator peaks if CUDA; unavailable if CPU; input/eligibility formulas separate"}
