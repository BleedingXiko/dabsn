"""Continuity with the paper-1 architecture experiments: fresh matched online/BPTT runs.

The full DABSN model (core, admitted read, long memory) trains on the original
paper-1 tasks and regimes, through the public online API (``train_step`` with an
``OnlineTemporalState``) or full BPTT. Both methods share the model factory,
data, rates and evaluation namespace. Historical values are not reused.

The A5 cohort is separate: online credit at the frozen rate, seed 0 as the
development seed and seeds 1-3 held out. Every measured outcome is kept.
"""
from __future__ import annotations

import json
import math
import random
import time

import torch
from torch.nn import functional as F
from dabsn.distributed import clip_grad_norm
from dabsn.runtime import train_step
from dabsn.temporal_credit import initial_online_temporal_state
from dabsn.reproductions import common as paper1

import harness
import models

NEEDS_NATIVE = True
PROTOCOL = None


def jobs(protocol):
    methods=[*protocol["methods"],*protocol.get("control_methods",[])]
    paired=[{"task":task,"seed":seed,"method":method}
            for task in protocol["suite_tasks"] for seed in protocol["seeds"] for method in methods]
    cohort=[{"task":"a5","seed":seed,"method":"online"}
            for seed in [protocol["a5_development_seed"],*protocol["a5_heldout_seeds"]]]
    return paired+cohort


def identifier(job):
    return f"{job['task']}-s{job['seed']}-{job['method']}"


def role(job):
    if job["task"]!="a5": return "paired"
    return "development" if job["seed"]==PROTOCOL["a5_development_seed"] else "heldout"


def precision_for(job):
    """The A5 cohort runs FP32 compute."""
    return "fp32" if job["task"]=="a5" else None


def configuration_for(job):
    cfg=dict(PROTOCOL["tasks"][job["task"]])
    if job["task"]=="a5" and job["method"]=="online":
        cfg["lr"]=PROTOCOL["a5_frozen_online_lr"]
    return cfg


def model_for(task,cfg,device):
    if task=="copy":
        net=paper1.model(cfg["vocab"]+1,cfg["vocab"],cfg["hidden"],PROTOCOL["geometry"],device)
    elif task in ("mqar","keyvalue"):
        net=paper1.model(3+cfg["keys"]+cfg["values"],cfg["values"],cfg["hidden"],
                         PROTOCOL["geometry"],device)
    else:
        net=paper1.model(60,60,cfg["hidden"],PROTOCOL["geometry"],device)
    count=sum(parameter.numel() for parameter in net.parameters())
    if any(parameter.dtype!=torch.float32 for parameter in net.parameters()):
        raise ValueError("Continuity requires FP32 master parameter storage")
    if count!=cfg["current_parameter_count"]:
        raise ValueError(f"Continuity architecture parameter count differs: {count}")
    return net


def task_batch(task,cfg,batch,length,device,table=None,identity=None):
    if task=="copy":
        x,targets=paper1.copy_batch(batch,length,cfg["vocab"],device)
    elif task=="mqar":
        x,targets,mask=paper1.mqar_batch(batch,length,cfg["keys"],cfg["values"],
                                        cfg["pairs"],torch.device("cpu"))
        x=x.to(device)
        targets=targets.masked_fill(~mask,-100).to(device)
    elif task=="keyvalue":
        x,last_target=paper1.keyvalue_batch(batch,length,cfg["keys"],cfg["values"],
            cfg["pairs"],cfg["filler_probability"],cfg["ordered_pairs"],torch.device("cpu"))
        x=x.to(device); last_target=last_target.to(device)
        targets=torch.full((batch,length),-100,dtype=torch.long,device=device)
        targets[:,-1]=last_target
    elif task=="a5":
        x,targets=paper1.a5_batch(batch,length,table,identity,device)
    else: raise ValueError("Unknown task")
    if x.shape[:2]!=targets.shape: raise AssertionError("Input/target episode shape mismatch")
    return x,targets


def objective(total_steps,total_valid):
    """Compensate runtime chunk weighting to equal full-sequence CE exactly."""
    if total_valid<=0: raise ValueError("No supervised target in episode")
    def loss_fn(logits,targets):
        chunk=targets.shape[1]
        summed=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                               targets.reshape(-1),ignore_index=-100,reduction="sum")
        return summed*(total_steps/chunk)/total_valid
    return loss_fn


def train_length(task,cfg,step):
    if task!="a5": return cfg["train_length"]
    lengths=[]; length=max(2,cfg["train_len_min"])
    while length<cfg["train_length"]:
        lengths.append(length); length*=2
    lengths.append(cfg["train_length"])
    unlocked=(max(1,math.ceil(min(1.,step/cfg["curriculum_steps"])*len(lengths)))
              if cfg["curriculum_steps"]>0 else len(lengths))
    return lengths[int(torch.randint(0,unlocked,(1,),device="cpu").item())]


def scheduled_lr(cfg,step):
    warmup=cfg["warmup_steps"]; base=cfg["lr"]; total=cfg["steps"]
    if warmup>0 and step<=warmup: return base*step/warmup
    if cfg["lr_schedule"]!="cosine" or total<=warmup: return base
    minimum=cfg["lr_min_frac"]
    progress=min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
    return base*(minimum+(1.-minimum)*.5*(1.+math.cos(math.pi*progress)))


def evaluation_batch_size(task,cfg,length):
    return cfg["eval_batch_by_length"][str(length)]


def regime_record(task,cfg):
    steps=sorted({0,1,cfg["warmup_steps"],cfg["warmup_steps"]+1,
                  (cfg["steps"]+cfg["warmup_steps"])//2,cfg["steps"]})
    return {"task":task,"protocol":PROTOCOL["protocol"],"configuration":cfg,
            "optimizer":{"name":PROTOCOL["optimizer"],"betas":PROTOCOL["betas"],
                         "eps":PROTOCOL["eps"],"weight_decay":PROTOCOL["weight_decay"],
                         "clip_norm":PROTOCOL["clip_norm"]},
            "lr_samples":[{"step":step,"lr":scheduled_lr(cfg,step)} for step in steps],
            "original_source_sha256":harness.digest_value(harness.MATRIX["continuity_regime_sources"]),
            "comparison_scope":PROTOCOL["comparison_scope"]}


def original_probe_draw(task,cfg,step,device,table,identity):
    """Preserve the original diagnostics' random draws without extra forwards."""
    if step!=1 and step%cfg["diagnostic_every"]!=0: return
    if task=="copy":
        paper1.copy_batch(min(cfg["batch"],16),max(cfg["eval_lengths"]),cfg["vocab"],device)
    elif task=="a5":
        paper1.a5_batch(min(cfg["batch"],8),cfg["train_length"],table,identity,device)
        # Original eval_pairwise uses eight batches over all 60*60 pairs.
        # Derive its cursor from the absolute step for exact checkpoint resume.
        pairs=60*60; batches=8
        diagnostic_index=0 if step==1 else step//cfg["diagnostic_every"]
        cursor=(diagnostic_index*batches*cfg["batch"])%pairs
        for _ in range(batches):
            if cursor==0: torch.randperm(pairs,device="cpu")
            cursor=(cursor+cfg["batch"])%pairs


def evaluate(net,task,cfg,seed,device,table,identity):
    net.eval(); results=[]
    task_index=list(PROTOCOL["tasks"]).index(task)
    for length in cfg["eval_lengths"]:
        paper1.seed_all(900000+10000*seed+100*task_index+length)
        losses=[]; accuracies=[]
        with torch.no_grad():
            for _ in range(cfg["val_batches"]):
                x,targets=task_batch(task,cfg,evaluation_batch_size(task,cfg,length),length,
                                     device,table,identity)
                with harness.autocast(device):
                    logits=net.forward_sequence(x)
                loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                                     targets.reshape(-1),ignore_index=-100)
                valid=targets!=-100
                accuracy=float(((logits.argmax(-1)==targets)&valid).sum()/valid.sum())
                losses.append(float(loss)); accuracies.append(accuracy)
        results.append({"eval_length":length,"eval_loss":sum(losses)/len(losses),
                        "eval_accuracy":sum(accuracies)/len(accuracies),
                        "evaluation_batches":cfg["val_batches"],
                        "evaluation_batch_size":evaluation_batch_size(task,cfg,length)})
    return results


def one_update(net,optimizer,method,x,targets,dtype_observations=None,frozen=None):
    """The exact paper-2 comparison update used by training and preflight."""
    if x.dtype!=torch.float32:
        raise ValueError("Continuity requires FP32 input storage")
    valid=int((targets!=-100).sum())
    loss_fn=objective(x.shape[1],valid)
    if method=="online":
        state=initial_online_temporal_state(net,x.shape[0],
            stream_policy="reset_each_batch",stream_ids=tuple(range(x.shape[0])),
            device=x.device,dtype=torch.float32,eligibility_dtype=torch.float32)
        # The native online API owns backward and disables autocast for its
        # analytic replay. Parameters, recurrent storage and eligibility stay FP32.
        with harness.autocast(x.device):
            loss,completed=train_step(net,x,targets,optimizer,loss_fn=loss_fn,
                temporal_state=state,temporal_chunk_size=PROTOCOL["online_chunk"],
                clip_grad_norm=PROTOCOL["clip_norm"])
        completed.assert_checkpointable()
        if dtype_observations is not None:
            recurrent={str(value.dtype) for layer in completed.layers
                       for value in (*layer.core_recurrent,*layer.long_recurrent)}
            eligibility={str(value.dtype) for layer in completed.layers
                         for value in (*layer.core_eligibility.tensors(),layer.long_eligibility.traces)}
            if recurrent!={"torch.float32"} or eligibility!={"torch.float32"}:
                raise AssertionError("Continuity recurrent or eligibility storage changed precision")
            dtype_observations.append({"method":method,"recurrent_dtypes":sorted(recurrent),
                                       "eligibility_dtypes":sorted(eligibility),"loss_dtype":"torch.float32"})
    elif method=="frozen_core":
        # The same online computation, one chunk per episode, with the core's
        # gradients (which arrive only through E) dropped before clipping.
        if frozen is None: raise ValueError("frozen_core requires a FrozenCore")
        valid=int((targets!=-100).sum())
        loss,_=models.net_chunked_update(net,optimizer,x,targets,"online",x.shape[1],
            PROTOCOL["clip_norm"],objective(x.shape[1],valid),frozen=frozen)
    elif method in ("online_chunked","reset","permute"):
        valid=int((targets!=-100).sum())
        loss,_=models.net_chunked_update(net,optimizer,x,targets,
            "online" if method=="online_chunked" else method,models.online_chunk(PROTOCOL["control_chunk"],x),
            PROTOCOL["clip_norm"],objective(x.shape[1],valid))
    elif method=="bptt":
        with harness.autocast(x.device):
            logits=net.forward_sequence(x)
        loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                             targets.reshape(-1),ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(net,PROTOCOL["clip_norm"])
        optimizer.step()
        if dtype_observations is not None:
            dtype_observations.append({"method":method,"eligibility_dtypes":[],
                                       "eligibility_scope":"No eligibility state in BPTT",
                                       "loss_dtype":str(loss.dtype)})
        loss=float(loss.detach())
    else:
        raise ValueError("Unknown continuity method")
    if not math.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
    return loss


def run(job, ctx):
    global PROTOCOL
    PROTOCOL=ctx.protocol; device=ctx.device; directory=ctx.directory
    task=job["task"]; cfg=configuration_for(job)
    paper1.seed_all(job["seed"])
    net=model_for(task,cfg,torch.device(device))
    initial_parameters_sha256=harness.parameter_digest(net)
    frozen=models.FrozenCore(models._NetView(net)) if job["method"]=="frozen_core" else None
    optimizer=torch.optim.AdamW(frozen.trainable(net) if frozen else net.parameters(),lr=cfg["lr"],
        betas=tuple(PROTOCOL["betas"]),eps=PROTOCOL["eps"],
        weight_decay=PROTOCOL["weight_decay"])
    constructed_groups=[{"lr":float(group["lr"]),"betas":list(group["betas"]),
        "eps":group["eps"],"weight_decay":group["weight_decay"],
        "capturable":group["capturable"],"fused":group["fused"]}
        for group in optimizer.param_groups]
    for group in constructed_groups:
        if (group["lr"]!=cfg["lr"] or group["betas"]!=PROTOCOL["betas"] or
            group["eps"]!=PROTOCOL["eps"] or group["weight_decay"]!=PROTOCOL["weight_decay"]):
            raise ValueError("Constructed optimizer differs from continuity regime")
    regime=regime_record(task,cfg)
    precision=harness.precision_record(device)
    regime["precision"]=precision
    regime["constructed_optimizer_groups"]=constructed_groups
    regime["current_parameter_count"]=sum(p.numel() for p in net.parameters())
    regime_path=directory/"regime.json"
    if regime_path.exists():
        if json.loads(regime_path.read_text())!=regime:
            raise ValueError("Saved continuity regime differs")
    table,group_identity=paper1.a5_table() if task=="a5" else (None,None)
    start_step=0; last_loss=None
    saved=ctx.latest() if ctx.resume else None
    if saved is not None:
        net.load_state_dict(saved["model"])
        harness.restore_optimizer_state(optimizer,saved["optimizer"])
        random.setstate(saved["python_random"])
        torch.set_rng_state(saved["torch_random"].cpu().clone())
        if device=="cuda":
            torch.cuda.set_rng_state_all([state.cpu().clone() for state in saved["cuda_random"]])
        start_step=saved["completed"]
        last_loss=saved["last_training_loss"]

    def payload(step):
        return {"completed":step,"model":net.state_dict(),"optimizer":optimizer.state_dict(),
                "python_random":random.getstate(),"torch_random":torch.get_rng_state(),
                "cuda_random":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "last_training_loss":last_loss}

    if not regime_path.exists(): harness.write_record(regime_path,regime)
    started=time.monotonic(); net.train()
    for step in range(start_step+1,cfg["steps"]+1):
        for group in optimizer.param_groups: group["lr"]=scheduled_lr(cfg,step)
        length=train_length(task,cfg,step)
        x,targets=task_batch(task,cfg,cfg["batch"],length,torch.device(device),
                             table,group_identity)
        last_loss=one_update(net,optimizer,job["method"],x,targets,frozen=frozen)
        original_probe_draw(task,cfg,step,torch.device(device),table,group_identity)
        if step%cfg["checkpoint_every"]==0 or step==cfg["steps"]:
            ctx.checkpoint(ctx.step_name(step),payload(step))
        harness.maybe_stop(ctx,lambda: ctx.checkpoint("recovery",payload(step),replace=True))
    elapsed=time.monotonic()-started
    evaluation=evaluate(net,task,cfg,job["seed"],torch.device(device),table,group_identity)
    return {"configuration":cfg,"method":job["method"],"seed":job["seed"],
            "initial_parameters_sha256":initial_parameters_sha256,
            "training_updates":cfg["steps"],"last_training_loss":last_loss,
            "training_seconds_this_attempt":elapsed,"evaluation":evaluation,
            "parameter_count":sum(p.numel() for p in net.parameters()),
            "execution_completed":True,
            "a5_exact_reproduction_passed":all(row["eval_accuracy"]==1. for row in evaluation) if task=="a5" else None,
            "role":role(job),
            "frozen_core_bitwise_unchanged":frozen.verify() if frozen else None,
            "checkpoint_sha256":ctx.checkpoint_hashes(),
            "regime":regime,"precision":precision,"comparison_scope":PROTOCOL["comparison_scope"]}
