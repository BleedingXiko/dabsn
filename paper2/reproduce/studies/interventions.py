"""Causal intervention on E at a frozen checkpoint of the central study.

At the start, middle and end of each online main-study run (updates 0, 1500, 4500),
paired episodes are processed up to a boundary 1 or 32 steps before the query. E is
then carried, zeroed, rolled across lanes, sign-flipped on its W trace, or
stripped of its local-control traces. Every branch shares the exact pre-update
weights, recurrent state, suffix and target, so only the credit differs. Each
branch records its core credit against the carried branch (cosine and norm ratio, in
total and by family), takes one true update (a clone of the source Adam state,
clipped at 1.0) and one equal-norm SGD diagnostic, and is evaluated on fresh
held-out episodes.
At the 1-step boundary the final input is also zeroed in a separate check, so W's credit is
purely historical: carried, W-trace-negated and erased E must give W credit, its negation, and zero.
"""
from __future__ import annotations

import copy
import itertools
import json
import math

import torch
from torch.nn import functional as F
from dabsn.distributed import clip_grad_norm
from dabsn.temporal_credit import OnlineCoreFrame

import credit
import harness
import models
from credit import CoreEligibilityState, initial_core_eligibility
from studies import delayed, delayed_tasks

NEEDS_NATIVE = True
PROTOCOL_MAIN = None
INTERVENTION = None
TOLERANCE = None
CORE_FAMILIES = ("W", "A", "Ug", "Wg", "local_controls", "shared_controls")


def jobs(protocol, rates):
    return [{"task":task,"seed":seed,"checkpoint_step":step,"length":length,"boundary_suffix":suffix,
             "source_lr":rates[task]}
            for task,seed,step,length,suffix in itertools.product(
                protocol["tasks"],protocol["training_seeds"],protocol["checkpoint_steps"],protocol["lengths"],
                protocol["boundary_suffixes"])]


def identifier(job):
    return f"{job['task']}-s{job['seed']}-c{job['checkpoint_step']}-l{job['length']}-b{job['boundary_suffix']}"


def source_model(job, ctx):
    """Load the online main-study model and optimizer exactly as saved at the checkpoint."""
    rate=job["source_lr"]
    main_job={"phase":"main","task":job["task"],"seed":job["seed"],"method":"online","lr":rate}
    name=delayed.identifier(main_job)
    directory,record=ctx.dependency("delayed",name)
    if record["job"]!=main_job:
        raise ValueError("Source online training job differs")
    precision=record["result"]["precision"]
    if precision!=harness.precision_record(ctx.device):
        raise ValueError("Intervention compute policy differs from source training")
    model=models.Block(PROTOCOL_MAIN["input_dim"],PROTOCOL_MAIN["state_dim"],
                       delayed_tasks.output_dim(job["task"]),job["seed"],ctx.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=rate,
        betas=tuple(PROTOCOL_MAIN["betas"]),weight_decay=PROTOCOL_MAIN["weight_decay"])
    step=job["checkpoint_step"]
    if step==0:
        if harness.parameter_digest(model)!=record["result"]["initial_parameters_sha256"]:
            raise ValueError("Initial parameters differ from the recorded training run")
        checkpoint_sha=None
    else:
        path=directory/f"step-{step:06d}.pt"
        checkpoint_sha=harness.sha256(path)
        if record["result"]["checkpoint_sha256"].get(path.name)!=checkpoint_sha:
            raise ValueError("Source checkpoint checksum differs")
        saved=torch.load(path,map_location=ctx.device,weights_only=True)
        identity=saved["identity"]
        if identity["job"]!=main_job or identity["device"]!=ctx.device or identity["precision"]!=precision:
            raise ValueError("Source checkpoint identity differs")
        model.load_state_dict(saved["model"])
        harness.restore_optimizer_state(optimizer,saved["optimizer"])
    if any(group["lr"]!=rate for group in optimizer.param_groups):
        raise ValueError("Restored optimizer rate differs from the selected rate")
    attempt=sorted(directory.glob("attempt-*.json"))[-1]
    return model,optimizer,{"job_name":name,"attempt_sha256":harness.sha256(attempt),
                            "checkpoint_step":step,"checkpoint_sha256":checkpoint_sha,
                            "selected_lr":rate,"precision":precision}


def prefix(model,x,boundary):
    """Carry the block's online state and credit through the episode up to the boundary."""
    chunk=PROTOCOL_MAIN["chunk"]
    state=models.block_online_state(model,x.shape[0],chunk,x.device)
    for start in range(0,boundary,chunk):
        with harness.autocast(x.device):
            output,pending=model.net.forward_sequence_online(x[:,start:min(start+chunk,boundary)],state)
        (output.float().sum()*0).backward()
        state=models.finalize_online_temporal_state(pending)
    model.zero_grad(set_to_none=True)
    return state


def change_eligibility(model,state,condition):
    """The edited branch state; forward state (recurrence, read bank) is identical in every branch."""
    if condition=="carry":
        return models._replace_eligibility(state,lambda c,l:(credit.rebuild(v.clone() for v in c.tensors()),
                                                              models.LongReadEligibilityState(l.traces.clone())))
    if condition=="zero":
        return models.erase_credit(model,state)
    if condition=="permute_lanes":
        return models.roll_credit(state)
    if condition=="negate_w":
        return models._replace_eligibility(state,lambda c,l:(credit.negate_w(c),l))
    if condition=="erase_local":
        return models._replace_eligibility(state,lambda c,l:(credit.erase_local(c),l))
    raise ValueError("Unknown eligibility intervention")


def core_vector(gradients, families):
    parts=[value.detach().flatten().double() for name,value in gradients if credit.family(name) in families]
    return torch.cat(parts) if parts else torch.zeros(0,dtype=torch.float64)


def cosine(a, b):
    na,nb=float(a.norm()),float(b.norm())
    return float(a@b)/(na*nb) if na>0 and nb>0 else None


def credit_against_carry(gradients, carry):
    """Core credit of a branch against the carried branch: cosine and norm ratio, total and by family."""
    total,reference=core_vector(gradients,CORE_FAMILIES),core_vector(carry,CORE_FAMILIES)
    by_family={}
    for name in CORE_FAMILIES:
        a,b=core_vector(gradients,(name,)),core_vector(carry,(name,))
        by_family[name]={"cosine":cosine(a,b),"norm_ratio":float(a.norm())/float(b.norm()) if float(b.norm())>0 else None}
    return {"core_cosine":cosine(total,reference),
            "core_norm_ratio":float(total.norm())/float(reference.norm()) if float(reference.norm())>0 else None,
            "by_family":by_family}


def family_norms(named_values):
    squared={}
    for name,value in named_values:
        owner=credit.family(name)
        squared[owner]=squared.get(owner,0.)+float(value.detach().float().square().sum())
    return {name:math.sqrt(value) for name,value in squared.items()}


def terminal_loss(logits,y,task):
    loss=(F.cross_entropy(logits.float(),y) if delayed_tasks.objective(task)=="classification"
          else F.mse_loss(logits.float(),y.float()))
    harness.finite(loss,"nonfinite intervention loss")
    return loss


def new_model(task,device,pre_state):
    model=models.Block(PROTOCOL_MAIN["input_dim"],PROTOCOL_MAIN["state_dim"],delayed_tasks.output_dim(task),0,device)
    model.load_state_dict(pre_state)
    return model


def immediate_credit(pre_state,x,y,state,task,boundary):
    """Immediate-only credit: the same forward, with all carried credit erased before the final step."""
    model=new_model(task,x.device,pre_state)
    if x.shape[1]-1>boundary:
        with harness.autocast(x.device):
            output,pending=model.net.forward_sequence_online(x[:,boundary:-1],state)
        (output.float().sum()*0).backward()
        state=models.finalize_online_temporal_state(pending)
        model.zero_grad(set_to_none=True)
    state=models.erase_credit(model,state)
    with harness.autocast(x.device):
        output,_=model.net.forward_sequence_online(x[:,-1:],state)
    logits=output[:,-1].float()
    loss=terminal_loss(logits,y,task)
    loss.backward()
    return {"loss":float(loss.detach()),"family_credit_norms":family_norms(
        (name,p.grad if p.grad is not None else torch.zeros_like(p))
        for name,p in model.named_parameters()),"logits":logits.detach()}


def isolated_w_pathway(pre_state,x,y,state,task):
    """Proposition (erased isolated historical pathway) at the checkpoint: with the final input zeroed,
    W's current-step credit vanishes, so its credit is purely historical. Carried, negated-W-trace and
    zeroed E share one forward; negation must flip W's credit and erasure must remove it."""
    zeroed=torch.zeros_like(x[:,-1:])
    credit_w={}; reference=None
    for condition in ("carry","negate_w","zero"):
        model=new_model(task,x.device,pre_state)
        with harness.autocast(x.device):
            output,_=model.net.forward_sequence_online(zeroed,change_eligibility(model,state,condition))
        logits=output[:,-1].float()
        if reference is None: reference=logits.detach()
        if float((logits.detach()-reference).abs().max())>1e-6:
            raise AssertionError("Isolated-pathway branches changed forward computation")
        terminal_loss(logits,y,task).backward()
        credit_w[condition]=core_vector([(n,p.grad if p.grad is not None else torch.zeros_like(p))
                                         for n,p in model.named_parameters()],("W",))
    carry=credit_w["carry"]
    return {"w_credit_norm":{k:float(v.norm()) for k,v in credit_w.items()},
            "negate_w_cosine_to_carry":cosine(credit_w["negate_w"],carry),
            "zero_norm_ratio_to_carry":float(credit_w["zero"].norm())/float(carry.norm()) if float(carry.norm())>0 else None}


def fork_batch(base,optimizer_state,x,y,task,rate,post_seed,post_start,suffix):
    boundary=x.shape[1]-suffix
    base.zero_grad(set_to_none=True)
    pre_state=copy.deepcopy(base.state_dict())
    state=prefix(new_model(task,x.device,pre_state),x,boundary)
    rows=[]; reference=None; carry_norm=None; carry_gradients=None
    for condition in INTERVENTION["conditions"]:
        model=new_model(task,x.device,pre_state)
        optimizer=torch.optim.Adam(model.parameters(),lr=rate,betas=tuple(PROTOCOL_MAIN["betas"]),
                                   weight_decay=PROTOCOL_MAIN["weight_decay"])
        harness.restore_optimizer_state(optimizer,copy.deepcopy(optimizer_state))
        changed=change_eligibility(model,state,condition)
        with harness.autocast(x.device):
            output,_=model.net.forward_sequence_online(x[:,boundary:],changed)
        logits=output[:,-1].float()
        loss=terminal_loss(logits,y,task)
        if reference is None:
            reference=(output.detach().float(),logits.detach(),loss.detach())
        forward_error=max(float((output.detach().float()-reference[0]).abs().max()),
                          float((logits.detach()-reference[1]).abs().max()),
                          float((loss.detach()-reference[2]).abs()))
        if forward_error>1e-6:
            raise AssertionError("Eligibility branch changed pre-update forward computation")
        loss.backward()
        gradients=[(name,p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                   for name,p in model.named_parameters()]
        credit_by_family=family_norms(gradients)
        total_norm=math.sqrt(sum(value*value for value in credit_by_family.values()))
        if condition=="carry": carry_norm=total_norm; carry_gradients=gradients
        if carry_norm is None: raise RuntimeError("Carry branch must be evaluated first")
        before={name:p.detach().clone() for name,p in model.named_parameters()}
        preclip=float(clip_grad_norm(model.net,PROTOCOL_MAIN["clip_norm"]))
        if not math.isfinite(preclip): raise FloatingPointError("Nonfinite intervention gradient")
        models.apply_optimizer_step(model.net,optimizer)
        changed_by_family=family_norms((name,p.detach()-before[name]) for name,p in model.named_parameters())
        real_post=delayed_tasks.delayed_evaluate(model,task,post_seed,x.shape[1],PROTOCOL_MAIN["batch"],
            PROTOCOL_MAIN["batch"],PROTOCOL_MAIN["input_dim"],PROTOCOL_MAIN["chunk"],x.device,start_index=post_start)
        normalized_post=None
        if total_norm>0 and carry_norm>0:
            diagnostic=new_model(task,x.device,pre_state)
            factor=rate*carry_norm/total_norm
            with torch.no_grad():
                for (name,parameter),(grad_name,gradient) in zip(diagnostic.named_parameters(),gradients):
                    if name!=grad_name: raise AssertionError("Diagnostic parameter order changed")
                    parameter.add_(gradient,alpha=-factor)
            normalized_post=delayed_tasks.delayed_evaluate(diagnostic,task,post_seed,x.shape[1],
                PROTOCOL_MAIN["batch"],PROTOCOL_MAIN["batch"],PROTOCOL_MAIN["input_dim"],PROTOCOL_MAIN["chunk"],
                x.device,start_index=post_start)
        rows.append({"condition":condition,"pre_update_loss":float(loss.detach()),
                     "forward_max_abs_difference":forward_error,"credit_norm_by_family":credit_by_family,
                     "core_credit_against_carry":credit_against_carry(gradients,carry_gradients),
                     "pre_clip_total_norm":preclip,"actual_parameter_change_by_family":changed_by_family,
                     "true_optimizer_post_update":real_post,"equal_norm_sgd_post_update":normalized_post})
    immediate=immediate_credit(pre_state,x,y,state,task,boundary)
    # Erasing credit before the final step splits the suffix into two online calls, which sum in a
    # different order than one call; the forward is held to the protocol's scaled FP32 tolerance.
    loss=rows[0]["pre_update_loss"]
    immediate_forward=max(float((immediate["logits"]-reference[1]).abs().max())/(1+float(reference[1].abs().max())),
                          abs(immediate["loss"]-loss)/(1+abs(loss)))
    if not math.isfinite(immediate_forward) or immediate_forward>TOLERANCE:
        raise AssertionError(f"Immediate-only diagnostic changed pre-update forward behavior: {immediate_forward}")
    immediate.pop("logits")
    immediate["forward_scaled_difference"]=immediate_forward
    if suffix==1:
        immediate["isolated_w_pathway"]=isolated_w_pathway(pre_state,x,y,state,task)
    return rows,immediate,{"max_forward_difference":max(r["forward_max_abs_difference"] for r in rows),
                           "precision":harness.precision_record(x.device),
                           "prefix_state_bytes":state.nbytes(),"carry_total_credit_norm":carry_norm}


def run(job, ctx):
    global PROTOCOL_MAIN, INTERVENTION, TOLERANCE
    PROTOCOL_MAIN=ctx.matrix["delayed"]["main"]; INTERVENTION=ctx.protocol
    TOLERANCE=ctx.matrix["delayed"]["execution"]["evaluation_batch_equivalence_scaled_tolerance"]
    model,optimizer,source=source_model(job,ctx)
    total=INTERVENTION["paired_episodes_per_condition"]
    batch_size=INTERVENTION["batch"]
    if total%batch_size: raise ValueError("Paired episode count must divide batch")
    rows=[]; immediate=[]; controls=[]
    for start in range(0,total,batch_size):
        batch_path=ctx.directory/f"batch-{start:04d}.json"
        if batch_path.exists():
            batch_record=json.loads(batch_path.read_text())
            if (batch_record["job"]!=job or batch_record["source"]!=source or
                batch_record["batch_start"]!=start):
                raise ValueError("Saved paired batch identity differs")
        else:
            x,y,identities=delayed_tasks.delayed_batch(job["task"],830021+job["seed"],start,job["length"],
                batch_size,PROTOCOL_MAIN["input_dim"],ctx.device)
            branch_rows,one_step,control=fork_batch(model,optimizer.state_dict(),x,y,
                job["task"],source["selected_lr"],850021+job["seed"],start,job["boundary_suffix"])
            batch_record={"job":job,"source":source,"batch_start":start,
                          "episode_ids":identities,"conditions":branch_rows,
                          "immediate_only":one_step,"controls":control}
            harness.write_record(batch_path,batch_record)
        rows.append({"batch_start":start,"episode_ids":batch_record["episode_ids"],
                     "conditions":batch_record["conditions"]})
        immediate.append({"batch_start":start,**batch_record["immediate_only"]})
        controls.append({"batch_start":start,**batch_record["controls"]})
        if ctx.stop_due():
            raise harness.Stopped(f"stopped after the paired batch at {start}")
    return {"source":source,"precision":harness.precision_record(ctx.device),
            "paired_episodes_per_condition":total,
            "batches":rows,"immediate_only":immediate,"controls":controls,
            "batch_sha256":{path.name:harness.sha256(path) for path in sorted(ctx.directory.glob("batch-*.json"))},
            "elapsed_seconds":ctx.elapsed(),
            "interpretation":"all conditions share exact pre-update weights, state, suffix and target"}
