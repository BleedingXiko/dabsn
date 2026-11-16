"""Models, data materialization, and one training update, shared by the learning studies.

``Model`` is one DABSN core with a linear readout. ``sequence`` runs one episode in
chunks under a credit method:

- ``online``: B-channel online credit carried across every chunk boundary;
- ``reset``: online credit, with E erased at every chunk boundary;
- ``permute``: online credit, with E rolled across lanes once after the first chunk;
- ``tbptt``: gradients truncated at every chunk boundary;
- ``bptt``: full backpropagation through the whole episode.

Every method sees identical parameters, data, and forward computation.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from dabsn.core import DABSNCore
from dabsn.distributed import clip_grad_norm
from dabsn.temporal_credit import CoreEligibilityState, OnlineCoreFrame, initial_core_eligibility

from harness import autocast, finite
import tasks


class Readout(nn.Linear):
    """Linear readout computed in FP32 under every compute policy.

    Under bf16 autocast a 16-bit readout output depends on the batch size by one
    bf16 rounding step (cuBLAS chooses its kernel by batch shape), which would
    make evaluation depend on how episodes are batched. The readout is tiny; it
    and the loss stay FP32. The core and its projections follow the policy.
    """

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), self.weight, self.bias)


class Model(nn.Module):
    def __init__(self, inputs, state, classes, seed, device):
        super().__init__()
        torch.manual_seed(seed)
        # The published DABSN core with its own default initialization, unmodified.
        # Initialized on CPU so every method starts from identical parameters.
        self.core = DABSNCore(inputs, state)
        self.head = Readout(state, classes)
        self.to(device=device, dtype=torch.float32)

def association_batch(seed, step, batch_size, length, classes, keys, associations, device, distractors=True):
    # CPU-generated data are identical across methods and independent of model RNG.
    rng=torch.Generator().manual_seed(seed+1000003*step)
    if length<=associations or keys<associations:
        raise ValueError("Episode must include writes and a later query")
    x=torch.zeros(batch_size,length,keys+classes+3)
    labels=torch.empty(batch_size,dtype=torch.long)
    for lane in range(batch_size):
        kk=torch.randperm(keys,generator=rng)[:associations]
        vv=torch.randint(classes,(associations,),generator=rng)
        query=int(torch.randint(associations,(),generator=rng))
        x[lane,torch.arange(associations),kk]=1
        x[lane,torch.arange(associations),keys+vv]=1
        x[lane,:associations,keys+classes]=1  # WRITE
        if distractors:
            positions=torch.arange(associations,length-1)
            # Independent component RNGs preserve associations and noise prefixes
            # when length or distractor presence changes.
            key_rng=torch.Generator().manual_seed(seed+1000003*step+3000017*(lane+1))
            value_rng=torch.Generator().manual_seed(seed+1000003*step+7000003*(lane+1))
            nk=torch.randint(keys,(len(positions),),generator=key_rng)
            nv=torch.randint(classes,(len(positions),),generator=value_rng)
            x[lane,positions,nk]=1; x[lane,positions,keys+nv]=1
            x[lane,positions,keys+classes+1]=1  # NOISE
        x[lane,-1,kk[query]]=1
        x[lane,-1,-1]=1  # QUERY; value coordinates remain zero.
        labels[lane]=vv[query]
    return x.to(device),labels.to(device)

def sequence(model, x, labels, method, chunk, backward=True, objective="classification"):
    c=model.core; b,t,_=x.shape
    state=c.initial_state(b,device=x.device,dtype=torch.float32)
    eligibility=initial_core_eligibility(c,b,device=x.device,dtype=torch.float32) if method in ("online","reset","permute") else None
    loss_value=None
    for start in range(0,t,chunk):
        stop=min(start+chunk,t); last=stop==t
        frame=OnlineCoreFrame(eligibility) if eligibility is not None else None
        with autocast(x.device):
            outputs,state=c.forward_from_state(
                x[:,start:stop],initial_state=state,return_final_state=True,
                temporal_credit="online" if frame is not None else "bptt",
                eligibility_state=eligibility,online_frame=frame)
            if last:
                logits=model.head(torch.tanh(outputs[0][:,-1,c.hidden_dim:]))
        # Native batched output state may already be BF16-rounded. Promote its
        # public carry storage without claiming FP32 internal recurrence math.
        # float() retains the graph for full BPTT and is a no-op for FP32 state.
        state=tuple(value.float() for value in state)
        if any(value.dtype!=torch.float32 for value in state):
            raise AssertionError('Predictive state must retain FP32 storage')
        if last:
            logits=logits.float()
            if objective=="classification":
                loss=F.cross_entropy(logits,labels)
            elif objective=="regression":
                loss=F.mse_loss(logits,labels.float())
            else:
                raise ValueError(f"Unknown objective: {objective}")
            finite(loss,"nonfinite loss")
            loss_value=float(loss.detach())
            if backward: loss.backward()
        elif frame is not None:
            # Replay with zero signals is necessary to advance eligibility.
            (outputs[0].float().sum()*0).backward()
        if frame is not None:
            if not frame.replayed or frame.resulting_eligibility is None:
                raise RuntimeError("online eligibility failed to advance")
            eligibility=frame.resulting_eligibility
            if any(value.dtype!=torch.float32 for value in eligibility.tensors()):
                raise AssertionError('Eligibility must retain FP32 storage')
            if not last and method=="reset":
                eligibility=initial_core_eligibility(c,b,device=x.device,dtype=torch.float32)
            elif not last and method=="permute" and start==0:
                eligibility=CoreEligibilityState(*(v.roll(1,0).contiguous() for v in eligibility.tensors()))
        if method!="bptt": state=tuple(v.detach() for v in state)
    return loss_value,logits.detach(),state,eligibility

@torch.no_grad()
def association_evaluate(model, cfg, length, device, distractors=True):
    # Common full forward pass: evaluation does not depend on the learning rule.
    correct=0; n=0; loss=0.
    for i in range(cfg["evaluation_batches"]):
        x,y=association_batch(cfg["test_seed"],i,cfg["batch"],length,cfg["classes"],
                  cfg["keys"],cfg["associations"],device,distractors)
        if is_block(model):
            logits=block_predict(model,x)
        else:
            with autocast(device):
                outputs=model.core.forward_from_state(x)
                logits=model.head(torch.tanh(outputs[0][:,-1,model.core.hidden_dim:]))
        logits=logits.float()
        finite(logits,"nonfinite evaluation logits")
        loss+=float(F.cross_entropy(logits,y,reduction="sum"))
        correct+=int((logits.argmax(-1)==y).sum()); n+=y.numel()
    return {"loss":loss/n,"accuracy":correct/n,"correct":correct,"examples":n}

def update(model,opt,x,y,method,chunk,clip,objective="classification",frozen=None):
    if is_block(model):
        loss,norm,logits=block_update(model,opt,x,y,method,chunk,clip,objective,frozen)
        return loss,norm,logits,None,None
    if frozen is not None: raise ValueError("Frozen-core arm requires the full block")
    opt.zero_grad(set_to_none=True)
    loss,logits,state,eligibility=sequence(model,x,y,method,chunk,objective=objective)
    if not bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()):
        raise FloatingPointError("nonfinite parameter credit")
    norm=float(clip_grad_norm(model,clip))
    if not math.isfinite(norm): raise FloatingPointError("nonfinite pre-clip norm")
    opt.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
        raise FloatingPointError("nonfinite updated parameter")
    return loss,norm,logits,state,eligibility


# ---------------------------------------------- delayed-learning tasks

def objective(task):
    return "classification" if task=="selective_retrieval" else "regression"


def output_dim(task):
    return 16 if task=="selective_retrieval" else 8


def delayed_batch(task,seed,start,length,batch_size,inputs,device,distractors=True):
    """Materialize public sparse episodes with identical IDs for every method."""
    x=torch.zeros(batch_size,length,inputs,dtype=torch.float32)
    labels=[]; identities=[]
    for lane in range(batch_size):
        episode=tasks.episode(task,seed,start+lane,length,inputs,distractors)
        for event in episode["events"]:
            x[lane,event["time"],event["indices"]]=torch.tensor(event["values"],dtype=x.dtype)
        labels.append(episode["target"])
        identities.append((seed,start+lane))
    label_dtype=torch.long if objective(task)=="classification" else torch.float32
    y=torch.tensor(labels,dtype=label_dtype)
    return x.to(device),y.to(device),identities


@torch.no_grad()
def delayed_predict(model,x,chunk):
    if is_block(model):
        return block_predict(model,x)
    state=model.core.initial_state(x.shape[0],device=x.device,dtype=x.dtype)
    last=None
    for start in range(0,x.shape[1],chunk):
        with autocast(x.device):
            output,state=model.core.forward_from_state(x[:,start:start+chunk],
                initial_state=state,return_final_state=True)
        state=tuple(value.float() for value in state)
        last=output[0][:,-1,model.core.hidden_dim:]
    with autocast(x.device):
        return model.head(torch.tanh(last))


@torch.no_grad()
def delayed_evaluate(model,task,seed,length,count,batch_size,inputs,chunk,device,start_index=0,
             input_policy="full"):
    if count%batch_size: raise ValueError("Evaluation count must divide batch")
    if input_policy not in ("full","final_chunk_only"):
        raise ValueError("Unknown evaluation input policy")
    summed=0.; correct=0; zero_summed=0.
    for start in range(0,count,batch_size):
        x,y,_=delayed_batch(task,seed,start_index+start,length,batch_size,inputs,device)
        if input_policy=="final_chunk_only":
            x=x.clone()
            x[:,:-chunk]=0
        logits=delayed_predict(model,x,chunk)
        finite(logits,"nonfinite evaluation output")
        if objective(task)=="classification":
            summed+=float(F.cross_entropy(logits.float(),y,reduction="sum"))
            correct+=int((logits.argmax(-1)==y).sum())
        else:
            summed+=float(F.mse_loss(logits.float(),y.float(),reduction="sum"))
            zero_summed+=float(y.square().sum())
    result={"examples":count,"loss":summed/count if objective(task)=="classification"
            else summed/(count*8)}
    if objective(task)=="classification":
        result.update(correct=correct,accuracy=correct/count)
    else:
        result.update(normalized_mse=result["loss"],
                      zero_predictor_normalized_mse=zero_summed/(count*8))
    return result


# ------------------------------------------------- the full DABSN block

import dataclasses

from dabsn import DABSNLayerSpec, DABSNModel
from dabsn.runtime.api import apply_optimizer_step
from dabsn.temporal_credit import (
    LongReadEligibilityState, bind_online_temporal_contract, finalize_online_temporal_state,
    initial_long_read_eligibility, initial_online_temporal_state, prepare_online_temporal_state,
)


class Block(nn.Module):
    """The full DABSN block (core + read) at its stock initialization: one layer, causal
    seq read, token readout. Used for every task that needs memory beyond the core."""

    def __init__(self, inputs, state, classes, seed, device):
        super().__init__()
        torch.manual_seed(seed)
        self.net = DABSNModel(inputs, classes,
                              [DABSNLayerSpec(hidden_dim=state, state_dim=state, read_geometry="seq")],
                              output_adapter="token")
        self.to(device=device, dtype=torch.float32)

    @property
    def block(self):
        return self.net.backbone.blocks[0]

    @property
    def core(self):
        return self.block.core


def is_block(model):
    return isinstance(model, Block)


class FrozenCore:
    """A frozen core that keeps the online machinery running.

    The core parameters keep requires_grad, so the online replay runs and E advances
    exactly as in the trained arms; the core is excluded from the optimizer and its
    gradients are dropped before clipping, so no core update is ever applied.
    ``verify`` proves the core is bitwise unchanged."""

    def __init__(self, model):
        self.parameters = list(model.core.parameters())
        self.ids = {id(p) for p in self.parameters}
        self.initial = [p.detach().clone() for p in self.parameters]

    def trainable(self, model):
        return [p for p in model.parameters() if id(p) not in self.ids]

    def drop_gradients(self):
        for p in self.parameters:
            p.grad = None

    def verify(self):
        for p, initial in zip(self.parameters, self.initial):
            if not torch.equal(p.detach(), initial):
                raise AssertionError("Frozen core parameters changed")
        return True


def _replace_eligibility(state, transform):
    """Apply transform(core_E, long_E) -> (core_E, long_E) to every layer; forward state untouched."""
    layers = []
    for layer in state.layers:
        core_e, long_e = transform(layer.core_eligibility, layer.long_eligibility)
        layers.append(dataclasses.replace(layer, core_eligibility=core_e, long_eligibility=long_e))
    return dataclasses.replace(state, layers=tuple(layers))


def erase_credit(model, state):
    """Erase carried credit (core E and long-read E); recurrent state and read bank are kept."""
    def fresh(core_e, long_e):
        batch = core_e.batch_size
        return (initial_core_eligibility(model.core, batch, device=core_e.device, dtype=core_e.dtype),
                initial_long_read_eligibility(model.block.read, batch, device=core_e.device, dtype=core_e.dtype))
    return _replace_eligibility(state, fresh)


def roll_credit(state, shift=1):
    """Give every lane another lane's carried credit; forward state untouched."""
    def roll(core_e, long_e):
        return (CoreEligibilityState(*(v.roll(shift, 0).contiguous() for v in core_e.tensors())),
                LongReadEligibilityState(long_e.traces.roll(shift, 0).contiguous()))
    return _replace_eligibility(state, roll)


def block_online_state(model, batch, chunk, device):
    ids = tuple(range(batch))
    state = initial_online_temporal_state(model.net, batch, stream_policy="reset_each_batch", stream_ids=ids,
                                          device=device, dtype=torch.float32, eligibility_dtype=torch.float32)
    state = prepare_online_temporal_state(model.net, state, stream_ids=ids, stream_reset_mask=None)
    return bind_online_temporal_contract(state, temporal_chunk_size=chunk, eligibility_batch_mode="vectorized",
                                         eligibility_lane_group_size=None)


def _final_loss(logits, labels, objective):
    logits = logits.float()
    if objective == "classification":
        return F.cross_entropy(logits, labels)
    if objective == "regression":
        return F.mse_loss(logits, labels.float())
    raise ValueError(f"Unknown objective: {objective}")


def block_sequence(model, x, labels, method, chunk, objective="classification"):
    """One episode on the full block; loss on the final step.

    online : carried credit across every chunk boundary (public online API);
    reset  : the same, with core and long-read E erased at every boundary;
    permute: the same, with E rolled across lanes once after the first chunk;
    bptt   : the full graph through the whole episode (learnability reference).
    """
    if method == "bptt":
        with autocast(x.device):
            logits = model.net.forward_sequence(x)[:, -1]
        loss = _final_loss(logits, labels, objective)
        finite(loss, "nonfinite loss")
        loss.backward()
        return float(loss.detach()), logits.detach().float()
    if method not in ("online", "reset", "permute"):
        raise ValueError(f"Unknown block method: {method}")
    total = x.shape[1]
    state = block_online_state(model, x.shape[0], chunk, x.device)
    loss_value = None; logits = None
    for start in range(0, total, chunk):
        stop = min(start + chunk, total); last = stop == total
        with autocast(x.device):
            output, pending = model.net.forward_sequence_online(x[:, start:stop], state)
        if last:
            logits = output[:, -1].float()
            loss = _final_loss(logits, labels, objective)
            finite(loss, "nonfinite loss")
            loss_value = float(loss.detach())
            loss.backward()
        else:
            # Zero cotangents still replay, which advances the carried credit.
            (output.float().sum() * 0).backward()
        state = finalize_online_temporal_state(pending)
        if not last and method == "reset":
            state = erase_credit(model, state)
        elif not last and method == "permute" and start == 0:
            state = roll_credit(state)
    return loss_value, logits.detach()


def online_chunk(chunk, x):
    """Temporal chunk for one episode: "episode" (or None) is one chunk spanning the
    episode, the framework's train_step default; an integer splits the episode."""
    return x.shape[1] if chunk in (None, "episode") else int(chunk)


def block_update(model, opt, x, y, method, chunk, clip, objective="classification", frozen=None):
    """One update of the full block. ``frozen_core`` runs the online path unchanged and
    drops the core gradients before clipping (see FrozenCore)."""
    if (method == "frozen_core") != (frozen is not None):
        raise ValueError("frozen_core requires a FrozenCore and only frozen_core accepts one")
    opt.zero_grad(set_to_none=True)
    chunk = x.shape[1] if method == "bptt" else online_chunk(chunk, x)
    loss, logits = block_sequence(model, x, y, "online" if method == "frozen_core" else method,
                                  chunk, objective)
    if frozen is not None:
        frozen.drop_gradients()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if grads and not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
        raise FloatingPointError("nonfinite parameter credit")
    norm = float(clip_grad_norm(model.net, clip))
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite pre-clip norm")
    apply_optimizer_step(model.net, opt)
    if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
        raise FloatingPointError("nonfinite updated parameter")
    return loss, norm, logits


@torch.no_grad()
def block_predict(model, x):
    """Common full forward, independent of the training method."""
    with autocast(x.device):
        return model.net.forward_sequence(x)[:, -1].float()


def permute_credit(state, permutation):
    """Lane i receives lane permutation[i]'s carried credit; forward state untouched."""
    def select(core_e, long_e):
        return (CoreEligibilityState(*(v.index_select(0, permutation).contiguous() for v in core_e.tensors())),
                LongReadEligibilityState(long_e.traces.index_select(0, permutation).contiguous()))
    return _replace_eligibility(state, select)


def block_stream_state(model, batch, chunk, device):
    """Online state for one uninterrupted stream per lane (stable lane identities)."""
    ids = tuple(range(batch))
    state = initial_online_temporal_state(model.net, batch, stream_policy="carry_lanes", stream_ids=ids,
                                          device=device, dtype=torch.float32, eligibility_dtype=torch.float32)
    return bind_online_temporal_contract(state, temporal_chunk_size=chunk, eligibility_batch_mode="vectorized",
                                         eligibility_lane_group_size=None)


def block_stream_step(model, state, x):
    """Advance one chunk of an uninterrupted stream. Returns (final-step logits, pending state)."""
    ids = tuple(range(x.shape[0]))
    state = prepare_online_temporal_state(model.net, state, stream_ids=ids, stream_reset_mask=None)
    with autocast(x.device):
        output, pending = model.net.forward_sequence_online(x, state)
    return output[:, -1].float(), output, pending


class _NetView:
    """Adapter so the credit controls work on any single-layer DABSNModel."""
    def __init__(self, net):
        self.net = net
        self.block = net.backbone.blocks[0]
        self.core = self.block.core


def net_chunked_update(net, optimizer, x, targets, method, chunk, clip, loss_fn, frozen=None):
    """One update of a full DABSN model over chunks with per-position loss.

    online : carried credit across every chunk boundary;
    reset  : credit (core E and long-read E) erased at every boundary;
    permute: credit rolled across lanes once after the first chunk.
    The forward computation (recurrence and read bank) is identical in all three.
    ``frozen`` (a FrozenCore) drops the core gradients before clipping."""
    if method not in ("online", "reset", "permute"):
        raise ValueError(f"Unknown chunked method: {method}")
    view = _NetView(net)
    optimizer.zero_grad(set_to_none=True)
    total = x.shape[1]
    state = block_online_state(view, x.shape[0], chunk, x.device)
    summed = 0.0
    for start in range(0, total, chunk):
        stop = min(start + chunk, total); last = stop == total
        with autocast(x.device):
            output, pending = net.forward_sequence_online(x[:, start:stop], state)
        part = loss_fn(output, targets[:, start:stop]) * ((stop - start) / total)
        finite(part, "nonfinite loss")
        summed += float(part.detach())
        part.backward()
        state = finalize_online_temporal_state(pending)
        if not last and method == "reset":
            state = erase_credit(view, state)
        elif not last and method == "permute" and start == 0:
            state = roll_credit(state)
    if frozen is not None:
        frozen.drop_gradients()
    norm = float(clip_grad_norm(net, clip))
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite pre-clip norm")
    apply_optimizer_step(net, optimizer)
    return summed, norm
