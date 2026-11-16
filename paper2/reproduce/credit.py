"""Operations on the B-channel online credit state E (``CoreEligibilityState``).

E holds, per batch lane: a budget-only trace for each of the four matrices
(W, A, Wg, Ug), a three-state (budget, energy, saturation) trace for each of the
sixteen per-channel controls, and a three-state trace for each of the four shared
controls. Every intervention here returns a new state; none mutates its input.
"""
from __future__ import annotations

import math

import torch
from dabsn.temporal_credit import (
    CORE_GLOBAL_PARAMETER_NAMES, CORE_LOCAL_PARAMETER_NAMES, CoreEligibilityState,
    initial_core_eligibility,
)

MATRIX_TRACES = {"W.weight": "w_budget", "A.weight": "a_budget",
                 "Wg.weight": "wg_budget", "Ug.weight": "ug_budget"}


def fresh(core, batch, *, device, dtype=torch.float32):
    return initial_core_eligibility(core, batch, device=device, dtype=dtype)


def rebuild(tensors):
    return CoreEligibilityState(*(value.contiguous() for value in tensors))


def zero(e):
    return rebuild(torch.zeros_like(value) for value in e.tensors())


def roll_lanes(e, shift=1):
    """Give every lane the trace of another lane (cyclic shift)."""
    return rebuild(value.roll(shift, 0) for value in e.tensors())


def flip_lanes(e):
    return rebuild(value.flip(0) for value in e.tensors())


def select_lanes(e, permutation):
    """Lane i receives the trace of lane permutation[i]."""
    return rebuild(value.index_select(0, permutation) for value in e.tensors())


def negate_w(e):
    tensors = [value.detach().clone() for value in e.tensors()]
    tensors[0] = -tensors[0]
    return rebuild(tensors)


def erase_local(e):
    tensors = [value.detach().clone() for value in e.tensors()]
    tensors[4] = torch.zeros_like(tensors[4])
    return rebuild(tensors)


def l2(e):
    return math.sqrt(sum(float(value.float().square().sum()) for value in e.tensors()))


def scalars(e):
    return sum(value.numel() for value in e.tensors())


def column_scales(core, e):
    """RMS response scale of every parameter coordinate's embedded column of E.

    The embedded column of a parameter coordinate is its trace placed in the
    3*B*S state coordinates it influences; its L2 norm over sqrt(3*B*S) is the
    coordinate's response scale. Returns ({parameter: scale tensor}, global mean
    over all core parameter coordinates including zero columns).
    """
    denominator = 3 * e.batch_size * e.state_dim
    result = {}
    for name, p in core.named_parameters():
        if name in MATRIX_TRACES:
            scale = (getattr(e, MATRIX_TRACES[name]).square().sum(0) / denominator).sqrt()
        elif name in CORE_LOCAL_PARAMETER_NAMES:
            k = CORE_LOCAL_PARAMETER_NAMES.index(name)
            scale = (e.local[:, :, :, k].square().sum((0, 2)) / denominator).sqrt()
        elif name in CORE_GLOBAL_PARAMETER_NAMES:
            k = CORE_GLOBAL_PARAMETER_NAMES.index(name)
            scale = (e.global_[:, :, :, k].square().sum() / denominator).sqrt()
        else:
            raise ValueError(f"Unmapped core parameter: {name}")
        result[p] = scale.reshape_as(p)
    count = sum(p.numel() for p in core.parameters())
    mean = sum(value.sum() for value in result.values()) / count
    return result, mean


def family(name):
    """Parameter family of a study-model parameter name (bare core, or the full block)."""
    core = name.split(".core.", 1)[1] if ".core." in name else name.removeprefix("core.")
    if core in MATRIX_TRACES:
        return core.removesuffix(".weight")
    if core in CORE_LOCAL_PARAMETER_NAMES:
        return "local_controls"
    if core in CORE_GLOBAL_PARAMETER_NAMES:
        return "shared_controls"
    if name.startswith("head.") or "output_adapter" in name:
        return "readout"
    if ".read." in name or name.startswith("read."):
        return "read"
    return "block_other"


def embedded_operator(core,eligibility):
    """Materialize K = [3*B*S, P] for microscopic cores only; training never allocates it."""
    b=eligibility.batch_size; s=eligibility.state_dim
    columns=[]; owners=[]; names=[]
    for name,parameter in core.named_parameters():
        if name in ("W.weight","A.weight","Wg.weight","Ug.weight"):
            trace=getattr(eligibility,{
                "W.weight":"w_budget","A.weight":"a_budget",
                "Wg.weight":"wg_budget","Ug.weight":"ug_budget"}[name])
            for i in range(s):
                for j in range(parameter.shape[1]):
                    column=torch.zeros(b,s,3,dtype=trace.dtype,device=trace.device)
                    column[:,i,0]=trace[:,i,j]
                    columns.append(column.reshape(-1)); owners.append(i)
                    names.append(f"{name}[{i},{j}]")
        elif name in CORE_LOCAL_PARAMETER_NAMES:
            k=CORE_LOCAL_PARAMETER_NAMES.index(name)
            for i in range(s):
                column=torch.zeros(b,s,3,dtype=eligibility.dtype,device=eligibility.device)
                column[:,i,:]=eligibility.local[:,i,:,k]
                columns.append(column.reshape(-1)); owners.append(i)
                names.append(f"{name}[{i}]")
        elif name in CORE_GLOBAL_PARAMETER_NAMES:
            k=CORE_GLOBAL_PARAMETER_NAMES.index(name)
            columns.append(eligibility.global_[:,:,:,k].reshape(-1))
            owners.append("shared"); names.append(name)
        else:
            raise ValueError(f"Unmapped core parameter: {name}")
    return torch.stack(columns,dim=1),owners,names


