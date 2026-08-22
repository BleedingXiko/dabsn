"""Registered compact dense admitted read with bounded forward/backward tiles."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor

from .admitted import (
    _admitted_three_way_read_backward,
    _admitted_three_way_read_op,
)


def dense_bmm_three_way_read_exact(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    bank_idx: Tensor,
    bank_valid: Tensor,
    *,
    mode: str,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
    query_offset: int = 0,
    total_steps: int | None = None,
) -> Tensor:
    """Exact deployed tensor-core BMM admitted read, shared by ABI and runtime."""
    batch, steps, _hidden = query.shape
    bank = bank_keys.shape[1]
    full_steps = int(total_steps) if total_steps is not None else steps
    scale_f = scale.to(torch.float32)
    compatibility = torch.bmm(query, bank_keys.to(query.dtype).transpose(1, 2)).float() * scale_f
    key_admission = (bank_key_bias.to(torch.float32) + bank_admission.to(torch.float32)).unsqueeze(
        1
    )
    short_scores = compatibility + key_admission
    cocktail_compatibility = torch.bmm(
        cocktail.to(query.dtype),
        bank_cocktail.to(query.dtype).transpose(1, 2),
    ).float()
    permanent_scores = short_scores + cocktail_compatibility * cocktail_gain.to(torch.float32)
    valid = bank_valid.unsqueeze(1)
    bank_positions = bank_idx.unsqueeze(1)
    if mode == "seq":
        query_positions = (torch.arange(steps, device=query.device) + int(query_offset)).view(
            1, steps, 1
        )
        allow = valid & (bank_positions <= query_positions)
        induct_allow = valid & (bank_positions < query_positions)
    elif mode == "field":
        allow = valid.expand(batch, steps, bank)
        induct_allow = valid & (bank_positions < (full_steps - 1))
    else:
        raise ValueError("dense BMM admitted-read mode must be seq or field")

    def read(scores: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        eligible = mask.any(dim=-1, keepdim=True)
        minimum = torch.finfo(scores.dtype).min
        weights = torch.softmax(scores.masked_fill(~mask, minimum), dim=-1)
        return torch.bmm(weights.to(values.dtype), values).float().mul_(eligible)

    short = read(short_scores, bank_writes, allow)
    permanent = read(permanent_scores, bank_writes, allow)
    induct = read(short_scores, next_writes, induct_allow)
    output = (
        short_gain.to(torch.float32) * short
        + pad_gain.to(torch.float32) * permanent
        + induct_gain.to(torch.float32) * induct
    )
    return output.to(query.dtype)


def _validate_mode_and_budget(mode: int, score_budget: int) -> None:
    if mode not in {0, 1}:
        raise ValueError("compact admitted-read mode must be 0 (seq) or 1 (field)")
    if score_budget <= 0:
        raise ValueError("compact admitted-read score_budget must be positive")


def _chunk_width(query: Tensor, bank_keys: Tensor, score_budget: int) -> int:
    batch = query.shape[0]
    bank = bank_keys.shape[1]
    return max(1, int(score_budget) // max(1, int(batch) * int(bank)))


def _compact_dense_bmm_forward(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    bank_idx: Tensor,
    bank_valid: Tensor,
    mode: int,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
    score_budget: int,
) -> Tensor:
    _validate_mode_and_budget(mode, score_budget)
    total_steps = query.shape[1]
    chunk = _chunk_width(query, bank_keys, score_budget)
    pieces = []
    for start in range(0, total_steps, chunk):
        stop = min(total_steps, start + chunk)
        pieces.append(
            dense_bmm_three_way_read_exact(
                query[:, start:stop],
                bank_keys,
                bank_writes,
                next_writes,
                cocktail[:, start:stop],
                bank_cocktail,
                bank_key_bias,
                bank_admission,
                scale,
                bank_idx,
                bank_valid,
                mode="seq" if mode == 0 else "field",
                short_gain=short_gain,
                pad_gain=pad_gain,
                induct_gain=induct_gain,
                cocktail_gain=cocktail_gain,
                query_offset=start,
                total_steps=total_steps,
            )
        )
    return torch.cat(pieces, dim=1)


_COMPOSITE_LIBRARY = torch.library.Library("dabsn", "FRAGMENT")
_COMPOSITE_LIBRARY.define(
    "admitted_three_way_read_compact_dense_bmm("
    "Tensor query, Tensor bank_keys, Tensor bank_writes, Tensor next_writes, "
    "Tensor cocktail, Tensor bank_cocktail, Tensor bank_key_bias, "
    "Tensor bank_admission, Tensor scale, Tensor bank_idx, Tensor bank_valid, "
    "int mode, Tensor short_gain, Tensor pad_gain, Tensor induct_gain, "
    "Tensor cocktail_gain, int score_budget) -> Tensor"
)
torch.library.impl(
    _COMPOSITE_LIBRARY,
    "admitted_three_way_read_compact_dense_bmm",
    "CompositeImplicitAutograd",
)(_compact_dense_bmm_forward)

_compact_dense_bmm_op = torch.ops.dabsn.admitted_three_way_read_compact_dense_bmm.default


@torch.library.register_vmap(_compact_dense_bmm_op)
def _compact_dense_bmm_vmap(info, in_dims, *inputs):
    def select(value, dim, index):
        return value if dim is None else value.movedim(dim, 0)[index]

    outputs = [
        _compact_dense_bmm_op(*(select(value, dim, index) for value, dim in zip(inputs, in_dims)))
        for index in range(info.batch_size)
    ]
    return torch.stack(outputs), 0


def _compact_masks(
    query: Tensor,
    bank_idx: Tensor,
    bank_valid: Tensor,
    *,
    mode: int,
    query_offset: int,
    total_steps: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, steps = query.shape[:2]
    bank = bank_idx.shape[1]
    valid = bank_valid.unsqueeze(1)
    positions = bank_idx.unsqueeze(1)
    if mode == 0:
        query_positions = (torch.arange(steps, device=query.device) + int(query_offset)).view(
            1, steps, 1
        )
        allow = valid & (positions <= query_positions)
        induct_allow = valid & (positions < query_positions)
    else:
        allow = valid.expand(batch, steps, bank)
        induct_allow = (valid & (positions < (int(total_steps) - 1))).expand(batch, steps, bank)
    return allow, induct_allow, allow.any(dim=-1), induct_allow.any(dim=-1)


def _compact_dense_forward(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    bank_idx: Tensor,
    bank_valid: Tensor,
    mode: int,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
    score_budget: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_mode_and_budget(mode, score_budget)
    if query.dim() != 3 or bank_keys.dim() != 3:
        raise ValueError("compact admitted read expects query/bank tensors with rank 3")
    if bank_idx.shape != bank_valid.shape or bank_idx.shape != bank_keys.shape[:2]:
        raise ValueError("bank_idx and bank_valid must match bank shape [B,N]")
    batch, total_steps, hidden = query.shape
    bank = bank_keys.shape[1]
    chunk = _chunk_width(query, bank_keys, score_budget)
    output = torch.empty((batch, total_steps, hidden), device=query.device, dtype=query.dtype)
    weight_dtype = torch.float32 if query.dtype in {torch.float16, torch.bfloat16} else query.dtype
    short_weights = torch.empty((batch, total_steps, bank), device=query.device, dtype=weight_dtype)
    permanent_weights = torch.empty_like(short_weights)
    induct_weights = torch.empty_like(short_weights)
    for start in range(0, total_steps, chunk):
        stop = min(total_steps, start + chunk)
        query_chunk = query[:, start:stop]
        allow, induct_allow, eligible, induct_eligible = _compact_masks(
            query_chunk,
            bank_idx,
            bank_valid,
            mode=mode,
            query_offset=start,
            total_steps=total_steps,
        )
        values = _admitted_three_way_read_op(
            query_chunk,
            bank_keys,
            bank_writes,
            next_writes,
            cocktail[:, start:stop],
            bank_cocktail,
            bank_key_bias,
            bank_admission,
            scale,
            allow,
            induct_allow,
            eligible,
            induct_eligible,
            short_gain,
            pad_gain,
            induct_gain,
            cocktail_gain,
        )
        output[:, start:stop].copy_(values[0])
        short_weights[:, start:stop].copy_(values[1])
        permanent_weights[:, start:stop].copy_(values[2])
        induct_weights[:, start:stop].copy_(values[3])
    return output, short_weights, permanent_weights, induct_weights


@torch.library.custom_op(
    "dabsn::admitted_three_way_read_compact_dense",
    mutates_args=(),
)
def _admitted_three_way_read_compact_dense_op(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    bank_idx: Tensor,
    bank_valid: Tensor,
    mode: int,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
    score_budget: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _compact_dense_forward(
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
        bank_cocktail,
        bank_key_bias,
        bank_admission,
        scale,
        bank_idx,
        bank_valid,
        mode,
        short_gain,
        pad_gain,
        induct_gain,
        cocktail_gain,
        score_budget,
    )


@_admitted_three_way_read_compact_dense_op.register_fake
def _admitted_three_way_read_compact_dense_fake(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    bank_idx: Tensor,
    bank_valid: Tensor,
    mode: int,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
    score_budget: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del (
        bank_writes,
        next_writes,
        cocktail,
        bank_cocktail,
        bank_key_bias,
        bank_admission,
        scale,
        bank_idx,
        bank_valid,
        mode,
        short_gain,
        pad_gain,
        induct_gain,
        cocktail_gain,
        score_budget,
    )
    batch, steps, hidden = query.shape
    bank = bank_keys.shape[1]
    weight_dtype = torch.float32 if query.dtype in {torch.float16, torch.bfloat16} else query.dtype
    weights = torch.empty((batch, steps, bank), device=query.device, dtype=weight_dtype)
    return (
        torch.empty((batch, steps, hidden), device=query.device, dtype=query.dtype),
        weights,
        weights.clone(),
        weights.clone(),
    )


_admitted_three_way_read_compact_dense_op.register_kernel(("cpu", "cuda"))(_compact_dense_forward)


def _compact_dense_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs[:11], *inputs[12:16], *output[1:])
    ctx.mode = int(inputs[11])
    ctx.score_budget = int(inputs[16])


def _compact_dense_backward(
    ctx,
    grad_output: Tensor | None,
    grad_short_weights: Tensor | None,
    grad_permanent_weights: Tensor | None,
    grad_induct_weights: Tensor | None,
):
    (
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
        bank_cocktail,
        bank_key_bias,
        bank_admission,
        scale,
        bank_idx,
        bank_valid,
        short_gain,
        pad_gain,
        induct_gain,
        cocktail_gain,
        short_weights,
        permanent_weights,
        induct_weights,
    ) = ctx.saved_tensors
    if grad_output is None:
        grad_output = torch.zeros_like(query)
    chunk = _chunk_width(query, bank_keys, ctx.score_budget)
    query_gradients = []
    cocktail_gradients = []
    accumulated: list[Tensor | None] = [None] * 13
    total_steps = query.shape[1]
    for start in range(0, total_steps, chunk):
        stop = min(total_steps, start + chunk)
        query_chunk = query[:, start:stop]
        cocktail_chunk = cocktail[:, start:stop]
        allow, induct_allow, eligible, induct_eligible = _compact_masks(
            query_chunk,
            bank_idx,
            bank_valid,
            mode=ctx.mode,
            query_offset=start,
            total_steps=total_steps,
        )
        context = SimpleNamespace(
            saved_tensors=(
                query_chunk,
                bank_keys,
                bank_writes,
                next_writes,
                cocktail_chunk,
                bank_cocktail,
                bank_key_bias,
                bank_admission,
                scale,
                allow,
                induct_allow,
                eligible,
                induct_eligible,
                short_gain,
                pad_gain,
                induct_gain,
                cocktail_gain,
                short_weights[:, start:stop],
                permanent_weights[:, start:stop],
                induct_weights[:, start:stop],
            )
        )
        gradients = _admitted_three_way_read_backward(
            context,
            grad_output[:, start:stop],
            None if grad_short_weights is None else grad_short_weights[:, start:stop],
            (None if grad_permanent_weights is None else grad_permanent_weights[:, start:stop]),
            None if grad_induct_weights is None else grad_induct_weights[:, start:stop],
        )
        query_gradients.append(gradients[0])
        cocktail_gradients.append(gradients[4])
        for target, source_index in enumerate((1, 2, 3, 5, 6, 7, 8, 13, 14, 15, 16)):
            value = gradients[source_index]
            slot = target + 2
            accumulated[slot] = value if accumulated[slot] is None else accumulated[slot] + value
    assert all(value is not None for value in accumulated[2:13])
    return (
        torch.cat(query_gradients, dim=1),
        accumulated[2],
        accumulated[3],
        accumulated[4],
        torch.cat(cocktail_gradients, dim=1),
        accumulated[5],
        accumulated[6],
        accumulated[7],
        accumulated[8],
        None,
        None,
        None,
        accumulated[9],
        accumulated[10],
        accumulated[11],
        accumulated[12],
        None,
    )


torch.library.register_autograd(
    _admitted_three_way_read_compact_dense_op,
    _compact_dense_backward,
    setup_context=_compact_dense_setup,
)
torch.library.register_autocast(
    _admitted_three_way_read_compact_dense_op,
    "cpu",
    torch.bfloat16,
)


@torch.library.register_vmap(_admitted_three_way_read_compact_dense_op)
def _compact_dense_vmap(info, in_dims, *inputs):
    del info

    def select(value, dim, index):
        return value if dim is None else value.movedim(dim, 0)[index]

    batch_size = next(value.shape[dim] for value, dim in zip(inputs, in_dims) if dim is not None)
    outputs = [
        _admitted_three_way_read_compact_dense_op(
            *(select(value, dim, index) for value, dim in zip(inputs, in_dims))
        )
        for index in range(batch_size)
    ]
    return tuple(torch.stack(values) for values in zip(*outputs)), (0, 0, 0, 0)


def admitted_three_way_read_compact_dense(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    bank_idx: Tensor,
    bank_valid: Tensor,
    *,
    mode: str,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
    score_budget: int,
) -> Tensor:
    if mode not in {"seq", "field"}:
        raise ValueError("compact admitted-read mode must be 'seq' or 'field'")
    return _admitted_three_way_read_compact_dense_op(
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
        bank_cocktail,
        bank_key_bias,
        bank_admission,
        scale,
        bank_idx,
        bank_valid,
        0 if mode == "seq" else 1,
        short_gain,
        pad_gain,
        induct_gain,
        cocktail_gain,
        int(score_budget),
    )[0]


def admitted_three_way_read_compact_dense_bmm(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    bank_idx: Tensor,
    bank_valid: Tensor,
    *,
    mode: str,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
    score_budget: int,
) -> Tensor:
    if mode not in {"seq", "field"}:
        raise ValueError("compact dense BMM admitted-read mode must be 'seq' or 'field'")
    return _compact_dense_bmm_op(
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
        bank_cocktail,
        bank_key_bias,
        bank_admission,
        scale,
        bank_idx,
        bank_valid,
        0 if mode == "seq" else 1,
        short_gain,
        pad_gain,
        induct_gain,
        cocktail_gain,
        int(score_budget),
    )


__all__ = [
    "admitted_three_way_read_compact_dense",
    "admitted_three_way_read_compact_dense_bmm",
]
