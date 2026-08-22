"""Output-only registered ABI for the front-packed compact flash admitted read."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import Tensor

from .compact_admitted import _compact_dense_backward, _compact_dense_forward

_REFERENCE_SCORE_BUDGET = 8_388_608
_CUDA_REGISTERED = False
CompactGradients = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


def _reference_forward(
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
) -> Tensor:
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
        _REFERENCE_SCORE_BUDGET,
    )[0]


@torch.library.custom_op("dabsn::admitted_three_way_read_compact_flash", mutates_args=())
def _compact_flash_op(
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
) -> Tensor:
    return _reference_forward(
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
    )


@_compact_flash_op.register_fake
def _compact_flash_fake(
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
) -> Tensor:
    del (
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
    )
    return torch.empty_like(query)


_compact_flash_op.register_kernel("cpu")(_reference_forward)


def _reference_backward(grad_output: Tensor, *inputs) -> tuple[Tensor, ...]:
    outputs = _compact_dense_forward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        inputs[9],
        inputs[10],
        inputs[11],
        inputs[12],
        inputs[13],
        inputs[14],
        inputs[15],
        _REFERENCE_SCORE_BUDGET,
    )
    context = SimpleNamespace(
        saved_tensors=(*inputs[:11], *inputs[12:16], *outputs[1:]),
        mode=int(inputs[11]),
        score_budget=_REFERENCE_SCORE_BUDGET,
    )
    gradients = _compact_dense_backward(context, grad_output, None, None, None)
    return (*gradients[:9], *gradients[12:16])


@torch.library.custom_op(
    "dabsn::_admitted_three_way_read_compact_flash_backward",
    mutates_args=(),
)
def _compact_flash_backward_op(
    grad_output: Tensor,
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
) -> CompactGradients:
    return cast(
        CompactGradients,
        _reference_backward(
            grad_output,
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
        ),
    )


@_compact_flash_backward_op.register_fake
def _compact_flash_backward_fake(
    grad_output: Tensor,
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
) -> CompactGradients:
    del grad_output, bank_idx, bank_valid, mode
    return cast(
        CompactGradients,
        tuple(
            torch.empty_like(value)
            for value in (
                query,
                bank_keys,
                bank_writes,
                next_writes,
                cocktail,
                bank_cocktail,
                bank_key_bias,
                bank_admission,
                scale,
                short_gain,
                pad_gain,
                induct_gain,
                cocktail_gain,
            )
        ),
    )


_compact_flash_backward_op.register_kernel("cpu")(_reference_backward)


def _compact_flash_setup(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs[:11], *inputs[12:])
    ctx.mode = int(inputs[11])


def _compact_flash_backward(ctx, grad_output: Tensor):
    tensors = ctx.saved_tensors
    gradients = _compact_flash_backward_op(
        grad_output,
        *tensors[:11],
        ctx.mode,
        *tensors[11:],
    )
    return (*gradients[:9], None, None, None, *gradients[9:])


torch.library.register_autograd(
    _compact_flash_op,
    _compact_flash_backward,
    setup_context=_compact_flash_setup,
)
torch.library.register_autocast(_compact_flash_op, "cpu", torch.bfloat16)


@torch.library.register_vmap(_compact_flash_op)
def _compact_flash_vmap(info, in_dims, *inputs):
    def select(value, dim, index):
        return value if dim is None else value.movedim(dim, 0)[index]

    outputs = [
        _compact_flash_op(*(select(value, dim, index) for value, dim in zip(inputs, in_dims)))
        for index in range(info.batch_size)
    ]
    return torch.stack(outputs), 0


def register_compact_flash_cuda(runtime: Any) -> bool:
    """Attach the existing exact Triton forward/backward to the stable operator ABI."""
    global _CUDA_REGISTERED
    if _CUDA_REGISTERED:
        return False

    def cuda_forward(*inputs):
        context = SimpleNamespace(save_for_backward=lambda *values: None)
        return runtime.CompactFlashAdmittedThreeWayReadTritonFunction.forward(context, *inputs)

    def cuda_backward(grad_output: Tensor, *inputs):
        context = SimpleNamespace(
            saved_tensors=(*inputs[:11], *inputs[12:]),
            mode=int(inputs[11]),
        )
        gradients = runtime.CompactFlashAdmittedThreeWayReadTritonFunction.backward(
            context, grad_output
        )
        return (*gradients[:9], *gradients[12:16])

    _compact_flash_op.register_kernel("cuda")(cuda_forward)
    _compact_flash_backward_op.register_kernel("cuda")(cuda_backward)
    _CUDA_REGISTERED = True
    return True


def admitted_three_way_read_compact_flash(
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
) -> Tensor:
    if mode not in {"seq", "field"}:
        raise ValueError("compact flash admitted-read mode must be 'seq' or 'field'")
    return _compact_flash_op(
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
    )


__all__ = ["admitted_three_way_read_compact_flash"]
