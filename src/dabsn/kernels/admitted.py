"""Registered dense admitted three-way read with an analytic backward."""

from __future__ import annotations

import torch
from torch import Tensor
from torch._subclasses.fake_tensor import is_fake

from dabsn._torch_compat import register_custom_op_autocast


def _read_crossover_calibration_bytes(
    dtype: torch.dtype,
    bank_entries: int,
    hidden: int,
    *,
    query_steps: int | None = None,
    batch: int = 4,
) -> int:
    """Conservative peak-byte estimate for the synthetic CUDA read calibration."""
    bank_entries = int(bank_entries)
    hidden = int(hidden)
    batch = int(batch)
    if bank_entries < 0 or hidden <= 0 or batch <= 0:
        raise ValueError("calibration shape requires bank_entries >= 0, hidden > 0, batch > 0")
    query_steps = int(
        query_steps if query_steps is not None else max(64, min(bank_entries, 256))
    )
    if query_steps <= 0:
        raise ValueError("calibration query_steps must be positive")
    value_bytes = torch.empty((), dtype=dtype).element_size()

    # Persistent synthetic inputs plus the compact path's possible front-pack
    # outputs: query, three full-width banks twice, and width-4/scalar metadata.
    value_elements = (
        batch * query_steps * hidden
        + 6 * batch * bank_entries * hidden
        + 2 * batch * query_steps * 4
        + 4 * batch * bank_entries * 4
    )
    # Two index banks (input + packed output), validity masks, and conservative
    # dense score/allow temporaries exercised by the comparison path.
    index_bytes = 2 * batch * bank_entries * torch.empty((), dtype=torch.int64).element_size()
    mask_bytes = 2 * batch * bank_entries
    score_bytes = 16 * batch * query_steps * bank_entries
    return int(value_elements * value_bytes + index_bytes + mask_bytes + score_bytes)


def _read_crossover_should_measure(
    dtype: torch.dtype,
    bank_entries: int,
    hidden: int,
    *,
    max_bytes: int,
    free_bytes: int | None = None,
) -> bool:
    """Whether synthetic read calibration fits both absolute and live budgets."""
    max_bytes = int(max_bytes)
    if max_bytes <= 0:
        raise ValueError("calibration max_bytes must be positive")
    budget = max_bytes
    if free_bytes is not None:
        free_bytes = int(free_bytes)
        if free_bytes < 0:
            raise ValueError("calibration free_bytes must be non-negative")
        budget = min(budget, max(free_bytes // 4, 1))
    return _read_crossover_calibration_bytes(dtype, bank_entries, hidden) <= budget


def _frontpack_admitted_bank(
    memory_key: Tensor,
    write_memory: Tensor,
    next_write_memory: Tensor,
    memory_cocktail: Tensor,
    key_bias: Tensor,
    admission_gate: Tensor,
    bank_idx: Tensor,
    bank_valid: Tensor,
) -> tuple[Tensor, ...]:
    """Front-pack a valid mask without copying an order that is already packed.

    A stable descending sort of a True-prefix mask is the identity.  Ordinary
    eager execution returns contiguous inputs directly in that case, avoiding
    redundant full-bank copies during large carried-memory inference.  Compiled
    and CUDA-graph-captured execution keeps the value-oblivious gather path
    because inspecting a device mask would introduce a host sync.
    """
    tensors = (
        memory_key,
        write_memory,
        next_write_memory,
        memory_cocktail,
        key_bias,
        admission_gate,
        bank_idx,
        bank_valid,
    )
    compiling = bool(torch.compiler.is_compiling())
    capturing = False
    if bank_valid.is_cuda and not compiling:
        try:
            capturing = bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError:
            # A CUDA runtime without an initialized current stream cannot be
            # capturing. Preserve the ordinary eager path in that case.
            capturing = False
    can_inspect = (
        not compiling
        and not capturing
        and not is_fake(bank_valid)
        and bank_valid.device.type != "meta"
        and bank_valid.dtype == torch.bool
        and bank_valid.ndim == 2
        and all(tensor.is_contiguous() for tensor in tensors)
    )
    if can_inspect:
        counts = bank_valid.to(torch.int64).sum(dim=1, keepdim=True)
        positions = torch.arange(bank_valid.shape[1], device=bank_valid.device).unsqueeze(0)
        if torch.equal(bank_valid, positions < counts):
            return tensors

    order = torch.argsort(
        bank_valid.detach().to(torch.int32),
        dim=1,
        descending=True,
        stable=True,
    )
    idx = order.unsqueeze(-1)
    return (
        torch.gather(memory_key, 1, idx.expand(-1, -1, memory_key.shape[-1])),
        torch.gather(write_memory, 1, idx.expand(-1, -1, write_memory.shape[-1])),
        torch.gather(
            next_write_memory,
            1,
            idx.expand(-1, -1, next_write_memory.shape[-1]),
        ),
        torch.gather(
            memory_cocktail,
            1,
            idx.expand(-1, -1, memory_cocktail.shape[-1]),
        ),
        torch.gather(key_bias, 1, order),
        torch.gather(admission_gate, 1, order),
        torch.gather(bank_idx, 1, order),
        torch.gather(bank_valid, 1, order),
    )


def _masked_softmax(scores: Tensor, allow: Tensor, eligible: Tensor) -> Tensor:
    masked = scores.masked_fill(~allow, float("-inf"))
    masked = masked.masked_fill(~eligible.unsqueeze(-1), 0.0)
    weights = torch.softmax(masked, dim=-1)
    return torch.where(eligible.unsqueeze(-1), weights, torch.zeros_like(weights))


@torch.library.custom_op("dabsn::admitted_three_way_read", mutates_args=())
def _admitted_three_way_read_op(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    allow: Tensor,
    induct_allow: Tensor,
    eligible: Tensor,
    induct_eligible: Tensor,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    output_dtype = query.dtype
    compute_dtype = torch.float32 if query.dtype in {torch.float16, torch.bfloat16} else query.dtype
    query = query.to(compute_dtype)
    bank_keys = bank_keys.to(compute_dtype)
    bank_writes = bank_writes.to(compute_dtype)
    next_writes = next_writes.to(compute_dtype)
    cocktail = cocktail.to(compute_dtype)
    bank_cocktail = bank_cocktail.to(compute_dtype)
    bank_key_bias = bank_key_bias.to(compute_dtype)
    bank_admission = bank_admission.to(compute_dtype)
    scale = scale.to(compute_dtype)
    short_gain = short_gain.to(compute_dtype)
    pad_gain = pad_gain.to(compute_dtype)
    induct_gain = induct_gain.to(compute_dtype)
    cocktail_gain = cocktail_gain.to(compute_dtype)
    compatibility = torch.bmm(query, bank_keys.transpose(1, 2)) * scale
    cocktail_compatibility = torch.bmm(cocktail, bank_cocktail.transpose(1, 2)) * cocktail_gain
    content = compatibility + bank_key_bias.unsqueeze(1)
    short_scores = content + bank_admission.unsqueeze(1)
    permanent_scores = content + cocktail_compatibility + bank_admission.unsqueeze(1)
    short_weights = _masked_softmax(short_scores, allow, eligible)
    permanent_weights = _masked_softmax(permanent_scores, allow, eligible)
    induct_weights = _masked_softmax(short_scores, induct_allow, induct_eligible)
    short_read = torch.bmm(short_weights, bank_writes)
    permanent_read = torch.bmm(permanent_weights, bank_writes)
    induct_read = torch.bmm(induct_weights, next_writes)
    output = (short_gain * short_read + pad_gain * permanent_read + induct_gain * induct_read).to(
        output_dtype
    )
    return output, short_weights, permanent_weights, induct_weights


@_admitted_three_way_read_op.register_fake
def _admitted_three_way_read_fake(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    allow: Tensor,
    induct_allow: Tensor,
    eligible: Tensor,
    induct_eligible: Tensor,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del (
        bank_writes,
        next_writes,
        cocktail,
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
    batch, steps, hidden = query.shape
    bank = bank_keys.shape[1]
    weight_dtype = torch.float32 if query.dtype in {torch.float16, torch.bfloat16} else query.dtype
    weights = query.new_empty((batch, steps, bank), dtype=weight_dtype)
    return query.new_empty((batch, steps, hidden)), weights, weights.clone(), weights.clone()


def _admitted_three_way_read_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs, *output[1:])


def _softmax_backward(
    grad_weights: Tensor,
    weights: Tensor,
    allow: Tensor,
    eligible: Tensor,
) -> Tensor:
    grad_scores = weights * (grad_weights - (grad_weights * weights).sum(dim=-1, keepdim=True))
    return grad_scores.masked_fill(~allow, 0.0).masked_fill(~eligible.unsqueeze(-1), 0.0)


def _admitted_three_way_read_backward(
    ctx,
    grad_output,
    grad_short_weights,
    grad_permanent_weights,
    grad_induct_weights,
):
    (
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
        bank_cocktail,
        _bank_key_bias,
        _bank_admission,
        scale,
        allow,
        induct_allow,
        eligible,
        induct_eligible,
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

    compute_dtype = short_weights.dtype
    grad_output_compute = grad_output.to(compute_dtype)
    query_compute = query.to(compute_dtype)
    bank_keys_compute = bank_keys.to(compute_dtype)
    bank_writes_compute = bank_writes.to(compute_dtype)
    next_writes_compute = next_writes.to(compute_dtype)
    cocktail_compute = cocktail.to(compute_dtype)
    bank_cocktail_compute = bank_cocktail.to(compute_dtype)
    scale_compute = scale.to(compute_dtype)
    short_gain_compute = short_gain.to(compute_dtype)
    pad_gain_compute = pad_gain.to(compute_dtype)
    induct_gain_compute = induct_gain.to(compute_dtype)
    cocktail_gain_compute = cocktail_gain.to(compute_dtype)

    short_read = torch.bmm(short_weights, bank_writes_compute)
    permanent_read = torch.bmm(permanent_weights, bank_writes_compute)
    induct_read = torch.bmm(induct_weights, next_writes_compute)
    grad_short = (
        torch.bmm(grad_output_compute, bank_writes_compute.transpose(1, 2)) * short_gain_compute
    )
    grad_permanent = (
        torch.bmm(grad_output_compute, bank_writes_compute.transpose(1, 2)) * pad_gain_compute
    )
    grad_induct = (
        torch.bmm(grad_output_compute, next_writes_compute.transpose(1, 2)) * induct_gain_compute
    )
    if grad_short_weights is not None:
        grad_short = grad_short + grad_short_weights
    if grad_permanent_weights is not None:
        grad_permanent = grad_permanent + grad_permanent_weights
    if grad_induct_weights is not None:
        grad_induct = grad_induct + grad_induct_weights

    grad_short_scores = _softmax_backward(grad_short, short_weights, allow, eligible)
    grad_permanent_scores = _softmax_backward(grad_permanent, permanent_weights, allow, eligible)
    grad_induct_scores = _softmax_backward(
        grad_induct, induct_weights, induct_allow, induct_eligible
    )
    grad_content = grad_short_scores + grad_permanent_scores + grad_induct_scores
    grad_cocktail_scores = grad_permanent_scores

    grad_query = torch.bmm(grad_content, bank_keys_compute) * scale_compute
    grad_bank_keys = torch.bmm(grad_content.transpose(1, 2), query_compute) * scale_compute
    raw_compatibility = torch.bmm(query_compute, bank_keys_compute.transpose(1, 2))
    grad_scale = (grad_content * raw_compatibility).sum().reshape_as(scale)
    grad_cocktail = torch.bmm(grad_cocktail_scores, bank_cocktail_compute) * cocktail_gain_compute
    grad_bank_cocktail = (
        torch.bmm(grad_cocktail_scores.transpose(1, 2), cocktail_compute) * cocktail_gain_compute
    )
    raw_cocktail = torch.bmm(cocktail_compute, bank_cocktail_compute.transpose(1, 2))
    grad_cocktail_gain = (grad_cocktail_scores * raw_cocktail).sum().reshape_as(cocktail_gain)
    grad_bank_writes = (
        torch.bmm(short_weights.transpose(1, 2), grad_output_compute) * short_gain_compute
        + torch.bmm(permanent_weights.transpose(1, 2), grad_output_compute) * pad_gain_compute
    )
    grad_next_writes = (
        torch.bmm(induct_weights.transpose(1, 2), grad_output_compute) * induct_gain_compute
    )
    grad_bias = grad_content.sum(dim=1)
    grad_admission = grad_content.sum(dim=1)
    grad_short_gain = (grad_output_compute * short_read).sum().reshape_as(short_gain)
    grad_pad_gain = (grad_output_compute * permanent_read).sum().reshape_as(pad_gain)
    grad_induct_gain = (grad_output_compute * induct_read).sum().reshape_as(induct_gain)
    return (
        grad_query.to(query.dtype),
        grad_bank_keys.to(bank_keys.dtype),
        grad_bank_writes.to(bank_writes.dtype),
        grad_next_writes.to(next_writes.dtype),
        grad_cocktail.to(cocktail.dtype),
        grad_bank_cocktail.to(bank_cocktail.dtype),
        grad_bias.to(_bank_key_bias.dtype),
        grad_admission.to(_bank_admission.dtype),
        grad_scale.to(scale.dtype),
        None,
        None,
        None,
        None,
        grad_short_gain.to(short_gain.dtype),
        grad_pad_gain.to(pad_gain.dtype),
        grad_induct_gain.to(induct_gain.dtype),
        grad_cocktail_gain.to(cocktail_gain.dtype),
    )


torch.library.register_autograd(
    _admitted_three_way_read_op,
    _admitted_three_way_read_backward,
    setup_context=_admitted_three_way_read_setup,
)


@torch.library.register_vmap(_admitted_three_way_read_op)
def _admitted_three_way_read_vmap(info, in_dims, *inputs):
    def select(value, dim, index):
        return value if dim is None else value.movedim(dim, 0)[index]

    outputs = [
        _admitted_three_way_read_op(
            *(select(value, dim, index) for value, dim in zip(inputs, in_dims))
        )
        for index in range(info.batch_size)
    ]
    return tuple(torch.stack(items) for items in zip(*outputs)), (0, 0, 0, 0)


def admitted_three_way_read(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    allow: Tensor,
    induct_allow: Tensor,
    eligible: Tensor,
    induct_eligible: Tensor,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
) -> Tensor:
    return _admitted_three_way_read_op(
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
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
    )[0]


def _native_backward_reference(
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
    allow: Tensor,
    induct_allow: Tensor,
    eligible: Tensor,
    induct_eligible: Tensor,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
) -> tuple[Tensor, ...]:
    outputs = _admitted_three_way_read_op(
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
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

    class Context:
        saved_tensors = (
            query,
            bank_keys,
            bank_writes,
            next_writes,
            cocktail,
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
            *outputs[1:],
        )

    gradients = _admitted_three_way_read_backward(Context(), grad_output, None, None, None)
    return (
        gradients[0],
        gradients[1],
        gradients[2],
        gradients[3],
        gradients[4],
        gradients[5],
        gradients[6],
        gradients[7],
        gradients[8],
        gradients[13],
        gradients[14],
        gradients[15],
        gradients[16],
    )


@torch.library.custom_op("dabsn::_admitted_three_way_read_backward", mutates_args=())
def _admitted_three_way_read_native_backward_op(
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
    allow: Tensor,
    induct_allow: Tensor,
    eligible: Tensor,
    induct_eligible: Tensor,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
) -> tuple[
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
]:
    gradients = _native_backward_reference(
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
        allow,
        induct_allow,
        eligible,
        induct_eligible,
        short_gain,
        pad_gain,
        induct_gain,
        cocktail_gain,
    )
    return (
        gradients[0],
        gradients[1],
        gradients[2],
        gradients[3],
        gradients[4],
        gradients[5],
        gradients[6],
        gradients[7],
        gradients[8],
        gradients[9],
        gradients[10],
        gradients[11],
        gradients[12],
    )


@_admitted_three_way_read_native_backward_op.register_fake
def _admitted_three_way_read_native_backward_fake(
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
    allow: Tensor,
    induct_allow: Tensor,
    eligible: Tensor,
    induct_eligible: Tensor,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
) -> tuple[
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
]:
    del grad_output, allow, induct_allow, eligible, induct_eligible
    return (
        torch.empty_like(query),
        torch.empty_like(bank_keys),
        torch.empty_like(bank_writes),
        torch.empty_like(next_writes),
        torch.empty_like(cocktail),
        torch.empty_like(bank_cocktail),
        torch.empty_like(bank_key_bias),
        torch.empty_like(bank_admission),
        torch.empty_like(scale),
        torch.empty_like(short_gain),
        torch.empty_like(pad_gain),
        torch.empty_like(induct_gain),
        torch.empty_like(cocktail_gain),
    )


@torch.library.custom_op("dabsn::admitted_three_way_read_native", mutates_args=())
def _admitted_three_way_read_native_op(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    allow: Tensor,
    induct_allow: Tensor,
    eligible: Tensor,
    induct_eligible: Tensor,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
) -> Tensor:
    return admitted_three_way_read(
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
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


@_admitted_three_way_read_native_op.register_fake
def _admitted_three_way_read_native_fake(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    allow: Tensor,
    induct_allow: Tensor,
    eligible: Tensor,
    induct_eligible: Tensor,
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
        allow,
        induct_allow,
        eligible,
        induct_eligible,
        short_gain,
        pad_gain,
        induct_gain,
        cocktail_gain,
    )
    return torch.empty_like(query)


def _admitted_three_way_read_native_setup(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _admitted_three_way_read_native_backward(ctx, grad_output: Tensor):
    gradients = _admitted_three_way_read_native_backward_op(grad_output, *ctx.saved_tensors)
    return (
        gradients[0],
        gradients[1],
        gradients[2],
        gradients[3],
        gradients[4],
        gradients[5],
        gradients[6],
        gradients[7],
        gradients[8],
        None,
        None,
        None,
        None,
        gradients[9],
        gradients[10],
        gradients[11],
        gradients[12],
    )


torch.library.register_autograd(
    _admitted_three_way_read_native_op,
    _admitted_three_way_read_native_backward,
    setup_context=_admitted_three_way_read_native_setup,
)
register_custom_op_autocast(
    _admitted_three_way_read_op,
    "cpu",
    torch.bfloat16,
)
register_custom_op_autocast(
    _admitted_three_way_read_native_op,
    "cpu",
    torch.bfloat16,
)


@torch.library.register_vmap(_admitted_three_way_read_native_op)
def _admitted_three_way_read_native_vmap(info, in_dims, *inputs):
    del info

    def select(value, dim, index):
        return value if dim is None else value.movedim(dim, 0)[index]

    batch_size = next(value.shape[dim] for value, dim in zip(inputs, in_dims) if dim is not None)
    outputs = [
        _admitted_three_way_read_native_op(
            *(select(value, dim, index) for value, dim in zip(inputs, in_dims))
        )
        for index in range(batch_size)
    ]
    return torch.stack(outputs), 0


def admitted_three_way_read_native(
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    allow: Tensor,
    induct_allow: Tensor,
    eligible: Tensor,
    induct_eligible: Tensor,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
) -> Tensor:
    return _admitted_three_way_read_native_op(
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
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


__all__ = ["admitted_three_way_read", "admitted_three_way_read_native"]
