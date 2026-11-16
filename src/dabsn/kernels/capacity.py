"""Capacity native dispatch, opaque compiler boundaries and analytic autograd.

The native sparse tape contains selected rows only: O(B*T*A), not O(B*T*C).
Online replay updates the persistent eligibility table at selected rows only.
Ordinary BPTT materializes parameter gradients once; online returns COO rows.

Sparse parameter-gradient scratch is sized by ``capacity_gradient_layout``:
one slot per selected event (B*T*A) or one slot per Capacity row per
sequence (B*C), whichever is smaller. Event slots alone grow with sequence
length and make long-context sparse training impractical.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch
from torch import Tensor

_BACKEND = "reference"
_EXT = None
_ERROR = None
_CALLS = {"cpu": 0, "cuda": 0}
_DEFER_ROWS = False
_DEFER_SLOTS = {}


@contextmanager
def capture_capacity_rows(module=None):
    """Keep data-dependent COO construction outside a CUDA graph recording."""
    global _DEFER_ROWS, _DEFER_SLOTS
    if module is not None:
        from ..capacity import DABSNCapacitySubsystem
        if not any(isinstance(child, DABSNCapacitySubsystem) and child.mode == "sparse"
                   for child in module.modules()):
            yield
            return
    previous = _DEFER_ROWS
    previous_slots = _DEFER_SLOTS
    _DEFER_ROWS = True
    _DEFER_SLOTS = {}
    original_forward = module.forward if module is not None else None
    owned_forward = module is not None and "forward" in module.__dict__

    @wraps(original_forward)
    def reset_slots(*args, **kwargs):
        _DEFER_SLOTS.clear()
        return original_forward(*args, **kwargs)

    # make_graphed_callables rejects module hooks. A scoped forward wrapper
    # resets slot numbering on warmup/capture without changing that contract.
    if module is not None:
        module.forward = reset_slots
    try:
        yield
    finally:
        _DEFER_ROWS = previous
        _DEFER_SLOTS = previous_slots
        # Successful graph construction installs a new forward. Preserve it;
        # on an eager/failed build, restore the original attribute ownership.
        if module is not None and module.forward is reset_slots:
            if owned_forward:
                module.forward = original_forward
            else:
                del module.forward


def finalize_capacity_gradients(module):
    """Materialize pending captured row contributions once, outside capture.

    The device generation counter advances on *every replay*. Python bookkeeping
    alone would reuse stale gradients or incorrectly skip subsequent replays.
    """
    parameters = module.parameters() if hasattr(module, "parameters") else module
    for parameter in parameters:
        slots = getattr(parameter, "_capacity_pending_rows", None)
        if slots is None:
            continue
        for pending in slots.values():
            generation = int(pending["generation"].item())
            if generation == pending["consumed"]:
                continue
            indices, values = pending["indices"], pending["values"]
            if pending.get("row_totals", False):
                # Fixed-size per-row totals: keep exactly the rows this replay
                # selected, including selected rows whose gradient is zero.
                indices = _bucket_rows(torch.unique(indices, sorted=True),
                                       pending.get("bucket_width", 0))
                values = values.index_select(0, indices)
            if parameter.grad is not None:
                prior = parameter.grad
                if not prior.is_sparse:
                    raise RuntimeError("captured Capacity rows require sparse gradient accumulation")
                indices = torch.cat((prior._indices()[0], indices))
                values = torch.cat((prior._values().to(values.dtype), values))
            rows, inverse = torch.unique(indices, sorted=True, return_inverse=True)
            reduced = _row_reduce(inverse, values, rows.numel()).to(parameter.dtype)
            parameter.grad = torch.sparse_coo_tensor(rows[None], reduced,
                size=parameter.shape, is_coalesced=True, check_invariants=False)
            pending["consumed"] = generation


def _bucket_rows(bases, width):
    """Expand sorted unique bucket base rows into their sorted selected rows."""
    if not width:
        return bases
    return (bases[:, None] + torch.arange(width, device=bases.device)).reshape(-1)


def _load_cpu():
    global _EXT, _ERROR
    if _EXT is not None:
        return _EXT
    if _ERROR is not None:
        raise RuntimeError(f"Capacity CPU extension unavailable: {_ERROR}")
    try:
        from torch.utils.cpp_extension import load
        from .cpu import _prepare_cpp_build_env
        _prepare_cpp_build_env()
        directory = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / "dabsn_capacity_cpu"
        directory.mkdir(parents=True, exist_ok=True)
        _EXT = load("dabsn_capacity_cpu", [str(Path(__file__).with_name("capacity_cpu.cpp"))],
                    build_directory=str(directory),
                    extra_cflags=["-O3", "-std=c++17", "-ffp-contract=off"], verbose=False)
    except Exception as exc:
        _ERROR = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"Capacity CPU extension unavailable: {_ERROR}") from exc
    return _EXT


def enable_capacity(backend: str, *, required: bool = False) -> bool:
    global _BACKEND
    if backend not in ("reference", "cpu", "cuda"):
        raise ValueError("Capacity backend must be reference, cpu or cuda")
    try:
        if backend == "cpu":
            _load_cpu()
        elif backend == "cuda":
            from .capacity_triton import available
            if not available():
                raise RuntimeError("Capacity Triton CUDA runtime is unavailable")
    except Exception:
        if required:
            raise
        return False
    _BACKEND = backend
    return True


def capacity_backend(x: Tensor) -> str:
    if _BACKEND == x.device.type:
        return "cpu_native_cpp" if x.device.type == "cpu" else "cuda_triton"
    return "reference"


def capacity_operand_dtype(x: Tensor) -> torch.dtype:
    if x.dtype != torch.float64 and torch.is_autocast_enabled(x.device.type):
        dtype = torch.get_autocast_dtype(x.device.type)
        if dtype == torch.bfloat16 and x.is_cuda:
            from ._arch import bf16_kernels_supported

            if not bf16_kernels_supported(x.device):
                # Pre-Ampere: no bf16 tensor cores; keep FP32 operands.
                return torch.float32
        return dtype
    return x.dtype


def capacity_gradient_slots(batch: int, steps: int, active: int, rows: int) -> dict[str, int]:
    """Parameter-gradient table rows each sparse layout needs (per parameter).

    ``event``: one slot per selected (sequence, token, active row) event.
    ``row``: one slot per Capacity row per sequence (the kernels' private-row
    accumulation), summed over sequences afterwards.
    """
    return {"event": int(batch) * int(steps) * int(active), "row": int(batch) * int(rows)}


def capacity_gradient_layout(batch: int, steps: int, active: int, rows: int) -> str:
    """Select the sparse gradient layout from shapes alone (graph-capture safe).

    Row totals are used only when strictly smaller. That condition also
    guarantees ``rows < B*T*A``, which keeps CUDA in its race-free private-row
    mode (each sequence lane owns its own row slots).
    """
    slots = capacity_gradient_slots(batch, steps, active, rows)
    return "row" if slots["row"] < slots["event"] else "event"


def capacity_runtime_status() -> dict:
    return {"backend": _BACKEND, "cpu_loaded": _EXT is not None,
            "cpu_error": _ERROR, "cpu_calls": _CALLS["cpu"], "cuda_calls": _CALLS["cuda"]}


@torch.library.custom_op("dabsn::capacity_row_reduce", mutates_args=())
def _row_reduce(indices: Tensor, values: Tensor, row_count: int) -> Tensor:
    """Reduce selected row contributions in a reproducible order."""
    if values.device.type == "cuda":
        from .capacity_triton import row_reduce
        return row_reduce(indices, values, row_count)
    return values.new_zeros((row_count, *values.shape[1:])).index_add_(0, indices, values)


@_row_reduce.register_fake
def _row_reduce_fake(indices, values, row_count):
    return values.new_empty((row_count, *values.shape[1:]))


@torch.library.custom_op("dabsn::capacity_address", mutates_args=())
def _address(x: Tensor, projection: Tensor, active: int) -> Tensor:
    if x.device.type == "cpu":
        return _load_cpu().address(x, projection, active)
    from .capacity_triton import address
    return address(x, projection, active)


@_address.register_fake
def _address_fake(x, projection, active):
    return x.new_empty((*x.shape[:2], active), dtype=torch.long)


def capacity_address(x, projection, active):
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    return _address(x.to(dtype).contiguous(), projection.to(device=x.device, dtype=dtype).contiguous(), active)


@torch.library.custom_op("dabsn::capacity_forward", mutates_args={"features"})
def _forward(x: Tensor, key: Tensor, bias: Tensor, logit: Tensor, value: Tensor,
             gain: Tensor, features: Tensor, indices: Tensor, sparse: bool,
             stable_slots: bool, operand_mode: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _CALLS[x.device.type] += 1
    with torch.autocast(device_type=x.device.type, enabled=False):
        if x.device.type == "cpu":
            return tuple(_load_cpu().forward(x, key, bias, logit, value, gain, features, indices, sparse, operand_mode))
        from .capacity_triton import forward
        return tuple(forward(x, key, bias, logit, value, gain, features, indices, sparse, stable_slots, operand_mode))


@_forward.register_fake
def _forward_fake(x, key, bias, logit, value, gain, features, indices, sparse, stable_slots, operand_mode):
    a = indices.shape[2] if sparse else key.shape[0]
    return (x.new_empty((*x.shape[:2], value.shape[1])), *(
        x.new_empty((*x.shape[:2], a)) for _ in range(3)))


@torch.library.custom_op("dabsn::capacity_backward", mutates_args={"ek", "eb", "er"})
def _backward(x: Tensor, key: Tensor, logit: Tensor, value: Tensor, gain: Tensor,
              indices: Tensor, local: Tensor, rows: int, acts: Tensor, old: Tensor,
              states: Tensor, gout: Tensor, gfinal: Tensor, ek: Tensor, eb: Tensor,
              er: Tensor, sparse: bool, online: bool, stable_slots: bool, operand_mode: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    with torch.autocast(device_type=x.device.type, enabled=False):
        if x.device.type == "cpu":
            return tuple(_load_cpu().backward(x, key, logit, value, gain, indices, local, rows,
                                       acts, old, states, gout, gfinal, ek, eb, er, sparse, online, operand_mode))
        from .capacity_triton import backward
        return tuple(backward(x, key, logit, value, gain, indices, local, rows,
                        acts, old, states, gout, gfinal, ek, eb, er, sparse, online, stable_slots, operand_mode))


@_backward.register_fake
def _backward_fake(x, key, logit, value, gain, indices, local, rows, acts, old,
                   states, gout, gfinal, ek, eb, er, sparse, online, stable_slots, operand_mode):
    return (torch.empty_like(x), x.new_empty((rows, key.shape[1])), x.new_empty((rows,)),
            x.new_empty((rows,)), x.new_empty((rows, value.shape[1])), x.new_empty(()),
            x.new_empty((0,)) if online else torch.empty_like(gfinal))


@torch.library.custom_op("dabsn::capacity_grouped_forward", mutates_args={"features"})
def _grouped_forward(x: Tensor, key: Tensor, bias: Tensor, logit: Tensor, value: Tensor,
                     gain: Tensor, features: Tensor, indices: Tensor, sparse: bool,
                     operand_mode: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _CALLS["cuda"] += 1
    from .capacity_triton import grouped_forward
    with torch.autocast(device_type="cuda", enabled=False):
        return tuple(grouped_forward(x, key, bias, logit, value, gain, features, indices,
                                     sparse, operand_mode))


@_grouped_forward.register_fake
def _grouped_forward_fake(x, key, bias, logit, value, gain, features, indices, sparse, operand_mode):
    a = indices.shape[2] if sparse else key.shape[0]
    return (features.new_empty((*x.shape[:2], value.shape[1])), *(
        features.new_empty((*x.shape[:2], a)) for _ in range(3)))


@torch.library.custom_op("dabsn::capacity_grouped_backward", mutates_args=())
def _grouped_backward(x: Tensor, key: Tensor, logit: Tensor, value: Tensor, gain: Tensor,
                      indices: Tensor, acts: Tensor, old: Tensor, states: Tensor, gout: Tensor,
                      gfinal: Tensor, sparse: bool, operand_mode: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    from .capacity_triton import grouped_backward
    with torch.autocast(device_type="cuda", enabled=False):
        return tuple(grouped_backward(x, key, logit, value, gain, indices, acts, old, states,
                                      gout, gfinal, sparse, operand_mode))


@_grouped_backward.register_fake
def _grouped_backward_fake(x, key, logit, value, gain, indices, acts, old, states, gout, gfinal,
                           sparse, operand_mode):
    c = key.shape[0]
    return (states.new_empty(x.shape), states.new_empty((c, key.shape[1])), states.new_empty((c,)),
            states.new_empty((c,)), states.new_empty((c, value.shape[1])), states.new_empty(()),
            torch.empty_like(gfinal))


@torch.library.custom_op("dabsn::capacity_grouped_online_backward", mutates_args={"ek", "eb", "er"})
def _grouped_online_backward(x: Tensor, key: Tensor, logit: Tensor, value: Tensor, gain: Tensor,
                             indices: Tensor, acts: Tensor, old: Tensor, states: Tensor, gout: Tensor,
                             ek: Tensor, eb: Tensor, er: Tensor, sparse: bool,
                             operand_mode: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    from .capacity_triton import grouped_online_backward
    with torch.autocast(device_type="cuda", enabled=False):
        return tuple(grouped_online_backward(x, key, logit, value, gain, indices, acts, old, states,
                                             gout, ek, eb, er, sparse, operand_mode))


@_grouped_online_backward.register_fake
def _grouped_online_backward_fake(x, key, logit, value, gain, indices, acts, old, states, gout,
                                  ek, eb, er, sparse, operand_mode):
    c = key.shape[0]
    return (states.new_empty(x.shape), states.new_empty((c, key.shape[1])), states.new_empty((c,)),
            states.new_empty((c,)), states.new_empty((c, value.shape[1])), states.new_empty(()))


class _CapacityScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, key, bias, logit, value, gain, features, indices, ek, eb, er, sparse, online,
                stable_slots, bucketed=False):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        ctx.operand_mode = 1 if x.dtype == torch.float16 else (2 if x.dtype == torch.bfloat16 else 0)
        ctx.original_dtypes = [v.dtype for v in (x, key, bias, logit, value, gain, features)]
        xx = x.to(dtype).contiguous()
        # CUDA loads convert selected rows in the kernel: never expand an entire
        # BF16 row table to FP32 just to execute a sparse selection.
        kk, bb, rr, vv, gg = [v.contiguous() if x.is_cuda or sparse else v.to(dtype).contiguous()
                            for v in (key, bias, logit, value, gain)]
        zz = features.contiguous() if x.is_cuda else features.to(dtype).contiguous()
        if not online:
            zz = zz.clone()
        # Canonical buckets (or dense execution) on CUDA FP32: grouped GEMMs.
        from .capacity_triton import grouped_supported
        ctx.grouped = bool((bucketed or not sparse) and grouped_supported(xx, sparse, online))
        if ctx.grouped:
            if ctx.operand_mode:
                # Autocast operands stay 16-bit: the input as given, the row
                # tables cast once per call and reused by backward.
                xx = x.contiguous()
                kk, vv = key.to(x.dtype).contiguous(), value.to(x.dtype).contiguous()
            out, acts, old, states = _grouped_forward(xx, kk, bb, rr, vv, gg, zz, indices, sparse,
                                                      ctx.operand_mode)
        else:
            out, acts, old, states = _forward(xx, kk, bb, rr, vv, gg, zz, indices, sparse,
                                             stable_slots, ctx.operand_mode)
        if online:
            if features.dtype != zz.dtype or not features.is_contiguous():
                features.copy_(zz)
            # Final recurrent carry intentionally has no online temporal VJP.
            final = features.detach()
            ctx.mark_non_differentiable(final)
        else:
            final = zz.to(features.dtype)
        ctx.sparse, ctx.online = sparse, online
        ctx.stable_slots = stable_slots
        ctx.defer_rows = bool(_DEFER_ROWS and sparse and online)
        if ctx.defer_rows:
            ctx.row_parameters = (key, bias, logit, value)
            ctx.row_slot = _DEFER_SLOTS.get(id(key), 0)
            _DEFER_SLOTS[id(key)] = ctx.row_slot + 1
        ctx.save_for_backward(xx, kk, rr, vv, gg, indices, acts, old, states, ek, eb, er)
        ctx.set_materialize_grads(False)
        return out.to(x.dtype), final

    @staticmethod
    def backward(ctx, gout, gfinal):
        x, key, logit, value, gain, indices, acts, old, states, ek, eb, er = ctx.saved_tensors
        if gout is None:
            gout = states.new_zeros((*x.shape[:2], value.shape[1]))
        if gfinal is None:
            gfinal = states.new_zeros((x.shape[0], key.shape[0])) if not ctx.online else states.new_empty(0)
        if ctx.grouped and not ctx.online:
            grads = _grouped_backward(x, key, logit, value, gain, indices, acts, old, states,
                                      gout.contiguous(), gfinal.to(states.dtype).contiguous(),
                                      ctx.sparse, ctx.operand_mode)
            dtypes = ctx.original_dtypes
            return (grads[0].to(dtypes[0]), grads[1].to(dtypes[1]), grads[2].to(dtypes[2]),
                    grads[3].to(dtypes[3]), grads[4].to(dtypes[4]), grads[5].to(dtypes[5]),
                    grads[6].to(dtypes[6]), None, None, None, None, None, None, None, None)
        if ctx.grouped:
            # Online grouped credit: dense row totals; traces advance in place.
            ekc, ebc, erc = [v.to(states.dtype).contiguous() for v in (ek, eb, er)]
            grads = list(_grouped_online_backward(x, key, logit, value, gain, indices, acts, old,
                                                  states, gout.contiguous(), ekc, ebc,
                                                  erc, ctx.sparse, ctx.operand_mode))
            grads.append(states.new_empty(0))
            row_totals = bool(ctx.sparse and ctx.defer_rows)
            # Canonical routes select whole buckets: one base row per token
            # identifies every selected row, so uniqueness is over B*T bases.
            rows = indices[..., 0].reshape(-1) if ctx.sparse else indices
            bucket_width = indices.shape[-1] if ctx.sparse else 0
            if ctx.sparse and not ctx.defer_rows:
                rows = _bucket_rows(torch.unique(rows, sorted=True), bucket_width)
                grads[1:5] = [g.index_select(0, rows) for g in grads[1:5]]
        else:
            bucket_width = 0
            row_totals =(ctx.sparse and (ctx.defer_rows or not ctx.online)
                          and capacity_gradient_layout(*indices.shape, key.shape[0]) == "row")
            if row_totals:
                # Accumulate by physical row: scratch is O(B*C), independent of T.
                # ``rows`` keeps the selected row IDs for deferred finalization.
                rows = indices.reshape(-1)
                local = indices
                count = key.shape[0]
            elif ctx.defer_rows or (ctx.sparse and not ctx.online):
                rows = indices.reshape(-1)
                local = torch.arange(indices.numel(), device=x.device).reshape(indices.shape)
                count = indices.numel()
            elif ctx.sparse and ctx.online:
                rows, local = torch.unique(indices, sorted=True, return_inverse=True)
                count = rows.numel()
            else:
                rows, local = indices, indices
                count = key.shape[0]
            # Usually already FP32; FP64 is retained for the numerical oracle.
            ekc, ebc, erc = [v.to(x.dtype).contiguous() for v in (ek, eb, er)]
            grads = _backward(x, key, logit, value, gain, indices, local, count, acts, old,
                              states, gout.to(x.dtype).contiguous(), gfinal.to(x.dtype).contiguous(),
                              ekc, ebc, erc, ctx.sparse, ctx.online, ctx.stable_slots, ctx.operand_mode)
        if ctx.online:
            for original, updated in zip((ek, eb, er), (ekc, ebc, erc)):
                if original.dtype != updated.dtype or not original.is_contiguous():
                    original.copy_(updated)
        shapes = [key.shape, logit.shape, logit.shape, value.shape]
        parameter_grads = []
        for i, shape in enumerate(shapes, 1):
            grad = grads[i]
            if ctx.defer_rows:
                parameter = ctx.row_parameters[i-1]
                slots = getattr(parameter, "_capacity_pending_rows", {})
                previous = slots.get(ctx.row_slot)
                generation = (x.new_zeros((), dtype=torch.int64) if previous is None
                              else previous["generation"])
                generation.add_(1)
                slots[ctx.row_slot] = {
                    "indices": rows, "values": grad, "generation": generation,
                    "consumed": 0 if previous is None else previous["consumed"],
                    "row_totals": row_totals,
                    "bucket_width": bucket_width if ctx.grouped else 0,
                }
                parameter._capacity_pending_rows = slots
                grad = None
            elif ctx.sparse and ctx.online:
                grad = torch.sparse_coo_tensor(rows[None], grad.to(ctx.original_dtypes[i]), size=shape,
                                               is_coalesced=True, check_invariants=False)
            elif ctx.sparse and row_totals:
                # The kernel already summed every selected event into its row.
                grad = grad.to(ctx.original_dtypes[i])
            elif ctx.sparse:
                # Ordinary autograd exposes v4's dense parameter gradients only
                # here. Native scratch contains selected event contributions.
                grad = _row_reduce(rows, grad, shape[0])
                grad = grad.to(ctx.original_dtypes[i])
            else:
                grad = grad.to(ctx.original_dtypes[i])
            parameter_grads.append(grad)
        return (grads[0].to(ctx.original_dtypes[0]), *parameter_grads,
                grads[5].to(ctx.original_dtypes[5]),
                None if ctx.online else grads[6].to(ctx.original_dtypes[6]),
                None, None, None, None, None, None, None, None)


def capacity_scan(x, key, bias, retention_logit, value, gain, features, indices,
                  *, eligibility=None, sparse=False, stable_slots=False, bucketed=False):
    """Native scan with explicit row IDs, also usable by row-sharded execution.

    ``bucketed=True`` promises canonical routes: every token selects one whole
    contiguous bucket, ``indices[b,t,a] == indices[b,t,0] + a`` with
    ``indices[b,t,0] % A == 0``. CUDA BPTT then runs as grouped GEMMs.

    ``features`` is consumed in online mode and immutable in ordinary BPTT.
    Explicit sparse routes must be distinct, valid rows within each token.
    ``stable_slots=True`` additionally promises row IDs modulo A equal their
    slot; canonical contiguous geometric buckets satisfy this. CUDA uses row
    parallelism only under this explicit promise (or dense execution).
    """
    if x.ndim != 3 or key.ndim != 2 or x.shape[-1] != key.shape[1]:
        raise ValueError("Capacity expects x[B,T,D] and key[C,D]")
    x = x.to(capacity_operand_dtype(x))
    c = key.shape[0]
    tensors = (key, bias, retention_logit, value, gain, features, indices)
    if any(t.device != x.device for t in tensors):
        raise ValueError("Capacity inputs, parameters, indices and state must share a device")
    if features.shape != (x.shape[0], c) or bias.shape != (c,) or retention_logit.shape != (c,):
        raise ValueError("Capacity row and recurrent state shapes disagree")
    if value.ndim != 2 or value.shape[0] != c or gain.numel() != 1:
        raise ValueError("Capacity value/gain shapes disagree")
    if sparse and (indices.dtype != torch.long or indices.ndim != 3 or indices.shape[:2] != x.shape[:2]):
        raise ValueError("Capacity sparse indices must be int64 [B,T,A]")
    if sparse:
        if indices.shape[-1] < 1:
            raise ValueError("Capacity needs at least one selected row per token")
        torch._assert_async(((indices >= 0) & (indices < c)).all(), "Capacity row index out of range")
        if stable_slots or bucketed:
            slots = torch.arange(indices.shape[-1],device=indices.device)
            torch._assert_async((indices.remainder(indices.shape[-1]) == slots).all(),
                                "Capacity stable slots must retain each row's slot")
        if bucketed:
            torch._assert_async((indices - indices[..., :1] == slots).all(),
                                "Capacity bucketed routes must select one contiguous bucket")
    if eligibility is None:
        eligibility = tuple(x.new_empty(0) for _ in range(3))
        online = False
    else:
        online = True
        if tuple(t.shape for t in eligibility) != ((x.shape[0], c, key.shape[1]), (x.shape[0], c), (x.shape[0], c)):
            raise ValueError("Capacity eligibility shapes disagree")
        if any(t.device != x.device for t in eligibility):
            raise ValueError("Capacity eligibility must share the input device")
    return _CapacityScan.apply(x, key, bias, retention_logit, value, gain, features,
                               indices.contiguous(), *eligibility, sparse, online, stable_slots,
                               bool(bucketed and sparse))
