"""Fused carried-state linear recurrence for canonical DABSN long memory."""

from __future__ import annotations

import os as _os
from typing import List, Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
    _TRITON_IMPORT_ERROR = None
except Exception as exc:
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = exc


def _linrec_reference(a: Tensor, g: Tensor, y_init: Tensor) -> Tensor:
    """Plain-autograd scan y_t = a_t*y_{t-1} + g_t, y_{-1}=y_init. Returns [B,T,h]."""
    b, t, h = a.shape
    y = y_init
    outs: List[Tensor] = []
    for i in range(t):
        y = a[:, i, :] * y + g[:, i, :]
        outs.append(y)
    return torch.stack(outs, dim=1)


# ---------------------------------------------------------------------------
# Torch forward/backward primitives (device-agnostic; analytic gradient).
# ---------------------------------------------------------------------------
def _linrec_forward(a: Tensor, g: Tensor, y_init: Tensor) -> Tensor:
    """Run the forward scan without building an autograd graph."""
    if (not a.is_cuda) and a.dtype in (torch.float32, torch.float16, torch.bfloat16):
        try:
            from .cpu import _load_ext
            ext = _load_ext()
            if ext is not None and hasattr(ext, "linrec_forward_cpu"):
                return ext.linrec_forward_cpu(a, g, y_init)
        except Exception:
            pass
    b, t, h = a.shape
    y = y_init
    outs: List[Tensor] = []
    for i in range(t):
        y = a[:, i, :] * y + g[:, i, :]
        outs.append(y)
    return torch.stack(outs, dim=1)


def _linrec_backward(
    a: Tensor, g: Tensor, y_init: Tensor, gy: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Analytic reverse pass. gy = dL/dy [B,T,h] -> (da, dg, dy_init).

    y_t = a_t y_{t-1} + g_t is linear in y_{t-1}, so with the total adjoint
    c_t = dL/dy_t = gy_t + a_{t+1} c_{t+1}  (reverse, c_T = 0):
        dg_t     = c_t
        da_t     = c_t * y_{t-1}
        dy_init  = a_0 * c_0
    Recomputes the y_{t-1} stack forward (O(T*h) transient, freed at return).
    """
    if (not a.is_cuda) and a.dtype in (torch.float32, torch.float16, torch.bfloat16):
        try:
            from .cpu import _load_ext
            ext = _load_ext()
            if ext is not None and hasattr(ext, "linrec_backward_cpu"):
                da, dg, dy_init = ext.linrec_backward_cpu(a, g, y_init, gy)
                return da, dg, dy_init
        except Exception:
            pass
    b, t, h = a.shape
    y_prev: List[Tensor] = []          # y_{t-1} for each t (y_{-1}=y_init)
    y = y_init
    for i in range(t):
        y_prev.append(y)
        y = a[:, i, :] * y + g[:, i, :]
    yprev = torch.stack(y_prev, dim=1)  # [B,T,h]

    da = torch.zeros_like(a)
    dg = torch.zeros_like(g)
    carry = torch.zeros(b, h, device=a.device, dtype=a.dtype)  # c_{t+1}
    for i in range(t - 1, -1, -1):
        a_next = a[:, i + 1, :] if i + 1 < t else torch.zeros_like(carry)
        c = gy[:, i, :] + a_next * carry          # c_t
        dg[:, i, :] = c
        da[:, i, :] = c * yprev[:, i, :]
        carry = c
    dy_init = a[:, 0, :] * carry                  # carry == c_0 after the loop
    return da, dg, dy_init


# ---------------------------------------------------------------------------
# Triton fused forward (single launch over T; one program per (batch, h-block)).
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _linrec_fwd_kernel(
        A, G, YINIT, OUT,
        B, T, H,
        stride_ab, stride_at, stride_ah,
        stride_gb, stride_gt, stride_gh,
        stride_ob, stride_ot, stride_oh,
        stride_ib, stride_ih,
        BH: tl.constexpr,
    ):
        # program owns one batch and a BH-block of hidden dims; carries y in registers
        # and streams the whole sequence through the diagonal recurrence.
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        h_off = pid_h * BH + tl.arange(0, BH)
        h_mask = h_off < H
        # Carry the recurrence state in fp32 (accumulation precision). bf16/fp16 inputs
        # are cast on load so the loop-carried `y` keeps one type for the whole loop;
        # Triton rejects a loop var whose type changes between iterations.
        y = tl.load(YINIT + pid_b * stride_ib + h_off * stride_ih, mask=h_mask, other=0.0).to(tl.float32)
        for i in range(0, T):
            a_off = pid_b * stride_ab + i * stride_at + h_off * stride_ah
            g_off = pid_b * stride_gb + i * stride_gt + h_off * stride_gh
            o_off = pid_b * stride_ob + i * stride_ot + h_off * stride_oh
            a = tl.load(A + a_off, mask=h_mask, other=0.0).to(tl.float32)
            g = tl.load(G + g_off, mask=h_mask, other=0.0).to(tl.float32)
            y = a * y + g
            tl.store(OUT + o_off, y, mask=h_mask)

    @triton.jit
    def _linrec_bwd_kernel(
        A, G, YINIT, GY, DA, DG, DYINIT, YPREV,
        B, T, H,
        stride_b, stride_t, stride_h,
        stride_ib, stride_ih,
        BH: tl.constexpr,
    ):
        # One program per (batch, h-block): recompute the forward y_{t-1} stack into
        # YPREV (registers carry y), then run the reverse adjoint scan in the same
        # launch. Mirrors _linrec_backward exactly:
        #   c_t = gy_t + a_{t+1} c_{t+1} (c_T=0); dg_t=c_t; da_t=c_t*y_{t-1};
        #   dy_init = a_0 * c_0.
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        h_off = pid_h * BH + tl.arange(0, BH)
        h_mask = h_off < H
        base = pid_b * stride_b + h_off * stride_h
        # forward: store y_{t-1} (the state BEFORE step t) at slot t.
        # All recurrence state in fp32 (loads cast on the way in) so the loop-carried
        # vars keep one consistent type -- Triton rejects a bf16->fp32 type change.
        y = tl.load(YINIT + pid_b * stride_ib + h_off * stride_ih, mask=h_mask, other=0.0).to(tl.float32)
        for i in range(0, T):
            off = base + i * stride_t
            tl.store(YPREV + off, y, mask=h_mask)
            a = tl.load(A + off, mask=h_mask, other=0.0).to(tl.float32)
            g = tl.load(G + off, mask=h_mask, other=0.0).to(tl.float32)
            y = a * y + g
        # reverse: total adjoint carry c_{t+1}.
        carry = tl.zeros((BH,), dtype=tl.float32)
        for ridx in range(0, T):
            i = T - 1 - ridx
            off = base + i * stride_t
            if i + 1 < T:
                a_next = tl.load(A + base + (i + 1) * stride_t, mask=h_mask, other=0.0).to(tl.float32)
            else:
                a_next = tl.zeros((BH,), dtype=tl.float32)
            c = tl.load(GY + off, mask=h_mask, other=0.0).to(tl.float32) + a_next * carry
            tl.store(DG + off, c, mask=h_mask)
            yprev = tl.load(YPREV + off, mask=h_mask, other=0.0).to(tl.float32)
            tl.store(DA + off, c * yprev, mask=h_mask)
            carry = c
        a0 = tl.load(A + base, mask=h_mask, other=0.0).to(tl.float32)          # a_0
        tl.store(DYINIT + pid_b * stride_ib + h_off * stride_ih, a0 * carry, mask=h_mask)


def _triton_linrec_backward_once(a: Tensor, g: Tensor, y_init: Tensor, gy: Tensor):
    """Compute ``da``, ``dg``, and ``dy_init`` in one Triton reverse scan."""
    b, t, h = a.shape
    da = torch.empty_like(a)
    dg = torch.empty_like(g)
    dy_init = torch.empty_like(y_init)
    yprev = torch.empty_like(a)                        # scratch y_{t-1} stack
    BH = triton.next_power_of_2(h)
    grid = (b, triton.cdiv(h, BH))
    _linrec_bwd_kernel[grid](
        a, g, y_init, gy, da, dg, dy_init, yprev,
        b, t, h,
        a.stride(0), a.stride(1), a.stride(2),
        y_init.stride(0), y_init.stride(1),
        BH=BH,
    )
    return da, dg, dy_init


def _triton_linrec_backward(a: Tensor, g: Tensor, y_init: Tensor, gy: Tensor):
    """Triton recurrence backward, chunked over time for long-context training.

    Reverse-mode dependencies cross chunk boundaries through exactly one tensor:
    the final state of a chunk is the initial state of the next chunk. Add the
    adjoint from the next chunk to the last local output, then run the proven
    single-chunk backward and carry its dy_init to the previous chunk.
    """
    t = int(a.shape[1])
    chunk = int(_os.environ.get("DABSN_LONG_SCAN_CHUNK", "8192"))
    if chunk <= 0 or t <= chunk:
        return _triton_linrec_backward_once(a, g, y_init, gy)

    starts = list(range(0, t, chunk))
    inits: List[Tensor] = []
    carry = y_init
    for start in starts:
        end = min(start + chunk, t)
        inits.append(carry)
        out = _triton_linrec_once(a[:, start:end, :], g[:, start:end, :], carry)
        carry = out[:, -1, :].contiguous()

    da = torch.empty_like(a)
    dg = torch.empty_like(g)
    carry_grad = torch.zeros_like(y_init)
    for start, init in zip(reversed(starts), reversed(inits)):
        end = min(start + chunk, t)
        gy_chunk = gy[:, start:end, :].contiguous().clone()
        gy_chunk[:, -1, :] = gy_chunk[:, -1, :] + carry_grad
        da_chunk, dg_chunk, carry_grad = _triton_linrec_backward_once(
            a[:, start:end, :].contiguous(),
            g[:, start:end, :].contiguous(),
            init.contiguous(),
            gy_chunk,
        )
        da[:, start:end, :] = da_chunk
        dg[:, start:end, :] = dg_chunk
    return da, dg, carry_grad


def _triton_linrec_once(a: Tensor, g: Tensor, y_init: Tensor) -> Tensor:
    b, t, h = a.shape
    out = torch.empty((b, t, h), device=a.device, dtype=a.dtype)
    BH = triton.next_power_of_2(h)
    grid = (b, triton.cdiv(h, BH))
    _linrec_fwd_kernel[grid](
        a, g, y_init, out,
        b, t, h,
        a.stride(0), a.stride(1), a.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        y_init.stride(0), y_init.stride(1),
        BH=BH,
    )
    return out


def _triton_linrec(a: Tensor, g: Tensor, y_init: Tensor) -> Tensor:
    """Triton recurrence forward, chunked over time for very long contexts.

    A single Triton program scans the whole T axis, which is fine for shorter
    sequences but too large for long-context runs. Chunking is implementation
    tiling: each chunk runs the same recurrence, and the exact final state of one
    chunk becomes the initial state of the next.
    """
    t = int(a.shape[1])
    chunk = int(_os.environ.get("DABSN_LONG_SCAN_CHUNK", "8192"))
    if chunk <= 0 or t <= chunk:
        return _triton_linrec_once(a, g, y_init)
    outs: List[Tensor] = []
    carry = y_init
    for start in range(0, t, chunk):
        end = min(start + chunk, t)
        out = _triton_linrec_once(a[:, start:end, :], g[:, start:end, :], carry)
        outs.append(out)
        carry = out[:, -1, :].contiguous()
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# autograd.Function: fused forward + analytic backward.
# ---------------------------------------------------------------------------
class LinearRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: Tensor, g: Tensor, y_init: Tensor) -> Tensor:
        ctx.save_for_backward(a, g, y_init)
        if a.is_cuda:
            if not _HAS_TRITON:
                raise RuntimeError("Triton long recurrence kernel is unavailable on CUDA")
            try:
                return _triton_linrec(a.contiguous(), g.contiguous(), y_init.contiguous())
            except Exception as exc:
                raise RuntimeError(f"Triton long recurrence forward failed: {exc}") from exc
        return _linrec_forward(a, g, y_init)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        a, g, y_init = ctx.saved_tensors
        if a.is_cuda:
            if not _HAS_TRITON:
                raise RuntimeError("Triton long recurrence backward kernel is unavailable on CUDA")
            try:
                da, dg, dy_init = _triton_linrec_backward(
                    a.contiguous(), g.contiguous(), y_init.contiguous(), grad_out.contiguous())
                return da, dg, dy_init
            except Exception as exc:
                raise RuntimeError(f"Triton long recurrence backward failed: {exc}") from exc
        da, dg, dy_init = _linrec_backward(a, g, y_init, grad_out.contiguous())
        return da, dg, dy_init


def linear_recurrence(a: Tensor, g: Tensor, y_init: Tensor) -> Tensor:
    return LinearRecurrence.apply(a, g, y_init)

def fused_long_scan(
    read,
    write: Tensor,
    plasticity: Tensor,
    novelty: Tensor,
) -> Tensor:
    return fused_long_scan_from_state(
        read,
        write,
        plasticity,
        novelty,
        initial_state=None,
        return_final_state=False,
    )


def fused_long_scan_from_state(
    read,
    write: Tensor,
    plasticity: Tensor,
    novelty: Tensor,
    *,
    initial_state: tuple[Tensor, Tensor, Tensor] | None = None,
    return_final_state: bool = False,
):
    del novelty
    if bool(getattr(type(read), "_long_native_required", False)) and not write.is_cuda:
        raise RuntimeError(
            "Triton/CUDA long-memory recurrence was required, but received a CPU tensor"
        )
    batch, _seq_len, hidden = write.shape
    zeros = torch.zeros(batch, hidden, device=write.device, dtype=write.dtype)
    ones = torch.ones(batch, hidden, device=write.device, dtype=write.dtype)
    if initial_state is None:
        long_initial, expected_initial, retention_initial = zeros, zeros, ones
    else:
        long_initial, expected_initial, retention_initial = (
            state.to(device=write.device, dtype=write.dtype) for state in initial_state
        )
        expected_shape = (batch, hidden)
        if any(state.shape != expected_shape for state in (
            long_initial, expected_initial, retention_initial
        )):
            raise ValueError(
                f"initial long-scan state must have shape {expected_shape}"
            )
    retain = torch.sigmoid(read.logit_retain)

    expectation_retain = torch.sigmoid(read.logit_expect_retain)
    expected = linear_recurrence(
        expectation_retain.expand_as(write),
        (1.0 - expectation_retain) * write,
        expected_initial,
    )
    expected_previous = torch.cat(
        [expected_initial.unsqueeze(1), expected[:, :-1, :]],
        dim=1,
    )
    prediction_error = torch.tanh((write - expected_previous).abs())
    plastic_salience = plasticity * prediction_error

    retention_decay = torch.sigmoid(read.logit_retention_decay)
    retention_strength = torch.sigmoid(read.logit_retention_strength)
    retention = linear_recurrence(
        retention_decay.expand_as(write),
        (1.0 - retention_decay) * (1.0 - prediction_error),
        retention_initial,
    )
    effective_retain = (
        retain + (1.0 - retain) * retention_strength * retention
    )
    long_update = (1.0 - effective_retain) * (plastic_salience * write)
    long_sequence = linear_recurrence(effective_retain, long_update, long_initial)
    read._last_long_backend = (
        "cuda_triton" if write.is_cuda else "cpu_native_cpp"
    )
    if return_final_state:
        return long_sequence, (
            long_sequence[:, -1, :],
            expected[:, -1, :],
            retention[:, -1, :],
        )
    return long_sequence


def enable_fused_long_read(*, required_cuda: bool = False) -> bool:
    from dabsn.read import DABSNRead

    if required_cuda and not (_HAS_TRITON and torch.cuda.is_available()):
        raise RuntimeError(
            "Triton/CUDA long-memory recurrence was required but is unavailable"
        )
    if not hasattr(DABSNRead, "_eager_long_scan"):
        DABSNRead._eager_long_scan = DABSNRead._long_scan
    if not hasattr(DABSNRead, "_eager_long_scan_from_state"):
        DABSNRead._eager_long_scan_from_state = DABSNRead._long_scan_from_state
    DABSNRead._long_scan = fused_long_scan
    DABSNRead._long_scan_from_state = fused_long_scan_from_state
    DABSNRead._long_native_enabled = True
    DABSNRead._long_native_required = bool(required_cuda)
    return True


def long_runtime_status() -> dict[str, object]:
    from dabsn.read import DABSNRead

    return {
        "triton_available": bool(_HAS_TRITON),
        "cuda_available": torch.cuda.is_available(),
        "enabled": bool(getattr(DABSNRead, "_long_native_enabled", False)),
        "import_error": None if _TRITON_IMPORT_ERROR is None else repr(_TRITON_IMPORT_ERROR),
    }
