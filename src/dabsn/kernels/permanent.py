"""Canonical permanent associative delta-scan primitive."""

from __future__ import annotations

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

_CPU_NATIVE_ENABLED = False
_CPU_NATIVE_REQUIRED = False
_CPU_NATIVE_HITS = 0


def enable_native_cpu_permanent(*, required: bool = False) -> bool:
    """Enable the C++/OpenMP permanent scan for CPU tensors."""

    global _CPU_NATIVE_ENABLED, _CPU_NATIVE_REQUIRED
    from .cpu import _load_ext

    extension = _load_ext(required=required)
    _CPU_NATIVE_ENABLED = extension is not None
    _CPU_NATIVE_REQUIRED = bool(required)
    return _CPU_NATIVE_ENABLED


def _permanent_reference(k: Tensor, v: Tensor, beta: Tensor) -> Tensor:
    """Plain-autograd reference delta scan returning ``[B,T,H]``."""
    b, t, h = k.shape
    state = torch.zeros(b, h, h, device=k.device, dtype=k.dtype)
    outs: List[Tensor] = []
    for i in range(t):
        ki = k[:, i, :]
        recon = torch.einsum("bvk,bk->bv", state, ki)
        outs.append(recon)
        err = v[:, i, :] - recon
        state = state + beta[:, i].view(b, 1, 1) * torch.einsum("bv,bk->bvk", err, ki)
    return torch.stack(outs, dim=1)


# ---------------------------------------------------------------------------
# Torch forward/backward primitives (device-agnostic; analytic gradient).
# ---------------------------------------------------------------------------
def _scan_forward(k: Tensor, v: Tensor, beta: Tensor) -> Tensor:
    """Run the recurrence without building an autograd graph."""
    b, t, h = k.shape
    state = torch.zeros(b, h, h, device=k.device, dtype=k.dtype)
    outs: List[Tensor] = []
    for i in range(t):
        ki = k[:, i, :]
        recon = torch.einsum("bvk,bk->bv", state, ki)
        outs.append(recon)
        err = v[:, i, :] - recon
        state = state + beta[:, i].view(b, 1, 1) * torch.einsum("bv,bk->bvk", err, ki)
    return torch.stack(outs, dim=1)


def _scan_backward(k: Tensor, v: Tensor, beta: Tensor, g: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Analytic reverse pass. g = dL/d(recons) [B,T,h] -> (dk, dv, dbeta).

    Recomputes the S_i stack forward (O(T*h^2) transient, freed at return), then
    walks i=T-1..0 applying the adjoint recurrence above.
    """
    b, t, h = k.shape
    # forward, caching S_i (state BEFORE step i) and r_i for the reverse pass
    state = torch.zeros(b, h, h, device=k.device, dtype=k.dtype)
    S_stack: List[Tensor] = []
    r_stack: List[Tensor] = []
    for i in range(t):
        ki = k[:, i, :]
        S_stack.append(state)
        recon = torch.einsum("bvk,bk->bv", state, ki)
        r_stack.append(recon)
        err = v[:, i, :] - recon
        state = state + beta[:, i].view(b, 1, 1) * torch.einsum("bv,bk->bvk", err, ki)

    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta)
    A = torch.zeros(b, h, h, device=k.device, dtype=k.dtype)  # dL/dS_{i+1}
    for i in range(t - 1, -1, -1):
        ki = k[:, i, :]
        vi = v[:, i, :]
        gi = g[:, i, :]
        Si = S_stack[i]
        ri = r_stack[i]
        bi = beta[:, i].view(b, 1)
        ei = vi - ri

        Ak = torch.einsum("bvk,bk->bv", A, ki)  # A k_i              [B,h_v]
        dv[:, i, :] = bi * Ak
        dbeta[:, i] = torch.einsum("bv,bv->b", Ak, ei)  # <A k_i, e_i>

        Stg = torch.einsum("bvk,bv->bk", Si, gi)  # S_i^T g_i          [B,h_k]
        AtV = torch.einsum("bvk,bv->bk", A, vi)  # A^T v_i            [B,h_k]
        StAk = torch.einsum("bvk,bv->bk", Si, Ak)  # S_i^T (A k_i)      [B,h_k]
        AtSk = torch.einsum("bvk,bv->bk", A, ri)  # A^T (S_i k_i)      [B,h_k]
        dk[:, i, :] = Stg + bi * AtV - bi * (StAk + AtSk)

        # A_i = g_i k_i^T + A - beta_i (A k_i) k_i^T
        A = (
            torch.einsum("bv,bk->bvk", gi, ki)
            + A
            - beta[:, i].view(b, 1, 1) * torch.einsum("bv,bk->bvk", Ak, ki)
        )
    return dk, dv, dbeta


# ---------------------------------------------------------------------------
# Triton fused forward (single launch over T; one program per (batch, v-row block)).
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _permanent_fwd_kernel(
        K,
        V,
        BETA,
        OUT,
        B,
        T,
        H,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_bb,
        stride_bt,
        BV: tl.constexpr,
        BH: tl.constexpr,
    ):
        # program owns one batch and a BV-block of value rows; holds that slab of
        # state [BV, H] in registers and streams the whole sequence through it.
        pid_b = tl.program_id(0)
        pid_v = tl.program_id(1)
        v_off = pid_v * BV + tl.arange(0, BV)  # [BV] value-dim rows
        h_off = tl.arange(0, BH)  # [BH] key-dim cols
        v_mask = v_off < H
        h_mask = h_off < H
        state = tl.zeros((BV, BH), dtype=tl.float32)  # fast-weight slab S[v,k]
        for i in range(0, T):
            kp = K + pid_b * stride_kb + i * stride_kt + h_off * stride_kh
            ki = tl.load(kp, mask=h_mask, other=0.0)  # [BH]
            recon = tl.sum(state * ki[None, :], axis=1)  # [BV]  S_i k_i
            op = OUT + pid_b * stride_kb + i * stride_kt + v_off * stride_kh
            tl.store(op, recon, mask=v_mask)
            vp = V + pid_b * stride_kb + i * stride_kt + v_off * stride_kh
            vi = tl.load(vp, mask=v_mask, other=0.0)  # [BV]
            bp = BETA + pid_b * stride_bb + i * stride_bt
            bi = tl.load(bp)
            err = vi - recon  # [BV]
            state = state + bi * err[:, None] * ki[None, :]  # rank-1 delta update

    @triton.jit
    def _permanent_bwd_kernel(
        K,
        V,
        BETA,
        GRAD_OUT,
        DKEY,
        DVALUE,
        DBETA,
        STATE_TAPE,
        B,
        T,
        H,
        stride_kb,
        stride_kt,
        stride_kh,
        stride_bb,
        stride_bt,
        BV: tl.constexpr,
        BH: tl.constexpr,
    ):
        # Each program owns one batch and a block of value rows. It first
        # reconstructs the exact pre-step state rows, then runs their reverse
        # adjoint. Key and beta gradients are reductions across row blocks, so
        # those two destinations use atomic accumulation.
        pid_b = tl.program_id(0)
        pid_v = tl.program_id(1)
        v_off = pid_v * BV + tl.arange(0, BV)
        h_off = tl.arange(0, BH)
        v_mask = v_off < H
        h_mask = h_off < H
        slab_mask = v_mask[:, None] & h_mask[None, :]
        state = tl.zeros((BV, BH), dtype=tl.float32)
        for i in range(0, T):
            tape_off = ((pid_b * T + i) * H + v_off[:, None]) * H + h_off[None, :]
            tl.store(STATE_TAPE + tape_off, state, mask=slab_mask)
            ki = tl.load(
                K + pid_b * stride_kb + i * stride_kt + h_off * stride_kh,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            recon = tl.sum(state * ki[None, :], axis=1)
            vi = tl.load(
                V + pid_b * stride_kb + i * stride_kt + v_off * stride_kh,
                mask=v_mask,
                other=0.0,
            ).to(tl.float32)
            bi = tl.load(BETA + pid_b * stride_bb + i * stride_bt).to(tl.float32)
            state = state + bi * (vi - recon)[:, None] * ki[None, :]

        adjoint = tl.zeros((BV, BH), dtype=tl.float32)
        for reverse_index in range(0, T):
            i = T - 1 - reverse_index
            tape_off = ((pid_b * T + i) * H + v_off[:, None]) * H + h_off[None, :]
            state_i = tl.load(STATE_TAPE + tape_off, mask=slab_mask, other=0.0).to(tl.float32)
            ki = tl.load(
                K + pid_b * stride_kb + i * stride_kt + h_off * stride_kh,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            vi = tl.load(
                V + pid_b * stride_kb + i * stride_kt + v_off * stride_kh,
                mask=v_mask,
                other=0.0,
            ).to(tl.float32)
            gi = tl.load(
                GRAD_OUT + pid_b * stride_kb + i * stride_kt + v_off * stride_kh,
                mask=v_mask,
                other=0.0,
            ).to(tl.float32)
            bi = tl.load(BETA + pid_b * stride_bb + i * stride_bt).to(tl.float32)
            recon = tl.sum(state_i * ki[None, :], axis=1)
            adjoint_key = tl.sum(adjoint * ki[None, :], axis=1)
            tl.store(
                DVALUE + pid_b * stride_kb + i * stride_kt + v_off * stride_kh,
                bi * adjoint_key,
                mask=v_mask,
            )
            beta_part = tl.sum(adjoint_key * (vi - recon), axis=0)
            tl.atomic_add(DBETA + pid_b * stride_bb + i * stride_bt, beta_part)
            key_part = tl.sum(
                state_i * gi[:, None]
                + bi * adjoint * vi[:, None]
                - bi * state_i * adjoint_key[:, None]
                - bi * adjoint * recon[:, None],
                axis=0,
            )
            tl.atomic_add(
                DKEY + pid_b * stride_kb + i * stride_kt + h_off * stride_kh,
                key_part,
                mask=h_mask,
            )
            adjoint = gi[:, None] * ki[None, :] + adjoint - bi * adjoint_key[:, None] * ki[None, :]


def _triton_forward(k: Tensor, v: Tensor, beta: Tensor) -> Tensor:
    b, t, h = k.shape
    out = torch.empty_like(v)
    BH = triton.next_power_of_2(h)
    BV = BH
    grid = (b, triton.cdiv(h, BV))
    _permanent_fwd_kernel[grid](
        k,
        v,
        beta,
        out,
        b,
        t,
        h,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        beta.stride(0),
        beta.stride(1),
        BV=BV,
        BH=BH,
    )
    return out


def _triton_backward(
    k: Tensor, v: Tensor, beta: Tensor, grad_out: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    b, t, h = k.shape
    dkey = torch.zeros_like(k)
    dvalue = torch.empty_like(v)
    dbeta = torch.zeros_like(beta)
    state_tape = torch.empty((b, t, h, h), device=k.device, dtype=torch.float32)
    BH = triton.next_power_of_2(h)
    BV = min(16, BH)
    grid = (b, triton.cdiv(h, BV))
    _permanent_bwd_kernel[grid](
        k,
        v,
        beta,
        grad_out,
        dkey,
        dvalue,
        dbeta,
        state_tape,
        b,
        t,
        h,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        beta.stride(0),
        beta.stride(1),
        BV=BV,
        BH=BH,
    )
    return dkey, dvalue, dbeta


# ---------------------------------------------------------------------------
# autograd.Function: fused forward + analytic backward.
# ---------------------------------------------------------------------------
class PermanentDeltaScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k: Tensor, v: Tensor, beta: Tensor) -> Tensor:
        if k.is_cuda:
            ctx.backend = "cuda"
            ctx.save_for_backward(k, v, beta)
            if not _HAS_TRITON:
                raise RuntimeError("Triton permanent-pad kernel is unavailable on CUDA")
            try:
                return _triton_forward(k.contiguous(), v.contiguous(), beta.contiguous())
            except Exception as exc:
                raise RuntimeError(f"Triton permanent-pad forward failed: {exc}") from exc
        if _CPU_NATIVE_ENABLED:
            global _CPU_NATIVE_HITS
            from .cpu import _load_ext

            extension = _load_ext(required=_CPU_NATIVE_REQUIRED)
            if extension is None:
                if _CPU_NATIVE_REQUIRED:
                    raise RuntimeError("required native CPU permanent scan is unavailable")
            else:
                key = k.float().contiguous()
                value = v.float().contiguous()
                gate = beta.float().contiguous()
                ctx.backend = "cpu"
                ctx.original_dtypes = (k.dtype, v.dtype, beta.dtype)
                ctx.save_for_backward(key, value, gate)
                _CPU_NATIVE_HITS += 1
                return extension.permanent_delta_scan_cpu(key, value, gate).to(v.dtype)
        ctx.backend = "reference"
        ctx.save_for_backward(k, v, beta)
        return _scan_forward(k, v, beta)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        k, v, beta = ctx.saved_tensors
        if ctx.backend == "cuda":
            if not _HAS_TRITON:
                raise RuntimeError("Triton permanent-pad backward is unavailable on CUDA")
            try:
                dk, dv, dbeta = _triton_backward(
                    k.float().contiguous(),
                    v.float().contiguous(),
                    beta.float().contiguous(),
                    grad_out.float().contiguous(),
                )
                return dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype)
            except Exception as exc:
                raise RuntimeError(f"Triton permanent-pad backward failed: {exc}") from exc
        if ctx.backend == "cpu":
            from .cpu import _load_ext

            extension = _load_ext(required=True)
            dk, dv, dbeta = extension.permanent_delta_scan_bwd_cpu(
                k, v, beta, grad_out.float().contiguous()
            )
            kd, vd, bd = ctx.original_dtypes
            return dk.to(kd), dv.to(vd), dbeta.to(bd)
        dk, dv, dbeta = _scan_backward(k, v, beta, grad_out.contiguous())
        return dk, dv, dbeta


@torch.library.custom_op("dabsn::permanent_delta_scan", mutates_args=())
def _permanent_delta_scan_op(k: Tensor, v: Tensor, beta: Tensor) -> Tensor:
    """Stable dispatcher boundary around the selected scan implementation."""

    return PermanentDeltaScan.apply(k, v, beta)


@_permanent_delta_scan_op.register_fake
def _permanent_delta_scan_fake(k: Tensor, v: Tensor, beta: Tensor) -> Tensor:
    if k.dim() != 3 or v.shape != k.shape or beta.shape != k.shape[:2]:
        raise ValueError("expected k/v [B,T,H] and beta [B,T]")
    return torch.empty_like(v)


def _permanent_delta_scan_setup(ctx, inputs, output) -> None:
    k, v, beta = inputs
    ctx.save_for_backward(k, v, beta)


def _permanent_delta_scan_registered_backward(ctx, grad_out: Tensor):
    k, v, beta = ctx.saved_tensors
    if k.is_cuda:
        dk, dv, dbeta = _triton_backward(
            k.float().contiguous(),
            v.float().contiguous(),
            beta.float().contiguous(),
            grad_out.float().contiguous(),
        )
        return dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype)
    if _CPU_NATIVE_ENABLED:
        from .cpu import _load_ext

        extension = _load_ext(required=_CPU_NATIVE_REQUIRED)
        if extension is not None:
            dk, dv, dbeta = extension.permanent_delta_scan_bwd_cpu(
                k.float().contiguous(),
                v.float().contiguous(),
                beta.float().contiguous(),
                grad_out.float().contiguous(),
            )
            return dk.to(k.dtype), dv.to(v.dtype), dbeta.to(beta.dtype)
    return _scan_backward(k, v, beta, grad_out.contiguous())


torch.library.register_autograd(
    _permanent_delta_scan_op,
    _permanent_delta_scan_registered_backward,
    setup_context=_permanent_delta_scan_setup,
)


@torch.library.register_vmap(_permanent_delta_scan_op)
def _permanent_delta_scan_vmap(info, in_dims, k, v, beta):
    batch_size = info.batch_size

    def select(value, dim, index):
        return value if dim is None else value.movedim(dim, 0)[index]

    outputs = [
        _permanent_delta_scan_op(
            select(k, in_dims[0], index),
            select(v, in_dims[1], index),
            select(beta, in_dims[2], index),
        )
        for index in range(batch_size)
    ]
    return torch.stack(outputs), 0


def permanent_delta_scan(k: Tensor, v: Tensor, beta: Tensor) -> Tensor:
    return _permanent_delta_scan_op(k, v, beta)


def permanent_delta_read(
    expression: Tensor,
    long_state: Tensor,
    progress: Tensor,
    plasticity: Tensor,
    energy: Tensor,
    saturation: Tensor,
    *,
    phase_gain: Tensor,
    progress_gain: Tensor,
    beta_gain: Tensor,
    consolidation_energy_gain: Tensor,
    consolidation_saturation_gain: Tensor,
    beta_bias: Tensor,
) -> Tensor:
    key = torch.nn.functional.normalize(
        expression + phase_gain * long_state + progress_gain * progress,
        dim=-1,
    )
    beta = torch.sigmoid(
        beta_gain * plasticity.mean(dim=-1)
        + consolidation_energy_gain * energy.mean(dim=-1)
        + consolidation_saturation_gain * saturation.mean(dim=-1)
        + beta_bias
    )
    return permanent_delta_scan(key, expression, beta)


def permanent_runtime_status() -> dict[str, object]:
    return {
        "cpu_native_enabled": bool(_CPU_NATIVE_ENABLED),
        "cpu_native_hits": int(_CPU_NATIVE_HITS),
        "triton_available": bool(_HAS_TRITON),
        "cuda_available": torch.cuda.is_available(),
        "import_error": None if _TRITON_IMPORT_ERROR is None else repr(_TRITON_IMPORT_ERROR),
    }
