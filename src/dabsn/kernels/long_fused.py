"""Fused CUDA long memory: the canonical LONG prediction-error recurrence in one pass.

Per batch row and hidden unit, exactly the reference recurrence of
``DABSNRead._long_scan_from_state``::

    pe_t  = tanh(|w_t - e_{t-1}|)
    e_t   = a_e e_{t-1} + (1 - a_e) w_t
    r_t   = a_r r_{t-1} + (1 - a_r)(1 - pe_t)
    eff_t = R + (1 - R) S r_t
    L_t   = eff_t L_{t-1} + (1 - eff_t) p_t pe_t w_t

with ``R = sigmoid(logit_retain)``, ``a_e = sigmoid(logit_expect_retain)``,
``a_r = sigmoid(logit_retention_decay)``, ``S = sigmoid(logit_retention_strength)``.

The forward is one kernel carrying all three states in FP32 registers (the
previous implementation ran three separate scans with the elementwise stages
in between). The backward is one reverse kernel with the analytic adjoint of
all three coupled recurrences, including the final-state gradients. Programs
cover (batch, 32-unit hidden slice), so long sequences at small batch still
run many programs. No atomics; parameter gradients reduce per program and then
in a fixed order, so the result is bitwise deterministic.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False

_BH = 32
_CHUNK = 256  # time-chunk length of the parallel long-sequence forward


if _HAS_TRITON:

    @triton.jit
    def _tanh(x):
        e = tl.exp(-2.0 * tl.abs(x))
        y = (1.0 - e) / (1.0 + e)
        return tl.where(x >= 0, y, -y)

    @triton.jit
    def _long_fwd_kernel(W, P, L0, E0, R0, LR, LE, LD, LS, OUT, EPREV, RSEQ, LF, EF, RF,
                         T, H, sw_b, sw_t, sw_h, sp_b, sp_t, sp_h,
                         SAVE: tl.constexpr, INIT: tl.constexpr, BH: tl.constexpr):
        pid = tl.program_id(0)
        nh = tl.cdiv(H, BH)
        b = (pid // nh).to(tl.int64)
        h = (pid % nh) * BH + tl.arange(0, BH)
        hm = h < H
        R = tl.sigmoid(tl.load(LR).to(tl.float32))
        ae = tl.sigmoid(tl.load(LE).to(tl.float32))
        ar = tl.sigmoid(tl.load(LD).to(tl.float32))
        S = tl.sigmoid(tl.load(LS).to(tl.float32))
        if INIT:
            L = tl.load(L0 + b * H + h, hm, 0.0)
            e = tl.load(E0 + b * H + h, hm, 0.0)
            r = tl.load(R0 + b * H + h, hm, 0.0)
        else:
            L = tl.zeros((BH,), tl.float32)
            e = tl.zeros((BH,), tl.float32)
            r = tl.full((BH,), 1.0, tl.float32)
        wb = W + b * sw_b + h * sw_h
        pb = P + b * sp_b + h * sp_h
        ob = b * T * H + h
        for t in range(0, T):
            w = tl.load(wb + t * sw_t, hm, 0.0).to(tl.float32)
            p = tl.load(pb + t * sp_t, hm, 0.0).to(tl.float32)
            if SAVE:
                tl.store(EPREV + ob + t * H, e, hm)
            pe = _tanh(tl.abs(w - e))
            e = ae * e + (1.0 - ae) * w
            r = ar * r + (1.0 - ar) * (1.0 - pe)
            eff = R + (1.0 - R) * S * r
            L = eff * L + (1.0 - eff) * (p * pe * w)
            if SAVE:
                tl.store(RSEQ + ob + t * H, r, hm)
            tl.store(OUT + ob + t * H, L.to(OUT.dtype.element_ty), hm)
        tl.store(LF + b * H + h, L, hm)
        tl.store(EF + b * H + h, e, hm)
        tl.store(RF + b * H + h, r, hm)

    @triton.jit
    def _long_chunk_kernel(W, P, LR, LE, LD, LS, ES, RS, LSTART, EL, RL, LL, LP, OUT, EPREV,
                           RSEQ, T, H, C, CL, sw_b, sw_t, sw_h, sp_b, sp_t, sp_h,
                           PHASE: tl.constexpr, SAVE: tl.constexpr, BH: tl.constexpr):
        # Time-chunked forward. Given the true starting states of a chunk, each
        # recurrence below the one being resolved is exact; the one being
        # resolved runs from zero (or with a running product) so its chunk-end
        # value is an affine function of its true start:
        #   phase 1: e local             -> e_end = a_e^len e_start + EL
        #   phase 2: r local (true e)    -> r_end = a_r^len r_start + RL
        #   phase 3: L local (true e, r) -> L_end = prod(eff) L_start + LL
        #   phase 4: everything true; writes the outputs (and training tapes).
        pid = tl.program_id(0)
        nh = tl.cdiv(H, BH)
        c = pid % C
        hb = (pid // C) % nh
        b = (pid // (C * nh)).to(tl.int64)
        h = hb * BH + tl.arange(0, BH)
        hm = h < H
        R = tl.sigmoid(tl.load(LR).to(tl.float32))
        ae = tl.sigmoid(tl.load(LE).to(tl.float32))
        ar = tl.sigmoid(tl.load(LD).to(tl.float32))
        S = tl.sigmoid(tl.load(LS).to(tl.float32))
        cb = (b * C + c) * H + h
        if PHASE == 1:
            e = tl.zeros((BH,), tl.float32)
        else:
            e = tl.load(ES + cb, hm, 0.0)
        if PHASE >= 3:
            r = tl.load(RS + cb, hm, 0.0)
        else:
            r = tl.zeros((BH,), tl.float32)
        if PHASE == 4:
            L = tl.load(LSTART + cb, hm, 0.0)
        else:
            L = tl.zeros((BH,), tl.float32)
        prod = tl.full((BH,), 1.0, tl.float32)
        wb = W + b * sw_b + h * sw_h
        pb = P + b * sp_b + h * sp_h
        ob = b * T * H + h
        t0 = c * CL
        t1 = tl.minimum(T, t0 + CL)
        for t in range(t0, t1):
            w = tl.load(wb + t * sw_t, hm, 0.0).to(tl.float32)
            if PHASE == 1:
                e = ae * e + (1.0 - ae) * w
            else:
                if SAVE and PHASE == 4:
                    tl.store(EPREV + ob + t * H, e, hm)
                pe = _tanh(tl.abs(w - e))
                e = ae * e + (1.0 - ae) * w
                r = ar * r + (1.0 - ar) * (1.0 - pe)
                if PHASE >= 3:
                    p = tl.load(pb + t * sp_t, hm, 0.0).to(tl.float32)
                    eff = R + (1.0 - R) * S * r
                    L = eff * L + (1.0 - eff) * (p * pe * w)
                    if PHASE == 3:
                        prod = prod * eff
                    else:
                        tl.store(OUT + ob + t * H, L.to(OUT.dtype.element_ty), hm)
                        if SAVE:
                            tl.store(RSEQ + ob + t * H, r, hm)
        if PHASE == 1:
            tl.store(EL + cb, e, hm)
        if PHASE == 2:
            tl.store(RL + cb, r, hm)
        if PHASE == 3:
            tl.store(LL + cb, L, hm)
            tl.store(LP + cb, prod, hm)

    @triton.jit
    def _long_carry_kernel(X0, LOGIT, LOC, PROD, START, FINAL, T, H, C, CL, init_value,
                           KIND: tl.constexpr, INIT: tl.constexpr, BH: tl.constexpr):
        # Exact chunk-to-chunk carry: x <- decay x + local, decay = a^len (KIND 0)
        # or the chunk's product of retain factors (KIND 1).
        pid = tl.program_id(0)
        nh = tl.cdiv(H, BH)
        b = (pid // nh).to(tl.int64)
        h = (pid % nh) * BH + tl.arange(0, BH)
        hm = h < H
        if INIT:
            x = tl.load(X0 + b * H + h, hm, 0.0)
        else:
            x = tl.full((BH,), init_value, tl.float32)
        log_a = tl.log(tl.sigmoid(tl.load(LOGIT).to(tl.float32)))
        for c in range(0, C):
            cb = (b * C + c) * H + h
            tl.store(START + cb, x, hm)
            n = tl.minimum(CL, T - c * CL).to(tl.float32)
            if KIND == 0:
                decay = tl.exp(n * log_a)
            else:
                decay = tl.load(PROD + cb, hm, 0.0)
            x = decay * x + tl.load(LOC + cb, hm, 0.0)
        tl.store(FINAL + b * H + h, x, hm)

    @triton.jit
    def _long_bwd_kernel(W, P, L0, E0, R0, LR, LE, LD, LS, OUT, EPREV, RSEQ, GOUT, GLF, GEF,
                         GRF, DW, DP, DL0, DE0, DR0, DPAR, T, H, sw_b, sw_t, sw_h, sp_b, sp_t,
                         sp_h, INIT: tl.constexpr, FINAL: tl.constexpr, BH: tl.constexpr):
        pid = tl.program_id(0)
        nh = tl.cdiv(H, BH)
        b = (pid // nh).to(tl.int64)
        h = (pid % nh) * BH + tl.arange(0, BH)
        hm = h < H
        R = tl.sigmoid(tl.load(LR).to(tl.float32))
        ae = tl.sigmoid(tl.load(LE).to(tl.float32))
        ar = tl.sigmoid(tl.load(LD).to(tl.float32))
        S = tl.sigmoid(tl.load(LS).to(tl.float32))
        if FINAL:
            cL = tl.load(GLF + b * H + h, hm, 0.0)
            ce = tl.load(GEF + b * H + h, hm, 0.0)
            cr = tl.load(GRF + b * H + h, hm, 0.0)
        else:
            cL = tl.zeros((BH,), tl.float32)
            ce = tl.zeros((BH,), tl.float32)
            cr = tl.zeros((BH,), tl.float32)
        gR = tl.zeros((BH,), tl.float32)
        gS = tl.zeros((BH,), tl.float32)
        gar = tl.zeros((BH,), tl.float32)
        gae = tl.zeros((BH,), tl.float32)
        wb = W + b * sw_b + h * sw_h
        pb = P + b * sp_b + h * sp_h
        ob = b * T * H + h
        if INIT:
            L0v = tl.load(L0 + b * H + h, hm, 0.0)
            R0v = tl.load(R0 + b * H + h, hm, 0.0)
        else:
            L0v = tl.zeros((BH,), tl.float32)
            R0v = tl.full((BH,), 1.0, tl.float32)
        for k in range(0, T):
            t = T - 1 - k
            w = tl.load(wb + t * sw_t, hm, 0.0).to(tl.float32)
            p = tl.load(pb + t * sp_t, hm, 0.0).to(tl.float32)
            ep = tl.load(EPREV + ob + t * H, hm, 0.0)
            rt = tl.load(RSEQ + ob + t * H, hm, 0.0)
            if t > 0:
                lp = tl.load(OUT + ob + (t - 1) * H, hm, 0.0).to(tl.float32)
                rp = tl.load(RSEQ + ob + (t - 1) * H, hm, 0.0)
            else:
                lp = L0v
                rp = R0v
            d = w - ep
            pe = _tanh(tl.abs(d))
            eff = R + (1.0 - R) * S * rt
            u = p * pe * w
            lam = tl.load(GOUT + ob + t * H, hm, 0.0).to(tl.float32) + cL
            g_eff = lam * (lp - u)
            g_u = lam * (1.0 - eff)
            cL = lam * eff
            gR += g_eff * (1.0 - S * rt)
            gS += g_eff * (1.0 - R) * rt
            lr = cr + g_eff * (1.0 - R) * S
            gar += lr * (rp - (1.0 - pe))
            cr = lr * ar
            g_pe = g_u * p * w - lr * (1.0 - ar)
            tl.store(DP + ob + t * H, g_u * pe * w, hm)
            sgn = tl.where(d > 0, 1.0, tl.where(d < 0, -1.0, 0.0))
            g_d = g_pe * (1.0 - pe * pe) * sgn
            gae += ce * (ep - w)
            tl.store(DW + ob + t * H, g_u * p * pe + ce * (1.0 - ae) + g_d, hm)
            ce = ce * ae - g_d
        if INIT:
            tl.store(DL0 + b * H + h, cL, hm)
            tl.store(DE0 + b * H + h, ce, hm)
            tl.store(DR0 + b * H + h, cr, hm)
        # Parameter gradients with respect to the logits: d sigmoid = s (1 - s).
        base = DPAR + pid * 4
        tl.store(base + 0, tl.sum(tl.where(hm, gR, 0.0), 0) * R * (1.0 - R))
        tl.store(base + 1, tl.sum(tl.where(hm, gae, 0.0), 0) * ae * (1.0 - ae))
        tl.store(base + 2, tl.sum(tl.where(hm, gar, 0.0), 0) * ar * (1.0 - ar))
        tl.store(base + 3, tl.sum(tl.where(hm, gS, 0.0), 0) * S * (1.0 - S))


def _grid(B: int, H: int) -> tuple[int]:
    return (B * triton.cdiv(H, _BH),)


@torch.library.custom_op("dabsn::long_memory_fused", mutates_args=())
def _long_op(w: Tensor, p: Tensor, l0: Optional[Tensor], e0: Optional[Tensor],
             r0: Optional[Tensor],
             lr: Tensor, le: Tensor, ld: Tensor, ls: Tensor, out_dtype: torch.dtype, save: bool
             ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    B, T, H = w.shape
    f = dict(device=w.device, dtype=torch.float32)
    out = torch.empty((B, T, H), device=w.device, dtype=out_dtype)
    eprev = torch.empty((B, T, H) if save else (0,), **f)
    rseq = torch.empty((B, T, H) if save else (0,), **f)
    lf, ef, rf = (torch.empty((B, H), **f) for _ in range(3))
    init = l0 is not None
    if T >= 2 * _CHUNK:
        _chunked_forward(w, p, l0, e0, r0, lr, le, ld, ls, out, eprev, rseq, lf, ef, rf, save)
        return out, lf, ef, rf, eprev, rseq
    _long_fwd_kernel[_grid(B, H)](w, p, l0 if init else w, e0 if init else w, r0 if init else w,
                                  lr, le, ld, ls, out, eprev if save else out,
                                  rseq if save else out, lf, ef, rf, T, H, *w.stride(),
                                  *p.stride(), SAVE=save, INIT=init, BH=_BH, num_warps=1,
                                  num_stages=3)
    return out, lf, ef, rf, eprev, rseq


def _chunked_forward(w, p, l0, e0, r0, lr, le, ld, ls, out, eprev, rseq, lf, ef, rf, save):
    """Long sequences: chunks run in parallel, exact affine carries between them."""
    B, T, H = w.shape
    CL = _CHUNK
    C = triton.cdiv(T, CL)
    f = dict(device=w.device, dtype=torch.float32)
    es, el, rs, rl, lst, ll, lp = (torch.empty((B, C, H), **f) for _ in range(7))
    init = l0 is not None
    grid = (B * triton.cdiv(H, _BH) * C,)
    carry_grid = _grid(B, H)
    strides = (*w.stride(), *p.stride())

    def phase(k):
        _long_chunk_kernel[grid](w, p, lr, le, ld, ls, es, rs, lst, el, rl, ll, lp, out,
                                 eprev if save else out, rseq if save else out, T, H, C, CL,
                                 *strides, PHASE=k, SAVE=save, BH=_BH, num_warps=1,
                                 num_stages=3)

    def carry(x0, logit, loc, prod, start, final, init_value, kind):
        _long_carry_kernel[carry_grid](x0 if init else loc, logit, loc, prod, start, final,
                                       T, H, C, CL, init_value, KIND=kind, INIT=init, BH=_BH,
                                       num_warps=1)

    phase(1)
    carry(e0, le, el, el, es, ef, 0.0, 0)
    phase(2)
    carry(r0, ld, rl, rl, rs, rf, 1.0, 0)
    phase(3)
    carry(l0, lr, ll, lp, lst, lf, 0.0, 1)
    phase(4)


@_long_op.register_fake
def _long_fake(w, p, l0, e0, r0, lr, le, ld, ls, out_dtype, save):
    B, T, H = w.shape
    f = torch.float32
    tape = (B, T, H) if save else (0,)
    return (w.new_empty((B, T, H), dtype=out_dtype), w.new_empty((B, H), dtype=f),
            w.new_empty((B, H), dtype=f), w.new_empty((B, H), dtype=f),
            w.new_empty(tape, dtype=f), w.new_empty(tape, dtype=f))


@torch.library.custom_op("dabsn::long_memory_fused_backward", mutates_args=())
def _long_backward_op(w: Tensor, p: Tensor, l0: Optional[Tensor], e0: Optional[Tensor],
                      r0: Optional[Tensor], lr: Tensor, le: Tensor, ld: Tensor, ls: Tensor,
                      out: Tensor, eprev: Tensor, rseq: Tensor, gout: Tensor,
                      glf: Optional[Tensor], gef: Optional[Tensor], grf: Optional[Tensor]
                      ) -> List[Tensor]:
    B, T, H = w.shape
    f = dict(device=w.device, dtype=torch.float32)
    init = l0 is not None
    final = glf is not None
    dw = torch.empty((B, T, H), **f)
    dp = torch.empty((B, T, H), **f)
    dl0, de0, dr0 = (torch.empty((B, H) if init else (0,), **f) for _ in range(3))
    grid = _grid(B, H)
    dpar = torch.empty((grid[0], 4), **f)
    _long_bwd_kernel[grid](w, p, l0 if init else w, e0 if init else w, r0 if init else w,
                           lr, le, ld, ls, out, eprev, rseq, gout, glf if final else w,
                           gef if final else w, grf if final else w, dw, dp,
                           dl0 if init else dw, de0 if init else dw, dr0 if init else dw, dpar,
                           T, H, *w.stride(), *p.stride(), INIT=init, FINAL=final, BH=_BH,
                           num_warps=1, num_stages=3)
    return [dw, dp, dl0, de0, dr0, dpar.sum(dim=0)]


@_long_backward_op.register_fake
def _long_backward_fake(w, p, l0, e0, r0, lr, le, ld, ls, out, eprev, rseq, gout, glf, gef, grf):
    f = torch.float32
    B, T, H = w.shape
    st = (B, H) if l0 is not None else (0,)
    return [w.new_empty((B, T, H), dtype=f), w.new_empty((B, T, H), dtype=f),
            w.new_empty(st, dtype=f), w.new_empty(st, dtype=f), w.new_empty(st, dtype=f),
            w.new_empty((4,), dtype=f)]


def _long_setup(ctx, inputs, output) -> None:
    w, p, l0, e0, r0, lr, le, ld, ls = inputs[:9]
    ctx.init = l0 is not None
    states = (l0, e0, r0) if ctx.init else ()
    ctx.save_for_backward(w, p, lr, le, ld, ls, output[0], output[4], output[5], *states)


def _long_backward(ctx, gout, glf, gef, grf, _ge, _gr):
    w, p, lr, le, ld, ls, out, eprev, rseq, *states = ctx.saved_tensors
    l0, e0, r0 = states if ctx.init else (None, None, None)
    B, T, H = w.shape
    if gout is None:
        gout = torch.zeros((B, T, H), device=w.device, dtype=torch.float32)
    final = any(g is not None for g in (glf, gef, grf))

    def state_grad(g):
        return torch.zeros((B, H), device=w.device, dtype=torch.float32) if g is None else (
            g.to(torch.float32).contiguous())

    dw, dp, dl0, de0, dr0, dpar = _long_backward_op(
        w, p, l0, e0, r0, lr, le, ld, ls, out, eprev, rseq,
        gout.to(torch.float32).contiguous(),
        state_grad(glf) if final else None, state_grad(gef) if final else None,
        state_grad(grf) if final else None)
    dstates = (dl0.to(l0.dtype), de0.to(e0.dtype), dr0.to(r0.dtype)) if ctx.init else (
        None, None, None)
    return (dw.to(w.dtype), dp.to(p.dtype), *dstates, dpar[0:1].to(lr.dtype).reshape(lr.shape),
            dpar[1:2].to(le.dtype).reshape(le.shape), dpar[2:3].to(ld.dtype).reshape(ld.shape),
            dpar[3:4].to(ls.dtype).reshape(ls.shape), None, None)


torch.library.register_autograd(_long_op, _long_backward, setup_context=_long_setup)


def fused_long_memory_supported(write: Tensor) -> bool:
    return (
        _HAS_TRITON
        and write.is_cuda
        and write.dim() == 3
        and write.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def fused_long_memory(read, write: Tensor, plasticity: Tensor, initial_state, return_final_state):
    """Drop-in for the CUDA branch of ``fused_long_scan_from_state``."""
    B, _T, H = write.shape
    out_dtype = torch.result_type(write, read.logit_retain)
    result_dtype = out_dtype
    logits = (read.logit_retain, read.logit_expect_retain, read.logit_retention_decay,
              read.logit_retention_strength)
    if torch.bfloat16 in (write.dtype, plasticity.dtype, out_dtype, *(t.dtype for t in logits)):
        from ._arch import bf16_kernels_supported

        if not bf16_kernels_supported(write.device):
            # Pre-Ampere (T4): Triton cannot lower bf16 there. The recurrence
            # is FP32 inside the kernel anyway, so upcast at the boundary.
            write, plasticity, out_dtype = write.float(), plasticity.float(), torch.float32
            logits = tuple(t.float() for t in logits)
    if initial_state is None:
        l0 = e0 = r0 = None
    else:
        f = dict(device=write.device, dtype=torch.float32)
        l0, e0, r0 = (s.to(**f).contiguous() for s in initial_state)
        if any(s.shape != (B, H) for s in (l0, e0, r0)):
            raise ValueError(f"initial long-scan state must have shape {(B, H)}")
    plasticity = plasticity.expand_as(write)
    save = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (write, plasticity, l0, e0, r0, *logits))
    out, lf, ef, rf, _, _ = _long_op(write, plasticity, l0, e0, r0, *logits, out_dtype, save)
    out = out.to(result_dtype)
    read._last_long_backend = "cuda_triton_fused"
    if return_final_state:
        return out, (lf, ef, rf)
    return out


__all__ = ["fused_long_memory", "fused_long_memory_supported"]
