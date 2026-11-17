"""Fused CUDA local VJP for Core online-credit replay.

``core_online_replay_contracted`` evaluates the closed-form current-step VJP
(``_core_step_local_vjp_analytic``) over the whole flattened ``B*T`` tape with
roughly a hundred full-tape PyTorch ops. This module computes the same values in
two elementwise Triton passes around the one recurrent-adjoint GEMM:

* pass A: everything up to the recurrent product, plus the matrix-trace
  coefficients and the direct per-channel/scalar parameter partial sums;
* GEMM:  ``g_expression @ A + g_gate_pre @ Ug`` (cuBLAS, FP32 matmul policy);
* pass B: the ``y`` pre-activation adjoint and the three-state incoming adjoint.

The small-state trace scan runs in analytic per-state form
(``_trace_scan_lanes``, about 10x faster than the dual-number scan it
replaces). Budget-trace weights and the chunk-level matrix contractions are
unchanged, so the eligibility rule and its reassociation are exactly those of
``core_online_replay_contracted``. Partial sums are reduced in a fixed order:
deterministic, no atomics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

import triton
import triton.language as tl

from ..temporal_credit import (
    CORE_GLOBAL_PARAMETER_NAMES,
    CORE_LOCAL_PARAMETER_NAMES,
    CoreEligibilityState,
    CoreSignalGrads,
    _budget_trace_weights,
    _contract_budget_trace,
)
from ._arch import on_tensor_device

_ROWS = 32
_COLS = 64


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _ld(Ptr, rows, cols, rs, mask, PRESENT: tl.constexpr):
    if PRESENT:
        return tl.load(Ptr + rows[:, None] * rs + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    else:
        return tl.zeros(mask.shape, dtype=tl.float32)


@triton.jit
def _vjp_a(
    Budget, Energy, Sat, Y, Expr, GatePre, Gate, Nov, Plas,
    s_b, s_e, s_s, s_y, s_x, s_gp, s_g, s_n, s_p,
    SNb, SE, SX, SNov, SP, SNs, SW,
    r_nb, r_e, r_x, r_nov, r_p, r_ns, r_w,
    UgDiag, ADiag, Scalars,
    GExpr, GGate, GBudget, Incoming, CoefW, CoefA, CoefG,
    ChanPart, ScalPart,
    R, H,
    HAS_NB: tl.constexpr, HAS_E: tl.constexpr, HAS_X: tl.constexpr, HAS_NOV: tl.constexpr,
    HAS_P: tl.constexpr, HAS_NS: tl.constexpr, HAS_W: tl.constexpr,
    BR: tl.constexpr, BC: tl.constexpr,
):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    rows = pr * BR + tl.arange(0, BR)
    cols = pc * BC + tl.arange(0, BC)
    mask = (rows[:, None] < R) & (cols[None, :] < H)
    fm = mask.to(tl.float32)
    budget = _ld(Budget, rows, cols, s_b, mask, True)
    energy = _ld(Energy, rows, cols, s_e, mask, True)
    saturation = _ld(Sat, rows, cols, s_s, mask, True)
    y = _ld(Y, rows, cols, s_y, mask, True)
    expression = _ld(Expr, rows, cols, s_x, mask, True)
    gate_pre = _ld(GatePre, rows, cols, s_gp, mask, True)
    gate = _ld(Gate, rows, cols, s_g, mask, True)
    novelty = _ld(Nov, rows, cols, s_n, mask, True)
    plasticity = _ld(Plas, rows, cols, s_p, mask, True)
    gnb = _ld(SNb, rows, cols, r_nb, mask, HAS_NB)
    g_energy = _ld(SE, rows, cols, r_e, mask, HAS_E)
    g_expr = _ld(SX, rows, cols, r_x, mask, HAS_X)
    g_nov = _ld(SNov, rows, cols, r_nov, mask, HAS_NOV)
    g_p = _ld(SP, rows, cols, r_p, mask, HAS_P)
    g_ns = _ld(SNs, rows, cols, r_ns, mask, HAS_NS)
    g_w = _ld(SW, rows, cols, r_w, mask, HAS_W)
    alpha = tl.load(Scalars + 0)
    lam = tl.load(Scalars + 1)
    decay = tl.load(Scalars + 2)
    sig_lam = tl.load(Scalars + 3)
    hard_slope = tl.load(Scalars + 4)

    g_budget = gnb * (1.0 - alpha)
    g_write = g_w + gnb * lam
    g_p = g_p + g_write * expression
    g_expr = g_expr + g_write * plasticity
    g_gate = g_p * energy
    g_energy = g_energy + g_p * gate
    stress = novelty * (1.0 - energy)
    g_sat = g_ns * decay
    g_stress = g_ns * (1.0 - decay)
    g_nov = g_nov + g_stress * (1.0 - energy)
    g_energy = g_energy - g_stress * novelty
    diff = expression - budget
    sgn = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    g_diff = g_nov * (1.0 - novelty * novelty) * sgn
    g_expr = g_expr + g_diff
    g_budget = g_budget - g_diff
    open_ = ((gate_pre > -3.0) & (gate_pre < 3.0)).to(tl.float32)
    g_gate_pre = g_gate * (open_ * hard_slope)

    idx = rows[:, None] * H + cols[None, :]
    tl.store(GExpr + idx, g_expr, mask=mask)
    tl.store(GGate + idx, g_gate_pre, mask=mask)
    tl.store(GBudget + idx, g_budget, mask=mask)
    tl.store(Incoming + idx * 3 + 1, g_energy, mask=mask)
    tl.store(Incoming + idx * 3 + 2, g_sat, mask=mask)

    slope = open_ / 6.0
    common = lam * energy
    ug_diag = tl.load(UgDiag + cols, mask=cols < H, other=0.0)[None, :]
    a_diag = tl.load(ADiag + cols, mask=cols < H, other=0.0)[None, :]
    tl.store(CoefG + idx, common * expression * slope, mask=mask)
    tl.store(CoefA + idx, common * gate, mask=mask)
    tl.store(CoefW + idx, common * (1.0 - y * y) * (expression * slope * ug_diag + gate * a_diag),
             mask=mask)

    cmask = cols < H
    base = ChanPart + pr * 2 * H + cols
    tl.store(base, tl.sum(gnb * fm, 0), mask=cmask)
    tl.store(base + H, tl.sum(g_gate_pre * fm, 0), mask=cmask)
    sp = ScalPart + (pr * tl.num_programs(1) + pc) * 3
    tl.store(sp + 0, tl.sum(tl.sum(gnb * (-budget) * alpha * (1.0 - alpha) * fm, 1), 0))
    tl.store(sp + 1, tl.sum(tl.sum(gnb * (plasticity * expression) * sig_lam * fm, 1), 0))
    tl.store(sp + 2, tl.sum(tl.sum(g_ns * (saturation - stress) * decay * (1.0 - decay) * fm, 1), 0))


@triton.jit
def _vjp_b(
    Y, s_y, SY, r_y, Recurrent, GBudget, GYPre, Incoming,
    R, H,
    HAS_Y: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    rows = pr * BR + tl.arange(0, BR)
    cols = pc * BC + tl.arange(0, BC)
    mask = (rows[:, None] < R) & (cols[None, :] < H)
    idx = rows[:, None] * H + cols[None, :]
    y = _ld(Y, rows, cols, s_y, mask, True)
    g_y = _ld(SY, rows, cols, r_y, mask, HAS_Y) + tl.load(Recurrent + idx, mask=mask, other=0.0)
    g_y_pre = g_y * (1.0 - y * y)
    tl.store(GYPre + idx, g_y_pre, mask=mask)
    tl.store(Incoming + idx * 3, tl.load(GBudget + idx, mask=mask, other=0.0) + g_y_pre, mask=mask)


@triton.jit
def _lane(eb, ee, es, g, ib, ie, iz, jb0, jb1, je0, je1, je2, js0, js1, js2, db, de, ds):
    g = g + ib * eb + ie * ee + iz * es
    return (jb0 * eb + jb1 * ee + db,
            je0 * eb + je1 * ee + je2 * es + de,
            js0 * eb + js1 * ee + js2 * es + ds,
            g)


@triton.jit
def _ldl(Ptr, base, o: tl.constexpr, j: tl.constexpr, W: tl.constexpr, mask):
    return tl.load(Ptr + base * (3 * W) + o * W + j, mask, other=0.0)


@triton.jit
def _stl(Ptr, base, o: tl.constexpr, j: tl.constexpr, W: tl.constexpr, value, mask):
    tl.store(Ptr + base * (3 * W) + o * W + j, value, mask)


@triton.jit
def _trace_scan_lanes(
    Budget, Energy, Saturation, Y, Expression, GatePre, Incoming,
    Params, Globals, ADiag, UgDiag, Local, Global, NextLocal, NextGlobal,
    LocalGrad, GlobalGrad, Jbb,
    S: tl.constexpr, T, BLOCK: tl.constexpr,
):
    """Analytic form of ``online_trace._scan``: same lanes, same eligibility rule.

    The dual-number scan carries 23 derivative lanes through every operation and
    extracts the 3x3 state Jacobian with cross-lane reductions. Here each thread
    owns one state: its step values and Jacobian are computed once as scalars,
    and every parameter lane's direct derivative is a scalar sensitivity times
    that parameter's local input (each parameter enters the step at one point).
    Lane p's parameter is named beside its update; lanes 3-18 are the local
    parameters (``CORE_LOCAL_PARAMETER_NAMES``), 19-22 the global ones.
    """
    b = tl.program_id(0)
    s = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    sv = s < S
    base = b * S + s
    a_diag = tl.load(ADiag + s, sv, other=0)
    ug_diag = tl.load(UgDiag + s, sv, other=0)
    kappa_raw = tl.load(Params + 2 * S + s, sv, other=0)
    kappa = tl.maximum(kappa_raw, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(kappa_raw)))
    sig_kappa = tl.sigmoid(kappa_raw)
    recover = tl.sigmoid(tl.load(Params + 3 * S + s, sv, other=0))
    rec_prior = recover * (1.0 - recover)
    ks = tl.load(Params + 4 * S + s, sv, other=0)
    ky = tl.load(Params + 5 * S + s, sv, other=0)
    kb = tl.load(Params + 6 * S + s, sv, other=0)
    kn = tl.load(Params + 7 * S + s, sv, other=0)
    kbias = tl.load(Params + 8 * S + s, sv, other=0)
    rs = tl.load(Params + 9 * S + s, sv, other=0)
    ry = tl.load(Params + 10 * S + s, sv, other=0)
    rb = tl.load(Params + 11 * S + s, sv, other=0)
    rn = tl.load(Params + 12 * S + s, sv, other=0)
    rbias = tl.load(Params + 13 * S + s, sv, other=0)
    ksat = tl.load(Params + 14 * S + s, sv, other=0)
    rsat = tl.load(Params + 15 * S + s, sv, other=0)
    alpha = tl.sigmoid(tl.load(Globals))
    loglam = tl.load(Globals + 1)
    lam = tl.maximum(loglam, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(loglam)))
    sig_lam = tl.sigmoid(loglam)
    decay = tl.sigmoid(tl.load(Globals + 2))
    suppress = tl.sigmoid(tl.load(Globals + 3))
    d_alpha = alpha * (1.0 - alpha)
    d_decay = decay * (1.0 - decay)
    d_suppress = suppress * (1.0 - suppress)

    # Incoming traces: three outputs (budget, energy, saturation) x 20 lanes.
    b3 = _ldl(Local, base, 0, 0, 16, sv); e3 = _ldl(Local, base, 1, 0, 16, sv); s3 = _ldl(Local, base, 2, 0, 16, sv)
    b4 = _ldl(Local, base, 0, 1, 16, sv); e4 = _ldl(Local, base, 1, 1, 16, sv); s4 = _ldl(Local, base, 2, 1, 16, sv)
    b5 = _ldl(Local, base, 0, 2, 16, sv); e5 = _ldl(Local, base, 1, 2, 16, sv); s5 = _ldl(Local, base, 2, 2, 16, sv)
    b6 = _ldl(Local, base, 0, 3, 16, sv); e6 = _ldl(Local, base, 1, 3, 16, sv); s6 = _ldl(Local, base, 2, 3, 16, sv)
    b7 = _ldl(Local, base, 0, 4, 16, sv); e7 = _ldl(Local, base, 1, 4, 16, sv); s7 = _ldl(Local, base, 2, 4, 16, sv)
    b8 = _ldl(Local, base, 0, 5, 16, sv); e8 = _ldl(Local, base, 1, 5, 16, sv); s8 = _ldl(Local, base, 2, 5, 16, sv)
    b9 = _ldl(Local, base, 0, 6, 16, sv); e9 = _ldl(Local, base, 1, 6, 16, sv); s9 = _ldl(Local, base, 2, 6, 16, sv)
    b10 = _ldl(Local, base, 0, 7, 16, sv); e10 = _ldl(Local, base, 1, 7, 16, sv); s10 = _ldl(Local, base, 2, 7, 16, sv)
    b11 = _ldl(Local, base, 0, 8, 16, sv); e11 = _ldl(Local, base, 1, 8, 16, sv); s11 = _ldl(Local, base, 2, 8, 16, sv)
    b12 = _ldl(Local, base, 0, 9, 16, sv); e12 = _ldl(Local, base, 1, 9, 16, sv); s12 = _ldl(Local, base, 2, 9, 16, sv)
    b13 = _ldl(Local, base, 0, 10, 16, sv); e13 = _ldl(Local, base, 1, 10, 16, sv); s13 = _ldl(Local, base, 2, 10, 16, sv)
    b14 = _ldl(Local, base, 0, 11, 16, sv); e14 = _ldl(Local, base, 1, 11, 16, sv); s14 = _ldl(Local, base, 2, 11, 16, sv)
    b15 = _ldl(Local, base, 0, 12, 16, sv); e15 = _ldl(Local, base, 1, 12, 16, sv); s15 = _ldl(Local, base, 2, 12, 16, sv)
    b16 = _ldl(Local, base, 0, 13, 16, sv); e16 = _ldl(Local, base, 1, 13, 16, sv); s16 = _ldl(Local, base, 2, 13, 16, sv)
    b17 = _ldl(Local, base, 0, 14, 16, sv); e17 = _ldl(Local, base, 1, 14, 16, sv); s17 = _ldl(Local, base, 2, 14, 16, sv)
    b18 = _ldl(Local, base, 0, 15, 16, sv); e18 = _ldl(Local, base, 1, 15, 16, sv); s18 = _ldl(Local, base, 2, 15, 16, sv)
    b19 = _ldl(Global, base, 0, 0, 4, sv); e19 = _ldl(Global, base, 1, 0, 4, sv); s19 = _ldl(Global, base, 2, 0, 4, sv)
    b20 = _ldl(Global, base, 0, 1, 4, sv); e20 = _ldl(Global, base, 1, 1, 4, sv); s20 = _ldl(Global, base, 2, 1, 4, sv)
    b21 = _ldl(Global, base, 0, 2, 4, sv); e21 = _ldl(Global, base, 1, 2, 4, sv); s21 = _ldl(Global, base, 2, 2, 4, sv)
    b22 = _ldl(Global, base, 0, 3, 4, sv); e22 = _ldl(Global, base, 1, 3, 4, sv); s22 = _ldl(Global, base, 2, 3, 4, sv)
    zero = tl.zeros((BLOCK,), dtype=tl.float32)
    g3 = zero; g4 = zero; g5 = zero; g6 = zero; g7 = zero; g8 = zero; g9 = zero; g10 = zero
    g11 = zero; g12 = zero; g13 = zero; g14 = zero; g15 = zero; g16 = zero; g17 = zero; g18 = zero
    g19 = zero; g20 = zero; g21 = zero; g22 = zero

    offset = (b * T) * S + s
    budget = tl.load(Budget + offset, sv, other=0)
    energy = tl.load(Energy + offset, sv, other=0)
    saturation = tl.load(Saturation + offset, sv, other=0)
    y = tl.load(Y + offset, sv, other=0)
    expression = tl.load(Expression + offset, sv, other=0)
    pre = tl.load(GatePre + offset, sv, other=0)
    ib = tl.load(Incoming + offset * 3, sv, other=0)
    ie = tl.load(Incoming + offset * 3 + 1, sv, other=0)
    iz = tl.load(Incoming + offset * 3 + 2, sv, other=0)
    for t in range(T):
        nxt = (b * T + t + 1) * S + s
        ok = sv & (t + 1 < T)
        n_budget = tl.load(Budget + nxt, ok, other=0)
        n_energy = tl.load(Energy + nxt, ok, other=0)
        n_saturation = tl.load(Saturation + nxt, ok, other=0)
        n_y = tl.load(Y + nxt, ok, other=0)
        n_expression = tl.load(Expression + nxt, ok, other=0)
        n_pre = tl.load(GatePre + nxt, ok, other=0)
        n_ib = tl.load(Incoming + nxt * 3, ok, other=0)
        n_ie = tl.load(Incoming + nxt * 3 + 1, ok, other=0)
        n_iz = tl.load(Incoming + nxt * 3 + 2, ok, other=0)

        gate = tl.minimum(tl.maximum((pre + 3.0) / 6.0, 0.0), 1.0)
        slope = ((pre > -3.0) & (pre < 3.0)).to(tl.float32) / 6.0
        delta = expression - budget
        novelty = _tanh(tl.abs(delta))
        sign = tl.where(delta > 0, 1.0, tl.where(delta < 0, -1.0, 0.0))
        one_minus_e = 1.0 - energy
        next_sat = decay * saturation + (1.0 - decay) * novelty * one_minus_e
        neff = novelty * (1.0 - suppress * next_sat)
        tb = _tanh(budget)
        k = ks * gate + ky * y + kb * tb + kn * neff + kbias + ksat * next_sat
        r = rs * gate + ry * y + rb * tb + rn * neff + rbias + rsat * next_sat
        tk = _tanh(k)
        tr = _tanh(r)
        ek = tl.exp(0.5 * tk)
        er = tl.exp(0.5 * tr)
        cost = kappa * ek
        raw_recovery = recover * er
        recovery = tl.minimum(tl.maximum(raw_recovery, 0.0), 1.0)
        rec_open = ((raw_recovery > 0) & (raw_recovery < 1)).to(tl.float32)
        plasticity = gate * energy
        raw_energy = energy + recovery * one_minus_e - cost * plasticity
        e_open = ((raw_energy > 0) & (raw_energy < 1)).to(tl.float32)
        ck = cost * 0.5 * (1.0 - tk * tk)
        cr = raw_recovery * 0.5 * (1.0 - tr * tr) * rec_open
        kfac = -ck * plasticity * e_open  # d next_e / d k
        rfac = cr * one_minus_e * e_open  # d next_e / d r

        dy0 = 1.0 - y * y
        dx0 = dy0 * a_diag
        dg0 = slope * dy0 * ug_diag
        dn0 = (1.0 - novelty * novelty) * sign * (dx0 - 1.0)
        dns0 = (1.0 - decay) * one_minus_e * dn0
        dneff0 = dn0 * (1.0 - suppress * next_sat) - novelty * suppress * dns0
        dtb0 = 1.0 - tb * tb
        dk0 = ks * dg0 + ky * dy0 + kb * dtb0 + kn * dneff0 + ksat * dns0
        dr0 = rs * dg0 + ry * dy0 + rb * dtb0 + rn * dneff0 + rsat * dns0
        dp0 = energy * dg0
        jb0 = (1.0 - alpha) + lam * (dp0 * expression + plasticity * dx0)
        je0 = rfac * dr0 + kfac * dk0 - cost * dp0 * e_open
        dns1 = -(1.0 - decay) * novelty
        dneff1 = -novelty * suppress * dns1
        jb1 = lam * gate * expression
        je1 = (1.0 - recovery - cost * gate) * e_open + rfac * (rn * dneff1 + rsat * dns1) + kfac * (kn * dneff1 + ksat * dns1)
        dneff2 = -novelty * suppress * decay
        je2 = rfac * (rn * dneff2 + rsat * decay) + kfac * (kn * dneff2 + ksat * decay)
        dns21 = (saturation - novelty * one_minus_e) * d_decay
        dneff21 = -novelty * suppress * dns21
        dneff22 = -novelty * next_sat * d_suppress

        b3, e3, s3, g3 = _lane(b3, e3, s3, g3, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                               lam * energy * slope * expression,
                               slope * (rs * rfac + ks * kfac - cost * energy * e_open), 0.0)  # Wg.bias
        b4, e4, s4, g4 = _lane(b4, e4, s4, g4, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                               1.0, 0.0, 0.0)  # beta
        b5, e5, s5, g5 = _lane(b5, e5, s5, g5, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                               0.0, -sig_kappa * ek * plasticity * e_open, 0.0)  # log_kappa
        b6, e6, s6, g6 = _lane(b6, e6, s6, g6, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                               0.0, rec_prior * er * rec_open * one_minus_e * e_open, 0.0)  # logit_recover
        b7, e7, s7, g7 = _lane(b7, e7, s7, g7, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                               0.0, kfac * gate, 0.0)  # k_s
        b8, e8, s8, g8 = _lane(b8, e8, s8, g8, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                               0.0, kfac * y, 0.0)  # k_y
        b9, e9, s9, g9 = _lane(b9, e9, s9, g9, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                               0.0, kfac * tb, 0.0)  # k_b
        b10, e10, s10, g10 = _lane(b10, e10, s10, g10, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, kfac * neff, 0.0)  # k_n
        b11, e11, s11, g11 = _lane(b11, e11, s11, g11, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, kfac, 0.0)  # k_bias
        b12, e12, s12, g12 = _lane(b12, e12, s12, g12, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, rfac * gate, 0.0)  # r_s
        b13, e13, s13, g13 = _lane(b13, e13, s13, g13, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, rfac * y, 0.0)  # r_y
        b14, e14, s14, g14 = _lane(b14, e14, s14, g14, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, rfac * tb, 0.0)  # r_b
        b15, e15, s15, g15 = _lane(b15, e15, s15, g15, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, rfac * neff, 0.0)  # r_n
        b16, e16, s16, g16 = _lane(b16, e16, s16, g16, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, rfac, 0.0)  # r_bias
        b17, e17, s17, g17 = _lane(b17, e17, s17, g17, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, kfac * next_sat, 0.0)  # k_saturation
        b18, e18, s18, g18 = _lane(b18, e18, s18, g18, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, rfac * next_sat, 0.0)  # r_saturation
        b19, e19, s19, g19 = _lane(b19, e19, s19, g19, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   -budget * d_alpha, 0.0, 0.0)  # logit_alpha
        b20, e20, s20, g20 = _lane(b20, e20, s20, g20, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   plasticity * expression * sig_lam, 0.0, 0.0)  # log_lambda
        b21, e21, s21, g21 = _lane(b21, e21, s21, g21, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, rfac * (rn * dneff21 + rsat * dns21) + kfac * (kn * dneff21 + ksat * dns21),
                                   dns21)  # logit_saturation_decay
        b22, e22, s22, g22 = _lane(b22, e22, s22, g22, ib, ie, iz, jb0, jb1, je0, je1, je2, dns0, dns1, decay,
                                   0.0, rfac * rn * dneff22 + kfac * kn * dneff22, 0.0)  # logit_saturation_suppress
        tl.store(Jbb + (b * T + t) * S + s, jb0, sv)

        budget = n_budget
        energy = n_energy
        saturation = n_saturation
        y = n_y
        expression = n_expression
        pre = n_pre
        ib = n_ib
        ie = n_ie
        iz = n_iz

    _stl(NextLocal, base, 0, 0, 16, b3, sv); _stl(NextLocal, base, 1, 0, 16, e3, sv); _stl(NextLocal, base, 2, 0, 16, s3, sv)
    _stl(NextLocal, base, 0, 1, 16, b4, sv); _stl(NextLocal, base, 1, 1, 16, e4, sv); _stl(NextLocal, base, 2, 1, 16, s4, sv)
    _stl(NextLocal, base, 0, 2, 16, b5, sv); _stl(NextLocal, base, 1, 2, 16, e5, sv); _stl(NextLocal, base, 2, 2, 16, s5, sv)
    _stl(NextLocal, base, 0, 3, 16, b6, sv); _stl(NextLocal, base, 1, 3, 16, e6, sv); _stl(NextLocal, base, 2, 3, 16, s6, sv)
    _stl(NextLocal, base, 0, 4, 16, b7, sv); _stl(NextLocal, base, 1, 4, 16, e7, sv); _stl(NextLocal, base, 2, 4, 16, s7, sv)
    _stl(NextLocal, base, 0, 5, 16, b8, sv); _stl(NextLocal, base, 1, 5, 16, e8, sv); _stl(NextLocal, base, 2, 5, 16, s8, sv)
    _stl(NextLocal, base, 0, 6, 16, b9, sv); _stl(NextLocal, base, 1, 6, 16, e9, sv); _stl(NextLocal, base, 2, 6, 16, s9, sv)
    _stl(NextLocal, base, 0, 7, 16, b10, sv); _stl(NextLocal, base, 1, 7, 16, e10, sv); _stl(NextLocal, base, 2, 7, 16, s10, sv)
    _stl(NextLocal, base, 0, 8, 16, b11, sv); _stl(NextLocal, base, 1, 8, 16, e11, sv); _stl(NextLocal, base, 2, 8, 16, s11, sv)
    _stl(NextLocal, base, 0, 9, 16, b12, sv); _stl(NextLocal, base, 1, 9, 16, e12, sv); _stl(NextLocal, base, 2, 9, 16, s12, sv)
    _stl(NextLocal, base, 0, 10, 16, b13, sv); _stl(NextLocal, base, 1, 10, 16, e13, sv); _stl(NextLocal, base, 2, 10, 16, s13, sv)
    _stl(NextLocal, base, 0, 11, 16, b14, sv); _stl(NextLocal, base, 1, 11, 16, e14, sv); _stl(NextLocal, base, 2, 11, 16, s14, sv)
    _stl(NextLocal, base, 0, 12, 16, b15, sv); _stl(NextLocal, base, 1, 12, 16, e15, sv); _stl(NextLocal, base, 2, 12, 16, s15, sv)
    _stl(NextLocal, base, 0, 13, 16, b16, sv); _stl(NextLocal, base, 1, 13, 16, e16, sv); _stl(NextLocal, base, 2, 13, 16, s16, sv)
    _stl(NextLocal, base, 0, 14, 16, b17, sv); _stl(NextLocal, base, 1, 14, 16, e17, sv); _stl(NextLocal, base, 2, 14, 16, s17, sv)
    _stl(NextLocal, base, 0, 15, 16, b18, sv); _stl(NextLocal, base, 1, 15, 16, e18, sv); _stl(NextLocal, base, 2, 15, 16, s18, sv)
    _stl(NextGlobal, base, 0, 0, 4, b19, sv); _stl(NextGlobal, base, 1, 0, 4, e19, sv); _stl(NextGlobal, base, 2, 0, 4, s19, sv)
    _stl(NextGlobal, base, 0, 1, 4, b20, sv); _stl(NextGlobal, base, 1, 1, 4, e20, sv); _stl(NextGlobal, base, 2, 1, 4, s20, sv)
    _stl(NextGlobal, base, 0, 2, 4, b21, sv); _stl(NextGlobal, base, 1, 2, 4, e21, sv); _stl(NextGlobal, base, 2, 2, 4, s21, sv)
    _stl(NextGlobal, base, 0, 3, 4, b22, sv); _stl(NextGlobal, base, 1, 3, 4, e22, sv); _stl(NextGlobal, base, 2, 3, 4, s22, sv)
    lg = LocalGrad + base * 16
    tl.store(lg + 0, g3, sv); tl.store(lg + 1, g4, sv); tl.store(lg + 2, g5, sv); tl.store(lg + 3, g6, sv)
    tl.store(lg + 4, g7, sv); tl.store(lg + 5, g8, sv); tl.store(lg + 6, g9, sv); tl.store(lg + 7, g10, sv)
    tl.store(lg + 8, g11, sv); tl.store(lg + 9, g12, sv); tl.store(lg + 10, g13, sv); tl.store(lg + 11, g14, sv)
    tl.store(lg + 12, g15, sv); tl.store(lg + 13, g16, sv); tl.store(lg + 14, g17, sv); tl.store(lg + 15, g18, sv)
    gg = GlobalGrad + base * 4
    tl.store(gg + 0, g19, sv); tl.store(gg + 1, g20, sv); tl.store(gg + 2, g21, sv); tl.store(gg + 3, g22, sv)


@on_tensor_device
def core_trace_scan_analytic(core, tape, incoming, eligibility):
    """Drop-in for ``online_trace.core_trace_scan`` (same outputs, same order)."""
    batch, steps, state, _ = incoming.shape
    names = ('beta', 'log_kappa', 'logit_recover', 'k_s', 'k_y', 'k_b',
             'k_n', 'k_bias', 'r_s', 'r_y', 'r_b', 'r_n', 'r_bias',
             'k_saturation', 'r_saturation')
    params = torch.stack([core.Wg.bias.detach().float().reshape(-1)] +
                         [getattr(core, name).detach().float().reshape(-1) for name in names])
    globals_ = torch.stack([getattr(core, name).detach().float().reshape(()) for name in
                           ('logit_alpha', 'log_lambda', 'logit_saturation_decay',
                            'logit_saturation_suppress')])
    next_local = torch.empty_like(eligibility.local)
    next_global = torch.empty_like(eligibility.global_)
    local_grad = torch.empty((batch, state, 16), device=incoming.device, dtype=torch.float32)
    global_grad = torch.empty((batch, state, 4), device=incoming.device, dtype=torch.float32)
    jbb = torch.empty((batch, steps, state), device=incoming.device, dtype=torch.float32)
    tapes = [getattr(tape, name).float().contiguous() for name in
             ('budget', 'energy', 'saturation', 'y', 'expression', 'gate_preactivation')]
    # One state per thread, 32 states per one-warp program (measured best on A40).
    block = 32
    _trace_scan_lanes[(batch, triton.cdiv(state, block))](
        *tapes, incoming.float().contiguous(), params, globals_,
        core.A.weight.detach().diagonal().float().contiguous(),
        core.Ug.weight.detach().diagonal().float().contiguous(),
        eligibility.local.contiguous(), eligibility.global_.contiguous(),
        next_local, next_global, local_grad, global_grad, jbb,
        state, steps, block, num_warps=1,
    )
    return next_local, next_global, local_grad.sum(0), global_grad.sum((0, 1)), jbb


def _rows(value: Tensor, R: int, H: int):
    """A [B,T,H] tensor as rows of H with a uniform row stride, without copying when possible."""
    if value.stride(-1) == 1 and value.stride(0) == value.shape[1] * value.stride(1):
        return value, value.stride(1)
    value = value.contiguous()
    return value, H


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=False)
@on_tensor_device
def core_online_replay_fused(core, tape, signal_grads: CoreSignalGrads, eligibility_state: CoreEligibilityState):
    """CUDA replacement for ``core_online_replay_contracted`` (same contract, same values)."""
    batch, steps, state = tape.batch_size, tape.steps, tape.state_dim
    if steps < 1:
        raise ValueError("online replay requires at least one time step")
    if (eligibility_state.batch_size, eligibility_state.input_dim, eligibility_state.state_dim) != (
        batch, tape.input_dim, state
    ):
        raise ValueError("eligibility dimensions do not match tape")
    for name in CoreSignalGrads.__dataclass_fields__:
        value = getattr(signal_grads, name)
        if value is not None and tuple(value.shape) != (batch, steps, state):
            raise ValueError(f"{name} cotangent must have shape {(batch, steps, state)}")
    dtype = eligibility_state.dtype
    if dtype != torch.float32 or tape.y.dtype != torch.float32:
        raise TypeError("fused online replay requires FP32 tapes and eligibility")

    R, H = batch * steps, state
    dev = tape.y.device
    tape_fields = [_rows(getattr(tape, name), R, H) for name in (
        "budget", "energy", "saturation", "y", "expression", "gate_preactivation", "gate",
        "novelty", "plasticity")]
    dummy = torch.empty(1, device=dev, dtype=torch.float32)
    sig = {}
    for name in ("next_budget", "energy", "expression", "novelty", "plasticity", "saturation", "write", "y"):
        value = getattr(signal_grads, name)
        sig[name] = (dummy, 0, False) if value is None else (*_rows(value, R, H), True)

    alpha = torch.sigmoid(core.logit_alpha.detach().float())
    lam = F.softplus(core.log_lambda.detach().float())
    decay = torch.sigmoid(core.logit_saturation_decay.detach().float())
    sig_lam = torch.sigmoid(core.log_lambda.detach().float())
    hard_slope = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=dev).reshape(1)
    scalars = torch.cat([alpha.reshape(1), lam.reshape(1), decay.reshape(1), sig_lam.reshape(1),
                         hard_slope]).contiguous()
    ug_diag = core.Ug.weight.detach().diagonal().float().contiguous()
    a_diag = core.A.weight.detach().diagonal().float().contiguous()

    empty = lambda: torch.empty((batch, steps, H), device=dev, dtype=torch.float32)  # noqa: E731
    g_expr, g_gate_pre, g_budget, g_y_pre = empty(), empty(), empty(), empty()
    coef_w, coef_a, coef_g = empty(), empty(), empty()
    incoming = torch.empty((batch, steps, H, 3), device=dev, dtype=torch.float32)
    grid = (triton.cdiv(R, _ROWS), triton.cdiv(H, _COLS))
    chan_part = torch.empty((grid[0], 2, H), device=dev, dtype=torch.float32)
    scal_part = torch.empty((grid[0] * grid[1], 3), device=dev, dtype=torch.float32)
    _vjp_a[grid](
        *(value for value, _ in tape_fields), *(stride for _, stride in tape_fields),
        *(sig[n][0] for n in ("next_budget", "energy", "expression", "novelty", "plasticity", "saturation", "write")),
        *(sig[n][1] for n in ("next_budget", "energy", "expression", "novelty", "plasticity", "saturation", "write")),
        ug_diag, a_diag, scalars,
        g_expr, g_gate_pre, g_budget, incoming, coef_w, coef_a, coef_g,
        chan_part, scal_part,
        R, H,
        *(sig[n][2] for n in ("next_budget", "energy", "expression", "novelty", "plasticity", "saturation", "write")),
        BR=_ROWS, BC=_COLS, num_warps=4,
    )
    flat = lambda value: value.reshape(R, H)  # noqa: E731
    recurrent = torch.mm(flat(g_expr), core.A.weight.detach().float())
    recurrent.addmm_(flat(g_gate_pre), core.Ug.weight.detach().float())
    y_rows, y_stride = tape_fields[3]
    _vjp_b[grid](
        y_rows, y_stride, sig["y"][0], sig["y"][1], recurrent, g_budget, g_y_pre, incoming,
        R, H, HAS_Y=sig["y"][2], BR=_ROWS, BC=_COLS, num_warps=4,
    )

    gradients = {parameter: torch.zeros_like(parameter, dtype=torch.promote_types(parameter.dtype, torch.float32))
                 for parameter in core.parameters()}
    chan = chan_part.sum(0)
    scal = scal_part.sum(0)
    x = tape.inputs.to(dtype)
    y = tape.y.to(dtype)
    x_flat = x.reshape(R, -1)
    y_flat = y.reshape(R, H)
    direct = {
        core.beta: chan[0],
        core.Wg.bias: chan[1],
        core.logit_alpha: scal[0].reshape_as(core.logit_alpha),
        core.log_lambda: scal[1].reshape_as(core.log_lambda),
        core.logit_saturation_decay: scal[2].reshape_as(core.logit_saturation_decay),
        core.W.weight: flat(g_y_pre).T @ x_flat,
        core.A.weight: flat(g_expr).T @ y_flat,
        core.Wg.weight: flat(g_gate_pre).T @ x_flat,
        core.Ug.weight: flat(g_gate_pre).T @ y_flat,
    }
    input_gradient = torch.mm(flat(g_y_pre), core.W.weight.detach().float())
    input_gradient.addmm_(flat(g_gate_pre), core.Wg.weight.detach().float())

    local, global_, local_gradient, global_gradient, jbb = core_trace_scan_analytic(
        core, tape, incoming, eligibility_state)
    weights = _budget_trace_weights(jbb, incoming[..., 0])
    matrices = []
    for parameter, initial, coefficient, inputs in (
        (core.W.weight, eligibility_state.w_budget, coef_w, x),
        (core.A.weight, eligibility_state.a_budget, coef_a, y),
        (core.Wg.weight, eligibility_state.wg_budget, coef_g, x),
        (core.Ug.weight, eligibility_state.ug_budget, coef_g, y),
    ):
        final, gradient = _contract_budget_trace(initial, coefficient, inputs, weights)
        matrices.append(final)
        gradients[parameter].add_(gradient.to(gradients[parameter].dtype))
    for index, name in enumerate(CORE_LOCAL_PARAMETER_NAMES):
        parameter = core.get_parameter(name)
        gradients[parameter].add_(local_gradient[:, index].to(gradients[parameter].dtype))
    for index, name in enumerate(CORE_GLOBAL_PARAMETER_NAMES):
        parameter = core.get_parameter(name)
        gradients[parameter].add_(global_gradient[index].reshape_as(parameter).to(gradients[parameter].dtype))
    for parameter, gradient in direct.items():
        gradients[parameter].add_(gradient.to(gradients[parameter].dtype))
    return (
        CoreEligibilityState(*matrices, local, global_),
        {parameter: gradient.to(parameter.dtype) for parameter, gradient in gradients.items()},
        input_gradient.reshape_as(tape.inputs).to(tape.inputs.dtype),
    )


__all__ = ["core_online_replay_fused"]
