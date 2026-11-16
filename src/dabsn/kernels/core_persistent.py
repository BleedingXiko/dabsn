"""Persistent single-launch DABSN core scans (forward and BPTT reverse).

The core recurrence is dense and nonlinear, so it cannot be scanned in parallel
over time. What it can avoid is a dozen tiny kernel launches per time step. Each
scan here is ONE launch: every program owns a tile of (batch rows, hidden units),
keeps its recurrent state private, and loops over time on chip. Per step it
computes its slice of the recurrent product on tensor cores and then the whole
pointwise step of its tile. The only cross-program dependency per step is the
vector the recurrent product consumes (``y`` forward, the ``[dgate_pre,
d_expression]`` adjoint in reverse). Programs exchange it through a
double-buffered global buffer and one grid-wide barrier per step.

Two layouts share one definition of the math (the ``_fwd_math``/``_bwd_local``
helpers):

* ``single``: one tile per program whenever every tile fits on its own SM. State,
  parameters, reverse carries and gradient accumulators live in registers for the
  whole scan, and the next step's inputs/tapes are prefetched before the barrier.
* ``general``: programs loop over several tiles with state in global memory; used
  only when the tile count exceeds the SM count.

The grid never exceeds the SM count (all programs co-resident), each tile is owned
by exactly one program, and reductions run in a fixed order, so both scans are
bitwise deterministic and use no data atomics. The math is a line-for-line
transcription of ``batched_runtime._forward_step`` and ``_backward_step``.

Precision follows the framework policy. The recurrent product takes operands in
``op_dtype`` (the activation dtype under autocast, FP32 otherwise), accumulates in
FP32, and FP32 operands follow ``torch.get_float32_matmul_precision()``. The
recurrent state chain and every pointwise operation stay FP32.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only installs
    triton = None
    tl = None
    _HAS_TRITON = False


def _input_precision(op_dtype: torch.dtype) -> str:
    if op_dtype != torch.float32:
        return "ieee"
    return "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"


def _triton_at_least(major: int, minor: int) -> bool:
    if not _HAS_TRITON:
        return False
    try:
        parts = triton.__version__.split("+")[0].split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (ValueError, IndexError):
        return False


# The persistent kernels are validated on Triton 3.4. Under Triton 3.2 (the
# cuda-turing pin) they compile but return wrong gradients (measured on the
# same GPU: 3.4 passes, 3.2 fails), so older Triton keeps the batched Core.
_PERSISTENT_TRITON_OK = _triton_at_least(3, 4)

_SM_COUNT: dict[int, int] = {}


def _sm_count(device) -> int:
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    if index not in _SM_COUNT:
        _SM_COUNT[index] = int(torch.cuda.get_device_properties(index).multi_processor_count)
    return _SM_COUNT[index]


def persistent_core_supported(Wx: Tensor) -> bool:
    # A real launch needs real CUDA storage: never under tracing or fake
    # tensors (which may claim is_cuda while living on CPU/meta).
    if not (_HAS_TRITON and Wx.device.type == "cuda") or torch.compiler.is_compiling():
        return False
    from torch._subclasses.fake_tensor import is_fake

    if is_fake(Wx):
        return False
    if os.environ.get("DABSN_CORE_PERSISTENT", "1") != "1" or not _PERSISTENT_TRITON_OK:
        return False
    if Wx.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    return Wx.shape[1] >= 1 and Wx.shape[-1] >= 1


def _tune() -> dict:
    raw = os.environ.get("DABSN_CORE_PERSIST_TUNE", "")
    return dict(item.split("=") for item in raw.split(",") if item)


_SINGLE_TILES = ((16, 16), (32, 16), (16, 32), (32, 32), (64, 16), (64, 32), (128, 16), (128, 32))


def _layout(B: int, H: int, device, op_dtype: torch.dtype) -> dict:
    """Pick the single-tile layout when it fits the SM count, else the general one."""
    sms = _sm_count(device)
    tune = _tune()
    cfg = {"BK": int(tune.get("BK", 32 if op_dtype == torch.float32 else 64)),
           "S": int(tune.get("S", 3))}
    if tune.get("MODE", "auto") != "general":
        for BN, BM in _SINGLE_TILES:
            BM = max(16, min(BM, triton.next_power_of_2(B)))
            n_units, n_batch = triton.cdiv(H, BN), triton.cdiv(B, BM)
            if n_units * n_batch <= sms:
                cfg.update(mode="single", BM=BM, BN=BN, n_units=n_units, n_batch=n_batch,
                           P=n_units * n_batch, W=int(tune.get("W", 4 if BM * BN <= 512 else 8)))
                return cfg
    BN = 16
    while triton.cdiv(H, BN) > sms and BN < 128:
        BN *= 2
    BM = min(32, max(16, triton.next_power_of_2(B)))
    n_units = triton.cdiv(H, BN)
    cfg.update(mode="general", BM=BM, BN=BN, n_units=n_units, n_batch=triton.cdiv(B, BM),
               P=min(n_units, sms), W=int(tune.get("W", 4)))
    return cfg


if _HAS_TRITON:

    @triton.jit
    def _tanh(x):
        return 2.0 * tl.sigmoid(2.0 * x) - 1.0

    @triton.jit
    def _softplus(x):
        return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

    @triton.jit
    def _grid_barrier(Counter, target):
        tl.debug_barrier()
        tl.atomic_add(Counter, 1, sem="release", scope="gpu")
        cur = tl.atomic_add(Counter, 0, sem="acquire", scope="gpu")
        while cur < target:
            cur = tl.atomic_add(Counter, 0, sem="acquire", scope="gpu")
        tl.debug_barrier()

    @triton.jit
    def _p(Params, idx: tl.constexpr, n, H):
        return tl.load(Params + idx * H + n, mask=n < H, other=0.0)[None, :]

    @triton.jit
    def _load_params(Params, n, H):
        """Derived per-unit parameters, [1, BN] each, in one fixed order."""
        lk = _p(Params, 1, n, H)
        lr = _p(Params, 2, n, H)
        loglam = _p(Params, 17, n, H)
        return (
            _p(Params, 0, n, H),  # beta
            _softplus(lk),  # kappa
            tl.sigmoid(lr),  # recover
            _p(Params, 3, n, H), _p(Params, 4, n, H), _p(Params, 5, n, H),
            _p(Params, 6, n, H), _p(Params, 7, n, H),  # k_s k_y k_b k_n k_bias
            _p(Params, 8, n, H), _p(Params, 9, n, H), _p(Params, 10, n, H),
            _p(Params, 11, n, H), _p(Params, 12, n, H),  # r_s r_y r_b r_n r_bias
            tl.sigmoid(_p(Params, 13, n, H)),  # decay
            _p(Params, 14, n, H), _p(Params, 15, n, H),  # k_c r_c
            tl.sigmoid(_p(Params, 16, n, H)),  # alpha
            _softplus(loglam),  # lambda
            tl.sigmoid(loglam),  # d softplus(lambda)
            tl.sigmoid(_p(Params, 18, n, H)),  # suppress
            tl.sigmoid(lk),  # d softplus(kappa)
        )

    @triton.jit
    def _fwd_math(y, acc_g, expression, b, e, c, wgx,
                  beta, kappa, recover, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
                  decay, kc, rc, alpha, lam, suppress):
        gate = tl.minimum(tl.maximum(wgx + acc_g + 3.0, 0.0), 6.0) / 6.0
        novelty = _tanh(tl.abs(expression - b))
        c_new = decay * c + (1.0 - decay) * (novelty * (1.0 - e))
        neff = novelty * (1.0 - suppress * c_new)
        tb = _tanh(b)
        k_sig = ks * gate + ky * y + kb * tb + kn * neff + kbias + kc * c_new
        r_sig = rs * gate + ry * y + rb * tb + rn * neff + rbias + rc * c_new
        write_cost = kappa * tl.exp(0.5 * _tanh(k_sig))
        recovery = tl.minimum(tl.maximum(recover * tl.exp(0.5 * _tanh(r_sig)), 0.0), 1.0)
        plast = gate * e
        write = plast * expression
        nb = (1.0 - alpha) * b + beta + lam * write
        ne = tl.minimum(tl.maximum(e + recovery * (1.0 - e) - write_cost * plast, 0.0), 1.0)
        return gate, novelty, c_new, plast, write, nb, ne, neff, write_cost, recovery

    @triton.jit
    def _bwd_local(y, b_prev, c_prev, e_prev, gate, p, ay, c_t,
                   gy0, gb_new, gen, gc_new, gnov0, gp0, gay0, gwrite, ge_out,
                   beta, kappa, recover, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
                   decay, kc, rc, alpha, lam, sig_lam, suppress, sig_lk):
        """Reverse step up to the recurrent product (``_backward_step`` order)."""
        nov = _tanh(tl.abs(ay - b_prev))
        tanh_b = _tanh(b_prev)
        neff = nov * (1.0 - suppress * c_t)
        k_sig = ks * gate + ky * y + kb * tanh_b + kn * neff + kbias + kc * c_t
        r_sig = rs * gate + ry * y + rb * tanh_b + rn * neff + rbias + rc * c_t
        tanh_k = _tanh(k_sig)
        tanh_r = _tanh(r_sig)
        exp_k = tl.exp(0.5 * tanh_k)
        exp_r = tl.exp(0.5 * tanh_r)
        write_cost = kappa * exp_k
        recovery_pre = recover * exp_r
        recovery = tl.minimum(tl.maximum(recovery_pre, 0.0), 1.0)
        rc_grad = ((recovery_pre >= 0.0) & (recovery_pre <= 1.0)).to(tl.float32)
        energy_pre = e_prev + recovery * (1.0 - e_prev) - write_cost * p
        ge_pre = gen * ((energy_pre >= 0.0) & (energy_pre <= 1.0)).to(tl.float32)
        g_bprev_state = gb_new * (1.0 - alpha)
        g_eprev_state = ge_pre * (1.0 - recovery)
        gp = gp0 + gb_new * lam * ay - ge_pre * write_cost
        gay = gay0 + gb_new * lam * p
        g_write_cost = -ge_pre * p
        g_recovery = ge_pre * (1.0 - e_prev)
        d_expr_main = gay + gwrite * p
        gp = gp + gwrite * ay
        g_k = g_write_cost * write_cost * 0.5 * (1.0 - tanh_k * tanh_k)
        g_rpre = g_recovery * rc_grad
        g_r = g_rpre * recovery_pre * 0.5 * (1.0 - tanh_r * tanh_r)
        g_gate_energy = g_k * ks + g_r * rs
        gy_energy = g_k * ky + g_r * ry
        gb_energy = (g_k * kb + g_r * rb) * (1.0 - tanh_b * tanh_b)
        gnov_eff = g_k * kn + g_r * rn
        gc_energy = g_k * kc + g_r * rc
        gate_open = ((gate > 0.0) & (gate < 1.0)).to(tl.float32)
        dgate_pre = (gp * e_prev + g_gate_energy) * (gate_open / 6.0)
        ge_gate = gp * gate
        stress = nov * (1.0 - e_prev)
        c_new = decay * c_prev + (1.0 - decay) * stress
        g_c_new = gc_new + gc_energy - gnov_eff * nov * suppress
        g_stress = g_c_new * (1.0 - decay)
        g_nov = gnov_eff * (1.0 - suppress * c_new) + g_stress * (1.0 - e_prev)
        g_e_prev_cort = -g_stress * nov
        diff = ay - b_prev
        nc = _tanh(tl.abs(diff))
        sgn = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
        d_nov = (gnov0 + g_nov) * (1.0 - nc * nc) * sgn
        d_expr = d_expr_main + d_nov
        return (
            dgate_pre,
            d_expr,
            gy0 + gy_energy,  # gy before the recurrent adjoint
            g_bprev_state + gb_energy - d_nov,  # next gb before the y pre-activation
            g_eprev_state + ge_gate + g_e_prev_cort + ge_out,  # next ge
            g_c_new * decay,  # next gc
            gb_new,
            g_write_cost * exp_k * sig_lk,
            g_rpre * exp_r * recover * (1.0 - recover),
            g_k * gate, g_k * y, g_k * tanh_b, g_k * neff, g_k,
            g_r * gate, g_r * y, g_r * tanh_b, g_r * neff, g_r,
            g_c_new * (c_prev - stress) * decay * (1.0 - decay),
            g_k * c_t, g_r * c_t,
            gb_new * (-b_prev) * alpha * (1.0 - alpha),
            gb_new * (p * ay) * sig_lam,
            gnov_eff * nov * (-c_new) * suppress * (1.0 - suppress),
        )

    @triton.jit
    def _fwd_product(YBuf, cur, W, m, n, B, H,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr):
        rk = tl.arange(0, BK)
        mmask = m < B
        nmask = n < H
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_e = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, H, BK):
            k = k0 + rk
            kmask = k < H
            ya = tl.load(YBuf + cur + m[:, None] * H + k[None, :],
                         mask=mmask[:, None] & kmask[None, :], other=0.0, cache_modifier=".cg")
            wg = tl.load(W + n[:, None] * H + k[None, :],
                         mask=nmask[:, None] & kmask[None, :], other=0.0)
            wa = tl.load(W + (H + n[:, None]) * H + k[None, :],
                         mask=nmask[:, None] & kmask[None, :], other=0.0)
            acc_g = tl.dot(ya, tl.trans(wg), acc_g, input_precision=IP)
            acc_e = tl.dot(ya, tl.trans(wa), acc_e, input_precision=IP)
        return acc_g, acc_e

    @triton.jit
    def _bwd_product(DBuf, dcur, Wt, m, n, B, H,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr):
        rk = tl.arange(0, BK)
        mmask = m < B
        nmask = n < H
        adj = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, 2 * H, BK):
            k = k0 + rk
            kmask = k < 2 * H
            d = tl.load(DBuf + dcur + m[:, None] * (2 * H) + k[None, :],
                        mask=mmask[:, None] & kmask[None, :], other=0.0, cache_modifier=".cg")
            wt = tl.load(Wt + n[:, None] * (2 * H) + k[None, :],
                         mask=nmask[:, None] & kmask[None, :], other=0.0)
            adj = tl.dot(d, tl.trans(wt), adj, input_precision=IP)
        return adj

    @triton.jit
    def _bwd_load(U, ETape, CTape, GTape, Plas, Expr, B0, C0,
                  GU, GNov, GP, GAy, GWr, GEo, GCo, m, n, t, B, T, H,
                  HAS_GU: tl.constexpr, HAS_GNOV: tl.constexpr, HAS_GP: tl.constexpr,
                  HAS_GAY: tl.constexpr, HAS_GWR: tl.constexpr, HAS_GEO: tl.constexpr,
                  HAS_GCO: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        """Every tape and output-gradient tile the reverse step ``t`` reads (t may be -1)."""
        valid = (m[:, None] < B) & (n[None, :] < H) & (t >= 0)
        ts = tl.maximum(t, 0)
        sidx = m[:, None] * H + n[None, :]
        tidx = (m[:, None] * T + ts) * H + n[None, :]
        uidx = (m[:, None] * T + ts) * (2 * H) + n[None, :]
        later = valid & (t > 0)
        first = valid & (t == 0)
        y = tl.load(U + uidx, mask=valid, other=0.0).to(tl.float32)
        b_prev = tl.where(t > 0, tl.load(U + uidx - H, mask=later, other=0.0).to(tl.float32),
                          tl.load(B0 + sidx, mask=first, other=0.0))
        c_prev = tl.where(t > 0, tl.load(CTape + tidx - H, mask=later, other=0.0).to(tl.float32),
                          tl.load(C0 + sidx, mask=first, other=0.0))
        e_prev = tl.load(ETape + tidx, mask=valid, other=0.0).to(tl.float32)
        gate = tl.load(GTape + tidx, mask=valid, other=0.0).to(tl.float32)
        p = tl.load(Plas + tidx, mask=valid, other=0.0).to(tl.float32)
        ay = tl.load(Expr + tidx, mask=valid, other=0.0).to(tl.float32)
        c_t = tl.load(CTape + tidx, mask=valid, other=0.0).to(tl.float32)
        zero = tl.zeros((BM, BN), dtype=tl.float32)
        if HAS_GU:
            gy0 = tl.load(GU + uidx, mask=valid, other=0.0).to(tl.float32)
            gbo = tl.load(GU + uidx + H, mask=valid, other=0.0).to(tl.float32)
        else:
            gy0 = zero
            gbo = zero
        gnov0 = tl.load(GNov + tidx, mask=valid, other=0.0).to(tl.float32) if HAS_GNOV else zero
        gp0 = tl.load(GP + tidx, mask=valid, other=0.0).to(tl.float32) if HAS_GP else zero
        gay0 = tl.load(GAy + tidx, mask=valid, other=0.0).to(tl.float32) if HAS_GAY else zero
        gwrite = tl.load(GWr + tidx, mask=valid, other=0.0).to(tl.float32) if HAS_GWR else zero
        ge_out = tl.load(GEo + tidx, mask=valid, other=0.0).to(tl.float32) if HAS_GEO else zero
        gco = tl.load(GCo + tidx, mask=valid, other=0.0).to(tl.float32) if HAS_GCO else zero
        return y, b_prev, c_prev, e_prev, gate, p, ay, c_t, gy0, gbo, gnov0, gp0, gay0, gwrite, ge_out, gco

    # ----------------------------------------------------------- forward, single
    @triton.jit
    def _core_fwd_single(
        Wx, Wgx, W, Params, B0, E0, C0, YBuf, Counter,
        U, Nov, Plas, Expr, Write, ETape, CTape, GTape, FB, FE, FC,
        BPrev, CPrev, GPre, NEff, WCost, Recov,
        B, T, H, P, N_UNITS,
        EXTRA: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        OPT = YBuf.dtype.element_ty
        TT = U.dtype.element_ty
        n = (pid % N_UNITS) * BN + tl.arange(0, BN)
        m = (pid // N_UNITS) * BM + tl.arange(0, BM)
        mask = (m[:, None] < B) & (n[None, :] < H)
        sidx = m[:, None] * H + n[None, :]
        (beta, kappa, recover, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
         decay, kc, rc, alpha, lam, sig_lam, suppress, sig_lk) = _load_params(Params, n, H)
        b = tl.load(B0 + sidx, mask=mask, other=0.0)
        e = tl.load(E0 + sidx, mask=mask, other=0.0)
        c = tl.load(C0 + sidx, mask=mask, other=0.0)
        row = m[:, None] * T
        wx = tl.load(Wx + row * H + n[None, :], mask=mask, other=0.0).to(tl.float32)
        wgx = tl.load(Wgx + row * H + n[None, :], mask=mask, other=0.0).to(tl.float32)
        y = _tanh(wx + b)
        tl.store(YBuf + sidx, y.to(OPT), mask=mask)
        _grid_barrier(Counter, P)
        for t in range(0, T):
            tidx = (row + t) * H + n[None, :]
            ahead = mask & (t + 1 < T)
            wx_n = tl.load(Wx + tidx + H, mask=ahead, other=0.0).to(tl.float32)
            wgx_n = tl.load(Wgx + tidx + H, mask=ahead, other=0.0).to(tl.float32)
            acc_g, expression = _fwd_product(YBuf, (t % 2) * B * H, W, m, n, B, H, BM, BN, BK, IP)
            c_prev = c
            gate, novelty, c, plast, write, nb, ne, neff, write_cost, recovery = _fwd_math(
                y, acc_g, expression, b, e, c, wgx,
                beta, kappa, recover, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
                decay, kc, rc, alpha, lam, suppress)
            y_n = _tanh(wx_n + nb)
            tl.store(YBuf + ((t + 1) % 2) * B * H + sidx, y_n.to(OPT), mask=ahead)
            uidx = (row + t) * (2 * H) + n[None, :]
            tl.store(U + uidx, y.to(TT), mask=mask)
            tl.store(U + uidx + H, nb.to(TT), mask=mask)
            tl.store(Nov + tidx, novelty.to(TT), mask=mask)
            tl.store(Plas + tidx, plast.to(TT), mask=mask)
            tl.store(Expr + tidx, expression.to(TT), mask=mask)
            tl.store(Write + tidx, write.to(TT), mask=mask)
            tl.store(ETape + tidx, e.to(TT), mask=mask)
            tl.store(CTape + tidx, c.to(TT), mask=mask)
            tl.store(GTape + tidx, gate.to(TT), mask=mask)
            if EXTRA:
                tl.store(BPrev + tidx, b, mask=mask)
                tl.store(CPrev + tidx, c_prev, mask=mask)
                tl.store(GPre + tidx, gate * 6.0 - 3.0, mask=mask)
                tl.store(NEff + tidx, neff, mask=mask)
                tl.store(WCost + tidx, write_cost, mask=mask)
                tl.store(Recov + tidx, recovery, mask=mask)
            b = nb
            e = ne
            y = y_n
            wgx = wgx_n
            if t + 1 < T:
                _grid_barrier(Counter, (t + 2) * P)
        tl.store(FB + sidx, b, mask=mask)
        tl.store(FE + sidx, e, mask=mask)
        tl.store(FC + sidx, c, mask=mask)

    # ---------------------------------------------------------- forward, general
    @triton.jit
    def _core_fwd_general(
        Wx, Wgx, W, Params, Bst, Est, Cst, YBuf, Counter,
        U, Nov, Plas, Expr, Write, ETape, CTape, GTape,
        BPrev, CPrev, GPre, NEff, WCost, Recov,
        B, T, H, P, N_UNITS, N_BATCH,
        EXTRA: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        OPT = YBuf.dtype.element_ty
        TT = U.dtype.element_ty
        for nt in range(pid, N_UNITS, P):
            n = nt * BN + tl.arange(0, BN)
            for mt in range(0, N_BATCH):
                m = mt * BM + tl.arange(0, BM)
                mask = (m[:, None] < B) & (n[None, :] < H)
                sidx = m[:, None] * H + n[None, :]
                b = tl.load(Bst + sidx, mask=mask, other=0.0)
                wx = tl.load(Wx + (m[:, None] * T) * H + n[None, :], mask=mask, other=0.0).to(tl.float32)
                tl.store(YBuf + sidx, _tanh(wx + b).to(OPT), mask=mask)
        _grid_barrier(Counter, P)
        for t in range(0, T):
            for nt in range(pid, N_UNITS, P):
                n = nt * BN + tl.arange(0, BN)
                (beta, kappa, recover, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
                 decay, kc, rc, alpha, lam, sig_lam, suppress, sig_lk) = _load_params(Params, n, H)
                for mt in range(0, N_BATCH):
                    m = mt * BM + tl.arange(0, BM)
                    mask = (m[:, None] < B) & (n[None, :] < H)
                    ahead = mask & (t + 1 < T)
                    sidx = m[:, None] * H + n[None, :]
                    tidx = (m[:, None] * T + t) * H + n[None, :]
                    acc_g, expression = _fwd_product(YBuf, (t % 2) * B * H, W, m, n, B, H, BM, BN, BK, IP)
                    b = tl.load(Bst + sidx, mask=mask, other=0.0)
                    e = tl.load(Est + sidx, mask=mask, other=0.0)
                    c = tl.load(Cst + sidx, mask=mask, other=0.0)
                    y = _tanh(tl.load(Wx + tidx, mask=mask, other=0.0).to(tl.float32) + b)
                    wgx = tl.load(Wgx + tidx, mask=mask, other=0.0).to(tl.float32)
                    c_prev = c
                    gate, novelty, c, plast, write, nb, ne, neff, write_cost, recovery = _fwd_math(
                        y, acc_g, expression, b, e, c, wgx,
                        beta, kappa, recover, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
                        decay, kc, rc, alpha, lam, suppress)
                    uidx = (m[:, None] * T + t) * (2 * H) + n[None, :]
                    tl.store(U + uidx, y.to(TT), mask=mask)
                    tl.store(U + uidx + H, nb.to(TT), mask=mask)
                    tl.store(Nov + tidx, novelty.to(TT), mask=mask)
                    tl.store(Plas + tidx, plast.to(TT), mask=mask)
                    tl.store(Expr + tidx, expression.to(TT), mask=mask)
                    tl.store(Write + tidx, write.to(TT), mask=mask)
                    tl.store(ETape + tidx, e.to(TT), mask=mask)
                    tl.store(CTape + tidx, c.to(TT), mask=mask)
                    tl.store(GTape + tidx, gate.to(TT), mask=mask)
                    if EXTRA:
                        tl.store(BPrev + tidx, b, mask=mask)
                        tl.store(CPrev + tidx, c_prev, mask=mask)
                        tl.store(GPre + tidx, gate * 6.0 - 3.0, mask=mask)
                        tl.store(NEff + tidx, neff, mask=mask)
                        tl.store(WCost + tidx, write_cost, mask=mask)
                        tl.store(Recov + tidx, recovery, mask=mask)
                    tl.store(Bst + sidx, nb, mask=mask)
                    tl.store(Est + sidx, ne, mask=mask)
                    tl.store(Cst + sidx, c, mask=mask)
                    wx1 = tl.load(Wx + tidx + H, mask=ahead, other=0.0).to(tl.float32)
                    tl.store(YBuf + ((t + 1) % 2) * B * H + sidx, _tanh(wx1 + nb).to(OPT), mask=ahead)
            if t + 1 < T:
                _grid_barrier(Counter, (t + 2) * P)

    # ----------------------------------------------------------- reverse, single
    @triton.jit
    def _core_bwd_single(
        U, ETape, CTape, GTape, Plas, Expr, B0, C0,
        GU, GNov, GP, GAy, GWr, GEo, GCo,
        Wt, Params, GB, GE, GC, DBuf, Counter,
        GWx, GWgx, Day, PPart, SPart,
        B, T, H, P, N_UNITS,
        HAS_GU: tl.constexpr, HAS_GNOV: tl.constexpr, HAS_GP: tl.constexpr,
        HAS_GAY: tl.constexpr, HAS_GWR: tl.constexpr, HAS_GEO: tl.constexpr,
        HAS_GCO: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        OPT = DBuf.dtype.element_ty
        GT = GWx.dtype.element_ty
        mt = pid // N_UNITS
        n = (pid % N_UNITS) * BN + tl.arange(0, BN)
        m = mt * BM + tl.arange(0, BM)
        mask = (m[:, None] < B) & (n[None, :] < H)
        fm = mask.to(tl.float32)
        sidx = m[:, None] * H + n[None, :]
        didx = m[:, None] * (2 * H) + n[None, :]
        (beta, kappa, recover, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
         decay, kc, rc, alpha, lam, sig_lam, suppress, sig_lk) = _load_params(Params, n, H)
        gb = tl.load(GB + sidx, mask=mask, other=0.0)
        ge = tl.load(GE + sidx, mask=mask, other=0.0)
        gc = tl.load(GC + sidx, mask=mask, other=0.0)
        a0 = tl.zeros((BM, BN), dtype=tl.float32)
        a1 = tl.zeros((BM, BN), dtype=tl.float32)
        a2 = tl.zeros((BM, BN), dtype=tl.float32)
        a3 = tl.zeros((BM, BN), dtype=tl.float32)
        a4 = tl.zeros((BM, BN), dtype=tl.float32)
        a5 = tl.zeros((BM, BN), dtype=tl.float32)
        a6 = tl.zeros((BM, BN), dtype=tl.float32)
        a7 = tl.zeros((BM, BN), dtype=tl.float32)
        a8 = tl.zeros((BM, BN), dtype=tl.float32)
        a9 = tl.zeros((BM, BN), dtype=tl.float32)
        a10 = tl.zeros((BM, BN), dtype=tl.float32)
        a11 = tl.zeros((BM, BN), dtype=tl.float32)
        a12 = tl.zeros((BM, BN), dtype=tl.float32)
        a13 = tl.zeros((BM, BN), dtype=tl.float32)
        a14 = tl.zeros((BM, BN), dtype=tl.float32)
        a15 = tl.zeros((BM, BN), dtype=tl.float32)
        s0 = tl.zeros((BM, BN), dtype=tl.float32)
        s1 = tl.zeros((BM, BN), dtype=tl.float32)
        s2 = tl.zeros((BM, BN), dtype=tl.float32)
        (y, b_prev, c_prev, e_prev, gate, p, ay, c_t, gy0, gbo, gnov0, gp0, gay0, gwrite, ge_out,
         gco) = _bwd_load(U, ETape, CTape, GTape, Plas, Expr, B0, C0, GU, GNov, GP, GAy, GWr, GEo,
                          GCo, m, n, T - 1, B, T, H, HAS_GU, HAS_GNOV, HAS_GP, HAS_GAY, HAS_GWR,
                          HAS_GEO, HAS_GCO, BM, BN)
        for i in range(0, T):
            t = T - 1 - i
            dcur = (t % 2) * B * (2 * H)
            tidx = (m[:, None] * T + t) * H + n[None, :]
            (dgate_pre, d_expr, gyp, gbp, ge, gc,
             p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
             q0, q1, q2) = _bwd_local(
                y, b_prev, c_prev, e_prev, gate, p, ay, c_t,
                gy0, gbo + gb, ge, gc + gco, gnov0, gp0, gay0, gwrite, ge_out,
                beta, kappa, recover, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
                decay, kc, rc, alpha, lam, sig_lam, suppress, sig_lk)
            tl.store(DBuf + dcur + didx, dgate_pre.to(OPT), mask=mask)
            tl.store(DBuf + dcur + didx + H, d_expr.to(OPT), mask=mask)
            tl.store(GWgx + tidx, dgate_pre.to(GT), mask=mask)
            tl.store(Day + tidx, d_expr.to(GT), mask=mask)
            a0 += p0 * fm
            a1 += p1 * fm
            a2 += p2 * fm
            a3 += p3 * fm
            a4 += p4 * fm
            a5 += p5 * fm
            a6 += p6 * fm
            a7 += p7 * fm
            a8 += p8 * fm
            a9 += p9 * fm
            a10 += p10 * fm
            a11 += p11 * fm
            a12 += p12 * fm
            a13 += p13 * fm
            a14 += p14 * fm
            a15 += p15 * fm
            s0 += q0 * fm
            s1 += q1 * fm
            s2 += q2 * fm
            y_t = y
            (y, b_prev, c_prev, e_prev, gate, p, ay, c_t, gy0, gbo, gnov0, gp0, gay0, gwrite,
             ge_out, gco) = _bwd_load(U, ETape, CTape, GTape, Plas, Expr, B0, C0, GU, GNov, GP,
                                      GAy, GWr, GEo, GCo, m, n, t - 1, B, T, H, HAS_GU, HAS_GNOV,
                                      HAS_GP, HAS_GAY, HAS_GWR, HAS_GEO, HAS_GCO, BM, BN)
            _grid_barrier(Counter, (i + 1) * P)
            adj = _bwd_product(DBuf, dcur, Wt, m, n, B, H, BM, BN, BK, IP)
            gpre = (gyp + adj) * (1.0 - y_t * y_t)
            gb = gbp + gpre
            tl.store(GWx + tidx, gpre.to(GT), mask=mask)
        tl.store(GB + sidx, gb, mask=mask)
        tl.store(GE + sidx, ge, mask=mask)
        tl.store(GC + sidx, gc, mask=mask)
        nmask = n < H
        base = PPart + mt * 16 * H + n
        tl.store(base + 0 * H, tl.sum(a0, 0), mask=nmask)
        tl.store(base + 1 * H, tl.sum(a1, 0), mask=nmask)
        tl.store(base + 2 * H, tl.sum(a2, 0), mask=nmask)
        tl.store(base + 3 * H, tl.sum(a3, 0), mask=nmask)
        tl.store(base + 4 * H, tl.sum(a4, 0), mask=nmask)
        tl.store(base + 5 * H, tl.sum(a5, 0), mask=nmask)
        tl.store(base + 6 * H, tl.sum(a6, 0), mask=nmask)
        tl.store(base + 7 * H, tl.sum(a7, 0), mask=nmask)
        tl.store(base + 8 * H, tl.sum(a8, 0), mask=nmask)
        tl.store(base + 9 * H, tl.sum(a9, 0), mask=nmask)
        tl.store(base + 10 * H, tl.sum(a10, 0), mask=nmask)
        tl.store(base + 11 * H, tl.sum(a11, 0), mask=nmask)
        tl.store(base + 12 * H, tl.sum(a12, 0), mask=nmask)
        tl.store(base + 13 * H, tl.sum(a13, 0), mask=nmask)
        tl.store(base + 14 * H, tl.sum(a14, 0), mask=nmask)
        tl.store(base + 15 * H, tl.sum(a15, 0), mask=nmask)
        tl.store(SPart + pid * 3 + 0, tl.sum(tl.sum(s0, 1), 0))
        tl.store(SPart + pid * 3 + 1, tl.sum(tl.sum(s1, 1), 0))
        tl.store(SPart + pid * 3 + 2, tl.sum(tl.sum(s2, 1), 0))

    # ---------------------------------------------------------- reverse, general
    @triton.jit
    def _core_bwd_general(
        U, ETape, CTape, GTape, Plas, Expr, B0, C0,
        GU, GNov, GP, GAy, GWr, GEo, GCo,
        Wt, Params, GB, GE, GC, GYP, GBP, DBuf, Counter,
        GWx, GWgx, Day, PAcc, SPart,
        B, T, H, P, N_UNITS, N_BATCH,
        HAS_GU: tl.constexpr, HAS_GNOV: tl.constexpr, HAS_GP: tl.constexpr,
        HAS_GAY: tl.constexpr, HAS_GWR: tl.constexpr, HAS_GEO: tl.constexpr,
        HAS_GCO: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        OPT = DBuf.dtype.element_ty
        GT = GWx.dtype.element_ty
        s_alpha = 0.0
        s_lam = 0.0
        s_sup = 0.0
        for i in range(0, T):
            t = T - 1 - i
            dcur = (t % 2) * B * (2 * H)
            for nt in range(pid, N_UNITS, P):
                n = nt * BN + tl.arange(0, BN)
                nmask = n < H
                (beta, kappa, recover, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
                 decay, kc, rc, alpha, lam, sig_lam, suppress, sig_lk) = _load_params(Params, n, H)
                a0 = tl.zeros((BN,), dtype=tl.float32)
                a1 = tl.zeros((BN,), dtype=tl.float32)
                a2 = tl.zeros((BN,), dtype=tl.float32)
                a3 = tl.zeros((BN,), dtype=tl.float32)
                a4 = tl.zeros((BN,), dtype=tl.float32)
                a5 = tl.zeros((BN,), dtype=tl.float32)
                a6 = tl.zeros((BN,), dtype=tl.float32)
                a7 = tl.zeros((BN,), dtype=tl.float32)
                a8 = tl.zeros((BN,), dtype=tl.float32)
                a9 = tl.zeros((BN,), dtype=tl.float32)
                a10 = tl.zeros((BN,), dtype=tl.float32)
                a11 = tl.zeros((BN,), dtype=tl.float32)
                a12 = tl.zeros((BN,), dtype=tl.float32)
                a13 = tl.zeros((BN,), dtype=tl.float32)
                a14 = tl.zeros((BN,), dtype=tl.float32)
                a15 = tl.zeros((BN,), dtype=tl.float32)
                for mt in range(0, N_BATCH):
                    m = mt * BM + tl.arange(0, BM)
                    mask = (m[:, None] < B) & nmask[None, :]
                    fm = mask.to(tl.float32)
                    sidx = m[:, None] * H + n[None, :]
                    tidx = (m[:, None] * T + t) * H + n[None, :]
                    didx = m[:, None] * (2 * H) + n[None, :]
                    (y, b_prev, c_prev, e_prev, gate, p, ay, c_t, gy0, gbo, gnov0, gp0, gay0,
                     gwrite, ge_out, gco) = _bwd_load(
                        U, ETape, CTape, GTape, Plas, Expr, B0, C0, GU, GNov, GP, GAy, GWr, GEo,
                        GCo, m, n, t, B, T, H, HAS_GU, HAS_GNOV, HAS_GP, HAS_GAY, HAS_GWR,
                        HAS_GEO, HAS_GCO, BM, BN)
                    gb = tl.load(GB + sidx, mask=mask, other=0.0)
                    ge = tl.load(GE + sidx, mask=mask, other=0.0)
                    gc = tl.load(GC + sidx, mask=mask, other=0.0)
                    (dgate_pre, d_expr, gyp, gbp, ge, gc,
                     p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
                     q0, q1, q2) = _bwd_local(
                        y, b_prev, c_prev, e_prev, gate, p, ay, c_t,
                        gy0, gbo + gb, ge, gc + gco, gnov0, gp0, gay0, gwrite, ge_out,
                        beta, kappa, recover, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
                        decay, kc, rc, alpha, lam, sig_lam, suppress, sig_lk)
                    tl.store(DBuf + dcur + didx, dgate_pre.to(OPT), mask=mask)
                    tl.store(DBuf + dcur + didx + H, d_expr.to(OPT), mask=mask)
                    tl.store(GWgx + tidx, dgate_pre.to(GT), mask=mask)
                    tl.store(Day + tidx, d_expr.to(GT), mask=mask)
                    tl.store(GYP + sidx, gyp, mask=mask)
                    tl.store(GBP + sidx, gbp, mask=mask)
                    tl.store(GE + sidx, ge, mask=mask)
                    tl.store(GC + sidx, gc, mask=mask)
                    a0 += tl.sum(p0 * fm, 0)
                    a1 += tl.sum(p1 * fm, 0)
                    a2 += tl.sum(p2 * fm, 0)
                    a3 += tl.sum(p3 * fm, 0)
                    a4 += tl.sum(p4 * fm, 0)
                    a5 += tl.sum(p5 * fm, 0)
                    a6 += tl.sum(p6 * fm, 0)
                    a7 += tl.sum(p7 * fm, 0)
                    a8 += tl.sum(p8 * fm, 0)
                    a9 += tl.sum(p9 * fm, 0)
                    a10 += tl.sum(p10 * fm, 0)
                    a11 += tl.sum(p11 * fm, 0)
                    a12 += tl.sum(p12 * fm, 0)
                    a13 += tl.sum(p13 * fm, 0)
                    a14 += tl.sum(p14 * fm, 0)
                    a15 += tl.sum(p15 * fm, 0)
                    s_alpha += tl.sum(tl.sum(q0 * fm, 1), 0)
                    s_lam += tl.sum(tl.sum(q1 * fm, 1), 0)
                    s_sup += tl.sum(tl.sum(q2 * fm, 1), 0)
                base = PAcc + n
                tl.store(base + 0 * H, tl.load(base + 0 * H, mask=nmask, other=0.0) + a0, mask=nmask)
                tl.store(base + 1 * H, tl.load(base + 1 * H, mask=nmask, other=0.0) + a1, mask=nmask)
                tl.store(base + 2 * H, tl.load(base + 2 * H, mask=nmask, other=0.0) + a2, mask=nmask)
                tl.store(base + 3 * H, tl.load(base + 3 * H, mask=nmask, other=0.0) + a3, mask=nmask)
                tl.store(base + 4 * H, tl.load(base + 4 * H, mask=nmask, other=0.0) + a4, mask=nmask)
                tl.store(base + 5 * H, tl.load(base + 5 * H, mask=nmask, other=0.0) + a5, mask=nmask)
                tl.store(base + 6 * H, tl.load(base + 6 * H, mask=nmask, other=0.0) + a6, mask=nmask)
                tl.store(base + 7 * H, tl.load(base + 7 * H, mask=nmask, other=0.0) + a7, mask=nmask)
                tl.store(base + 8 * H, tl.load(base + 8 * H, mask=nmask, other=0.0) + a8, mask=nmask)
                tl.store(base + 9 * H, tl.load(base + 9 * H, mask=nmask, other=0.0) + a9, mask=nmask)
                tl.store(base + 10 * H, tl.load(base + 10 * H, mask=nmask, other=0.0) + a10, mask=nmask)
                tl.store(base + 11 * H, tl.load(base + 11 * H, mask=nmask, other=0.0) + a11, mask=nmask)
                tl.store(base + 12 * H, tl.load(base + 12 * H, mask=nmask, other=0.0) + a12, mask=nmask)
                tl.store(base + 13 * H, tl.load(base + 13 * H, mask=nmask, other=0.0) + a13, mask=nmask)
                tl.store(base + 14 * H, tl.load(base + 14 * H, mask=nmask, other=0.0) + a14, mask=nmask)
                tl.store(base + 15 * H, tl.load(base + 15 * H, mask=nmask, other=0.0) + a15, mask=nmask)
            _grid_barrier(Counter, (i + 1) * P)
            for nt in range(pid, N_UNITS, P):
                n = nt * BN + tl.arange(0, BN)
                for mt in range(0, N_BATCH):
                    m = mt * BM + tl.arange(0, BM)
                    mask = (m[:, None] < B) & (n[None, :] < H)
                    sidx = m[:, None] * H + n[None, :]
                    tidx = (m[:, None] * T + t) * H + n[None, :]
                    adj = _bwd_product(DBuf, dcur, Wt, m, n, B, H, BM, BN, BK, IP)
                    y = tl.load(U + (m[:, None] * T + t) * (2 * H) + n[None, :], mask=mask,
                                other=0.0).to(tl.float32)
                    gpre = (tl.load(GYP + sidx, mask=mask, other=0.0) + adj) * (1.0 - y * y)
                    tl.store(GB + sidx, tl.load(GBP + sidx, mask=mask, other=0.0) + gpre, mask=mask)
                    tl.store(GWx + tidx, gpre.to(GT), mask=mask)
        tl.store(SPart + pid * 3 + 0, s_alpha)
        tl.store(SPart + pid * 3 + 1, s_lam)
        tl.store(SPart + pid * 3 + 2, s_sup)


def pack_core_params(params) -> Tensor:
    """[19, H] FP32 in the batched-runtime parameter order (scalars broadcast)."""
    H = max(int(value.numel()) for value in params)
    return torch.stack(
        [value.detach().float().reshape(-1).expand(H) if value.numel() == 1
         else value.detach().float().reshape(H) for value in params]
    ).contiguous()


def persistent_forward_tapes(
    Wx: Tensor,
    Wgx: Tensor,
    recurrent: Tensor,
    params,
    state: tuple[Tensor, Tensor, Tensor],
    *,
    tape_dtype: torch.dtype,
    online_extras: bool = False,
):
    """One-launch forward scan. Returns (U, nov, plas, expr, write, e, c, gate, b, e, c).

    ``recurrent`` is ``cat(Ug, A)`` already in the GEMM operand dtype. The
    returned final state is FP32. With ``online_extras`` six more FP32 tapes
    follow: (budget_prev, saturation_prev, gate_pre, novelty_effective,
    write_cost, recovery), exactly the extra values ``CoreOnlineTape`` holds,
    with ``gate_pre = 6 * gate - 3`` as the online tape defines it.
    """
    B, T, H = Wx.shape
    dev = Wx.device
    Wx = Wx.contiguous()
    Wgx = Wgx.contiguous()
    recurrent = recurrent.contiguous()
    packed = pack_core_params(params)
    b0, e0, c0 = (value.to(device=dev, dtype=torch.float32).contiguous() for value in state)
    U = torch.empty((B, T, 2 * H), device=dev, dtype=tape_dtype)
    tapes = [torch.empty((B, T, H), device=dev, dtype=tape_dtype) for _ in range(7)]
    ybuf = torch.empty((2, B, H), device=dev, dtype=recurrent.dtype)
    counter = torch.zeros(1, device=dev, dtype=torch.int32)
    if online_extras:
        extras = [torch.empty((B, T, H), device=dev, dtype=torch.float32) for _ in range(6)]
    else:
        extras = [ybuf] * 6  # never touched: EXTRA is a constexpr
    cfg = _layout(B, H, dev, recurrent.dtype)
    meta = dict(BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], IP=_input_precision(recurrent.dtype),
                EXTRA=bool(online_extras), num_warps=cfg["W"], num_stages=cfg["S"])
    if cfg["mode"] == "single":
        final = [torch.empty((B, H), device=dev, dtype=torch.float32) for _ in range(3)]
        _core_fwd_single[(cfg["P"],)](
            Wx, Wgx, recurrent, packed, b0, e0, c0, ybuf, counter,
            U, *tapes, *final, *extras,
            B, T, H, cfg["P"], cfg["n_units"], **meta,
        )
    else:
        final = [b0.clone(), e0.clone(), c0.clone()]
        _core_fwd_general[(cfg["P"],)](
            Wx, Wgx, recurrent, packed, *final, ybuf, counter,
            U, *tapes, *extras,
            B, T, H, cfg["P"], cfg["n_units"], cfg["n_batch"], **meta,
        )
    if online_extras:
        return (U, *tapes, *final, *extras)
    return (U, *tapes, *final)


def persistent_backward(
    *,
    reads,
    grads,
    initial,
    final_grads,
    recurrent: Tensor,
    params,
    out_dtype: torch.dtype,
):
    """One-launch BPTT reverse. Returns gWx, gWgx, day, param_acc[16,H], scalar[3], gb, ge, gc."""
    U, e_tape, c_tape, s_tape, plasticity, expression = (value.contiguous() for value in reads)
    B, T, H = plasticity.shape
    dev = U.device
    initial_b, initial_c = (value.to(device=dev, dtype=torch.float32).contiguous() for value in initial)
    dummy = torch.empty(1, device=dev, dtype=torch.float32)
    flags, gptrs = [], []
    for value in grads:
        flags.append(value is not None)
        gptrs.append(dummy if value is None else value.contiguous())
    packed = pack_core_params(params)
    gb, ge, gc = (value.to(device=dev, dtype=torch.float32).contiguous().clone() for value in final_grads)
    recurrent_t = recurrent.t().contiguous()  # [H, 2H]: contiguous K for the reverse product
    dbuf = torch.empty((2, B, 2 * H), device=dev, dtype=recurrent.dtype)
    counter = torch.zeros(1, device=dev, dtype=torch.int32)
    gWx = torch.empty((B, T, H), device=dev, dtype=out_dtype)
    gWgx = torch.empty_like(gWx)
    day = torch.empty_like(gWx)
    cfg = _layout(B, H, dev, recurrent.dtype)
    meta = dict(BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], IP=_input_precision(recurrent.dtype),
                num_warps=cfg["W"], num_stages=cfg["S"])
    spart = torch.zeros((cfg["P"], 3), device=dev, dtype=torch.float32)
    if cfg["mode"] == "single":
        ppart = torch.empty((cfg["n_batch"], 16, H), device=dev, dtype=torch.float32)
        _core_bwd_single[(cfg["P"],)](
            U, e_tape, c_tape, s_tape, plasticity, expression, initial_b, initial_c,
            *gptrs, recurrent_t, packed, gb, ge, gc, dbuf, counter,
            gWx, gWgx, day, ppart, spart,
            B, T, H, cfg["P"], cfg["n_units"], *flags, **meta,
        )
        pacc = ppart.sum(0)
    else:
        gyp = torch.empty((B, H), device=dev, dtype=torch.float32)
        gbp = torch.empty_like(gyp)
        pacc = torch.zeros((16, H), device=dev, dtype=torch.float32)
        _core_bwd_general[(cfg["P"],)](
            U, e_tape, c_tape, s_tape, plasticity, expression, initial_b, initial_c,
            *gptrs, recurrent_t, packed, gb, ge, gc, gyp, gbp, dbuf, counter,
            gWx, gWgx, day, pacc, spart,
            B, T, H, cfg["P"], cfg["n_units"], cfg["n_batch"], *flags, **meta,
        )
    return gWx, gWgx, day, pacc, spart.sum(0), gb, ge, gc
