"""Triton/CUDA runtime for the canonical DABSN core and admitted read."""

from __future__ import annotations

import math
import os as _os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import triton
import triton.language as tl


def _chunk_autotune_configs():
    # Full tuning compiles and benchmarks 18 configurations for each live shape.
    # The default keeps first-call compilation bounded with one balanced config.
    # Set DABSN_AUTOTUNE_FULL=1 for exhaustive tuning on a target GPU, or override
    # the default block, warp, and stage values through the environment.
    if _os.environ.get("DABSN_AUTOTUNE_FULL", "0") == "1":
        cfgs = []
        for block_k in (32, 64, 128):
            for num_warps in (4, 8, 16):
                for num_stages in (2, 3):
                    cfgs.append(triton.Config({"BLOCK_K": block_k},
                                              num_warps=num_warps, num_stages=num_stages))
        return cfgs
    return [triton.Config(
        {"BLOCK_K": int(_os.environ.get("DABSN_AUTOTUNE_BLOCK_K", 64))},
        num_warps=int(_os.environ.get("DABSN_AUTOTUNE_WARPS", 8)),
        num_stages=int(_os.environ.get("DABSN_AUTOTUNE_STAGES", 2)))]


_FWD_CHUNK_AUTOTUNE = _chunk_autotune_configs()
_BWD_CHUNK_AUTOTUNE = _chunk_autotune_configs()
_BWD_RESET_ZERO = ["gWx", "gWgx", "DAy", "DUg", "gparam_p", "gscal_p"]


@triton.jit
def _tl_tanh(x):
    # Portable tanh: some Triton builds drop tl.tanh. tanh(x) == 2*sigmoid(2x)-1.
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _tl_abs(x):
    # Portable abs: avoid relying on tl.abs across Triton versions.
    return tl.where(x >= 0.0, x, -x)


SUPPORTED_DTYPES = {torch.float32, torch.bfloat16, torch.float16}


def _check_supported_dtype(dtype: torch.dtype, label: str) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"{label} dtype must be one of fp32/bf16/fp16, got {dtype}")


def _reference_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in {torch.bfloat16, torch.float16} else dtype


def _softplus_scalar(x: Tensor) -> Tensor:
    return F.softplus(x.reshape(()))


def _dabsn_core_scan_reference(
    Wx: Tensor,
    Wgx: Tensor,
    Ug: Tensor,
    A: Tensor,
    beta: Tensor,
    log_kappa: Tensor,
    logit_recover: Tensor,
    k_s: Tensor,
    k_y: Tensor,
    k_b: Tensor,
    k_n: Tensor,
    k_bias: Tensor,
    r_s: Tensor,
    r_y: Tensor,
    r_b: Tensor,
    r_n: Tensor,
    r_bias: Tensor,
    logit_c_decay: Tensor,
    k_c: Tensor,
    r_c: Tensor,
    logit_alpha: Tensor,
    log_lambda: Tensor,
    logit_c_suppress: Tensor,
    *,
    return_tape: bool = False,
) -> Tuple[Tensor, ...]:
    """Differentiable PyTorch reference matching the Triton scan boundary."""
    out_dtype = Wx.dtype
    compute_dtype = _reference_compute_dtype(out_dtype)
    Wx = Wx.to(compute_dtype)
    Wgx = Wgx.to(compute_dtype)
    Ug = Ug.to(compute_dtype)
    A = A.to(compute_dtype)
    beta = beta.to(compute_dtype)
    log_kappa = log_kappa.to(compute_dtype)
    logit_recover = logit_recover.to(compute_dtype)
    k_s = k_s.to(compute_dtype)
    k_y = k_y.to(compute_dtype)
    k_b = k_b.to(compute_dtype)
    k_n = k_n.to(compute_dtype)
    k_bias = k_bias.to(compute_dtype)
    r_s = r_s.to(compute_dtype)
    r_y = r_y.to(compute_dtype)
    r_b = r_b.to(compute_dtype)
    r_n = r_n.to(compute_dtype)
    r_bias = r_bias.to(compute_dtype)
    logit_c_decay = logit_c_decay.to(compute_dtype)
    k_c = k_c.to(compute_dtype)
    r_c = r_c.to(compute_dtype)
    logit_alpha = logit_alpha.to(compute_dtype)
    log_lambda = log_lambda.to(compute_dtype)
    logit_c_suppress = logit_c_suppress.to(compute_dtype)

    B, T, H = Wx.shape
    gate_alpha = torch.sigmoid(logit_alpha.reshape(()))
    lam = _softplus_scalar(log_lambda)
    kappa = F.softplus(log_kappa)
    recover = torch.sigmoid(logit_recover)
    c_decay = torch.sigmoid(logit_c_decay)
    c_suppress = torch.sigmoid(logit_c_suppress.reshape(()))

    b = torch.zeros(B, H, dtype=Wx.dtype, device=Wx.device)
    e = torch.ones_like(b)
    c = torch.zeros_like(b)
    outs: list[Tensor] = []
    nov_outs: list[Tensor] = []
    p_outs: list[Tensor] = []
    ay_outs: list[Tensor] = []
    write_outs: list[Tensor] = []
    e_outs: list[Tensor] = []
    c_outs: list[Tensor] = []
    s_outs: list[Tensor] = []

    GA_w = torch.cat([Ug, A], dim=0)
    for t in range(T):
        y = torch.tanh(Wx[:, t, :] + b)
        ug, ay = F.linear(y, GA_w).chunk(2, dim=1)
        s = F.hardsigmoid(Wgx[:, t, :] + ug)
        novelty = torch.tanh((ay - b).abs())
        stress = novelty * (1.0 - e)
        c = c_decay * c + (1.0 - c_decay) * stress
        novelty_eff = novelty * (1.0 - c_suppress * c)

        energy_nov = novelty_eff
        k_signal = k_s * s + k_y * y + k_b * torch.tanh(b) + k_n * energy_nov + k_bias
        r_signal = r_s * s + r_y * y + r_b * torch.tanh(b) + r_n * energy_nov + r_bias
        k_signal = k_signal + k_c * c
        r_signal = r_signal + r_c * c

        k_t = kappa * torch.exp(0.5 * torch.tanh(k_signal))
        r_t = torch.clamp(recover * torch.exp(0.5 * torch.tanh(r_signal)), 0.0, 1.0)
        p = s * e
        if return_tape:
            e_outs.append(e)
            c_outs.append(c)
            s_outs.append(s)

        b = (1.0 - gate_alpha) * b + beta + lam * (p * ay)
        e = torch.clamp(e + r_t * (1.0 - e) - k_t * p, 0.0, 1.0)

        outs.append(torch.cat([y, b], dim=-1))
        nov_outs.append(novelty)
        p_outs.append(p)
        ay_outs.append(ay)
        write_outs.append(p * ay)

    outs_tuple = (
        torch.stack(outs, dim=1),
        torch.stack(nov_outs, dim=1),
        torch.stack(p_outs, dim=1),
        torch.stack(ay_outs, dim=1),
        torch.stack(write_outs, dim=1),
    )
    if return_tape:
        outs_tuple = (
            *outs_tuple,
            torch.stack(e_outs, dim=1),
            torch.stack(c_outs, dim=1),
            torch.stack(s_outs, dim=1),
        )
    return tuple(t.to(out_dtype) for t in outs_tuple)


@triton.jit
def _dabsn_core_scan_fwd(
    Wx, Wgx, Ug, A,
    beta, log_kappa, logit_recover,
    k_s, k_y, k_b, k_n, k_bias,
    r_s, r_y, r_b, r_n, r_bias,
    logit_c_decay, k_c, r_c,
    initial_b, initial_e, initial_c,
    final_b, final_e, final_c,
    U, novelty_o, p_o, ay_o, write_o,
    e_o, c_o, s_o,
    logit_alpha,
    log_lambda,
    logit_c_suppress,
    B,
    T,
    H: tl.constexpr,
    STORE_TAPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    bidx = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    mask = h < H

    state_base = bidx * H
    b_state = tl.load(initial_b + state_base + h, mask=mask, other=0.0).to(tl.float32)
    e_state = tl.load(initial_e + state_base + h, mask=mask, other=1.0).to(tl.float32)
    c_state = tl.load(initial_c + state_base + h, mask=mask, other=0.0).to(tl.float32)

    gate_alpha = tl.sigmoid(tl.load(logit_alpha).to(tl.float32))
    lam = tl.log(1.0 + tl.exp(tl.load(log_lambda).to(tl.float32)))
    c_suppress = tl.sigmoid(tl.load(logit_c_suppress).to(tl.float32))

    beta_v = tl.load(beta + h, mask=mask, other=0.0).to(tl.float32)
    log_kappa_v = tl.load(log_kappa + h, mask=mask, other=-80.0).to(tl.float32)
    logit_recover_v = tl.load(logit_recover + h, mask=mask, other=-80.0).to(tl.float32)
    kappa = tl.log(1.0 + tl.exp(log_kappa_v))
    recover = tl.sigmoid(logit_recover_v)

    ks = tl.load(k_s + h, mask=mask, other=0.0).to(tl.float32)
    ky = tl.load(k_y + h, mask=mask, other=0.0).to(tl.float32)
    kb = tl.load(k_b + h, mask=mask, other=0.0).to(tl.float32)
    kn = tl.load(k_n + h, mask=mask, other=0.0).to(tl.float32)
    kbias = tl.load(k_bias + h, mask=mask, other=0.0).to(tl.float32)
    rs = tl.load(r_s + h, mask=mask, other=0.0).to(tl.float32)
    ry = tl.load(r_y + h, mask=mask, other=0.0).to(tl.float32)
    rb = tl.load(r_b + h, mask=mask, other=0.0).to(tl.float32)
    rn = tl.load(r_n + h, mask=mask, other=0.0).to(tl.float32)
    rbias = tl.load(r_bias + h, mask=mask, other=0.0).to(tl.float32)
    logit_c_decay_v = tl.load(logit_c_decay + h, mask=mask, other=-80.0).to(tl.float32)
    c_decay = tl.sigmoid(logit_c_decay_v)
    kc = tl.load(k_c + h, mask=mask, other=0.0).to(tl.float32)
    rc = tl.load(r_c + h, mask=mask, other=0.0).to(tl.float32)

    row = h[:, None]
    col = h[None, :]
    mat_mask = (row < H) & (col < H)
    ug_mat = tl.load(Ug + row * H + col, mask=mat_mask, other=0.0).to(tl.float32)
    a_mat = tl.load(A + row * H + col, mask=mat_mask, other=0.0).to(tl.float32)

    for t in range(0, T):
        base = (bidx * T + t) * H
        wx = tl.load(Wx + base + h, mask=mask, other=0.0).to(tl.float32)
        wgx = tl.load(Wgx + base + h, mask=mask, other=0.0).to(tl.float32)

        y = _tl_tanh(wx + b_state)
        ug = tl.sum(ug_mat * y[None, :], axis=1)
        ay = tl.sum(a_mat * y[None, :], axis=1)

        s = tl.minimum(1.0, tl.maximum(0.0, (wgx + ug) / 6.0 + 0.5))
        novelty = _tl_tanh(_tl_abs(ay - b_state))
        stress = novelty * (1.0 - e_state)
        c_state = c_decay * c_state + (1.0 - c_decay) * stress
        novelty_eff = novelty * (1.0 - c_suppress * c_state)

        energy_nov = novelty_eff

        k_signal = ks * s + ky * y + kb * _tl_tanh(b_state) + kn * energy_nov + kbias
        r_signal = rs * s + ry * y + rb * _tl_tanh(b_state) + rn * energy_nov + rbias
        k_signal += kc * c_state
        r_signal += rc * c_state

        k_t = kappa * tl.exp(0.5 * _tl_tanh(k_signal))
        r_t = tl.minimum(1.0, tl.maximum(0.0, recover * tl.exp(0.5 * _tl_tanh(r_signal))))
        p = s * e_state
        new_b = (1.0 - gate_alpha) * b_state + beta_v + lam * (p * ay)
        new_e = tl.minimum(1.0, tl.maximum(0.0, e_state + r_t * (1.0 - e_state) - k_t * p))

        u_base = (bidx * T + t) * (2 * H)
        tl.store(U + u_base + h, y, mask=mask)
        tl.store(U + u_base + H + h, new_b, mask=mask)
        tl.store(novelty_o + base + h, novelty, mask=mask)
        tl.store(p_o + base + h, p, mask=mask)
        tl.store(ay_o + base + h, ay, mask=mask)
        tl.store(write_o + base + h, p * ay, mask=mask)
        if STORE_TAPE:
            tl.store(e_o + base + h, e_state, mask=mask)
            tl.store(c_o + base + h, c_state, mask=mask)
            tl.store(s_o + base + h, s, mask=mask)

        b_state = new_b
        e_state = new_e

    tl.store(final_b + state_base + h, b_state, mask=mask)
    tl.store(final_e + state_base + h, e_state, mask=mask)
    tl.store(final_c + state_base + h, c_state, mask=mask)


@triton.autotune(configs=_FWD_CHUNK_AUTOTUNE, key=["H"])
@triton.jit
def _dabsn_core_scan_fwd_chunked(
    Wx, Wgx, Ug, A,
    beta, log_kappa, logit_recover,
    k_s, k_y, k_b, k_n, k_bias,
    r_s, r_y, r_b, r_n, r_bias,
    logit_c_decay, k_c, r_c,
    initial_b, initial_e, initial_c,
    final_b, final_e, final_c,
    U, novelty_o, p_o, ay_o, write_o,
    e_o, c_o, s_o,
    y_scratch,
    logit_alpha,
    log_lambda,
    logit_c_suppress,
    B,
    T,
    H: tl.constexpr,
    STORE_TAPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Identical math to `_dabsn_core_scan_fwd`, but `ay = A.y` and `ug = Ug.y` are
    # evaluated as a loop over BLOCK_K column chunks of H instead of holding the whole
    # (BLOCK_H, BLOCK_H) matrix tile resident for the entire scan. A/Ug are constant
    # across all T and B -> L2-resident, so per-chunk reloads are L2 hits. Peak live
    # registers drop from O(BLOCK_H^2) to O(BLOCK_H*BLOCK_K): no spill, no H cap.
    # Only `y` is needed column-wise (b/e/c are row-indexed only), so we stash y in a
    # per-row global scratch each step and read the column slices back, fenced by a
    # block barrier so cross-thread reads see the writes.
    bidx = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    mask = h < H

    state_base = bidx * H
    b_state = tl.load(initial_b + state_base + h, mask=mask, other=0.0).to(tl.float32)
    e_state = tl.load(initial_e + state_base + h, mask=mask, other=1.0).to(tl.float32)
    c_state = tl.load(initial_c + state_base + h, mask=mask, other=0.0).to(tl.float32)

    gate_alpha = tl.sigmoid(tl.load(logit_alpha).to(tl.float32))
    lam = tl.log(1.0 + tl.exp(tl.load(log_lambda).to(tl.float32)))
    c_suppress = tl.sigmoid(tl.load(logit_c_suppress).to(tl.float32))

    beta_v = tl.load(beta + h, mask=mask, other=0.0).to(tl.float32)
    log_kappa_v = tl.load(log_kappa + h, mask=mask, other=-80.0).to(tl.float32)
    logit_recover_v = tl.load(logit_recover + h, mask=mask, other=-80.0).to(tl.float32)
    kappa = tl.log(1.0 + tl.exp(log_kappa_v))
    recover = tl.sigmoid(logit_recover_v)

    ks = tl.load(k_s + h, mask=mask, other=0.0).to(tl.float32)
    ky = tl.load(k_y + h, mask=mask, other=0.0).to(tl.float32)
    kb = tl.load(k_b + h, mask=mask, other=0.0).to(tl.float32)
    kn = tl.load(k_n + h, mask=mask, other=0.0).to(tl.float32)
    kbias = tl.load(k_bias + h, mask=mask, other=0.0).to(tl.float32)
    rs = tl.load(r_s + h, mask=mask, other=0.0).to(tl.float32)
    ry = tl.load(r_y + h, mask=mask, other=0.0).to(tl.float32)
    rb = tl.load(r_b + h, mask=mask, other=0.0).to(tl.float32)
    rn = tl.load(r_n + h, mask=mask, other=0.0).to(tl.float32)
    rbias = tl.load(r_bias + h, mask=mask, other=0.0).to(tl.float32)
    logit_c_decay_v = tl.load(logit_c_decay + h, mask=mask, other=-80.0).to(tl.float32)
    c_decay = tl.sigmoid(logit_c_decay_v)
    kc = tl.load(k_c + h, mask=mask, other=0.0).to(tl.float32)
    rc = tl.load(r_c + h, mask=mask, other=0.0).to(tl.float32)

    y_row_base = bidx * H

    for t in range(0, T):
        base = (bidx * T + t) * H
        wx = tl.load(Wx + base + h, mask=mask, other=0.0).to(tl.float32)
        wgx = tl.load(Wgx + base + h, mask=mask, other=0.0).to(tl.float32)

        y = _tl_tanh(wx + b_state)

        # Stash y so the column-chunk matvec can read slices owned by other threads.
        # Barrier BEFORE the store guards the previous step's chunk reads from the
        # overwrite; barrier AFTER makes this step's y visible block-wide.
        tl.debug_barrier()
        tl.store(y_scratch + y_row_base + h, y, mask=mask)
        tl.debug_barrier()

        ay = tl.zeros((BLOCK_H,), tl.float32)
        ug = tl.zeros((BLOCK_H,), tl.float32)
        for k0 in range(0, H, BLOCK_K):
            kcol = k0 + tl.arange(0, BLOCK_K)
            kmask = kcol < H
            y_k = tl.load(y_scratch + y_row_base + kcol, mask=kmask, other=0.0)
            blk_mask = mask[:, None] & kmask[None, :]
            a_blk = tl.load(A + h[:, None] * H + kcol[None, :], mask=blk_mask, other=0.0).to(tl.float32)
            ug_blk = tl.load(Ug + h[:, None] * H + kcol[None, :], mask=blk_mask, other=0.0).to(tl.float32)
            ay += tl.sum(a_blk * y_k[None, :], axis=1)
            ug += tl.sum(ug_blk * y_k[None, :], axis=1)

        s = tl.minimum(1.0, tl.maximum(0.0, (wgx + ug) / 6.0 + 0.5))
        novelty = _tl_tanh(_tl_abs(ay - b_state))
        stress = novelty * (1.0 - e_state)
        c_state = c_decay * c_state + (1.0 - c_decay) * stress
        novelty_eff = novelty * (1.0 - c_suppress * c_state)

        energy_nov = novelty_eff

        k_signal = ks * s + ky * y + kb * _tl_tanh(b_state) + kn * energy_nov + kbias
        r_signal = rs * s + ry * y + rb * _tl_tanh(b_state) + rn * energy_nov + rbias
        k_signal += kc * c_state
        r_signal += rc * c_state

        k_t = kappa * tl.exp(0.5 * _tl_tanh(k_signal))
        r_t = tl.minimum(1.0, tl.maximum(0.0, recover * tl.exp(0.5 * _tl_tanh(r_signal))))
        p = s * e_state
        new_b = (1.0 - gate_alpha) * b_state + beta_v + lam * (p * ay)
        new_e = tl.minimum(1.0, tl.maximum(0.0, e_state + r_t * (1.0 - e_state) - k_t * p))

        u_base = (bidx * T + t) * (2 * H)
        tl.store(U + u_base + h, y, mask=mask)
        tl.store(U + u_base + H + h, new_b, mask=mask)
        tl.store(novelty_o + base + h, novelty, mask=mask)
        tl.store(p_o + base + h, p, mask=mask)
        tl.store(ay_o + base + h, ay, mask=mask)
        tl.store(write_o + base + h, p * ay, mask=mask)
        if STORE_TAPE:
            tl.store(e_o + base + h, e_state, mask=mask)
            tl.store(c_o + base + h, c_state, mask=mask)
            tl.store(s_o + base + h, s, mask=mask)

        b_state = new_b
        e_state = new_e

    tl.store(final_b + state_base + h, b_state, mask=mask)
    tl.store(final_e + state_base + h, e_state, mask=mask)
    tl.store(final_c + state_base + h, c_state, mask=mask)


# Widths up to this power-of-two tile use the one-tile forward kernel while the
# recurrent matrix fits its register budget. Larger widths use chunked matvec.
_ONE_TILE_MAX = 256


def _block_k(hidden_dim: int) -> int:
    """Column width for a chunked matvec tile that stays within register budget."""
    return 64


def _block_h(hidden_dim: int) -> int:
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    # Next power of two; chunked matvec handles widths beyond a single tile.
    return 1 << (hidden_dim - 1).bit_length()


def _launch_fwd_scan(
    Wx, Wgx, Ug, A,
    beta, log_kappa, logit_recover,
    k_s, k_y, k_b, k_n, k_bias,
    r_s, r_y, r_b, r_n, r_bias,
    logit_c_decay, k_c, r_c,
    initial_b, initial_e, initial_c,
    final_b, final_e, final_c,
    U, novelty, p, ay, write,
    e_tape, c_tape, s_tape,
    logit_alpha, log_lambda, logit_c_suppress,
    B, T, H, store_tape,
) -> None:
    """Pick the one-tile (small H) or chunked-matvec (large H) forward kernel."""
    block = _block_h(H)
    if block <= _ONE_TILE_MAX:
        _dabsn_core_scan_fwd[(B,)](
            Wx, Wgx, Ug, A,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            initial_b, initial_e, initial_c,
            final_b, final_e, final_c,
            U, novelty, p, ay, write,
            e_tape, c_tape, s_tape,
            logit_alpha, log_lambda, logit_c_suppress,
            B, T, H,
            store_tape,
            BLOCK_H=block,
            num_warps=_fused_num_warps(block),
        )
        return
    y_scratch = torch.empty((B, H), device=Wx.device, dtype=torch.float32)
    # BLOCK_K / num_warps / num_stages are chosen by @triton.autotune (keyed on H).
    _dabsn_core_scan_fwd_chunked[(B,)](
        Wx, Wgx, Ug, A,
        beta, log_kappa, logit_recover,
        k_s, k_y, k_b, k_n, k_bias,
        r_s, r_y, r_b, r_n, r_bias,
        logit_c_decay, k_c, r_c,
        initial_b, initial_e, initial_c,
        final_b, final_e, final_c,
        U, novelty, p, ay, write,
        e_tape, c_tape, s_tape,
        y_scratch,
        logit_alpha, log_lambda, logit_c_suppress,
        B, T, H,
        store_tape,
        BLOCK_H=block,
    )


def dabsn_core_scan_triton(
    Wx: Tensor,
    Wgx: Tensor,
    Ug: Tensor,
    A: Tensor,
    beta: Tensor,
    log_kappa: Tensor,
    logit_recover: Tensor,
    k_s: Tensor,
    k_y: Tensor,
    k_b: Tensor,
    k_n: Tensor,
    k_bias: Tensor,
    r_s: Tensor,
    r_y: Tensor,
    r_b: Tensor,
    r_n: Tensor,
    r_bias: Tensor,
    logit_c_decay: Tensor,
    k_c: Tensor,
    r_c: Tensor,
    *,
    logit_alpha: Tensor,
    log_lambda: Tensor,
    logit_c_suppress: Tensor,
    initial_state: tuple[Tensor, Tensor, Tensor] | None = None,
    return_final_state: bool = False,
) -> Tuple[Tensor, ...]:
    """Run fused Triton forward over pre-projected `Wx` and `Wgx`.

    Shapes:
      Wx/Wgx: B,T,H contiguous fp32/bf16/fp16 CUDA
      Ug/A: H,H contiguous fp32/bf16/fp16 CUDA
      vector params: H contiguous fp32/bf16/fp16 CUDA
    """
    if not Wx.is_cuda:
        raise TypeError("Triton kernel requires CUDA tensors")
    _check_supported_dtype(Wx.dtype, "Wx")
    if not (Wx.is_contiguous() and Wgx.is_contiguous() and Ug.is_contiguous() and A.is_contiguous()):
        raise ValueError("Wx, Wgx, Ug, and A must be contiguous")
    B, T, H = Wx.shape
    if Wgx.shape != Wx.shape:
        raise ValueError("Wgx shape must match Wx")
    if Ug.shape != (H, H) or A.shape != (H, H):
        raise ValueError("Ug and A must be H,H")
    if initial_state is None:
        initial_state = (
            torch.zeros((B, H), device=Wx.device, dtype=Wx.dtype),
            torch.ones((B, H), device=Wx.device, dtype=Wx.dtype),
            torch.zeros((B, H), device=Wx.device, dtype=Wx.dtype),
        )
    initial_b, initial_e, initial_c = (
        value.to(device=Wx.device, dtype=Wx.dtype).contiguous() for value in initial_state
    )
    if any(value.shape != (B, H) for value in (initial_b, initial_e, initial_c)):
        raise ValueError(f"initial_state tensors must have shape {(B, H)}")
    U = torch.empty((B, T, 2 * H), device=Wx.device, dtype=Wx.dtype)
    novelty = torch.empty_like(Wx)
    p = torch.empty_like(Wx)
    ay = torch.empty_like(Wx)
    write = torch.empty_like(Wx)
    final_b = torch.empty((B, H), device=Wx.device, dtype=Wx.dtype)
    final_e = torch.empty_like(final_b)
    final_c = torch.empty_like(final_b)
    dummy_tape = U
    _launch_fwd_scan(
        Wx, Wgx, Ug, A,
        beta, log_kappa, logit_recover,
        k_s, k_y, k_b, k_n, k_bias,
        r_s, r_y, r_b, r_n, r_bias,
        logit_c_decay, k_c, r_c,
        initial_b, initial_e, initial_c,
        final_b, final_e, final_c,
        U, novelty, p, ay, write,
        dummy_tape, dummy_tape, dummy_tape,
        logit_alpha, log_lambda, logit_c_suppress,
        B, T, H,
        False,
    )
    outputs = (U, novelty, p, ay, write)
    if return_final_state:
        return (*outputs, final_b, final_e, final_c)
    return outputs


def dabsn_core_scan_triton_tape(
    Wx: Tensor,
    Wgx: Tensor,
    Ug: Tensor,
    A: Tensor,
    beta: Tensor,
    log_kappa: Tensor,
    logit_recover: Tensor,
    k_s: Tensor,
    k_y: Tensor,
    k_b: Tensor,
    k_n: Tensor,
    k_bias: Tensor,
    r_s: Tensor,
    r_y: Tensor,
    r_b: Tensor,
    r_n: Tensor,
    r_bias: Tensor,
    logit_c_decay: Tensor,
    k_c: Tensor,
    r_c: Tensor,
    *,
    logit_alpha: Tensor,
    log_lambda: Tensor,
    logit_c_suppress: Tensor,
    tape_dtype: torch.dtype | None = None,
    initial_state: tuple[Tensor, Tensor, Tensor] | None = None,
    return_final_state: bool = False,
) -> Tuple[Tensor, ...]:
    """Forward scan plus state tape for recurrent-core backward.

    Low-precision training keeps the public output dtype, but the custom
    backward can request fp32 tape storage so wide-H gradients use the same
    fp32 intermediates as the differentiable reference.
    """
    if not Wx.is_cuda:
        raise TypeError("Triton kernel requires CUDA tensors")
    _check_supported_dtype(Wx.dtype, "Wx")
    if not (Wx.is_contiguous() and Wgx.is_contiguous() and Ug.is_contiguous() and A.is_contiguous()):
        raise ValueError("Wx, Wgx, Ug, and A must be contiguous")
    B, T, H = Wx.shape
    if Wgx.shape != Wx.shape:
        raise ValueError("Wgx shape must match Wx")
    if Ug.shape != (H, H) or A.shape != (H, H):
        raise ValueError("Ug and A must be H,H")
    store_dtype = Wx.dtype if tape_dtype is None else tape_dtype
    _check_supported_dtype(store_dtype, "tape_dtype")
    if initial_state is None:
        initial_state = (
            torch.zeros((B, H), device=Wx.device, dtype=Wx.dtype),
            torch.ones((B, H), device=Wx.device, dtype=Wx.dtype),
            torch.zeros((B, H), device=Wx.device, dtype=Wx.dtype),
        )
    initial_b, initial_e, initial_c = (
        value.to(device=Wx.device, dtype=Wx.dtype).contiguous() for value in initial_state
    )
    if any(value.shape != (B, H) for value in (initial_b, initial_e, initial_c)):
        raise ValueError(f"initial_state tensors must have shape {(B, H)}")
    U = torch.empty((B, T, 2 * H), device=Wx.device, dtype=store_dtype)
    novelty = torch.empty((B, T, H), device=Wx.device, dtype=store_dtype)
    p = torch.empty((B, T, H), device=Wx.device, dtype=store_dtype)
    ay = torch.empty((B, T, H), device=Wx.device, dtype=store_dtype)
    write = torch.empty((B, T, H), device=Wx.device, dtype=store_dtype)
    e_tape = torch.empty((B, T, H), device=Wx.device, dtype=store_dtype)
    c_tape = torch.empty((B, T, H), device=Wx.device, dtype=store_dtype)
    s_tape = torch.empty((B, T, H), device=Wx.device, dtype=store_dtype)
    final_b = torch.empty((B, H), device=Wx.device, dtype=store_dtype)
    final_e = torch.empty_like(final_b)
    final_c = torch.empty_like(final_b)

    _launch_fwd_scan(
        Wx, Wgx, Ug, A,
        beta, log_kappa, logit_recover,
        k_s, k_y, k_b, k_n, k_bias,
        r_s, r_y, r_b, r_n, r_bias,
        logit_c_decay, k_c, r_c,
        initial_b, initial_e, initial_c,
        final_b, final_e, final_c,
        U, novelty, p, ay, write,
        e_tape, c_tape, s_tape,
        logit_alpha, log_lambda, logit_c_suppress,
        B, T, H,
        True,
    )
    outputs = (U, novelty, p, ay, write, e_tape, c_tape, s_tape)
    if return_final_state:
        return (*outputs, final_b, final_e, final_c)
    return outputs


_FUSED_NUM_PARAM_VECS = 16  # beta, log_kappa, logit_recover, k_*, r_*, logit_c_decay, k_c, r_c


def _fused_block_h(hidden_dim: int) -> int:
    """Tile width for the fused reverse-scan backward (split-H ceiling 1024)."""
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    block = 1 << (hidden_dim - 1).bit_length()
    if block > 1024:
        raise ValueError("fused backward supports hidden_dim <= 1024")
    return block


def _fused_num_warps(block: int) -> int:
    if block <= 64:
        return 4
    if block <= 256:
        return 8
    return 16


@triton.jit
def _dabsn_core_scan_bwd(
    U, Novelty, P, Ay, Etape, Ctape, Stape,
    InitialB, InitialC,
    A, Ug,
    beta, log_kappa, logit_recover,
    k_s, k_y, k_b, k_n, k_bias,
    r_s, r_y, r_b, r_n, r_bias,
    logit_c_decay, k_c, r_c,
    gU, gNov, gP, gAy, gWrite, gEOut, gCOut, gFinalB, gFinalE, gFinalC,
    gWx, gWgx, DAy, DUg,
    gInitialB, gInitialE, gInitialC,
    gparam_p, gscal_p,
    logit_alpha, log_lambda, logit_c_suppress,
    T,
    H: tl.constexpr,
    NP: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Single fused reverse-scan backward for the dabsn_core recurrent core.

    One program per batch row walks time in reverse, carrying grad_b / grad_e /
    grad_c in fp32 registers and composing the exact algebra of the tested
    backward slices inline. Matrix grads grad_A / grad_Ug are emitted as the
    per-step adjoint tapes `DAy` (combined d_ay) and `DUg` (d_ug = dpre); the host
    turns them into GEMMs. Per-dim / scalar param grads accumulate in registers
    and are written once to per-program partial buffers.
    """
    bidx = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    mask = h < H

    # ---- constant per-dim params (loaded once) ----
    beta_v = tl.load(beta + h, mask=mask, other=0.0).to(tl.float32)
    lk = tl.load(log_kappa + h, mask=mask, other=-80.0).to(tl.float32)
    lr = tl.load(logit_recover + h, mask=mask, other=-80.0).to(tl.float32)
    kappa = tl.log(1.0 + tl.exp(lk))
    recover = tl.sigmoid(lr)
    ks = tl.load(k_s + h, mask=mask, other=0.0).to(tl.float32)
    ky = tl.load(k_y + h, mask=mask, other=0.0).to(tl.float32)
    kb = tl.load(k_b + h, mask=mask, other=0.0).to(tl.float32)
    kn = tl.load(k_n + h, mask=mask, other=0.0).to(tl.float32)
    kbias = tl.load(k_bias + h, mask=mask, other=0.0).to(tl.float32)
    rs = tl.load(r_s + h, mask=mask, other=0.0).to(tl.float32)
    ry = tl.load(r_y + h, mask=mask, other=0.0).to(tl.float32)
    rb = tl.load(r_b + h, mask=mask, other=0.0).to(tl.float32)
    rn = tl.load(r_n + h, mask=mask, other=0.0).to(tl.float32)
    rbias = tl.load(r_bias + h, mask=mask, other=0.0).to(tl.float32)
    cdecay_logit = tl.load(logit_c_decay + h, mask=mask, other=-80.0).to(tl.float32)
    decay = tl.sigmoid(cdecay_logit)
    kc = tl.load(k_c + h, mask=mask, other=0.0).to(tl.float32)
    rc = tl.load(r_c + h, mask=mask, other=0.0).to(tl.float32)

    logit_alpha_v = tl.load(logit_alpha).to(tl.float32)
    log_lambda_v = tl.load(log_lambda).to(tl.float32)
    logit_c_suppress_v = tl.load(logit_c_suppress).to(tl.float32)
    alpha = tl.sigmoid(logit_alpha_v)
    lam = tl.log(1.0 + tl.exp(log_lambda_v))
    sig_lam = tl.sigmoid(log_lambda_v)
    suppress = tl.sigmoid(logit_c_suppress_v)

    # ---- persistent matrix tiles for the adjoint mat-vecs ----
    row = h[:, None]
    col = h[None, :]
    mat_mask = (row < H) & (col < H)
    a_mat = tl.load(A + row * H + col, mask=mat_mask, other=0.0).to(tl.float32)
    ug_mat = tl.load(Ug + row * H + col, mask=mat_mask, other=0.0).to(tl.float32)

    # ---- loop-carried fp32 accumulators ----
    state_base = bidx * H
    gbn = tl.load(gFinalB + state_base + h, mask=mask, other=0.0).to(tl.float32)
    gen = tl.load(gFinalE + state_base + h, mask=mask, other=0.0).to(tl.float32)
    gcn = tl.load(gFinalC + state_base + h, mask=mask, other=0.0).to(tl.float32)
    a_beta = tl.zeros((BLOCK_H,), tl.float32)
    a_logk = tl.zeros((BLOCK_H,), tl.float32)
    a_logr = tl.zeros((BLOCK_H,), tl.float32)
    a_ks = tl.zeros((BLOCK_H,), tl.float32)
    a_ky = tl.zeros((BLOCK_H,), tl.float32)
    a_kb = tl.zeros((BLOCK_H,), tl.float32)
    a_kn = tl.zeros((BLOCK_H,), tl.float32)
    a_kbias = tl.zeros((BLOCK_H,), tl.float32)
    a_rs = tl.zeros((BLOCK_H,), tl.float32)
    a_ry = tl.zeros((BLOCK_H,), tl.float32)
    a_rb = tl.zeros((BLOCK_H,), tl.float32)
    a_rn = tl.zeros((BLOCK_H,), tl.float32)
    a_rbias = tl.zeros((BLOCK_H,), tl.float32)
    a_cdecay = tl.zeros((BLOCK_H,), tl.float32)
    a_kc = tl.zeros((BLOCK_H,), tl.float32)
    a_rc = tl.zeros((BLOCK_H,), tl.float32)
    a_alpha = tl.zeros((), tl.float32)
    a_lambda = tl.zeros((), tl.float32)
    a_csup = tl.zeros((), tl.float32)

    for i in range(0, T):
        t = T - 1 - i
        base = (bidx * T + t) * H
        ubase = (bidx * T + t) * (2 * H)

        y = tl.load(U + ubase + h, mask=mask, other=0.0).to(tl.float32)
        tprev = tl.maximum(t - 1, 0)
        ub_prev = (bidx * T + tprev) * (2 * H)
        base_prev = (bidx * T + tprev) * H
        first_f = tl.where(t == 0, 1.0, 0.0)  # scalar 1.0 at t==0 else 0.0
        prev_f = 1.0 - first_f
        b_prev = (
            prev_f * tl.load(U + ub_prev + H + h, mask=mask, other=0.0).to(tl.float32)
            + first_f * tl.load(InitialB + state_base + h, mask=mask, other=0.0).to(tl.float32)
        )
        e_prev = tl.load(Etape + base + h, mask=mask, other=1.0).to(tl.float32)
        c_prev = (
            prev_f * tl.load(Ctape + base_prev + h, mask=mask, other=0.0).to(tl.float32)
            + first_f * tl.load(InitialC + state_base + h, mask=mask, other=0.0).to(tl.float32)
        )

        s = tl.load(Stape + base + h, mask=mask, other=0.0).to(tl.float32)
        nov = tl.load(Novelty + base + h, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(P + base + h, mask=mask, other=0.0).to(tl.float32)
        ay = tl.load(Ay + base + h, mask=mask, other=0.0).to(tl.float32)
        c_t = tl.load(Ctape + base + h, mask=mask, other=0.0).to(tl.float32)

        gy0 = tl.load(gU + ubase + h, mask=mask, other=0.0).to(tl.float32)
        gb_new = tl.load(gU + ubase + H + h, mask=mask, other=0.0).to(tl.float32) + gbn
        ge_out = tl.load(gEOut + base + h, mask=mask, other=0.0).to(tl.float32)
        gc_out = tl.load(gCOut + base + h, mask=mask, other=0.0).to(tl.float32)
        ge_new = gen
        gc_new = gcn + gc_out
        gnov0 = tl.load(gNov + base + h, mask=mask, other=0.0).to(tl.float32)
        gp0 = tl.load(gP + base + h, mask=mask, other=0.0).to(tl.float32)
        gay0 = tl.load(gAy + base + h, mask=mask, other=0.0).to(tl.float32)
        gwrite = tl.load(gWrite + base + h, mask=mask, other=0.0).to(tl.float32)

        # ---- recompute forward coefficients from the tape ----
        tanh_b = _tl_tanh(b_prev)
        novelty_eff = nov * (1.0 - suppress * c_t)
        energy_nov = novelty_eff
        k_signal = ks * s + ky * y + kb * tanh_b + kbias
        r_signal = rs * s + ry * y + rb * tanh_b + rbias
        k_signal += kn * energy_nov
        r_signal += rn * energy_nov
        k_signal += kc * c_t
        r_signal += rc * c_t
        tanh_k = _tl_tanh(k_signal)
        tanh_r = _tl_tanh(r_signal)
        exp_k = tl.exp(0.5 * tanh_k)
        exp_r = tl.exp(0.5 * tanh_r)
        k_t = kappa * exp_k
        r_pre = recover * exp_r
        r_t = tl.minimum(1.0, tl.maximum(0.0, r_pre))
        r_clamp_grad = tl.where((r_pre >= 0.0) & (r_pre <= 1.0), 1.0, 0.0)

        # ---- slice 1: b/e state update ----
        e_pre = e_prev + r_t * (1.0 - e_prev) - k_t * p
        clamp_grad = tl.where((e_pre >= 0.0) & (e_pre <= 1.0), 1.0, 0.0)
        ge_pre = ge_new * clamp_grad
        g_bprev_state = gb_new * (1.0 - alpha)
        g_eprev_state = ge_pre * (1.0 - r_t)
        gp = gp0 + gb_new * lam * ay + ge_pre * (-k_t)
        gay = gay0 + gb_new * lam * p
        gkt = ge_pre * (-p)
        grt = ge_pre * (1.0 - e_prev)
        a_beta += gb_new
        a_alpha += tl.sum(tl.where(mask, gb_new * (-b_prev) * alpha * (1.0 - alpha), 0.0), axis=0)
        a_lambda += tl.sum(tl.where(mask, gb_new * (p * ay) * sig_lam, 0.0), axis=0)

        # ---- slice 2: ay = A@y, write = p*ay (main path) ----
        d_ay_main = gay + gwrite * p
        gp += gwrite * ay  # grad_p from write = grad_write * ay

        # ---- slice 3: energy / recovery coefficients ----
        g_k_signal = gkt * k_t * 0.5 * (1.0 - tanh_k * tanh_k)
        g_r_pre = grt * r_clamp_grad
        g_r_signal = g_r_pre * r_pre * 0.5 * (1.0 - tanh_r * tanh_r)
        gs_energy = g_k_signal * ks + g_r_signal * rs
        gy_energy = g_k_signal * ky + g_r_signal * ry
        gb_energy = (g_k_signal * kb + g_r_signal * rb) * (1.0 - tanh_b * tanh_b)
        gnov_eff = g_k_signal * kn + g_r_signal * rn
        gc_energy = g_k_signal * kc + g_r_signal * rc
        a_logk += gkt * exp_k * tl.sigmoid(lk)
        a_logr += g_r_pre * exp_r * recover * (1.0 - recover)
        a_ks += g_k_signal * s
        a_ky += g_k_signal * y
        a_kb += g_k_signal * tanh_b
        a_kbias += g_k_signal
        a_rs += g_r_signal * s
        a_ry += g_r_signal * y
        a_rb += g_r_signal * tanh_b
        a_rbias += g_r_signal
        a_kn += g_k_signal * energy_nov
        a_rn += g_r_signal * energy_nov
        a_kc += g_k_signal * c_t
        a_rc += g_r_signal * c_t
        gy_accum = gy0 + gy_energy

        # ---- slice 4: gate s = hardsigmoid(Wgx + Ug@y); p = s*e_prev ----
        ds_dpre = tl.where((s > 0.0) & (s < 1.0), 1.0 / 6.0, 0.0)
        dpre = (gp * e_prev + gs_energy) * ds_dpre
        gy_accum += tl.sum(ug_mat * dpre[:, None], axis=0)  # Ug^T @ dpre
        ge_gate = gp * s
        tl.store(gWgx + base + h, dpre, mask=mask)
        tl.store(DUg + base + h, dpre, mask=mask)

        # ---- slice 5: saturation state + novelty suppression ----
        g_c_new_outer = gc_new + gc_energy
        stress = nov * (1.0 - e_prev)
        c_new = decay * c_prev + (1.0 - decay) * stress
        g_c_new = g_c_new_outer + gnov_eff * nov * (-suppress)
        g_stress = g_c_new * (1.0 - decay)
        g_novelty = gnov_eff * (1.0 - suppress * c_new) + g_stress * (1.0 - e_prev)
        g_e_prev_cort = g_stress * (-nov)
        g_c_prev = g_c_new * decay
        g_decay = g_c_new * (c_prev - stress)
        a_cdecay += g_decay * decay * (1.0 - decay)
        a_csup += tl.sum(tl.where(mask, gnov_eff * nov * (-c_new) * suppress * (1.0 - suppress), 0.0), axis=0)
        gnov_total = gnov0 + g_novelty

        # ---- slice 6: novelty = tanh(|ay - b_prev|) ----
        diff = ay - b_prev
        nov_chk = _tl_tanh(_tl_abs(diff))
        sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
        d_nov = gnov_total * (1.0 - nov_chk * nov_chk) * sign
        gb_nov = -d_nov

        # ---- slice 7: novelty-path ay = A@y, combine adjoint d_ay ----
        d_ay_total = d_ay_main + d_nov
        gy_accum += tl.sum(a_mat * d_ay_total[:, None], axis=0)  # A^T @ d_ay
        tl.store(DAy + base + h, d_ay_total, mask=mask)

        # ---- slice 8: y = tanh(Wx + b_prev) ----
        gpre = gy_accum * (1.0 - y * y)
        tl.store(gWx + base + h, gpre, mask=mask)
        gb_y = gpre

        # ---- carry to t-1 ----
        gbn = g_bprev_state + gb_energy + gb_nov + gb_y
        gen = g_eprev_state + ge_gate + g_e_prev_cort + ge_out
        gcn = g_c_prev

    # ---- flush per-program partials ----
    tl.store(gInitialB + state_base + h, gbn, mask=mask)
    tl.store(gInitialE + state_base + h, gen, mask=mask)
    tl.store(gInitialC + state_base + h, gcn, mask=mask)
    pp = bidx * NP * H
    tl.store(gparam_p + pp + 0 * H + h, a_beta, mask=mask)
    tl.store(gparam_p + pp + 1 * H + h, a_logk, mask=mask)
    tl.store(gparam_p + pp + 2 * H + h, a_logr, mask=mask)
    tl.store(gparam_p + pp + 3 * H + h, a_ks, mask=mask)
    tl.store(gparam_p + pp + 4 * H + h, a_ky, mask=mask)
    tl.store(gparam_p + pp + 5 * H + h, a_kb, mask=mask)
    tl.store(gparam_p + pp + 6 * H + h, a_kn, mask=mask)
    tl.store(gparam_p + pp + 7 * H + h, a_kbias, mask=mask)
    tl.store(gparam_p + pp + 8 * H + h, a_rs, mask=mask)
    tl.store(gparam_p + pp + 9 * H + h, a_ry, mask=mask)
    tl.store(gparam_p + pp + 10 * H + h, a_rb, mask=mask)
    tl.store(gparam_p + pp + 11 * H + h, a_rn, mask=mask)
    tl.store(gparam_p + pp + 12 * H + h, a_rbias, mask=mask)
    tl.store(gparam_p + pp + 13 * H + h, a_cdecay, mask=mask)
    tl.store(gparam_p + pp + 14 * H + h, a_kc, mask=mask)
    tl.store(gparam_p + pp + 15 * H + h, a_rc, mask=mask)
    tl.store(gscal_p + bidx * 3 + 0, a_alpha)
    tl.store(gscal_p + bidx * 3 + 1, a_lambda)
    tl.store(gscal_p + bidx * 3 + 2, a_csup)


@triton.autotune(configs=_BWD_CHUNK_AUTOTUNE, key=["H"], reset_to_zero=_BWD_RESET_ZERO)
@triton.jit
def _dabsn_core_scan_bwd_chunked(
    U, Novelty, P, Ay, Etape, Ctape, Stape,
    InitialB, InitialC,
    A, Ug,
    beta, log_kappa, logit_recover,
    k_s, k_y, k_b, k_n, k_bias,
    r_s, r_y, r_b, r_n, r_bias,
    logit_c_decay, k_c, r_c,
    gU, gNov, gP, gAy, gWrite, gEOut, gCOut, gFinalB, gFinalE, gFinalC,
    gWx, gWgx, DAy, DUg,
    gInitialB, gInitialE, gInitialC,
    gparam_p, gscal_p,
    dpre_scratch, day_scratch,
    logit_alpha, log_lambda, logit_c_suppress,
    T,
    H: tl.constexpr,
    NP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Identical math to `_dabsn_core_scan_bwd`, with the two transpose adjoint
    mat-vecs (`Ug^T @ dpre` and `A^T @ d_ay_total`, both reducing over the row index)
    are evaluated as a loop over BLOCK_K row chunks of H instead of holding the whole
    (BLOCK_H, BLOCK_H) matrix tile resident for the whole reverse scan. They are merged
    into one chunk loop (dpre is finalized early, the add is deferred until d_ay_total
    exists). dpre and d_ay_total are the only row-sliced vectors -> stashed in per-row
    global scratch, fenced by a block barrier. Peak live registers are
    O(BLOCK_H*BLOCK_K). DAy/DUg tapes for grad_A/grad_Ug are unchanged.
    """
    bidx = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    mask = h < H

    beta_v = tl.load(beta + h, mask=mask, other=0.0).to(tl.float32)
    lk = tl.load(log_kappa + h, mask=mask, other=-80.0).to(tl.float32)
    lr = tl.load(logit_recover + h, mask=mask, other=-80.0).to(tl.float32)
    kappa = tl.log(1.0 + tl.exp(lk))
    recover = tl.sigmoid(lr)
    ks = tl.load(k_s + h, mask=mask, other=0.0).to(tl.float32)
    ky = tl.load(k_y + h, mask=mask, other=0.0).to(tl.float32)
    kb = tl.load(k_b + h, mask=mask, other=0.0).to(tl.float32)
    kn = tl.load(k_n + h, mask=mask, other=0.0).to(tl.float32)
    kbias = tl.load(k_bias + h, mask=mask, other=0.0).to(tl.float32)
    rs = tl.load(r_s + h, mask=mask, other=0.0).to(tl.float32)
    ry = tl.load(r_y + h, mask=mask, other=0.0).to(tl.float32)
    rb = tl.load(r_b + h, mask=mask, other=0.0).to(tl.float32)
    rn = tl.load(r_n + h, mask=mask, other=0.0).to(tl.float32)
    rbias = tl.load(r_bias + h, mask=mask, other=0.0).to(tl.float32)
    cdecay_logit = tl.load(logit_c_decay + h, mask=mask, other=-80.0).to(tl.float32)
    decay = tl.sigmoid(cdecay_logit)
    kc = tl.load(k_c + h, mask=mask, other=0.0).to(tl.float32)
    rc = tl.load(r_c + h, mask=mask, other=0.0).to(tl.float32)

    logit_alpha_v = tl.load(logit_alpha).to(tl.float32)
    log_lambda_v = tl.load(log_lambda).to(tl.float32)
    logit_c_suppress_v = tl.load(logit_c_suppress).to(tl.float32)
    alpha = tl.sigmoid(logit_alpha_v)
    lam = tl.log(1.0 + tl.exp(log_lambda_v))
    sig_lam = tl.sigmoid(log_lambda_v)
    suppress = tl.sigmoid(logit_c_suppress_v)

    state_base = bidx * H
    gbn = tl.load(gFinalB + state_base + h, mask=mask, other=0.0).to(tl.float32)
    gen = tl.load(gFinalE + state_base + h, mask=mask, other=0.0).to(tl.float32)
    gcn = tl.load(gFinalC + state_base + h, mask=mask, other=0.0).to(tl.float32)
    a_beta = tl.zeros((BLOCK_H,), tl.float32)
    a_logk = tl.zeros((BLOCK_H,), tl.float32)
    a_logr = tl.zeros((BLOCK_H,), tl.float32)
    a_ks = tl.zeros((BLOCK_H,), tl.float32)
    a_ky = tl.zeros((BLOCK_H,), tl.float32)
    a_kb = tl.zeros((BLOCK_H,), tl.float32)
    a_kn = tl.zeros((BLOCK_H,), tl.float32)
    a_kbias = tl.zeros((BLOCK_H,), tl.float32)
    a_rs = tl.zeros((BLOCK_H,), tl.float32)
    a_ry = tl.zeros((BLOCK_H,), tl.float32)
    a_rb = tl.zeros((BLOCK_H,), tl.float32)
    a_rn = tl.zeros((BLOCK_H,), tl.float32)
    a_rbias = tl.zeros((BLOCK_H,), tl.float32)
    a_cdecay = tl.zeros((BLOCK_H,), tl.float32)
    a_kc = tl.zeros((BLOCK_H,), tl.float32)
    a_rc = tl.zeros((BLOCK_H,), tl.float32)
    a_alpha = tl.zeros((), tl.float32)
    a_lambda = tl.zeros((), tl.float32)
    a_csup = tl.zeros((), tl.float32)

    for i in range(0, T):
        t = T - 1 - i
        base = (bidx * T + t) * H
        ubase = (bidx * T + t) * (2 * H)

        y = tl.load(U + ubase + h, mask=mask, other=0.0).to(tl.float32)
        tprev = tl.maximum(t - 1, 0)
        ub_prev = (bidx * T + tprev) * (2 * H)
        base_prev = (bidx * T + tprev) * H
        first_f = tl.where(t == 0, 1.0, 0.0)
        prev_f = 1.0 - first_f
        b_prev = (
            prev_f * tl.load(U + ub_prev + H + h, mask=mask, other=0.0).to(tl.float32)
            + first_f * tl.load(InitialB + state_base + h, mask=mask, other=0.0).to(tl.float32)
        )
        e_prev = tl.load(Etape + base + h, mask=mask, other=1.0).to(tl.float32)
        c_prev = (
            prev_f * tl.load(Ctape + base_prev + h, mask=mask, other=0.0).to(tl.float32)
            + first_f * tl.load(InitialC + state_base + h, mask=mask, other=0.0).to(tl.float32)
        )

        s = tl.load(Stape + base + h, mask=mask, other=0.0).to(tl.float32)
        nov = tl.load(Novelty + base + h, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(P + base + h, mask=mask, other=0.0).to(tl.float32)
        ay = tl.load(Ay + base + h, mask=mask, other=0.0).to(tl.float32)
        c_t = tl.load(Ctape + base + h, mask=mask, other=0.0).to(tl.float32)

        gy0 = tl.load(gU + ubase + h, mask=mask, other=0.0).to(tl.float32)
        gb_new = tl.load(gU + ubase + H + h, mask=mask, other=0.0).to(tl.float32) + gbn
        ge_out = tl.load(gEOut + base + h, mask=mask, other=0.0).to(tl.float32)
        gc_out = tl.load(gCOut + base + h, mask=mask, other=0.0).to(tl.float32)
        ge_new = gen
        gc_new = gcn + gc_out
        gnov0 = tl.load(gNov + base + h, mask=mask, other=0.0).to(tl.float32)
        gp0 = tl.load(gP + base + h, mask=mask, other=0.0).to(tl.float32)
        gay0 = tl.load(gAy + base + h, mask=mask, other=0.0).to(tl.float32)
        gwrite = tl.load(gWrite + base + h, mask=mask, other=0.0).to(tl.float32)

        tanh_b = _tl_tanh(b_prev)
        novelty_eff = nov * (1.0 - suppress * c_t)
        energy_nov = novelty_eff
        k_signal = ks * s + ky * y + kb * tanh_b + kbias
        r_signal = rs * s + ry * y + rb * tanh_b + rbias
        k_signal += kn * energy_nov
        r_signal += rn * energy_nov
        k_signal += kc * c_t
        r_signal += rc * c_t
        tanh_k = _tl_tanh(k_signal)
        tanh_r = _tl_tanh(r_signal)
        exp_k = tl.exp(0.5 * tanh_k)
        exp_r = tl.exp(0.5 * tanh_r)
        k_t = kappa * exp_k
        r_pre = recover * exp_r
        r_t = tl.minimum(1.0, tl.maximum(0.0, r_pre))
        r_clamp_grad = tl.where((r_pre >= 0.0) & (r_pre <= 1.0), 1.0, 0.0)

        # ---- slice 1: b/e state update ----
        e_pre = e_prev + r_t * (1.0 - e_prev) - k_t * p
        clamp_grad = tl.where((e_pre >= 0.0) & (e_pre <= 1.0), 1.0, 0.0)
        ge_pre = ge_new * clamp_grad
        g_bprev_state = gb_new * (1.0 - alpha)
        g_eprev_state = ge_pre * (1.0 - r_t)
        gp = gp0 + gb_new * lam * ay + ge_pre * (-k_t)
        gay = gay0 + gb_new * lam * p
        gkt = ge_pre * (-p)
        grt = ge_pre * (1.0 - e_prev)
        a_beta += gb_new
        a_alpha += tl.sum(tl.where(mask, gb_new * (-b_prev) * alpha * (1.0 - alpha), 0.0), axis=0)
        a_lambda += tl.sum(tl.where(mask, gb_new * (p * ay) * sig_lam, 0.0), axis=0)

        # ---- slice 2: ay = A@y, write = p*ay (main path) ----
        d_ay_main = gay + gwrite * p
        gp += gwrite * ay

        # ---- slice 3: energy / recovery coefficients ----
        g_k_signal = gkt * k_t * 0.5 * (1.0 - tanh_k * tanh_k)
        g_r_pre = grt * r_clamp_grad
        g_r_signal = g_r_pre * r_pre * 0.5 * (1.0 - tanh_r * tanh_r)
        gs_energy = g_k_signal * ks + g_r_signal * rs
        gy_energy = g_k_signal * ky + g_r_signal * ry
        gb_energy = (g_k_signal * kb + g_r_signal * rb) * (1.0 - tanh_b * tanh_b)
        gnov_eff = g_k_signal * kn + g_r_signal * rn
        gc_energy = g_k_signal * kc + g_r_signal * rc
        a_logk += gkt * exp_k * tl.sigmoid(lk)
        a_logr += g_r_pre * exp_r * recover * (1.0 - recover)
        a_ks += g_k_signal * s
        a_ky += g_k_signal * y
        a_kb += g_k_signal * tanh_b
        a_kbias += g_k_signal
        a_rs += g_r_signal * s
        a_ry += g_r_signal * y
        a_rb += g_r_signal * tanh_b
        a_rbias += g_r_signal
        a_kn += g_k_signal * energy_nov
        a_rn += g_r_signal * energy_nov
        a_kc += g_k_signal * c_t
        a_rc += g_r_signal * c_t
        gy_accum = gy0 + gy_energy

        # ---- slice 4: gate s = hardsigmoid(Wgx + Ug@y); p = s*e_prev ----
        # dpre is finalized here, but its Ug^T mat-vec is DEFERRED to the merged
        # chunked transpose below (so both transpose mat-vecs share one row loop).
        ds_dpre = tl.where((s > 0.0) & (s < 1.0), 1.0 / 6.0, 0.0)
        dpre = (gp * e_prev + gs_energy) * ds_dpre
        ge_gate = gp * s
        tl.store(gWgx + base + h, dpre, mask=mask)
        tl.store(DUg + base + h, dpre, mask=mask)

        # ---- slice 5: saturation state + novelty suppression ----
        g_c_new_outer = gc_new + gc_energy
        stress = nov * (1.0 - e_prev)
        c_new = decay * c_prev + (1.0 - decay) * stress
        g_c_new = g_c_new_outer + gnov_eff * nov * (-suppress)
        g_stress = g_c_new * (1.0 - decay)
        g_novelty = gnov_eff * (1.0 - suppress * c_new) + g_stress * (1.0 - e_prev)
        g_e_prev_cort = g_stress * (-nov)
        g_c_prev = g_c_new * decay
        g_decay = g_c_new * (c_prev - stress)
        a_cdecay += g_decay * decay * (1.0 - decay)
        a_csup += tl.sum(tl.where(mask, gnov_eff * nov * (-c_new) * suppress * (1.0 - suppress), 0.0), axis=0)
        gnov_total = gnov0 + g_novelty

        # ---- slice 6: novelty = tanh(|ay - b_prev|) ----
        diff = ay - b_prev
        nov_chk = _tl_tanh(_tl_abs(diff))
        sign = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
        d_nov = gnov_total * (1.0 - nov_chk * nov_chk) * sign
        gb_nov = -d_nov

        # ---- slice 7: novelty-path ay = A@y, combine adjoint d_ay ----
        d_ay_total = d_ay_main + d_nov
        tl.store(DAy + base + h, d_ay_total, mask=mask)

        # ---- merged chunked transpose mat-vecs: gy_accum += Ug^T@dpre + A^T@d_ay ----
        # Both reduce over the ROW index i; stash the two row-vectors and read i-slices.
        tl.debug_barrier()
        tl.store(dpre_scratch + bidx * H + h, dpre, mask=mask)
        tl.store(day_scratch + bidx * H + h, d_ay_total, mask=mask)
        tl.debug_barrier()
        gy_mat = tl.zeros((BLOCK_H,), tl.float32)
        for i0 in range(0, H, BLOCK_K):
            irow = i0 + tl.arange(0, BLOCK_K)
            imask = irow < H
            dpre_i = tl.load(dpre_scratch + bidx * H + irow, mask=imask, other=0.0)
            day_i = tl.load(day_scratch + bidx * H + irow, mask=imask, other=0.0)
            blk_mask = imask[:, None] & mask[None, :]
            ug_blk = tl.load(Ug + irow[:, None] * H + h[None, :], mask=blk_mask, other=0.0).to(tl.float32)
            a_blk = tl.load(A + irow[:, None] * H + h[None, :], mask=blk_mask, other=0.0).to(tl.float32)
            gy_mat += tl.sum(ug_blk * dpre_i[:, None], axis=0)
            gy_mat += tl.sum(a_blk * day_i[:, None], axis=0)
        gy_accum += gy_mat

        # ---- slice 8: y = tanh(Wx + b_prev) ----
        gpre = gy_accum * (1.0 - y * y)
        tl.store(gWx + base + h, gpre, mask=mask)
        gb_y = gpre

        # ---- carry to t-1 ----
        gbn = g_bprev_state + gb_energy + gb_nov + gb_y
        gen = g_eprev_state + ge_gate + g_e_prev_cort + ge_out
        gcn = g_c_prev

    tl.store(gInitialB + state_base + h, gbn, mask=mask)
    tl.store(gInitialE + state_base + h, gen, mask=mask)
    tl.store(gInitialC + state_base + h, gcn, mask=mask)
    pp = bidx * NP * H
    tl.store(gparam_p + pp + 0 * H + h, a_beta, mask=mask)
    tl.store(gparam_p + pp + 1 * H + h, a_logk, mask=mask)
    tl.store(gparam_p + pp + 2 * H + h, a_logr, mask=mask)
    tl.store(gparam_p + pp + 3 * H + h, a_ks, mask=mask)
    tl.store(gparam_p + pp + 4 * H + h, a_ky, mask=mask)
    tl.store(gparam_p + pp + 5 * H + h, a_kb, mask=mask)
    tl.store(gparam_p + pp + 6 * H + h, a_kn, mask=mask)
    tl.store(gparam_p + pp + 7 * H + h, a_kbias, mask=mask)
    tl.store(gparam_p + pp + 8 * H + h, a_rs, mask=mask)
    tl.store(gparam_p + pp + 9 * H + h, a_ry, mask=mask)
    tl.store(gparam_p + pp + 10 * H + h, a_rb, mask=mask)
    tl.store(gparam_p + pp + 11 * H + h, a_rn, mask=mask)
    tl.store(gparam_p + pp + 12 * H + h, a_rbias, mask=mask)
    tl.store(gparam_p + pp + 13 * H + h, a_cdecay, mask=mask)
    tl.store(gparam_p + pp + 14 * H + h, a_kc, mask=mask)
    tl.store(gparam_p + pp + 15 * H + h, a_rc, mask=mask)
    tl.store(gscal_p + bidx * 3 + 0, a_alpha)
    tl.store(gscal_p + bidx * 3 + 1, a_lambda)
    tl.store(gscal_p + bidx * 3 + 2, a_csup)


def _dabsn_core_fused_backward(
    U, novelty, p, ay, e_tape, c_tape, s_tape,
    initial_b, initial_c,
    A, Ug,
    beta, log_kappa, logit_recover,
    k_s, k_y, k_b, k_n, k_bias,
    r_s, r_y, r_b, r_n, r_bias,
    logit_c_decay, k_c, r_c,
    logit_alpha, log_lambda, logit_c_suppress,
    gU, gnov, gp, gay, gwrite, ge_out, gc_out, gfinal_b, gfinal_e, gfinal_c,
):
    """Launch the fused reverse-scan kernel and finish grad_A / grad_Ug via GEMM."""
    B, T, _ = novelty.shape
    H = int(novelty.shape[2])
    device = novelty.device
    block = _block_h(H)  # uncapped; one-tile <= 256, chunked above
    NP = _FUSED_NUM_PARAM_VECS

    gWx = torch.zeros((B, T, H), device=device, dtype=torch.float32)
    gWgx = torch.zeros((B, T, H), device=device, dtype=torch.float32)
    DAy = torch.zeros((B, T, H), device=device, dtype=torch.float32)
    DUg = torch.zeros((B, T, H), device=device, dtype=torch.float32)
    gparam_p = torch.zeros((B, NP, H), device=device, dtype=torch.float32)
    gscal_p = torch.zeros((B, 3), device=device, dtype=torch.float32)
    ginitial_b = torch.empty((B, H), device=device, dtype=torch.float32)
    ginitial_e = torch.empty_like(ginitial_b)
    ginitial_c = torch.empty_like(ginitial_b)

    common_args = (
        U.contiguous(), novelty.contiguous(), p.contiguous(), ay.contiguous(),
        e_tape.contiguous(), c_tape.contiguous(), s_tape.contiguous(),
        initial_b.contiguous(), initial_c.contiguous(),
        A.contiguous(), Ug.contiguous(),
        beta.contiguous(), log_kappa.contiguous(), logit_recover.contiguous(),
        k_s.contiguous(), k_y.contiguous(), k_b.contiguous(), k_n.contiguous(), k_bias.contiguous(),
        r_s.contiguous(), r_y.contiguous(), r_b.contiguous(), r_n.contiguous(), r_bias.contiguous(),
        logit_c_decay.contiguous(), k_c.contiguous(), r_c.contiguous(),
        gU.contiguous(), gnov.contiguous(), gp.contiguous(), gay.contiguous(), gwrite.contiguous(),
        ge_out.contiguous(), gc_out.contiguous(),
        gfinal_b.contiguous(), gfinal_e.contiguous(), gfinal_c.contiguous(),
        gWx, gWgx, DAy, DUg,
        ginitial_b, ginitial_e, ginitial_c,
        gparam_p, gscal_p,
    )
    tail_args = (
        logit_alpha.contiguous(), log_lambda.contiguous(), logit_c_suppress.contiguous(),
        T, H, NP,
    )
    if block <= _ONE_TILE_MAX:
        _dabsn_core_scan_bwd[(B,)](
            *common_args, *tail_args,
            BLOCK_H=block,
            num_warps=_fused_num_warps(block),
        )
    else:
        dpre_scratch = torch.zeros((B, H), device=device, dtype=torch.float32)
        day_scratch = torch.zeros((B, H), device=device, dtype=torch.float32)
        # BLOCK_K / num_warps / num_stages chosen by @triton.autotune (keyed on H).
        _dabsn_core_scan_bwd_chunked[(B,)](
            *common_args, dpre_scratch, day_scratch, *tail_args,
            BLOCK_H=block,
        )

    y_flat = U[:, :, :H].reshape(B * T, H).to(torch.float32)
    grad_A = DAy.reshape(B * T, H).transpose(0, 1) @ y_flat
    grad_Ug = DUg.reshape(B * T, H).transpose(0, 1) @ y_flat
    gparam = gparam_p.sum(dim=0)
    gscal = gscal_p.sum(dim=0)
    return gWx, gWgx, grad_A, grad_Ug, gparam, gscal, ginitial_b, ginitial_e, ginitial_c


class DABSNCoreScanTritonFusedBackward(torch.autograd.Function):
    """Triton forward plus fused one-tile or chunked reverse-scan backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        Wx: Tensor,
        Wgx: Tensor,
        Ug: Tensor,
        A: Tensor,
        beta: Tensor,
        log_kappa: Tensor,
        logit_recover: Tensor,
        k_s: Tensor,
        k_y: Tensor,
        k_b: Tensor,
        k_n: Tensor,
        k_bias: Tensor,
        r_s: Tensor,
        r_y: Tensor,
        r_b: Tensor,
        r_n: Tensor,
        r_bias: Tensor,
        logit_c_decay: Tensor,
        k_c: Tensor,
        r_c: Tensor,
        logit_alpha: Tensor,
        log_lambda: Tensor,
        logit_c_suppress: Tensor,
        initial_b: Tensor,
        initial_e: Tensor,
        initial_c: Tensor,
        return_tape: bool,
        precise_tape: bool,
        return_final_state: bool,
    ) -> Tuple[Tensor, ...]:
        ctx.return_tape = bool(return_tape)
        ctx.precise_tape = bool(precise_tape)
        ctx.return_final_state = bool(return_final_state)
        outs = dabsn_core_scan_triton_tape(
            Wx, Wgx, Ug, A,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha=logit_alpha,
            log_lambda=log_lambda,
            logit_c_suppress=logit_c_suppress,
            tape_dtype=_reference_compute_dtype(Wx.dtype) if bool(precise_tape) else Wx.dtype,
            initial_state=(initial_b, initial_e, initial_c),
            return_final_state=True,
        )
        ctx.save_for_backward(
            Wx, Wgx, Ug, A,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha, log_lambda, logit_c_suppress,
            initial_b, initial_e, initial_c,
            *outs,
        )
        public = list(outs[:7] if bool(return_tape) else outs[:5])
        if bool(return_final_state):
            public.extend(outs[8:11])
        return tuple(t.to(Wx.dtype) for t in public)

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        saved = ctx.saved_tensors
        (
            Wx, Wgx, Ug, A,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha, log_lambda, logit_c_suppress,
            initial_b, initial_e, initial_c,
            U, novelty, p, ay, write, e_tape, c_tape, s_tape,
            final_b, final_e, final_c,
        ) = saved
        gU, gnov_out, gp_out, gay_out, gwrite_out = (
            torch.zeros_like(out) if grad is None else grad.contiguous()
            for out, grad in zip((U, novelty, p, ay, write), grad_outputs)
        )
        ge_out = (
            torch.zeros_like(e_tape)
            if not ctx.return_tape or grad_outputs[5] is None
            else grad_outputs[5].contiguous()
        )
        gc_out = (
            torch.zeros_like(c_tape)
            if not ctx.return_tape or grad_outputs[6] is None
            else grad_outputs[6].contiguous()
        )
        final_offset = 7 if ctx.return_tape else 5
        if ctx.return_final_state:
            gfinal_b = (
                torch.zeros_like(final_b)
                if grad_outputs[final_offset] is None
                else grad_outputs[final_offset].contiguous()
            )
            gfinal_e = (
                torch.zeros_like(final_e)
                if grad_outputs[final_offset + 1] is None
                else grad_outputs[final_offset + 1].contiguous()
            )
            gfinal_c = (
                torch.zeros_like(final_c)
                if grad_outputs[final_offset + 2] is None
                else grad_outputs[final_offset + 2].contiguous()
            )
        else:
            gfinal_b = torch.zeros_like(final_b)
            gfinal_e = torch.zeros_like(final_e)
            gfinal_c = torch.zeros_like(final_c)

        gWx, gWgx, grad_A, grad_Ug, gparam, gscal, ginitial_b, ginitial_e, ginitial_c = _dabsn_core_fused_backward(
            U, novelty, p, ay, e_tape, c_tape, s_tape,
            initial_b, initial_c,
            A, Ug,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha, log_lambda, logit_c_suppress,
            gU, gnov_out, gp_out, gay_out, gwrite_out, ge_out, gc_out,
            gfinal_b, gfinal_e, gfinal_c,
        )

        return (
            gWx.to(Wx.dtype),
            gWgx.to(Wgx.dtype),
            grad_Ug.to(Ug.dtype),
            grad_A.to(A.dtype),
            gparam[0].to(beta.dtype),
            gparam[1].to(log_kappa.dtype),
            gparam[2].to(logit_recover.dtype),
            gparam[3].to(k_s.dtype),
            gparam[4].to(k_y.dtype),
            gparam[5].to(k_b.dtype),
            gparam[6].to(k_n.dtype),
            gparam[7].to(k_bias.dtype),
            gparam[8].to(r_s.dtype),
            gparam[9].to(r_y.dtype),
            gparam[10].to(r_b.dtype),
            gparam[11].to(r_n.dtype),
            gparam[12].to(r_bias.dtype),
            gparam[13].to(logit_c_decay.dtype),
            gparam[14].to(k_c.dtype),
            gparam[15].to(r_c.dtype),
            gscal[0].reshape_as(logit_alpha).to(logit_alpha.dtype),
            gscal[1].reshape_as(log_lambda).to(log_lambda.dtype),
            gscal[2].reshape_as(logit_c_suppress).to(logit_c_suppress.dtype),
            ginitial_b.to(initial_b.dtype),
            ginitial_e.to(initial_e.dtype),
            ginitial_c.to(initial_c.dtype),
            None,
            None,
            None,
        )


def dabsn_core_scan_trainable_fused(
    Wx: Tensor,
    Wgx: Tensor,
    Ug: Tensor,
    A: Tensor,
    beta: Tensor,
    log_kappa: Tensor,
    logit_recover: Tensor,
    k_s: Tensor,
    k_y: Tensor,
    k_b: Tensor,
    k_n: Tensor,
    k_bias: Tensor,
    r_s: Tensor,
    r_y: Tensor,
    r_b: Tensor,
    r_n: Tensor,
    r_bias: Tensor,
    logit_c_decay: Tensor,
    k_c: Tensor,
    r_c: Tensor,
    logit_alpha: Tensor,
    log_lambda: Tensor,
    logit_c_suppress: Tensor,
    *,
    return_tape: bool = False,
    precise_tape: bool = False,
    initial_state: tuple[Tensor, Tensor, Tensor] | None = None,
    return_final_state: bool = False,
) -> Tuple[Tensor, ...]:
    """Triton forward tape with fused one-tile or chunked reverse-scan backward."""
    B, _T, H = Wx.shape
    if initial_state is None:
        initial_state = (
            torch.zeros((B, H), device=Wx.device, dtype=Wx.dtype),
            torch.ones((B, H), device=Wx.device, dtype=Wx.dtype),
            torch.zeros((B, H), device=Wx.device, dtype=Wx.dtype),
        )
    initial_b, initial_e, initial_c = (
        value.to(device=Wx.device, dtype=Wx.dtype).contiguous() for value in initial_state
    )
    if any(value.shape != (B, H) for value in (initial_b, initial_e, initial_c)):
        raise ValueError(f"initial_state tensors must have shape {(B, H)}")
    return DABSNCoreScanTritonFusedBackward.apply(
        Wx, Wgx, Ug, A,
        beta, log_kappa, logit_recover,
        k_s, k_y, k_b, k_n, k_bias,
        r_s, r_y, r_b, r_n, r_bias,
        logit_c_decay, k_c, r_c,
        logit_alpha, log_lambda, logit_c_suppress,
        initial_b, initial_e, initial_c,
        bool(return_tape), bool(precise_tape),
        bool(return_final_state),
    )


@triton.jit
def _dabsn_admitted_three_way_fwd(
    Q, KB, WB, WBNext, Cocktail, CB, KeyBias, Adm, Allow, InductAllow, HasElig, InductElig, Out,
    Scale, ShortGain, PadGain, InductGain, CocktailGain,
    T: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    bid_t = tl.program_id(0)
    bidx = bid_t // T
    tidx = bid_t - bidx * T
    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    scale = tl.load(Scale).to(tl.float32)
    short_gain = tl.load(ShortGain).to(tl.float32)
    pad_gain = tl.load(PadGain).to(tl.float32)
    induct_gain = tl.load(InductGain).to(tl.float32)
    cocktail_gain = tl.load(CocktailGain).to(tl.float32)

    q = tl.load(Q + (bidx * T + tidx) * H + h, mask=hmask, other=0.0).to(tl.float32)
    c0 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 0).to(tl.float32)
    c1 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 1).to(tl.float32)
    c2 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 2).to(tl.float32)
    c3 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 3).to(tl.float32)
    has = tl.load(HasElig + bidx * T + tidx) != 0
    induct_has = tl.load(InductElig + bidx * T + tidx) != 0

    m_s = tl.full((), -3.402823e38, tl.float32)
    l_s = tl.full((), 0.0, tl.float32)
    acc_s = tl.zeros((BLOCK_H,), tl.float32)
    m_p = tl.full((), -3.402823e38, tl.float32)
    l_p = tl.full((), 0.0, tl.float32)
    acc_p = tl.zeros((BLOCK_H,), tl.float32)
    m_i = tl.full((), -3.402823e38, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc_i = tl.zeros((BLOCK_H,), tl.float32)

    for j in range(0, N):
        k = tl.load(KB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wb = tl.load(WB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wbn = tl.load(WBNext + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        dot = tl.sum(q * k, axis=0) * scale
        key_adm = tl.load(KeyBias + bidx * N + j).to(tl.float32) + tl.load(Adm + bidx * N + j).to(tl.float32)
        short_score = dot + key_adm
        cb0 = tl.load(CB + (bidx * N + j) * 4 + 0).to(tl.float32)
        cb1 = tl.load(CB + (bidx * N + j) * 4 + 1).to(tl.float32)
        cb2 = tl.load(CB + (bidx * N + j) * 4 + 2).to(tl.float32)
        cb3 = tl.load(CB + (bidx * N + j) * 4 + 3).to(tl.float32)
        perm_score = short_score + (c0 * cb0 + c1 * cb1 + c2 * cb2 + c3 * cb3) * cocktail_gain

        a = (tl.load(Allow + (bidx * T + tidx) * N + j) != 0) & has
        ia = (tl.load(InductAllow + (bidx * T + tidx) * N + j) != 0) & induct_has

        s_score = tl.where(a, short_score, -3.402823e38)
        p_score = tl.where(a, perm_score, -3.402823e38)
        i_score = tl.where(ia, short_score, -3.402823e38)

        new_m_s = tl.maximum(m_s, s_score)
        alpha_s = tl.exp(m_s - new_m_s)
        beta_s = tl.exp(s_score - new_m_s)
        beta_s = tl.where(a, beta_s, 0.0)
        acc_s = acc_s * alpha_s + wb * beta_s
        l_s = l_s * alpha_s + beta_s
        m_s = new_m_s

        new_m_p = tl.maximum(m_p, p_score)
        alpha_p = tl.exp(m_p - new_m_p)
        beta_p = tl.exp(p_score - new_m_p)
        beta_p = tl.where(a, beta_p, 0.0)
        acc_p = acc_p * alpha_p + wb * beta_p
        l_p = l_p * alpha_p + beta_p
        m_p = new_m_p

        new_m_i = tl.maximum(m_i, i_score)
        alpha_i = tl.exp(m_i - new_m_i)
        beta_i = tl.exp(i_score - new_m_i)
        beta_i = tl.where(ia, beta_i, 0.0)
        acc_i = acc_i * alpha_i + wbn * beta_i
        l_i = l_i * alpha_i + beta_i
        m_i = new_m_i

    short = tl.where(has, acc_s / tl.maximum(l_s, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    perm = tl.where(has, acc_p / tl.maximum(l_p, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    induct = tl.where(induct_has, acc_i / tl.maximum(l_i, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    out = short_gain * short + pad_gain * perm + induct_gain * induct
    tl.store(Out + (bidx * T + tidx) * H + h, out, mask=hmask)


@triton.jit
def _dabsn_admitted_three_way_compact_fwd(
    Q, KB, WB, WBNext, Cocktail, CB, KeyBias, Adm, BankIdx, BankValid, Count, Out,
    Scale, ShortGain, PadGain, InductGain, CocktailGain,
    T: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    bid_t = tl.program_id(0)
    bidx = bid_t // T
    tidx = bid_t - bidx * T
    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    scale = tl.load(Scale).to(tl.float32)
    short_gain = tl.load(ShortGain).to(tl.float32)
    pad_gain = tl.load(PadGain).to(tl.float32)
    induct_gain = tl.load(InductGain).to(tl.float32)
    cocktail_gain = tl.load(CocktailGain).to(tl.float32)

    q = tl.load(Q + (bidx * T + tidx) * H + h, mask=hmask, other=0.0).to(tl.float32)
    c0 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 0).to(tl.float32)
    c1 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 1).to(tl.float32)
    c2 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 2).to(tl.float32)
    c3 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 3).to(tl.float32)

    m_s = tl.full((), -3.402823e38, tl.float32)
    l_s = tl.full((), 0.0, tl.float32)
    acc_s = tl.zeros((BLOCK_H,), tl.float32)
    m_p = tl.full((), -3.402823e38, tl.float32)
    l_p = tl.full((), 0.0, tl.float32)
    acc_p = tl.zeros((BLOCK_H,), tl.float32)
    m_i = tl.full((), -3.402823e38, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc_i = tl.zeros((BLOCK_H,), tl.float32)
    has = tl.full((), 0, tl.int32)
    induct_has = tl.full((), 0, tl.int32)

    # Front-packed members and a device count visit only live entries.
    count_b = tl.load(Count + bidx).to(tl.int32)
    for j in range(0, count_b):
        bank_pos = tl.load(BankIdx + bidx * N + j).to(tl.int64)
        valid = tl.load(BankValid + bidx * N + j) != 0
        if MODE == 0:
            a = valid & (bank_pos <= tidx)
            ia = valid & (bank_pos < tidx)
        else:
            a = valid
            ia = valid & (bank_pos < T - 1)
        has += tl.where(a, 1, 0)
        induct_has += tl.where(ia, 1, 0)

        k = tl.load(KB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wb = tl.load(WB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wbn = tl.load(WBNext + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        dot = tl.sum(q * k, axis=0) * scale
        key_adm = tl.load(KeyBias + bidx * N + j).to(tl.float32) + tl.load(Adm + bidx * N + j).to(tl.float32)
        short_score = dot + key_adm
        cb0 = tl.load(CB + (bidx * N + j) * 4 + 0).to(tl.float32)
        cb1 = tl.load(CB + (bidx * N + j) * 4 + 1).to(tl.float32)
        cb2 = tl.load(CB + (bidx * N + j) * 4 + 2).to(tl.float32)
        cb3 = tl.load(CB + (bidx * N + j) * 4 + 3).to(tl.float32)
        perm_score = short_score + (c0 * cb0 + c1 * cb1 + c2 * cb2 + c3 * cb3) * cocktail_gain

        s_score = tl.where(a, short_score, -3.402823e38)
        p_score = tl.where(a, perm_score, -3.402823e38)
        i_score = tl.where(ia, short_score, -3.402823e38)

        new_m_s = tl.maximum(m_s, s_score)
        alpha_s = tl.exp(m_s - new_m_s)
        beta_s = tl.exp(s_score - new_m_s)
        beta_s = tl.where(a, beta_s, 0.0)
        acc_s = acc_s * alpha_s + wb * beta_s
        l_s = l_s * alpha_s + beta_s
        m_s = new_m_s

        new_m_p = tl.maximum(m_p, p_score)
        alpha_p = tl.exp(m_p - new_m_p)
        beta_p = tl.exp(p_score - new_m_p)
        beta_p = tl.where(a, beta_p, 0.0)
        acc_p = acc_p * alpha_p + wb * beta_p
        l_p = l_p * alpha_p + beta_p
        m_p = new_m_p

        new_m_i = tl.maximum(m_i, i_score)
        alpha_i = tl.exp(m_i - new_m_i)
        beta_i = tl.exp(i_score - new_m_i)
        beta_i = tl.where(ia, beta_i, 0.0)
        acc_i = acc_i * alpha_i + wbn * beta_i
        l_i = l_i * alpha_i + beta_i
        m_i = new_m_i

    short = tl.where(has != 0, acc_s / tl.maximum(l_s, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    perm = tl.where(has != 0, acc_p / tl.maximum(l_p, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    induct = tl.where(induct_has != 0, acc_i / tl.maximum(l_i, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    out = short_gain * short + pad_gain * perm + induct_gain * induct
    tl.store(Out + (bidx * T + tidx) * H + h, out, mask=hmask)


@triton.jit
def _dabsn_admitted_three_way_compact_flash_fwd(
    Q, KB, WB, WBNext, Cocktail, CB, KeyBias, Adm, BankIdx, BankValid, Count, Out,
    Scale, ShortGain, PadGain, InductGain, CocktailGain,
    T: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DOT: tl.constexpr,
    BLOCK_V: tl.constexpr,
    PREC: tl.constexpr,
):
    """Forward-only tiled exact admitted read for large eval/inference banks.

    This is the flash-shaped version of `_dabsn_admitted_three_way_compact_fwd`:
    one program computes a query tile and output-value tile while streaming key
    tiles with online softmax. It preserves the full admitted-write semantics but
    avoids one-program-per-query serial bank loops and Python-level torch tiling.
    """
    q_block = tl.program_id(0)
    v_block = tl.program_id(1)
    bidx = tl.program_id(2)

    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    m_valid = offs_m < T
    v_valid = offs_v < H

    c0 = tl.load(Cocktail + (bidx * T + offs_m) * 4 + 0, mask=m_valid, other=0.0).to(tl.float32)
    c1 = tl.load(Cocktail + (bidx * T + offs_m) * 4 + 1, mask=m_valid, other=0.0).to(tl.float32)
    c2 = tl.load(Cocktail + (bidx * T + offs_m) * 4 + 2, mask=m_valid, other=0.0).to(tl.float32)
    c3 = tl.load(Cocktail + (bidx * T + offs_m) * 4 + 3, mask=m_valid, other=0.0).to(tl.float32)
    scale = tl.load(Scale).to(tl.float32)
    short_gain = tl.load(ShortGain).to(tl.float32)
    pad_gain = tl.load(PadGain).to(tl.float32)
    induct_gain = tl.load(InductGain).to(tl.float32)
    cocktail_gain = tl.load(CocktailGain).to(tl.float32)

    m_s = tl.full((BLOCK_M,), -3.402823e38, tl.float32)
    l_s = tl.zeros((BLOCK_M,), tl.float32)
    acc_s = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)
    m_p = tl.full((BLOCK_M,), -3.402823e38, tl.float32)
    l_p = tl.zeros((BLOCK_M,), tl.float32)
    acc_p = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)
    m_i = tl.full((BLOCK_M,), -3.402823e38, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)

    # The launcher front-packs live entries and passes a device-side count, so the
    # key loop visits only the active bank without a host synchronization.
    count_b = tl.load(Count + bidx).to(tl.int32)
    for n0 in range(0, count_b, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < count_b
        bank_pos = tl.load(BankIdx + bidx * N + offs_n, mask=n_valid, other=T).to(tl.int64)
        valid = (tl.load(BankValid + bidx * N + offs_n, mask=n_valid, other=0) != 0) & n_valid

        scores = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for d0 in range(0, H, BLOCK_DOT):
            offs_d = d0 + tl.arange(0, BLOCK_DOT)
            d_valid = offs_d < H
            q = tl.load(
                Q + (bidx * T + offs_m[:, None]) * H + offs_d[None, :],
                mask=m_valid[:, None] & d_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                KB + (bidx * N + offs_n[:, None]) * H + offs_d[None, :],
                mask=n_valid[:, None] & d_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            # Tensor-core QK contraction. tf32x3 is the balanced default; ieee and
            # tf32 remain selectable precision modes.
            if PREC == "ieee":
                scores += tl.dot(q, tl.trans(k), input_precision="ieee")
            elif PREC == "tf32":
                scores += tl.dot(q, tl.trans(k), input_precision="tf32")
            else:
                scores += tl.dot(q, tl.trans(k), input_precision="tf32x3")

        scores *= scale
        key_adm = (
            tl.load(KeyBias + bidx * N + offs_n, mask=n_valid, other=0.0).to(tl.float32)
            + tl.load(Adm + bidx * N + offs_n, mask=n_valid, other=0.0).to(tl.float32)
        )
        cb0 = tl.load(CB + (bidx * N + offs_n) * 4 + 0, mask=n_valid, other=0.0).to(tl.float32)
        cb1 = tl.load(CB + (bidx * N + offs_n) * 4 + 1, mask=n_valid, other=0.0).to(tl.float32)
        cb2 = tl.load(CB + (bidx * N + offs_n) * 4 + 2, mask=n_valid, other=0.0).to(tl.float32)
        cb3 = tl.load(CB + (bidx * N + offs_n) * 4 + 3, mask=n_valid, other=0.0).to(tl.float32)
        short_scores = scores + key_adm[None, :]
        cocktail_scores = (c0[:, None] * cb0[None, :] + c1[:, None] * cb1[None, :] + c2[:, None] * cb2[None, :] + c3[:, None] * cb3[None, :]) * cocktail_gain
        perm_scores = short_scores + cocktail_scores

        if MODE == 0:
            allow = valid[None, :] & (bank_pos[None, :] <= offs_m[:, None])
            induct_allow = valid[None, :] & (bank_pos[None, :] < offs_m[:, None])
        else:
            allow = valid[None, :]
            induct_allow = valid[None, :] & (bank_pos[None, :] < (T - 1))
        allow = allow & m_valid[:, None]
        induct_allow = induct_allow & m_valid[:, None]

        values = tl.load(
            WB + (bidx * N + offs_n[:, None]) * H + offs_v[None, :],
            mask=n_valid[:, None] & v_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        next_values = tl.load(
            WBNext + (bidx * N + offs_n[:, None]) * H + offs_v[None, :],
            mask=n_valid[:, None] & v_valid[None, :],
            other=0.0,
        ).to(tl.float32)

        s_score = tl.where(allow, short_scores, -3.402823e38)
        p_score = tl.where(allow, perm_scores, -3.402823e38)
        i_score = tl.where(induct_allow, short_scores, -3.402823e38)

        new_m_s = tl.maximum(m_s, tl.max(s_score, axis=1))
        alpha_s = tl.exp(m_s - new_m_s)
        beta_s = tl.exp(s_score - new_m_s[:, None])
        beta_s = tl.where(allow, beta_s, 0.0)
        l_s = l_s * alpha_s + tl.sum(beta_s, axis=1)
        if PREC == "ieee":
            acc_s = acc_s * alpha_s[:, None] + tl.dot(beta_s.to(values.dtype), values, input_precision="ieee")
        elif PREC == "tf32":
            acc_s = acc_s * alpha_s[:, None] + tl.dot(beta_s.to(values.dtype), values, input_precision="tf32")
        else:
            acc_s = acc_s * alpha_s[:, None] + tl.dot(beta_s.to(values.dtype), values, input_precision="tf32x3")
        m_s = new_m_s

        new_m_p = tl.maximum(m_p, tl.max(p_score, axis=1))
        alpha_p = tl.exp(m_p - new_m_p)
        beta_p = tl.exp(p_score - new_m_p[:, None])
        beta_p = tl.where(allow, beta_p, 0.0)
        l_p = l_p * alpha_p + tl.sum(beta_p, axis=1)
        if PREC == "ieee":
            acc_p = acc_p * alpha_p[:, None] + tl.dot(beta_p.to(values.dtype), values, input_precision="ieee")
        elif PREC == "tf32":
            acc_p = acc_p * alpha_p[:, None] + tl.dot(beta_p.to(values.dtype), values, input_precision="tf32")
        else:
            acc_p = acc_p * alpha_p[:, None] + tl.dot(beta_p.to(values.dtype), values, input_precision="tf32x3")
        m_p = new_m_p

        new_m_i = tl.maximum(m_i, tl.max(i_score, axis=1))
        alpha_i = tl.exp(m_i - new_m_i)
        beta_i = tl.exp(i_score - new_m_i[:, None])
        beta_i = tl.where(induct_allow, beta_i, 0.0)
        l_i = l_i * alpha_i + tl.sum(beta_i, axis=1)
        if PREC == "ieee":
            acc_i = acc_i * alpha_i[:, None] + tl.dot(beta_i.to(next_values.dtype), next_values, input_precision="ieee")
        elif PREC == "tf32":
            acc_i = acc_i * alpha_i[:, None] + tl.dot(beta_i.to(next_values.dtype), next_values, input_precision="tf32")
        else:
            acc_i = acc_i * alpha_i[:, None] + tl.dot(beta_i.to(next_values.dtype), next_values, input_precision="tf32x3")
        m_i = new_m_i

    short = acc_s / tl.maximum(l_s, 1.0e-12)[:, None]
    perm = acc_p / tl.maximum(l_p, 1.0e-12)[:, None]
    induct = acc_i / tl.maximum(l_i, 1.0e-12)[:, None]
    short = tl.where((l_s > 0.0)[:, None], short, tl.zeros((BLOCK_M, BLOCK_V), tl.float32))
    perm = tl.where((l_p > 0.0)[:, None], perm, tl.zeros((BLOCK_M, BLOCK_V), tl.float32))
    induct = tl.where((l_i > 0.0)[:, None], induct, tl.zeros((BLOCK_M, BLOCK_V), tl.float32))
    out = short_gain * short + pad_gain * perm + induct_gain * induct
    tl.store(
        Out + (bidx * T + offs_m[:, None]) * H + offs_v[None, :],
        out,
        mask=m_valid[:, None] & v_valid[None, :],
    )


@triton.jit
def _dabsn_admitted_three_way_bwd(
    Q, KB, WB, WBNext, Cocktail, CB, KeyBias, Adm,
    Allow, InductAllow, HasElig, InductElig, GradOut,
    GradQ, GradKB, GradWB, GradWBNext, GradCocktail, GradCB, GradKeyBias, GradAdm,
    GradScaleParts, GradShortGainParts, GradPadGainParts, GradInductGainParts, GradCocktailGainParts,
    Scale, ShortGain, PadGain, InductGain, CocktailGain,
    T: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    bid_t = tl.program_id(0)
    bidx = bid_t // T
    tidx = bid_t - bidx * T
    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    scale = tl.load(Scale).to(tl.float32)
    short_gain = tl.load(ShortGain).to(tl.float32)
    pad_gain = tl.load(PadGain).to(tl.float32)
    induct_gain = tl.load(InductGain).to(tl.float32)
    cocktail_gain = tl.load(CocktailGain).to(tl.float32)

    q = tl.load(Q + (bidx * T + tidx) * H + h, mask=hmask, other=0.0).to(tl.float32)
    gout = tl.load(GradOut + (bidx * T + tidx) * H + h, mask=hmask, other=0.0).to(tl.float32)
    c0 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 0).to(tl.float32)
    c1 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 1).to(tl.float32)
    c2 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 2).to(tl.float32)
    c3 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 3).to(tl.float32)
    has = tl.load(HasElig + bidx * T + tidx) != 0
    induct_has = tl.load(InductElig + bidx * T + tidx) != 0

    # Pass 1: recompute online-softmax normalizers and read vectors.
    m_s = tl.full((), -3.402823e38, tl.float32)
    l_s = tl.full((), 0.0, tl.float32)
    acc_s = tl.zeros((BLOCK_H,), tl.float32)
    m_p = tl.full((), -3.402823e38, tl.float32)
    l_p = tl.full((), 0.0, tl.float32)
    acc_p = tl.zeros((BLOCK_H,), tl.float32)
    m_i = tl.full((), -3.402823e38, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc_i = tl.zeros((BLOCK_H,), tl.float32)

    for j in range(0, N):
        k = tl.load(KB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wb = tl.load(WB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wbn = tl.load(WBNext + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        dot = tl.sum(q * k, axis=0) * scale
        key_adm = tl.load(KeyBias + bidx * N + j).to(tl.float32) + tl.load(Adm + bidx * N + j).to(tl.float32)
        short_score = dot + key_adm
        cb0 = tl.load(CB + (bidx * N + j) * 4 + 0).to(tl.float32)
        cb1 = tl.load(CB + (bidx * N + j) * 4 + 1).to(tl.float32)
        cb2 = tl.load(CB + (bidx * N + j) * 4 + 2).to(tl.float32)
        cb3 = tl.load(CB + (bidx * N + j) * 4 + 3).to(tl.float32)
        cocktail_dot = c0 * cb0 + c1 * cb1 + c2 * cb2 + c3 * cb3
        perm_score = short_score + cocktail_dot * cocktail_gain
        a = (tl.load(Allow + (bidx * T + tidx) * N + j) != 0) & has
        ia = (tl.load(InductAllow + (bidx * T + tidx) * N + j) != 0) & induct_has

        s_score = tl.where(a, short_score, -3.402823e38)
        p_score = tl.where(a, perm_score, -3.402823e38)
        i_score = tl.where(ia, short_score, -3.402823e38)

        new_m_s = tl.maximum(m_s, s_score)
        alpha_s = tl.exp(m_s - new_m_s)
        beta_s = tl.exp(s_score - new_m_s)
        beta_s = tl.where(a, beta_s, 0.0)
        acc_s = acc_s * alpha_s + wb * beta_s
        l_s = l_s * alpha_s + beta_s
        m_s = new_m_s

        new_m_p = tl.maximum(m_p, p_score)
        alpha_p = tl.exp(m_p - new_m_p)
        beta_p = tl.exp(p_score - new_m_p)
        beta_p = tl.where(a, beta_p, 0.0)
        acc_p = acc_p * alpha_p + wb * beta_p
        l_p = l_p * alpha_p + beta_p
        m_p = new_m_p

        new_m_i = tl.maximum(m_i, i_score)
        alpha_i = tl.exp(m_i - new_m_i)
        beta_i = tl.exp(i_score - new_m_i)
        beta_i = tl.where(ia, beta_i, 0.0)
        acc_i = acc_i * alpha_i + wbn * beta_i
        l_i = l_i * alpha_i + beta_i
        m_i = new_m_i

    read_s = tl.where(has, acc_s / tl.maximum(l_s, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    read_p = tl.where(has, acc_p / tl.maximum(l_p, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    read_i = tl.where(induct_has, acc_i / tl.maximum(l_i, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    gs = gout * short_gain
    gp = gout * pad_gain
    gi = gout * induct_gain
    dot_gs_read = tl.sum(gs * read_s, axis=0)
    dot_gp_read = tl.sum(gp * read_p, axis=0)
    dot_gi_read = tl.sum(gi * read_i, axis=0)

    gq = tl.zeros((BLOCK_H,), tl.float32)
    gc0 = tl.full((), 0.0, tl.float32)
    gc1 = tl.full((), 0.0, tl.float32)
    gc2 = tl.full((), 0.0, tl.float32)
    gc3 = tl.full((), 0.0, tl.float32)
    g_scale = tl.full((), 0.0, tl.float32)
    g_cocktail_gain = tl.full((), 0.0, tl.float32)

    # Pass 2: gradients wrt values and scores.
    for j in range(0, N):
        k = tl.load(KB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wb = tl.load(WB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wbn = tl.load(WBNext + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        dot_raw = tl.sum(q * k, axis=0)
        short_score = dot_raw * scale + tl.load(KeyBias + bidx * N + j).to(tl.float32) + tl.load(Adm + bidx * N + j).to(tl.float32)
        cb0 = tl.load(CB + (bidx * N + j) * 4 + 0).to(tl.float32)
        cb1 = tl.load(CB + (bidx * N + j) * 4 + 1).to(tl.float32)
        cb2 = tl.load(CB + (bidx * N + j) * 4 + 2).to(tl.float32)
        cb3 = tl.load(CB + (bidx * N + j) * 4 + 3).to(tl.float32)
        cocktail_dot = c0 * cb0 + c1 * cb1 + c2 * cb2 + c3 * cb3
        perm_score = short_score + cocktail_dot * cocktail_gain
        a = (tl.load(Allow + (bidx * T + tidx) * N + j) != 0) & has
        ia = (tl.load(InductAllow + (bidx * T + tidx) * N + j) != 0) & induct_has

        ws = tl.where(a, tl.exp(short_score - m_s) / tl.maximum(l_s, 1.0e-12), 0.0)
        wp = tl.where(a, tl.exp(perm_score - m_p) / tl.maximum(l_p, 1.0e-12), 0.0)
        wi = tl.where(ia, tl.exp(short_score - m_i) / tl.maximum(l_i, 1.0e-12), 0.0)

        ds = ws * (tl.sum(gs * wb, axis=0) - dot_gs_read)
        dp = wp * (tl.sum(gp * wb, axis=0) - dot_gp_read)
        di = wi * (tl.sum(gi * wbn, axis=0) - dot_gi_read)
        d_short_score = ds + dp + di

        tl.atomic_add(GradWB + (bidx * N + j) * H + h, ws * gs + wp * gp, sem="relaxed", mask=hmask)
        tl.atomic_add(GradWBNext + (bidx * N + j) * H + h, wi * gi, sem="relaxed", mask=hmask)
        tl.atomic_add(GradKB + (bidx * N + j) * H + h, d_short_score * q * scale, sem="relaxed", mask=hmask)
        gq += d_short_score * k * scale
        tl.atomic_add(GradKeyBias + bidx * N + j, d_short_score, sem="relaxed")
        tl.atomic_add(GradAdm + bidx * N + j, d_short_score, sem="relaxed")
        g_scale += d_short_score * dot_raw

        # Only the permanent/cocktail read has cocktail-specific score terms.
        gc0 += dp * cocktail_gain * cb0
        gc1 += dp * cocktail_gain * cb1
        gc2 += dp * cocktail_gain * cb2
        gc3 += dp * cocktail_gain * cb3
        tl.atomic_add(GradCB + (bidx * N + j) * 4 + 0, dp * cocktail_gain * c0, sem="relaxed")
        tl.atomic_add(GradCB + (bidx * N + j) * 4 + 1, dp * cocktail_gain * c1, sem="relaxed")
        tl.atomic_add(GradCB + (bidx * N + j) * 4 + 2, dp * cocktail_gain * c2, sem="relaxed")
        tl.atomic_add(GradCB + (bidx * N + j) * 4 + 3, dp * cocktail_gain * c3, sem="relaxed")
        g_cocktail_gain += dp * cocktail_dot

    tl.store(GradQ + (bidx * T + tidx) * H + h, gq, mask=hmask)
    tl.store(GradCocktail + (bidx * T + tidx) * 4 + 0, gc0)
    tl.store(GradCocktail + (bidx * T + tidx) * 4 + 1, gc1)
    tl.store(GradCocktail + (bidx * T + tidx) * 4 + 2, gc2)
    tl.store(GradCocktail + (bidx * T + tidx) * 4 + 3, gc3)
    part = bidx * T + tidx
    tl.store(GradScaleParts + part, g_scale)
    tl.store(GradShortGainParts + part, tl.sum(gout * read_s, axis=0))
    tl.store(GradPadGainParts + part, tl.sum(gout * read_p, axis=0))
    tl.store(GradInductGainParts + part, tl.sum(gout * read_i, axis=0))
    tl.store(GradCocktailGainParts + part, g_cocktail_gain)


@triton.jit
def _dabsn_admitted_three_way_compact_bwd(
    Q, KB, WB, WBNext, Cocktail, CB, KeyBias, Adm,
    BankIdx, BankValid, Count, GradOut,
    GradQ, GradKB, GradWB, GradWBNext, GradCocktail, GradCB, GradKeyBias, GradAdm,
    GradScaleParts, GradShortGainParts, GradPadGainParts, GradInductGainParts, GradCocktailGainParts,
    Scale, ShortGain, PadGain, InductGain, CocktailGain,
    T: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    bid_t = tl.program_id(0)
    bidx = bid_t // T
    tidx = bid_t - bidx * T
    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    scale = tl.load(Scale).to(tl.float32)
    short_gain = tl.load(ShortGain).to(tl.float32)
    pad_gain = tl.load(PadGain).to(tl.float32)
    induct_gain = tl.load(InductGain).to(tl.float32)
    cocktail_gain = tl.load(CocktailGain).to(tl.float32)

    q = tl.load(Q + (bidx * T + tidx) * H + h, mask=hmask, other=0.0).to(tl.float32)
    gout = tl.load(GradOut + (bidx * T + tidx) * H + h, mask=hmask, other=0.0).to(tl.float32)
    c0 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 0).to(tl.float32)
    c1 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 1).to(tl.float32)
    c2 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 2).to(tl.float32)
    c3 = tl.load(Cocktail + (bidx * T + tidx) * 4 + 3).to(tl.float32)

    m_s = tl.full((), -3.402823e38, tl.float32)
    l_s = tl.full((), 0.0, tl.float32)
    acc_s = tl.zeros((BLOCK_H,), tl.float32)
    m_p = tl.full((), -3.402823e38, tl.float32)
    l_p = tl.full((), 0.0, tl.float32)
    acc_p = tl.zeros((BLOCK_H,), tl.float32)
    m_i = tl.full((), -3.402823e38, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc_i = tl.zeros((BLOCK_H,), tl.float32)
    has = tl.full((), 0, tl.int32)
    induct_has = tl.full((), 0, tl.int32)

    # Front-packed members and a device count visit only live entries.
    count_b = tl.load(Count + bidx).to(tl.int32)
    for j in range(0, count_b):
        bank_pos = tl.load(BankIdx + bidx * N + j).to(tl.int64)
        valid = tl.load(BankValid + bidx * N + j) != 0
        if MODE == 0:
            a = valid & (bank_pos <= tidx)
            ia = valid & (bank_pos < tidx)
        else:
            a = valid
            ia = valid & (bank_pos < T - 1)
        has += tl.where(a, 1, 0)
        induct_has += tl.where(ia, 1, 0)

        k = tl.load(KB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wb = tl.load(WB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wbn = tl.load(WBNext + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        dot = tl.sum(q * k, axis=0) * scale
        key_adm = tl.load(KeyBias + bidx * N + j).to(tl.float32) + tl.load(Adm + bidx * N + j).to(tl.float32)
        short_score = dot + key_adm
        cb0 = tl.load(CB + (bidx * N + j) * 4 + 0).to(tl.float32)
        cb1 = tl.load(CB + (bidx * N + j) * 4 + 1).to(tl.float32)
        cb2 = tl.load(CB + (bidx * N + j) * 4 + 2).to(tl.float32)
        cb3 = tl.load(CB + (bidx * N + j) * 4 + 3).to(tl.float32)
        cocktail_dot = c0 * cb0 + c1 * cb1 + c2 * cb2 + c3 * cb3
        perm_score = short_score + cocktail_dot * cocktail_gain

        s_score = tl.where(a, short_score, -3.402823e38)
        p_score = tl.where(a, perm_score, -3.402823e38)
        i_score = tl.where(ia, short_score, -3.402823e38)

        new_m_s = tl.maximum(m_s, s_score)
        alpha_s = tl.exp(m_s - new_m_s)
        beta_s = tl.exp(s_score - new_m_s)
        beta_s = tl.where(a, beta_s, 0.0)
        acc_s = acc_s * alpha_s + wb * beta_s
        l_s = l_s * alpha_s + beta_s
        m_s = new_m_s

        new_m_p = tl.maximum(m_p, p_score)
        alpha_p = tl.exp(m_p - new_m_p)
        beta_p = tl.exp(p_score - new_m_p)
        beta_p = tl.where(a, beta_p, 0.0)
        acc_p = acc_p * alpha_p + wb * beta_p
        l_p = l_p * alpha_p + beta_p
        m_p = new_m_p

        new_m_i = tl.maximum(m_i, i_score)
        alpha_i = tl.exp(m_i - new_m_i)
        beta_i = tl.exp(i_score - new_m_i)
        beta_i = tl.where(ia, beta_i, 0.0)
        acc_i = acc_i * alpha_i + wbn * beta_i
        l_i = l_i * alpha_i + beta_i
        m_i = new_m_i

    read_s = tl.where(has != 0, acc_s / tl.maximum(l_s, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    read_p = tl.where(has != 0, acc_p / tl.maximum(l_p, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    read_i = tl.where(induct_has != 0, acc_i / tl.maximum(l_i, 1.0e-12), tl.zeros((BLOCK_H,), tl.float32))
    gs = gout * short_gain
    gp = gout * pad_gain
    gi = gout * induct_gain
    dot_gs_read = tl.sum(gs * read_s, axis=0)
    dot_gp_read = tl.sum(gp * read_p, axis=0)
    dot_gi_read = tl.sum(gi * read_i, axis=0)

    gq = tl.zeros((BLOCK_H,), tl.float32)
    gc0 = tl.full((), 0.0, tl.float32)
    gc1 = tl.full((), 0.0, tl.float32)
    gc2 = tl.full((), 0.0, tl.float32)
    gc3 = tl.full((), 0.0, tl.float32)
    g_scale = tl.full((), 0.0, tl.float32)
    g_cocktail_gain = tl.full((), 0.0, tl.float32)

    for j in range(0, N):
        bank_pos = tl.load(BankIdx + bidx * N + j).to(tl.int64)
        valid = tl.load(BankValid + bidx * N + j) != 0
        if MODE == 0:
            a = valid & (bank_pos <= tidx)
            ia = valid & (bank_pos < tidx)
        else:
            a = valid
            ia = valid & (bank_pos < T - 1)

        k = tl.load(KB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wb = tl.load(WB + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        wbn = tl.load(WBNext + (bidx * N + j) * H + h, mask=hmask, other=0.0).to(tl.float32)
        dot_raw = tl.sum(q * k, axis=0)
        short_score = dot_raw * scale + tl.load(KeyBias + bidx * N + j).to(tl.float32) + tl.load(Adm + bidx * N + j).to(tl.float32)
        cb0 = tl.load(CB + (bidx * N + j) * 4 + 0).to(tl.float32)
        cb1 = tl.load(CB + (bidx * N + j) * 4 + 1).to(tl.float32)
        cb2 = tl.load(CB + (bidx * N + j) * 4 + 2).to(tl.float32)
        cb3 = tl.load(CB + (bidx * N + j) * 4 + 3).to(tl.float32)
        cocktail_dot = c0 * cb0 + c1 * cb1 + c2 * cb2 + c3 * cb3
        perm_score = short_score + cocktail_dot * cocktail_gain

        ws = tl.where(a, tl.exp(short_score - m_s) / tl.maximum(l_s, 1.0e-12), 0.0)
        wp = tl.where(a, tl.exp(perm_score - m_p) / tl.maximum(l_p, 1.0e-12), 0.0)
        wi = tl.where(ia, tl.exp(short_score - m_i) / tl.maximum(l_i, 1.0e-12), 0.0)

        ds = ws * (tl.sum(gs * wb, axis=0) - dot_gs_read)
        dp = wp * (tl.sum(gp * wb, axis=0) - dot_gp_read)
        di = wi * (tl.sum(gi * wbn, axis=0) - dot_gi_read)
        d_short_score = ds + dp + di

        tl.atomic_add(GradWB + (bidx * N + j) * H + h, ws * gs + wp * gp, sem="relaxed", mask=hmask)
        tl.atomic_add(GradWBNext + (bidx * N + j) * H + h, wi * gi, sem="relaxed", mask=hmask)
        tl.atomic_add(GradKB + (bidx * N + j) * H + h, d_short_score * q * scale, sem="relaxed", mask=hmask)
        gq += d_short_score * k * scale
        tl.atomic_add(GradKeyBias + bidx * N + j, d_short_score, sem="relaxed")
        tl.atomic_add(GradAdm + bidx * N + j, d_short_score, sem="relaxed")
        g_scale += d_short_score * dot_raw

        gc0 += dp * cocktail_gain * cb0
        gc1 += dp * cocktail_gain * cb1
        gc2 += dp * cocktail_gain * cb2
        gc3 += dp * cocktail_gain * cb3
        tl.atomic_add(GradCB + (bidx * N + j) * 4 + 0, dp * cocktail_gain * c0, sem="relaxed")
        tl.atomic_add(GradCB + (bidx * N + j) * 4 + 1, dp * cocktail_gain * c1, sem="relaxed")
        tl.atomic_add(GradCB + (bidx * N + j) * 4 + 2, dp * cocktail_gain * c2, sem="relaxed")
        tl.atomic_add(GradCB + (bidx * N + j) * 4 + 3, dp * cocktail_gain * c3, sem="relaxed")
        g_cocktail_gain += dp * cocktail_dot

    tl.store(GradQ + (bidx * T + tidx) * H + h, gq, mask=hmask)
    tl.store(GradCocktail + (bidx * T + tidx) * 4 + 0, gc0)
    tl.store(GradCocktail + (bidx * T + tidx) * 4 + 1, gc1)
    tl.store(GradCocktail + (bidx * T + tidx) * 4 + 2, gc2)
    tl.store(GradCocktail + (bidx * T + tidx) * 4 + 3, gc3)
    part = bidx * T + tidx
    tl.store(GradScaleParts + part, g_scale)
    tl.store(GradShortGainParts + part, tl.sum(gout * read_s, axis=0))
    tl.store(GradPadGainParts + part, tl.sum(gout * read_p, axis=0))
    tl.store(GradInductGainParts + part, tl.sum(gout * read_i, axis=0))
    tl.store(GradCocktailGainParts + part, g_cocktail_gain)


def _admitted_three_way_eager(
    read_state: Tensor,
    memory_key: Tensor,
    write_memory: Tensor,
    next_write_memory: Tensor,
    read_cocktail: Tensor,
    memory_cocktail: Tensor,
    key_bias: Tensor,
    admission_gate: Tensor,
    scale: Tensor,
    allow: Tensor,
    induct_allow: Tensor,
    has_elig: Tensor,
    induct_elig: Tensor,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
) -> Tensor:
    out_dtype = read_state.dtype
    compute_dtype = _reference_compute_dtype(out_dtype)
    read_state = read_state.to(compute_dtype)
    memory_key = memory_key.to(compute_dtype)
    write_memory = write_memory.to(compute_dtype)
    next_write_memory = next_write_memory.to(compute_dtype)
    read_cocktail = read_cocktail.to(compute_dtype)
    memory_cocktail = memory_cocktail.to(compute_dtype)
    key_bias = key_bias.to(compute_dtype)
    admission_gate = admission_gate.to(compute_dtype)
    scale = scale.to(compute_dtype)
    short_gain = short_gain.to(compute_dtype)
    pad_gain = pad_gain.to(compute_dtype)
    induct_gain = induct_gain.to(compute_dtype)
    cocktail_gain = cocktail_gain.to(compute_dtype)

    compat = torch.bmm(read_state, memory_key.transpose(1, 2)) * scale
    cocktail_compat = torch.bmm(read_cocktail, memory_cocktail.transpose(1, 2)) * cocktail_gain
    content = compat + key_bias.unsqueeze(1)
    short_scores = content + admission_gate.unsqueeze(1)
    perm_scores = content + cocktail_compat + admission_gate.unsqueeze(1)

    def read(scores: Tensor, values: Tensor, mask: Tensor, elig: Tensor) -> Tensor:
        scores = scores.masked_fill(~mask, float("-inf"))
        scores = scores.masked_fill(~elig.unsqueeze(-1), 0.0)
        weights = F.softmax(scores, dim=-1)
        weights = torch.where(elig.unsqueeze(-1), weights, torch.zeros_like(weights))
        return torch.bmm(weights, values)

    out = (
        short_gain * read(short_scores, write_memory, allow, has_elig)
        + pad_gain * read(perm_scores, write_memory, allow, has_elig)
        + induct_gain * read(short_scores, next_write_memory, induct_allow, induct_elig)
    )
    return out.to(out_dtype)


class AdmittedThreeWayReadTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        read_state: Tensor,
        memory_key: Tensor,
        write_memory: Tensor,
        next_write_memory: Tensor,
        read_cocktail: Tensor,
        memory_cocktail: Tensor,
        key_bias: Tensor,
        admission_gate: Tensor,
        scale: Tensor,
        allow: Tensor,
        induct_allow: Tensor,
        has_elig: Tensor,
        induct_elig: Tensor,
        short_gain: Tensor,
        pad_gain: Tensor,
        induct_gain: Tensor,
        cocktail_gain: Tensor,
    ) -> Tensor:
        _check_supported_dtype(read_state.dtype, "read_state")
        if not read_state.is_cuda:
            raise TypeError("admitted three-way Triton read requires CUDA tensors")
        B, T, H = read_state.shape
        N = memory_key.shape[1]
        if read_cocktail.shape[-1] != 4 or memory_cocktail.shape[-1] != 4:
            raise ValueError("admitted three-way Triton read expects read/memory cocktail width 4")
        block = _block_h(H)
        out = torch.empty((B, T, H), device=read_state.device, dtype=read_state.dtype)
        _dabsn_admitted_three_way_fwd[(B * T,)](
            read_state.contiguous(), memory_key.contiguous(), write_memory.contiguous(), next_write_memory.contiguous(),
            read_cocktail.contiguous(), memory_cocktail.contiguous(), key_bias.contiguous(), admission_gate.contiguous(),
            allow.contiguous(), induct_allow.contiguous(), has_elig.contiguous(), induct_elig.contiguous(),
            out,
            scale.contiguous(),
            short_gain.contiguous(),
            pad_gain.contiguous(),
            induct_gain.contiguous(),
            cocktail_gain.contiguous(),
            T, N, H,
            BLOCK_H=block,
            num_warps=_fused_num_warps(block),
        )
        ctx.save_for_backward(
            read_state, memory_key, write_memory, next_write_memory, read_cocktail, memory_cocktail, key_bias, admission_gate, scale,
            allow, induct_allow, has_elig, induct_elig,
            short_gain, pad_gain, induct_gain, cocktail_gain,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        saved = ctx.saved_tensors
        read_state, memory_key, write_memory, next_write_memory, read_cocktail, memory_cocktail, key_bias, admission_gate, scale, allow, induct_allow, has_elig, induct_elig, short_gain, pad_gain, induct_gain, cocktail_gain = saved
        if not grad_out.is_cuda:
            raise RuntimeError("admitted three-way Triton backward requires CUDA grad_out")
        B, T, H = read_state.shape
        N = memory_key.shape[1]
        block = _block_h(H)
        grad_read_state = torch.empty((B, T, H), device=read_state.device, dtype=torch.float32)
        grad_memory_key = torch.zeros((B, N, H), device=memory_key.device, dtype=torch.float32)
        grad_write_memory = torch.zeros((B, N, H), device=write_memory.device, dtype=torch.float32)
        grad_next_write_memory = torch.zeros((B, N, H), device=next_write_memory.device, dtype=torch.float32)
        grad_read_cocktail = torch.empty((B, T, read_cocktail.shape[-1]), device=read_cocktail.device, dtype=torch.float32)
        grad_memory_cocktail = torch.zeros((B, N, memory_cocktail.shape[-1]), device=memory_cocktail.device, dtype=torch.float32)
        grad_key_bias = torch.zeros((B, N), device=key_bias.device, dtype=torch.float32)
        grad_admission_gate = torch.zeros((B, N), device=admission_gate.device, dtype=torch.float32)
        grad_scale_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_short_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_pad_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_induct_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_cocktail_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        _dabsn_admitted_three_way_bwd[(B * T,)](
            read_state.contiguous(), memory_key.contiguous(), write_memory.contiguous(), next_write_memory.contiguous(),
            read_cocktail.contiguous(), memory_cocktail.contiguous(), key_bias.contiguous(), admission_gate.contiguous(),
            allow.contiguous(), induct_allow.contiguous(), has_elig.contiguous(), induct_elig.contiguous(),
            grad_out.contiguous(),
            grad_read_state, grad_memory_key, grad_write_memory, grad_next_write_memory, grad_read_cocktail, grad_memory_cocktail, grad_key_bias, grad_admission_gate,
            grad_scale_parts, grad_short_gain_parts, grad_pad_gain_parts, grad_induct_gain_parts, grad_cocktail_gain_parts,
            scale.contiguous(),
            short_gain.contiguous(),
            pad_gain.contiguous(),
            induct_gain.contiguous(),
            cocktail_gain.contiguous(),
            T, N, H,
            BLOCK_H=block,
            num_warps=_fused_num_warps(block),
        )
        grad_scale = grad_scale_parts.sum().to(scale.dtype).reshape_as(scale)
        grad_short_gain = grad_short_gain_parts.sum().to(short_gain.dtype).reshape_as(short_gain)
        grad_pad_gain = grad_pad_gain_parts.sum().to(pad_gain.dtype).reshape_as(pad_gain)
        grad_induct_gain = grad_induct_gain_parts.sum().to(induct_gain.dtype).reshape_as(induct_gain)
        grad_cocktail_gain = grad_cocktail_gain_parts.sum().to(cocktail_gain.dtype).reshape_as(cocktail_gain)
        return (
            grad_read_state.to(read_state.dtype), grad_memory_key.to(memory_key.dtype), grad_write_memory.to(write_memory.dtype), grad_next_write_memory.to(next_write_memory.dtype),
            grad_read_cocktail.to(read_cocktail.dtype), grad_memory_cocktail.to(memory_cocktail.dtype),
            grad_key_bias.to(key_bias.dtype), grad_admission_gate.to(admission_gate.dtype), grad_scale,
            None, None, None, None,
            grad_short_gain, grad_pad_gain, grad_induct_gain, grad_cocktail_gain,
        )


class CompactAdmittedThreeWayReadTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        read_state: Tensor,
        memory_key: Tensor,
        write_memory: Tensor,
        next_write_memory: Tensor,
        read_cocktail: Tensor,
        memory_cocktail: Tensor,
        key_bias: Tensor,
        admission_gate: Tensor,
        scale: Tensor,
        bank_idx: Tensor,
        bank_valid: Tensor,
        mode: int,
        short_gain: Tensor,
        pad_gain: Tensor,
        induct_gain: Tensor,
        cocktail_gain: Tensor,
    ) -> Tensor:
        _check_supported_dtype(read_state.dtype, "read_state")
        if not read_state.is_cuda:
            raise TypeError("compact admitted three-way Triton read requires CUDA tensors")
        B, T, H = read_state.shape
        N = memory_key.shape[1]
        if read_cocktail.shape[-1] != 4 or memory_cocktail.shape[-1] != 4:
            raise ValueError("compact admitted three-way read expects cocktail width 4")
        if bank_idx.shape != bank_valid.shape or bank_idx.shape != (B, N):
            raise ValueError("bank_idx/bank_valid must have shape [B,N]")
        block = _block_h(H)
        out = torch.empty((B, T, H), device=read_state.device, dtype=read_state.dtype)
        bank_count = bank_valid.to(torch.int32).sum(dim=1).contiguous()  # Device count for the front-packed bank.
        _dabsn_admitted_three_way_compact_fwd[(B * T,)](
            read_state.contiguous(), memory_key.contiguous(), write_memory.contiguous(), next_write_memory.contiguous(),
            read_cocktail.contiguous(), memory_cocktail.contiguous(), key_bias.contiguous(), admission_gate.contiguous(),
            bank_idx.contiguous(), bank_valid.contiguous(), bank_count, out,
            scale.contiguous(),
            short_gain.contiguous(),
            pad_gain.contiguous(),
            induct_gain.contiguous(),
            cocktail_gain.contiguous(),
            T, N, H,
            MODE=int(mode),
            BLOCK_H=block,
            num_warps=_fused_num_warps(block),
        )
        ctx.save_for_backward(
            read_state, memory_key, write_memory, next_write_memory, read_cocktail, memory_cocktail, key_bias, admission_gate,
            scale, bank_idx, bank_valid, short_gain, pad_gain, induct_gain, cocktail_gain,
        )
        ctx.mode = int(mode)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        saved = ctx.saved_tensors
        read_state, memory_key, write_memory, next_write_memory, read_cocktail, memory_cocktail, key_bias, admission_gate, scale, bank_idx, bank_valid, short_gain, pad_gain, induct_gain, cocktail_gain = saved
        if not grad_out.is_cuda:
            raise RuntimeError("compact admitted three-way Triton backward requires CUDA grad_out")
        B, T, H = read_state.shape
        N = memory_key.shape[1]
        block = _block_h(H)
        grad_read_state = torch.empty((B, T, H), device=read_state.device, dtype=torch.float32)
        grad_memory_key = torch.zeros((B, N, H), device=memory_key.device, dtype=torch.float32)
        grad_write_memory = torch.zeros((B, N, H), device=write_memory.device, dtype=torch.float32)
        grad_next_write_memory = torch.zeros((B, N, H), device=next_write_memory.device, dtype=torch.float32)
        grad_read_cocktail = torch.empty((B, T, read_cocktail.shape[-1]), device=read_cocktail.device, dtype=torch.float32)
        grad_memory_cocktail = torch.zeros((B, N, memory_cocktail.shape[-1]), device=memory_cocktail.device, dtype=torch.float32)
        grad_key_bias = torch.zeros((B, N), device=key_bias.device, dtype=torch.float32)
        grad_admission_gate = torch.zeros((B, N), device=admission_gate.device, dtype=torch.float32)
        grad_scale_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_short_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_pad_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_induct_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_cocktail_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        bank_count = bank_valid.to(torch.int32).sum(dim=1).contiguous()  # Device count for the front-packed bank.
        _dabsn_admitted_three_way_compact_bwd[(B * T,)](
            read_state.contiguous(), memory_key.contiguous(), write_memory.contiguous(), next_write_memory.contiguous(),
            read_cocktail.contiguous(), memory_cocktail.contiguous(), key_bias.contiguous(), admission_gate.contiguous(),
            bank_idx.contiguous(), bank_valid.contiguous(), bank_count, grad_out.contiguous(),
            grad_read_state, grad_memory_key, grad_write_memory, grad_next_write_memory, grad_read_cocktail, grad_memory_cocktail, grad_key_bias, grad_admission_gate,
            grad_scale_parts, grad_short_gain_parts, grad_pad_gain_parts, grad_induct_gain_parts, grad_cocktail_gain_parts,
            scale.contiguous(),
            short_gain.contiguous(),
            pad_gain.contiguous(),
            induct_gain.contiguous(),
            cocktail_gain.contiguous(),
            T, N, H,
            MODE=int(ctx.mode),
            BLOCK_H=block,
            num_warps=_fused_num_warps(block),
        )
        grad_scale = grad_scale_parts.sum().to(scale.dtype).reshape_as(scale)
        grad_short_gain = grad_short_gain_parts.sum().to(short_gain.dtype).reshape_as(short_gain)
        grad_pad_gain = grad_pad_gain_parts.sum().to(pad_gain.dtype).reshape_as(pad_gain)
        grad_induct_gain = grad_induct_gain_parts.sum().to(induct_gain.dtype).reshape_as(induct_gain)
        grad_cocktail_gain = grad_cocktail_gain_parts.sum().to(cocktail_gain.dtype).reshape_as(cocktail_gain)
        return (
            grad_read_state.to(read_state.dtype), grad_memory_key.to(memory_key.dtype), grad_write_memory.to(write_memory.dtype), grad_next_write_memory.to(next_write_memory.dtype),
            grad_read_cocktail.to(read_cocktail.dtype), grad_memory_cocktail.to(memory_cocktail.dtype),
            grad_key_bias.to(key_bias.dtype), grad_admission_gate.to(admission_gate.dtype), grad_scale,
            None, None, None,
            grad_short_gain, grad_pad_gain, grad_induct_gain, grad_cocktail_gain,
        )


class CompactFlashAdmittedThreeWayReadTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        read_state: Tensor,
        memory_key: Tensor,
        write_memory: Tensor,
        next_write_memory: Tensor,
        read_cocktail: Tensor,
        memory_cocktail: Tensor,
        key_bias: Tensor,
        admission_gate: Tensor,
        scale: Tensor,
        bank_idx: Tensor,
        bank_valid: Tensor,
        mode: int,
        short_gain: Tensor,
        pad_gain: Tensor,
        induct_gain: Tensor,
        cocktail_gain: Tensor,
    ) -> Tensor:
        _check_supported_dtype(read_state.dtype, "read_state")
        if not read_state.is_cuda:
            raise TypeError("compact flash admitted three-way Triton read requires CUDA tensors")
        B, T, H = read_state.shape
        N = memory_key.shape[1]
        if read_cocktail.shape[-1] != 4 or memory_cocktail.shape[-1] != 4:
            raise ValueError("compact flash admitted three-way read expects cocktail width 4")
        if bank_idx.shape != bank_valid.shape or bank_idx.shape != (B, N):
            raise ValueError("bank_idx/bank_valid must have shape [B,N]")
        block_m = int(_os.environ.get("DABSN_FLASH_QUERY_BLOCK", "16"))
        block_n = int(_os.environ.get("DABSN_FLASH_KEY_BLOCK", "64"))
        block_dot = int(_os.environ.get("DABSN_FLASH_DOT_BLOCK", "64"))
        block_v = int(_os.environ.get("DABSN_FLASH_VALUE_BLOCK", "64"))
        block_m = max(1, min(block_m, 64))
        block_n = max(1, min(block_n, 128))
        block_dot = max(16, min(block_dot, _block_h(min(H, 128))))
        block_v = max(16, min(block_v, _block_h(min(H, 128))))
        # Keep QK and AV contractions on tensor cores. tf32x3 is the balanced
        # default; ieee and tf32 are selectable through DABSN_FLASH_PREC.
        prec = _os.environ.get("DABSN_FLASH_PREC", "tf32x3")
        out = torch.empty((B, T, H), device=read_state.device, dtype=read_state.dtype)
        grid = (triton.cdiv(T, block_m), triton.cdiv(H, block_v), B)
        bank_count = bank_valid.to(torch.int32).sum(dim=1).contiguous()  # Device count for the front-packed bank.
        _dabsn_admitted_three_way_compact_flash_fwd[grid](
            read_state.contiguous(), memory_key.contiguous(), write_memory.contiguous(), next_write_memory.contiguous(),
            read_cocktail.contiguous(), memory_cocktail.contiguous(), key_bias.contiguous(), admission_gate.contiguous(),
            bank_idx.contiguous(), bank_valid.contiguous(), bank_count, out,
            scale.contiguous(),
            short_gain.contiguous(),
            pad_gain.contiguous(),
            induct_gain.contiguous(),
            cocktail_gain.contiguous(),
            T, N, H,
            MODE=int(mode),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_DOT=block_dot,
            BLOCK_V=block_v,
            PREC=prec,
            num_warps=int(_os.environ.get("DABSN_FLASH_WARPS", "4")),
            num_stages=int(_os.environ.get("DABSN_FLASH_STAGES", "3")),
        )
        ctx.save_for_backward(
            read_state, memory_key, write_memory, next_write_memory, read_cocktail, memory_cocktail, key_bias, admission_gate,
            scale, bank_idx, bank_valid, short_gain, pad_gain, induct_gain, cocktail_gain,
        )
        ctx.mode = int(mode)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        saved = ctx.saved_tensors
        read_state, memory_key, write_memory, next_write_memory, read_cocktail, memory_cocktail, key_bias, admission_gate, scale, bank_idx, bank_valid, short_gain, pad_gain, induct_gain, cocktail_gain = saved
        if not grad_out.is_cuda:
            raise RuntimeError("compact flash admitted three-way Triton backward requires CUDA grad_out")
        B, T, H = read_state.shape
        N = memory_key.shape[1]
        block = _block_h(H)
        grad_read_state = torch.empty((B, T, H), device=read_state.device, dtype=torch.float32)
        grad_memory_key = torch.zeros((B, N, H), device=memory_key.device, dtype=torch.float32)
        grad_write_memory = torch.zeros((B, N, H), device=write_memory.device, dtype=torch.float32)
        grad_next_write_memory = torch.zeros((B, N, H), device=next_write_memory.device, dtype=torch.float32)
        grad_read_cocktail = torch.empty((B, T, read_cocktail.shape[-1]), device=read_cocktail.device, dtype=torch.float32)
        grad_memory_cocktail = torch.zeros((B, N, memory_cocktail.shape[-1]), device=memory_cocktail.device, dtype=torch.float32)
        grad_key_bias = torch.zeros((B, N), device=key_bias.device, dtype=torch.float32)
        grad_admission_gate = torch.zeros((B, N), device=admission_gate.device, dtype=torch.float32)
        grad_scale_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_short_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_pad_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_induct_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        grad_cocktail_gain_parts = torch.empty((B, T), device=read_state.device, dtype=torch.float32)
        bank_count = bank_valid.to(torch.int32).sum(dim=1).contiguous()  # Device count for the front-packed bank.
        _dabsn_admitted_three_way_compact_bwd[(B * T,)](
            read_state.contiguous(), memory_key.contiguous(), write_memory.contiguous(), next_write_memory.contiguous(),
            read_cocktail.contiguous(), memory_cocktail.contiguous(), key_bias.contiguous(), admission_gate.contiguous(),
            bank_idx.contiguous(), bank_valid.contiguous(), bank_count, grad_out.contiguous(),
            grad_read_state, grad_memory_key, grad_write_memory, grad_next_write_memory, grad_read_cocktail, grad_memory_cocktail, grad_key_bias, grad_admission_gate,
            grad_scale_parts, grad_short_gain_parts, grad_pad_gain_parts, grad_induct_gain_parts, grad_cocktail_gain_parts,
            scale.contiguous(),
            short_gain.contiguous(),
            pad_gain.contiguous(),
            induct_gain.contiguous(),
            cocktail_gain.contiguous(),
            T, N, H,
            MODE=int(ctx.mode),
            BLOCK_H=block,
            num_warps=_fused_num_warps(block),
        )
        grad_scale = grad_scale_parts.sum().to(scale.dtype).reshape_as(scale)
        grad_short_gain = grad_short_gain_parts.sum().to(short_gain.dtype).reshape_as(short_gain)
        grad_pad_gain = grad_pad_gain_parts.sum().to(pad_gain.dtype).reshape_as(pad_gain)
        grad_induct_gain = grad_induct_gain_parts.sum().to(induct_gain.dtype).reshape_as(induct_gain)
        grad_cocktail_gain = grad_cocktail_gain_parts.sum().to(cocktail_gain.dtype).reshape_as(cocktail_gain)
        return (
            grad_read_state.to(read_state.dtype), grad_memory_key.to(memory_key.dtype), grad_write_memory.to(write_memory.dtype), grad_next_write_memory.to(next_write_memory.dtype),
            grad_read_cocktail.to(read_cocktail.dtype), grad_memory_cocktail.to(memory_cocktail.dtype),
            grad_key_bias.to(key_bias.dtype), grad_admission_gate.to(admission_gate.dtype), grad_scale,
            None, None, None,
            grad_short_gain, grad_pad_gain, grad_induct_gain, grad_cocktail_gain,
        )


def admitted_three_way_read_trainable(
    read_state: Tensor,
    memory_key: Tensor,
    write_memory: Tensor,
    next_write_memory: Tensor,
    read_cocktail: Tensor,
    memory_cocktail: Tensor,
    key_bias: Tensor,
    admission_gate: Tensor,
    scale: Tensor,
    allow: Tensor,
    induct_allow: Tensor,
    has_elig: Tensor,
    induct_elig: Tensor,
    *,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
) -> Tensor:
    return AdmittedThreeWayReadTritonFunction.apply(
        read_state, memory_key, write_memory, next_write_memory, read_cocktail, memory_cocktail, key_bias, admission_gate, scale,
        allow, induct_allow, has_elig, induct_elig,
        short_gain, pad_gain, induct_gain, cocktail_gain,
    )


def _frontpack_admitted_bank(memory_key, write_memory, next_write_memory, memory_cocktail,
                             key_bias, admission_gate, bank_idx, bank_valid):
    """Move live entries to the front for count-bounded kernel traversal.

    The order comes from a detached mask, while tensor gathers remain tracked by
    autograd so backward scatters gradients to their original bank positions.
    Query-side tensors and scalar controls are unchanged.
    """
    order = torch.argsort(bank_valid.detach().to(torch.int32), dim=1, descending=True, stable=True)
    idx = order.unsqueeze(-1)
    mk = torch.gather(memory_key, 1, idx.expand(-1, -1, memory_key.shape[-1]))
    wm = torch.gather(write_memory, 1, idx.expand(-1, -1, write_memory.shape[-1]))
    nwm = torch.gather(next_write_memory, 1, idx.expand(-1, -1, next_write_memory.shape[-1]))
    mc = torch.gather(memory_cocktail, 1, idx.expand(-1, -1, memory_cocktail.shape[-1]))
    kb = torch.gather(key_bias, 1, order)
    ag = torch.gather(admission_gate, 1, order)
    bi = torch.gather(bank_idx, 1, order)
    bv = torch.gather(bank_valid, 1, order)
    return mk, wm, nwm, mc, kb, ag, bi, bv


def admitted_three_way_read_compact_trainable(
    read_state: Tensor,
    memory_key: Tensor,
    write_memory: Tensor,
    next_write_memory: Tensor,
    read_cocktail: Tensor,
    memory_cocktail: Tensor,
    key_bias: Tensor,
    admission_gate: Tensor,
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
    mode_id = 0 if mode == "seq" else 1
    memory_key, write_memory, next_write_memory, memory_cocktail, key_bias, admission_gate, bank_idx, bank_valid = \
        _frontpack_admitted_bank(memory_key, write_memory, next_write_memory, memory_cocktail,
                                 key_bias, admission_gate, bank_idx, bank_valid)
    return CompactAdmittedThreeWayReadTritonFunction.apply(
        read_state, memory_key, write_memory, next_write_memory, read_cocktail, memory_cocktail,
        key_bias, admission_gate, scale, bank_idx, bank_valid, mode_id,
        short_gain, pad_gain, induct_gain, cocktail_gain,
    )


def admitted_three_way_read_compact_flash_trainable(
    read_state: Tensor,
    memory_key: Tensor,
    write_memory: Tensor,
    next_write_memory: Tensor,
    read_cocktail: Tensor,
    memory_cocktail: Tensor,
    key_bias: Tensor,
    admission_gate: Tensor,
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
    """Trainable compact admitted read using the production flash forward.

    Forward is the same tiled online-softmax kernel used by inference. Backward is
    the compact Triton gradient kernel, so gradients are real and no dense
    [T,N] score/mask tensor is materialized. The backward is still one-query
    programs over the bank, so reports must not claim it is flash-tiled backward.
    """
    mode_id = 0 if mode == "seq" else 1
    memory_key, write_memory, next_write_memory, memory_cocktail, key_bias, admission_gate, bank_idx, bank_valid = \
        _frontpack_admitted_bank(memory_key, write_memory, next_write_memory, memory_cocktail,
                                 key_bias, admission_gate, bank_idx, bank_valid)
    return CompactFlashAdmittedThreeWayReadTritonFunction.apply(
        read_state, memory_key, write_memory, next_write_memory, read_cocktail, memory_cocktail,
        key_bias, admission_gate, scale, bank_idx, bank_valid, mode_id,
        short_gain, pad_gain, induct_gain, cocktail_gain,
    )


def admitted_three_way_read_compact_infer(
    read_state: Tensor,
    memory_key: Tensor,
    write_memory: Tensor,
    next_write_memory: Tensor,
    read_cocktail: Tensor,
    memory_cocktail: Tensor,
    key_bias: Tensor,
    admission_gate: Tensor,
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
    """Forward-only exact compact admitted read for eval/inference.

    This is the large-bank production forward path. It is task-agnostic: callers
    pass the admitted bank and seq/field eligibility mode, and the kernel streams
    query/key/value tiles with online softmax. Training still uses the autograd
    compact kernel until a matching tiled backward is implemented.
    """
    _check_supported_dtype(read_state.dtype, "read_state")
    if not read_state.is_cuda:
        raise TypeError("compact admitted three-way infer read requires CUDA tensors")
    B, T, H = read_state.shape
    N = memory_key.shape[1]
    if read_cocktail.shape[-1] != 4 or memory_cocktail.shape[-1] != 4:
        raise ValueError("compact admitted three-way infer read expects cocktail width 4")
    if bank_idx.shape != bank_valid.shape or bank_idx.shape != (B, N):
        raise ValueError("bank_idx/bank_valid must have shape [B,N]")
    mode_id = 0 if mode == "seq" else 1
    block_m = int(_os.environ.get("DABSN_FLASH_QUERY_BLOCK", "16"))
    block_n = int(_os.environ.get("DABSN_FLASH_KEY_BLOCK", "64"))
    block_dot = int(_os.environ.get("DABSN_FLASH_DOT_BLOCK", "64"))
    block_v = int(_os.environ.get("DABSN_FLASH_VALUE_BLOCK", "64"))
    block_m = max(1, min(block_m, 64))
    block_n = max(1, min(block_n, 128))
    block_dot = max(16, min(block_dot, _block_h(min(H, 128))))
    block_v = max(16, min(block_v, _block_h(min(H, 128))))
    # Tensor-core contractions (see trainable launcher note). tf32x3 default.
    prec = _os.environ.get("DABSN_FLASH_PREC", "tf32x3")
    out = torch.empty((B, T, H), device=read_state.device, dtype=read_state.dtype)
    grid = (triton.cdiv(T, block_m), triton.cdiv(H, block_v), B)
    # Front-pack live entries for count-bounded traversal of any bank mask.
    memory_key, write_memory, next_write_memory, memory_cocktail, key_bias, admission_gate, bank_idx, bank_valid = \
        _frontpack_admitted_bank(memory_key, write_memory, next_write_memory, memory_cocktail,
                                 key_bias, admission_gate, bank_idx, bank_valid)
    bank_count = bank_valid.to(torch.int32).sum(dim=1).contiguous()  # Device count for the front-packed bank.
    _dabsn_admitted_three_way_compact_flash_fwd[grid](
        read_state.contiguous(), memory_key.contiguous(), write_memory.contiguous(), next_write_memory.contiguous(),
        read_cocktail.contiguous(), memory_cocktail.contiguous(), key_bias.contiguous(), admission_gate.contiguous(),
        bank_idx.contiguous(), bank_valid.contiguous(), bank_count, out,
        scale.contiguous(),
        short_gain.contiguous(),
        pad_gain.contiguous(),
        induct_gain.contiguous(),
        cocktail_gain.contiguous(),
        T, N, H,
        MODE=mode_id,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DOT=block_dot,
        BLOCK_V=block_v,
        PREC=prec,
        num_warps=int(_os.environ.get("DABSN_FLASH_WARPS", "4")),
        num_stages=int(_os.environ.get("DABSN_FLASH_STAGES", "3")),
    )
    return out


def dense_bmm_three_way_read(
    read_state: Tensor,
    memory_key: Tensor,
    write_memory: Tensor,
    next_write_memory: Tensor,
    read_cocktail: Tensor,
    memory_cocktail: Tensor,
    key_bias: Tensor,
    admission_gate: Tensor,
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
    """Dense tensor-core implementation of the three-way admitted read.

    It uses the same equations as the compact flash kernel for sequence and field
    geometry. Near-full banks route here because native ``torch.bmm`` is more
    efficient than a count-bounded member loop at high density.
    """
    B, T, H = read_state.shape
    N = memory_key.shape[1]
    scale_f = scale.to(torch.float32)
    q = read_state.to(torch.float32)
    kb = memory_key.to(torch.float32)
    compat = torch.bmm(q, kb.transpose(1, 2)) * scale_f                       # [B,T,N]
    key_adm = (key_bias.to(torch.float32) + admission_gate.to(torch.float32)).unsqueeze(1)  # [B,1,N]
    short_scores = compat + key_adm
    cc = torch.bmm(read_cocktail.to(torch.float32), memory_cocktail.to(torch.float32).transpose(1, 2))
    perm_scores = short_scores + cc * cocktail_gain.to(torch.float32)
    valid = bank_valid.unsqueeze(1)                                           # [B,1,N]
    bank_pos = bank_idx.unsqueeze(1)                                          # [B,1,N]
    if mode == "seq":
        qpos = torch.arange(T, device=read_state.device).view(1, T, 1)
        allow = valid & (bank_pos <= qpos)
        induct_allow = valid & (bank_pos < qpos)
    else:
        allow = valid.expand(B, T, N)
        induct_allow = valid & (bank_pos < (T - 1))

    def rd(scores: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        elig = mask.any(dim=-1, keepdim=True)                                # [B,T,1]
        w = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        w = torch.where(elig, torch.nan_to_num(w), torch.zeros_like(w))
        return torch.bmm(w, values.to(torch.float32))

    short = rd(short_scores, write_memory, allow)
    perm = rd(perm_scores, write_memory, allow)
    induct = rd(short_scores, next_write_memory, induct_allow)
    out = (short_gain.to(torch.float32) * short
           + pad_gain.to(torch.float32) * perm
           + induct_gain.to(torch.float32) * induct)
    return out.to(read_state.dtype)


# Per-device and shape crossover-density cache. The dispatcher measures where
# dense BMM overtakes the count-bounded flash kernel and reuses that result.
_READ_CROSSOVER_CACHE: dict[tuple, float] = {}


def measure_read_crossover(
    dev: torch.device,
    dtype: torch.dtype,
    N: int,
    H: int,
    *,
    T: int | None = None,
    B: int = 4,
    prec: str = "tf32x3",
    mode: str = "seq",
    densities: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    iters: int = 8,
    warmup: int = 3,
) -> float:
    """Empirically find the density at which the dense bmm read overtakes the flash
    kernel for this (device, shape, dtype, prec, mode). Returns the crossover density:
    read at or above it should route to bmm, below it to the flash kernel. If the flash
    kernel wins at every density, returns 1.0 (never switch); if bmm always wins, 0.0.

    Both runtime paths are timed directly so the threshold reflects the target
    device rather than a fixed density assumption.
    """
    if dev.type != "cuda" or not torch.cuda.is_available():
        return 1.0  # The flash path is CUDA-only; CPU execution uses BMM.
    T = int(T if T is not None else max(64, min(N, 256)))
    _cpu_rng = torch.get_rng_state()
    _cuda_rng = torch.cuda.get_rng_state_all()
    torch.manual_seed(11)
    q = torch.randn(B, T, H, device=dev, dtype=dtype)
    kb = torch.randn(B, N, H, device=dev, dtype=dtype)
    wb = torch.randn(B, N, H, device=dev, dtype=dtype)
    wbn = torch.randn(B, N, H, device=dev, dtype=dtype)
    rc = F.normalize(torch.randn(B, T, 4, device=dev, dtype=dtype), dim=-1)
    cb = F.normalize(torch.randn(B, N, 4, device=dev, dtype=dtype), dim=-1)
    kbias = torch.randn(B, N, device=dev, dtype=dtype) * 0.1
    adm = torch.randn(B, N, device=dev, dtype=dtype) * 0.1
    bank_idx = torch.randint(0, max(T, 1), (B, N), device=dev).sort(dim=-1).values.to(torch.long)
    gains = dict(
        short_gain=torch.tensor(0.8, device=dev, dtype=dtype),
        pad_gain=torch.tensor(0.6, device=dev, dtype=dtype),
        induct_gain=torch.tensor(0.4, device=dev, dtype=dtype),
        cocktail_gain=torch.tensor(1.1, device=dev, dtype=dtype),
    )
    scale = torch.tensor(1.7, device=dev, dtype=dtype)

    def _time(run) -> float:
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            run()
        e.record()
        torch.cuda.synchronize()
        return float(s.elapsed_time(e)) / iters

    _prev_prec = _os.environ.get("DABSN_FLASH_PREC")
    _os.environ["DABSN_FLASH_PREC"] = prec
    crossover = 1.0
    try:
        with torch.no_grad():
            for d in densities:
                bank_valid = torch.rand(B, N, device=dev) < d
                ms_flash = _time(lambda bv=bank_valid: admitted_three_way_read_compact_infer(
                    q, kb, wb, wbn, rc, cb, kbias, adm, scale, bank_idx, bv,
                    mode=mode, **gains))
                ms_dense = _time(lambda bv=bank_valid: dense_bmm_three_way_read(
                    q, kb, wb, wbn, rc, cb, kbias, adm, scale, bank_idx, bv,
                    mode=mode, **gains))
                if ms_dense < ms_flash:
                    crossover = float(d)  # first density where dense wins -> switch here
                    break
    finally:
        if _prev_prec is None:
            _os.environ.pop("DABSN_FLASH_PREC", None)
        else:
            _os.environ["DABSN_FLASH_PREC"] = _prev_prec
        torch.set_rng_state(_cpu_rng)
        torch.cuda.set_rng_state_all(_cuda_rng)
    return crossover


def _read_crossover_for(dev: torch.device, dtype: torch.dtype, N: int, H: int,
                        prec: str, mode: str) -> float:
    """Return a cached on-device crossover, or a pinned environment value."""
    pin = _os.environ.get("DABSN_DENSE_CROSSOVER")
    if pin is not None:
        return float(pin)
    key = (dev.type, str(dtype), int(N), int(H), prec, mode)
    hit = _READ_CROSSOVER_CACHE.get(key)
    if hit is None:
        hit = measure_read_crossover(dev, dtype, N, H, prec=prec, mode=mode)
        _READ_CROSSOVER_CACHE[key] = hit
    return hit


def admitted_three_way_read_dispatch(
    read_state: Tensor,
    memory_key: Tensor,
    write_memory: Tensor,
    next_write_memory: Tensor,
    read_cocktail: Tensor,
    memory_cocktail: Tensor,
    key_bias: Tensor,
    admission_gate: Tensor,
    scale: Tensor,
    bank_idx: Tensor,
    bank_valid: Tensor,
    *,
    mode: str,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
    return_backend: bool = False,
) -> Tensor | tuple[Tensor, str]:
    """Forward-only admitted read with density-aware sparse/dense dispatch.

    Sparse banks use the count-bounded flash kernel; dense banks use the
    tensor-core BMM path. The crossover is measured and cached for each device,
    shape, dtype, precision, and geometry bucket. ``DABSN_DENSE_CROSSOVER`` can
    pin the threshold and skip calibration. The inference-only routing scalar is
    a cost heuristic, not a data-dependent tensor shape; training retains the
    device-side count-bounded path.
    """
    N = memory_key.shape[1]
    if N == 0:
        density = 0.0
    else:
        density = float(bank_valid.to(torch.float32).mean().item())
    prec = _os.environ.get("DABSN_FLASH_PREC", "tf32x3")
    crossover = _read_crossover_for(read_state.device, read_state.dtype,
                                    N, read_state.shape[-1], prec, mode)
    if density >= crossover:
        output = dense_bmm_three_way_read(
            read_state, memory_key, write_memory, next_write_memory, read_cocktail,
            memory_cocktail, key_bias, admission_gate, scale, bank_idx, bank_valid,
            mode=mode, short_gain=short_gain, pad_gain=pad_gain,
            induct_gain=induct_gain, cocktail_gain=cocktail_gain,
        )
        backend = "dense_bmm_cuda"
    else:
        output = admitted_three_way_read_compact_infer(
            read_state, memory_key, write_memory, next_write_memory, read_cocktail,
            memory_cocktail, key_bias, admission_gate, scale, bank_idx, bank_valid,
            mode=mode, short_gain=short_gain, pad_gain=pad_gain,
            induct_gain=induct_gain, cocktail_gain=cocktail_gain,
        )
        backend = "compact_flash_infer"
    return (output, backend) if return_backend else output


def dabsn_core_scan_triton_from_core(core, e_in: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Adapter for `DABSNCore` / dabsn_core core objects.

    This does not mutate or monkeypatch the core. It only reads parameters.
    """
    _check_supported_dtype(e_in.dtype, "e_in")
    e_in = e_in.contiguous()
    H = int(core.hidden_dim)

    logit_c_decay = core.logit_saturation_decay.expand(H).contiguous()
    k_c = core.k_saturation.contiguous()
    r_c = core.r_saturation.contiguous()
    logit_c_suppress = core.logit_saturation_suppress.reshape(())

    return dabsn_core_scan_triton(
        core.W(e_in).contiguous(),
        core.Wg(e_in).contiguous(),
        core.Ug.weight.contiguous(),
        core.A.weight.contiguous(),
        core.beta.contiguous(),
        core.log_kappa.contiguous(),
        core.logit_recover.contiguous(),
        core.k_s.contiguous(),
        core.k_y.contiguous(),
        core.k_b.contiguous(),
        core.k_n.contiguous(),
        core.k_bias.contiguous(),
        core.r_s.contiguous(),
        core.r_y.contiguous(),
        core.r_b.contiguous(),
        core.r_n.contiguous(),
        core.r_bias.contiguous(),
        logit_c_decay,
        k_c,
        r_c,
        logit_alpha=core.logit_alpha.reshape(()),
        log_lambda=core.log_lambda.reshape(()),
        logit_c_suppress=logit_c_suppress,
    )


def dabsn_core_scan_triton_tape_from_core(core, e_in: Tensor) -> Tuple[Tensor, ...]:
    """Core adapter for the tape-producing scan."""
    _check_supported_dtype(e_in.dtype, "e_in")
    e_in = e_in.contiguous()
    H = int(core.hidden_dim)
    logit_c_decay = core.logit_saturation_decay.expand(H).contiguous()
    k_c = core.k_saturation.contiguous()
    r_c = core.r_saturation.contiguous()
    logit_c_suppress = core.logit_saturation_suppress.reshape(())

    return dabsn_core_scan_triton_tape(
        core.W(e_in).contiguous(),
        core.Wg(e_in).contiguous(),
        core.Ug.weight.contiguous(),
        core.A.weight.contiguous(),
        core.beta.contiguous(),
        core.log_kappa.contiguous(),
        core.logit_recover.contiguous(),
        core.k_s.contiguous(),
        core.k_y.contiguous(),
        core.k_b.contiguous(),
        core.k_n.contiguous(),
        core.k_bias.contiguous(),
        core.r_s.contiguous(),
        core.r_y.contiguous(),
        core.r_b.contiguous(),
        core.r_n.contiguous(),
        core.r_bias.contiguous(),
        logit_c_decay,
        k_c,
        r_c,
        logit_alpha=core.logit_alpha.reshape(()),
        log_lambda=core.log_lambda.reshape(()),
        logit_c_suppress=logit_c_suppress,
    )


def dabsn_core_scan_trainable_from_core(core, e_in: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run Triton forward with the fused reverse-scan backward.

    Widths above the single-tile limit use chunked backward. Gradients flow
    through both input projections because their outputs enter the custom
    autograd function as graph tensors. Use
    ``core_fused_backward_parity_check`` when validating a new GPU/runtime pair.
    """
    _check_supported_dtype(e_in.dtype, "e_in")
    e_in = e_in.contiguous()
    H = int(core.hidden_dim)
    logit_c_decay = core.logit_saturation_decay.expand(H).contiguous()
    k_c = core.k_saturation.contiguous()
    r_c = core.r_saturation.contiguous()
    logit_c_suppress = core.logit_saturation_suppress.reshape(())

    return dabsn_core_scan_trainable_fused(
        core.W(e_in).contiguous(),
        core.Wg(e_in).contiguous(),
        core.Ug.weight.contiguous(),
        core.A.weight.contiguous(),
        core.beta.contiguous(),
        core.log_kappa.contiguous(),
        core.logit_recover.contiguous(),
        core.k_s.contiguous(),
        core.k_y.contiguous(),
        core.k_b.contiguous(),
        core.k_n.contiguous(),
        core.k_bias.contiguous(),
        core.r_s.contiguous(),
        core.r_y.contiguous(),
        core.r_b.contiguous(),
        core.r_n.contiguous(),
        core.r_bias.contiguous(),
        logit_c_decay,
        k_c,
        r_c,
        core.logit_alpha.reshape(()),
        core.log_lambda.reshape(()),
        logit_c_suppress,
    )
