"""Fused flash-style admitted three-way read for CUDA (forward and backward).

Exactly the canonical admitted read (``admitted.py``), computed without ever
materializing a ``[B, T, N]`` score, weight, or mask tensor:

    content    = scale * q.k + key_bias + admission
    short      = softmax_allow(content)               . writes
    permanent  = softmax_allow(content + cg * cq.ck)   . writes
    induction  = softmax_induct(content)              . next_writes
    output     = short_gain*short + pad_gain*permanent + induct_gain*induction

Admission is untouched: the caller still builds the content-dependent admitted
bank. These kernels only read it. Eligibility comes from bank positions in the
kernel: ``seq`` allows ``position <= t`` (induction ``< t``); ``field`` allows
every valid entry (induction ``position < T-1``). Rows with no eligible entry
read zero, exactly as the reference.

Two entry points share one set of kernels:

* ``admitted_three_way_read_flash`` takes an already gathered bank (normalized
  keys, writes, next writes), the ABI of the compact reads.
* ``admitted_three_way_read_tokens`` takes the per-token tensors (raw
  expression, write, cocktail, key bias + admission) and the admitted bank's
  positions. Bank rows are gathered, and query/key rows normalized, inside the
  kernels; gradients come back per token through one deterministic combine
  pass. Nothing bank-shaped is materialized.

Forward: one pass per query tile with an online softmax per read (one score tile
feeds all three), storing the three log-sum-exps and, when training, the three
per-path reads. Backward: recompute. The softmax-backward row terms
``E = rowsum(dO * O)`` come from the saved reads; a query pass forms dq and the
cocktail/scale partials; a bank pass forms dk, dv, dnext, the key-bias/admission
column sums and the bank cocktail gradient. No atomics: bitwise deterministic.
No host syncs: CUDA-graph capturable. Precision follows the framework policy:
16-bit operands with FP32 accumulation under autocast/16-bit inputs; FP32
operands follow ``torch.get_float32_matmul_precision()`` (``highest`` = IEEE,
else TF32).
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor
from ._arch import on_tensor_device

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only installs
    triton = None
    tl = None
    _HAS_TRITON = False

_INVALID = 1 << 30
_EPS = 1e-12  # F.normalize default


def _input_precision() -> str:
    return "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"


def _operand_dtype(query: Tensor) -> torch.dtype:
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    if query.dtype in (torch.float16, torch.bfloat16):
        return query.dtype
    return torch.float32


def _operand_dtype_ok(tensor: Tensor) -> bool:
    if _operand_dtype(tensor) != torch.bfloat16:
        return True
    from ._arch import bf16_kernels_supported

    return bf16_kernels_supported(tensor.device)


def flash_read_supported(query: Tensor, bank_keys: Tensor, cocktail: Tensor) -> bool:
    return (
        _HAS_TRITON
        and query.is_cuda
        and _operand_dtype_ok(query)
        and query.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and query.dim() == 3
        and bank_keys.dim() == 3
        and bank_keys.shape[1] >= 1
        and cocktail.shape[-1] == 4
    )


_DEFAULT_CONFIG = {
    "fwd": {"BM": 32, "BN": 64, "BK": 64, "BH": 128, "num_warps": 4, "num_stages": 1},
    "query": {"BM": 128, "BN": 32, "BK": 64, "BH": 128, "num_warps": 8, "num_stages": 2},
    "bank": {"BM": 128, "BN": 32, "BK": 64, "BH": 128, "num_warps": 8, "num_stages": 2},
}


def _config(kind: str, H: int) -> dict[str, int]:
    import json
    import os

    cfg = dict(_DEFAULT_CONFIG[kind])
    override = os.environ.get("DABSN_READ_FLASH_CONFIG")
    if override:
        cfg.update(json.loads(override).get(kind, {}))
    width = max(16, triton.next_power_of_2(H))
    cfg["BK"] = min(cfg["BK"], width)
    if "BH" in cfg:
        cfg["BH"] = min(cfg["BH"], width)
    return cfg


_WORKING: dict[tuple, dict[str, int]] = {}


def _shrink(cfg: dict[str, int]) -> dict[str, int] | None:
    cfg = dict(cfg)
    if cfg["num_stages"] > 1:
        cfg["num_stages"] = 1
    elif cfg.get("BH", 16) > 32:
        cfg["BH"] //= 2
    elif cfg["BN"] > 16:
        cfg["BN"] //= 2
    elif cfg["BM"] > 16:
        cfg["BM"] //= 2
    elif cfg["BK"] > 16:
        cfg["BK"] //= 2
    else:
        return None
    return cfg


def _launch(kind: str, H: int, dtype: torch.dtype, gather: bool, launch) -> None:
    """Launch with the tuned tiles, shrinking them if a GPU/dtype lacks the
    shared memory (FP32 operands need twice the 16-bit footprint). The working
    tiles are cached per kernel, width, operand dtype and bank addressing."""
    key = (kind, max(16, triton.next_power_of_2(H)), dtype, gather)
    cfg = _WORKING.get(key) or _config(kind, H)
    while True:
        try:
            launch(cfg)
        except triton.runtime.errors.OutOfResources:
            cfg = _shrink(cfg)
            if cfg is None:
                raise
            continue
        _WORKING[key] = cfg
        return


if _HAS_TRITON:

    @triton.jit
    def _bank_rows(KROW, NROW, b64, cols, cmask, N, GATHER: tl.constexpr):
        if GATHER:
            krow = tl.load(KROW + b64 * N + cols, cmask, 0)
            nrow = tl.load(NROW + b64 * N + cols, cmask, 0)
        else:
            krow = cols
            nrow = cols
        return krow, nrow

    @triton.jit
    def _inv_norm(NRM, b64, idx, mask, R, NORM: tl.constexpr):
        if NORM:
            n = tl.load(NRM + b64 * R + idx, mask, 1.0)
            return 1.0 / tl.maximum(n, 1e-12)
        else:
            return tl.full(idx.shape, 1.0, tl.float32)

    @triton.jit
    def _scores(Q, K, b64, rows, krow, rmask, cmask, T, S, H, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr):
        raw = tl.zeros((BM, BN), dtype=tl.float32)
        for h0 in range(0, H, BK):
            hk = h0 + tl.arange(0, BK)
            hm = hk < H
            q = tl.load(Q + b64 * T * H + rows[:, None] * H + hk[None, :],
                        rmask[:, None] & hm[None, :], 0.0)
            k = tl.load(K + b64 * S * H + krow[:, None] * H + hk[None, :],
                        cmask[:, None] & hm[None, :], 0.0)
            raw = tl.dot(q, tl.trans(k), raw, input_precision=IP)
        return raw

    @triton.jit
    def _scores3(Q, K, DO, V, VN, b64, rows, krow, nrow, rmask, cmask, T, S, H,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IP: tl.constexpr):
        raw = tl.zeros((BM, BN), dtype=tl.float32)
        u = tl.zeros((BM, BN), dtype=tl.float32)
        un = tl.zeros((BM, BN), dtype=tl.float32)
        for h0 in range(0, H, BK):
            hk = h0 + tl.arange(0, BK)
            hm = hk < H
            qm = rmask[:, None] & hm[None, :]
            km = cmask[:, None] & hm[None, :]
            qo = b64 * T * H + rows[:, None] * H + hk[None, :]
            ko = b64 * S * H + krow[:, None] * H + hk[None, :]
            no = b64 * S * H + nrow[:, None] * H + hk[None, :]
            q = tl.load(Q + qo, qm, 0.0)
            do = tl.load(DO + qo, qm, 0.0)
            k = tl.load(K + ko, km, 0.0)
            v = tl.load(V + ko, km, 0.0)
            vn = tl.load(VN + no, km, 0.0)
            raw = tl.dot(q, tl.trans(k), raw, input_precision=IP)
            u = tl.dot(do, tl.trans(v), u, input_precision=IP)
            un = tl.dot(do, tl.trans(vn), un, input_precision=IP)
        return raw, u, un

    @triton.jit
    def _cocktail(CQ, CK, b64, rows, krow, rmask, cmask, T, S):
        cc = tl.zeros((rows.shape[0], krow.shape[0]), dtype=tl.float32)
        for j in tl.static_range(4):
            cq = tl.load(CQ + b64 * T * 4 + rows * 4 + j, rmask, 0.0)
            ck = tl.load(CK + b64 * S * 4 + krow * 4 + j, cmask, 0.0)
            cc += cq[:, None] * ck[None, :]
        return cc

    @triton.jit
    def _limits(rows, T, SEQ: tl.constexpr):
        if SEQ:
            lim_a = rows
            lim_i = rows
        else:
            lim_a = tl.zeros_like(rows) + (1 << 30) - 1
            lim_i = tl.zeros_like(rows) + T - 1
        return lim_a, lim_i

    @triton.jit
    def _online_acc(m, l, acc, x, mask, v, IP: tl.constexpr):
        xm = tl.where(mask, x, float("-inf"))
        m_new = tl.maximum(m, tl.max(xm, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m - m_use)
        pr = tl.exp(xm - m_use[:, None])
        l = l * alpha + tl.sum(pr, 1)
        acc = tl.dot(pr.to(v.dtype), v, acc * alpha[:, None], input_precision=IP)
        return m_new, l, acc

    @triton.jit
    def _finish(m, l, acc):
        inv = tl.where(l > 0, 1.0 / l, 0.0)
        return acc * inv[:, None], tl.where(l > 0, m + tl.log(l), float("inf"))

    @triton.jit
    def _fwd_kernel(Q, QN, K, KN, V, VN, KROW, NROW, CQ, CK, KB, POS, NHI, SCALE, SG, PG, IG,
                    CG, OUT, LSE, OP, T, N, S, H, SEQ: tl.constexpr, SAVE: tl.constexpr,
                    GATHER: tl.constexpr, NORM: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr, BH: tl.constexpr, IP: tl.constexpr):
        # Single pass, online softmax: one score tile feeds all three reads.
        pid = tl.program_id(0)
        nh = tl.cdiv(H, BH)
        nm = tl.cdiv(T, BM)
        hb = pid % nh
        mb = (pid // nh) % nm
        b64 = (pid // (nh * nm)).to(tl.int64)
        rows = mb * BM + tl.arange(0, BM)
        rmask = rows < T
        n_hi = tl.load(NHI + (pid // (nh * nm)) * nm + mb)
        lim_a, lim_i = _limits(rows, T, SEQ)
        iq = _inv_norm(QN, b64, rows, rmask, T, NORM)
        scale = tl.load(SCALE).to(tl.float32)
        cg = tl.load(CG).to(tl.float32)
        hv = hb * BH + tl.arange(0, BH)
        hvm = hv < H
        m_s = tl.full((BM,), float("-inf"), tl.float32)
        m_p = tl.full((BM,), float("-inf"), tl.float32)
        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_s = tl.zeros((BM,), tl.float32)
        l_p = tl.zeros((BM,), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        a_s = tl.zeros((BM, BH), tl.float32)
        a_p = tl.zeros((BM, BH), tl.float32)
        a_i = tl.zeros((BM, BH), tl.float32)
        for n0 in range(0, n_hi, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < N
            krow, nrow = _bank_rows(KROW, NROW, b64, cols, cmask, N, GATHER)
            pos = tl.load(POS + b64 * N + cols, cmask, 1 << 30)
            raw = _scores(Q, K, b64, rows, krow, rmask, cmask, T, S, H, BM, BN, BK, IP)
            ik = _inv_norm(KN, b64, krow, cmask, S, NORM)
            raw = raw * iq[:, None] * ik[None, :]
            kb = tl.load(KB + b64 * S + krow, cmask, 0.0)
            s = raw * scale + kb[None, :]
            p = s + cg * _cocktail(CQ, CK, b64, rows, krow, rmask, cmask, T, S)
            allow = pos[None, :] <= lim_a[:, None]
            induct = pos[None, :] < lim_i[:, None]
            vmask = cmask[:, None] & hvm[None, :]
            v = tl.load(V + b64 * S * H + krow[:, None] * H + hv[None, :], vmask, 0.0)
            vn = tl.load(VN + b64 * S * H + nrow[:, None] * H + hv[None, :], vmask, 0.0)
            m_s, l_s, a_s = _online_acc(m_s, l_s, a_s, s, allow, v, IP)
            m_p, l_p, a_p = _online_acc(m_p, l_p, a_p, p, allow, v, IP)
            m_i, l_i, a_i = _online_acc(m_i, l_i, a_i, s, induct, vn, IP)
        o_s, lse_s = _finish(m_s, l_s, a_s)
        o_p, lse_p = _finish(m_p, l_p, a_p)
        o_i, lse_i = _finish(m_i, l_i, a_i)
        sg = tl.load(SG).to(tl.float32)
        pg = tl.load(PG).to(tl.float32)
        ig = tl.load(IG).to(tl.float32)
        out = sg * o_s + pg * o_p + ig * o_i
        oo = b64 * T * H + rows[:, None] * H + hv[None, :]
        om = rmask[:, None] & hvm[None, :]
        tl.store(OUT + oo, out.to(OUT.dtype.element_ty), om)
        if SAVE:
            ob = b64 * 3 * T * H + rows[:, None] * H + hv[None, :]
            tl.store(OP + ob, o_s, om)
            tl.store(OP + ob + T * H, o_p, om)
            tl.store(OP + ob + 2 * T * H, o_i, om)
        if hb == 0:
            base = LSE + b64 * 3 * T
            tl.store(base + rows, lse_s, rmask)
            tl.store(base + T + rows, lse_p, rmask)
            tl.store(base + 2 * T + rows, lse_i, rmask)

    @triton.jit
    def _weights(s, p, pos, lim_a, lim_i, LSE, b64, rows, rmask, T):
        base = b64 * 3 * T
        lse_s = tl.load(LSE + base + rows, rmask, float("inf"))
        lse_p = tl.load(LSE + base + T + rows, rmask, float("inf"))
        lse_i = tl.load(LSE + base + 2 * T + rows, rmask, float("inf"))
        allow = pos[None, :] <= lim_a[:, None]
        induct = pos[None, :] < lim_i[:, None]
        w_s = tl.where(allow, tl.exp(s - lse_s[:, None]), 0.0)
        w_p = tl.where(allow, tl.exp(p - lse_p[:, None]), 0.0)
        w_i = tl.where(induct, tl.exp(s - lse_i[:, None]), 0.0)
        return w_s, w_p, w_i

    @triton.jit
    def _grad_scores(w_s, w_p, w_i, u, un, E, b64, rows, rmask, T, sg, pg, ig):
        base = E + b64 * 3 * T
        e_s = tl.load(base + rows, rmask, 0.0)
        e_p = tl.load(base + T + rows, rmask, 0.0)
        e_i = tl.load(base + 2 * T + rows, rmask, 0.0)
        gp = pg * w_p * (u - e_p[:, None])
        g = sg * w_s * (u - e_s[:, None]) + gp + ig * w_i * (un - e_i[:, None])
        return g, gp

    @triton.jit
    def _bwd_query_kernel(Q, QN, K, KN, V, VN, KROW, NROW, DO, CQ, CK, KB, POS, NHI, SCALE, SG,
                          PG, IG, CG, LSE, E, DQ, DCQ, PART, T, N, S, H, SEQ: tl.constexpr,
                          GATHER: tl.constexpr, NORM: tl.constexpr, BM: tl.constexpr,
                          BN: tl.constexpr, BK: tl.constexpr, BH: tl.constexpr,
                          IP: tl.constexpr):
        # dq here is the gradient with respect to the normalized query.
        pid = tl.program_id(0)
        nh = tl.cdiv(H, BH)
        nm = tl.cdiv(T, BM)
        hb = pid % nh
        mb = (pid // nh) % nm
        bi = pid // (nh * nm)
        b64 = bi.to(tl.int64)
        rows = mb * BM + tl.arange(0, BM)
        rmask = rows < T
        n_hi = tl.load(NHI + bi * nm + mb)
        lim_a, lim_i = _limits(rows, T, SEQ)
        iq = _inv_norm(QN, b64, rows, rmask, T, NORM)
        scale = tl.load(SCALE).to(tl.float32)
        cg = tl.load(CG).to(tl.float32)
        sg = tl.load(SG).to(tl.float32)
        pg = tl.load(PG).to(tl.float32)
        ig = tl.load(IG).to(tl.float32)
        hv = hb * BH + tl.arange(0, BH)
        hvm = hv < H
        acc = tl.zeros((BM, BH), tl.float32)
        dcq0 = tl.zeros((BM,), tl.float32)
        dcq1 = tl.zeros((BM,), tl.float32)
        dcq2 = tl.zeros((BM,), tl.float32)
        dcq3 = tl.zeros((BM,), tl.float32)
        dscale = tl.zeros((BM,), tl.float32)
        dcg = tl.zeros((BM,), tl.float32)
        for n0 in range(0, n_hi, BN):
            cols = n0 + tl.arange(0, BN)
            cmask = cols < N
            krow, nrow = _bank_rows(KROW, NROW, b64, cols, cmask, N, GATHER)
            pos = tl.load(POS + b64 * N + cols, cmask, 1 << 30)
            raw, u, un = _scores3(Q, K, DO, V, VN, b64, rows, krow, nrow, rmask, cmask, T, S, H,
                                  BM, BN, BK, IP)
            ik = _inv_norm(KN, b64, krow, cmask, S, NORM)
            raw = raw * iq[:, None] * ik[None, :]
            kb = tl.load(KB + b64 * S + krow, cmask, 0.0)
            s = raw * scale + kb[None, :]
            cc = _cocktail(CQ, CK, b64, rows, krow, rmask, cmask, T, S)
            p = s + cg * cc
            w_s, w_p, w_i = _weights(s, p, pos, lim_a, lim_i, LSE, b64, rows, rmask, T)
            g, gp = _grad_scores(w_s, w_p, w_i, u, un, E, b64, rows, rmask, T, sg, pg, ig)
            k = tl.load(K + b64 * S * H + krow[:, None] * H + hv[None, :],
                        cmask[:, None] & hvm[None, :], 0.0)
            acc = tl.dot((g * ik[None, :]).to(k.dtype), k, acc, input_precision=IP)
            dscale += tl.sum(g * raw, 1)
            dcg += tl.sum(gp * cc, 1)
            ckb = CK + b64 * S * 4 + krow * 4
            dcq0 += tl.sum(gp * tl.load(ckb + 0, cmask, 0.0)[None, :], 1)
            dcq1 += tl.sum(gp * tl.load(ckb + 1, cmask, 0.0)[None, :], 1)
            dcq2 += tl.sum(gp * tl.load(ckb + 2, cmask, 0.0)[None, :], 1)
            dcq3 += tl.sum(gp * tl.load(ckb + 3, cmask, 0.0)[None, :], 1)
        tl.store(DQ + b64 * T * H + rows[:, None] * H + hv[None, :], acc * scale,
                 rmask[:, None] & hvm[None, :])
        if hb == 0:
            cq = DCQ + b64 * T * 4 + rows * 4
            tl.store(cq + 0, dcq0 * cg, rmask)
            tl.store(cq + 1, dcq1 * cg, rmask)
            tl.store(cq + 2, dcq2 * cg, rmask)
            tl.store(cq + 3, dcq3 * cg, rmask)
            slot = PART + (bi * nm + mb) * 2
            tl.store(slot, tl.sum(tl.where(rmask, dscale, 0.0), 0))
            tl.store(slot + 1, tl.sum(tl.where(rmask, dcg, 0.0), 0))

    @triton.jit
    def _bwd_bank_kernel(Q, QN, K, KN, V, VN, KROW, NROW, DO, CQ, CK, KB, POS, SCALE, SG, PG, IG,
                         CG, LSE, E, DK, DV, DVN, DKB, DCK, T, N, S, H, SEQ: tl.constexpr,
                         GATHER: tl.constexpr, NORM: tl.constexpr, BM: tl.constexpr,
                         BN: tl.constexpr, BK: tl.constexpr, BH: tl.constexpr,
                         IP: tl.constexpr):
        # Per bank entry: dk (w.r.t. the normalized key), dv, dnext, d(bias+admission), dck.
        pid = tl.program_id(0)
        nh = tl.cdiv(H, BH)
        nn = tl.cdiv(N, BN)
        hb = pid % nh
        nb = (pid // nh) % nn
        b64 = (pid // (nh * nn)).to(tl.int64)
        cols = nb * BN + tl.arange(0, BN)
        cmask = cols < N
        krow, nrow = _bank_rows(KROW, NROW, b64, cols, cmask, N, GATHER)
        pos = tl.load(POS + b64 * N + cols, cmask, 1 << 30)
        ik = _inv_norm(KN, b64, krow, cmask, S, NORM)
        kb = tl.load(KB + b64 * S + krow, cmask, 0.0)
        scale = tl.load(SCALE).to(tl.float32)
        cg = tl.load(CG).to(tl.float32)
        sg = tl.load(SG).to(tl.float32)
        pg = tl.load(PG).to(tl.float32)
        ig = tl.load(IG).to(tl.float32)
        hv = hb * BH + tl.arange(0, BH)
        hvm = hv < H
        dk = tl.zeros((BN, BH), tl.float32)
        dv = tl.zeros((BN, BH), tl.float32)
        dvn = tl.zeros((BN, BH), tl.float32)
        dkb = tl.zeros((BN,), tl.float32)
        dck0 = tl.zeros((BN,), tl.float32)
        dck1 = tl.zeros((BN,), tl.float32)
        dck2 = tl.zeros((BN,), tl.float32)
        dck3 = tl.zeros((BN,), tl.float32)
        if SEQ:
            # Carried bank entries (online chunks) have negative positions relative to
            # the chunk: every query row sees them. Never start before query row 0.
            first = tl.min(pos, 0)
            m_lo = (tl.maximum(tl.minimum(first, T), 0) // BM) * BM
        else:
            m_lo = 0
        for m0 in range(m_lo, T, BM):
            rows = m0 + tl.arange(0, BM)
            rmask = rows < T
            lim_a, lim_i = _limits(rows, T, SEQ)
            raw, u, un = _scores3(Q, K, DO, V, VN, b64, rows, krow, nrow, rmask, cmask, T, S, H,
                                  BM, BN, BK, IP)
            iq = _inv_norm(QN, b64, rows, rmask, T, NORM)
            raw = raw * iq[:, None] * ik[None, :]
            s = raw * scale + kb[None, :]
            p = s + cg * _cocktail(CQ, CK, b64, rows, krow, rmask, cmask, T, S)
            w_s, w_p, w_i = _weights(s, p, pos, lim_a, lim_i, LSE, b64, rows, rmask, T)
            g, gp = _grad_scores(w_s, w_p, w_i, u, un, E, b64, rows, rmask, T, sg, pg, ig)
            qo = b64 * T * H + rows[:, None] * H + hv[None, :]
            qm = rmask[:, None] & hvm[None, :]
            q = tl.load(Q + qo, qm, 0.0)
            do = tl.load(DO + qo, qm, 0.0)
            dk = tl.dot(tl.trans((g * iq[:, None]).to(q.dtype)), q, dk, input_precision=IP)
            dv = tl.dot(tl.trans((sg * w_s + pg * w_p).to(do.dtype)), do, dv, input_precision=IP)
            dvn = tl.dot(tl.trans((ig * w_i).to(do.dtype)), do, dvn, input_precision=IP)
            dkb += tl.sum(g, 0)
            cqb = CQ + b64 * T * 4 + rows * 4
            dck0 += tl.sum(gp * tl.load(cqb + 0, rmask, 0.0)[:, None], 0)
            dck1 += tl.sum(gp * tl.load(cqb + 1, rmask, 0.0)[:, None], 0)
            dck2 += tl.sum(gp * tl.load(cqb + 2, rmask, 0.0)[:, None], 0)
            dck3 += tl.sum(gp * tl.load(cqb + 3, rmask, 0.0)[:, None], 0)
        ko = b64 * N * H + cols[:, None] * H + hv[None, :]
        km = cmask[:, None] & hvm[None, :]
        tl.store(DK + ko, dk * scale, km)
        tl.store(DV + ko, dv, km)
        tl.store(DVN + ko, dvn, km)
        if hb == 0:
            tl.store(DKB + b64 * N + cols, dkb, cmask)
            ckb = DCK + b64 * N * 4 + cols * 4
            tl.store(ckb + 0, dck0 * cg, cmask)
            tl.store(ckb + 1, dck1 * cg, cmask)
            tl.store(ckb + 2, dck2 * cg, cmask)
            tl.store(ckb + 3, dck3 * cg, cmask)

    @triton.jit
    def _norm_back(X, DY, NRM, b64, t, T, H, BH: tl.constexpr):
        # Gradient of y = x / max(|x|, eps) for one row: ((dy - y (y.dy)) / |x|),
        # or dy / eps when the norm is clamped.
        n = tl.load(NRM + b64 * T + t)
        scaled = n > 1e-12
        inv = 1.0 / tl.maximum(n, 1e-12)
        dot = tl.zeros((BH,), tl.float32)
        for h0 in range(0, H, BH):
            h = h0 + tl.arange(0, BH)
            hm = h < H
            x = tl.load(X + (b64 * T + t) * H + h, hm, 0.0).to(tl.float32)
            dy = tl.load(DY + h, hm, 0.0)
            dot += x * dy
        coef = tl.where(scaled, tl.sum(dot, 0) * inv * inv, 0.0)
        return inv, coef

    @triton.jit
    def _combine_kernel(X, XN, W, WN, DQ, DK, DV, DVN, DKB, DCK, DCQ, INV, DX, DW, DKBT, DCOCK,
                        T, N, H, BH: tl.constexpr):
        # Per token: query-side and bank-side gradients onto the per-token tensors.
        pid = tl.program_id(0)
        b64 = (pid // T).to(tl.int64)
        t = pid % T
        n = tl.load(INV + b64 * T + t)
        prev = tl.load(INV + b64 * T + t - 1, mask=t > 0, other=-1)
        has = n >= 0
        nn = tl.maximum(n, 0).to(tl.int64)
        pp = tl.maximum(prev, 0).to(tl.int64)
        clamp_self = has & (t == T - 1)
        inv_q, coef_q = _norm_back(X, DQ + (b64 * T + t) * H, XN, b64, t, T, H, BH)
        inv_k, coef_k = _norm_back(W, DK + (b64 * N + nn) * H, WN, b64, t, T, H, BH)
        for h0 in range(0, H, BH):
            h = h0 + tl.arange(0, BH)
            hm = h < H
            row = (b64 * T + t) * H + h
            x = tl.load(X + row, hm, 0.0).to(tl.float32)
            dq = tl.load(DQ + row, hm, 0.0)
            tl.store(DX + row, (dq - x * coef_q) * inv_q, hm)
            w = tl.load(W + row, hm, 0.0).to(tl.float32)
            bank = (b64 * N + nn) * H + h
            dk = tl.load(DK + bank, hm & has, 0.0)
            dw = tl.where(has, (dk - w * coef_k) * inv_k, 0.0)
            dw += tl.load(DV + bank, hm & has, 0.0)
            dw += tl.load(DVN + (b64 * N + pp) * H + h, hm & (prev >= 0), 0.0)
            dw += tl.load(DVN + bank, hm & clamp_self, 0.0)
            tl.store(DW + row, dw, hm)
        tl.store(DKBT + b64 * T + t, tl.load(DKB + b64 * N + nn, has, 0.0))
        j = tl.arange(0, 4)
        dc = tl.load(DCQ + (b64 * T + t) * 4 + j) + tl.load(DCK + (b64 * N + nn) * 4 + j,
                                                           has & (j < 4), 0.0)
        tl.store(DCOCK + (b64 * T + t) * 4 + j, dc)


_IP = {0: "ieee", 1: "tf32"}


def _bank_limits(pos: Tensor, T: int, BM: int, mode: int) -> Tensor:
    """Per (batch, query tile): one past the last bank entry any row can see.

    The suffix minimum of bank positions is non-decreasing, so a searchsorted of
    the tile's last row gives an exact loop bound for any bank order (tight for
    the front-packed banks the read builds). Device-only: no host sync."""
    B, N = pos.shape
    nm = triton.cdiv(T, BM)
    if mode != 0:
        return torch.full((B, nm), N, device=pos.device, dtype=torch.int32)
    suffix = pos.flip(1).cummin(1).values.flip(1).contiguous()
    last = (torch.arange(nm, device=pos.device, dtype=torch.int32) * BM + BM - 1).clamp_(max=T - 1)
    return torch.searchsorted(suffix, last.expand(B, nm).contiguous(), right=True).to(torch.int32)


@on_tensor_device
def _launch_forward(q, qn, k, kn, v, vn, krow, nrow, cq, ck, kb, pos, scal, mode, ip, save,
                    gather):
    B, T, H = q.shape
    N = pos.shape[1]
    S = k.shape[1]
    norm = qn is not None
    qn = q if qn is None else qn
    kn = q if kn is None else kn
    krow = pos if krow is None else krow
    nrow = pos if nrow is None else nrow
    out = torch.empty_like(q)
    lse = torch.empty((B, 3, T), device=q.device, dtype=torch.float32)
    reads = torch.empty((B, 3, T, H) if save else (0,), device=q.device, dtype=torch.float32)

    def launch(cfg):
        grid = (B * triton.cdiv(T, cfg["BM"]) * triton.cdiv(H, cfg["BH"]),)
        _fwd_kernel[grid](q, qn, k, kn, v, vn, krow, nrow, cq, ck, kb, pos,
                          _bank_limits(pos, T, cfg["BM"], mode), scal[0:1], scal[1:2],
                          scal[2:3], scal[3:4], scal[4:5], out, lse, reads, T, N, S, H,
                          SEQ=mode == 0, SAVE=save, GATHER=gather, NORM=norm, IP=_IP[ip], **cfg)

    _launch("fwd", H, q.dtype, gather, launch)
    return out, lse, reads


@on_tensor_device
def _launch_backward(do, q, qn, k, kn, v, vn, krow, nrow, cq, ck, kb, pos, scal, lse, reads,
                     mode, ip, gather):
    """Per-entry gradients: dq (normalized query), dk (normalized key), dv, dnext,
    d(bias+admission), dck, dcq, and the five scalar gradients."""
    B, T, H = q.shape
    N = pos.shape[1]
    S = k.shape[1]
    seq = mode == 0
    IP = _IP[ip]
    norm = qn is not None
    qn = q if qn is None else qn
    kn = q if kn is None else kn
    krow = pos if krow is None else krow
    nrow = pos if nrow is None else nrow
    f = dict(device=q.device, dtype=torch.float32)
    # Softmax-backward row terms E = rowsum(dO * O) per read, from the saved
    # per-path reads (no extra pass over the bank).
    e = torch.einsum("bth,bjth->bjt", do.to(torch.float32), reads).contiguous()
    dq = torch.empty((B, T, H), **f)
    dcq = torch.empty((B, T, 4), **f)
    parts = {}
    common = dict(SEQ=seq, GATHER=gather, NORM=norm, IP=IP)

    def launch_query(cfg):
        nm = triton.cdiv(T, cfg["BM"])
        parts["part"] = part = torch.empty((B * nm, 2), **f)
        _bwd_query_kernel[(B * nm * triton.cdiv(H, cfg["BH"]),)](
            q, qn, k, kn, v, vn, krow, nrow, do, cq, ck, kb, pos,
            _bank_limits(pos, T, cfg["BM"], mode), scal[0:1], scal[1:2], scal[2:3], scal[3:4],
            scal[4:5], lse, e, dq, dcq, part, T, N, S, H, **common, **cfg)

    _launch("query", H, q.dtype, gather, launch_query)
    dk = torch.empty((B, N, H), **f)
    dv = torch.empty_like(dk)
    dvn = torch.empty_like(dk)
    dkb = torch.empty((B, N), **f)
    dck = torch.empty((B, N, 4), **f)

    def launch_bank(cfg):
        _bwd_bank_kernel[(B * triton.cdiv(N, cfg["BN"]) * triton.cdiv(H, cfg["BH"]),)](
            q, qn, k, kn, v, vn, krow, nrow, do, cq, ck, kb, pos, scal[0:1], scal[1:2],
            scal[2:3], scal[3:4], scal[4:5], lse, e, dk, dv, dvn, dkb, dck, T, N, S, H,
            **common, **cfg)

    _launch("bank", H, q.dtype, gather, launch_bank)
    sums = e.sum(dim=(0, 2))
    tail = parts["part"].sum(dim=0)
    dscal = torch.stack([tail[0], sums[0], sums[1], sums[2], tail[1]])
    return dq, dk, dv, dvn, dkb, dck, dcq, dscal


def _f32_scalars(scale, short_gain, pad_gain, induct_gain, cocktail_gain) -> Tensor:
    return torch.cat([t.reshape(1) for t in (scale, short_gain, pad_gain, induct_gain,
                                              cocktail_gain)]).to(torch.float32)


# ---------------------------------------------------------------------------
# ABI 1: gathered bank (compact-read contract).
# ---------------------------------------------------------------------------
@torch.library.custom_op("dabsn::admitted_read_flash", mutates_args=())
def _flash_op(q: Tensor, k: Tensor, v: Tensor, vn: Tensor, cq: Tensor, ck: Tensor, kb: Tensor,
              pos: Tensor, scal: Tensor, mode: int, ip: int, save: bool
              ) -> tuple[Tensor, Tensor, Tensor]:
    return _launch_forward(q, None, k, None, v, vn, None, None, cq, ck, kb, pos, scal, mode, ip,
                           save, False)


@_flash_op.register_fake
def _flash_fake(q, k, v, vn, cq, ck, kb, pos, scal, mode, ip, save):
    B, T, H = q.shape
    f = torch.float32
    return (torch.empty_like(q), q.new_empty((B, 3, T), dtype=f),
            q.new_empty((B, 3, T, H) if save else (0,), dtype=f))


@torch.library.custom_op("dabsn::admitted_read_flash_backward", mutates_args=())
def _flash_backward_op(do: Tensor, q: Tensor, k: Tensor, v: Tensor, vn: Tensor, cq: Tensor,
                       ck: Tensor, kb: Tensor, pos: Tensor, scal: Tensor, lse: Tensor,
                       reads: Tensor, mode: int, ip: int) -> List[Tensor]:
    dq, dk, dv, dvn, dkb, dck, dcq, dscal = _launch_backward(
        do, q, None, k, None, v, vn, None, None, cq, ck, kb, pos, scal, lse, reads, mode, ip,
        False)
    return [dq, dk, dv, dvn, dcq, dck, dkb, dscal]


@_flash_backward_op.register_fake
def _flash_backward_fake(do, q, k, v, vn, cq, ck, kb, pos, scal, lse, reads, mode, ip):
    f = torch.float32
    return [q.new_empty(q.shape, dtype=f), k.new_empty(k.shape, dtype=f),
            k.new_empty(k.shape, dtype=f), k.new_empty(k.shape, dtype=f),
            cq.new_empty(cq.shape, dtype=f), ck.new_empty(ck.shape, dtype=f),
            kb.new_empty(kb.shape, dtype=f), scal.new_empty((5,), dtype=f)]


def _flash_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs[:9], output[1], output[2])
    ctx.mode, ctx.ip = int(inputs[9]), int(inputs[10])


def _flash_backward(ctx, grad_out: Tensor, _grad_lse, _grad_reads):
    q, k, v, vn, cq, ck, kb, pos, scal, lse, reads = ctx.saved_tensors
    dq, dk, dv, dvn, dcq, dck, dkb, dscal = _flash_backward_op(
        grad_out.to(q.dtype).contiguous(), q, k, v, vn, cq, ck, kb, pos, scal, lse, reads,
        ctx.mode, ctx.ip)
    return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dvn.to(vn.dtype), dcq, dck, dkb,
            None, dscal, None, None, None)


torch.library.register_autograd(_flash_op, _flash_backward, setup_context=_flash_setup)


def admitted_three_way_read_flash(
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
    """Fused CUDA admitted three-way read on a gathered bank (compact-read ABI)."""
    if mode not in {"seq", "field"}:
        raise ValueError("flash admitted-read mode must be 'seq' or 'field'")
    dt = _operand_dtype(query)
    f32 = torch.float32
    pos = torch.where(bank_valid, bank_idx, torch.full_like(bank_idx, _INVALID)).to(torch.int32)
    tensors = (query, bank_keys, bank_writes, next_writes, cocktail, bank_cocktail, bank_key_bias,
               bank_admission, scale, short_gain, pad_gain, induct_gain, cocktail_gain)
    save = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    out, _lse, _reads = _flash_op(
        query.to(dt).contiguous(),
        bank_keys.to(dt).contiguous(),
        bank_writes.to(dt).contiguous(),
        next_writes.to(dt).contiguous(),
        cocktail.to(f32).contiguous(),
        bank_cocktail.to(f32).contiguous(),
        (bank_key_bias.to(f32) + bank_admission.to(f32)).contiguous(),
        pos.contiguous(),
        _f32_scalars(scale, short_gain, pad_gain, induct_gain, cocktail_gain),
        0 if mode == "seq" else 1,
        0 if dt != f32 or _input_precision() == "ieee" else 1,
        save,
    )
    return out.to(query.dtype)


# ---------------------------------------------------------------------------
# ABI 2: per-token tensors; the bank is addressed by position inside the kernels.
# ---------------------------------------------------------------------------
def _bank_addresses(bank_idx: Tensor, bank_valid: Tensor, T: int):
    krow = bank_idx.to(torch.int32).contiguous()
    nrow = (bank_idx + 1).clamp_(max=T - 1).to(torch.int32).contiguous()
    pos = torch.where(bank_valid, krow, torch.full_like(krow, _INVALID)).contiguous()
    return krow, nrow, pos


@torch.library.custom_op("dabsn::admitted_read_tokens", mutates_args=())
def _tokens_op(x: Tensor, w: Tensor, cock: Tensor, kba: Tensor, krow: Tensor, nrow: Tensor,
               pos: Tensor, scal: Tensor, mode: int, ip: int, save: bool
               ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    xn = torch.linalg.vector_norm(x, dim=-1, dtype=torch.float32)
    wn = torch.linalg.vector_norm(w, dim=-1, dtype=torch.float32)
    out, lse, reads = _launch_forward(x, xn, w, wn, w, w, krow, nrow, cock, cock, kba, pos, scal,
                                      mode, ip, save, True)
    return out, lse, reads, xn, wn


@_tokens_op.register_fake
def _tokens_fake(x, w, cock, kba, krow, nrow, pos, scal, mode, ip, save):
    B, T, H = x.shape
    f = torch.float32
    return (torch.empty_like(x), x.new_empty((B, 3, T), dtype=f),
            x.new_empty((B, 3, T, H) if save else (0,), dtype=f),
            x.new_empty((B, T), dtype=f), x.new_empty((B, T), dtype=f))


@torch.library.custom_op("dabsn::admitted_read_tokens_backward", mutates_args=())
def _tokens_backward_op(do: Tensor, x: Tensor, w: Tensor, cock: Tensor, kba: Tensor,
                        krow: Tensor, nrow: Tensor, pos: Tensor, scal: Tensor, lse: Tensor,
                        reads: Tensor, xn: Tensor, wn: Tensor, mode: int, ip: int
                        ) -> List[Tensor]:
    B, T, H = x.shape
    N = pos.shape[1]
    dq, dk, dv, dvn, dkb, dck, dcq, dscal = _launch_backward(
        do, x, xn, w, wn, w, w, krow, nrow, cock, cock, kba, pos, scal, lse, reads, mode, ip,
        True)
    inv = torch.full((B, T), -1, device=x.device, dtype=torch.int32)
    inv.scatter_(1, krow.long(), torch.arange(N, device=x.device, dtype=torch.int32).expand(B, N))
    f = dict(device=x.device, dtype=torch.float32)
    dx = torch.empty((B, T, H), **f)
    dw = torch.empty((B, T, H), **f)
    dkba = torch.empty((B, T), **f)
    dcock = torch.empty((B, T, 4), **f)
    _combine_kernel[(B * T,)](x, xn, w, wn, dq, dk, dv, dvn, dkb, dck, dcq, inv, dx, dw, dkba,
                              dcock, T, N, H, BH=min(1024, max(16, triton.next_power_of_2(H))),
                              num_warps=4)
    return [dx, dw, dcock, dkba, dscal]


@_tokens_backward_op.register_fake
def _tokens_backward_fake(do, x, w, cock, kba, krow, nrow, pos, scal, lse, reads, xn, wn, mode,
                          ip):
    f = torch.float32
    return [x.new_empty(x.shape, dtype=f), w.new_empty(w.shape, dtype=f),
            cock.new_empty(cock.shape, dtype=f), kba.new_empty(kba.shape, dtype=f),
            scal.new_empty((5,), dtype=f)]


def _tokens_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs[:8], output[1], output[2], output[3], output[4])
    ctx.mode, ctx.ip = int(inputs[8]), int(inputs[9])


@on_tensor_device
def _tokens_backward(ctx, grad_out: Tensor, *_unused):
    x, w, cock, kba, krow, nrow, pos, scal, lse, reads, xn, wn = ctx.saved_tensors
    dx, dw, dcock, dkba, dscal = _tokens_backward_op(
        grad_out.to(x.dtype).contiguous(), x, w, cock, kba, krow, nrow, pos, scal, lse, reads,
        xn, wn, ctx.mode, ctx.ip)
    return (dx.to(x.dtype), dw.to(w.dtype), dcock, dkba, None, None, None, dscal, None, None,
            None)


torch.library.register_autograd(_tokens_op, _tokens_backward, setup_context=_tokens_setup)


def admitted_three_way_read_tokens(
    expression: Tensor,
    write: Tensor,
    cocktail: Tensor,
    key_admission: Tensor,
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
    """Fused admitted read from per-token tensors.

    ``expression`` is the raw query (normalized in the kernel), ``write`` the
    raw write stream (bank keys are its normalized rows, values its rows, next
    values the following rows), ``key_admission`` the per-token
    ``key_bias + admission``, and ``bank_idx``/``bank_valid`` the admitted bank
    built by admission. Same values and gradients as gathering the bank and
    calling the compact read."""
    if mode not in {"seq", "field"}:
        raise ValueError("token admitted-read mode must be 'seq' or 'field'")
    T = expression.shape[1]
    dt = _operand_dtype(expression)
    f32 = torch.float32
    krow, nrow, pos = _bank_addresses(bank_idx, bank_valid, T)
    tensors = (expression, write, cocktail, key_admission, scale, short_gain, pad_gain,
               induct_gain, cocktail_gain)
    save = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    out = _tokens_op(
        expression.to(dt).contiguous(),
        write.to(dt).contiguous(),
        cocktail.to(f32).contiguous(),
        key_admission.to(f32).contiguous(),
        krow,
        nrow,
        pos,
        _f32_scalars(scale, short_gain, pad_gain, induct_gain, cocktail_gain),
        0 if mode == "seq" else 1,
        0 if dt != f32 or _input_precision() == "ieee" else 1,
        save,
    )[0]
    return out.to(expression.dtype)


def token_read_supported(expression: Tensor, write: Tensor) -> bool:
    return (
        _HAS_TRITON
        and expression.is_cuda
        and _operand_dtype_ok(expression)
        and expression.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and write.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and expression.dim() == 3
        and write.shape == expression.shape
        and expression.shape[1] >= 1
    )


__all__ = [
    "admitted_three_way_read_flash",
    "admitted_three_way_read_tokens",
    "flash_read_supported",
    "token_read_supported",
]
