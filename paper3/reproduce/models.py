"""Transformer model zoo for the paper 3 matrix.

One pre-LN decoder-only transformer whose attention can be:

* ``none``  : NoPE, causal softmax attention with no positional signal
* ``rope``  : rotary positions; at evaluation optionally rescaled with position
              interpolation (``pi``) or YaRN (``yarn``), both applied at test time
* ``alibi`` : linear distance biases
* sliding-window NoPE (``window``)
* Infini-attention (``infini_segment``): segment-local softmax attention mixed
  with a compressive linear memory (delta-rule update), in every layer

Optionally one or more residual inserts ``h <- h + branch(LN(h))`` sit at
``front`` (before the first block), ``middle`` (before block L//2) or ``back``
(after the last block). The branch is ``DABSNHostMemory`` or, for the
parameter-matched control, a two-layer MLP with the same parameter count.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dabsn import DABSNHostMemory

# ----------------------------------------------------------------------
# Named variants. Values that depend on the run size (window, segment) are
# resolved from the size dict by ``model_spec``.

ZOO = {
    "nope": {"pe": "none"},
    "rope": {"pe": "rope"},
    "alibi": {"pe": "alibi"},
    "swa": {"pe": "none", "window": True},
    "infini": {"pe": "rope", "infini": True},
    "nope_mem": {"pe": "none", "insert": "dabsn", "placements": ["middle"]},
    "swa_mem": {"pe": "none", "window": True, "insert": "dabsn", "placements": ["middle"]},
    "nope_ctrl": {"pe": "none", "insert": "mlp_matched", "placements": ["middle"]},
    "nope_mem_front": {"pe": "none", "insert": "dabsn", "placements": ["front"]},
    "nope_ctrl_front": {"pe": "none", "insert": "mlp_matched", "placements": ["front"]},
    "swa_mem_front": {"pe": "none", "window": True, "insert": "dabsn", "placements": ["front"]},
    "nope_mem_back": {"pe": "none", "insert": "dabsn", "placements": ["back"]},
    "nope_mem_x2": {"pe": "none", "insert": "dabsn", "placements": ["front", "back"]},
}

# RoPE-trained checkpoints are additionally evaluated with these test-time
# rescalings; they are not separate trainings.
EVAL_VARIANTS = {"rope": ["base", "pi", "yarn"]}


def model_spec(name: str, size: dict) -> dict:
    base = ZOO[name]
    return {
        "vocab": int(size["vocab"]),
        "d_model": int(size["d_model"]),
        "n_layers": int(size["n_layers"]),
        "n_heads": int(size["n_heads"]),
        "mlp_mult": int(size.get("mlp_mult", 4)),
        "pe": base["pe"],
        "window": int(size["window"]) if base.get("window") else None,
        "infini_segment": int(size["segment"]) if base.get("infini") else None,
        "insert": base.get("insert"),
        "placements": list(base.get("placements", [])),
        "rope_base": float(size.get("rope_base", 10000.0)),
        **({"memory_amp": "host_autocast"} if base.get("insert") == "dabsn" else {}),
    }


# ----------------------------------------------------------------------
# Positional machinery

def rope_tables(positions: torch.Tensor, head_dim: int, base: float, mode: str,
                train_len: int | None, eval_len: int, min_scale: float = 1.0):
    """cos/sin tables [T, head_dim]. ``mode`` in {base, pi, yarn}.

    The scale is ``max(min_scale, eval_len / train_len)``; a YaRN model
    fine-tuned at scale s uses ``min_scale = s`` at every length."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64,
                                       device=positions.device) / head_dim))
    mscale = 1.0
    s = 1.0 if train_len is None else max(float(min_scale), eval_len / train_len)
    if mode == "pi" and s > 1:
        inv = inv / s
    elif mode == "yarn" and s > 1:
        # NTK-by-parts interpolation with attention temperature (Peng et al.).
        alpha, beta = 1.0, 32.0
        wavelen = 2 * math.pi / inv
        r = train_len / wavelen
        gamma = ((r - alpha) / (beta - alpha)).clamp(0.0, 1.0)
        inv = inv * ((1 - gamma) / s + gamma)
        mscale = 0.1 * math.log(s) + 1.0
    elif mode not in {"base", "pi", "yarn"}:
        raise ValueError(mode)
    freqs = positions.to(torch.float64)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos() * mscale).float(), (emb.sin() * mscale).float()


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos.to(x.dtype) + _rotate_half(x) * sin.to(x.dtype)


def alibi_slopes(n_heads: int) -> torch.Tensor:
    return torch.tensor([2.0 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)])


# ----------------------------------------------------------------------
# Attention kernels (all exact; chunking only bounds memory)

def causal_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def alibi_attention(q, k, v, slopes, budget: int = 1 << 25):
    B, H, T, _ = q.shape
    outs = []
    qb = max(1, min(T, budget // max(1, B * H * T)))
    slopes = slopes.to(q.device, torch.float32)
    for s in range(0, T, qb):
        e = min(T, s + qb)
        qi = torch.arange(s, e, device=q.device)
        kj = torch.arange(0, e, device=q.device)
        dist = (qi[:, None] - kj[None, :]).float()
        bias = -slopes[:, None, None] * dist
        bias = bias.masked_fill(dist < 0, float("-inf")).to(q.dtype)
        outs.append(F.scaled_dot_product_attention(q[:, :, s:e], k[:, :, :e], v[:, :, :e],
                                                   attn_mask=bias.unsqueeze(0)))
    return torch.cat(outs, dim=2)


def sliding_window_attention(q, k, v, window: int):
    """Causal attention to the previous ``window`` positions (incl. self)."""
    B, H, T, dh = q.shape
    w = window
    if T <= 4 * w:
        i = torch.arange(T, device=q.device)
        rel = i[:, None] - i[None, :]
        mask = (rel >= 0) & (rel < w)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    nb = -(-T // w)
    pad = nb * w - T
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    qb = q.reshape(B, H * nb, w, dh)
    kb = k.reshape(B, H, nb, w, dh)
    vb = v.reshape(B, H, nb, w, dh)
    kk = torch.cat([F.pad(kb, (0, 0, 0, 0, 1, 0))[:, :, :-1], kb], dim=3)
    vv = torch.cat([F.pad(vb, (0, 0, 0, 0, 1, 0))[:, :, :-1], vb], dim=3)
    i = torch.arange(w, device=q.device)[:, None]
    j = torch.arange(2 * w, device=q.device)[None, :]
    rel = i - (j - w)
    m = ((rel >= 0) & (rel < w)).expand(nb, w, 2 * w).clone()
    m[0, :, :w] = False  # block 0 has no previous block
    out = F.scaled_dot_product_attention(
        qb, kk.reshape(B, H * nb, 2 * w, dh), vv.reshape(B, H * nb, 2 * w, dh),
        attn_mask=m.repeat(H, 1, 1),
    )
    return out.reshape(B, H, nb * w, dh)[:, :, :T]


def infini_attention(q, k, v, beta, segment: int):
    """Infini-attention (Munkhdalai et al., 2024), linear + delta memory."""
    B, H, T, dh = q.shape
    S = segment
    ns = -(-T // S)
    pad = ns * S - T
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    local = F.scaled_dot_product_attention(
        q.reshape(B, H * ns, S, dh), k.reshape(B, H * ns, S, dh),
        v.reshape(B, H * ns, S, dh), is_causal=True,
    ).reshape(B, H, ns * S, dh)
    with _no_autocast(q):
        sq = F.elu(q.float()) + 1.0
        sk = F.elu(k.float()) + 1.0
        vf = v.float()
        M = q.new_zeros((B, H, dh, dh), dtype=torch.float32)
        z = q.new_zeros((B, H, 1, dh), dtype=torch.float32)
        mem = []
        for s in range(ns):
            sl = slice(s * S, (s + 1) * S)
            qs, ks, vs = sq[:, :, sl], sk[:, :, sl], vf[:, :, sl]
            mem.append((qs @ M) / ((qs * z).sum(-1, keepdim=True) + 1e-6))
            retrieved = (ks @ M) / ((ks * z).sum(-1, keepdim=True) + 1e-6)
            M = M + ks.transpose(-1, -2) @ (vs - retrieved)
            z = z + ks.sum(2, keepdim=True)
        a_mem = torch.cat(mem, dim=2)
        g = torch.sigmoid(beta.float())[None, :, None, None]
        out = g * a_mem + (1 - g) * local.float()
    return out[:, :, :T].to(q.dtype)


def _no_autocast(t: torch.Tensor):
    return torch.autocast(device_type=t.device.type, enabled=False)


# ----------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d, h = spec["d_model"], spec["n_heads"]
        self.h, self.dh = h, d // h
        self.pe = spec["pe"]
        self.window = spec["window"]
        self.segment = spec["infini_segment"]
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        if self.segment:
            self.beta = nn.Parameter(torch.zeros(h))
        if self.pe == "alibi":
            self.register_buffer("slopes", alibi_slopes(h), persistent=False)

    def forward(self, x, ctx):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.pe == "rope":
            q, k = apply_rope(q, ctx["cos"], ctx["sin"]), apply_rope(k, ctx["cos"], ctx["sin"])
        if self.segment:
            o = infini_attention(q, k, v, self.beta, self.segment)
        elif self.window:
            o = sliding_window_attention(q, k, v, self.window)
        elif self.pe == "alibi":
            o = alibi_attention(q, k, v, self.slopes)
        else:
            o = causal_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc = nn.Linear(d, hidden)
        self.out = nn.Linear(hidden, d)

    def forward(self, x):
        return self.out(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d = spec["d_model"]
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(spec)
        self.mlp = MLP(d, spec["mlp_mult"] * d)

    def forward(self, h, ctx):
        h = h + self.attn(self.ln1(h), ctx)
        return h + self.mlp(self.ln2(h))


def memory_param_count(d: int) -> int:
    m = DABSNHostMemory(d, kernels="reference")
    return sum(p.numel() for p in m.parameters())


class Insert(nn.Module):
    """Residual insert ``h + branch(LN(h))`` under the host's autocast."""

    def __init__(self, d: int, kind: str, kernels: str):
        super().__init__()
        self.kind = kind
        self.ln = nn.LayerNorm(d)
        if kind == "dabsn":
            self.branch = DABSNHostMemory(d, kernels=kernels)
        elif kind == "mlp_matched":
            target = memory_param_count(d)
            hidden = max(1, round((target - d) / (2 * d + 1)))
            self.branch = MLP(d, hidden)
        else:
            raise ValueError(kind)

    def forward(self, h):
        # The branch runs under the host's autocast (bf16 on the GPU), exactly
        # like the rest of the model and like the original plugin.
        return h + self.branch(self.ln(h))


class Transformer(nn.Module):
    def __init__(self, spec: dict, memory_kernels: str = "auto"):
        super().__init__()
        self.spec = dict(spec)
        d, L = spec["d_model"], spec["n_layers"]
        self.embed = nn.Embedding(spec["vocab"], d)
        self.blocks = nn.ModuleList(Block(spec) for _ in range(L))
        where = {"front": 0, "middle": L // 2, "back": L}
        self.inserts = nn.ModuleDict()
        for p in spec["placements"]:
            self.inserts[str(where[p])] = Insert(d, spec["insert"], memory_kernels)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, spec["vocab"], bias=False)

    def _ctx(self, T, device, rope_mode, train_len):
        s = self.spec
        if s["pe"] != "rope":
            return {}
        dh = s["d_model"] // s["n_heads"]
        if s["infini_segment"]:
            pos = torch.arange(T, device=device) % s["infini_segment"]
            cos, sin = rope_tables(pos, dh, s["rope_base"], "base", None, T)
        elif s.get("rope_scaling"):
            # YaRN-fine-tuned RoPE: the scaling is part of the model.
            sc = s["rope_scaling"]
            pos = torch.arange(T, device=device)
            cos, sin = rope_tables(pos, dh, s["rope_base"], "yarn", sc["orig_len"], T,
                                   min_scale=sc["min_scale"])
        else:
            pos = torch.arange(T, device=device)
            cos, sin = rope_tables(pos, dh, s["rope_base"], rope_mode, train_len, T)
        return {"cos": cos, "sin": sin}

    def hidden(self, x, rope_mode: str = "base", train_len: int | None = None):
        h = self.embed(x)
        ctx = self._ctx(x.shape[1], x.device, rope_mode, train_len)
        for i, blk in enumerate(self.blocks):
            if str(i) in self.inserts:
                h = self.inserts[str(i)](h)
            h = blk(h, ctx)
        last = str(len(self.blocks))
        if last in self.inserts:
            h = self.inserts[last](h)
        return self.ln_f(h)

    def memory_stats(self) -> list[dict]:
        out = []
        for key, ins in self.inserts.items():
            if ins.kind == "dabsn":
                c = ins.branch.read_contract
                out.append({
                    "at": int(key),
                    "admitted_bank": c.get("max_admitted_bank"),
                    "read_work": c.get("estimated_read_work"),
                })
        return out


def param_groups(model: Transformer, weight_decay: float):
    """Decay only the 2-D weights of attention, block MLPs and the head.

    Embeddings, norms, biases, Infini gates and every residual insert (the DABSN
    memory and its parameter-matched MLP control alike) are not decayed.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_block_matrix = name.startswith("blocks.") and p.ndim == 2
        if in_block_matrix or name == "head.weight":
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
