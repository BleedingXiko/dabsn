"""Memory-frugal fused linear + cross-entropy for language-model training.

At vocabulary V=50k the readout logits ``[B, T, V]`` -- and the FP32 upcast
cross-entropy makes of them -- are the single largest activation in a DABSN
training step, dwarfing the recurrent tapes. This module computes the identical
mean next-token loss without ever materializing the full logits: it walks the
flattened ``B*T`` rows in chunks, forming ``[chunk, V]`` logits, reducing each
chunk to per-row ``logsumexp - gathered`` in FP32, and recomputing the same
chunks in the backward to build the parameter and hidden-state gradients. Peak
logit memory drops from ``B*T*V`` to ``chunk*V``.

The result is tolerance-bound, not bit-identical, to ``F.cross_entropy`` (the
running-sum reduction order differs), and the exact single-shot path is always
available as a fallback -- chunking auto-engages only when the full logits would
exceed the element budget.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import Tensor

# Default element budget for a single logit chunk (rows * V). 2**24 ~= 16.7M
# elements = 67 MB in FP32; small enough to bound peak, large enough to keep the
# readout GEMM efficient.
_DEFAULT_CHUNK_BUDGET = 1 << 24


def _chunk_rows(num_rows: int, vocab: int, budget: int | None = None) -> int:
    """Rows per chunk so that rows*V stays within the element budget."""
    if budget is None:
        budget = int(os.environ.get("DABSN_LOSS_CHUNK_SCORES", str(_DEFAULT_CHUNK_BUDGET)))
    if budget <= 0:
        return num_rows
    rows = max(1, budget // max(1, vocab))
    return min(rows, num_rows)


def _accum_dtype(dtype: torch.dtype) -> torch.dtype:
    """Compute precision for the loss: at least FP32, but keep FP64 if given.

    Upcasting bf16/fp16 logits to FP32 is the numerically stable default; a
    higher-precision input (FP64, used by the exactness tests) is preserved so
    the chunked math can be checked against the reference to full precision.
    """
    return torch.promote_types(dtype, torch.float32)


class ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Fused readout + mean cross-entropy that never materializes ``[N, V]``.

    Forward saves only ``hidden, weight, bias, targets`` (never the logits);
    backward recomputes each chunk's logits and softmax to form the grads.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,  # [N, H] (already flattened)
        weight: Tensor,  # [V, H]
        bias: Tensor | None,  # [V] or None
        targets: Tensor,  # [N] int64
        chunk_rows: int,
        ignore_index: int,
    ) -> Tensor:
        n_rows, _ = hidden.shape
        vocab = weight.shape[0]
        acc = _accum_dtype(hidden.dtype)
        loss_sum = hidden.new_zeros((), dtype=acc)
        valid_count = hidden.new_zeros((), dtype=acc)
        for start in range(0, n_rows, chunk_rows):
            stop = min(n_rows, start + chunk_rows)
            hid_c = hidden[start:stop]
            tgt_c = targets[start:stop]
            logits_c = F.linear(hid_c, weight, bias).to(acc)
            valid = tgt_c != ignore_index
            # Clamp ignored targets to a valid index for the gather; their
            # contribution is masked out below.
            safe_tgt = torch.where(valid, tgt_c, torch.zeros_like(tgt_c))
            lse = torch.logsumexp(logits_c, dim=-1)
            gathered = logits_c.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)
            nll = (lse - gathered) * valid.to(logits_c.dtype)
            loss_sum = loss_sum + nll.sum()
            valid_count = valid_count + valid.sum().to(acc)
        denom = valid_count.clamp_min(1.0)
        ctx.save_for_backward(hidden, weight, bias, targets, denom)
        ctx.chunk_rows = int(chunk_rows)
        ctx.ignore_index = int(ignore_index)
        ctx.has_bias = bias is not None
        ctx.vocab = int(vocab)
        ctx.acc = acc
        return loss_sum / denom

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, bias, targets, denom = ctx.saved_tensors
        chunk_rows = ctx.chunk_rows
        ignore_index = ctx.ignore_index
        acc = ctx.acc
        n_rows, _ = hidden.shape
        scale = (grad_output / denom).to(acc)
        weight_acc = weight.to(acc)

        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight, dtype=acc)
        grad_bias = torch.zeros_like(bias, dtype=acc) if ctx.has_bias else None
        for start in range(0, n_rows, chunk_rows):
            stop = min(n_rows, start + chunk_rows)
            hid_c = hidden[start:stop]
            tgt_c = targets[start:stop]
            logits_c = F.linear(hid_c, weight, bias).to(acc)
            valid = tgt_c != ignore_index
            safe_tgt = torch.where(valid, tgt_c, torch.zeros_like(tgt_c))
            probs = torch.softmax(logits_c, dim=-1)
            # d/dlogit CE = softmax - onehot(target), masked for ignore_index.
            probs.scatter_add_(
                -1,
                safe_tgt.unsqueeze(-1),
                -torch.ones_like(safe_tgt, dtype=probs.dtype).unsqueeze(-1),
            )
            probs = probs * valid.to(probs.dtype).unsqueeze(-1) * scale
            # grad wrt hidden chunk: probs @ weight, cast back to hidden dtype.
            grad_hidden[start:stop] = (probs @ weight_acc).to(hidden.dtype)
            grad_weight += probs.transpose(0, 1) @ hid_c.to(acc)
            if grad_bias is not None:
                grad_bias += probs.sum(dim=0)

        grad_weight = grad_weight.to(weight.dtype)
        if grad_bias is not None:
            grad_bias = grad_bias.to(bias.dtype)
        # gradients for: hidden, weight, bias, targets, chunk_rows, ignore_index
        return grad_hidden, grad_weight, grad_bias, None, None, None


def chunked_linear_cross_entropy(
    hidden: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    targets: Tensor,
    *,
    ignore_index: int = -100,
    chunk_budget: int | None = None,
) -> Tensor:
    """Mean next-token cross-entropy of ``readout(hidden)`` vs ``targets``.

    ``hidden`` is ``[B, T, H]`` or ``[N, H]``; ``targets`` matches without the H
    axis. Auto-engages chunking only when the full logits ``N*V`` exceed the
    element budget, so small models keep the exact single-shot path.
    ``DABSN_LOSS_CHUNK_SCORES=off`` forces the exact fallback.
    """
    if hidden.dim() == 3:
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        targets_flat = targets.reshape(-1)
    else:
        hidden_flat = hidden
        targets_flat = targets
    n_rows = hidden_flat.shape[0]
    vocab = weight.shape[0]

    acc = _accum_dtype(hidden_flat.dtype)
    env = os.environ.get("DABSN_LOSS_CHUNK_SCORES", "")
    if env.lower() == "off":
        logits = F.linear(hidden_flat, weight, bias).to(acc)
        return F.cross_entropy(logits, targets_flat, ignore_index=ignore_index)

    budget = chunk_budget
    if budget is None and env:
        try:
            budget = int(env)
        except ValueError:
            budget = None
    effective_budget = budget if budget is not None else _DEFAULT_CHUNK_BUDGET
    if n_rows * vocab <= effective_budget:
        # Small enough to compute exactly in one shot (still upcast CE).
        logits = F.linear(hidden_flat, weight, bias).to(acc)
        return F.cross_entropy(logits, targets_flat, ignore_index=ignore_index)

    rows = _chunk_rows(n_rows, vocab, budget)
    return ChunkedLinearCrossEntropy.apply(
        hidden_flat, weight, bias, targets_flat, rows, ignore_index
    )


def chunked_cross_entropy_from_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int = -100,
    chunk_budget: int | None = None,
) -> Tensor:
    """Mean cross-entropy over already-materialized logits without the full
    FP32 upcast peak.

    Use this where logits already exist (e.g. a graphed forward returned them):
    it chunks the FP32 upcast so only ``chunk*V`` FP32 elements live at once,
    instead of a full ``[N, V]`` FP32 copy alongside the bf16 logits.
    """
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    n_rows, vocab = logits_flat.shape

    env = os.environ.get("DABSN_LOSS_CHUNK_SCORES", "")
    budget = chunk_budget
    if budget is None and env and env.lower() != "off":
        try:
            budget = int(env)
        except ValueError:
            budget = None
    effective_budget = budget if budget is not None else _DEFAULT_CHUNK_BUDGET
    acc = _accum_dtype(logits_flat.dtype)
    if env.lower() == "off" or n_rows * vocab <= effective_budget:
        return F.cross_entropy(logits_flat.to(acc), targets_flat, ignore_index=ignore_index)

    rows = _chunk_rows(n_rows, vocab, budget)
    loss_sum = logits_flat.new_zeros((), dtype=acc)
    valid_count = logits_flat.new_zeros((), dtype=acc)
    for start in range(0, n_rows, rows):
        stop = min(n_rows, start + rows)
        logits_c = logits_flat[start:stop].to(acc)
        tgt_c = targets_flat[start:stop]
        valid = tgt_c != ignore_index
        safe_tgt = torch.where(valid, tgt_c, torch.zeros_like(tgt_c))
        lse = torch.logsumexp(logits_c, dim=-1)
        gathered = logits_c.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)
        nll = (lse - gathered) * valid.to(logits_c.dtype)
        loss_sum = loss_sum + nll.sum()
        valid_count = valid_count + valid.sum().to(torch.float32)
    return loss_sum / valid_count.clamp_min(1.0)
