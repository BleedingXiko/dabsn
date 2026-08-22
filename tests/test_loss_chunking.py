"""CPU numerics tests for the chunked fused linear+cross-entropy (Phase 4).

The chunked loss must match the naive readout+F.cross_entropy in value and in
every gradient (hidden, readout weight, readout bias) within FP32 tolerance, and
be invariant to the chunk size -- otherwise the memory win would change the math.
"""

from __future__ import annotations

import pytest
import torch

from dabsn.runtime.loss import (
    chunked_cross_entropy_from_logits,
    chunked_linear_cross_entropy,
)


def _naive(hidden, weight, bias, targets, ignore_index=-100):
    import torch.nn.functional as F

    # Preserve the input precision (FP64 in these tests) so the comparison
    # against the chunked loss -- which promotes to at least FP32, i.e. keeps
    # FP64 here -- is an exactness check, not a precision-mismatch.
    logits = F.linear(hidden, weight, bias)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=ignore_index
    )


def test_forward_matches_naive_cross_entropy():
    torch.manual_seed(0)
    B, T, H, V = 2, 6, 8, 40
    hidden = torch.randn(B, T, H, dtype=torch.float64)
    weight = torch.randn(V, H, dtype=torch.float64)
    bias = torch.randn(V, dtype=torch.float64)
    targets = torch.randint(0, V, (B, T))
    ref = _naive(hidden, weight, bias, targets)
    # Force chunking with a tiny budget.
    got = chunked_linear_cross_entropy(hidden, weight, bias, targets, chunk_budget=1)
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("chunk_budget", [1, 7 * 40, 10_000])
def test_backward_grads_match_and_are_chunk_invariant(chunk_budget):
    torch.manual_seed(1)
    B, T, H, V = 3, 5, 7, 50
    hidden = torch.randn(B, T, H, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(V, H, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(V, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, V, (B, T))

    hidden_r = hidden.detach().clone().requires_grad_(True)
    weight_r = weight.detach().clone().requires_grad_(True)
    bias_r = bias.detach().clone().requires_grad_(True)

    loss = chunked_linear_cross_entropy(hidden, weight, bias, targets, chunk_budget=chunk_budget)
    loss.backward()
    ref = _naive(hidden_r, weight_r, bias_r, targets)
    ref.backward()

    torch.testing.assert_close(loss, ref, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(hidden.grad, hidden_r.grad, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(weight.grad, weight_r.grad, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(bias.grad, bias_r.grad, rtol=1e-8, atol=1e-8)


def test_ignore_index_masks_positions():
    torch.manual_seed(2)
    B, T, H, V = 2, 8, 6, 30
    hidden = torch.randn(B, T, H, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(V, H, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(V, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    targets[:, ::2] = -100  # mask half the positions

    hidden_r = hidden.detach().clone().requires_grad_(True)
    weight_r = weight.detach().clone().requires_grad_(True)
    bias_r = bias.detach().clone().requires_grad_(True)

    loss = chunked_linear_cross_entropy(
        hidden, weight, bias, targets, chunk_budget=1, ignore_index=-100
    )
    loss.backward()
    ref = _naive(hidden_r, weight_r, bias_r, targets, ignore_index=-100)
    ref.backward()

    torch.testing.assert_close(loss, ref, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(hidden.grad, hidden_r.grad, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(weight.grad, weight_r.grad, rtol=1e-8, atol=1e-8)


def test_small_model_takes_exact_single_shot_path():
    # Under the budget, the exact F.cross_entropy path is used (no Function).
    torch.manual_seed(3)
    B, T, H, V = 2, 4, 5, 16
    hidden = torch.randn(B, T, H, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(V, H, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(V, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    # Huge budget -> single shot; still must equal naive.
    got = chunked_linear_cross_entropy(hidden, weight, bias, targets, chunk_budget=1 << 30)
    ref = _naive(hidden, weight, bias, targets)
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-9)


def test_from_logits_variant_matches():
    torch.manual_seed(4)
    N, V = 37, 60
    logits = torch.randn(N, V, dtype=torch.float64)
    targets = torch.randint(0, V, (N,))
    import torch.nn.functional as F

    ref = F.cross_entropy(logits, targets)
    got = chunked_cross_entropy_from_logits(logits, targets, chunk_budget=1)
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-9)


def test_off_env_forces_exact(monkeypatch):
    monkeypatch.setenv("DABSN_LOSS_CHUNK_SCORES", "off")
    torch.manual_seed(5)
    B, T, H, V = 2, 6, 8, 40
    hidden = torch.randn(B, T, H, dtype=torch.float64)
    weight = torch.randn(V, H, dtype=torch.float64)
    bias = torch.randn(V, dtype=torch.float64)
    targets = torch.randint(0, V, (B, T))
    got = chunked_linear_cross_entropy(hidden, weight, bias, targets)
    ref = _naive(hidden, weight, bias, targets)
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-9)
