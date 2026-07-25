"""Phase 6: time-chunked core scan in a DABSNBlock must match the unchunked
block exactly in the forward and within tolerance in the backward.

The core carries (budget, energy, saturation) across chunk boundaries, so the
assembled full-T tapes are identical to a single scan -- the same carried-state
parity the release gate pins. Backward differs only by checkpoint recompute
reduction order.
"""

from __future__ import annotations

import copy

import pytest
import torch

from dabsn.model import DABSNBlock


@pytest.mark.parametrize("geometry", ["seq", "field", "hybrid"])
@pytest.mark.parametrize("chunk_t", [1, 3, 5])
def test_chunked_core_matches_unchunked(monkeypatch, geometry, chunk_t):
    torch.manual_seed(0)
    block = DABSNBlock(input_dim=6, hidden_dim=7, state_dim=7, read_geometry=geometry)
    block.train()
    chunked = copy.deepcopy(block)

    x = torch.randn(2, 8, 6, dtype=torch.float64, requires_grad=True)
    x_ck = x.detach().clone().requires_grad_(True)

    # Cast both to float64 for a tight exactness check.
    block.double()
    chunked.double()

    monkeypatch.delenv("DABSN_BLOCK_CHUNK_T", raising=False)
    out = block(x)

    monkeypatch.setenv("DABSN_BLOCK_CHUNK_T", str(chunk_t))
    out_ck = chunked(x_ck)

    # Forward is bit-identical: carried-state chunking reproduces the full scan.
    torch.testing.assert_close(out_ck, out, rtol=1e-10, atol=1e-10)

    out.sum().backward()
    out_ck.sum().backward()
    # Backward is tolerance-bound (checkpoint recompute reduction order).
    torch.testing.assert_close(x_ck.grad, x.grad, rtol=1e-6, atol=1e-6)
    for (_, p), (_, p_ck) in zip(block.named_parameters(), chunked.named_parameters()):
        if p.grad is None:
            continue
        torch.testing.assert_close(p_ck.grad, p.grad, rtol=1e-6, atol=1e-6)


def test_block_chunk_off_is_unchunked(monkeypatch):
    # -1 forces the unchunked path even if a width was otherwise chosen.
    torch.manual_seed(1)
    block = DABSNBlock(input_dim=4, hidden_dim=5, state_dim=5, read_geometry="seq").double()
    x = torch.randn(2, 6, 4, dtype=torch.float64)
    monkeypatch.setenv("DABSN_BLOCK_CHUNK_T", "-1")
    off = block(x)
    monkeypatch.delenv("DABSN_BLOCK_CHUNK_T", raising=False)
    default = block(x)
    torch.testing.assert_close(off, default, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# CUDA-graph capture safety
#
# Checkpoint recompute cannot be captured: it re-runs forward during backward
# against capture-pool addresses that no longer hold those activations at
# replay, which surfaces as cudaErrorIllegalAddress at the next sync -- and the
# *auto* chunk decision reads live free memory, so warmup and capture can
# disagree on the chunk count and record a graph whose structure never matched.
# Both hazards are attribute-level, so both are provable without a GPU.
# ---------------------------------------------------------------------------


def test_capture_no_chunk_overrides_every_setting(monkeypatch):
    block = DABSNBlock(input_dim=4, hidden_dim=5, state_dim=5, read_geometry="seq")
    x = torch.randn(2, 16, 4)

    monkeypatch.setenv("DABSN_BLOCK_CHUNK_T", "4")
    assert block._resolve_block_chunk_t(x) == 4

    # Capture wins over an explicit width rather than crashing inside the graph.
    block._capture_no_chunk = True
    assert block._resolve_block_chunk_t(x) == 0
    monkeypatch.delenv("DABSN_BLOCK_CHUNK_T", raising=False)
    assert block._resolve_block_chunk_t(x) == 0

    del block._capture_no_chunk
    monkeypatch.setenv("DABSN_BLOCK_CHUNK_T", "4")
    assert block._resolve_block_chunk_t(x) == 4


def test_auto_chunk_decision_is_stable_per_shape(monkeypatch):
    # The auto decision must be made once per execution shape. Re-deciding from
    # whatever free memory exists at that instant would let the same shape chunk
    # on one step and not the next -- nondeterministic structure, and exactly the
    # warmup/capture mismatch that corrupts a recorded graph.
    monkeypatch.delenv("DABSN_BLOCK_CHUNK_T", raising=False)
    block = DABSNBlock(input_dim=4, hidden_dim=5, state_dim=5, read_geometry="seq")
    x = torch.randn(2, 16, 4)

    # Seed the cache with a decision, then prove a later call reuses it instead
    # of re-querying the device.
    block._resolve_block_chunk_t(x)
    cache = block.__dict__["_block_chunk_auto"]
    key = (2, 16, 5, x.dtype)
    assert key in cache
    cache[key] = 7
    assert block._resolve_block_chunk_t(x) == 7
    # A different shape is a different decision.
    assert (3, 16, 5, x.dtype) not in cache


def test_pin_capture_safe_recompute_disables_and_restores():
    from dabsn.model import DABSNBackbone
    from dabsn.runtime.graph import _pin_capture_safe_recompute

    backbone = DABSNBackbone(4, [{"hidden_dim": 5, "state_dim": 5}], True)
    blocks = list(backbone.blocks)
    assert backbone.grad_checkpoint is True
    assert not any(getattr(b, "_capture_no_chunk", False) for b in blocks)

    restore = _pin_capture_safe_recompute(backbone)
    assert backbone.grad_checkpoint is False
    assert all(b._capture_no_chunk for b in blocks)

    restore()
    assert backbone.grad_checkpoint is True
    # The attribute is removed, not left False, so nothing lingers on the module.
    assert not any(hasattr(b, "_capture_no_chunk") for b in blocks)
