"""CPU-provable unit tests for work-aware core-scan dispatch.

The core-scan backend must be chosen from the *execution shape* (B, H), not a
hard ``B >= 64`` rule that stranded large-H / small-microbatch training on the
GEMV persistent path. These tests pin the decision function so a regression in
the dispatch policy is caught without a GPU.
"""

from __future__ import annotations

import pytest

from dabsn.kernels.triton import select_core_backend


def test_no_grad_always_persistent():
    # Inference/eval never wants the training GEMM scans regardless of shape.
    assert select_core_backend(1024, 4096, requested="auto", grad_enabled=False) == "persistent"
    assert select_core_backend(1, 8, requested="batched", grad_enabled=False) == "persistent"


def test_large_batch_small_hidden_uses_batched():
    # The original B>=64 rule: big device batch, GEMM amortizes.
    assert select_core_backend(64, 64, requested="auto") == "batched"
    assert select_core_backend(128, 32, requested="auto") == "batched"


def test_small_microbatch_large_hidden_uses_batched():
    # The defect this fixes: a 1B model uses microbatch 4-16 with grad-accum,
    # yet H is large -- per-sequence GEMV is catastrophic. Work-aware dispatch
    # must route these to the tensor-core batched scan.
    assert select_core_backend(4, 2048, requested="auto") == "batched"
    assert select_core_backend(8, 1024, requested="auto") == "batched"
    assert select_core_backend(2, 4096, requested="auto") == "batched"


def test_small_model_small_batch_stays_persistent():
    # A genuinely small model at latency batch keeps the low-launch persistent
    # scan: neither the batch nor the work floor is cleared.
    assert select_core_backend(4, 64, requested="auto") == "persistent"
    assert select_core_backend(1, 512, requested="auto") == "persistent"


def test_work_floor_boundary_is_configurable():
    # B*H == min_work is inclusive; just under stays persistent.
    assert select_core_backend(4, 1024, requested="auto", min_batch=64, min_work=4096) == "batched"
    assert select_core_backend(4, 1023, requested="auto", min_batch=64, min_work=4096) == "persistent"
    # Raising the floor keeps a mid-size shape on the persistent path.
    assert select_core_backend(4, 2048, requested="auto", min_batch=64, min_work=1 << 20) == "persistent"


def test_explicit_overrides_win():
    assert select_core_backend(1, 8, requested="batched") == "batched"
    assert select_core_backend(4096, 8192, requested="persistent") == "persistent"


def test_batched_fused_only_by_explicit_request():
    # auto must never select the (uncertified) single-launch fused kernel...
    for b, h in [(4, 4096), (256, 1024), (1, 8)]:
        assert select_core_backend(b, h, requested="auto") != "batched_fused"
    # ...but an explicit request routes to it.
    assert select_core_backend(4, 2048, requested="batched_fused") == "batched_fused"
    # and no-grad still collapses to persistent.
    assert select_core_backend(4, 2048, requested="batched_fused", grad_enabled=False) == "persistent"


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        select_core_backend(4, 128, requested="nonsense")


def test_env_overrides_thresholds(monkeypatch):
    monkeypatch.setenv("DABSN_BATCHED_CORE_MIN_WORK", "1000000000")
    monkeypatch.setenv("DABSN_BATCHED_CORE_MIN_BATCH", "1000000")
    # With both thresholds pushed sky-high, auto falls back to persistent.
    assert select_core_backend(8, 1024, requested="auto") == "persistent"
