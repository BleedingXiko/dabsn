"""CPU-provable unit tests for work-aware core-scan dispatch.

The core-scan backend must be chosen from the *execution shape* (B, H), not a
hard ``B >= 64`` rule that stranded large-H / small-microbatch training on the
GEMV persistent path. These tests pin the decision function so a regression in
the dispatch policy is caught without a GPU.
"""

from __future__ import annotations

import pytest

from dabsn.kernels.triton import select_core_backend


def test_no_grad_routes_on_work_not_on_grad_mode():
    """No-grad is not automatically a latency workload.

    Routing every no-grad call to the persistent GEMV scan put batch inference
    and evaluation off the tensor cores entirely: at batch 256 a forward pass
    measured SLOWER than the same shapes with autograd enabled, because the grad
    path got the batched GEMM and the no-grad path did not. The work floor
    already separates latency shapes from throughput shapes.
    """
    # Batch inference at a real width belongs on the GEMM scan.
    assert select_core_backend(1024, 4096, requested="auto", grad_enabled=False) == "batched"
    assert select_core_backend(256, 2048, requested="auto", grad_enabled=False) == "batched"
    # A single sequence is still a latency workload.
    assert select_core_backend(1, 8, requested="auto", grad_enabled=False) == "persistent"
    assert select_core_backend(1, 512, requested="auto", grad_enabled=False) == "persistent"
    # Explicit requests are still honored.
    assert select_core_backend(1, 8, requested="batched", grad_enabled=False) == "batched"
    # ...except the fused Function, whose saved backward context is dead weight
    # under no_grad: same math on the plain batched GEMM instead.
    assert select_core_backend(64, 128, requested="batched_fused", grad_enabled=False) == "batched"


def test_large_batch_small_hidden_uses_fused():
    # The original B>=64 rule: big device batch, GEMM amortizes. At these narrow
    # widths the state fits one register tile, so the single-launch fused scan --
    # parity-proven against the reference on CPU and on an A100 -- is the pick.
    assert select_core_backend(64, 64, requested="auto") == "batched_fused"
    assert select_core_backend(128, 32, requested="auto") == "batched_fused"
    # Past the one-tile width the fused kernel cannot carry the state, so the
    # unbounded batched GEMM scan takes over.
    assert select_core_backend(64, 1024, requested="auto") == "batched"


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
    assert (
        select_core_backend(4, 1023, requested="auto", min_batch=64, min_work=4096) == "persistent"
    )
    # Raising the floor keeps a mid-size shape on the persistent path.
    assert (
        select_core_backend(4, 2048, requested="auto", min_batch=64, min_work=1 << 20)
        == "persistent"
    )


def test_explicit_overrides_win():
    assert select_core_backend(1, 8, requested="batched") == "batched"
    assert select_core_backend(4096, 8192, requested="persistent") == "persistent"


def test_fused_never_selected_past_its_width_or_below_the_work_floor():
    # auto reaches for the fused kernel only where it is actually valid: never
    # past the one-tile width, and never before there is tensor-core work.
    for b, h in [(4, 4096), (256, 1024), (1, 8)]:
        assert select_core_backend(b, h, requested="auto") != "batched_fused"
    # An explicit request at a width the fused kernel supports routes to it.
    assert select_core_backend(4, 128, requested="batched_fused") == "batched_fused"
    assert select_core_backend(4, 256, requested="batched_fused") == "batched_fused"
    # Under no_grad the fused Function's saved tape is dead weight -> batched.
    assert select_core_backend(4, 128, requested="batched_fused", grad_enabled=False) == "batched"


def test_explicit_fused_too_wide_routes_to_batched():
    # H beyond the single-launch width would hard-fail deep in the kernel;
    # instead route to the width-safe batched scan (a one-time warning explains
    # why) so training never crashes on an unactionable error.
    assert select_core_backend(4, 2048, requested="batched_fused") == "batched"
    assert select_core_backend(4, 257, requested="batched_fused") == "batched"
    # The width bound is configurable.
    assert (
        select_core_backend(4, 300, requested="batched_fused", fused_max_h=512) == "batched_fused"
    )


def test_auto_selects_fused_when_width_safe_and_work_floor_cleared():
    # Width-safe + enough work -> the single-launch fused scan.
    # (16,256): B*H=4096 clears the work floor and H=256 is one-tile-safe.
    assert select_core_backend(16, 256, requested="auto") == "batched_fused"
    # too wide for one tile -> batched GEMM scan (no width bound there)
    assert select_core_backend(4, 2048, requested="auto") == "batched"
    # width-safe but below the work floor -> persistent (no tensor-core work yet)
    assert select_core_backend(1, 64, requested="auto") == "persistent"
    # the width bound is configurable, and auto respects it
    assert select_core_backend(16, 512, requested="auto", fused_max_h=512) == "batched_fused"
    assert select_core_backend(16, 512, requested="auto", fused_max_h=256) == "batched"


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        select_core_backend(4, 128, requested="nonsense")


def test_env_overrides_thresholds(monkeypatch):
    monkeypatch.setenv("DABSN_BATCHED_CORE_MIN_WORK", "1000000000")
    monkeypatch.setenv("DABSN_BATCHED_CORE_MIN_BATCH", "1000000")
    # With both thresholds pushed sky-high, auto falls back to persistent.
    assert select_core_backend(8, 1024, requested="auto") == "persistent"
