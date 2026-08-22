"""CPU-provable tests for Phase 3 routing observability + OOM robustness.

None of this needs CUDA: the log-once contract, the allocator-config helper,
and the actionable-OOM message builder are all pure host logic.
"""

from __future__ import annotations

import logging

from dabsn.runtime.dispatch import (
    log_routing_once,
    reset_routing_log,
    warn_routing_once,
)


def test_log_routing_once_emits_once_per_decision(caplog):
    reset_routing_log()
    with caplog.at_level(logging.INFO, logger="dabsn.dispatch"):
        log_routing_once("core_scan", "batched", batch=8, hidden=1024)
        log_routing_once("core_scan", "batched", batch=8, hidden=1024)
        log_routing_once("core_scan", "batched", batch=8, hidden=2048)  # new shape
    records = [r for r in caplog.records if r.name == "dabsn.dispatch"]
    assert len(records) == 2  # duplicate suppressed, new shape logged


def test_warn_routing_once_is_deduped(caplog):
    reset_routing_log()
    with caplog.at_level(logging.WARNING, logger="dabsn.dispatch"):
        for _ in range(5):
            warn_routing_once("core_scan_fused", "fused unavailable (H>256)", hidden=512)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_silence_env_mutes_routing(caplog, monkeypatch):
    reset_routing_log()
    monkeypatch.setenv("DABSN_SILENCE_ROUTING", "1")
    with caplog.at_level(logging.INFO, logger="dabsn.dispatch"):
        log_routing_once("core_scan", "batched", batch=1, hidden=8)
        warn_routing_once("core_scan_fused", "anything", hidden=8)
    assert not [r for r in caplog.records if r.name == "dabsn.dispatch"]


def test_configure_cuda_allocator_appends_without_clobbering(monkeypatch):
    import importlib.util
    import os

    tool = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "scale_bench.py"
    )
    spec = importlib.util.spec_from_file_location("scale_bench_alloc", tool)
    import sys

    sb = importlib.util.module_from_spec(spec)
    sys.modules["scale_bench_alloc"] = sb
    spec.loader.exec_module(sb)

    # Fresh: adds expandable_segments.
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("DABSN_DISABLE_EXPANDABLE_SEGMENTS", raising=False)
    sb._configure_cuda_allocator()
    assert "expandable_segments:True" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    # Existing config is preserved, not clobbered.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    sb._configure_cuda_allocator()
    conf = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    assert "max_split_size_mb:128" in conf and "expandable_segments:True" in conf

    # Opt-out is honored.
    monkeypatch.setenv("DABSN_DISABLE_EXPANDABLE_SEGMENTS", "1")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    sb._configure_cuda_allocator()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:64"


def test_oom_message_is_actionable():
    from types import SimpleNamespace

    from dabsn.config import DABSNPretrainConfig
    from dabsn.pretrain import _oom_actionable_message

    config = DABSNPretrainConfig(corpus_bin="dummy.bin")
    state = SimpleNamespace(device=None)
    msg = _oom_actionable_message(config, state, RuntimeError("CUDA out of memory"), "pretraining")
    # Names the live shape and offers ranked, concrete levers.
    assert "batch=" in msg and "grad_accum=" in msg and "precision=" in msg
    assert "grad_accum_steps" in msg  # a concrete lever
    assert "1." in msg and "2." in msg  # ranked list
