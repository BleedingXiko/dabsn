"""Reproducible performance reports and confidence-bound release gates."""

from __future__ import annotations

import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class HardwareFingerprint:
    platform: str
    machine: str
    python: str
    torch: str
    device: str
    accelerator: str | None
    compute_capability: str | None


@dataclass(frozen=True)
class PerformanceReport:
    name: str
    configuration: Mapping[str, object]
    hardware: HardwareFingerprint
    warmup: int
    trials: int
    synchronized: bool
    samples_seconds: tuple[float, ...]
    mean_seconds: float
    ci95_seconds: float
    throughput_per_second: float
    peak_memory_bytes: int | None
    graph_count: int | None = None
    compile_seconds: float | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["hardware"] = asdict(self.hardware)
        return data


@dataclass(frozen=True)
class PerformanceComparison:
    passed: bool
    candidate_ratio: float
    conservative_regression: float
    permitted_regression: float
    reason: str


@dataclass(frozen=True)
class OperatorTrace:
    operator_counts: Mapping[str, int]
    tensor_copy_events: int
    synchronization_events: int


@dataclass(frozen=True)
class OperatorTraceComparison:
    passed: bool
    added_operators: Mapping[str, int]
    added_tensor_copy_events: int
    added_synchronization_events: int


def hardware_fingerprint(device: torch.device) -> HardwareFingerprint:
    accelerator = None
    capability = None
    if device.type == "cuda" and torch.cuda.is_available():
        accelerator = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        capability = f"sm{major}{minor}"
    return HardwareFingerprint(
        platform=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        torch=torch.__version__,
        device=str(device),
        accelerator=accelerator,
        compute_capability=capability,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    name: str,
    function: Callable[[], object],
    *,
    work_items: int,
    configuration: Mapping[str, object],
    device: str | torch.device = "cpu",
    warmup: int = 5,
    trials: int = 30,
    graph_count: int | None = None,
    compile_seconds: float | None = None,
) -> PerformanceReport:
    if warmup < 1:
        raise ValueError("performance reports require at least one warmup iteration")
    if trials < 5:
        raise ValueError("performance reports require at least five trials")
    if work_items <= 0:
        raise ValueError("work_items must be positive")
    selected = torch.device(device)
    for _ in range(warmup):
        function()
    _synchronize(selected)
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
    samples = []
    for _ in range(trials):
        _synchronize(selected)
        started = time.perf_counter()
        function()
        _synchronize(selected)
        samples.append(time.perf_counter() - started)
    mean = statistics.fmean(samples)
    standard_error = statistics.stdev(samples) / math.sqrt(trials)
    ci95 = 1.96 * standard_error
    peak = int(torch.cuda.max_memory_allocated(selected)) if selected.type == "cuda" else None
    return PerformanceReport(
        name=name,
        configuration=dict(configuration),
        hardware=hardware_fingerprint(selected),
        warmup=warmup,
        trials=trials,
        synchronized=True,
        samples_seconds=tuple(samples),
        mean_seconds=mean,
        ci95_seconds=ci95,
        throughput_per_second=work_items / mean,
        peak_memory_bytes=peak,
        graph_count=graph_count,
        compile_seconds=compile_seconds,
    )


def compare_performance(
    baseline: PerformanceReport,
    candidate: PerformanceReport,
    *,
    permitted_regression: float = 0.01,
) -> PerformanceComparison:
    if baseline.hardware != candidate.hardware:
        return PerformanceComparison(
            False, math.nan, math.inf, permitted_regression, "hardware fingerprints differ"
        )
    if baseline.configuration != candidate.configuration:
        return PerformanceComparison(
            False, math.nan, math.inf, permitted_regression, "configurations differ"
        )
    ratio = candidate.mean_seconds / baseline.mean_seconds
    conservative_candidate = candidate.mean_seconds + candidate.ci95_seconds
    conservative_baseline = max(
        baseline.mean_seconds - baseline.ci95_seconds,
        torch.finfo(torch.float64).tiny,
    )
    regression = conservative_candidate / conservative_baseline - 1.0
    passed = regression <= permitted_regression
    return PerformanceComparison(
        passed,
        ratio,
        regression,
        permitted_regression,
        "pass" if passed else "95% confidence bound exceeds permitted regression",
    )


def profile_operator_trace(
    function: Callable[[], object],
    *,
    device: str | torch.device = "cpu",
    warmup: int = 2,
) -> OperatorTrace:
    """Record the Torch operator multiset for one already-resolved call."""

    selected = torch.device(device)
    for _ in range(warmup):
        function()
    _synchronize(selected)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if selected.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profiler:
        function()
    counts = {
        event.key: int(event.count)
        for event in profiler.key_averages()
        if not event.key.startswith("ProfilerStep")
    }
    copies = sum(
        count
        for name, count in counts.items()
        if any(marker in name for marker in ("copy_", "contiguous", "_to_copy"))
    )
    synchronizations = sum(
        count for name, count in counts.items() if "Synchronize" in name or "synchronize" in name
    )
    return OperatorTrace(counts, copies, synchronizations)


def compare_operator_traces(
    direct: OperatorTrace,
    composed: OperatorTrace,
) -> OperatorTraceComparison:
    added = {
        name: count - int(direct.operator_counts.get(name, 0))
        for name, count in composed.operator_counts.items()
        if count > int(direct.operator_counts.get(name, 0))
    }
    added_copies = composed.tensor_copy_events - direct.tensor_copy_events
    added_syncs = composed.synchronization_events - direct.synchronization_events
    return OperatorTraceComparison(
        not added and added_copies <= 0 and added_syncs <= 0,
        added,
        added_copies,
        added_syncs,
    )


__all__ = [
    "HardwareFingerprint",
    "OperatorTrace",
    "OperatorTraceComparison",
    "PerformanceComparison",
    "PerformanceReport",
    "benchmark_callable",
    "compare_performance",
    "compare_operator_traces",
    "hardware_fingerprint",
    "profile_operator_trace",
]
