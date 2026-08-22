from dataclasses import replace

import pytest
import torch

from dabsn import (
    AxisContract,
    ComponentContract,
    DABSNGraph,
    ValueContract,
    benchmark_callable,
    bind_module,
    compare_performance,
)
from dabsn.benchmarking import compare_operator_traces, profile_operator_trace


def test_report_contains_every_required_methodology_field():
    value = torch.randn(32, 32)
    report = benchmark_callable(
        "fixture",
        lambda: value @ value,
        work_items=32,
        configuration={"width": 32, "dtype": "fp32"},
        warmup=1,
        trials=5,
    )
    assert report.warmup == 1
    assert report.trials == 5
    assert report.synchronized
    assert len(report.samples_seconds) == 5
    assert report.mean_seconds > 0
    assert report.ci95_seconds >= 0
    assert report.throughput_per_second > 0
    assert report.hardware.torch
    assert report.hardware.python


def test_reports_without_warmup_or_repeated_trials_are_rejected():
    with pytest.raises(ValueError, match="warmup"):
        benchmark_callable("bad", lambda: None, work_items=1, configuration={}, warmup=0, trials=5)
    with pytest.raises(ValueError, match="five trials"):
        benchmark_callable("bad", lambda: None, work_items=1, configuration={}, warmup=1, trials=1)


def test_comparison_uses_conservative_95_percent_bound():
    value = torch.randn(8, 8)
    baseline = benchmark_callable(
        "base",
        lambda: value + 1,
        work_items=8,
        configuration={"shape": 8},
        warmup=1,
        trials=5,
    )
    exact = replace(
        baseline,
        name="candidate",
        samples_seconds=baseline.samples_seconds,
        ci95_seconds=0.0,
    )
    base_exact = replace(baseline, ci95_seconds=0.0)
    assert compare_performance(base_exact, exact).passed
    slow = replace(exact, mean_seconds=base_exact.mean_seconds * 1.02)
    comparison = compare_performance(base_exact, slow, permitted_regression=0.01)
    assert not comparison.passed
    assert comparison.conservative_regression == pytest.approx(0.02)


def test_mismatched_hardware_or_configuration_cannot_gate_release():
    value = torch.randn(8)
    baseline = benchmark_callable(
        "base", lambda: value + 1, work_items=8, configuration={"shape": 8}, warmup=1, trials=5
    )
    mismatch = replace(baseline, configuration={"shape": 16})
    comparison = compare_performance(baseline, mismatch)
    assert not comparison.passed
    assert "configurations differ" in comparison.reason


def test_component_boundary_adds_no_torch_operator_copy_or_sync():
    module = torch.nn.Linear(8, 8, bias=False)
    value = ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("experience", "T", dynamic=True),
        AxisContract("world", 8),
    )
    graph = DABSNGraph([bind_module("linear", module, ComponentContract(value, value))])
    sample = torch.randn(2, 4, 8)
    direct = profile_operator_trace(lambda: module(sample), warmup=1)
    composed = profile_operator_trace(lambda: graph(sample), warmup=1)
    comparison = compare_operator_traces(direct, composed)
    assert comparison.passed, comparison
