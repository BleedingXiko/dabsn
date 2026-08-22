"""Gates that must be off while this code is traced rather than executed.

Three optimizations exist purely to speed up the *eager* path: the per-step
`torch.compile`, and the two captured reverse/forward scan graphs. Every one of
them is illegal while something is tracing the surrounding code -- you cannot
nest a dynamo-optimized function inside an FX trace, and you cannot record a
CUDA graph against tensors that are being walked symbolically rather than
executed.

These gates cost a machine with two T4s three full gate runs to get right, so
they are pinned here rather than left to be rediscovered:

1. The first attempt tested only `is_cuda`, so whole-graph compiling the block
   failed with "Detected that you are using FX to symbolically trace a
   dynamo-optimized function".
2. The second added `torch.compiler.is_compiling()`, which is set while Dynamo
   traces the forward but **not** while AOTAutograd traces the backward -- so
   `aot_eager` failed on the backward step compile in exactly the same way.

The signal that covers both is a fake tensor, and the trap that makes this
untestable by inspection is that a fake CUDA tensor reports `is_cuda` as True on
a machine with no GPU at all. That is precisely why these run here: they
reproduce the two-T4 failure on any laptop.
"""

from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from dabsn.kernels.batched_runtime import _being_traced, _step_compile_enabled


def _fake_cuda_tensor(mode: FakeTensorMode) -> torch.Tensor:
    with mode:
        return torch.empty(4, 8, device="cuda")


def test_a_fake_cuda_tensor_still_reports_is_cuda():
    """The premise. If this ever stops holding, the gates below are moot."""
    with FakeTensorMode() as mode:
        tensor = _fake_cuda_tensor(mode)
        assert tensor.is_cuda, "fake CUDA tensors are expected to report is_cuda"
        assert not torch.compiler.is_compiling(), (
            "a fake tensor outside Dynamo must not set is_compiling, which is the "
            "whole reason is_compiling alone was insufficient"
        )


def test_step_compile_is_off_under_fake_tensors_despite_is_cuda():
    """The regression: `is_cuda and env` was true here, and it must not be.

    This is the AOTAutograd-backward case that `aot_eager` hit on hardware.
    """
    with FakeTensorMode() as mode:
        tensor = _fake_cuda_tensor(mode)
        assert _being_traced(tensor)
        assert not _step_compile_enabled(tensor)


def test_step_compile_is_still_available_on_ordinary_tensors():
    """The gate must not have been closed by simply disabling the optimization.

    A CPU tensor is refused for its own reason (the step compile is CUDA-only),
    so what is checked here is that an ordinary tensor is not treated as traced.
    """
    tensor = torch.zeros(4, 8)
    assert not _being_traced(tensor)


def test_tracing_guard_precedes_every_symbolic_comparison():
    """Presence is not enough -- the guard has to come FIRST.

    `or` short-circuits left to right, so a tracing guard placed after a clause
    like `T < _SCAN_GRAPH_MIN_STEPS` never runs: the shape comparison is
    evaluated first, and under tracing that shape is symbolic, which raises out
    of the symbolic-shape machinery before the guard is reached.

    This is not hypothetical ordering pedantry. With the guard in second
    position the gate passed on CPU -- where `not is_cuda` short-circuits ahead
    of the comparison -- and failed on CUDA with `'int' object has no attribute
    'is_Add'`, which is why it survived several rounds of local testing.
    """
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src/dabsn/kernels/batched_runtime.py"
    text = source.read_text(encoding="utf-8")

    guarded = 0
    for block in text.split("    if (")[1:]:
        condition = block.split("    ):")[0]
        if "_SCAN_GRAPH_MIN_STEPS" not in condition:
            continue
        guarded += 1
        clauses = [
            line.strip()
            for line in condition.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert clauses, "scan-graph condition parsed as empty"
        assert clauses[0].startswith("_being_traced("), (
            "the tracing guard must be the first clause of the scan-graph gate; "
            f"found {clauses[0]!r} ahead of it, which is evaluated under tracing"
        )
    assert guarded == 2, f"expected two scan-graph gates, inspected {guarded}"


def test_scan_graph_gates_consult_the_same_signal():
    """Both captured scan paths must gate on tracing, not only on capture.

    Asserted against the source because the alternative is a CUDA-only,
    two-rank, torch-version-specific integration test -- which is exactly the
    setup that let this ship twice.
    """
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src/dabsn/kernels/batched_runtime.py"
    text = source.read_text(encoding="utf-8")
    assert text.count("_being_traced(") >= 3, (
        "the step-compile gate and both scan-graph gates must test for tracing"
    )
    assert "_being_traced(U)" in text
    assert "_being_traced(Wx)" in text
