"""Regression coverage for the compilation boundary used by CUDA online credit."""
import pytest
import torch

from dabsn import DABSNCore
from dabsn.temporal_credit import (
    CoreSignalGrads, core_online_forward_reference, core_online_replay_analytic,
    initial_core_eligibility,
)


def test_online_replay_visits_more_than_eight_shapes_without_global_limit_changes(monkeypatch):
    from dabsn.kernels.online_compile import call_online_compiled

    # Exercise real Dynamo full-graph tracing on CPU. Only CUDA code generation
    # is replaced; guard creation/cache limits are exactly the failing subsystem.
    original_compile = torch.compile
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def compile_on_cpu(function, **options):
        return original_compile(function, **dict(options, backend=backend))

    monkeypatch.setattr(torch, "compile", compile_on_cpu)
    limit = torch._dynamo.config.recompile_limit
    torch.manual_seed(122)
    fixtures = []
    with torch.no_grad():
        for hidden in range(2, 12):
            core = DABSNCore(3, hidden)
            eligibility = initial_core_eligibility(core, 1, device="cpu")
            _, tape = core_online_forward_reference(
                core, torch.randn(1, 1, 3), core.initial_state(1), eligibility,
            )
            signals = CoreSignalGrads(y=torch.randn(1, 1, hidden))
            fixtures.append((core, tape, signals, eligibility))
            expected = core_online_replay_analytic(*fixtures[-1])
            actual = call_online_compiled(core_online_replay_analytic, *fixtures[-1])
            for left, right in zip(actual[0].tensors(), expected[0].tensors()):
                torch.testing.assert_close(left, right)
            for parameter in core.parameters():
                torch.testing.assert_close(actual[1][parameter], expected[1][parameter])
            torch.testing.assert_close(actual[2], expected[2])
        assert len(graphs) == 10
        # Optimizer updates change values, not the execution contract. Reuse the
        # compiled graph while consuming the new weights (no frozen snapshots).
        core, tape, signals, eligibility = fixtures[0]
        core.A.weight.add_(.1)
        actual = call_online_compiled(core_online_replay_analytic, *fixtures[0])
        expected = core_online_replay_analytic(*fixtures[0])
        torch.testing.assert_close(actual[2], expected[2])
        assert len(graphs) == 10
    assert torch._dynamo.config.recompile_limit == limit


def test_bf16_emulation_is_not_reported_as_native_online_compilation(monkeypatch):
    from dabsn.kernels.online_compile import require_native_online_dtype

    # Hardware-query boundary only: CPU CI cannot expose a T4 capability.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with pytest.raises(RuntimeError, match="native BF16"):
        require_native_online_dtype(torch.device("cuda", 0), torch.bfloat16)
    require_native_online_dtype(torch.device("cuda", 0), torch.float16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    require_native_online_dtype(torch.device("cuda", 0), torch.bfloat16)


def test_online_replay_cache_distinguishes_shared_output_gradients(monkeypatch):
    from dabsn.kernels.online_compile import call_online_compiled

    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **options:
                        original_compile(fn, **dict(options, backend="eager")))
    torch.manual_seed(126)
    core = DABSNCore(2, 3)
    eligibility = initial_core_eligibility(core, 1, device="cpu")
    _, tape = core_online_forward_reference(
        core, torch.randn(1, 1, 2), core.initial_state(1), eligibility,
    )
    names = tuple(CoreSignalGrads.__dataclass_fields__)
    with torch.no_grad():
        for pattern in (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15):
            shared = torch.randn(1, 1, 3)
            signals = CoreSignalGrads(**{
                name: shared if pattern & (1 << index) else torch.randn(1, 1, 3)
                for index, name in enumerate(names)
            })
            actual = call_online_compiled(core_online_replay_analytic, core, tape, signals, eligibility)
            expected = core_online_replay_analytic(core, tape, signals, eligibility)
            torch.testing.assert_close(actual[2], expected[2])
            for parameter in core.parameters():
                torch.testing.assert_close(actual[1][parameter], expected[1][parameter])


def test_online_compiler_decline_never_silently_runs_eager(monkeypatch):
    import importlib
    from torch._dynamo.exc import SkipFrame
    from dabsn.kernels.online_compile import call_online_compiled

    def decline(graph, inputs, **options):
        raise SkipFrame("simulated unsupported device/dtype")

    monkeypatch.setattr(importlib.import_module("torch._inductor.compile_fx"), "compile_fx", decline)

    def numeric_step(value):
        return value.sin() + 1

    with pytest.raises(RuntimeError, match="eager fallback"):
        call_online_compiled(numeric_step, torch.randn(3))
