"""Public component conformance checks used by providers and CI."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch

from .components import (
    AxisEffect,
    BuildContext,
    ComponentOutput,
    ComponentSpec,
    component_registry,
)
from .events import EventCode, emit_event
from .graph import DABSNGraph


@dataclass(frozen=True)
class ConformanceCheck:
    name: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class ConformanceReport:
    provider_key: str
    checks: tuple[ConformanceCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_key": self.provider_key,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def _axis_extent(axis, free_size: int) -> int:
    """Concrete length to expect for one declared axis.

    A ``COLLAPSE`` axis is reduced to a single entry by the component, so its
    length is one no matter what the input carried -- the whole point of
    declaring the effect. Otherwise a statically sized axis keeps its size and
    everything else takes the free size the caller chose.
    """
    if axis.effect is AxisEffect.COLLAPSE:
        return 1
    if isinstance(axis.size, int):
        return axis.size
    return free_size


def _shape_from_contract(contract) -> tuple[int, ...]:
    if len(contract.leaves) != 1:
        raise ValueError("the built-in conformance runner currently requires one tensor leaf")
    return tuple(_axis_extent(axis, 2) for axis in contract.leaves[0].axes)


def _shape_with_dynamic_size(contract, dynamic_size: int) -> tuple[int, ...]:
    if len(contract.leaves) != 1:
        raise ValueError("the built-in conformance runner currently requires one tensor leaf")
    return tuple(
        _axis_extent(axis, dynamic_size if axis.dynamic else 2)
        for axis in contract.leaves[0].axes
    )


def _record(checks, name, function, *, skip: str | None = None):
    if skip is not None:
        checks.append(ConformanceCheck(name, "skip", skip))
        return None
    try:
        result = function()
    except Exception as exc:
        checks.append(ConformanceCheck(name, "fail", f"{type(exc).__name__}: {exc}"))
        return None
    checks.append(ConformanceCheck(name, "pass"))
    return result


def check_component(
    provider_key: str,
    config: Mapping[str, object],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    trust_provider: bool = False,
    distributed_state=None,
) -> ConformanceReport:
    """Run the provider's declared local conformance matrix."""

    selected_device = torch.device(device)
    checks: list[ConformanceCheck] = []
    component_registry.discover()
    if trust_provider:
        component_registry.authorize(provider_key)
    binding = _record(
        checks,
        "config-schema-and-build",
        lambda: component_registry.build(
            ComponentSpec("conformance.0", provider_key, dict(config)),
            BuildContext(device=selected_device, dtype=dtype),
        ),
    )
    if binding is None:
        return ConformanceReport(provider_key, tuple(checks))
    binding.module.to(device=selected_device, dtype=dtype)
    shape = _shape_from_contract(binding.contract.input)
    output_shape = _shape_from_contract(binding.contract.output)

    record = component_registry.resolve(provider_key)

    def migration_check():
        if record.provider is None:
            raise RuntimeError("trusted provider did not load")
        migrated = record.provider.migrate_config(
            binding.config_schema_version,
            dict(config),
        )
        if dict(migrated) != dict(config):
            raise RuntimeError("current-schema migration changed configuration")

    _record(checks, "config-schema-migration", migration_check)

    def fresh_input(*, noncontiguous=False):
        value = torch.randn(*shape, device=selected_device, dtype=dtype)
        if noncontiguous and value.dim() >= 2:
            value = value.transpose(0, 1).contiguous().transpose(0, 1)
            assert not value.is_contiguous()
        return value.requires_grad_(True)

    def eager_backward(noncontiguous=False):
        binding.module.zero_grad(set_to_none=True)
        value = fresh_input(noncontiguous=noncontiguous)
        output = binding.module(value)
        if tuple(output.shape) != output_shape:
            raise ValueError(f"output shape {tuple(output.shape)} != declared {output_shape}")
        output.float().square().mean().backward()
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise RuntimeError("input gradients are absent or non-finite")
        return output

    _record(checks, "eager-forward-backward", eager_backward)
    _record(checks, "non-contiguous-input", lambda: eager_backward(True))

    def dynamic_axes():
        for dynamic_size in (1, 3):
            input_dynamic = _shape_with_dynamic_size(binding.contract.input, dynamic_size)
            output_dynamic = _shape_with_dynamic_size(binding.contract.output, dynamic_size)
            value = torch.randn(
                *input_dynamic,
                device=selected_device,
                dtype=dtype,
            )
            output = binding.module(value)
            if tuple(output.shape) != output_dynamic:
                raise RuntimeError(
                    f"dynamic output {tuple(output.shape)} != declared {output_dynamic}"
                )

    _record(
        checks,
        "dynamic-axis-contract",
        dynamic_axes,
        skip=None if binding.capabilities.dynamic_shapes else "provider declares unsupported",
    )

    _record(
        checks,
        "determinism-declaration",
        lambda: (
            None
            if binding.capabilities.deterministic is not None
            else (_ for _ in ()).throw(RuntimeError("provider omitted determinism declaration"))
        ),
    )

    def autocast_check(precision: torch.dtype):
        binding.module.zero_grad(set_to_none=True)
        value = fresh_input()
        with torch.autocast(device_type=selected_device.type, dtype=precision):
            output = binding.module(value)
            loss = output.float().square().mean()
        loss.backward()
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise RuntimeError("autocast gradients are absent or non-finite")
        emit_event(
            EventCode.DTYPE_CHANGE,
            component_id=binding.component_id,
            requested_dtype=str(precision),
            input_dtype=str(dtype),
            device=str(selected_device),
        )

    bf16_available = selected_device.type == "cpu" or (
        selected_device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    _record(
        checks,
        "amp-bf16",
        lambda: autocast_check(torch.bfloat16),
        skip=(
            None
            if binding.capabilities.amp_bf16 and bf16_available
            else "provider/device declares unsupported"
        ),
    )
    _record(
        checks,
        "amp-fp16",
        lambda: autocast_check(torch.float16),
        skip=(
            None
            if binding.capabilities.amp_fp16 and selected_device.type == "cuda"
            else "provider/device declares unsupported"
        ),
    )

    graph = DABSNGraph([binding])

    def terms():
        result = graph.forward_with_terms(fresh_input())
        if not isinstance(result, ComponentOutput):
            raise TypeError("graph training path did not return ComponentOutput")
        if len(result.loss_terms) != len(binding.loss_terms):
            raise RuntimeError("declared and returned loss-term arity disagree")
        if len(result.reports) != len(binding.reports):
            raise RuntimeError("declared and returned report arity disagree")

    _record(checks, "fixed-training-output", terms)

    def state_carry():
        if not binding.states:
            return
        first = graph.forward_with_state(fresh_input(), None)
        if len(first.next_state) != len(binding.states):
            raise RuntimeError("initial state result has incorrect arity")
        second = graph.forward_with_state(fresh_input(), first.next_state)
        if len(second.next_state) != len(binding.states):
            raise RuntimeError("carried state result has incorrect arity")

    _record(
        checks,
        "streaming-state-carry",
        state_carry,
        skip=None if binding.capabilities.streaming_state else "provider declares stateless",
    )

    def compile_check():
        compiled = torch.compile(binding.module, backend="aot_eager", fullgraph=True)
        compiled(fresh_input()).float().sum().backward()

    _record(
        checks,
        "compile-fullgraph",
        compile_check,
        skip=None if binding.capabilities.compile_fullgraph else "provider declares unsupported",
    )
    compile_result = checks[-1]
    emit_event(
        EventCode.COMPILE_RESULT,
        component_id=binding.component_id,
        status=compile_result.status,
        detail=compile_result.detail,
        fullgraph=True,
    )

    def fake_check():
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode(allow_non_fake_inputs=True):
            binding.module(torch.empty(shape, device=selected_device, dtype=dtype))

    _record(checks, "fake-tensor", fake_check)

    def export_check():
        torch.export.export(binding.module, (fresh_input(),), strict=True)

    _record(
        checks,
        "torch-export",
        export_check,
        skip=None if binding.capabilities.export else "provider declares unsupported",
    )

    def reconstruction():
        spec = binding.to_spec()
        rebuilt = component_registry.build(spec)
        rebuilt.module.to(device=selected_device, dtype=dtype)
        rebuilt.module.load_state_dict(binding.module.state_dict())
        sample = fresh_input().detach()
        original_training = binding.module.training
        rebuilt_training = rebuilt.module.training
        try:
            binding.module.eval()
            rebuilt.module.eval()
            expected = binding.module(sample)
            actual = rebuilt.module(sample)
        finally:
            binding.module.train(original_training)
            rebuilt.module.train(rebuilt_training)
        if expected.shape != actual.shape:
            raise RuntimeError("reconstructed component changed output shape")
        torch.testing.assert_close(actual, expected)

    _record(checks, "save-load-reconstruction", reconstruction)

    def lifecycle():
        graph.post_optimizer_step(step_applied=False)
        graph.post_optimizer_step(step_applied=True)

    _record(checks, "lifecycle-timing", lifecycle)

    def direct_vs_composed():
        sample = fresh_input().detach()
        for _ in range(2):
            binding.module(sample)
            graph(sample)
        started = time.perf_counter()
        for _ in range(10):
            binding.module(sample)
        direct = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(10):
            graph(sample)
        composed = time.perf_counter() - started
        # This local smoke check records gross accidental overhead. Formal <=1%
        # release gates use synchronized repeated trials and confidence bounds.
        if composed > direct * 5 + 1.0e-3:
            raise RuntimeError(f"gross composition overhead direct={direct} composed={composed}")

    _record(checks, "direct-versus-composed-smoke", direct_vs_composed)

    def cuda_graph_replay():
        stream = torch.cuda.Stream(device=selected_device)
        stream.wait_stream(torch.cuda.current_stream(selected_device))
        static_input = fresh_input().detach()
        with torch.cuda.stream(stream):
            for _ in range(3):
                binding.module(static_input)
        torch.cuda.current_stream(selected_device).wait_stream(stream)
        graph_capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_capture, stream=stream):
            static_output = binding.module(static_input)
        replacement = torch.randn_like(static_input)
        expected = binding.module(replacement)
        static_input.copy_(replacement)
        graph_capture.replay()
        torch.cuda.synchronize(selected_device)
        torch.testing.assert_close(static_output, expected)

    cuda_skip = None
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        cuda_skip = "CUDA device not selected or unavailable"
    elif not binding.capabilities.cuda_graph:
        cuda_skip = "provider declares unsupported"
    _record(checks, "cuda-graph-replay", cuda_graph_replay, skip=cuda_skip)
    capture_result = checks[-1]
    emit_event(
        EventCode.CAPTURE_RESULT,
        component_id=binding.component_id,
        status=capture_result.status,
        detail=capture_result.detail,
    )

    def distributed_wrapping():
        if not binding.capabilities.distributed:
            raise RuntimeError("provider declares distributed execution unsupported")
        from .distributed import wrap_distributed

        precision = {
            torch.float32: "fp32",
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
        }[dtype]
        binding.module.zero_grad(set_to_none=True)
        wrapped = wrap_distributed(graph, distributed_state, precision=precision)
        output = wrapped(fresh_input())
        output.float().square().mean().backward()

    _record(
        checks,
        "fsdp-wrapping",
        distributed_wrapping,
        skip=(
            None
            if distributed_state is not None and distributed_state.enabled
            else "requires an initialized multi-rank process group"
        ),
    )
    return ConformanceReport(provider_key, tuple(checks))


__all__ = ["ConformanceCheck", "ConformanceReport", "check_component"]
