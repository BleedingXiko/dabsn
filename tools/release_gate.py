#!/usr/bin/env python3
"""Small, non-interactive DABSN native-runtime release gate.

Run this from a wheel-installed checkout through ``tools/cpu_check.sh`` or
``tools/gpu_check.sh``. Every check reports independently and the process exits
nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable

import torch

from dabsn import DABSNCore, DABSNLayerSpec, DABSNModel, load_dabsn, save_dabsn
from dabsn.kernels import enable, local_field_gather, permanent_delta_scan, status
from dabsn.kernels.long import _linrec_reference, linear_recurrence
from dabsn.kernels.permanent import _permanent_reference


class Gate:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.rows: list[dict[str, object]] = []

    def check(self, name: str, function: Callable[[], object]) -> None:
        started = time.perf_counter()
        try:
            detail = function()
        except Exception as exc:  # continue so one run shows every failure
            row = {
                "name": name,
                "passed": False,
                "seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            self.rows.append(row)
            print(f"[FAIL] {name}: {row['error']}", flush=True)
        else:
            row = {
                "name": name,
                "passed": True,
                "seconds": round(time.perf_counter() - started, 3),
                "detail": detail,
            }
            self.rows.append(row)
            suffix = "" if detail is None else f": {detail}"
            print(f"[PASS] {name}{suffix}", flush=True)

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(bool(row["passed"]) for row in self.rows)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    backend: str,
    label: str,
) -> None:
    if backend == "cuda":
        atol, rtol = 2.0e-3, 2.0e-3
    else:
        atol, rtol = 3.0e-4, 3.0e-4
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=label)


def _environment(backend: str) -> dict[str, object]:
    data: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "backend": backend,
    }
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch cannot see CUDA. On Windows, run this inside WSL2 Ubuntu "
                "with the NVIDIA driver installed on the host."
            )
        capability = torch.cuda.get_device_capability(0)
        if capability < (7, 0):
            raise RuntimeError(f"CUDA capability {capability} is below the release-gate minimum")
        import triton

        data.update(
            device=torch.cuda.get_device_name(0),
            capability=f"{capability[0]}.{capability[1]}",
            cuda=torch.version.cuda,
            triton=triton.__version__,
        )
    return data


def _model_reference(
    device: torch.device,
) -> tuple[DABSNModel, DABSNModel, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(20260715)
    layers = [DABSNLayerSpec(8, 8, geometry) for geometry in ("seq", "field", "hybrid")]
    reference = DABSNModel(5, 3, layers, output_adapter="token").to(device)
    native = copy.deepcopy(reference)
    base = torch.randn(2, 8, 5, device=device)
    reference_input = base.detach().clone().requires_grad_(True)
    reference_output = reference.forward_sequence(reference_input)
    loss = reference_output.square().mean() + 0.01 * reference_output.mean()
    loss.backward()
    reference_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }
    return (
        native,
        reference,
        base,
        reference_output.detach(),
        reference_input.grad.detach().clone(),
        reference_grads,
    )


def _run_core_state(
    core: DABSNCore,
    base: torch.Tensor,
    initial: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    chunked: bool,
) -> dict[str, object]:
    core.zero_grad(set_to_none=True)
    inputs = base.detach().clone().requires_grad_(True)
    state = tuple(value.detach().clone().requires_grad_(True) for value in initial)
    if chunked:
        left, carry = core.forward_from_state(
            inputs[:, :3],
            initial_state=state,
            return_writes=True,
            return_cocktail=True,
            return_final_state=True,
        )
        right, final = core.forward_from_state(
            inputs[:, 3:],
            initial_state=carry,
            return_writes=True,
            return_cocktail=True,
            return_final_state=True,
        )
        outputs = tuple(torch.cat([a, b], dim=1) for a, b in zip(left, right))
    else:
        outputs, final = core.forward_from_state(
            inputs,
            initial_state=state,
            return_writes=True,
            return_cocktail=True,
            return_final_state=True,
        )
    loss = sum((index + 1) * value.square().mean() for index, value in enumerate(outputs))
    loss = loss + sum((index + 8) * value.square().mean() for index, value in enumerate(final))
    loss.backward()
    return {
        "outputs": tuple(value.detach() for value in outputs),
        "final": tuple(value.detach() for value in final),
        "input_grad": inputs.grad.detach(),
        "state_grads": tuple(value.grad.detach() for value in state),
        "parameter_grads": {
            name: parameter.grad.detach().clone()
            for name, parameter in core.named_parameters()
            if parameter.grad is not None
        },
        "backend": getattr(core, "_last_core_backend", None),
    }


def _core_state_reference(device: torch.device) -> dict[str, object]:
    torch.manual_seed(20260716)
    reference = DABSNCore(5, 8).to(device)
    full_native = copy.deepcopy(reference)
    chunk_native = copy.deepcopy(reference)
    base = torch.randn(2, 7, 5, device=device)
    initial = (
        torch.randn(2, 8, device=device) * 0.1,
        torch.sigmoid(torch.randn(2, 8, device=device)),
        torch.randn(2, 8, device=device) * 0.05,
    )
    return {
        "full_native": full_native,
        "chunk_native": chunk_native,
        "base": base,
        "initial": initial,
        "reference": _run_core_state(reference, base, initial, chunked=False),
    }


def _core_state_native_parity(backend: str, captured: dict[str, object]) -> dict[str, float]:
    full = _run_core_state(
        captured["full_native"], captured["base"], captured["initial"], chunked=False
    )
    chunk = _run_core_state(
        captured["chunk_native"], captured["base"], captured["initial"], chunked=True
    )
    reference = captured["reference"]
    expected_backend = "cuda_triton" if backend == "cuda" else "cpu_native_cpp"
    if full["backend"] != expected_backend or chunk["backend"] != expected_backend:
        raise AssertionError(
            f"core state route was full={full['backend']} chunk={chunk['backend']}, expected {expected_backend}"
        )
    for label, actual, expected in (
        ("full outputs", full["outputs"], reference["outputs"]),
        ("full final state", full["final"], reference["final"]),
        ("full state gradients", full["state_grads"], reference["state_grads"]),
        ("chunk outputs", chunk["outputs"], full["outputs"]),
        ("chunk final state", chunk["final"], full["final"]),
        ("chunk state gradients", chunk["state_grads"], full["state_grads"]),
    ):
        for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
            _assert_close(actual_tensor, expected_tensor, backend=backend, label=f"{label} {index}")
    for label, actual, expected in (
        ("full input gradient", full["input_grad"], reference["input_grad"]),
        ("chunk input gradient", chunk["input_grad"], full["input_grad"]),
    ):
        _assert_close(actual, expected, backend=backend, label=label)
    for label, actual, expected in (
        ("full parameter gradients", full["parameter_grads"], reference["parameter_grads"]),
        ("chunk parameter gradients", chunk["parameter_grads"], full["parameter_grads"]),
    ):
        if set(actual) != set(expected):
            raise AssertionError(f"{label} coverage differs")
        for name in actual:
            _assert_close(actual[name], expected[name], backend=backend, label=f"{label} {name}")
    return {
        "full_reference_max_abs": max(
            _max_abs(actual, expected)
            for actual, expected in zip(full["outputs"], reference["outputs"])
        ),
        "chunk_full_max_abs": max(
            _max_abs(actual, expected)
            for actual, expected in zip(chunk["outputs"], full["outputs"])
        ),
    }


def _backend_status(backend: str) -> dict[str, object]:
    report = enable(backend, required=True)
    if report["active_backend"] != backend:
        raise AssertionError(report)
    if backend == "cpu":
        cpu = report["cpu"]
        required = ("extension_available", "core_scan_enabled", "three_way_read_enabled")
        if not all(bool(cpu.get(key)) for key in required):
            raise AssertionError(cpu)
        if not bool(report["permanent"].get("cpu_native_enabled")):
            raise AssertionError(report["permanent"])
        if not bool(report["local_field"].get("cpu_native_enabled")):
            raise AssertionError(report["local_field"])
    else:
        cuda = report["cuda"]
        required = ("cuda_available", "triton_available", "core_scan_enabled", "admitted_three_way_read_enabled")
        if not all(bool(cuda.get(key)) for key in required):
            raise AssertionError(cuda)
        if not bool(report["long_memory"].get("enabled")):
            raise AssertionError(report["long_memory"])
        if not bool(report["permanent"].get("triton_available")):
            raise AssertionError(report["permanent"])
        if not bool(report["local_field"].get("triton_available")):
            raise AssertionError(report["local_field"])
    return report


def _model_native_parity(
    backend: str,
    native: DABSNModel,
    base: torch.Tensor,
    reference_output: torch.Tensor,
    reference_input_grad: torch.Tensor,
    reference_grads: dict[str, torch.Tensor],
) -> dict[str, object]:
    native.zero_grad(set_to_none=True)
    native_input = base.detach().clone().requires_grad_(True)
    native_output = native.forward_sequence(native_input)
    loss = native_output.square().mean() + 0.01 * native_output.mean()
    loss.backward()
    _assert_close(native_output, reference_output, backend=backend, label="model output")
    _assert_close(native_input.grad, reference_input_grad, backend=backend, label="input gradient")

    native_grads = {
        name: parameter.grad
        for name, parameter in native.named_parameters()
        if parameter.grad is not None
    }
    if set(native_grads) != set(reference_grads):
        raise AssertionError(
            f"parameter-gradient coverage differs: native-only={sorted(set(native_grads) - set(reference_grads))}, "
            f"reference-only={sorted(set(reference_grads) - set(native_grads))}"
        )
    for name, expected in reference_grads.items():
        _assert_close(native_grads[name], expected, backend=backend, label=f"parameter gradient {name}")

    traces = native.read_traces()
    for index, block in enumerate(native.backbone.blocks):
        expected = "cuda_triton" if backend == "cuda" else "cpu_native_cpp"
        if getattr(block.core, "_last_core_backend", None) != expected:
            raise AssertionError(f"block {index} core did not route through {expected}")
        if getattr(block.read, "_last_long_backend", None) != expected:
            raise AssertionError(f"block {index} long read did not route through {expected}")
        read_backend = traces[index]["read_contract"]["kernel_backend"]
        if backend == "cuda" and "triton" not in str(read_backend) and "flash" not in str(read_backend):
            raise AssertionError(f"block {index} admitted read backend: {read_backend}")
        if backend == "cpu" and read_backend != "cpu_native_cpp":
            raise AssertionError(f"block {index} admitted read backend: {read_backend}")

    return {
        "geometries": ["seq", "field", "hybrid"],
        "output_max_abs": _max_abs(native_output, reference_output),
        "input_grad_max_abs": _max_abs(native_input.grad, reference_input_grad),
        "parameter_gradients": len(native_grads),
    }


def _checkpoint_roundtrip(model: DABSNModel, base: torch.Tensor) -> str:
    model.eval()
    with tempfile.TemporaryDirectory(prefix="dabsn-gate-") as directory:
        path = Path(directory) / "model.safetensors"
        with torch.no_grad():
            expected = model.forward_sequence(base)
        save_dabsn(model, path)
        restored = load_dabsn(path, map_location=base.device).eval()
        with torch.no_grad():
            actual = restored.forward_sequence(base)
        if not torch.equal(actual, expected):
            raise AssertionError(f"checkpoint output max abs diff: {_max_abs(actual, expected)}")
    return "configuration + weights reload bit-exact"


def _primitive_parity(backend: str, device: torch.device) -> dict[str, float]:
    torch.manual_seed(44)
    a_cpu = (torch.rand(2, 7, 6) * 0.7).requires_grad_(True)
    g_cpu = torch.randn(2, 7, 6, requires_grad=True)
    y_cpu = torch.randn(2, 6, requires_grad=True)
    upstream = torch.randn(2, 7, 6)
    recurrence_cpu = _linrec_reference(a_cpu, g_cpu, y_cpu)
    (recurrence_cpu * upstream).sum().backward()

    a_dev = a_cpu.detach().to(device).requires_grad_(True)
    g_dev = g_cpu.detach().to(device).requires_grad_(True)
    y_dev = y_cpu.detach().to(device).requires_grad_(True)
    recurrence_dev = linear_recurrence(a_dev, g_dev, y_dev)
    (recurrence_dev * upstream.to(device)).sum().backward()
    _assert_close(recurrence_dev.cpu(), recurrence_cpu.detach(), backend=backend, label="long recurrence")
    for label, actual, expected in (
        ("long a gradient", a_dev.grad.cpu(), a_cpu.grad),
        ("long g gradient", g_dev.grad.cpu(), g_cpu.grad),
        ("long initial gradient", y_dev.grad.cpu(), y_cpu.grad),
    ):
        _assert_close(actual, expected, backend=backend, label=label)

    key_cpu = torch.randn(2, 7, 6, requires_grad=True)
    value_cpu = torch.randn(2, 7, 6, requires_grad=True)
    beta_cpu = torch.sigmoid(torch.randn(2, 7, requires_grad=True)).detach().requires_grad_(True)
    permanent_cpu = _permanent_reference(key_cpu, value_cpu, beta_cpu)
    (permanent_cpu * upstream).sum().backward()
    key_dev = key_cpu.detach().to(device).requires_grad_(True)
    value_dev = value_cpu.detach().to(device).requires_grad_(True)
    beta_dev = beta_cpu.detach().to(device).requires_grad_(True)
    permanent_dev = permanent_delta_scan(key_dev, value_dev, beta_dev)
    (permanent_dev * upstream.to(device)).sum().backward()
    _assert_close(permanent_dev.cpu(), permanent_cpu.detach(), backend=backend, label="permanent scan")
    for label, actual, expected in (
        ("permanent key gradient", key_dev.grad.cpu(), key_cpu.grad),
        ("permanent value gradient", value_dev.grad.cpu(), value_cpu.grad),
        ("permanent beta gradient", beta_dev.grad.cpu(), beta_cpu.grad),
    ):
        _assert_close(actual, expected, backend=backend, label=label)

    field_cpu = torch.randn(2, 9, 6, requires_grad=True)
    patch_cpu = torch.tensor(
        [[index, (index + 1) % 9, (index + 3) % 9] for index in range(9)],
        dtype=torch.long,
    )
    field_reference = field_cpu[:, patch_cpu]
    field_upstream = torch.randn_like(field_reference)
    (field_reference * field_upstream).sum().backward()
    field_dev = field_cpu.detach().to(device).requires_grad_(True)
    gathered, label = local_field_gather(field_dev, patch_cpu.to(device))
    (gathered * field_upstream.to(device)).sum().backward()
    if backend == "cuda" and label != "triton_patch_gather_then_shared_core":
        raise AssertionError(f"local-field gather fell back: {label}")
    if backend == "cpu" and label != "cpu_native_cpp_patch_gather_then_shared_core":
        raise AssertionError(f"local-field gather fell back: {label}")
    _assert_close(gathered.cpu(), field_reference.detach(), backend=backend, label="local gather")
    _assert_close(field_dev.grad.cpu(), field_cpu.grad, backend=backend, label="local gather gradient")

    return {
        "long_max_abs": _max_abs(recurrence_dev.cpu(), recurrence_cpu.detach()),
        "permanent_max_abs": _max_abs(permanent_dev.cpu(), permanent_cpu.detach()),
        "local_field_max_abs": _max_abs(gathered.cpu(), field_reference.detach()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--report", type=Path, default=Path("dabsn-release-gate.json"))
    args = parser.parse_args()

    gate = Gate(args.backend)
    environment: dict[str, object] = {}
    captured: dict[str, object] = {}
    device = torch.device("cuda" if args.backend == "cuda" else "cpu")

    def environment_check() -> dict[str, object]:
        environment.update(_environment(args.backend))
        return environment

    def reference_check() -> str:
        native, _reference, base, output, input_grad, parameter_grads = _model_reference(device)
        captured.update(
            native=native,
            base=base,
            output=output,
            input_grad=input_grad,
            parameter_grads=parameter_grads,
            core_state=_core_state_reference(device),
        )
        return "eager forward/backward reference captured before native activation"

    gate.check("environment", environment_check)
    gate.check("eager-reference", reference_check)
    gate.check("required-backend-activation", lambda: _backend_status(args.backend))
    if captured:
        gate.check(
            "native-model-forward-backward-parity",
            lambda: _model_native_parity(
                args.backend,
                captured["native"],
                captured["base"],
                captured["output"],
                captured["input_grad"],
                captured["parameter_grads"],
            ),
        )
        gate.check(
            "native-carried-core-state-forward-backward-parity",
            lambda: _core_state_native_parity(args.backend, captured["core_state"]),
        )
        gate.check(
            "checkpoint-roundtrip",
            lambda: _checkpoint_roundtrip(captured["native"], captured["base"]),
        )
    else:
        gate.check("native-model-forward-backward-parity", lambda: (_ for _ in ()).throw(RuntimeError("reference setup failed")))
        gate.check("native-carried-core-state-forward-backward-parity", lambda: (_ for _ in ()).throw(RuntimeError("reference setup failed")))
        gate.check("checkpoint-roundtrip", lambda: (_ for _ in ()).throw(RuntimeError("reference setup failed")))
    gate.check("standalone-primitive-forward-backward-parity", lambda: _primitive_parity(args.backend, device))

    final_status = status()
    report = {
        "passed": gate.passed,
        "backend": args.backend,
        "environment": environment,
        "checks": gate.rows,
        "kernel_status": final_status,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print("", flush=True)
    print(f"DABSN RELEASE GATE: {'PASS' if gate.passed else 'FAIL'}", flush=True)
    print(f"Report: {args.report.resolve()}", flush=True)
    if not gate.passed:
        print("Failed checks:", flush=True)
        for row in gate.rows:
            if not row["passed"]:
                print(f"  - {row['name']}: {row['error']}", flush=True)
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
