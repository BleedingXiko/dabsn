import subprocess
import sys


def test_required_native_cpu_installed_surface():
    program = r"""
import torch
from dabsn import DABSNLayerSpec, DABSNModel
from dabsn.read import DABSNRead
from dabsn.kernels import enable, local_field_gather, permanent_delta_scan, status
original_read = DABSNRead._three_way_read
enable("cpu", required=True)
assert torch._C._dispatch_has_kernel_for_dispatch_key("dabsn::core_scan_batched", "CPU")
assert torch._C._dispatch_has_kernel_for_dispatch_key(
    "dabsn::admitted_three_way_read_native", "CPU"
)
assert torch._C._dispatch_has_kernel_for_dispatch_key(
    "dabsn::_admitted_three_way_read_backward", "CPU"
)
assert DABSNRead._three_way_read is original_read
model = DABSNModel(4, 2, [DABSNLayerSpec(6, read_geometry="seq")], output_adapter="token")
x = torch.randn(1, 4, 4, requires_grad=True)
y = model.forward_sequence(x)
y.square().mean().backward()
assert torch.isfinite(y).all() and x.grad is not None
compiled_model = DABSNModel(
    4, 2, [DABSNLayerSpec(6, read_geometry="seq")], output_adapter="token"
)
compiled_x = torch.randn(1, 4, 4, requires_grad=True)
compiled = torch.compile(compiled_model.forward_sequence, backend="aot_eager", fullgraph=True)
compiled_y = compiled(compiled_x)
compiled_y.square().mean().backward()
assert torch.isfinite(compiled_y).all() and compiled_x.grad is not None
single = model.forward_sequence(torch.randn(1, 1, 4))
assert torch.isfinite(single).all()

core = model.backbone.blocks[0].core
state = tuple(value.requires_grad_(True) for value in core.initial_state(1))
core_input = torch.randn(1, 5, 4, requires_grad=True)
left, carry = core.forward_from_state(
    core_input[:, :2], initial_state=state, return_writes=True,
    return_cocktail=True, return_final_state=True,
)
right, final = core.forward_from_state(
    core_input[:, 2:], initial_state=carry, return_writes=True,
    return_cocktail=True, return_final_state=True,
)
loss = sum(value.square().mean() for value in (*left, *right, *final))
loss.backward()
assert core._last_core_backend == "cpu_native_cpp"
assert core_input.grad is not None and all(value.grad is not None for value in state)

key = torch.randn(2, 5, 6, requires_grad=True)
value = torch.randn(2, 5, 6, requires_grad=True)
beta = torch.sigmoid(torch.randn(2, 5)).requires_grad_(True)
permanent_delta_scan(key, value, beta).square().mean().backward()
assert key.grad is not None and value.grad is not None and beta.grad is not None

field = torch.randn(2, 7, 4, requires_grad=True)
patch = torch.tensor([[i, (i + 1) % 7, (i + 3) % 7] for i in range(7)])
gathered, route = local_field_gather(field, patch)
gathered.square().mean().backward()
assert route == "cpu_native_cpp_patch_gather_then_shared_core"
assert field.grad is not None
report = status()
assert report["permanent"]["cpu_native_hits"] > 0
assert report["local_field"]["cpu_hits"] > 0
"""
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_optional_cpu_runtime_falls_back_when_extension_is_unavailable():
    program = r"""
import torch
from dabsn import DABSNLayerSpec, DABSNModel
import dabsn.kernels.cpu as cpu

cpu._load_ext = lambda required=False: None
result = cpu.enable_native_cpu_kernels(required=False)
assert result == {"core_scan": False, "three_way_read": False}
model = DABSNModel(4, 2, [DABSNLayerSpec(6, read_geometry="seq")], output_adapter="token")
x = torch.randn(1, 4, 4, requires_grad=True)
y = model.forward_sequence(x)
y.square().mean().backward()
assert torch.isfinite(y).all() and x.grad is not None
"""
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_registered_cpu_core_low_precision_fallback_is_explicit_and_strict():
    program = r"""
import torch
from dabsn import (
    DABSNLayerSpec,
    DABSNModel,
    EventCode,
    StrictFallbackError,
    add_event_listener,
    remove_event_listener,
    strict_events,
)
from dabsn.kernels import enable
from dabsn.runtime.dispatch import reset_routing_log

enable("cpu", required=True)
model = DABSNModel(4, 2, [DABSNLayerSpec(6, read_geometry="seq")], output_adapter="token")
x = torch.randn(1, 4, 4)
with strict_events(), torch.autocast("cpu", dtype=torch.bfloat16):
    try:
        model.forward_sequence(x)
    except StrictFallbackError:
        pass
    else:
        raise AssertionError("strict mode accepted a low-precision CPU core fallback")

reset_routing_log()
events = []
add_event_listener(events.append)
try:
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.forward_sequence(x)
finally:
    remove_event_listener(events.append)
fallback = next(event for event in events if event.code == EventCode.PERFORMANCE_FALLBACK)
assert fallback.component_id == "core_scan_batched"
assert fallback.fields["requested_path"] == "cpu_native_cpp"
assert fallback.fields["selected_path"] == "registered_reference"
assert model.backbone.blocks[0].core._last_core_backend == "registered_reference"
assert torch.isfinite(output).all()
"""
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_registered_native_cpu_core_matches_reference_forward_and_backward():
    program = r"""
import copy
import torch
from dabsn import DABSNLayerSpec, DABSNModel
from dabsn.kernels import enable

torch.manual_seed(441)
reference = DABSNModel(
    4,
    3,
    [DABSNLayerSpec(6, read_geometry="seq")],
    output_adapter="token",
)
native = copy.deepcopy(reference)
x_reference = torch.randn(2, 5, 4, requires_grad=True)
x_native = x_reference.detach().clone().requires_grad_(True)
expected = reference.forward_sequence(x_reference)
expected.square().mean().backward()

enable("cpu", required=True)
actual = native.forward_sequence(x_native)
actual.square().mean().backward()
torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
torch.testing.assert_close(x_native.grad, x_reference.grad, atol=3e-5, rtol=3e-5)
for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
    reference.named_parameters(), native.named_parameters()
):
    assert actual_name == expected_name
    if expected_parameter.grad is None:
        assert actual_parameter.grad is None
    else:
        assert actual_parameter.grad is not None
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            atol=5e-5,
            rtol=5e-5,
        )
"""
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
