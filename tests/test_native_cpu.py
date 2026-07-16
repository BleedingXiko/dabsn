import subprocess
import sys


def test_required_native_cpu_installed_surface():
    program = r'''
import torch
from dabsn import DABSNLayerSpec, DABSNModel
from dabsn.kernels import enable, local_field_gather, permanent_delta_scan, status
enable("cpu", required=True)
model = DABSNModel(4, 2, [DABSNLayerSpec(6, read_geometry="seq")], output_adapter="token")
x = torch.randn(1, 4, 4, requires_grad=True)
y = model.forward_sequence(x)
y.square().mean().backward()
assert torch.isfinite(y).all() and x.grad is not None
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
'''
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_optional_cpu_runtime_falls_back_when_extension_is_unavailable():
    program = r'''
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
'''
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
