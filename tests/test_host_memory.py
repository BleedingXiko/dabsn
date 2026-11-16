"""DABSNHostMemory: the canonical DABSN read as a sidecar for any host model."""
import math

import pytest
import torch

from dabsn import DABSNBlock, DABSNCore, DABSNHostMemory, DABSNRead
from dabsn.components import BuildContext, ComponentSpec, component_registry
from dabsn.conformance import check_component
from dabsn.host_memory import HostMemoryComponent, _HostMemoryReferenceRead


def _memory(**kwargs):
    torch.manual_seed(3)
    return DABSNHostMemory(16, kernels="reference", **kwargs).double()


def test_no_dabsn_core_or_block_is_instantiated():
    memory = _memory()
    assert isinstance(memory.read, DABSNRead)
    assert not any(isinstance(m, (DABSNCore, DABSNBlock)) for m in memory.modules())
    assert memory.kernel_status["core_scan_instantiated"] is False


def test_forward_is_the_canonical_read_and_memory_only():
    memory = _memory()
    h = torch.randn(2, 12, 16, dtype=torch.float64)
    signals = memory.signals_from_host(h)
    assert torch.count_nonzero(signals["y"]) == 0 and torch.count_nonzero(signals["budget"]) == 0
    # Four tags are stride-0 views of one scalar per token.
    for name in ("novelty", "plasticity", "energy", "saturation"):
        assert signals[name].stride(-1) == 0
    torch.testing.assert_close(memory(h), memory.read(**signals), rtol=0, atol=0)


def test_admission_window_initialized_to_host_calibration():
    for window in (1.5, 0.25, 4.0):
        memory = DABSNHostMemory(8, kernels="reference", admit_window=window)
        assert math.isclose(float(torch.nn.functional.softplus(memory.read.admit_window)),
                            window, rel_tol=1e-6)
    with pytest.raises(ValueError):
        DABSNHostMemory(8, admit_window=0.0)


@pytest.mark.parametrize("geometry", ["seq"])
def test_seq_memory_is_causal(geometry):
    memory = _memory(read_geometry=geometry)
    h = torch.randn(2, 20, 16, dtype=torch.float64)
    changed = h.clone()
    changed[:, 11:] = torch.randn_like(changed[:, 11:]) * 3.0
    torch.testing.assert_close(memory(h)[:, :11], memory(changed)[:, :11], rtol=0, atol=1e-12)


def test_use_long_false_disables_and_freezes_long_memory():
    memory = _memory(use_long=False)
    assert float(memory.read.long_gain) == 0.0
    assert not memory.read.long_gain.requires_grad


def test_reference_mode_never_inherits_accelerated_kernel_flags():
    DABSNHostMemory(8, kernels="auto")
    reference = DABSNHostMemory(8, kernels="reference")
    assert type(reference.read) is _HostMemoryReferenceRead
    for flag in ("_long_native_enabled", "_cuda_native_enabled"):
        assert not getattr(type(reference.read), flag, False)
    # Ordinary DABSN reads are untouched by host-memory kernel selection.
    assert not getattr(DABSNRead, "_cuda_native_enabled", False) or torch.cuda.is_available()


def test_gradients_reach_host_latents_and_adapters():
    memory = _memory()
    h = torch.randn(2, 10, 16, dtype=torch.float64, requires_grad=True)
    memory(h).square().mean().backward()
    assert torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0
    for module in (memory.write_tags, memory.query_proj, memory.write_proj):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0


def test_provider_builds_residual_component_and_passes_conformance():
    binding = component_registry.build(
        ComponentSpec("memory.0", "dabsn:host_memory", {"dim": 16, "kernels": "reference"}),
        BuildContext(device=torch.device("cpu"), dtype=torch.float32),
    )
    assert isinstance(binding.module, HostMemoryComponent)
    x = torch.randn(2, 6, 16)
    torch.testing.assert_close(binding.module(x), x + binding.module.memory(x))
    report = check_component("dabsn:host_memory", {"dim": 16, "kernels": "reference"})
    failed = [c for c in report.checks if c.status == "fail"]
    assert not failed, failed


def test_provider_rejects_unknown_or_invalid_config():
    for bad in ({"dim": 16, "bogus": 1}, {"dim": 0}, {"dim": 16, "read_geometry": "grid"},
                {"dim": 16, "admit_window": -1.0}, {"dim": 16, "kernels": "fast"}):
        with pytest.raises(Exception):
            component_registry.build(ComponentSpec("m", "dabsn:host_memory", bad), BuildContext())
