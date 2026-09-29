"""Tensor boundaries needed by FSDP hooks and compiled numerical kernels."""

import pytest
import torch
from torch.utils._pytree import GetAttrKey, tree_flatten, tree_flatten_with_path, tree_unflatten

from dabsn.components import ComponentOutput


def test_component_output_exposes_all_tensors_to_distributed_backward_hooks():
    value, loss, report, state = [torch.randn(2, requires_grad=True) for _ in range(4)]
    output = ComponentOutput({"world": value}, (loss,), (report,), (state,))
    leaves, spec = tree_flatten(output)
    assert len(leaves) == 4
    assert all(actual is expected for actual, expected in zip(leaves, (value, loss, report, state)))
    restored = tree_unflatten(leaves, spec)
    assert isinstance(restored, ComponentOutput)
    assert restored.value["world"] is value
    assert restored.loss_terms[0] is loss
    assert restored.reports[0] is report
    assert restored.next_state[0] is state
    paths, _ = tree_flatten_with_path(output)
    assert [path[0] for path, _tensor in paths] == [
        GetAttrKey("value"), GetAttrKey("loss_terms"), GetAttrKey("reports"), GetAttrKey("next_state")
    ]


@pytest.mark.parametrize("backward_only", [False, True])
def test_compiled_scan_handles_fsdp_flat_parameter_gradient_metadata(monkeypatch, backward_only):
    """Full gathered views may retain a base whose gradient is still sharded."""
    import dabsn.kernels.batched_runtime as runtime

    torch._dynamo.reset()
    hidden = 2
    values = (
        torch.randn(1, 3, hidden), torch.randn(1, 3, hidden),
        torch.randn(hidden, hidden), torch.randn(hidden, hidden),
        *[torch.randn(hidden) * .1 for _ in range(16)],
        torch.tensor([-2.]), torch.tensor([-1.]), torch.tensor([-6.]),
        torch.zeros(1, hidden), torch.ones(1, hidden), torch.zeros(1, hidden),
    )
    reference = tuple(value.clone().requires_grad_() for value in values)
    flat_values = torch.cat([value.flatten() for value in values[2:23]])
    # This is the storage transition FSDP1 makes: .grad stays shard-sized
    # while .data is replaced by the all-gathered full parameter storage.
    flat = torch.nn.Parameter(flat_values[:flat_values.numel() // 2].clone())
    flat.grad = torch.zeros_like(flat)
    flat.data = flat_values
    inputs = [value.clone().requires_grad_() for value in values[:2]]
    offset = 0
    for value in values[2:23]:
        inputs.append(flat[offset:offset + value.numel()].view_as(value))
        offset += value.numel()
    inputs.extend(value.clone().requires_grad_() for value in values[23:])

    compiled = {}
    def compile_step(fn, which, sample):
        if which not in compiled:
            compiled[which] = torch.compile(fn, backend="eager", fullgraph=True)
        return compiled[which]

    # Build the reference autograd graph without the manual scan's mutable
    # tape/carry buffers, so PyTorch independently differentiates its recurrence.
    wx, wgx, ug, a, *rest = reference
    budget, energy, saturation = rest[-3:]
    columns = [[] for _ in range(7)]
    for step in range(wx.shape[1]):
        y, budget, energy, saturation, novelty, plasticity, expression, write, energy_before, _gate = (
            runtime._forward_step(wx[:, step], wgx[:, step], torch.cat((ug, a)),
                                  budget, energy, saturation, *rest[:-3])
        )
        for column, value in zip(columns, (torch.cat((y, budget), -1), novelty,
                                          plasticity, expression, write, energy_before, saturation)):
            column.append(value)
    expected = (*[torch.stack(column, 1) for column in columns], budget, energy, saturation)
    expected_grads = torch.autograd.grad(sum(t.square().sum() for t in expected), reference)
    monkeypatch.setattr(runtime, "_compiled_step", compile_step)
    monkeypatch.setattr(runtime, "_step_compile_enabled", lambda tensor: True)
    if backward_only:
        # Isolate backward conversion rather than failing first in forward.
        with monkeypatch.context() as forward_patch:
            forward_patch.setattr(runtime, "_step_compile_enabled", lambda tensor: False)
            actual = runtime.DABSNCoreScanBatched.apply(*inputs, True, True)
    else:
        actual = runtime.DABSNCoreScanBatched.apply(*inputs, True, True)
    actual_grads = torch.autograd.grad(sum(t.square().sum() for t in actual), inputs)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want)
    assert "backward" in compiled
    if not backward_only:
        assert "forward" in compiled
    torch._dynamo.reset()
