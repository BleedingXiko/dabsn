import pytest
import torch

from dabsn import (
    BuildContext,
    ComponentSpec,
    ReLU2MLPExpertGroup,
    component_registry,
    dropless_dispatch,
)


def _dispatch(hidden=8, experts=4, items=13, top_k=2):
    inputs = torch.randn(items, hidden)
    indices = torch.randint(0, experts, (items, top_k))
    weights = torch.softmax(torch.randn(items, top_k), -1)
    return dropless_dispatch(inputs, indices, weights, experts)


def test_grouped_relu2_matches_reference_forward_and_backward():
    torch.manual_seed(31)
    reference = ReLU2MLPExpertGroup(4, 8, 16, backend="reference")
    grouped = ReLU2MLPExpertGroup(4, 8, 16, backend="grouped")
    grouped.load_state_dict(reference.state_dict())
    dispatch = _dispatch()
    ref_inputs = dispatch.sorted_inputs.detach().clone().requires_grad_(True)
    fast_inputs = dispatch.sorted_inputs.detach().clone().requires_grad_(True)
    expected = reference.forward_dispatched(ref_inputs, dispatch.sorted_experts, dispatch.offsets)
    actual = grouped.forward_dispatched(fast_inputs, dispatch.sorted_experts, dispatch.offsets)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(fast_inputs.grad, ref_inputs.grad)
    torch.testing.assert_close(grouped.w1.grad, reference.w1.grad)
    torch.testing.assert_close(grouped.w2.grad, reference.w2.grad)


def test_grouped_backend_refuses_unsupported_alignment_instead_of_falling_back():
    group = ReLU2MLPExpertGroup(3, 5, 7, backend="grouped")
    dispatch = _dispatch(hidden=5, experts=3)
    with pytest.raises(RuntimeError, match="select backend='reference'"):
        group.forward_dispatched(dispatch.sorted_inputs, dispatch.sorted_experts, dispatch.offsets)


def test_provider_capability_is_configuration_dependent():
    common = {
        "hidden_dim": 8,
        "experts": 4,
        "top_k": 2,
        "inner_dim": 16,
        "router": "switch",
        "balance_coefficient": 0.01,
        "normalization": "none",
    }
    reference = component_registry.build(
        ComponentSpec("ref", "dabsn:sparse_moe", {**common, "backend": "reference"})
    )
    grouped = component_registry.build(
        ComponentSpec("fast", "dabsn:sparse_moe", {**common, "backend": "grouped"})
    )
    assert not reference.capabilities.compile_fullgraph
    assert not reference.capabilities.cuda_graph
    assert not grouped.capabilities.compile_fullgraph
    assert not grouped.capabilities.cuda_graph
    declared_cuda = component_registry.build(
        ComponentSpec(
            "fast-cuda",
            "dabsn:sparse_moe",
            {**common, "backend": "grouped"},
        ),
        BuildContext(device=torch.device("cuda"), dtype=torch.bfloat16),
    )
    assert declared_cuda.capabilities.compile_fullgraph
    assert declared_cuda.capabilities.cuda_graph
