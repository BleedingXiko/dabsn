import pytest
import torch
import torch.nn as nn

from dabsn import (
    AuxLossFreeTopKRouter,
    ComponentSpec,
    DABSNGraph,
    GenericExpertGroup,
    ReLU2MLPExpertGroup,
    SparseMoEComponent,
    SwitchTopKRouter,
    component_registry,
    dropless_dispatch,
)


def test_dropless_dispatch_conserves_exact_n_times_k_assignments_stably():
    inputs = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    indices = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 1], [0, 2]])
    weights = torch.full((5, 2), 0.5)
    dispatched = dropless_dispatch(inputs, indices, weights, experts=3)
    assert dispatched.sorted_inputs.shape[0] == 5 * 2
    assert dispatched.counts.tolist() == [3, 3, 4]
    assert dispatched.offsets.tolist() == [0, 3, 6, 10]
    assert dispatched.sorted_experts.tolist() == sorted(indices.flatten().tolist())
    assert dispatched.assignments.numel() == 10


def test_relu2_group_matches_independent_mathematical_experts_forward_backward():
    torch.manual_seed(4)
    group = ReLU2MLPExpertGroup(3, 5, 7).double()
    inputs = torch.randn(11, 5, dtype=torch.double, requires_grad=True)
    expert_ids = torch.tensor([0, 2, 1, 1, 0, 2, 2, 0, 1, 2, 0])
    actual = group.forward_dispatched(inputs, expert_ids)
    expected = torch.empty_like(inputs)
    for index in range(3):
        positions = torch.nonzero(expert_ids == index).flatten()
        expected[positions] = group.forward_expert(index, inputs[positions])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert group.w1.grad is not None and torch.isfinite(group.w1.grad).all()
    assert group.w2.grad is not None and torch.isfinite(group.w2.grad).all()


def test_sparse_moe_matches_direct_topk_oracle_and_has_zero_drops():
    torch.manual_seed(5)
    router = SwitchTopKRouter(4, 3, 2)
    group = ReLU2MLPExpertGroup(3, 4, 6)
    moe = SparseMoEComponent(4, router, group)
    value = torch.randn(2, 5, 4)
    result = moe.forward_with_terms(value)

    items = value.reshape(-1, 4)
    routed = router(items)
    oracle = torch.zeros_like(items)
    for item in range(items.shape[0]):
        for slot in range(2):
            expert = int(routed.expert_indices[item, slot])
            oracle[item] += (
                routed.expert_weights[item, slot]
                * group.forward_expert(expert, items[item : item + 1])[0]
            )
    torch.testing.assert_close(result.value, oracle.reshape_as(value))
    assert int(result.reports[-2]) == items.shape[0] * 2
    assert int(result.reports[-1]) == 0


def test_router_is_nonsymmetric_and_receives_balance_gradients():
    torch.manual_seed(6)
    router = SwitchTopKRouter(8, 4, 2)
    assert not torch.equal(router.proj.weight[0], router.proj.weight[1])
    output = router(torch.randn(13, 8))
    output.balance_loss.backward()
    assert router.proj.weight.grad is not None
    assert torch.isfinite(router.proj.weight.grad).all()


def test_generic_experts_run_their_own_modules_without_architecture_branches():
    class Custom(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(float(scale)))

        def forward(self, value):
            return value * self.scale

    experts = GenericExpertGroup([Custom(1), Custom(2), Custom(3)])
    router = SwitchTopKRouter(4, 3, 1)
    moe = SparseMoEComponent(4, router, experts)
    value = torch.randn(2, 5, 4, requires_grad=True)
    output = moe(value)
    output.sum().backward()
    assert output.shape == value.shape
    assert any(expert.scale.grad is not None for expert in experts.experts)


def test_aux_loss_free_bias_changes_only_after_real_optimizer_step():
    torch.manual_seed(7)
    router = AuxLossFreeTopKRouter(4, 3, 1, bias_update_rate=0.1)
    group = ReLU2MLPExpertGroup(3, 4, 5)
    moe = SparseMoEComponent(4, router, group)
    binding = component_registry.build(
        ComponentSpec(
            "portable",
            "dabsn:sparse_moe",
            {
                "hidden_dim": 4,
                "experts": 3,
                "top_k": 1,
                "inner_dim": 5,
                "router": "aux_loss_free",
            },
        )
    )
    # Use the local module in a binding with the provider's static declarations.
    binding.module = moe
    graph = DABSNGraph([binding])
    before = router.selection_bias.clone()
    graph.forward_with_terms(torch.randn(2, 3, 4))
    graph.post_optimizer_step(step_applied=False)
    torch.testing.assert_close(router.selection_bias, before)
    graph.post_optimizer_step(step_applied=True)
    assert not torch.equal(router.selection_bias, before)
    assert torch.count_nonzero(router._pending_counts) == 0


def test_aux_loss_free_router_reduces_counts_before_distributed_update(monkeypatch):
    router = AuxLossFreeTopKRouter(4, 3, 1, bias_update_rate=0.1)
    router._pending_counts.copy_(torch.tensor([4.0, 1.0, 1.0]))
    calls = []
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)

    def all_reduce(value, *, op, group=None):
        assert group is None
        calls.append(op)
        value.add_(torch.tensor([0.0, 3.0, 3.0]))

    monkeypatch.setattr("torch.distributed.all_reduce", all_reduce)
    router.post_optimizer_step()
    assert calls == [torch.distributed.ReduceOp.SUM]
    torch.testing.assert_close(router.selection_bias, torch.zeros(3))
    torch.testing.assert_close(router._pending_counts, torch.zeros(3))


def test_zero_output_initialization_is_explicit_not_framework_default():
    normal = ReLU2MLPExpertGroup(3, 4, 5)
    zero = ReLU2MLPExpertGroup(3, 4, 5, zero_output=True)
    assert torch.count_nonzero(normal.w2) > 0
    assert torch.count_nonzero(zero.w2) == 0


def test_portable_moe_provider_declares_fixed_terms_reports_and_roundtrips_spec():
    binding = component_registry.build(
        ComponentSpec(
            "moe.0",
            "dabsn:sparse_moe",
            {
                "hidden_dim": 8,
                "experts": 24,
                "top_k": 4,
                "inner_dim": 16,
                "router": "switch",
                "balance_coefficient": 0.01,
                "normalization": "none",
                "routing_granularity": "individual_h",
            },
        )
    )
    assert len(binding.loss_terms) == 1
    assert len(binding.reports) == 10
    assert binding.to_spec().provider_key == "dabsn:sparse_moe"
    result = binding.module.forward_with_terms(torch.randn(2, 3, 8))
    assert len(result.loss_terms) == 1
    assert len(result.reports) == 10


def test_reported_confidence_is_selected_raw_probability_not_one_over_k():
    torch.manual_seed(81)
    router = SwitchTopKRouter(8, 24, 4)
    group = ReLU2MLPExpertGroup(24, 8, 16, backend="reference")
    moe = SparseMoEComponent(8, router, group)
    value = torch.randn(2, 3, 8)
    result = moe.forward_with_terms(value)
    routed = router(value.reshape(-1, 8))
    expected = torch.gather(routed.probabilities, 1, routed.expert_indices).mean()
    torch.testing.assert_close(result.reports[6], expected)
    assert float(result.reports[6].detach()) != pytest.approx(1 / 4)


def test_nested_moe_is_an_ordinary_generic_expert():
    torch.manual_seed(82)
    inner = SparseMoEComponent(
        8,
        SwitchTopKRouter(8, 2, 1),
        ReLU2MLPExpertGroup(2, 8, 16, backend="reference"),
    )
    outer = SparseMoEComponent(
        8,
        SwitchTopKRouter(8, 2, 1),
        GenericExpertGroup([inner, nn.Linear(8, 8, bias=False)]),
    )
    value = torch.randn(2, 3, 8, requires_grad=True)
    output = outer(value)
    assert output.shape == value.shape
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_single_expert_balance_entropy_is_well_defined():
    moe = SparseMoEComponent(
        8,
        SwitchTopKRouter(8, 1, 1),
        ReLU2MLPExpertGroup(1, 8, 16, backend="reference"),
    )
    result = moe.forward_with_terms(torch.randn(2, 3, 8))
    torch.testing.assert_close(result.reports[2], torch.ones(()))


def test_invalid_capacity_drop_configuration_is_not_part_of_production_provider():
    with pytest.raises(ValueError, match="unknown dabsn:sparse_moe"):
        component_registry.build(
            ComponentSpec(
                "moe",
                "dabsn:sparse_moe",
                {
                    "hidden_dim": 8,
                    "experts": 4,
                    "top_k": 2,
                    "inner_dim": 16,
                    "capacity_factor": 1.0,
                    "routing_granularity": "token_drop",
                },
            )
        )
