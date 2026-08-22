import pytest
import torch
import torch.nn as nn

from dabsn.components import (
    AnonymousComponentError,
    AxisContract,
    ComponentCapabilities,
    ComponentContract,
    ComponentContractError,
    ComponentOutput,
    ResultDeclaration,
    StateDeclaration,
    ValueContract,
    bind_module,
)
from dabsn.graph import DABSNGraph, UnsupportedExecutionModeError


def _world(width):
    return ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("experience", "T", dynamic=True),
        AxisContract("world", width),
    )


def _binding(name, module, in_width, out_width=None, **kwargs):
    out_width = in_width if out_width is None else out_width
    return bind_module(
        name,
        module,
        ComponentContract(_world(in_width), _world(out_width)),
        **kwargs,
    )


def test_graph_is_ordered_and_components_may_be_terminal():
    order = []

    class Mark(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, value):
            order.append(self.name)
            return value + 1

    graph = DABSNGraph(
        [_binding("a", Mark("a"), 8), _binding("b", Mark("b"), 8)],
        input_contract=_world(8),
    )
    value = graph(torch.zeros(2, 3, 8))
    assert order == ["a", "b"]
    torch.testing.assert_close(value, torch.full_like(value, 2))


def test_graph_rejects_contract_mismatch_before_any_forward():
    with pytest.raises(ComponentContractError) as caught:
        DABSNGraph(
            [
                _binding("producer", nn.Identity(), 8, 8),
                _binding("consumer", nn.Identity(), 16, 16),
            ]
        )
    text = str(caught.value)
    assert "producer" in text and "consumer" in text
    assert "world" in text and "expected size 16, received 8" in text


def test_language_graph_requires_dabsn_world_builder_first():
    component = _binding("plain", nn.Identity(), 8)
    with pytest.raises(ValueError, match="DABSN world-building"):
        DABSNGraph([component], require_world_builder=True)
    component.capabilities = ComponentCapabilities(world_builder=True)
    DABSNGraph([component], require_world_builder=True)


def test_forward_with_terms_flattens_declared_fixed_results():
    class WithTerms(nn.Module):
        def forward(self, value):
            return value * 2

        def forward_with_terms(self, value):
            output = value * 2
            return ComponentOutput(output, (output.mean(),), (output.norm(),))

    graph = DABSNGraph(
        [
            _binding(
                "terms",
                WithTerms(),
                8,
                loss_terms=(ResultDeclaration("balance", "mean"),),
                reports=(ResultDeclaration("norm", "none"),),
            ),
            _binding("plain", nn.Identity(), 8),
        ]
    )
    result = graph.forward_with_terms(torch.ones(2, 3, 8))
    assert isinstance(result, ComponentOutput)
    assert len(result.loss_terms) == len(graph.loss_declarations) == 1
    assert len(result.reports) == len(graph.report_declarations) == 1
    torch.testing.assert_close(result.value, torch.full_like(result.value, 2))


def test_graph_initializes_and_carries_explicit_streaming_state():
    class Stateful(nn.Module):
        def forward(self, value):
            return value

        def forward_with_state(self, value, state):
            previous = value.new_zeros(()) if state is None else state[0]
            current = previous + value.sum()
            return ComponentOutput(value + current, next_state=(current,))

    graph = DABSNGraph(
        [
            _binding(
                "stateful",
                Stateful(),
                8,
                capabilities=ComponentCapabilities(streaming_state=True),
                states=(StateDeclaration("accumulator"),),
            )
        ]
    )
    value = torch.ones(2, 3, 8)
    first = graph.forward_with_state(value)
    second = graph.forward_with_state(value, first.next_state)
    assert graph.state_declarations == (StateDeclaration("accumulator"),)
    torch.testing.assert_close(first.next_state[0], torch.tensor(48.0))
    torch.testing.assert_close(second.next_state[0], torch.tensor(96.0))


def test_graph_rejects_streaming_state_arity_mismatches():
    class BadState(nn.Module):
        def forward(self, value):
            return value

        def forward_with_state(self, value, state):
            return ComponentOutput(value)

    graph = DABSNGraph(
        [
            _binding(
                "bad-state",
                BadState(),
                8,
                capabilities=ComponentCapabilities(streaming_state=True),
                states=(StateDeclaration("memory"),),
            )
        ]
    )
    with pytest.raises(ValueError, match="expected 1 state tensors"):
        graph.forward_with_state(torch.ones(2, 3, 8), ())
    with pytest.raises(RuntimeError, match="declared 1 state tensors"):
        graph.forward_with_state(torch.ones(2, 3, 8))


def test_binding_requires_state_capability_and_declarations_to_agree():
    with pytest.raises(ValueError, match="state slots must be declared exactly"):
        _binding(
            "undeclared-state",
            nn.Identity(),
            8,
            capabilities=ComponentCapabilities(streaming_state=True),
        )


def test_post_step_actions_never_run_for_accumulation_or_skipped_amp_step():
    class Lifecycle(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            return value

        def post_optimizer_step(self):
            self.calls += 1

    module = Lifecycle()
    graph = DABSNGraph([_binding("lifecycle", module, 8)])
    graph.post_optimizer_step(step_applied=False)
    assert module.calls == 0
    graph.post_optimizer_step(step_applied=True)
    assert module.calls == 1


def test_capability_report_fails_before_requested_unsupported_execution():
    graph = DABSNGraph(
        [
            _binding(
                "limited",
                nn.Identity(),
                8,
                capabilities=ComponentCapabilities(
                    eager=True, compile_fullgraph=False, export=False
                ),
            )
        ]
    )
    assert graph.capability_report().supported["eager"]
    assert not graph.capability_report().supported["compile_fullgraph"]
    with pytest.raises(UnsupportedExecutionModeError, match="limited"):
        graph.require_capabilities("compile_fullgraph")


def test_graph_does_not_claim_anonymous_component_is_portable():
    graph = DABSNGraph([_binding("plain", nn.Identity(), 8)])
    with pytest.raises(AnonymousComponentError):
        graph.component_specs(portable=True)


def test_only_world_builders_own_dabsn_memory():
    first = _binding("dabsn", nn.Identity(), 8)
    first.capabilities = ComponentCapabilities(world_builder=True, dabsn_memory_owner=True)
    graph = DABSNGraph([first, _binding("mlp", nn.Identity(), 8)])
    assert graph.dabsn_memory_count == 1
