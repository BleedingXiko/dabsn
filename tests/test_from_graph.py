import pytest
import torch
import torch.nn as nn

from dabsn import (
    AxisContract,
    ComponentCapabilities,
    ComponentContract,
    DABSNGraph,
    DABSNSequenceLM,
    ValueContract,
    bind_module,
)


def _world(width):
    return ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("experience", "T", dynamic=True),
        AxisContract("world", width),
    )


def _graph(in_width=6, out_width=6):
    first = bind_module(
        "dabsn.0",
        nn.Linear(in_width, out_width, bias=False),
        ComponentContract(_world(in_width), _world(out_width)),
        capabilities=ComponentCapabilities(world_builder=True, dabsn_memory_owner=True),
    )
    downstream = bind_module(
        "user.0",
        nn.Linear(out_width, out_width, bias=False),
        ComponentContract(_world(out_width), _world(out_width)),
    )
    return DABSNGraph([first, downstream], require_world_builder=True)


def test_from_graph_builds_language_model_over_exact_graph():
    graph = _graph()
    model = DABSNSequenceLM.from_graph(graph, vocab=31)
    assert model.graph is graph
    assert model.depth == 1
    assert model.forward_sequence(torch.randint(0, 31, (2, 5))).shape == (2, 5, 31)


def test_from_graph_rejects_non_dabsn_first_component():
    contract = ComponentContract(_world(6), _world(6))
    graph = DABSNGraph([bind_module("plain", nn.Identity(), contract)])
    with pytest.raises(ValueError, match="begin with DABSN"):
        DABSNSequenceLM.from_graph(graph, vocab=31)


def test_from_graph_tied_embeddings_require_same_endpoint_width():
    with pytest.raises(ValueError, match="equal graph input/output"):
        DABSNSequenceLM.from_graph(_graph(6, 8), vocab=31, tie_embeddings=True)


def test_from_graph_training_path_projects_value_and_preserves_terms():
    model = DABSNSequenceLM.from_graph(_graph(), vocab=31)
    result = model.forward_with_terms(torch.randint(0, 31, (2, 5)))
    assert result.value.shape == (2, 5, 31)
    assert result.loss_terms == ()


def test_from_graph_diagnostic_surface_is_defined_without_legacy_backbone():
    model = DABSNSequenceLM.from_graph(_graph(), vocab=31)
    assert model.read_traces() == []
    assert model.signal_traces() == []
    model.set_hybrid_gate_override(0.25)
