import os

import pytest
import torch

from dabsn import (
    ComponentSpec,
    DABSNGraph,
    DABSNLayerSpec,
    DABSNModel,
    DABSNTaskModel,
    component_registry,
)
from dabsn.model import DABSNSequenceLM
from dabsn.runtime import train_step
from dabsn.temporal_credit import initial_online_temporal_state

DEVICE = os.environ.get("DABSN_TEST_COMPOSITION_DEVICE", "cpu")


@pytest.fixture(autouse=True, scope="module")
def _composition_device():
    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA composition gate requires CUDA")
        from dabsn.kernels import enable
        enable("cuda", required=True)
    previous = torch.get_default_device()
    torch.set_default_device(DEVICE)
    try:
        yield
    finally:
        torch.set_default_device(previous)


@pytest.mark.parametrize("geometry", ["seq", "field", "hybrid"])
@pytest.mark.parametrize("depth", [1, 2])
def test_online_model_composition_forward_and_update(geometry: str, depth: int) -> None:
    torch.manual_seed(501 + depth)
    layers = [
        DABSNLayerSpec(7, 5, geometry),
        DABSNLayerSpec(6, 4, geometry),
    ][:depth]
    model = DABSNModel(
        input_dim=4,
        out_dim=3,
        layers=layers,
        output_adapter="token",
        residual=True,
        mlp_ratio=1.5,
        mlp_middle_depth=2 if depth == 2 else 0,
        mlp_depth_index=0,
    )
    inputs = torch.randn(2, 4, 4)
    targets = torch.randn(2, 4, 3)
    with torch.no_grad():
        expected = model.forward_sequence(inputs)
    state = initial_online_temporal_state(
        model,
        2,
        stream_policy="reset_each_batch",
        stream_ids=(0, 1),
        device=DEVICE,
        dtype=torch.float32,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    output, pending = model.forward_sequence_online(inputs, state)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    output.square().mean().backward()
    optimizer.zero_grad(set_to_none=True)
    loss, completed = train_step(
        model, inputs, targets, optimizer, temporal_state=state
    )
    assert torch.isfinite(torch.tensor(loss))
    completed.assert_checkpointable()


def test_task_adapter_online_composition() -> None:
    torch.manual_seed(510)
    model = DABSNTaskModel(
        raw_input_dim=3,
        model_input_dim=5,
        out_dim=2,
        layers=[DABSNLayerSpec(6, 4, "seq")],
        input_adapter="linear",
        output_adapter="token",
        residual=True,
    )
    inputs = torch.randn(2, 4, 3)
    state = initial_online_temporal_state(
        model,
        2,
        stream_policy="reset_each_batch",
        stream_ids=(0, 1),
        device=DEVICE,
        dtype=torch.float32,
    )
    with torch.no_grad():
        expected = model.forward_sequence(inputs)
    actual, pending = model.forward_sequence_online(inputs, state)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    assert pending.active_frames


def test_sequence_lm_online_training_composition() -> None:
    torch.manual_seed(511)
    model = DABSNSequenceLM(
        vocab=29,
        hidden_dim=8,
        depth=2,
        layers=[
            DABSNLayerSpec(9, 7, "seq"),
            DABSNLayerSpec(8, 6, "seq"),
        ],
        residual=True,
        mlp_ratio=2.0,
        mlp_middle_depth=1,
        mlp_depth_index=0,
    )
    ids = torch.randint(0, 29, (2, 5))
    targets = torch.randint(0, 29, (2, 5))
    state = initial_online_temporal_state(
        model,
        2,
        stream_policy="reset_each_batch",
        stream_ids=(0, 1),
        device=DEVICE,
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss, completed = train_step(
        model,
        ids,
        targets,
        optimizer,
        temporal_state=state,
        temporal_chunk_size=2,
    )
    assert torch.isfinite(torch.tensor(loss))
    completed.assert_checkpointable()
    assert model.embed.weight.grad is not None
    assert model.readout.weight.grad is not None


def _binding(component_id: str, provider: str, config: dict[str, object]):
    return component_registry.build(ComponentSpec(component_id, provider, config))


def test_builtin_providers_declare_online_temporal_policy() -> None:
    dabsn = _binding(
        "dabsn.0",
        "dabsn:block",
        {"input_dim": 6, "hidden_dim": 6, "state_dim": 5, "read_geometry": "seq"},
    )
    mlp = _binding("mlp.0", "dabsn:residual_mlp", {"dim": 6, "ratio": 2.0})
    moe = _binding(
        "moe.0",
        "dabsn:sparse_moe",
        {
            "hidden_dim": 6,
            "experts": 3,
            "top_k": 1,
            "inner_dim": 8,
            "router": "switch",
            "balance_coefficient": 0.01,
        },
    )
    assert dabsn.capabilities.temporal_credit == "dabsn"
    assert mlp.capabilities.temporal_credit == "stateless"
    assert moe.capabilities.temporal_credit == "stateless"


def test_graph_dabsn_mlp_dabsn_trains_online() -> None:
    torch.manual_seed(512)
    graph = DABSNGraph(
        [
            _binding(
                "dabsn.0",
                "dabsn:block",
                {
                    "input_dim": 6,
                    "hidden_dim": 6,
                    "state_dim": 5,
                    "read_geometry": "seq",
                },
            ),
            _binding("mlp.0", "dabsn:residual_mlp", {"dim": 6, "ratio": 2.0}),
            _binding(
                "dabsn.1",
                "dabsn:block",
                {
                    "input_dim": 6,
                    "hidden_dim": 6,
                    "state_dim": 4,
                    "read_geometry": "seq",
                },
            ),
        ],
        require_world_builder=True,
    )
    model = DABSNSequenceLM.from_graph(graph, vocab=19, temporal_credit="online")
    state = initial_online_temporal_state(
        model,
        2,
        stream_policy="reset_each_batch",
        stream_ids=(0, 1),
        device=DEVICE,
        dtype=torch.float32,
    )
    loss, completed = train_step(
        model,
        torch.randint(0, 19, (2, 4)),
        torch.randint(0, 19, (2, 4)),
        torch.optim.SGD(model.parameters(), lr=1e-4),
        temporal_state=state,
        temporal_chunk_size=2,
    )
    assert torch.isfinite(torch.tensor(loss))
    assert len(completed.layers) == 2
    completed.assert_checkpointable()


def test_graph_dabsn_moe_auxiliary_objective_trains_online() -> None:
    torch.manual_seed(513)
    graph = DABSNGraph(
        [
            _binding(
                "dabsn.0",
                "dabsn:block",
                {
                    "input_dim": 6,
                    "hidden_dim": 6,
                    "state_dim": 5,
                    "read_geometry": "seq",
                },
            ),
            _binding(
                "moe.0",
                "dabsn:sparse_moe",
                {
                    "hidden_dim": 6,
                    "experts": 3,
                    "top_k": 1,
                    "inner_dim": 8,
                    "router": "switch",
                    "balance_coefficient": 0.01,
                },
            ),
        ],
        require_world_builder=True,
    )
    model = DABSNSequenceLM.from_graph(graph, vocab=19, temporal_credit="online")
    state = initial_online_temporal_state(
        model,
        2,
        stream_policy="reset_each_batch",
        stream_ids=(0, 1),
        device=DEVICE,
        dtype=torch.float32,
    )
    metrics = {}
    loss, completed = train_step(
        model,
        torch.randint(0, 19, (2, 4)),
        torch.randint(0, 19, (2, 4)),
        torch.optim.SGD(model.parameters(), lr=1e-4),
        temporal_state=state,
        temporal_chunk_size=2,
        metrics_callback=metrics.update,
    )
    assert torch.isfinite(torch.tensor(loss))
    assert "loss_term/0:router_balance" in metrics
    assert graph.bindings[0].module.core.W.weight.grad is not None
    assert graph.bindings[1].module.router.proj.weight.grad is not None
    assert graph.bindings[1].module.expert_group.w1.grad is not None
    completed.assert_checkpointable()
