import pytest
import torch

from dabsn import DABSNLayerSpec, DABSNModel
from dabsn.temporal_credit import (
    finalize_online_temporal_state,
    initial_online_temporal_state,
)


def _model(depth: int = 2) -> DABSNModel:
    layers = [
        DABSNLayerSpec(hidden_dim=5, state_dim=3, read_geometry="seq"),
        DABSNLayerSpec(hidden_dim=6, state_dim=4, read_geometry="seq"),
    ][:depth]
    return DABSNModel(
        input_dim=4,
        out_dim=2,
        layers=layers,
        output_adapter="token",
        residual=True,
        mlp_ratio=2.0,
    )


def test_multicore_online_forward_replays_each_layer_and_detaches_carry() -> None:
    torch.manual_seed(301)
    model = _model(2)
    inputs = torch.randn(2, 5, 4, requires_grad=True)
    with torch.no_grad():
        expected = model.forward_sequence(inputs.detach())
    state = initial_online_temporal_state(
        model,
        batch_size=2,
        stream_policy="reset_each_batch",
        stream_ids=(0, 1),
        device=inputs.device,
        dtype=inputs.dtype,
    )

    actual, pending = model.forward_sequence_online(inputs, state)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(RuntimeError, match="pending online replay"):
        pending.assert_checkpointable()
    actual.square().mean().backward()
    completed = finalize_online_temporal_state(pending)

    completed.assert_checkpointable()
    assert completed.chunk_position == 1
    assert not completed.active_frames
    assert all(
        tensor.grad_fn is None
        for layer in completed.layers
        for tensor in (*layer.core_recurrent, *layer.long_recurrent)
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    for block in model.backbone.blocks:
        assert all(parameter.grad is not None for parameter in block.core.parameters())
        assert all(
            getattr(block.read, name).grad is not None
            for name in (
                "logit_retain",
                "logit_expect_retain",
                "logit_retention_decay",
                "logit_retention_strength",
            )
        )


def test_stateless_mlp_and_head_gradients_match_one_chunk_bptt() -> None:
    torch.manual_seed(302)
    reference = _model(1)
    actual = _model(1)
    actual.load_state_dict(reference.state_dict())
    inputs = torch.randn(2, 4, 4)
    target = torch.randn(2, 4, 2)

    torch.nn.functional.mse_loss(reference.forward_sequence(inputs), target).backward()
    state = initial_online_temporal_state(
        actual,
        batch_size=2,
        stream_policy="reset_each_batch",
        stream_ids=(0, 1),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    output, pending = actual.forward_sequence_online(inputs, state)
    torch.nn.functional.mse_loss(output, target).backward()
    finalize_online_temporal_state(pending)

    pairs = (
        (actual.backbone.blocks[0].mlp_fc1, reference.backbone.blocks[0].mlp_fc1),
        (actual.backbone.blocks[0].mlp_fc2, reference.backbone.blocks[0].mlp_fc2),
        (actual.output_adapter, reference.output_adapter),
    )
    for actual_module, reference_module in pairs:
        for actual_parameter, reference_parameter in zip(
            actual_module.parameters(), reference_module.parameters()
        ):
            torch.testing.assert_close(
                actual_parameter.grad,
                reference_parameter.grad,
                rtol=2e-5,
                atol=2e-6,
            )
