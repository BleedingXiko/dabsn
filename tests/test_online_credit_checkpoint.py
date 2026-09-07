import copy

import pytest
import torch

from dabsn import DABSNLayerSpec, DABSNModel
from dabsn.runtime import train_step
from dabsn.runtime.api import _online_backward_step, apply_optimizer_step
from dabsn.temporal_credit import (
    OnlineTemporalState,
    initial_online_temporal_state,
    load_online_temporal_state,
    load_online_training_state,
    online_temporal_state_dict,
    online_training_state_dict,
)


def _model(*, state_dim: int = 5) -> DABSNModel:
    return DABSNModel(
        input_dim=4,
        out_dim=3,
        layers=[DABSNLayerSpec(4, state_dim, "seq")],
        output_adapter="token",
    )


def _initial(model: DABSNModel) -> OnlineTemporalState:
    return initial_online_temporal_state(
        model,
        2,
        stream_policy="carry_lanes",
        stream_ids=("left", "right"),
        device="cpu",
        dtype=torch.float32,
    )


def _assert_state_equal(actual: OnlineTemporalState, expected: OnlineTemporalState) -> None:
    assert actual.stream_policy == expected.stream_policy
    assert actual.stream_ids == expected.stream_ids
    assert actual.chunk_position == expected.chunk_position
    assert actual.accumulation_position == expected.accumulation_position
    torch.testing.assert_close(actual.pending_reset_mask, expected.pending_reset_mask)
    assert len(actual.layers) == len(expected.layers)
    for actual_layer, expected_layer in zip(actual.layers, expected.layers):
        for lhs, rhs in zip(actual_layer.core_recurrent, expected_layer.core_recurrent):
            torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
        for lhs, rhs in zip(
            actual_layer.core_eligibility.tensors(),
            expected_layer.core_eligibility.tensors(),
        ):
            torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
        for lhs, rhs in zip(actual_layer.long_recurrent, expected_layer.long_recurrent):
            torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
        torch.testing.assert_close(
            actual_layer.long_eligibility.traces,
            expected_layer.long_eligibility.traces,
            atol=0,
            rtol=0,
        )
        assert set(actual_layer.read_bank) == set(expected_layer.read_bank)
        for name in actual_layer.read_bank:
            torch.testing.assert_close(
                actual_layer.read_bank[name], expected_layer.read_bank[name], atol=0, rtol=0
            )


def test_online_temporal_state_dict_roundtrips_every_exact_resume_field() -> None:
    torch.manual_seed(501)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _, state = train_step(
        model,
        torch.randn(2, 6, 4),
        torch.randn(2, 6, 3),
        optimizer,
        temporal_state=_initial(model),
        stream_ids=("left", "right"),
        temporal_chunk_size=2,
    )

    payload = online_temporal_state_dict(state)
    restored = load_online_temporal_state(
        model,
        payload,
        expected_stream_policy="carry_lanes",
        expected_stream_ids=("left", "right"),
    )

    _assert_state_equal(restored, state)
    assert restored.active_frames == ()
    assert payload["temporal_credit"] == "online"


def test_online_temporal_state_restore_refuses_mismatch_before_returning_state() -> None:
    model = _model()
    payload = online_temporal_state_dict(_initial(model))

    bad_schema = copy.deepcopy(payload)
    bad_schema["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        load_online_temporal_state(model, bad_schema)

    with pytest.raises(ValueError, match="stream policy"):
        load_online_temporal_state(
            model, payload, expected_stream_policy="reset_each_batch"
        )
    with pytest.raises(ValueError, match="stream lane identifiers"):
        load_online_temporal_state(
            model, payload, expected_stream_ids=("left", "different")
        )
    with pytest.raises(ValueError, match="model layer 0"):
        load_online_temporal_state(_model(state_dim=6), payload)


def test_online_temporal_state_cannot_serialize_pending_replay() -> None:
    model = _model()
    state = _initial(model)
    _, pending = model.forward_sequence_online(torch.randn(2, 2, 4), state)
    with pytest.raises(RuntimeError, match="pending online replay"):
        online_temporal_state_dict(pending)


def test_exact_resume_matches_uninterrupted_next_update() -> None:
    torch.manual_seed(502)
    uninterrupted = _model()
    resumed = _model()
    resumed.load_state_dict(uninterrupted.state_dict())
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=2e-3)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=2e-3)
    first_inputs = torch.randn(2, 5, 4)
    first_targets = torch.randn(2, 5, 3)

    _, state = train_step(
        uninterrupted,
        first_inputs,
        first_targets,
        uninterrupted_optimizer,
        temporal_state=_initial(uninterrupted),
        stream_ids=("left", "right"),
        temporal_chunk_size=2,
    )
    resumed.load_state_dict(uninterrupted.state_dict())
    resumed_optimizer.load_state_dict(copy.deepcopy(uninterrupted_optimizer.state_dict()))
    restored_state = load_online_temporal_state(
        resumed,
        online_temporal_state_dict(state),
        expected_stream_policy="carry_lanes",
        expected_stream_ids=("left", "right"),
    )

    next_inputs = torch.randn(2, 4, 4)
    next_targets = torch.randn(2, 4, 3)
    loss_a, state_a = train_step(
        uninterrupted,
        next_inputs,
        next_targets,
        uninterrupted_optimizer,
        temporal_state=state,
        stream_ids=("left", "right"),
        temporal_chunk_size=2,
    )
    loss_b, state_b = train_step(
        resumed,
        next_inputs,
        next_targets,
        resumed_optimizer,
        temporal_state=restored_state,
        stream_ids=("left", "right"),
        temporal_chunk_size=2,
    )

    assert loss_a == loss_b
    for lhs, rhs in zip(uninterrupted.parameters(), resumed.parameters()):
        torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
    _assert_state_equal(state_a, state_b)
    assert uninterrupted_optimizer.state_dict()["state"].keys() == resumed_optimizer.state_dict()[
        "state"
    ].keys()
    for key, actual in uninterrupted_optimizer.state_dict()["state"].items():
        expected = resumed_optimizer.state_dict()["state"][key]
        for name in actual:
            if isinstance(actual[name], torch.Tensor):
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
            else:
                assert actual[name] == expected[name]


def test_exact_resume_inside_accumulation_window_restores_gradients() -> None:
    torch.manual_seed(503)
    uninterrupted = _model()
    resumed = _model()
    resumed.load_state_dict(uninterrupted.state_dict())
    first_inputs = torch.randn(2, 4, 4)
    first_targets = torch.randn(2, 4, 3)
    second_inputs = torch.randn(2, 4, 4)
    second_targets = torch.randn(2, 4, 3)
    uninterrupted.zero_grad(set_to_none=True)
    _, state, _ = _online_backward_step(
        uninterrupted,
        first_inputs,
        first_targets,
        temporal_state=_initial(uninterrupted),
        temporal_chunk_size=2,
        stream_ids=("left", "right"),
        loss_scale=0.5,
        accumulation_position=1,
    )
    payload = online_training_state_dict(uninterrupted, state)
    restored_state = load_online_training_state(
        resumed,
        payload,
        expected_stream_policy="carry_lanes",
        expected_stream_ids=("left", "right"),
        expected_temporal_chunk_size=2,
        expected_accumulation_position=1,
    )
    for lhs, rhs in zip(uninterrupted.parameters(), resumed.parameters()):
        torch.testing.assert_close(lhs.grad, rhs.grad, atol=0, rtol=0)

    _, state_a, _ = _online_backward_step(
        uninterrupted,
        second_inputs,
        second_targets,
        temporal_state=state,
        temporal_chunk_size=2,
        stream_ids=("left", "right"),
        loss_scale=0.5,
        accumulation_position=0,
    )
    _, state_b, _ = _online_backward_step(
        resumed,
        second_inputs,
        second_targets,
        temporal_state=restored_state,
        temporal_chunk_size=2,
        stream_ids=("left", "right"),
        loss_scale=0.5,
        accumulation_position=0,
    )
    optimizer_a = torch.optim.SGD(uninterrupted.parameters(), lr=1e-3)
    optimizer_b = torch.optim.SGD(resumed.parameters(), lr=1e-3)
    apply_optimizer_step(uninterrupted, optimizer_a)
    apply_optimizer_step(resumed, optimizer_b)
    for lhs, rhs in zip(uninterrupted.parameters(), resumed.parameters()):
        torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)
    _assert_state_equal(state_a, state_b)


def test_exact_resume_refuses_chunk_or_accumulation_contract_mismatch() -> None:
    model = _model()
    model.zero_grad(set_to_none=True)
    _, state, _ = _online_backward_step(
        model,
        torch.randn(2, 4, 4),
        torch.randn(2, 4, 3),
        temporal_state=_initial(model),
        temporal_chunk_size=2,
        stream_ids=("left", "right"),
        loss_scale=0.5,
        accumulation_position=1,
    )
    payload = online_training_state_dict(model, state)
    with pytest.raises(ValueError, match="chunk contract"):
        load_online_training_state(
            _model(), payload, expected_temporal_chunk_size=3
        )
    with pytest.raises(ValueError, match="accumulation position"):
        load_online_training_state(
            _model(), payload, expected_accumulation_position=0
        )
