import torch

from dabsn import DABSNLayerSpec, DABSNModel
from dabsn.runtime import train_step
from dabsn.runtime.api import _online_backward_step
from dabsn.temporal_credit import (
    initial_online_temporal_state,
    prepare_online_temporal_state,
)


def _model() -> DABSNModel:
    return DABSNModel(
        input_dim=4,
        out_dim=3,
        layers=[DABSNLayerSpec(5, 4, "seq")],
        output_adapter="token",
        residual=True,
        mlp_ratio=2.0,
    )


def test_online_train_step_applies_update_and_returns_completed_state() -> None:
    torch.manual_seed(401)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 6, 4)
    targets = torch.randn(2, 6, 3)
    state = initial_online_temporal_state(
        model,
        2,
        stream_policy="reset_each_batch",
        stream_ids=(0, 1),
        device="cpu",
        dtype=torch.float32,
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    loss, next_state = train_step(
        model,
        inputs,
        targets,
        optimizer,
        temporal_state=state,
        clip_grad_norm=1.0,
    )

    assert isinstance(loss, float) and torch.isfinite(torch.tensor(loss))
    next_state.assert_checkpointable()
    assert next_state.chunk_position == 1
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_existing_bptt_train_step_return_contract_is_unchanged() -> None:
    torch.manual_seed(402)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = train_step(
        model,
        torch.randn(2, 3, 4),
        torch.randn(2, 3, 3),
        optimizer,
    )
    assert isinstance(loss, float)


def test_split_online_step_carries_seq_read_bank_without_historical_graphs() -> None:
    torch.manual_seed(403)
    full_model = _model()
    split_model = _model()
    split_model.load_state_dict(full_model.state_dict())
    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.0)
    split_optimizer = torch.optim.SGD(split_model.parameters(), lr=0.0)
    inputs = torch.randn(2, 7, 4)
    targets = torch.randn(2, 7, 3)

    def state(model):
        return initial_online_temporal_state(
            model,
            2,
            stream_policy="reset_each_batch",
            stream_ids=(0, 1),
            device="cpu",
            dtype=torch.float32,
        )

    full_loss, full_state = train_step(
        full_model, inputs, targets, full_optimizer, temporal_state=state(full_model)
    )
    split_loss, split_state = train_step(
        split_model,
        inputs,
        targets,
        split_optimizer,
        temporal_state=state(split_model),
        temporal_chunk_size=3,
    )

    assert abs(full_loss - split_loss) < 1e-6
    full_layer, split_layer = full_state.layers[0], split_state.layers[0]
    for actual, expected in zip(split_layer.core_recurrent, full_layer.core_recurrent):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(split_layer.long_recurrent, full_layer.long_recurrent):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(
        split_layer.core_eligibility.tensors(), full_layer.core_eligibility.tensors()
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        split_layer.long_eligibility.traces,
        full_layer.long_eligibility.traces,
        rtol=2e-5,
        atol=2e-6,
    )
    assert split_layer.read_bank["write"].shape == (2, 7, 4)
    assert all(value.grad_fn is None for value in split_layer.read_bank.values())


def test_carry_lane_identity_change_requires_reset_and_resets_only_that_lane() -> None:
    torch.manual_seed(404)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    initial = initial_online_temporal_state(
        model,
        2,
        stream_policy="carry_lanes",
        stream_ids=("a", "b"),
        device="cpu",
        dtype=torch.float32,
    )
    _, carried = train_step(
        model,
        torch.randn(2, 5, 4),
        torch.randn(2, 5, 3),
        optimizer,
        temporal_state=initial,
        stream_ids=("a", "b"),
        temporal_chunk_size=2,
    )

    try:
        prepare_online_temporal_state(
            model,
            carried,
            stream_ids=("a", "new"),
            stream_reset_mask=torch.tensor([False, False]),
        )
    except ValueError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("changed lane identity was accepted without reset")

    prepared = prepare_online_temporal_state(
        model,
        carried,
        stream_ids=("a", "new"),
        stream_reset_mask=torch.tensor([False, True]),
    )
    old_layer, new_layer = carried.layers[0], prepared.layers[0]
    for old, new in zip(old_layer.core_recurrent, new_layer.core_recurrent):
        torch.testing.assert_close(new[0], old[0])
    assert torch.count_nonzero(new_layer.core_recurrent[0][1]) == 0
    assert torch.count_nonzero(new_layer.core_eligibility.w_budget[1]) == 0
    assert torch.count_nonzero(new_layer.long_eligibility.traces[1]) == 0
    assert new_layer.read_bank["valid"][0].all()
    assert not new_layer.read_bank["valid"][1].any()
    assert prepared.stream_ids == ("a", "new")


def test_reset_each_batch_discards_previous_online_state() -> None:
    torch.manual_seed(405)
    model = _model()
    state = initial_online_temporal_state(
        model,
        2,
        stream_policy="reset_each_batch",
        stream_ids=(0, 1),
        device="cpu",
        dtype=torch.float32,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    _, carried = train_step(
        model,
        torch.randn(2, 4, 4),
        torch.randn(2, 4, 3),
        optimizer,
        temporal_state=state,
    )
    prepared = prepare_online_temporal_state(
        model,
        carried,
        stream_ids=(7, 8),
        stream_reset_mask=None,
    )
    fresh = initial_online_temporal_state(
        model,
        2,
        stream_policy="reset_each_batch",
        stream_ids=(7, 8),
        device="cpu",
        dtype=torch.float32,
    )
    assert not prepared.layers[0].read_bank
    for actual, expected in zip(
        prepared.layers[0].core_eligibility.tensors(),
        fresh.layers[0].core_eligibility.tensors(),
    ):
        torch.testing.assert_close(actual, expected)


def test_lane_grouped_online_step_matches_vectorized_batch() -> None:
    torch.manual_seed(406)
    vectorized = _model()
    grouped = _model()
    grouped.load_state_dict(vectorized.state_dict())
    vectorized_optimizer = torch.optim.SGD(vectorized.parameters(), lr=0.0)
    grouped_optimizer = torch.optim.SGD(grouped.parameters(), lr=0.0)
    inputs = torch.randn(4, 5, 4)
    targets = torch.randn(4, 5, 3)

    def state(model):
        return initial_online_temporal_state(
            model,
            4,
            stream_policy="carry_lanes",
            stream_ids=("a", "b", "c", "d"),
            device="cpu",
            dtype=torch.float32,
        )

    loss_a, state_a = train_step(
        vectorized,
        inputs,
        targets,
        vectorized_optimizer,
        temporal_state=state(vectorized),
        temporal_chunk_size=2,
        stream_ids=("a", "b", "c", "d"),
    )
    loss_b, state_b = train_step(
        grouped,
        inputs,
        targets,
        grouped_optimizer,
        temporal_state=state(grouped),
        temporal_chunk_size=2,
        stream_ids=("a", "b", "c", "d"),
        eligibility_batch_mode="lane_grouped",
        eligibility_lane_group_size=2,
    )

    # Lane grouping changes FP32 reduction order; one ULP near this loss is
    # already 1.19e-7. Keep a dtype-scale bound rather than a sub-ULP threshold.
    eps = torch.finfo(torch.float32).eps
    torch.testing.assert_close(torch.tensor(loss_a), torch.tensor(loss_b),
                               rtol=4 * eps, atol=eps)
    for lhs, rhs in zip(vectorized.parameters(), grouped.parameters()):
        torch.testing.assert_close(lhs.grad, rhs.grad, rtol=2e-5, atol=2e-6)
    for layer_a, layer_b in zip(state_a.layers, state_b.layers):
        for lhs, rhs in zip(
            layer_a.core_eligibility.tensors(), layer_b.core_eligibility.tensors()
        ):
            torch.testing.assert_close(lhs, rhs, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(
            layer_a.long_eligibility.traces,
            layer_b.long_eligibility.traces,
            rtol=2e-5,
            atol=2e-6,
        )


def test_online_microbatch_accumulation_matches_combined_effective_batch() -> None:
    torch.manual_seed(407)
    combined = _model()
    accumulated = _model()
    accumulated.load_state_dict(combined.state_dict())
    first_inputs = torch.randn(2, 4, 4)
    second_inputs = torch.randn(2, 4, 4)
    first_targets = torch.randn(2, 4, 3)
    second_targets = torch.randn(2, 4, 3)
    combined.zero_grad(set_to_none=True)
    accumulated.zero_grad(set_to_none=True)

    _online_backward_step(
        combined,
        torch.cat([first_inputs, second_inputs]),
        torch.cat([first_targets, second_targets]),
        temporal_state=initial_online_temporal_state(
            combined,
            4,
            stream_policy="reset_each_batch",
            stream_ids=(0, 1, 2, 3),
            device="cpu",
            dtype=torch.float32,
        ),
        temporal_chunk_size=2,
    )
    accumulation_state = initial_online_temporal_state(
        accumulated,
        2,
        stream_policy="reset_each_batch",
        stream_ids=(0, 1),
        device="cpu",
        dtype=torch.float32,
    )
    _, accumulation_state, _ = _online_backward_step(
        accumulated,
        first_inputs,
        first_targets,
        temporal_state=accumulation_state,
        temporal_chunk_size=2,
        loss_scale=0.5,
        accumulation_position=1,
    )
    _, accumulation_state, _ = _online_backward_step(
        accumulated,
        second_inputs,
        second_targets,
        temporal_state=accumulation_state,
        temporal_chunk_size=2,
        loss_scale=0.5,
        accumulation_position=0,
    )

    assert accumulation_state.accumulation_position == 0
    for lhs, rhs in zip(combined.parameters(), accumulated.parameters()):
        torch.testing.assert_close(lhs.grad, rhs.grad, rtol=2e-5, atol=2e-6)
