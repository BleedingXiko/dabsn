from __future__ import annotations

import pytest
import torch

from dabsn import DABSNCore
from dabsn.read import DABSNRead
from dabsn.temporal_credit import (
    CORE_GLOBAL_PARAMETER_NAMES,
    CORE_LOCAL_PARAMETER_NAMES,
    CoreEligibilityState,
    LongReadEligibilityState,
    OnlineCoreFrame,
    OnlineLayerState,
    OnlineTemporalState,
    initial_core_eligibility,
    initial_long_read_eligibility,
)


def test_initial_core_eligibility_supports_unequal_input_and_state_widths() -> None:
    core = DABSNCore(input_dim=3, hidden_dim=5).to(dtype=torch.float64)

    state = initial_core_eligibility(core, batch_size=2, device="cpu")

    assert state.w_budget.shape == (2, 5, 3)
    assert state.a_budget.shape == (2, 5, 5)
    assert state.wg_budget.shape == (2, 5, 3)
    assert state.ug_budget.shape == (2, 5, 5)
    assert state.local.shape == (2, 5, 3, len(CORE_LOCAL_PARAMETER_NAMES))
    assert state.global_.shape == (2, 5, 3, len(CORE_GLOBAL_PARAMETER_NAMES))
    assert state.dtype == torch.float32
    assert state.device == torch.device("cpu")
    assert state.schema_version == 1
    assert state.batch_size == 2
    assert state.input_dim == 3
    assert state.state_dim == 5
    assert all(tensor.is_contiguous() for tensor in state.tensors())
    assert all(torch.count_nonzero(tensor) == 0 for tensor in state.tensors())


def test_core_eligibility_byte_accounting_includes_the_batch_factor() -> None:
    core = DABSNCore(input_dim=3, hidden_dim=5)

    one = initial_core_eligibility(core, batch_size=1, device="cpu")
    four = initial_core_eligibility(core, batch_size=4, device="cpu")

    expected_elements = 2 * 5 * 3 + 2 * 5 * 5 + 5 * 3 * 16 + 5 * 3 * 4
    assert one.nbytes() == expected_elements * torch.tensor([], dtype=torch.float32).element_size()
    assert four.nbytes() == 4 * one.nbytes()


def test_core_eligibility_rejects_a_noncontiguous_or_wrong_shape_tensor() -> None:
    core = DABSNCore(input_dim=3, hidden_dim=5)
    valid = initial_core_eligibility(core, batch_size=2, device="cpu")

    with pytest.raises(ValueError, match="w_budget.*shape"):
        CoreEligibilityState(
            w_budget=torch.zeros(2, 5, 4),
            a_budget=valid.a_budget,
            wg_budget=valid.wg_budget,
            ug_budget=valid.ug_budget,
            local=valid.local,
            global_=valid.global_,
        )

    with pytest.raises(ValueError, match="contiguous"):
        noncontiguous_local = valid.local.transpose(-1, -2).contiguous().transpose(-1, -2)
        CoreEligibilityState(
            w_budget=valid.w_budget,
            a_budget=valid.a_budget,
            wg_budget=valid.wg_budget,
            ug_budget=valid.ug_budget,
            local=noncontiguous_local,
            global_=valid.global_,
        )


def test_core_eligibility_rejects_mixed_dtype_or_device() -> None:
    core = DABSNCore(input_dim=3, hidden_dim=5)
    valid = initial_core_eligibility(core, batch_size=2, device="cpu")

    with pytest.raises(ValueError, match="same dtype and device"):
        CoreEligibilityState(
            w_budget=valid.w_budget,
            a_budget=valid.a_budget,
            wg_budget=valid.wg_budget,
            ug_budget=valid.ug_budget,
            local=valid.local.to(torch.float64),
            global_=valid.global_,
        )


@pytest.mark.parametrize("batch_size", [0, -1])
def test_initial_core_eligibility_rejects_nonpositive_batch(batch_size: int) -> None:
    core = DABSNCore(input_dim=3, hidden_dim=5)

    with pytest.raises(ValueError, match="batch_size"):
        initial_core_eligibility(core, batch_size=batch_size, device="cpu")


def test_long_read_eligibility_is_versioned_and_batch_factored() -> None:
    read = DABSNRead(hidden_dim=5, read_geometry="seq")
    one = initial_long_read_eligibility(read, 1, device="cpu")
    four = initial_long_read_eligibility(read, 4, device="cpu")

    assert isinstance(one, LongReadEligibilityState)
    assert one.traces.shape == (1, 5, 3, 4)
    assert one.schema_version == 1
    assert one.nbytes() == 5 * 3 * 4 * 4
    assert four.nbytes() == 4 * one.nbytes()
    assert torch.count_nonzero(one.traces) == 0


def test_online_layer_and_temporal_state_validate_lane_contracts() -> None:
    core = DABSNCore(input_dim=3, hidden_dim=5)
    read = DABSNRead(hidden_dim=5, read_geometry="seq")
    core_state = tuple(value.detach() for value in core.initial_state(2))
    long_state = tuple(value.detach() for value in read.initial_long_scan_state(2))
    layer = OnlineLayerState(
        core_recurrent=core_state,
        core_eligibility=initial_core_eligibility(core, 2, device="cpu"),
        long_recurrent=long_state,
        long_eligibility=initial_long_read_eligibility(read, 2, device="cpu"),
    )
    state = OnlineTemporalState(
        layers=(layer,),
        stream_policy="carry_lanes",
        stream_ids=("left", "right"),
        pending_reset_mask=torch.tensor([False, True]),
        chunk_position=7,
        accumulation_position=2,
    )

    assert state.batch_size == 2
    assert state.nbytes() == layer.nbytes() + 2
    state.assert_checkpointable()
    with pytest.raises(ValueError, match="stream_ids"):
        OnlineTemporalState(
            layers=(layer,),
            stream_policy="carry_lanes",
            stream_ids=("only-one",),
            pending_reset_mask=torch.zeros(2, dtype=torch.bool),
        )


def test_online_temporal_state_rejects_checkpoint_during_pending_replay() -> None:
    core = DABSNCore(input_dim=3, hidden_dim=5)
    read = DABSNRead(hidden_dim=5, read_geometry="seq")
    eligibility = initial_core_eligibility(core, 1, device="cpu")
    frame = OnlineCoreFrame(eligibility)
    frame.begin(eligibility)
    layer = OnlineLayerState(
        core_recurrent=tuple(value.detach() for value in core.initial_state(1)),
        core_eligibility=eligibility,
        long_recurrent=tuple(value.detach() for value in read.initial_long_scan_state(1)),
        long_eligibility=initial_long_read_eligibility(read, 1, device="cpu"),
    )
    state = OnlineTemporalState(
        layers=(layer,),
        stream_policy="reset_each_batch",
        stream_ids=(0,),
        pending_reset_mask=torch.zeros(1, dtype=torch.bool),
        active_frames=(frame,),
    )

    with pytest.raises(RuntimeError, match="pending online replay"):
        state.assert_checkpointable()
