"""Capacity contracts shared by causal and noncausal Read geometries."""
import copy

import pytest
import torch

from dabsn import build_dabsn, initial_online_temporal_state, finalize_online_temporal_state
from dabsn.kernels.capacity import enable_capacity


def dense(gradient):
    return gradient.to_dense() if gradient.is_sparse else gradient


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("geometry", ["seq", "field", "hybrid"])
@pytest.mark.parametrize("online", [False, True])
def test_learned_native_whole_model_all_geometries(mode, geometry, online):
    torch.manual_seed(192)
    reference = build_dabsn(3, 2, 5, state_dim=4, depth=2, read_geometry=geometry,
        capacity={"mode": mode, "size": 16, "active_units": 4}).double()
    native = copy.deepcopy(reference)
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    target = torch.randn(2, 4, 2, dtype=torch.float64)
    try:
        for _ in range(2):
            results = []
            for model, backend in ((reference, "reference"), (native, "cpu")):
                enable_capacity(backend, required=True)
                model.zero_grad(set_to_none=True)
                value = x.clone().requires_grad_()
                if online:
                    state = initial_online_temporal_state(model, 2, device="cpu", dtype=torch.float64,
                        stream_policy="reset_each_batch", stream_ids=("a", "b"))
                    output, state = model.forward_sequence_online(value, state)
                else:
                    output = model.forward_sequence(value)
                (output - target).square().mean().backward()
                if online:
                    state = finalize_online_temporal_state(state)
                    state.assert_checkpointable()
                results.append((output.detach(), value.grad, {
                    name: dense(p.grad).clone() for name, p in model.named_parameters() if p.grad is not None
                }))
                torch.optim.SGD(model.parameters(), lr=.1).step()
            for a, b in zip(results[0][:2], results[1][:2]):
                torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-7)
            assert results[0][2].keys() == results[1][2].keys()
            for name, grad in results[0][2].items():
                torch.testing.assert_close(grad, results[1][2][name], atol=2e-8, rtol=2e-7)
        assert all(block.capacity_module.last_backend == "cpu_native_cpp" for block in native.backbone.blocks)
    finally:
        enable_capacity("reference", required=True)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_capacity_lane_slice_merge_reset_and_checkpoint(mode):
    from dabsn.temporal_credit import (_slice_online_temporal_state, _merge_online_temporal_states,
        prepare_online_temporal_state, online_temporal_state_dict, load_online_temporal_state)
    model = build_dabsn(3, 2, 5, state_dim=4,
        capacity={"mode": mode, "size": 16, "active_units": 4})
    state = initial_online_temporal_state(model, 3, device="cpu", dtype=torch.float32,
        stream_policy="carry_lanes", stream_ids=("a", "b", "c"))
    output, state = model.forward_sequence_online(torch.randn(3, 4, 3), state)
    output.square().mean().backward()
    state = finalize_online_temporal_state(state)
    left = _slice_online_temporal_state(state, 0, 1)
    right = _slice_online_temporal_state(state, 1, 3)
    merged = _merge_online_temporal_states((left, right))
    restored = load_online_temporal_state(model, online_temporal_state_dict(merged))
    for original, actual in zip(state.layers, restored.layers):
        assert original.capacity_contract == actual.capacity_contract
        for name in original.capacity_eligibility:
            torch.testing.assert_close(original.capacity_eligibility[name], actual.capacity_eligibility[name])
        torch.testing.assert_close(original.capacity_recurrent["features"], actual.capacity_recurrent["features"])
    reset = prepare_online_temporal_state(model, restored, stream_ids=("a", "new", "c"),
        stream_reset_mask=torch.tensor([False, True, False]))
    for original, actual in zip(restored.layers, reset.layers):
        assert torch.count_nonzero(actual.capacity_recurrent["features"][1]) == 0
        torch.testing.assert_close(original.capacity_recurrent["features"][[0, 2]],
                                   actual.capacity_recurrent["features"][[0, 2]])
        for trace in actual.capacity_eligibility.values():
            assert torch.count_nonzero(trace[1]) == 0


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("backend", ["reference", "cpu"])
def test_online_capacity_delayed_loss_chunk_partition(mode, backend):
    from dabsn.capacity import DABSNCapacitySubsystem
    from dabsn import CapacityConfig
    torch.manual_seed(614)
    enable_capacity(backend, required=True)
    whole = DABSNCapacitySubsystem(3, 2, CapacityConfig(mode=mode, size=8, active_units=2)).double()
    x = torch.randn(2, 7, 3, dtype=torch.float64)
    whole.forward_sequence(x).sum().backward()
    torch.optim.SGD(whole.parameters(), lr=.05).step()
    whole.zero_grad(set_to_none=True)
    split = copy.deepcopy(whole)
    results = []
    try:
        for module, lengths in ((whole, (7,)), (split, (2, 1, 4))):
            recurrent, eligibility = module.initial_online_state(2, device="cpu",
                recurrent_dtype=torch.float64, eligibility_dtype=torch.float64)
            outputs = []
            offset = 0
            for length in lengths:
                output, recurrent, eligibility = module.forward_online(
                    x[:, offset:offset+length], recurrent, eligibility)
                offset += length
                # Earlier chunks have no loss contribution, but replay must still
                # advance eligibility for the final delayed supervision.
                loss = output[:, -1].square().sum() if offset == 7 else output.sum() * 0
                loss.backward()
                outputs.append(output.detach())
            results.append((torch.cat(outputs, 1), recurrent, eligibility))
        torch.testing.assert_close(results[0][0], results[1][0])
        for left, right in zip(whole.parameters(), split.parameters()):
            torch.testing.assert_close(dense(left.grad), dense(right.grad), atol=2e-10, rtol=2e-9)
        for index in (1, 2):
            for name in results[0][index]:
                torch.testing.assert_close(results[0][index][name], results[1][index][name])
    finally:
        enable_capacity("reference", required=True)
