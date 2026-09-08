from dataclasses import replace

import pytest
import torch

from dabsn import DABSNCore, DABSNSequenceLM, initial_online_temporal_state
from dabsn.temporal_credit import (
    CoreSignalGrads,
    core_online_forward_reference,
    core_online_replay_reference,
    initial_core_eligibility,
    initial_long_read_eligibility,
    load_online_temporal_state,
    online_temporal_state_dict,
    prepare_online_temporal_state,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("override", [None, torch.float64, torch.bfloat16])
def test_initial_temporal_precision_is_explicit(dtype, override):
    model = DABSNSequenceLM(vocab=17, hidden_dim=4, state_dim=3, depth=2).to(dtype)
    state = initial_online_temporal_state(
        model, 2, device="cpu", dtype=dtype, eligibility_dtype=override,
        stream_policy="carry_lanes", stream_ids=(0, 1),
    )
    expected = (torch.float64 if dtype == torch.float64 else torch.float32) if override is None else override
    for layer in state.layers:
        assert all(value.dtype == dtype for value in (*layer.core_recurrent, *layer.long_recurrent))
        assert layer.core_eligibility.dtype == expected
        assert layer.long_eligibility.traces.dtype == expected


@pytest.mark.parametrize("policy", ["reset_each_batch", "carry_lanes"])
def test_restore_and_reset_preserve_each_layer_eligibility_precision(policy):
    model = DABSNSequenceLM(vocab=17, hidden_dim=4, state_dim=3, depth=2)
    state = initial_online_temporal_state(
        model, 2, device="cpu", dtype=torch.float32,
        stream_policy=policy, stream_ids=(0, 1), temporal_chunk_size=2,
    )
    layers = []
    for index, (layer, block) in enumerate(zip(state.layers, model.backbone.blocks)):
        core_dtype = (torch.float64, torch.bfloat16)[index]
        long_dtype = (torch.bfloat16, torch.float64)[index]
        core = initial_core_eligibility(block.core, 2, device="cpu", dtype=core_dtype)
        long = initial_long_read_eligibility(block.read, 2, device="cpu", dtype=long_dtype)
        for value in (*core.tensors(), long.traces):
            value.fill_(1)
        layers.append(replace(layer, core_eligibility=core, long_eligibility=long))
    state = replace(state, layers=tuple(layers), chunk_position=3)
    restored = load_online_temporal_state(model, online_temporal_state_dict(state))
    reset = prepare_online_temporal_state(
        model, restored, stream_ids=(2, 1), stream_reset_mask=torch.tensor([True, False]),
    )
    for original, loaded, fresh in zip(state.layers, restored.layers, reset.layers):
        for expected, actual, result in zip(
            (*original.core_eligibility.tensors(), original.long_eligibility.traces),
            (*loaded.core_eligibility.tensors(), loaded.long_eligibility.traces),
            (*fresh.core_eligibility.tensors(), fresh.long_eligibility.traces),
        ):
            torch.testing.assert_close(actual, expected)
            assert result.dtype == expected.dtype
            assert torch.count_nonzero(result[0]) == 0
            torch.testing.assert_close(result[1], torch.zeros_like(result[1]) if policy == "reset_each_batch" else expected[1])
    assert reset.chunk_position == (0 if policy == "reset_each_batch" else 3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("required", [False, True])
def test_native_cpu_replay_never_converts_eligibility_precision(monkeypatch, dtype, required):
    from dabsn.kernels import cpu

    torch.manual_seed(731)
    core = DABSNCore(3, 2)
    inputs = torch.randn(1, 2, 3) * 0.1
    eligibility = initial_core_eligibility(core, 1, device="cpu", dtype=dtype)
    _, tape = core_online_forward_reference(core, inputs, core.initial_state(1), eligibility)
    signals = CoreSignalGrads(y=torch.ones(1, 2, 2))

    def unsupported_extension(**kwargs):
        raise AssertionError("native extension must not load for unsupported eligibility dtype")

    monkeypatch.setattr(cpu, "_load_ext", unsupported_extension)
    if required:
        with pytest.raises(RuntimeError, match="float32.*eligibility"):
            cpu.native_core_online_replay(core, tape, signals, eligibility, required=True)
    else:
        expected = core_online_replay_reference(core, tape, signals, eligibility)
        actual = cpu.native_core_online_replay(core, tape, signals, eligibility)
        for result, reference in zip(actual[0].tensors(), expected[0].tensors()):
            torch.testing.assert_close(result, reference)
            assert result.dtype == dtype
        for parameter in core.parameters():
            torch.testing.assert_close(actual[1][parameter], expected[1][parameter])
        torch.testing.assert_close(actual[2], expected[2])
