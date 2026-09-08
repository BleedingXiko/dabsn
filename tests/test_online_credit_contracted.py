import dataclasses

import pytest
import torch

from dabsn.core import DABSNCore
import dabsn.temporal_credit as credit


@pytest.mark.parametrize("batch,steps,inputs,state", [(1, 1, 3, 5), (2, 5, 7, 3), (3, 17, 4, 9)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chunk_contraction_preserves_all_credit(batch, steps, inputs, state, dtype):
    torch.manual_seed(1920)
    core = DABSNCore(inputs, state).to(dtype)
    x = torch.randn(batch, steps, inputs, dtype=dtype) * .3
    eligibility = credit.initial_core_eligibility(core, batch, device="cpu", dtype=dtype)
    for tensor in eligibility.tensors():
        tensor.normal_(0, .04)
    old_tensors = tuple(t.clone() for t in eligibility.tensors())
    recurrent = core.initial_state(batch, device="cpu", dtype=dtype)
    _, tape = credit.core_online_forward_reference(core, x, recurrent, eligibility)
    signals = credit.CoreSignalGrads(**{
        field.name: torch.randn(batch, steps, state, dtype=dtype)
        for field in dataclasses.fields(credit.CoreSignalGrads)
    })
    expected = credit.core_online_replay_analytic(core, tape, signals, eligibility)
    actual = credit.core_online_replay_contracted(core, tape, signals, eligibility)
    tolerance = dict(rtol=3e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-12)
    for left, right in zip(actual[0].tensors(), expected[0].tensors()):
        torch.testing.assert_close(left, right, **tolerance)
    for parameter in core.parameters():
        torch.testing.assert_close(actual[1][parameter], expected[1][parameter], **tolerance)
    torch.testing.assert_close(actual[2], expected[2], **tolerance)
    for left, right in zip(eligibility.tensors(), old_tensors):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_matrix_contraction_handles_zero_and_negative_transition_without_division():
    torch.manual_seed(92)
    b, t, s, i = 2, 6, 3, 5
    j = torch.randn(b, t, s, dtype=torch.float64)
    j[:, 2] = 0
    incoming, coefficient = torch.randn(2, b, t, s, dtype=torch.float64)
    x = torch.randn(b, t, i, dtype=torch.float64)
    initial = torch.randn(b, s, i, dtype=torch.float64)
    expected_state = initial.clone()
    expected_grad = torch.zeros(s, i, dtype=torch.float64)
    for step in range(t):
        expected_grad += torch.einsum("bs,bsi->si", incoming[:, step], expected_state)
        expected_state = j[:, step, :, None] * expected_state + coefficient[:, step, :, None] * x[:, step, None, :]
    weights = credit._budget_trace_weights(j, incoming)
    actual_state, actual_grad = credit._contract_budget_trace(initial, coefficient, x, weights)
    torch.testing.assert_close(actual_state, expected_state, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("native", [False, True])
def test_replay_keeps_fp32_arithmetic_inside_autocast(native):
    from dabsn.kernels.cpu import native_core_online_replay

    torch.manual_seed(81)
    core = DABSNCore(5, 7)
    eligibility = credit.initial_core_eligibility(core, 2, device="cpu")
    for tensor in eligibility.tensors():
        tensor.normal_(std=.1)
    x = torch.randn(2, 9, 5)
    _, tape = credit.core_online_forward_reference(core, x, core.initial_state(2, device="cpu"), eligibility)
    signals = credit.CoreSignalGrads(**{name: torch.randn(2, 9, 7) for name in credit.CoreSignalGrads.__dataclass_fields__})
    oracle = credit.core_online_replay_analytic(core, tape, signals, eligibility)
    replay = native_core_online_replay if native else credit.core_online_replay_contracted
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = replay(core, tape, signals, eligibility)
    for left, right in zip(actual[0].tensors(), oracle[0].tensors()):
        torch.testing.assert_close(left, right, rtol=5e-5, atol=5e-6)
    for parameter in core.parameters():
        torch.testing.assert_close(actual[1][parameter], oracle[1][parameter], rtol=5e-5, atol=5e-6)
