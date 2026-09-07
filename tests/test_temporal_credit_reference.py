from __future__ import annotations

import pytest
import torch

from dabsn import DABSNCore
from dabsn.temporal_credit import (
    CoreSignalGrads,
    OnlineCoreFrame,
    core_online_forward_reference,
    core_online_replay_analytic,
    core_online_replay_reference,
    initial_core_eligibility,
)


def _assert_core_outputs_equal(actual, expected) -> None:
    torch.testing.assert_close(actual.trajectory, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual.novelty, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual.plasticity, expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual.expression, expected[3], rtol=0, atol=0)
    torch.testing.assert_close(actual.write, expected[4], rtol=0, atol=0)
    torch.testing.assert_close(actual.energy, expected[5], rtol=0, atol=0)
    torch.testing.assert_close(actual.saturation, expected[6], rtol=0, atol=0)


def test_online_reference_forward_matches_canonical_core_with_unequal_widths() -> None:
    torch.manual_seed(7)
    core = DABSNCore(input_dim=3, hidden_dim=5).to(dtype=torch.float64)
    inputs = torch.randn(2, 4, 3, dtype=torch.float64)
    initial_state = (
        torch.randn(2, 5, dtype=torch.float64) * 0.1,
        torch.sigmoid(torch.randn(2, 5, dtype=torch.float64)),
        torch.sigmoid(torch.randn(2, 5, dtype=torch.float64)),
    )
    eligibility = initial_core_eligibility(
        core,
        batch_size=2,
        device="cpu",
        dtype=torch.float64,
    )

    expected, expected_final = core._reference_forward_from_state(
        inputs,
        initial_state=initial_state,
        return_writes=True,
        return_cocktail=True,
        return_final_state=True,
    )
    actual, tape = core_online_forward_reference(
        core,
        inputs,
        initial_state,
        eligibility,
    )

    _assert_core_outputs_equal(actual, expected)
    for actual_state, expected_state in zip(actual.final_state, expected_final):
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    assert tape.batch_size == 2
    assert tape.steps == 4
    assert tape.input_dim == 3
    assert tape.state_dim == 5


def test_online_reference_forward_does_not_advance_or_mutate_eligibility() -> None:
    core = DABSNCore(input_dim=3, hidden_dim=5)
    inputs = torch.randn(2, 3, 3)
    recurrent = core.initial_state(2)
    eligibility = initial_core_eligibility(core, 2, device="cpu")
    before = tuple(tensor.clone() for tensor in eligibility.tensors())

    _, tape = core_online_forward_reference(core, inputs, recurrent, eligibility)

    for actual, expected in zip(eligibility.tensors(), before):
        assert torch.equal(actual, expected)
    assert all(tensor.grad_fn is None for tensor in tape.tensors())


def test_y_only_online_replay_matches_exact_single_step_gradients() -> None:
    torch.manual_seed(11)
    core = DABSNCore(input_dim=3, hidden_dim=5).to(dtype=torch.float64)
    inputs = torch.randn(2, 1, 3, dtype=torch.float64, requires_grad=True)
    recurrent = core.initial_state(2, dtype=torch.float64)
    signal = torch.randn(2, 1, 5, dtype=torch.float64)
    parameters = tuple(core.parameters())

    expected_outputs = core._reference_forward_from_state(
        inputs,
        initial_state=recurrent,
        return_writes=True,
        return_cocktail=True,
    )
    expected_loss = (expected_outputs[0][..., :5] * signal).sum()
    expected = torch.autograd.grad(
        expected_loss,
        (*parameters, inputs),
        allow_unused=True,
    )

    eligibility = initial_core_eligibility(
        core,
        batch_size=2,
        device="cpu",
        dtype=torch.float64,
    )
    _, tape = core_online_forward_reference(core, inputs.detach(), recurrent, eligibility)
    next_eligibility, actual_parameters, actual_input = core_online_replay_reference(
        core,
        tape,
        CoreSignalGrads(y=signal),
        eligibility,
    )

    for parameter, expected_gradient in zip(parameters, expected[:-1]):
        actual_gradient = actual_parameters.get(parameter)
        if expected_gradient is None:
            assert actual_gradient is None or torch.count_nonzero(actual_gradient) == 0
        else:
            assert actual_gradient is not None
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_input, expected[-1], rtol=1e-12, atol=1e-12)
    assert any(torch.count_nonzero(tensor) > 0 for tensor in next_eligibility.tensors())


def test_y_only_online_rule_is_invariant_to_chunk_splits() -> None:
    torch.manual_seed(17)
    core = DABSNCore(input_dim=3, hidden_dim=4).to(dtype=torch.float64)
    inputs = torch.randn(2, 6, 3, dtype=torch.float64)
    signal = torch.randn(2, 6, 4, dtype=torch.float64)
    initial_recurrent = core.initial_state(2, dtype=torch.float64)
    initial_eligibility = initial_core_eligibility(
        core,
        batch_size=2,
        device="cpu",
        dtype=torch.float64,
    )

    full_forward, full_tape = core_online_forward_reference(
        core,
        inputs,
        initial_recurrent,
        initial_eligibility,
    )
    full_eligibility, full_gradients, full_input_gradient = core_online_replay_reference(
        core,
        full_tape,
        CoreSignalGrads(y=signal),
        initial_eligibility,
    )

    recurrent = initial_recurrent
    eligibility = initial_eligibility
    split_gradients = {parameter: torch.zeros_like(parameter) for parameter in core.parameters()}
    split_input_gradients = []
    start = 0
    for width in (2, 1, 3):
        stop = start + width
        chunk_forward, chunk_tape = core_online_forward_reference(
            core,
            inputs[:, start:stop],
            recurrent,
            eligibility,
        )
        eligibility, gradients, input_gradient = core_online_replay_reference(
            core,
            chunk_tape,
            CoreSignalGrads(y=signal[:, start:stop]),
            eligibility,
        )
        for parameter, gradient in gradients.items():
            split_gradients[parameter].add_(gradient)
        split_input_gradients.append(input_gradient)
        recurrent = chunk_forward.final_state
        start = stop

    for actual, expected in zip(eligibility.tensors(), full_eligibility.tensors()):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for parameter, expected in full_gradients.items():
        torch.testing.assert_close(
            split_gradients[parameter],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
    torch.testing.assert_close(
        torch.cat(split_input_gradients, dim=1),
        full_input_gradient,
        rtol=1e-12,
        atol=1e-12,
    )
    for actual, expected in zip(recurrent, full_forward.final_state):
        # Splitting the projection GEMM changes its reduction grouping on BLAS
        # implementations; require FP64 roundoff agreement, not bit identity.
        eps = torch.finfo(torch.float64).eps
        torch.testing.assert_close(actual, expected, rtol=8 * eps, atol=8 * eps)


def _random_signal_grads(result) -> CoreSignalGrads:
    state = result.novelty.shape[-1]
    return CoreSignalGrads(
        y=torch.randn_like(result.trajectory[..., :state]),
        next_budget=torch.randn_like(result.trajectory[..., state:]),
        novelty=torch.randn_like(result.novelty),
        plasticity=torch.randn_like(result.plasticity),
        expression=torch.randn_like(result.expression),
        write=torch.randn_like(result.write),
        energy=torch.randn_like(result.energy),
        saturation=torch.randn_like(result.saturation),
    )


def _signal_dot(result, grads: CoreSignalGrads) -> torch.Tensor:
    state = result.novelty.shape[-1]
    return sum(
        (
            (result.trajectory[..., :state] * grads.y).sum(),
            (result.trajectory[..., state:] * grads.next_budget).sum(),
            (result.novelty * grads.novelty).sum(),
            (result.plasticity * grads.plasticity).sum(),
            (result.expression * grads.expression).sum(),
            (result.write * grads.write).sum(),
            (result.energy * grads.energy).sum(),
            (result.saturation * grads.saturation).sum(),
        ),
        torch.zeros((), dtype=result.trajectory.dtype),
    )


def test_analytic_replay_matches_autograd_oracle_for_full_signals() -> None:
    torch.manual_seed(211)
    core = DABSNCore(input_dim=3, hidden_dim=5).to(dtype=torch.float64)
    inputs = torch.randn(2, 4, 3, dtype=torch.float64)
    recurrent = core.initial_state(2, dtype=torch.float64)
    eligibility = initial_core_eligibility(core, 2, device="cpu", dtype=torch.float64)
    result, tape = core_online_forward_reference(core, inputs, recurrent, eligibility)
    signals = _random_signal_grads(result)

    expected_state, expected_gradients, expected_input = core_online_replay_reference(
        core, tape, signals, eligibility
    )
    actual_state, actual_gradients, actual_input = core_online_replay_analytic(
        core, tape, signals, eligibility
    )

    for actual, expected in zip(actual_state.tensors(), expected_state.tensors()):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
    for parameter in core.parameters():
        torch.testing.assert_close(
            actual_gradients[parameter], expected_gradients[parameter], rtol=2e-11, atol=2e-11
        )
    torch.testing.assert_close(actual_input, expected_input, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_analytic_local_vjp_accepts_low_precision_weights_with_fp32_tape(dtype):
    import copy
    from dabsn.temporal_credit import _core_step_local_vjp_analytic, _core_step_local_vjp

    torch.manual_seed(213)
    core = DABSNCore(input_dim=3, hidden_dim=5).float()
    inputs = torch.randn(2, 2, 3)
    eligibility = initial_core_eligibility(core, 2, device="cpu")
    result, tape = core_online_forward_reference(core, inputs, core.initial_state(2), eligibility)
    signals = _random_signal_grads(result)
    # CUDA retains an FP32 tape even when model storage is FP16/BF16.
    core.to(dtype)
    incoming, gradients, input_gradient = _core_step_local_vjp_analytic(core, tape, 0, signals)
    assert input_gradient.dtype == torch.float32
    assert all(bool(torch.isfinite(value).all()) for value in (*incoming, *gradients.values(), input_gradient))
    oracle = copy.deepcopy(core).float()
    expected_incoming, expected_gradients, expected_input = _core_step_local_vjp(oracle, tape, 0, signals)
    for actual, expected in zip(incoming, expected_incoming):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    for parameter, expected_parameter in zip(core.parameters(), oracle.parameters()):
        if parameter in gradients:
            torch.testing.assert_close(gradients[parameter], expected_gradients[expected_parameter], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(input_gradient, expected_input, rtol=2e-6, atol=2e-6)


def test_full_signal_replay_matches_exact_single_step_gradients() -> None:
    torch.manual_seed(23)
    core = DABSNCore(input_dim=3, hidden_dim=5).to(dtype=torch.float64)
    inputs = torch.randn(2, 1, 3, dtype=torch.float64, requires_grad=True)
    recurrent = tuple(
        value.detach().requires_grad_(True)
        for value in core.initial_state(2, dtype=torch.float64)
    )
    parameters = tuple(core.parameters())
    expected = core._reference_forward_from_state(
        inputs,
        initial_state=recurrent,
        return_writes=True,
        return_cocktail=True,
        return_final_state=True,
    )
    expected_result = type("Expected", (), {})()
    (
        expected_result.trajectory,
        expected_result.novelty,
        expected_result.plasticity,
        expected_result.expression,
        expected_result.write,
        expected_result.energy,
        expected_result.saturation,
    ) = expected[0]
    signal_grads = _random_signal_grads(expected_result)
    expected_gradients = torch.autograd.grad(
        _signal_dot(expected_result, signal_grads),
        (*parameters, inputs, *recurrent),
        allow_unused=True,
    )

    eligibility = initial_core_eligibility(
        core,
        batch_size=2,
        device="cpu",
        dtype=torch.float64,
    )
    _, tape = core_online_forward_reference(
        core,
        inputs.detach(),
        tuple(value.detach() for value in recurrent),
        eligibility,
    )
    _, actual_parameters, actual_input, actual_incoming_state = core_online_replay_reference(
        core,
        tape,
        signal_grads,
        eligibility,
        return_incoming_state_grad=True,
    )

    for parameter, expected_gradient in zip(parameters, expected_gradients[: len(parameters)]):
        actual_gradient = actual_parameters.get(parameter)
        if expected_gradient is None:
            assert actual_gradient is None or torch.count_nonzero(actual_gradient) == 0
        else:
            assert actual_gradient is not None
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        actual_input,
        expected_gradients[len(parameters)],
        rtol=1e-11,
        atol=1e-11,
    )
    for actual, expected_gradient in zip(
        actual_incoming_state,
        expected_gradients[len(parameters) + 1 :],
    ):
        torch.testing.assert_close(actual, expected_gradient, rtol=1e-11, atol=1e-11)


def test_full_signal_online_rule_is_invariant_to_chunk_splits() -> None:
    torch.manual_seed(29)
    core = DABSNCore(input_dim=2, hidden_dim=3).to(dtype=torch.float64)
    inputs = torch.randn(2, 5, 2, dtype=torch.float64)
    initial_recurrent = core.initial_state(2, dtype=torch.float64)
    initial_eligibility = initial_core_eligibility(
        core,
        batch_size=2,
        device="cpu",
        dtype=torch.float64,
    )
    full_forward, full_tape = core_online_forward_reference(
        core,
        inputs,
        initial_recurrent,
        initial_eligibility,
    )
    full_signals = _random_signal_grads(full_forward)
    full_eligibility, full_gradients, full_input_gradient = core_online_replay_reference(
        core,
        full_tape,
        full_signals,
        initial_eligibility,
    )

    recurrent = initial_recurrent
    eligibility = initial_eligibility
    split_gradients = {parameter: torch.zeros_like(parameter) for parameter in core.parameters()}
    split_input_gradients = []
    start = 0
    for width in (1, 3, 1):
        stop = start + width
        chunk_forward, chunk_tape = core_online_forward_reference(
            core,
            inputs[:, start:stop],
            recurrent,
            eligibility,
        )
        chunk_signals = CoreSignalGrads(
            **{
                name: getattr(full_signals, name)[:, start:stop]
                for name in CoreSignalGrads.__dataclass_fields__
            }
        )
        eligibility, gradients, input_gradient = core_online_replay_reference(
            core,
            chunk_tape,
            chunk_signals,
            eligibility,
        )
        for parameter, gradient in gradients.items():
            split_gradients[parameter].add_(gradient)
        split_input_gradients.append(input_gradient)
        recurrent = chunk_forward.final_state
        start = stop

    for actual, expected in zip(eligibility.tensors(), full_eligibility.tensors()):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    for parameter, expected in full_gradients.items():
        torch.testing.assert_close(split_gradients[parameter], expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        torch.cat(split_input_gradients, dim=1),
        full_input_gradient,
        rtol=1e-11,
        atol=1e-11,
    )


def test_core_online_dispatch_preserves_forward_schema_and_replays_on_backward() -> None:
    torch.manual_seed(31)
    core = DABSNCore(input_dim=2, hidden_dim=3).to(dtype=torch.float64)
    inputs = torch.randn(2, 4, 2, dtype=torch.float64, requires_grad=True)
    recurrent = core.initial_state(2, dtype=torch.float64)
    eligibility = initial_core_eligibility(
        core,
        batch_size=2,
        device="cpu",
        dtype=torch.float64,
    )
    frame = OnlineCoreFrame(eligibility)
    expected, expected_final = core._reference_forward_from_state(
        inputs.detach(),
        initial_state=recurrent,
        return_writes=True,
        return_cocktail=True,
        return_final_state=True,
    )

    actual, actual_final = core.forward_from_state(
        inputs,
        initial_state=recurrent,
        return_writes=True,
        return_cocktail=True,
        return_final_state=True,
        temporal_credit="online",
        eligibility_state=eligibility,
        online_frame=frame,
    )

    assert len(actual) == 7
    for actual_signal, expected_signal in zip(actual, expected):
        torch.testing.assert_close(actual_signal, expected_signal, rtol=0, atol=0)
    for actual_state, expected_state in zip(actual_final, expected_final):
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    assert not frame.replayed
    assert frame.resulting_eligibility is None

    loss = sum(signal.square().mean() for signal in actual)
    loss.backward()

    assert frame.replayed
    assert frame.resulting_eligibility is not None
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in core.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in core.parameters())


def test_bf16_replay_sums_all_512_unit_beta_contributions_in_fp32():
    core = DABSNCore(1, 1).float()
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.zero_()
        core.logit_alpha.fill_(100)
    eligibility = initial_core_eligibility(core, 1, device="cpu")
    _, tape = core_online_forward_reference(core, torch.zeros(1, 512, 1), core.initial_state(1), eligibility)
    core.bfloat16()
    with torch.no_grad():
        _, gradients, _ = core_online_replay_analytic(
            core, tape, CoreSignalGrads(next_budget=torch.ones(1, 512, 1)), eligibility)
    assert float(gradients[core.beta]) == 512.0
    assert gradients[core.beta].dtype == torch.bfloat16


def test_core_online_dispatch_rejects_frame_reuse() -> None:
    core = DABSNCore(input_dim=2, hidden_dim=3)
    inputs = torch.randn(1, 2, 2, requires_grad=True)
    recurrent = core.initial_state(1)
    eligibility = initial_core_eligibility(core, 1, device="cpu")
    frame = OnlineCoreFrame(eligibility)

    result = core.forward_from_state(
        inputs,
        initial_state=recurrent,
        return_writes=True,
        return_cocktail=True,
        temporal_credit="online",
        eligibility_state=eligibility,
        online_frame=frame,
    )
    sum(signal.sum() for signal in result).backward()

    with torch.no_grad(), pytest.raises(RuntimeError, match="already replayed"):
        core.forward_from_state(
            inputs.detach(),
            initial_state=recurrent,
            return_writes=True,
            return_cocktail=True,
            temporal_credit="online",
            eligibility_state=eligibility,
            online_frame=frame,
        )
