import runpy
import subprocess
import sys
from pathlib import Path

import torch

from dabsn.core import DABSNCore
from dabsn.temporal_credit import (
    CoreSignalGrads,
    OnlineCoreFrame,
    core_online_forward_reference,
    core_online_replay_reference,
    initial_core_eligibility,
)


def _randomize_core(core: DABSNCore) -> None:
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.copy_(0.2 * torch.randn_like(parameter))
        core.logit_alpha.fill_(-1.3)
        core.log_lambda.fill_(-0.7)
        core.logit_saturation_decay.fill_(0.4)
        core.logit_saturation_suppress.fill_(-1.1)


def test_native_cpu_online_replay_matches_full_signal_reference() -> None:
    from dabsn.kernels.cpu import native_core_online_replay

    torch.manual_seed(108)
    batch, steps, input_dim, state_dim = 2, 4, 3, 2
    core = DABSNCore(input_dim, state_dim)
    _randomize_core(core)
    inputs = torch.randn(batch, steps, input_dim)
    initial_state = (
        0.15 * torch.randn(batch, state_dim),
        torch.sigmoid(torch.randn(batch, state_dim)),
        torch.sigmoid(torch.randn(batch, state_dim)),
    )
    eligibility = initial_core_eligibility(
        core,
        batch,
        device=inputs.device,
        dtype=torch.float32,
    )
    _, tape = core_online_forward_reference(core, inputs, initial_state, eligibility)
    shape = (batch, steps, state_dim)
    signal_grads = CoreSignalGrads(
        y=torch.randn(shape),
        next_budget=torch.randn(shape),
        novelty=torch.randn(shape),
        plasticity=torch.randn(shape),
        expression=torch.randn(shape),
        write=torch.randn(shape),
        energy=torch.randn(shape),
        saturation=torch.randn(shape),
    )

    expected_state, expected_parameters, expected_input = core_online_replay_reference(
        core,
        tape,
        signal_grads,
        eligibility,
    )
    actual_state, actual_parameters, actual_input = native_core_online_replay(
        core,
        tape,
        signal_grads,
        eligibility,
        required=True,
    )

    for actual, expected in zip(actual_state.tensors(), expected_state.tensors()):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for parameter in core.parameters():
        torch.testing.assert_close(
            actual_parameters[parameter],
            expected_parameters[parameter],
            rtol=2e-5,
            atol=2e-6,
        )
    torch.testing.assert_close(actual_input, expected_input, rtol=2e-5, atol=2e-6)


def _online_core_boundary_dispatch_check() -> None:
    from dabsn.kernels.cpu import enable_native_cpu_scan

    torch.manual_seed(109)
    batch, steps, input_dim, state_dim = 2, 3, 4, 3
    reference_core = DABSNCore(input_dim, state_dim)
    _randomize_core(reference_core)
    actual_core = DABSNCore(input_dim, state_dim)
    actual_core.load_state_dict(reference_core.state_dict())
    inputs = torch.randn(batch, steps, input_dim)
    initial_state = (
        0.1 * torch.randn(batch, state_dim),
        torch.sigmoid(torch.randn(batch, state_dim)),
        torch.sigmoid(torch.randn(batch, state_dim)),
    )
    eligibility = initial_core_eligibility(
        reference_core,
        batch,
        device=inputs.device,
        dtype=torch.float32,
    )
    reference_result, tape = core_online_forward_reference(
        reference_core,
        inputs,
        initial_state,
        eligibility,
    )
    shape = (batch, steps, state_dim)
    signals = CoreSignalGrads(
        y=torch.randn(shape),
        next_budget=torch.randn(shape),
        novelty=torch.randn(shape),
        plasticity=torch.randn(shape),
        expression=torch.randn(shape),
        write=torch.randn(shape),
        energy=torch.randn(shape),
        saturation=torch.randn(shape),
    )
    expected_state, expected_parameters, expected_input = core_online_replay_reference(
        reference_core,
        tape,
        signals,
        eligibility,
    )

    assert enable_native_cpu_scan(required=True)
    actual_input = inputs.clone().requires_grad_(True)
    frame = OnlineCoreFrame(eligibility)
    result, final_state = actual_core.forward_from_state(
        actual_input,
        initial_state=initial_state,
        return_writes=True,
        return_cocktail=True,
        return_final_state=True,
        temporal_credit="online",
        eligibility_state=eligibility,
        online_frame=frame,
    )
    actual_trajectory, actual_novelty, actual_plasticity, actual_expression, actual_write, actual_energy, actual_saturation = result
    for actual, expected in zip(
        (*result, *final_state),
        (
            reference_result.trajectory,
            reference_result.novelty,
            reference_result.plasticity,
            reference_result.expression,
            reference_result.write,
            reference_result.energy,
            reference_result.saturation,
            *reference_result.final_state,
        ),
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    loss = (
        (actual_trajectory[..., :state_dim] * signals.y).sum()
        + (actual_trajectory[..., state_dim:] * signals.next_budget).sum()
        + (actual_novelty * signals.novelty).sum()
        + (actual_plasticity * signals.plasticity).sum()
        + (actual_expression * signals.expression).sum()
        + (actual_write * signals.write).sum()
        + (actual_energy * signals.energy).sum()
        + (actual_saturation * signals.saturation).sum()
    )
    loss.backward()

    assert actual_core._last_core_backend == "cpu_native_cpp_online"
    assert frame.replayed and frame.resulting_eligibility is not None
    for actual, expected in zip(frame.resulting_eligibility.tensors(), expected_state.tensors()):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for (name, actual_parameter), reference_parameter in zip(
        actual_core.named_parameters(),
        reference_core.parameters(),
    ):
        assert actual_parameter.grad is not None, name
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameters[reference_parameter],
            rtol=2e-5,
            atol=2e-6,
        )
    torch.testing.assert_close(actual_input.grad, expected_input, rtol=2e-5, atol=2e-6)


def test_online_core_boundary_dispatches_native_cpu_forward_and_replay() -> None:
    path = repr(str(Path(__file__).resolve()))
    program = f"import runpy; runpy.run_path({path})['_online_core_boundary_dispatch_check']()"
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
