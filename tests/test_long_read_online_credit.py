import pytest
import torch

from dabsn.read import DABSNRead
from dabsn.temporal_credit import (
    OnlineLongReadFrame,
    initial_long_read_eligibility,
    long_read_online_autograd,
    long_read_online_forward,
    long_read_online_replay_analytic,
    long_read_online_replay,
)


LONG_PARAMETER_NAMES = (
    "logit_retain",
    "logit_expect_retain",
    "logit_retention_decay",
    "logit_retention_strength",
)


def _randomize_long_parameters(read: DABSNRead) -> None:
    with torch.no_grad():
        for name in LONG_PARAMETER_NAMES:
            getattr(read, name).copy_(0.5 * torch.randn_like(getattr(read, name)))


@pytest.mark.parametrize("steps", [1, 3, 6])
def test_long_read_online_credit_matches_exact_bptt_within_chunk(steps: int) -> None:
    torch.manual_seed(210 + steps)
    batch, hidden = 2, 3
    read = DABSNRead(hidden, "seq").to(torch.float64)
    _randomize_long_parameters(read)
    write = torch.randn(batch, steps, hidden, dtype=torch.float64, requires_grad=True)
    plasticity = torch.sigmoid(
        torch.randn(batch, steps, hidden, dtype=torch.float64)
    ).requires_grad_(True)
    novelty = torch.randn(batch, steps, hidden, dtype=torch.float64, requires_grad=True)
    initial_state = tuple(
        value.to(torch.float64).requires_grad_(True)
        for value in read.initial_long_scan_state(batch)
    )
    cotangent = torch.randn(batch, steps, hidden, dtype=torch.float64)

    expected_output, expected_final = read._long_scan_from_state(
        write,
        plasticity,
        novelty,
        initial_state=initial_state,
        return_final_state=True,
    )
    expected_loss = (expected_output * cotangent).sum()
    expected = torch.autograd.grad(
        expected_loss,
        (
            write,
            plasticity,
            novelty,
            *initial_state,
            *(getattr(read, name) for name in LONG_PARAMETER_NAMES),
        ),
        allow_unused=True,
    )

    eligibility = initial_long_read_eligibility(
        read,
        batch,
        device=write.device,
        dtype=torch.float64,
    )
    actual_output, tape, actual_final = long_read_online_forward(
        read,
        write.detach(),
        plasticity.detach(),
        novelty.detach(),
        tuple(value.detach() for value in initial_state),
        eligibility,
    )
    _, parameter_gradients, input_gradients, incoming_state = long_read_online_replay(
        read,
        tape,
        cotangent,
        eligibility,
        return_incoming_state_grad=True,
    )

    torch.testing.assert_close(actual_output, expected_output.detach())
    for actual, reference in zip(actual_final, expected_final):
        torch.testing.assert_close(actual, reference.detach())
    for actual, reference in zip(input_gradients, expected[:3]):
        expected_tensor = torch.zeros_like(actual) if reference is None else reference
        torch.testing.assert_close(actual, expected_tensor, rtol=1e-9, atol=1e-10)
    for actual, reference in zip(incoming_state, expected[3:6]):
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-10)
    for name, reference in zip(LONG_PARAMETER_NAMES, expected[6:]):
        parameter = getattr(read, name)
        reference = torch.zeros_like(parameter) if reference is None else reference
        torch.testing.assert_close(
            parameter_gradients[parameter], reference, rtol=1e-9, atol=1e-10
        )


def test_long_read_parameter_eligibility_is_chunk_invariant() -> None:
    torch.manual_seed(220)
    batch, steps, hidden = 2, 7, 3
    read = DABSNRead(hidden, "seq").to(torch.float64)
    _randomize_long_parameters(read)
    write = torch.randn(batch, steps, hidden, dtype=torch.float64)
    plasticity = torch.sigmoid(torch.randn(batch, steps, hidden, dtype=torch.float64))
    novelty = torch.randn(batch, steps, hidden, dtype=torch.float64)
    cotangent = torch.randn(batch, steps, hidden, dtype=torch.float64)
    initial_state = tuple(value.to(torch.float64) for value in read.initial_long_scan_state(batch))
    initial_eligibility = initial_long_read_eligibility(
        read, batch, device=write.device, dtype=torch.float64
    )

    _, full_tape, full_final = long_read_online_forward(
        read, write, plasticity, novelty, initial_state, initial_eligibility
    )
    full_eligibility, full_parameters, _ = long_read_online_replay(
        read, full_tape, cotangent, initial_eligibility
    )

    state = initial_state
    eligibility = initial_eligibility
    accumulated = {getattr(read, name): torch.zeros_like(getattr(read, name)) for name in LONG_PARAMETER_NAMES}
    for start, stop in ((0, 2), (2, 5), (5, 7)):
        _, tape, state = long_read_online_forward(
            read,
            write[:, start:stop],
            plasticity[:, start:stop],
            novelty[:, start:stop],
            state,
            eligibility,
        )
        eligibility, gradients, _ = long_read_online_replay(
            read, tape, cotangent[:, start:stop], eligibility
        )
        for parameter, gradient in gradients.items():
            accumulated[parameter] += gradient

    for actual, expected in zip(state, full_final):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        eligibility.traces, full_eligibility.traces, rtol=1e-12, atol=1e-12
    )
    for parameter in accumulated:
        torch.testing.assert_close(
            accumulated[parameter], full_parameters[parameter], rtol=1e-9, atol=1e-10
        )


def test_long_read_analytic_replay_matches_autograd_oracle() -> None:
    torch.manual_seed(229)
    batch, steps, hidden = 2, 6, 4
    read = DABSNRead(hidden, "seq").to(torch.float64)
    _randomize_long_parameters(read)
    write = torch.randn(batch, steps, hidden, dtype=torch.float64)
    plasticity = torch.sigmoid(torch.randn(batch, steps, hidden, dtype=torch.float64))
    novelty = torch.randn(batch, steps, hidden, dtype=torch.float64)
    cotangent = torch.randn(batch, steps, hidden, dtype=torch.float64)
    recurrent = tuple(value.to(torch.float64) for value in read.initial_long_scan_state(batch))
    eligibility = initial_long_read_eligibility(
        read, batch, device="cpu", dtype=torch.float64
    )
    _, tape, _ = long_read_online_forward(
        read, write, plasticity, novelty, recurrent, eligibility
    )
    expected_state, expected_gradients, expected_inputs = long_read_online_replay(
        read, tape, cotangent, eligibility
    )
    actual_state, actual_gradients, actual_inputs = long_read_online_replay_analytic(
        read, tape, cotangent, eligibility
    )

    torch.testing.assert_close(actual_state.traces, expected_state.traces, rtol=1e-12, atol=1e-12)
    for parameter in expected_gradients:
        torch.testing.assert_close(
            actual_gradients[parameter], expected_gradients[parameter], rtol=2e-11, atol=2e-11
        )
    for actual, expected in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)


def test_long_read_autograd_boundary_installs_gradients_and_advances_once() -> None:
    torch.manual_seed(221)
    batch, steps, hidden = 2, 4, 3
    read = DABSNRead(hidden, "seq").to(torch.float64)
    _randomize_long_parameters(read)
    write = torch.randn(batch, steps, hidden, dtype=torch.float64, requires_grad=True)
    plasticity = torch.sigmoid(
        torch.randn(batch, steps, hidden, dtype=torch.float64)
    ).requires_grad_(True)
    novelty = torch.randn(batch, steps, hidden, dtype=torch.float64, requires_grad=True)
    initial_state = tuple(value.to(torch.float64) for value in read.initial_long_scan_state(batch))
    eligibility = initial_long_read_eligibility(
        read, batch, device=write.device, dtype=torch.float64
    )
    frame = OnlineLongReadFrame(eligibility)
    output, final_state = long_read_online_autograd(
        read,
        write,
        plasticity,
        novelty,
        initial_state,
        eligibility,
        frame,
    )
    assert frame.active and not frame.replayed
    output.square().mean().backward()

    assert frame.replayed and frame.resulting_eligibility is not None
    assert write.grad is not None and plasticity.grad is not None and novelty.grad is not None
    assert all(getattr(read, name).grad is not None for name in LONG_PARAMETER_NAMES)
    assert all(value.grad_fn is not None for value in final_state)
    with pytest.raises(RuntimeError, match="already replayed"):
        long_read_online_autograd(
            read,
            write,
            plasticity,
            novelty,
            initial_state,
            eligibility,
            frame,
        )


def test_full_read_online_long_path_matches_one_chunk_bptt() -> None:
    torch.manual_seed(222)
    batch, steps, hidden = 1, 4, 3
    reference_read = DABSNRead(hidden, "seq").to(torch.float64)
    actual_read = DABSNRead(hidden, "seq").to(torch.float64)
    actual_read.load_state_dict(reference_read.state_dict())

    def inputs():
        return tuple(
            value.requires_grad_(True)
            for value in (
                torch.randn(batch, steps, hidden, dtype=torch.float64),
                torch.randn(batch, steps, hidden, dtype=torch.float64),
                torch.randn(batch, steps, hidden, dtype=torch.float64),
                torch.randn(batch, steps, hidden, dtype=torch.float64),
                torch.rand(batch, steps, hidden, dtype=torch.float64),
                torch.rand(batch, steps, hidden, dtype=torch.float64),
                torch.rand(batch, steps, hidden, dtype=torch.float64),
                torch.rand(batch, steps, hidden, dtype=torch.float64),
            )
        )

    reference_inputs = inputs()
    actual_inputs = tuple(value.detach().clone().requires_grad_(True) for value in reference_inputs)
    reference_output = reference_read(*reference_inputs)
    cotangent = torch.randn_like(reference_output)
    (reference_output * cotangent).sum().backward()

    eligibility = initial_long_read_eligibility(
        actual_read, batch, device="cpu", dtype=torch.float64
    )
    frame = OnlineLongReadFrame(eligibility)
    actual_output, final_state = actual_read(
        *actual_inputs,
        temporal_credit="online",
        initial_long_state=actual_read.initial_long_scan_state(
            batch, device="cpu", dtype=torch.float64
        ),
        long_eligibility_state=eligibility,
        online_long_frame=frame,
        return_long_state=True,
    )
    torch.testing.assert_close(actual_output, reference_output)
    assert all(value.grad_fn is not None for value in final_state)
    (actual_output * cotangent).sum().backward()

    assert frame.replayed
    for actual, reference in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=1e-9, atol=1e-10)
    for (name, actual), reference in zip(
        actual_read.named_parameters(), reference_read.parameters()
    ):
        actual_gradient = torch.zeros_like(actual) if actual.grad is None else actual.grad
        reference_gradient = (
            torch.zeros_like(reference) if reference.grad is None else reference.grad
        )
        torch.testing.assert_close(
            actual_gradient, reference_gradient, rtol=1e-9, atol=1e-10, msg=name
        )
