"""Small-trace scans must retain the supplied hybrid credit recurrence."""

import importlib.util

import pytest
import torch

from dabsn.core import DABSNCore
from dabsn.temporal_credit import (
    _advance_core_eligibility,
    core_online_forward_reference,
    initial_core_eligibility,
)


def _fixture(device, steps=5, clamped=False):
    torch.manual_seed(3041)
    core = DABSNCore(5, 7).to(device)
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.add_(torch.randn_like(parameter) * .15)
        if clamped:
            core.log_kappa.fill_(5.)
            core.logit_recover.fill_(8.)
    eligibility = initial_core_eligibility(core, 2, device=device)
    eligibility.local.normal_(std=.2)
    eligibility.global_.normal_(std=.2)
    inputs = torch.randn(2, steps, 5, device=device) * .4
    recurrent = (torch.randn(2, 7, device=device) * .2,
                 torch.rand(2, 7, device=device), torch.rand(2, 7, device=device))
    _, tape = core_online_forward_reference(core, inputs, recurrent, eligibility)
    incoming = torch.randn(2, steps, 7, 3, device=device)
    return core, tape, incoming, eligibility


def _expected(core, tape, incoming, eligibility):
    local_grad = torch.zeros_like(eligibility.local[:, :, 0])
    global_grad = torch.zeros_like(eligibility.global_[:, :, 0])
    jacobians = []
    for step in range(tape.inputs.shape[1]):
        local_grad += torch.einsum('bsi,bsip->bsp', incoming[:, step], eligibility.local)
        global_grad += torch.einsum('bsi,bsip->bsp', incoming[:, step], eligibility.global_)
        # Independent one-channel formula for the budget self Jacobian.
        y = tape.y[:, step]
        gate = tape.gate[:, step]
        hard = ((tape.gate_preactivation[:, step] > -3) &
                (tape.gate_preactivation[:, step] < 3)).float() / 6
        jacobians.append(1 - core.logit_alpha.sigmoid() +
            torch.nn.functional.softplus(core.log_lambda) * tape.energy[:, step] *
            (1-y.square()) * (gate * core.A.weight.diag() +
            tape.expression[:, step] * hard * core.Ug.weight.diag()))
        eligibility = _advance_core_eligibility(core, tape, step, eligibility)
    return (eligibility.local, eligibility.global_, local_grad.sum(0),
            global_grad.sum((0, 1)), torch.stack(jacobians, 1))


def test_small_trace_reference_preserves_nonzero_boundary_credit():
    from dabsn.kernels.online_trace import core_trace_scan_reference

    args = _fixture('cpu')
    actual = core_trace_scan_reference(*args)
    expected = _expected(*args)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or
                    importlib.util.find_spec('triton') is None,
                    reason='native trace scan requires CUDA and Triton')
@pytest.mark.parametrize('steps', [1, 5, 33])
def test_native_trace_scan_matches_nonzero_boundary_credit(steps):
    from dabsn.kernels.online_trace import core_trace_scan

    args = _fixture('cuda', steps)
    actual = core_trace_scan(*args)
    expected = _expected(*args)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=5e-6, rtol=5e-5)


def _weight_fixture(device):
    # Zero and negative Jacobians prohibit a cumulative-product division trick.
    jbb = torch.tensor([[[.5, 0.], [-.25, .8], [0., .4], [.9, -.6]]], device=device)
    incoming = torch.tensor([[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]], device=device)
    expected_boundary = torch.tensor([[1.875, 2.]], device=device)
    expected_multiplier = torch.tensor([[0., 0.]], device=device)
    expected_future = torch.tensor([[[1.75, 11.36], [5., 9.2], [7., 8.], [0., 0.]]], device=device)
    expected_suffix = torch.tensor([[[0., -.192], [0., -.24], [.9, -.6], [1., 1.]]], device=device)
    return jbb, incoming, (expected_boundary, expected_multiplier, expected_future, expected_suffix)


def test_budget_weight_reference_handles_zero_and_negative_jacobians():
    from dabsn.temporal_credit import _budget_trace_weights

    jbb, incoming, expected = _weight_fixture('cpu')
    for left, right in zip(_budget_trace_weights(jbb, incoming), expected):
        torch.testing.assert_close(left, right)


@pytest.mark.skipif(not torch.cuda.is_available() or
                    importlib.util.find_spec('triton') is None,
                    reason='native budget scan requires CUDA and Triton')
def test_native_budget_weights_handle_zero_and_negative_jacobians():
    from dabsn.kernels.online_trace import budget_trace_weights

    jbb, incoming, expected = _weight_fixture('cuda')
    for left, right in zip(budget_trace_weights(jbb, incoming), expected):
        torch.testing.assert_close(left, right)


def _kernel_bodies():
    from pathlib import Path
    from triton_cpu_emulator import load_kernel_bodies

    path = Path(__file__).parents[1] / 'src/dabsn/kernels/online_trace.py'
    return load_kernel_bodies(path, ('_tanh', '_softplus', '_param', '_scan', '_weights'))


@pytest.mark.parametrize('steps,clamped', [(1, False), (5, False), (33, False), (5, True)])
def test_native_trace_body_equations_and_indexing_on_cpu(steps, clamped):
    # Executes the shipped kernel body, including masking and pointer offsets;
    # this is not evidence of Triton compilation or GPU performance.
    scope, language, pointer = _kernel_bodies()
    core, tape, incoming, eligibility = _fixture('cpu', steps, clamped)
    names = ('beta', 'log_kappa', 'logit_recover', 'k_s', 'k_y', 'k_b',
             'k_n', 'k_bias', 'r_s', 'r_y', 'r_b', 'r_n', 'r_bias',
             'k_saturation', 'r_saturation')
    params = torch.stack([core.Wg.bias.detach().reshape(-1)] +
                         [getattr(core, name).detach().reshape(-1) for name in names])
    globals_ = torch.stack([getattr(core, name).detach().reshape(()) for name in
                           ('logit_alpha', 'log_lambda', 'logit_saturation_decay',
                            'logit_saturation_suppress')])
    output = (torch.empty_like(eligibility.local), torch.empty_like(eligibility.global_),
              torch.empty(2, 7, 16), torch.empty(2, 7, 4), torch.empty(2, steps, 7))
    tapes = [getattr(tape, name).contiguous() for name in
             ('budget', 'energy', 'saturation', 'y', 'expression', 'gate_preactivation')]
    arguments = (*tapes, incoming, params, globals_,
                 core.A.weight.detach().diag(), core.Ug.weight.detach().diag(),
                 eligibility.local, eligibility.global_, *output)
    for batch in range(2):
        for block in range(2):
            language.program = (batch, block)
            scope['_scan'](*(pointer(value) for value in arguments), 7, steps, 4)
    actual = (*output[:2], output[2].sum(0), output[3].sum((0, 1)), output[4])
    for left, right in zip(actual, _expected(core, tape, incoming, eligibility)):
        torch.testing.assert_close(left, right, atol=5e-6, rtol=5e-5)


def test_native_budget_body_equations_and_indexing_on_cpu():
    scope, language, pointer = _kernel_bodies()
    jbb, incoming, expected = _weight_fixture('cpu')
    output = tuple(torch.empty_like(value) for value in expected)
    language.program = (0, 0)
    scope['_weights'](*(pointer(value) for value in (jbb, incoming, *output)), 2, 4, 128)
    for left, right in zip(output, expected):
        torch.testing.assert_close(left, right)
