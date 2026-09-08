"""Native long-read equations exercised on CPU, with separate CUDA parity."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dabsn.read import DABSNRead
from dabsn.temporal_credit import (
    LongReadOnlineTape,
    initial_long_read_eligibility,
    long_read_online_forward,
    long_read_online_replay,
)


@pytest.mark.parametrize("batch,steps,hidden", [(1, 1, 3), (2, 5, 7), (2, 9, 19)])
def test_actual_long_kernel_bodies_match_reference_on_cpu(batch, steps, hidden):
    from triton_cpu_emulator import load_kernel_bodies

    path = Path(__file__).parents[1] / "src/dabsn/kernels/online_long_triton.py"
    ns, tl, raw_pointer = load_kernel_bodies(path, {"_long_forward_kernel", "_long_replay_kernel"})
    def pointer(value):
        result = raw_pointer(value)
        result.dtype = SimpleNamespace(element_ty=value.dtype)
        return result
    torch.manual_seed(188)
    read = DABSNRead(hidden_dim=hidden, read_geometry="seq")
    values = tuple(torch.randn(batch, steps, hidden) * .3 for _ in range(3))
    state = tuple(torch.rand(batch, hidden) for _ in range(3))
    eligibility = initial_long_read_eligibility(read, batch, device="cpu")
    eligibility.traces.normal_(std=.1)
    rates = torch.sigmoid(torch.stack([value.detach().reshape(()) for value in (
        read.logit_retain, read.logit_expect_retain, read.logit_retention_decay, read.logit_retention_strength,
    )]))
    expected, tape, final = long_read_online_forward(read, *values, state, eligibility)
    output = torch.empty_like(values[0])
    saved = tuple(torch.empty_like(output) for _ in range(3))
    actual_final = tuple(torch.empty_like(state[0]) for _ in range(3))
    block = 16
    for index in range((batch * hidden + block - 1) // block):
        tl.program = (index,)
        ns["_long_forward_kernel"](
            *(pointer(value) for value in (*values[:2], rates, *state, output, *saved, *actual_final)),
            batch, hidden, steps, block,
        )
    torch.testing.assert_close(output, expected, rtol=3e-6, atol=3e-7)
    for left, right in zip(actual_final, final):
        torch.testing.assert_close(left, right, rtol=3e-6, atol=3e-7)
    actual_tape = LongReadOnlineTape(*values, *saved)
    gradient = torch.randn_like(output)
    expected = long_read_online_replay(read, tape, gradient, eligibility)
    traces = torch.empty_like(eligibility.traces)
    partials = torch.empty(batch * hidden, 4)
    input_gradients = tuple(torch.empty_like(output) for _ in range(2))
    for index in range((batch * hidden + block - 1) // block):
        tl.program = (index,)
        ns["_long_replay_kernel"](
            *(pointer(value) for value in (
                *values[:2], actual_tape.long, actual_tape.expected, actual_tape.retention,
                rates, rates, gradient, eligibility.traces, traces, partials, *input_gradients,
            )), batch, hidden, steps, block,
        )
    torch.testing.assert_close(traces, expected[0].traces, rtol=5e-6, atol=5e-7)
    for left, right in zip(partials.sum(0), expected[1].values()):
        torch.testing.assert_close(left.reshape_as(right), right, rtol=5e-6, atol=5e-7)
    for left, right in zip(input_gradients, expected[2][:2]):
        torch.testing.assert_close(left, right, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("batch,steps,hidden", [(1, 1, 3), (2, 5, 7)])
def test_portable_long_native_boundary_matches_canonical(batch, steps, hidden):
    from dabsn.kernels.online_long import long_online_forward, long_online_replay

    torch.manual_seed(304)
    read = DABSNRead(hidden_dim=hidden, read_geometry="seq")
    write, plasticity, novelty = (torch.randn(batch, steps, hidden) for _ in range(3))
    state = tuple(torch.rand(batch, hidden) for _ in range(3))
    eligibility = initial_long_read_eligibility(read, batch, device="cpu")
    eligibility.traces.normal_(std=.1)
    expected, tape, final = long_read_online_forward(read, write, plasticity, novelty, state, eligibility)
    actual, actual_tape, actual_final = long_online_forward(read, write, plasticity, novelty, state, eligibility)
    torch.testing.assert_close(actual, expected)
    for left, right in zip(actual_final, final):
        torch.testing.assert_close(left, right)
    gradient = torch.randn_like(write)
    expected = long_read_online_replay(read, tape, gradient, eligibility)
    actual = long_online_replay(read, actual_tape, gradient, eligibility)
    torch.testing.assert_close(actual[0].traces, expected[0].traces)
    for parameter in expected[1]:
        torch.testing.assert_close(actual[1][parameter], expected[1][parameter])
    for left, right in zip(actual[2], expected[2]):
        torch.testing.assert_close(left, right)


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
                    reason="CUDA and Triton are required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_long_native_forward_replay_parity(dtype):
    from dabsn.kernels.online_long import long_online_forward, long_online_replay

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("native BF16 hardware is required")
    torch.manual_seed(881)
    read = DABSNRead(hidden_dim=7, read_geometry="seq").to(dtype)
    gpu_read = copy.deepcopy(read).cuda()
    values = tuple(torch.randn(2, 5, 7, dtype=dtype) * .2 for _ in range(3))
    state = tuple(torch.rand(2, 7, dtype=dtype) for _ in range(3))
    eligibility = initial_long_read_eligibility(read, 2, device="cpu")
    eligibility.traces.normal_(std=.1)
    gpu_eligibility = type(eligibility)(eligibility.traces.cuda())
    expected, tape, final = long_read_online_forward(read, *values, state, eligibility)
    actual, gpu_tape, gpu_final = long_online_forward(
        gpu_read, *(value.cuda() for value in values), tuple(value.cuda() for value in state), gpu_eligibility,
    )
    tolerance = 3e-5 if dtype == torch.float32 else (.005 if dtype == torch.float16 else .04)
    torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
    for left, right in zip(gpu_final, final):
        torch.testing.assert_close(left.cpu(), right, rtol=tolerance, atol=tolerance)
    gradient = torch.randn_like(values[0])
    expected = long_read_online_replay(read, tape, gradient, eligibility)
    actual = long_online_replay(gpu_read, gpu_tape, gradient.cuda(), gpu_eligibility)
    torch.testing.assert_close(actual[0].traces.cpu(), expected[0].traces, rtol=tolerance, atol=tolerance)
    for left, right in zip(actual[1].values(), expected[1].values()):
        torch.testing.assert_close(left.cpu(), right, rtol=tolerance, atol=tolerance)
    for left, right in zip(actual[2], expected[2]):
        torch.testing.assert_close(left.cpu(), right, rtol=tolerance, atol=tolerance)
