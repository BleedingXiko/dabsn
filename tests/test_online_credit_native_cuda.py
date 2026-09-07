import importlib.util
import copy

import pytest
import torch

from dabsn.core import DABSNCore
from dabsn.read import DABSNRead
from dabsn.temporal_credit import (
    CoreSignalGrads,
    core_online_forward_reference,
    core_online_replay_reference,
    initial_core_eligibility,
    initial_long_read_eligibility,
    long_read_online_forward,
    long_read_online_replay,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton are required for the online CUDA parity gate",
)


def _signals(result) -> CoreSignalGrads:
    return CoreSignalGrads(
        y=torch.randn_like(result.trajectory[..., : result.novelty.shape[-1]]),
        next_budget=torch.randn_like(result.novelty),
        novelty=torch.randn_like(result.novelty),
        plasticity=torch.randn_like(result.plasticity),
        expression=torch.randn_like(result.expression),
        write=torch.randn_like(result.write),
        energy=torch.randn_like(result.energy),
        saturation=torch.randn_like(result.saturation),
    )


def test_cuda_online_forward_and_replay_match_reference() -> None:
    from dabsn.kernels.triton import cuda_core_online_forward, cuda_core_online_replay

    torch.manual_seed(901)
    device = torch.device("cuda")
    core = DABSNCore(input_dim=5, hidden_dim=7).to(device)
    inputs = torch.randn(2, 4, 5, device=device)
    recurrent = (
        torch.randn(2, 7, device=device) * 0.1,
        torch.rand(2, 7, device=device),
        torch.rand(2, 7, device=device) * 0.1,
    )
    eligibility = initial_core_eligibility(
        core, 2, device=device, dtype=torch.float32
    )
    expected_result, expected_tape = core_online_forward_reference(
        core, inputs, recurrent, eligibility
    )
    actual_result, actual_tape = cuda_core_online_forward(
        core, inputs, recurrent, eligibility
    )
    for actual, expected in zip(
        (
            actual_result.trajectory,
            actual_result.novelty,
            actual_result.plasticity,
            actual_result.expression,
            actual_result.write,
            actual_result.energy,
            actual_result.saturation,
            *actual_result.final_state,
        ),
        (
            expected_result.trajectory,
            expected_result.novelty,
            expected_result.plasticity,
            expected_result.expression,
            expected_result.write,
            expected_result.energy,
            expected_result.saturation,
            *expected_result.final_state,
        ),
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    signals = _signals(expected_result)
    expected_state, expected_gradients, expected_input = core_online_replay_reference(
        core, expected_tape, signals, eligibility
    )
    actual_state, actual_gradients, actual_input = cuda_core_online_replay(
        core, actual_tape, signals, eligibility
    )
    for actual, expected in zip(actual_state.tensors(), expected_state.tensors()):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    for parameter in core.parameters():
        torch.testing.assert_close(
            actual_gradients[parameter],
            expected_gradients[parameter],
            rtol=3e-5,
            atol=3e-6,
        )
    torch.testing.assert_close(actual_input, expected_input, rtol=3e-5, atol=3e-6)
    assert (
        core._last_core_online_backend
        == "cuda_triton_forward_inductor_fused_replay"
    )


def test_cuda_online_long_forward_and_replay_match_reference() -> None:
    from dabsn.kernels.triton import (
        cuda_long_read_online_forward,
        cuda_long_read_online_replay,
    )

    torch.manual_seed(907)
    read = DABSNRead(7, "seq").cuda()
    write = torch.randn(2, 5, 7, device="cuda")
    plasticity = torch.rand_like(write)
    novelty = torch.rand_like(write)
    recurrent = read.initial_long_scan_state(2, device="cuda")
    eligibility = initial_long_read_eligibility(read, 2, device="cuda")
    expected_output, expected_tape, expected_final = long_read_online_forward(
        read, write, plasticity, novelty, recurrent, eligibility
    )
    actual_output, actual_tape, actual_final = cuda_long_read_online_forward(
        read, write, plasticity, novelty, recurrent, eligibility
    )
    torch.testing.assert_close(actual_output, expected_output, rtol=2e-5, atol=2e-6)
    for actual, expected in zip(actual_final, expected_final):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    cotangent = torch.randn_like(actual_output)
    expected_state, expected_gradients, expected_inputs = long_read_online_replay(
        read, expected_tape, cotangent, eligibility
    )
    actual_state, actual_gradients, actual_inputs = cuda_long_read_online_replay(
        read, actual_tape, cotangent, eligibility
    )
    torch.testing.assert_close(
        actual_state.traces, expected_state.traces, rtol=3e-5, atol=3e-6
    )
    for parameter in expected_gradients:
        torch.testing.assert_close(
            actual_gradients[parameter], expected_gradients[parameter], rtol=3e-5, atol=3e-6
        )
    for actual, expected in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert read._last_long_online_backend == "cuda_inductor_fused_forward_replay"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_low_precision_core_retains_fp32_eligibility(dtype):
    from dabsn.kernels.triton import cuda_core_online_forward, cuda_core_online_replay

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("Native BF16 compilation requires CUDA compute capability >= 8.0; emulation is not a native kernel")
    torch.manual_seed(921)
    core = DABSNCore(5, 7).cuda().to(dtype)
    reference = copy.deepcopy(core).float()
    inputs = torch.randn(2, 3, 5, device="cuda", dtype=dtype) * .1
    state = core.initial_state(2, device="cuda", dtype=torch.float32)
    eligibility = initial_core_eligibility(core, 2, device="cuda", dtype=torch.float32)
    expected, _ = core_online_forward_reference(reference, inputs.float(), state, eligibility)
    actual, tape = cuda_core_online_forward(core, inputs, state, eligibility)
    tolerance = .002 if dtype == torch.float16 else .02
    torch.testing.assert_close(actual.trajectory.float(), expected.trajectory, rtol=tolerance, atol=tolerance)
    assert all(value.dtype == torch.float32 for value in eligibility.tensors())
    assert tape.y.dtype == torch.float32
    signals = _signals(actual)
    # Use the same FP32 native tape in the autograd oracle, isolating replay
    # correctness from low-precision forward rounding.
    expected_state, expected_gradients, expected_input = core_online_replay_reference(reference, tape, signals, eligibility)
    actual_state, actual_gradients, actual_input = cuda_core_online_replay(core, tape, signals, eligibility)
    for left, right in zip(actual_state.tensors(), expected_state.tensors()):
        torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)
    for parameter, reference_parameter in zip(core.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_gradients[parameter].float(), expected_gradients[reference_parameter], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual_input.float(), expected_input.float(), rtol=tolerance, atol=tolerance)


def test_cuda_online_rejects_emulated_bf16_before_compilation():
    from dabsn.kernels.online_compile import require_native_online_dtype

    if torch.cuda.is_bf16_supported(including_emulation=False):
        require_native_online_dtype(torch.device("cuda"), torch.bfloat16)
    else:
        with pytest.raises(RuntimeError, match="native BF16"):
            require_native_online_dtype(torch.device("cuda"), torch.bfloat16)
