"""Online forward dispatch and precise batched tape contract."""

import copy
import importlib.util
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from dabsn.core import DABSNCore
from dabsn.temporal_credit import core_online_forward_reference, initial_core_eligibility


@pytest.mark.parametrize(
    "batch,input_dim,state_dim,requested",
    [
        (8, 1024, 6144, "auto"),
        (4, 17, 1025, "auto"),
        (64, 3, 7, "auto"),
        (1, 3, 5, "batched"),
    ],
)
def test_wide_online_forward_uses_batched_dispatch_without_persistent_runtime(
    monkeypatch, batch, input_dim, state_dim, requested
):
    from dabsn.kernels import triton as dispatch

    monkeypatch.setenv("DABSN_CORE_BACKEND", requested)

    def unavailable_persistent_runtime():
        pytest.fail("wide online forward reached the persistent GEMV runtime")

    monkeypatch.setattr(dispatch, "_runtime", unavailable_persistent_runtime)
    with torch.device("meta"):
        core = DABSNCore(input_dim, state_dim).bfloat16()
    mode = FakeTensorMode()
    # CPU-only PyTorch cannot index fake CUDA tensors. Simulate the CUDA
    # admission flag while executing the real dispatcher with metadata-only
    # CPU tensors; numerical and real-CUDA parity are separate tests below.
    core._apply(lambda value: FakeTensor(mode, value, torch.device("cpu")))
    monkeypatch.setattr(FakeTensor, "is_cuda", property(lambda self: True))
    with mode, torch.no_grad():
        inputs = torch.empty(batch, 2, input_dim, dtype=torch.bfloat16)
        state = tuple(torch.empty(batch, state_dim) for _ in range(3))
        result, tape = dispatch.cuda_core_online_forward(core, inputs, state, None)
        assert result.trajectory.shape == (batch, 2, 2 * state_dim)
        assert result.trajectory.dtype == torch.bfloat16
        assert all(
            getattr(tape, field.name).dtype == torch.float32
            for field in fields(tape)
            if field.name != "inputs"
        )
        assert core._last_core_online_backend == "cuda_batched_fp32_forward"


@pytest.mark.parametrize(
    "batch,steps,input_dim,state_dim", [(1, 1, 5, 3), (3, 4, 3, 7), (2, 3, 7, 5)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_precise_batched_online_tapes_match_reference(
    batch, steps, input_dim, state_dim, dtype, autocast
):
    from dabsn.kernels.online_forward import batched_online_forward_tapes

    torch.manual_seed(113)
    core = DABSNCore(input_dim, state_dim).to(dtype)
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)
    inputs = torch.randn(batch, steps, input_dim, dtype=dtype)
    state = (
        torch.randn(batch, state_dim) * 0.2,
        torch.rand(batch, state_dim),
        torch.rand(batch, state_dim),
    )
    before = tuple(value.clone() for value in state)
    projected, gate_projected = core.W(inputs), core.Wg(inputs)
    reference = copy.deepcopy(core).float()
    # The scan receives already-rounded projections. Hold that boundary fixed
    # to isolate recurrence precision from the input projection GEMMs.
    reference.W.forward = lambda _inputs: projected.float()
    reference.Wg.forward = lambda _inputs: gate_projected.float()
    eligibility = initial_core_eligibility(reference, batch, device="cpu", dtype=torch.float32)
    expected, tape = core_online_forward_reference(reference, inputs.float(), state, eligibility)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = batched_online_forward_tapes(core, projected, gate_projected, state)
    expected_values = (
        expected.trajectory,
        expected.novelty,
        expected.plasticity,
        expected.expression,
        expected.write,
        expected.energy,
        expected.saturation,
        tape.gate,
        *expected.final_state,
    )
    assert len(actual) == 11
    for observed, wanted in zip(actual, expected_values):
        assert observed.dtype == torch.float32
        assert not observed.requires_grad
        torch.testing.assert_close(observed, wanted, rtol=2e-6, atol=2e-6)
    for observed, wanted in zip(state, before):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
    assert all(parameter.dtype == dtype for parameter in core.parameters())


@pytest.mark.parametrize("target", ["inputs", "parameters", "state"])
def test_precise_forward_rejects_fp64_instead_of_silent_narrowing(target):
    from dabsn.kernels.online_forward import batched_online_forward_tapes

    core = DABSNCore(3, 5)
    projected = torch.randn(2, 3, 5)
    state = core.initial_state(2, device="cpu", dtype=torch.float32)
    if target == "inputs":
        projected = projected.double()
    elif target == "parameters":
        core.double()
    else:
        state = tuple(value.double() for value in state)
    with pytest.raises(TypeError, match="FP16, BF16, or FP32"):
        batched_online_forward_tapes(core, projected, projected, state)


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="real CUDA and Triton are required for fused online forward parity",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_dim,state_dim", [(5, 7), (7, 257)])
def test_fused_cuda_batched_online_matches_cpu_tapes(monkeypatch, dtype, input_dim, state_dim):
    from dabsn.kernels.online_forward import batched_online_forward_tapes

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("native BF16 hardware is required")
    torch.manual_seed(512)
    core = DABSNCore(input_dim, state_dim).to(dtype)
    inputs = torch.randn(3, 4, input_dim, dtype=dtype) * 0.2
    projected, gate_projected = core.W(inputs).detach(), core.Wg(inputs).detach()
    state = (
        torch.randn(3, state_dim) * 0.1,
        torch.rand(3, state_dim),
        torch.rand(3, state_dim) * 0.1,
    )
    expected = batched_online_forward_tapes(core, projected, gate_projected, state)
    gpu_core = copy.deepcopy(core).cuda()
    previous_precision = torch.get_float32_matmul_precision()
    try:
        # Isolate kernel arithmetic from the caller's optional TF32 policy.
        torch.set_float32_matmul_precision("highest")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            actual = batched_online_forward_tapes(
                gpu_core,
                projected.cuda(),
                gate_projected.cuda(),
                tuple(value.cuda() for value in state),
            )
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    for left, right in zip(actual, expected):
        assert left.dtype == torch.float32
        torch.testing.assert_close(left.cpu(), right, rtol=3e-5, atol=3e-6)


def _emulated_pointwise_kernels():
    """Execute kernel bodies on CPU; this deliberately does not emulate JIT."""
    from triton_cpu_emulator import load_kernel_bodies

    from dabsn.kernels import online_forward

    path = Path(online_forward.__file__).with_name("online_forward_triton.py")
    return load_kernel_bodies(path, {"_tanh", "_softplus", "_prepare_y", "_finish_step"})


@pytest.mark.parametrize(
    "hidden,dtype", [(7, torch.float32), (257, torch.float32), (13, torch.bfloat16)]
)
def test_fused_kernel_equations_and_masked_tape_offsets_on_cpu(hidden, dtype):
    from dabsn.kernels.online_forward import batched_online_forward_tapes

    torch.manual_seed(907)
    batch, steps = 3, 4
    core = DABSNCore(5, hidden).to(dtype)
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)
    inputs = torch.randn(batch, steps, 5, dtype=dtype)
    projected, gate_projected = core.W(inputs).detach(), core.Wg(inputs).detach()
    state = (
        torch.randn(batch, hidden) * 0.1,
        torch.rand(batch, hidden),
        torch.rand(batch, hidden) * 0.1,
    )
    expected = batched_online_forward_tapes(core, projected, gate_projected, state)
    recurrent = torch.cat((core.Ug.weight.float(), core.A.weight.float())).detach()
    params = torch.stack(
        tuple(
            getattr(core, name).detach().float().expand(hidden)
            for name in (
                "beta",
                "log_kappa",
                "logit_recover",
                "k_s",
                "k_y",
                "k_b",
                "k_n",
                "k_bias",
                "r_s",
                "r_y",
                "r_b",
                "r_n",
                "r_bias",
                "logit_saturation_decay",
                "k_saturation",
                "r_saturation",
                "logit_alpha",
                "log_lambda",
                "logit_saturation_suppress",
            )
        )
    )
    actual = tuple(torch.empty_like(value) for value in expected[:8])
    budget, energy, saturation = (value.clone() for value in state)
    y = torch.empty(batch, hidden)
    kernels, language, pointer = _emulated_pointwise_kernels()
    for step in range(steps):
        for block in range((batch * hidden + 255) // 256):
            language.program = block
            kernels["_prepare_y"](
                pointer(projected), pointer(budget), pointer(y), batch, steps, hidden, step, 256
            )
        recurrence = y @ recurrent.T
        for block in range((batch * hidden + 255) // 256):
            language.program = block
            kernels["_finish_step"](
                *(
                    pointer(value)
                    for value in (
                        gate_projected,
                        y,
                        recurrence,
                        params,
                        budget,
                        energy,
                        saturation,
                        *actual,
                    )
                ),
                batch,
                steps,
                hidden,
                step,
                256,
            )
    for observed, wanted in zip((*actual, budget, energy, saturation), expected):
        torch.testing.assert_close(observed, wanted, rtol=2e-5, atol=2e-6)
