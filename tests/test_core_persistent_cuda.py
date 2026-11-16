"""CUDA regression gate for the persistent Core scans and the fused online replay.

Every check compares against the FP64 eager reference (BPTT) or the FP64 CPU
online reference, in FP32 with IEEE matmuls, plus bitwise determinism, CUDA
graph capture, both tile layouts, and the analytic trace scan against the
dual-number scan it replaces.
"""

import copy

import pytest
import torch

def _persistent_triton_ok() -> bool:
    try:
        from dabsn.kernels.core_persistent import _PERSISTENT_TRITON_OK
    except Exception:
        return False
    return bool(_PERSISTENT_TRITON_OK)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not _persistent_triton_ok(),
        reason="persistent Core is enabled only on Triton >= 3.4 (wrong under 3.2)",
    ),
]


@pytest.fixture(autouse=True)
def _ieee_fp32():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    from dabsn.kernels.triton import enable_triton_kernels

    enable_triton_kernels(required=True)
    yield
    torch.set_float32_matmul_precision(previous)


def _core(B, H, T, seed=0):
    from dabsn.core import DABSNCore

    torch.manual_seed(seed + B * 1000 + H + T)
    core = DABSNCore(H, H)
    with torch.no_grad():
        for name in ("k_s", "k_y", "k_b", "k_n", "k_bias", "r_s", "r_y", "r_b", "r_n",
                     "r_bias", "k_saturation", "r_saturation", "beta"):
            getattr(core, name).normal_(std=0.3)
        core.logit_recover.normal_(-1.0, 0.5)
        core.log_kappa.normal_(-1.0, 0.5)
        core.logit_saturation_suppress.fill_(0.3)
    return core


def _bptt(core, x, cots, *, device, dtype):
    c = copy.deepcopy(core).to(device=device, dtype=dtype)
    xi = x.to(device=device, dtype=dtype).requires_grad_(True)
    if dtype == torch.float64:
        outs = c._reference_forward_from_state(xi, return_writes=True, return_cocktail=True)
    else:
        outs = c(xi, return_writes=True, return_cocktail=True)
    sum((o.double() * k.to(o.device).double()).sum() for o, k in zip(outs, cots)).backward()
    result = {f"out{i}": o.detach().double().cpu() for i, o in enumerate(outs)}
    result["dx"] = xi.grad.double().cpu()
    result.update({f"d{n}": p.grad.double().cpu() for n, p in c.named_parameters()})
    return result


@pytest.mark.parametrize("layout", ["auto", "general"])
@pytest.mark.parametrize("B,H,T", [(3, 40, 37), (16, 96, 33), (8, 256, 24)])
def test_persistent_bptt_matches_fp64_reference_and_is_deterministic(monkeypatch, layout, B, H, T):
    if layout == "general":
        monkeypatch.setenv("DABSN_CORE_PERSIST_TUNE", "MODE=general")
    monkeypatch.setenv("DABSN_CORE_PERSISTENT", "1")
    core = _core(B, H, T)
    x = torch.randn(B, T, H)
    with torch.no_grad():
        shapes = [o.shape for o in core._reference_forward_from_state(x, return_writes=True, return_cocktail=True)]
    g = torch.Generator().manual_seed(7)
    cots = [torch.randn(shape, generator=g) for shape in shapes]
    ref = _bptt(core, x, cots, device="cpu", dtype=torch.float64)
    new = _bptt(core, x, cots, device="cuda", dtype=torch.float32)
    again = _bptt(core, x, cots, device="cuda", dtype=torch.float32)
    for key in ref:
        scale = ref[key].abs().max().clamp_min(1e-12)
        assert (new[key] - ref[key]).abs().max() / scale < 1e-3, key
        assert torch.equal(new[key], again[key]), f"{key} not bitwise deterministic"


def test_persistent_bptt_carried_state_and_final_state_gradients(monkeypatch):
    monkeypatch.setenv("DABSN_CORE_PERSISTENT", "1")
    B, H, T = 4, 48, 19
    core = _core(B, H, T)
    x = torch.randn(B, T, H)
    state = (torch.randn(B, H) * 0.3, torch.rand(B, H), torch.rand(B, H))

    def run(device, dtype):
        c = copy.deepcopy(core).to(device=device, dtype=dtype)
        xi = x.to(device, dtype).requires_grad_(True)
        init = tuple(v.to(device, dtype).requires_grad_(True) for v in state)
        if dtype == torch.float64:
            (outs, final) = c._reference_forward_from_state(
                xi, initial_state=init, return_writes=True, return_final_state=True)
        else:
            outs, final = c.forward_from_state(xi, initial_state=init, return_writes=True,
                                               return_final_state=True)
        loss = sum(o.double().sin().sum() for o in outs) + sum(f.double().square().sum() for f in final)
        loss.backward()
        return [xi.grad.double().cpu(), *(v.grad.double().cpu() for v in init),
                *(f.detach().double().cpu() for f in final)]

    for got, want in zip(run("cuda", torch.float32), run("cpu", torch.float64)):
        assert (got - want).abs().max() / want.abs().max().clamp_min(1e-12) < 1e-3


def test_persistent_forward_replays_under_cuda_graph(monkeypatch):
    monkeypatch.setenv("DABSN_CORE_PERSISTENT", "1")
    B, H, T = 8, 64, 40
    core = _core(B, H, T).cuda()
    x = torch.randn(B, T, H, device="cuda")
    with torch.no_grad():
        eager = core(x, return_writes=True)
        static_x = x.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            core(static_x, return_writes=True)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = core(static_x, return_writes=True)
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()
    for a, b in zip(eager, captured):
        assert torch.equal(a, b)


def test_persistent_path_refuses_fake_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from dabsn.kernels.core_persistent import persistent_core_supported

    with FakeTensorMode():
        fake = torch.empty(2, 3, 4, device="cuda")
    assert not persistent_core_supported(fake)
    assert persistent_core_supported(torch.empty(2, 3, 4, device="cuda"))


@pytest.mark.parametrize("B,H,T", [(3, 40, 37), (16, 96, 20)])
def test_online_credit_fused_replay_matches_fp64_reference_over_two_chunks(monkeypatch, B, H, T):
    from dabsn.temporal_credit import OnlineCoreFrame, initial_core_eligibility

    monkeypatch.setenv("DABSN_CORE_PERSISTENT", "1")
    core = _core(B, H, T)
    xs = [torch.randn(B, T, H) for _ in range(2)]
    g = torch.Generator().manual_seed(11)
    cots = [torch.randn(B, T, 2 * H, generator=g), *(torch.randn(B, T, H, generator=g) for _ in range(4))]

    def run(device, dtype):
        c = copy.deepcopy(core).to(device=device, dtype=dtype)
        elig = initial_core_eligibility(c, B, device=device, dtype=dtype)
        state = c.initial_state(B, device=device, dtype=dtype)
        out = []
        for x in xs:
            xi = x.to(device, dtype).requires_grad_(True)
            frame = OnlineCoreFrame(elig)
            outs, state = c.forward_from_state(
                xi, initial_state=state, return_writes=True, return_final_state=True,
                temporal_credit="online", eligibility_state=elig, online_frame=frame)
            state = tuple(v.detach() for v in state)
            for p in c.parameters():
                p.grad = None
            sum((o.double() * k.to(o.device).double()).sum() for o, k in zip(outs, cots)).backward()
            elig = frame.resulting_eligibility
            out += [xi.grad, *(p.grad for p in c.parameters()), *elig.tensors()]
        if device == "cuda":
            assert c._last_core_online_backend == "cuda_persistent_forward_fused_replay"
        return [v.detach().double().cpu() for v in out]

    ref = run("cpu", torch.float64)
    new = run("cuda", torch.float32)
    again = run("cuda", torch.float32)
    for got, rep, want in zip(new, again, ref):
        assert (got - want).abs().max() / want.abs().max().clamp_min(1e-12) < 1e-3
        assert torch.equal(got, rep)


@pytest.mark.parametrize("B,H,T", [(3, 40, 37), (5, 70, 129)])
def test_analytic_trace_scan_matches_dual_number_scan(B, H, T):
    from dabsn.kernels.core_online_replay import core_trace_scan_analytic
    from dabsn.kernels.online_trace import core_trace_scan
    from dabsn.kernels.triton import cuda_core_online_forward
    from dabsn.temporal_credit import initial_core_eligibility

    core = _core(B, H, T).cuda()
    elig = initial_core_eligibility(core, B, device="cuda")
    elig.local.normal_()
    elig.global_.normal_()
    with torch.no_grad():
        _, tape = cuda_core_online_forward(
            core, torch.randn(B, T, H, device="cuda"),
            core.initial_state(B, device="cuda", dtype=torch.float32), elig)
    incoming = torch.randn(B, T, H, 3, device="cuda")
    for got, want in zip(core_trace_scan_analytic(core, tape, incoming, elig),
                         core_trace_scan(core, tape, incoming, elig)):
        assert (got - want).abs().max() / want.abs().max().clamp_min(1e-12) < 1e-3
