"""CUDA parity, determinism and capture for the fused admitted read and fused long memory."""
import importlib.util

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton are required",
)


def _bank(B, T, H, frac, dtype, seed=1):
    g = torch.Generator().manual_seed(seed)
    member = torch.rand(B, T, generator=g) < frac
    member[:, 0] = True
    counts = member.sum(1)
    n = int(counts.max())
    order = torch.argsort(member.float(), dim=1, descending=True, stable=True)
    idx = order[:, :n]
    valid = torch.arange(n).unsqueeze(0) < counts.unsqueeze(1)

    def r(*s):
        return torch.randn(*s, generator=g, dtype=torch.float64)

    writes = r(B, T, H)
    bw = torch.gather(writes, 1, idx.unsqueeze(-1).expand(-1, -1, H))
    nw = torch.gather(writes, 1, (idx + 1).clamp(max=T - 1).unsqueeze(-1).expand(-1, -1, H))
    floats = [
        F.normalize(r(B, T, H), dim=-1), F.normalize(bw, dim=-1), bw, nw,
        F.normalize(r(B, T, 4).abs(), dim=-1), F.normalize(r(B, n, 4).abs(), dim=-1),
        r(B, n) * 0.5, r(B, n) * 0.5, torch.tensor([3.0], dtype=torch.float64),
        *[torch.tensor([v], dtype=torch.float64) for v in (1.1, 0.7, 0.3, 0.9)],
    ]
    floats = [t.cuda().to(dtype).requires_grad_() for t in floats]
    return floats, idx.cuda(), valid.cuda()


def _run(fn, floats, idx, valid, cot, mode):
    q, bk, bw, nw, cq, ck, kb, ad, sc, sg, pg, ig, cg = floats
    for t in floats:
        t.grad = None
    y = fn(q, bk, bw, nw, cq, ck, kb, ad, sc, idx, valid, mode=mode,
           short_gain=sg, pad_gain=pg, induct_gain=ig, cocktail_gain=cg)
    (y.double() * cot).sum().backward()
    return [y.detach().double()] + [t.grad.detach().double() for t in floats]


def _assert_fp32_parity(actual, expected):
    # Scalar parameter gradients are cancelling sums over every score; bound
    # them against the scalar-gradient scale rather than their own (small) value.
    scalar_scale = max(e.abs().item() for e in expected if e.numel() == 1)
    for a, e in zip(actual, expected):
        if e.numel() == 1:
            assert (a - e).abs().item() < 2e-5 * scalar_scale
        else:
            assert ((a - e).abs().max() / (e.abs().max() + 1e-30)).item() < 2e-5


def _reference(*args, **kwargs):
    from dabsn.kernels.compact_admitted import admitted_three_way_read_compact_dense

    return admitted_three_way_read_compact_dense(*args, score_budget=1 << 30, **kwargs)


@pytest.mark.parametrize("mode", ["seq", "field"])
@pytest.mark.parametrize("B,T,H,frac", [(3, 200, 40, 0.4), (2, 130, 100, 0.15), (1, 70, 768, 1.0)])
def test_fused_read_matches_fp64_reference(mode, B, T, H, frac):
    from dabsn.kernels.read_flash import admitted_three_way_read_flash

    torch.set_float32_matmul_precision("highest")
    cot = torch.randn(B, T, H, device="cuda", dtype=torch.float64)
    expected = _run(_reference, *_bank(B, T, H, frac, torch.float64), cot, mode)
    actual = _run(admitted_three_way_read_flash, *_bank(B, T, H, frac, torch.float32), cot, mode)
    _assert_fp32_parity(actual, expected)


@pytest.mark.parametrize("shift", [5, 150])
def test_fused_read_carried_bank_negative_positions(shift):
    """Online chunks carry bank entries from earlier chunks; relative to the chunk
    their positions are negative, so every query row sees them. The bank backward
    must still only visit query rows 0..T-1 (it once started at a negative row)."""
    from dabsn.kernels.read_flash import admitted_three_way_read_flash

    torch.set_float32_matmul_precision("highest")
    B, T, H = 2, 200, 40
    cot = torch.randn(B, T, H, device="cuda", dtype=torch.float64)
    expected_inputs = _bank(B, T, H, 0.4, torch.float64)
    actual_inputs = _bank(B, T, H, 0.4, torch.float32)
    assert torch.equal(expected_inputs[1], actual_inputs[1])
    idx = expected_inputs[1] - shift
    assert (idx < 0).any()
    expected = _run(_reference, expected_inputs[0], idx, expected_inputs[2], cot, "seq")
    actual = _run(admitted_three_way_read_flash, actual_inputs[0], idx, actual_inputs[2], cot,
                  "seq")
    _assert_fp32_parity(actual, expected)


def test_fused_read_bf16_within_operand_error_and_deterministic():
    """bf16 operands, FP32 parameters (the autocast situation), against FP64 on the
    same rounded inputs. Tensor gradients stay within bf16 operand error. The
    five scalar gradients are sums over B*T*N terms with heavy cancellation, so
    their error is bounded relative to the scalar-gradient scale; the existing
    dense bf16 read measures in the same class on these inputs."""
    from dabsn.kernels.read_flash import admitted_three_way_read_flash

    floats, idx, valid = _bank(3, 200, 64, 0.5, torch.bfloat16)
    floats = [t.detach().float().requires_grad_() if t.numel() == 1 else t for t in floats]
    cot = torch.randn(3, 200, 64, device="cuda", dtype=torch.float64)
    expected = _run(_reference, [t.detach().double().requires_grad_() for t in floats], idx,
                    valid, cot, "seq")
    first = _run(admitted_three_way_read_flash, floats, idx, valid, cot, "seq")
    second = _run(admitted_three_way_read_flash, floats, idx, valid, cot, "seq")
    scalar_scale = max(e.abs().item() for e in expected if e.numel() == 1)
    for a, b, e in zip(first, second, expected):
        assert torch.equal(a, b)
        if e.numel() == 1:
            assert (a - e).abs().item() < 2 ** -5 * scalar_scale
        else:
            assert ((a - e).abs().max() / (e.abs().max() + 1e-30)).item() < 2 ** -6


def test_fused_read_is_graph_capturable():
    from dabsn.kernels.read_flash import admitted_three_way_read_flash

    floats, idx, valid = _bank(2, 96, 32, 0.5, torch.float32)
    q, bk, bw, nw, cq, ck, kb, ad, sc, sg, pg, ig, cg = [t.detach() for t in floats]

    def call():
        return admitted_three_way_read_flash(q, bk, bw, nw, cq, ck, kb, ad, sc, idx, valid,
                                             mode="seq", short_gain=sg, pad_gain=pg,
                                             induct_gain=ig, cocktail_gain=cg)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        eager = call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(eager, captured)


@pytest.mark.parametrize("T", [41, 700])
@pytest.mark.parametrize("carried,expand", [(True, True), (False, False)])
def test_fused_long_memory_matches_reference(carried, expand, T):
    """T=41 runs the sequential kernel; T=700 the time-chunked forward."""
    from dabsn.kernels.long_fused import fused_long_memory
    from dabsn.read import DABSNRead

    torch.manual_seed(0)
    B, H = 3, 37
    read = DABSNRead(H).cuda().double()
    with torch.no_grad():
        read.logit_retain.fill_(0.7)
        read.logit_expect_retain.fill_(1.3)
        read.logit_retention_decay.fill_(-0.4)
        read.logit_retention_strength.fill_(0.9)
    w = torch.randn(B, T, H, device="cuda", dtype=torch.float64)
    p = torch.rand(B, T, 1 if expand else H, device="cuda", dtype=torch.float64)
    state = [torch.randn(B, H, device="cuda", dtype=torch.float64) * 0.5 for _ in range(3)]
    gy = torch.randn(B, T, H, device="cuda", dtype=torch.float64)
    gf = [torch.randn(B, H, device="cuda", dtype=torch.float64) for _ in range(3)]
    logits = [read.logit_retain, read.logit_expect_retain, read.logit_retention_decay,
              read.logit_retention_strength]

    def go(fn, dtype):
        ww = w.detach().to(dtype).clone().requires_grad_()
        pp = p.detach().to(dtype).clone().requires_grad_()
        ss = [s.detach().to(dtype).clone().requires_grad_() for s in state] if carried else None
        for logit in logits:
            logit.grad = None
        out, final = fn(ww, pp.expand(B, T, H), ss)
        loss = (out.double() * gy).sum() + sum((f.double() * g).sum() for f, g in zip(final, gf))
        loss.backward()
        values = [out.double(), *[f.double() for f in final], ww.grad.double(), pp.grad.double()]
        values += [s.grad.double() for s in ss] if carried else []
        return values + [logit.grad.double().clone() for logit in logits]

    expected = go(lambda ww, pp, ss: read._long_scan_from_state(
        ww, pp, pp, initial_state=ss, return_final_state=True), torch.float64)
    actual = go(lambda ww, pp, ss: fused_long_memory(read, ww, pp, ss, True), torch.float32)
    # The last four are the logit gradients: cancelling sums over B*T*H terms,
    # bounded against the scalar-gradient scale like the read's scalars.
    scalar_scale = max(e.abs().item() for e in expected[-4:])
    for a, e in zip(actual[:-4], expected[:-4]):
        assert ((a - e).abs().max() / (e.abs().max() + 1e-30)).item() < 1e-5
    for a, e in zip(actual[-4:], expected[-4:]):
        assert (a - e).abs().item() < 1e-5 * scalar_scale


@pytest.mark.parametrize("mode", ["seq", "field"])
@pytest.mark.parametrize("B,T,H,frac", [(3, 150, 40, 0.4), (2, 97, 130, 0.2), (2, 64, 32, 1.0)])
def test_token_read_matches_gathered_fp64_reference(mode, B, T, H, frac):
    from dabsn.kernels.read_flash import admitted_three_way_read_tokens

    torch.set_float32_matmul_precision("highest")
    g = torch.Generator().manual_seed(7)
    member = torch.rand(B, T, generator=g) < frac
    member[:, 0] = True
    member[0, T - 1] = True  # exercises the clamped next-write row
    counts = member.sum(1)
    n = int(counts.max())
    order = torch.argsort(member.float(), dim=1, descending=True, stable=True)
    idx = order[:, :n].cuda()
    valid = (torch.arange(n).unsqueeze(0) < counts.unsqueeze(1)).cuda()
    base = [torch.randn(B, T, H, generator=g, dtype=torch.float64) * 2,
            torch.randn(B, T, H, generator=g, dtype=torch.float64),
            F.normalize(torch.rand(B, T, 4, generator=g, dtype=torch.float64), dim=-1),
            torch.randn(B, T, generator=g, dtype=torch.float64),
            *[torch.tensor([v], dtype=torch.float64) for v in (2.5, 1.1, 0.7, 0.3, 0.9)]]
    cot = torch.randn(B, T, H, device="cuda", dtype=torch.float64)

    def run(dtype, fused):
        leaves = [t.cuda().to(dtype).requires_grad_() for t in base]
        x, w, cock, kba, sc, sg, pg, ig, cg = leaves
        gains = dict(short_gain=sg, pad_gain=pg, induct_gain=ig, cocktail_gain=cg)
        if fused:
            y = admitted_three_way_read_tokens(x, w, cock, kba, sc, idx, valid, mode=mode, **gains)
        else:
            gather = lambda t: torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
            bw = gather(w)
            nw = torch.gather(w, 1, (idx + 1).clamp(max=T - 1).unsqueeze(-1).expand(-1, -1, H))
            y = _reference(F.normalize(x, dim=-1), F.normalize(bw, dim=-1), bw, nw, cock,
                           gather(cock), torch.gather(kba, 1, idx), torch.zeros_like(
                               torch.gather(kba, 1, idx)), sc, idx, valid, mode=mode, **gains)
        (y.double() * cot).sum().backward()
        return [y.detach().double()] + [t.grad.double() for t in leaves]

    expected = run(torch.float64, False)
    actual = run(torch.float32, True)
    _assert_fp32_parity(actual, expected)


@pytest.mark.parametrize("geometry", ["seq", "field", "hybrid"])
def test_host_memory_cuda_kernels_match_reference_path(geometry):
    from dabsn import DABSNHostMemory

    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(5)
    fast = DABSNHostMemory(48, kernels="auto", read_geometry=geometry).cuda()
    slow = DABSNHostMemory(48, kernels="reference", read_geometry=geometry).cuda()
    slow.load_state_dict(fast.state_dict())
    h = torch.randn(4, 90, 48, device="cuda")
    outs, grads = [], []
    for memory in (fast, slow):
        x = h.clone().requires_grad_()
        y = memory(x)
        y.square().sum().backward()
        outs.append(y.detach())
        grads.append([x.grad] + [p.grad for p in memory.parameters()])
    assert fast.kernel_status["last_three_way_backend"] == "registered_fused_tokens_trainable"
    assert fast.kernel_status["last_long_backend"] == "cuda_triton_fused"
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-4, atol=1e-5)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
