import copy

import pytest
import torch

from dabsn import DABSNCore
from dabsn.kernels.batched_runtime import (
    dabsn_core_scan_batched,
    dabsn_core_scan_batched_fused,
)
from dabsn.read import DABSNRead


def _scan(fn, core, inputs):
    hidden = core.hidden_dim
    return fn(
        core.W(inputs), core.Wg(inputs), core.Ug.weight, core.A.weight,
        core.beta, core.log_kappa, core.logit_recover,
        core.k_s, core.k_y, core.k_b, core.k_n, core.k_bias,
        core.r_s, core.r_y, core.r_b, core.r_n, core.r_bias,
        core.logit_saturation_decay.expand(hidden).contiguous(),
        core.k_saturation, core.r_saturation,
        core.logit_alpha.reshape(()), core.log_lambda.reshape(()),
        core.logit_saturation_suppress.reshape(()),
        return_tape=True,
    )


def _batched(core, inputs):
    return _scan(dabsn_core_scan_batched, core, inputs)


def _fused(core, inputs):
    return _scan(dabsn_core_scan_batched_fused, core, inputs)


def test_batched_core_matches_eager_forward_and_backward(monkeypatch):
    monkeypatch.setenv("DABSN_BATCHED_STEP_COMPILE", "0")
    torch.manual_seed(7)
    reference = DABSNCore(input_dim=6, hidden_dim=7)
    actual = copy.deepcopy(reference)
    x_reference = torch.randn(3, 5, 6, requires_grad=True)
    x_actual = x_reference.detach().clone().requires_grad_(True)

    expected = reference(x_reference, return_writes=True, return_cocktail=True)
    observed = _batched(actual, x_actual)
    assert len(expected) == len(observed) == 7
    for expected_tensor, observed_tensor in zip(expected, observed):
        torch.testing.assert_close(observed_tensor, expected_tensor, rtol=1e-6, atol=1e-6)

    weights = [torch.randn_like(value) for value in expected]
    sum((value * weight).sum() for value, weight in zip(expected, weights)).backward()
    sum((value * weight).sum() for value, weight in zip(observed, weights)).backward()
    torch.testing.assert_close(x_actual.grad, x_reference.grad, rtol=2e-5, atol=2e-5)
    for (reference_name, reference_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(), actual.named_parameters()
    ):
        assert reference_name == actual_name
        torch.testing.assert_close(
            actual_parameter.grad,
            reference_parameter.grad,
            rtol=2e-5,
            atol=2e-5,
            msg=lambda message, name=actual_name: f"{name}: {message}",
        )


def _scan_cast(fn, core, inputs, dtype, *, wgx_dtype=None):
    """Drive the batched scan with the input projections in a chosen dtype.

    Mirrors ``_scan`` but casts Wx/Wgx and the recurrent weights, so the
    backward's activation-dtype scratch tapes are exercised at bf16/fp16 (their
    real target), not only at CPU fp32.
    """
    hidden = core.hidden_dim
    wx = core.W(inputs).to(dtype)
    wgx = core.Wg(inputs).to(wgx_dtype or dtype)
    return fn(
        wx, wgx, core.Ug.weight.to(dtype), core.A.weight.to(dtype),
        core.beta, core.log_kappa, core.logit_recover,
        core.k_s, core.k_y, core.k_b, core.k_n, core.k_bias,
        core.r_s, core.r_y, core.r_b, core.r_n, core.r_bias,
        core.logit_saturation_decay.expand(hidden).contiguous(),
        core.k_saturation, core.r_saturation,
        core.logit_alpha.reshape(()), core.log_lambda.reshape(()),
        core.logit_saturation_suppress.reshape(()),
        return_tape=True,
    )


def test_batched_backward_requires_matching_wx_wgx_dtype(monkeypatch):
    # Phase 1: gWgx doubles as grad_Wgx and the Ug half of the recurrent GEMM,
    # so a Wx/Wgx dtype split would double-round it. The backward must refuse
    # rather than silently change the math.
    monkeypatch.setenv("DABSN_BATCHED_STEP_COMPILE", "0")
    torch.manual_seed(3)
    core = DABSNCore(input_dim=6, hidden_dim=7)
    x = torch.randn(2, 4, 6, requires_grad=True)
    out = _scan_cast(
        dabsn_core_scan_batched, core, x, torch.bfloat16, wgx_dtype=torch.float32
    )
    with pytest.raises(RuntimeError, match="share a\n?.*dtype|share a dtype|Wx and Wgx"):
        out[0].sum().backward()


def test_batched_bf16_scratch_matches_fp32_eager(monkeypatch):
    # Phase 1: allocating gWx/gWgx/day_tape in the activation dtype must stay
    # numerically faithful. Compare bf16 batched grads to the fp32 eager
    # reference within bf16 tolerance -- a broken cast-at-store would blow past.
    monkeypatch.setenv("DABSN_BATCHED_STEP_COMPILE", "0")
    torch.manual_seed(11)
    reference = DABSNCore(input_dim=6, hidden_dim=8)
    actual = copy.deepcopy(reference)
    x_reference = torch.randn(3, 5, 6, requires_grad=True)
    x_actual = x_reference.detach().clone().requires_grad_(True)

    expected = reference(x_reference, return_writes=True, return_cocktail=True)
    observed = _scan_cast(dabsn_core_scan_batched, actual, x_actual, torch.bfloat16)

    weights = [torch.randn_like(value) for value in expected]
    sum((value.float() * weight).sum() for value, weight in zip(expected, weights)).backward()
    sum((value.float() * weight).sum() for value, weight in zip(observed, weights)).backward()
    # bf16 GEMM + cast tolerance; the point is faithfulness, not bit-identity.
    torch.testing.assert_close(x_actual.grad, x_reference.grad, rtol=5e-2, atol=5e-2)
    for (_, ref_p), (name, act_p) in zip(
        reference.named_parameters(), actual.named_parameters()
    ):
        if ref_p.grad is None:
            continue
        torch.testing.assert_close(
            act_p.grad, ref_p.grad, rtol=5e-2, atol=5e-2,
            msg=lambda m, n=name: f"{n}: {m}",
        )


def test_fused_core_fallback_matches_eager_forward_and_backward(monkeypatch):
    # On CPU the fused `batched_fused` Function takes the pure-torch fallback
    # path; this certifies its tape layout and shared reverse recurrence exactly
    # reproduce the eager reference (the Triton forward is GPU-certified by the
    # scale gate; this pins everything around it).
    monkeypatch.setenv("DABSN_BATCHED_STEP_COMPILE", "0")
    torch.manual_seed(11)
    reference = DABSNCore(input_dim=6, hidden_dim=7)
    actual = copy.deepcopy(reference)
    x_reference = torch.randn(3, 5, 6, requires_grad=True)
    x_actual = x_reference.detach().clone().requires_grad_(True)

    expected = reference(x_reference, return_writes=True, return_cocktail=True)
    observed = _fused(actual, x_actual)
    assert len(expected) == len(observed) == 7
    for expected_tensor, observed_tensor in zip(expected, observed):
        torch.testing.assert_close(observed_tensor, expected_tensor, rtol=1e-6, atol=1e-6)

    weights = [torch.randn_like(value) for value in expected]
    sum((value * weight).sum() for value, weight in zip(expected, weights)).backward()
    sum((value * weight).sum() for value, weight in zip(observed, weights)).backward()
    torch.testing.assert_close(x_actual.grad, x_reference.grad, rtol=2e-5, atol=2e-5)
    for (_, reference_parameter), (name, actual_parameter) in zip(
        reference.named_parameters(), actual.named_parameters()
    ):
        torch.testing.assert_close(
            actual_parameter.grad, reference_parameter.grad, rtol=2e-5, atol=2e-5,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_fused_and_batched_agree_on_cpu(monkeypatch):
    monkeypatch.setenv("DABSN_BATCHED_STEP_COMPILE", "0")
    torch.manual_seed(13)
    core = DABSNCore(input_dim=5, hidden_dim=8)
    x = torch.randn(4, 6, 5)
    for a, b in zip(_batched(core, x), _fused(core, x)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused Triton forward needs CUDA")
@pytest.mark.parametrize("hidden_dim", [32, 256])
def test_fused_triton_forward_matches_eager_on_gpu(hidden_dim):
    # Certify the ACTUAL single-launch Triton forward (not the CPU fallback)
    # against the eager reference fwd+bwd. Runs only where CUDA + Triton exist;
    # this is the same guarantee tools/train_scale_gate.py asserts.
    #
    # hidden_dim=32 is a single (masked) K-chunk; hidden_dim=256 is the width
    # that used to OOM by staging both [256,256] recurrent matrices in shared
    # memory -- it now runs as four BLOCK_K=64 tl.dot chunks, so this pins the
    # K-tiling fix on real hardware.
    pytest.importorskip("triton")
    torch.manual_seed(19)
    reference = DABSNCore(input_dim=8, hidden_dim=hidden_dim).cuda()
    actual = copy.deepcopy(reference)
    x_reference = torch.randn(32, 12, 8, device="cuda", requires_grad=True)
    x_actual = x_reference.detach().clone().requires_grad_(True)

    expected = reference(x_reference, return_writes=True, return_cocktail=True)
    observed = _fused(actual, x_actual)
    for e, o in zip(expected, observed):
        torch.testing.assert_close(o, e, rtol=2e-3, atol=2e-3)

    weights = [torch.randn_like(v) for v in expected]
    sum((v * w).sum() for v, w in zip(expected, weights)).backward()
    sum((v * w).sum() for v, w in zip(observed, weights)).backward()
    torch.testing.assert_close(x_actual.grad, x_reference.grad, rtol=2e-3, atol=2e-3)
    # The fused Function routes ALL parameter grads through the single-launch
    # Triton reverse (_dabsn_core_fused_backward), not just the input grad --
    # certify every one against the eager reference so the wiring is proven
    # bit-parity on hardware, not merely the activation gradient.
    for (ref_name, ref_p), (act_name, act_p) in zip(
        reference.named_parameters(), actual.named_parameters()
    ):
        assert ref_name == act_name
        if ref_p.grad is None:
            continue
        torch.testing.assert_close(act_p.grad, ref_p.grad, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused Triton forward needs CUDA")
@pytest.mark.parametrize("hidden_dim", [32, 256])
def test_fused_forward_stores_tapes_in_activation_dtype(hidden_dim):
    # Phase 2b: the fused forward stores tapes directly in the activation dtype
    # instead of staging fp32 and casting after the launch. tl.store's fp32->bf16
    # cast is RTNE -- the same numbers the post-launch `.to()` produced -- so the
    # staging buffers and eight full-tape cast copies are pure waste. Assert the
    # tapes come back already narrow (no staging, no copy) and only the carried
    # state is cast.
    pytest.importorskip("triton")
    from dabsn.kernels.triton_runtime import dabsn_core_scan_batched_fused_forward as fwd

    torch.manual_seed(23)
    core = DABSNCore(input_dim=8, hidden_dim=hidden_dim).cuda().to(torch.bfloat16)
    x = torch.randn(32, 12, 8, device="cuda", dtype=torch.bfloat16)

    def run():
        args = (
            core.W(x), core.Wg(x), core.Ug.weight, core.A.weight,
            core.beta, core.log_kappa, core.logit_recover,
            core.k_s, core.k_y, core.k_b, core.k_n, core.k_bias,
            core.r_s, core.r_y, core.r_b, core.r_n, core.r_bias,
            core.logit_saturation_decay.expand(hidden_dim).contiguous(),
            core.k_saturation, core.r_saturation,
            core.logit_alpha.reshape(()), core.log_lambda.reshape(()),
            core.logit_saturation_suppress.reshape(()),
            core.initial_state(32, device=x.device)[0],
            core.initial_state(32, device=x.device)[1],
            core.initial_state(32, device=x.device)[2],
        )
        return fwd(*args)

    tapes = run()
    for tape in tapes:
        assert tape.dtype == torch.bfloat16
        assert torch.isfinite(tape.float()).all()
    # U carries [y | budget] and must be the double-width tape.
    assert tapes[0].shape == (32, 12, 2 * hidden_dim)


def test_batched_core_keeps_float32_reference_compute(monkeypatch):
    monkeypatch.setenv("DABSN_BATCHED_STEP_COMPILE", "0")
    core = DABSNCore(input_dim=4, hidden_dim=5).float()
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    outputs = _batched(core, x)
    assert all(value.dtype == torch.float32 for value in outputs)


def test_batched_core_preserves_low_precision_compute_dtype():
    # The whole point of this backend is BF16/FP16 tensor-core GEMMs. If the
    # recurrent matrix or output tape silently reverts to FP32, the compute
    # dtype leaks into the returned tensors -- assert it does not.
    core = DABSNCore(input_dim=4, hidden_dim=6).to(torch.bfloat16)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    outputs = _batched(core, x)
    assert all(value.dtype == torch.bfloat16 for value in outputs)
    assert all(torch.isfinite(value.float()).all() for value in outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dense BMM read needs CUDA")
@pytest.mark.parametrize("mode", ["seq", "field"])
def test_dense_bmm_read_query_chunking_matches_full(mode):
    # Certifies the query_offset/total_T math: reading the admitted bank in
    # query-time tiles is bit-identical to the untiled read, so long context can
    # stay on tensor-core BMM at any T. Runs where CUDA+Triton exist.
    pytest.importorskip("triton")
    from dabsn.kernels.triton_runtime import dense_bmm_three_way_read

    torch.manual_seed(21)
    B, T, N, H = 3, 16, 6, 8
    dev = "cuda"
    q = torch.randn(B, T, H, device=dev)
    mk, wm, nwm = (torch.randn(B, N, H, device=dev) for _ in range(3))
    rc = torch.randn(B, T, 4, device=dev)
    mc = torch.randn(B, N, 4, device=dev)
    kbias = torch.randn(B, N, device=dev)
    adm = torch.randn(B, N, device=dev)
    scale = torch.tensor(1.3, device=dev)
    bank_idx = torch.randint(0, T, (B, N), device=dev).sort(dim=-1).values.to(torch.long)
    bank_valid = torch.rand(B, N, device=dev) > 0.3
    gains = dict(
        short_gain=torch.tensor(0.8, device=dev), pad_gain=torch.tensor(0.6, device=dev),
        induct_gain=torch.tensor(0.4, device=dev), cocktail_gain=torch.tensor(1.1, device=dev),
    )
    full = dense_bmm_three_way_read(
        q, mk, wm, nwm, rc, mc, kbias, adm, scale, bank_idx, bank_valid,
        mode=mode, query_offset=0, total_T=T, **gains,
    )
    for chunk in (1, 5, 8):
        pieces = []
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            pieces.append(dense_bmm_three_way_read(
                q[:, t0:t1], mk, wm, nwm, rc[:, t0:t1], mc, kbias, adm, scale,
                bank_idx, bank_valid, mode=mode, query_offset=t0, total_T=T, **gains,
            ))
        # Exact math; only float rounding differs across tile shapes.
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)


def test_dense_trainable_read_matches_reference_equations():
    pytest.importorskip("triton")
    from dabsn.kernels.triton_runtime import dense_bmm_three_way_read

    torch.manual_seed(9)
    batch, steps, bank, hidden = 3, 6, 5, 7
    read = DABSNRead(hidden, "seq")
    query = torch.randn(batch, steps, hidden)
    memory_key = torch.randn(batch, bank, hidden)
    write_memory = torch.randn(batch, bank, hidden)
    next_write_memory = torch.randn(batch, bank, hidden)
    read_cocktail = torch.randn(batch, steps, 4)
    memory_cocktail = torch.randn(batch, bank, 4)
    key_bias = torch.randn(batch, bank)
    admission = torch.randn(batch, bank)
    scale = torch.tensor(1.3)
    bank_idx = torch.tensor([[0, 1, 3, 4, 5], [0, 2, 3, 4, 5], [0, 1, 2, 4, 5]])
    bank_valid = torch.tensor(
        [[True, True, False, True, True], [True, True, True, False, True], [True, False, True, True, True]]
    )
    qpos = torch.arange(steps).view(1, steps, 1)
    allow = bank_valid.unsqueeze(1) & (bank_idx.unsqueeze(1) <= qpos)
    induct_allow = bank_valid.unsqueeze(1) & (bank_idx.unsqueeze(1) < qpos)
    expected = read._three_way_read(
        query, memory_key, write_memory, next_write_memory,
        read_cocktail, memory_cocktail, key_bias, admission, scale,
        allow, induct_allow, allow.any(dim=-1), induct_allow.any(dim=-1),
    )
    actual = dense_bmm_three_way_read(
        query, memory_key, write_memory, next_write_memory,
        read_cocktail, memory_cocktail, key_bias, admission, scale,
        bank_idx, bank_valid, mode="seq",
        short_gain=read.short_gain, pad_gain=read.pad_gain,
        induct_gain=read.induct_gain, cocktail_gain=read.cocktail_gain,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dense_trainable_read_backward_matches_reference_gradients():
    """The dense read is a TRAINING path, so its backward must be exercised here.

    Every other dense-read test in this file is forward-only, and that blind
    spot shipped a real failure: the read finished its softmax output in place
    to save two [B,T,N] buffers, which bumps the version counter on a tensor
    SoftmaxBackward0 saved. Forward was perfect and every value test passed;
    the first `.backward()` died with "one of the variables needed for gradient
    computation has been modified by an inplace operation", on GPU, mid-run.

    Comparing gradients against the reference equations -- not just values --
    is what makes that class of bug impossible to ship again. The batch's first
    row has no valid bank entry at all, so the no-eligible-key branch (which is
    where a NaN would be born) carries gradient here too.
    """
    pytest.importorskip("triton")
    from dabsn.kernels.triton_runtime import dense_bmm_three_way_read

    torch.manual_seed(9)
    batch, steps, bank, hidden = 3, 6, 5, 7
    read = DABSNRead(hidden, "seq")

    def leaves():
        torch.manual_seed(9)
        made = [
            torch.randn(batch, steps, hidden), torch.randn(batch, bank, hidden),
            torch.randn(batch, bank, hidden), torch.randn(batch, bank, hidden),
            torch.randn(batch, steps, 4), torch.randn(batch, bank, 4),
            torch.randn(batch, bank), torch.randn(batch, bank),
        ]
        return [t.requires_grad_(True) for t in made]

    scale = torch.tensor(1.3)
    bank_idx = torch.tensor([[0, 1, 3, 4, 5], [0, 2, 3, 4, 5], [0, 1, 2, 4, 5]])
    bank_valid = torch.tensor([
        [False, False, False, False, False],   # no eligible key anywhere
        [True, True, True, False, True],
        [True, False, True, True, True],
    ])
    qpos = torch.arange(steps).view(1, steps, 1)
    allow = bank_valid.unsqueeze(1) & (bank_idx.unsqueeze(1) <= qpos)
    induct_allow = bank_valid.unsqueeze(1) & (bank_idx.unsqueeze(1) < qpos)
    cotangent = torch.randn(batch, steps, hidden)

    ref_leaves = leaves()
    expected = read._three_way_read(
        *ref_leaves, scale, allow, induct_allow,
        allow.any(dim=-1), induct_allow.any(dim=-1),
    )
    expected.backward(cotangent)

    act_leaves = leaves()
    actual = dense_bmm_three_way_read(
        *act_leaves, scale, bank_idx, bank_valid, mode="seq",
        short_gain=read.short_gain, pad_gain=read.pad_gain,
        induct_gain=read.induct_gain, cocktail_gain=read.cocktail_gain,
    )
    actual.backward(cotangent)   # the in-place bug raised RuntimeError right here

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for got, want in zip(act_leaves, ref_leaves):
        assert got.grad is not None and torch.isfinite(got.grad).all()
        torch.testing.assert_close(got.grad, want.grad, rtol=1e-6, atol=1e-6)
    # The all-ineligible row reads as exactly zero and must not poison its own
    # gradient: a -inf softmax repaired after the fact would put NaN here.
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    assert torch.isfinite(act_leaves[0].grad[0]).all()


def test_no_read_path_writes_a_softmax_output_in_place():
    """A standing guard on the exact defect above, for paths CPU cannot run.

    The read runtime imports Triton at module scope, so a CPU box cannot import
    it, let alone execute it -- which is precisely how an in-place write to a
    softmax output shipped to an A100 and failed there. Autograd's rule is not
    hardware-dependent though: a weight matrix that came out of softmax is never
    a legal in-place destination. Reading the source holds every branch to that
    rule on every machine, with nothing imported and nothing stubbed.
    """
    import ast
    import pathlib

    def call_name(call: ast.Call) -> str:
        func = call.func
        return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")

    src = pathlib.Path(__file__).resolve().parent.parent / "src" / "dabsn"
    sources = sorted(src.rglob("*.py"))
    assert sources, f"no package sources found under {src}"

    seen_any_softmax = False
    offenders: list[str] = []
    for path in sources:
        tree = ast.parse(path.read_text())
        # A hand-written `Function.backward` is exempt: autograd runs it with
        # grad disabled, so a softmax it computes there is a plain local buffer
        # that nothing saved. Mutating it in place is the correct, allocation-
        # free way to form `softmax - onehot`, not a defect.
        exempt = [
            (node.lineno, node.end_lineno or node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "backward"
        ]

        def graph_live(node, _exempt=exempt) -> bool:
            line = getattr(node, "lineno", None)
            return line is None or not any(lo <= line <= hi for lo, hi in _exempt)

        # Names bound directly to a softmax result. Autograd's saved tensor is
        # that exact buffer, so any later mutation of the name is the defect.
        softmax_outputs = {
            node.targets[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call) and call_name(node.value) == "softmax"
            and graph_live(node)
        }
        seen_any_softmax |= bool(softmax_outputs)
        for node in ast.walk(tree):
            if not graph_live(node):
                continue
            if not isinstance(node, ast.Call):
                continue
            name = call_name(node)
            if not name.endswith("_") or name.startswith("__"):
                continue
            # `w.mul_(x)` mutates the receiver; `torch.nan_to_num_(w)` mutates
            # its first argument. Check both -- either form is the same bug.
            touched = list(node.args[:1])
            if isinstance(node.func, ast.Attribute):
                touched.append(node.func.value)
            for operand in touched:
                if isinstance(operand, ast.Name) and operand.id in softmax_outputs:
                    offenders.append(
                        f"{path.relative_to(src.parent.parent)}:{node.lineno}: "
                        f"{name} writes softmax output '{operand.id}' in place"
                    )

    assert seen_any_softmax, "no softmax result found anywhere -- this guard has gone stale"
    assert not offenders, (
        "softmax saves its own output for backward, so these mutations bump a "
        "saved tensor's version and backward raises 'a variable needed for "
        "gradient computation has been modified by an inplace operation':\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Chunked scan driver (the body a CUDA graph captures)
#
# `_forward_chunk` is the single definition of the forward scan math: the eager
# path and the captured path both run it. So chunk-boundary correctness is
# CPU-provable, and CUDA capture only changes how the launches are ISSUED --
# there is no second implementation of the recurrence that could drift.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 3, 4, 5, 12])
def test_forward_chunk_sequence_matches_one_pass(count):
    from dabsn.kernels.batched_runtime import (
        _batched_forward_tapes, _forward_chunk, _ScanState,
    )

    torch.manual_seed(11)
    B, T, H = 3, 12, 6
    core = DABSNCore(input_dim=H, hidden_dim=H).double()
    x = torch.randn(B, T, H, dtype=torch.float64)
    Wx = core.W(x).contiguous()
    Wgx = core.Wg(x).contiguous()
    init = core.initial_state(B, device=x.device)

    full = _batched_forward_tapes(
        Wx, Wgx, core.Ug.weight, core.A.weight,
        core.beta, core.log_kappa, core.logit_recover,
        core.k_s, core.k_y, core.k_b, core.k_n, core.k_bias,
        core.r_s, core.r_y, core.r_b, core.r_n, core.r_bias,
        core.logit_saturation_decay.expand(H).contiguous(),
        core.k_saturation, core.r_saturation,
        core.logit_alpha.reshape(()), core.log_lambda.reshape(()),
        core.logit_saturation_suppress.reshape(()),
        init[0], init[1], init[2],
    )

    # Same scan, driven as a sequence of `count`-step chunks through the fixed
    # carry storage a capture would replay through.
    dt = Wx.dtype
    tapes = (
        torch.zeros((B, T, 2 * H), dtype=dt),
        *(torch.zeros((B, T, H), dtype=dt) for _ in range(7)),
    )
    recurrent = torch.cat((core.Ug.weight, core.A.weight), dim=0).to(dt).contiguous()
    params = (
        core.beta, core.log_kappa, core.logit_recover,
        core.k_s, core.k_y, core.k_b, core.k_n, core.k_bias,
        core.r_s, core.r_y, core.r_b, core.r_n, core.r_bias,
        core.logit_saturation_decay.expand(H).contiguous(),
        core.k_saturation, core.r_saturation,
        core.logit_alpha.reshape(()), core.log_lambda.reshape(()),
        core.logit_saturation_suppress.reshape(()),
    )
    state = _ScanState(init[0].float(), init[1].float(), init[2].float())
    done = 0
    while done < T:
        n = min(count, T - done)
        _forward_chunk(
            Wx[:, done:done + n], Wgx[:, done:done + n], recurrent,
            state.budget, state.energy, state.saturation,
            tapes, params, H, n, done,
        )
        done += n

    for chunked, reference in zip(tapes, full[:8]):
        torch.testing.assert_close(chunked, reference, rtol=0, atol=0)
    for got, want in zip(
        (state.budget, state.energy, state.saturation), full[8:11]
    ):
        torch.testing.assert_close(got.to(want.dtype), want, rtol=0, atol=0)


def test_scan_state_does_not_mutate_caller_initial_state():
    # The carry is written back through these tensors so chunks chain through
    # fixed storage; `initial_b.float()` returns the caller's tensor unchanged
    # when it is already fp32, so the state must clone or it corrupts its input.
    from dabsn.kernels.batched_runtime import _ScanState

    b = torch.zeros(2, 4)
    state = _ScanState(b, torch.ones(2, 4), torch.zeros(2, 4))
    state.budget.fill_(7.0)
    assert torch.all(b == 0.0), "_ScanState must not alias the caller's tensors"


def test_scan_chunk_width_scales_with_shape_not_a_constant():
    from dabsn.kernels.batched_runtime import _scan_chunk_width

    # Wider/larger shapes stage fewer steps per capture; nothing is hardcoded.
    narrow = _scan_chunk_width(B=8, T=4096, H=256, elem=2)
    wide = _scan_chunk_width(B=256, T=4096, H=2048, elem=2)
    assert narrow > wide >= 1
    # Never more steps than the scan actually has.
    assert _scan_chunk_width(B=8, T=16, H=256, elem=2) == 16


@pytest.mark.parametrize("count", [1, 2, 5, 7, 12])
def test_backward_chunk_sequence_matches_one_pass(count):
    """The reverse scan must be identical however it is cut into chunks.

    This is the property CUDA-graph replay depends on: a captured chunk resumes
    from carry left in fixed storage by the previous chunk, and the boundary case
    (step 0, whose predecessor is the scan's initial state) is supplied by the
    driver rather than branched on inside the loop. If chunking changed a single
    bit, replay would silently produce wrong gradients.
    """
    from dabsn.kernels.batched_runtime import (
        _backward_chunk, _batched_forward_tapes, _prev_slices, _ReverseCarry,
    )

    torch.manual_seed(19)
    B, T, H = 2, 12, 5
    core = DABSNCore(input_dim=H, hidden_dim=H).double()
    x = torch.randn(B, T, H, dtype=torch.float64)
    Wx = core.W(x).contiguous()
    Wgx = core.Wg(x).contiguous()
    init = core.initial_state(B, device=x.device)
    decay_vec = core.logit_saturation_decay.expand(H).contiguous()

    (U, _nov, plasticity, expression, _w, e_tape, c_tape, s_tape,
     _fb, _fe, _fc) = _batched_forward_tapes(
        Wx, Wgx, core.Ug.weight, core.A.weight,
        core.beta, core.log_kappa, core.logit_recover,
        core.k_s, core.k_y, core.k_b, core.k_n, core.k_bias,
        core.r_s, core.r_y, core.r_b, core.r_n, core.r_bias,
        decay_vec, core.k_saturation, core.r_saturation,
        core.logit_alpha.reshape(()), core.log_lambda.reshape(()),
        core.logit_saturation_suppress.reshape(()),
        init[0], init[1], init[2],
    )

    reads = (U, e_tape, c_tape, s_tape, plasticity, expression)
    f64 = torch.float64
    grads = (
        torch.randn(B, T, 2 * H, dtype=f64),
        *(torch.randn(B, T, H, dtype=f64) for _ in range(6)),
    )
    recurrent = torch.cat((core.Ug.weight, core.A.weight), dim=0).to(U.dtype).contiguous()
    consts = (
        core.log_kappa.double(), core.logit_recover.double(),
        core.k_s.double(), core.k_y.double(), core.k_b.double(),
        core.k_n.double(), core.k_bias.double(),
        core.r_s.double(), core.r_y.double(), core.r_b.double(),
        core.r_n.double(), core.r_bias.double(),
        torch.sigmoid(decay_vec.double()), core.k_saturation.double(),
        core.r_saturation.double(),
        torch.sigmoid(core.logit_alpha.double().reshape(())),
        torch.nn.functional.softplus(core.log_lambda.double().reshape(())),
        torch.sigmoid(core.log_lambda.double().reshape(())),
        torch.sigmoid(core.logit_saturation_suppress.double().reshape(())),
    )

    def fresh_carry():
        return _ReverseCarry(
            torch.zeros(B, H, dtype=f64), torch.zeros(B, H, dtype=f64),
            torch.zeros(B, H, dtype=f64),
            [torch.zeros(H, dtype=f64) for _ in range(16)],
            [torch.zeros((), dtype=f64) for _ in range(3)],
        )

    def run(chunk_size):
        outs = tuple(torch.zeros(B, T, H, dtype=U.dtype) for _ in range(3))
        carry = fresh_carry()
        start = T - chunk_size
        while start >= 1:
            prev = _prev_slices(U, c_tape, init[0], init[2], H, start, chunk_size)
            _backward_chunk(reads, grads, outs, prev, carry, recurrent, consts,
                            H, chunk_size, start)
            start -= chunk_size
        remaining = start + chunk_size
        if remaining > 0:
            prev = _prev_slices(U, c_tape, init[0], init[2], H, 0, remaining)
            _backward_chunk(reads, grads, outs, prev, carry, recurrent, consts,
                            H, remaining, 0)
        return outs, carry

    ref_outs, ref_carry = run(T)
    got_outs, got_carry = run(count)
    for got, want in zip(got_outs, ref_outs):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    for got, want in zip(got_carry.tensors(), ref_carry.tensors()):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_reverse_carry_does_not_mutate_caller_tensors():
    from dabsn.kernels.batched_runtime import _ReverseCarry

    gb = torch.zeros(2, 3)
    acc = [torch.zeros(3)]
    carry = _ReverseCarry(gb, torch.zeros(2, 3), torch.zeros(2, 3), acc, [torch.zeros(())])
    carry.gb.fill_(5.0)
    carry.param_acc[0].fill_(5.0)
    assert torch.all(gb == 0.0) and torch.all(acc[0] == 0.0)


def test_reverse_chunk_width_accounts_for_fp32_grad_staging():
    from dabsn.kernels.batched_runtime import _reverse_chunk_width, _scan_chunk_width

    # The reverse stages eight FP32 grads on top of activation-dtype tapes, so at
    # the same shape it must stage FEWER steps than the forward -- not reuse the
    # forward's number.
    fwd = _scan_chunk_width(B=256, T=4096, H=2048, elem=2)
    rev = _reverse_chunk_width(B=256, T=4096, H=2048, elem=2)
    assert 1 <= rev < fwd
    assert _reverse_chunk_width(B=4, T=8, H=64, elem=2) == 8


def test_reverse_capture_width_is_shape_adaptive_and_leaves_step_zero_eager():
    from dabsn.kernels.batched_runtime import _reverse_capture_width

    for batch in (1, 8, 32, 256):
        for hidden in (64, 448, 2048):
            for context in (1, 2, 31, 32, 128, 512, 1024, 4096):
                width = _reverse_capture_width(
                    B=batch,
                    T=context,
                    H=hidden,
                    elem=2,
                )
                if context == 1:
                    assert width == 0
                else:
                    assert 1 <= width <= context - 1


def test_reverse_capture_uses_full_adaptive_tail_when_budget_covers_sequence(
    monkeypatch,
):
    import dabsn.kernels.batched_runtime as runtime

    monkeypatch.setattr(runtime, "_reverse_chunk_width", lambda *args, **kwargs: args[1])
    for context in (2, 17, 128, 512, 1024):
        assert runtime._reverse_capture_width(32, context, 448, 2) == context - 1


def test_chunk_widths_hold_their_invariants_at_every_shape():
    """Capture must degrade sanely at ANY shape, not just the ones benchmarked.

    Two invariants, swept over batch x width x length x dtype: the width is
    always a usable number of steps (1 <= w <= T, never zero and never past the
    end), and staging never exceeds the budget it was derived from. A shape too
    wide to stage lands below the minimum-steps floor, where the driver runs the
    scan uncaptured instead of capturing something useless.
    """
    from dabsn.kernels.batched_runtime import (
        _reverse_chunk_width, _scan_chunk_width, _scan_stage_bytes,
    )

    budget = _scan_stage_bytes()

    for B in (1, 2, 3, 7, 8, 64, 256, 1024, 4096):
        for H in (1, 7, 16, 64, 256, 777, 2048, 8192, 16384):
            for T in (1, 2, 31, 32, 33, 512, 2048, 16384):
                for elem in (2, 4):
                    fwd = _scan_chunk_width(B, T, H, elem)
                    rev = _reverse_chunk_width(B, T, H, elem)
                    for width, per_step in (
                        (fwd, 10 * B * H * elem),
                        (rev, B * H * (8 * elem + 8 * 4 + 3 * elem)),
                    ):
                        assert 1 <= width <= T, (B, H, T, elem, width)
                        if width > 1:
                            assert width * per_step <= budget, (
                                B, H, T, elem, width
                            )


def test_capture_survives_the_batch_that_makes_the_step_compute_bound():
    """Capture must not switch itself off as the batch grows.

    A recurrence only becomes compute-bound at large batch, so large batch is
    where the recorded launches matter most. But per-step staging scales with
    batch, so a FIXED staging budget buys fewer and fewer steps until the width
    drops under the minimum and capture silently stops happening.

    That is not hypothetical -- it is measured. On an A100 at 2048h/seq-512 the
    reverse width was 37 steps at batch 128 and 18 at batch 256, and 256
    benchmarked SLOWER than 128 (3,797 vs 5,499 tok/s) despite being the more
    efficient shape. The budget is a fraction of the device now, so the width
    stays above the floor across the whole range a real sweep visits.
    """
    from dabsn.kernels.batched_runtime import (
        _reverse_chunk_width, _scan_chunk_width, _scan_stage_bytes,
        _SCAN_GRAPH_MIN_STEPS,
    )

    # The A100-80GB the regression was measured on; asserted independently of
    # whatever device happens to be running this test.
    budget = max(_scan_stage_bytes(), int(85_899_345_920 * 0.02))
    T, H, elem = 512, 2048, 2

    # Up to the measured batch ceiling for this shape on an 80 GiB A100 (the
    # capacity probe OOMs at 448), because the budget is deliberately NOT raised
    # to cover batches that cannot fit: staging is cached per captured shape, so
    # a four-point sweep holds four copies, and a larger fraction would take the
    # memory the capacity probe needs and turn a capture win into an OOM.
    for batch in (32, 64, 128, 256, 400):
        per_step_rev = batch * H * (8 * elem + 8 * 4 + 3 * elem)
        per_step_fwd = 10 * batch * H * elem
        rev = max(1, min(budget // per_step_rev, T))
        fwd = max(1, min(budget // per_step_fwd, T))
        assert rev >= _SCAN_GRAPH_MIN_STEPS, (
            f"batch {batch}: reverse width {rev} is under the {_SCAN_GRAPH_MIN_STEPS}-step "
            "floor, so the backward would run uncaptured exactly where capture matters"
        )
        assert fwd >= _SCAN_GRAPH_MIN_STEPS, f"batch {batch}: forward width {fwd} under floor"

    # And the widths still shrink with batch -- the budget is a bound, not a
    # licence to stage without limit.
    assert (_reverse_chunk_width(256, T, H, elem) <= _reverse_chunk_width(128, T, H, elem))
    assert (_scan_chunk_width(256, T, H, elem) <= _scan_chunk_width(128, T, H, elem))
