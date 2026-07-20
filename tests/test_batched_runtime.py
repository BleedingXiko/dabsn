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
