import copy

import pytest
import torch

from dabsn import DABSNCore
from dabsn.kernels.batched_runtime import dabsn_core_scan_batched
from dabsn.read import DABSNRead


def _batched(core, inputs):
    hidden = core.hidden_dim
    return dabsn_core_scan_batched(
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
