from __future__ import annotations

import torch

from dabsn.kernels.admitted import _frontpack_admitted_bank


def _inputs(
    bank_valid: torch.Tensor,
    *,
    requires_grad: bool = False,
    noncontiguous: bool = False,
) -> tuple[torch.Tensor, ...]:
    batch, entries = bank_valid.shape

    def vector(width: int) -> torch.Tensor:
        if noncontiguous:
            tensor = torch.randn(batch, entries * 2, width)[:, ::2, :]
        else:
            tensor = torch.randn(batch, entries, width)
        return tensor.requires_grad_(requires_grad)

    def scalar() -> torch.Tensor:
        if noncontiguous:
            tensor = torch.randn(batch, entries * 2)[:, ::2]
        else:
            tensor = torch.randn(batch, entries)
        return tensor.requires_grad_(requires_grad)

    if noncontiguous:
        bank_idx = torch.arange(entries * 2).view(1, -1).expand(batch, -1)[:, ::2]
        valid_source = torch.zeros(batch, entries * 2, dtype=torch.bool)
        valid_source[:, ::2] = bank_valid
        bank_valid = valid_source[:, ::2]
    else:
        bank_idx = torch.arange(entries).view(1, -1).expand(batch, -1).contiguous()
        bank_valid = bank_valid.contiguous()
    return (
        vector(7),
        vector(7),
        vector(7),
        vector(4),
        scalar(),
        scalar(),
        bank_idx,
        bank_valid,
    )


def _reference_frontpack(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    memory_key, write_memory, next_write_memory, memory_cocktail = inputs[:4]
    key_bias, admission_gate, bank_idx, bank_valid = inputs[4:]
    order = torch.argsort(
        bank_valid.detach().to(torch.int32),
        dim=1,
        descending=True,
        stable=True,
    )
    idx = order.unsqueeze(-1)
    return (
        torch.gather(memory_key, 1, idx.expand(-1, -1, memory_key.shape[-1])),
        torch.gather(write_memory, 1, idx.expand(-1, -1, write_memory.shape[-1])),
        torch.gather(
            next_write_memory,
            1,
            idx.expand(-1, -1, next_write_memory.shape[-1]),
        ),
        torch.gather(
            memory_cocktail,
            1,
            idx.expand(-1, -1, memory_cocktail.shape[-1]),
        ),
        torch.gather(key_bias, 1, order),
        torch.gather(admission_gate, 1, order),
        torch.gather(bank_idx, 1, order),
        torch.gather(bank_valid, 1, order),
    )


def _assert_equal(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> None:
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=0.0, atol=0.0)


def test_prefix_packed_bank_returns_original_tensors_without_gathers():
    valid = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    inputs = _inputs(valid)
    expected = _reference_frontpack(inputs)
    actual = _frontpack_admitted_bank(*inputs)

    _assert_equal(actual, expected)
    assert all(observed is source for observed, source in zip(actual, inputs, strict=True))


def test_all_valid_and_all_invalid_banks_are_identity_orders():
    for valid in (
        torch.ones((1, 9), dtype=torch.bool),
        torch.zeros((1, 9), dtype=torch.bool),
    ):
        inputs = _inputs(valid)
        actual = _frontpack_admitted_bank(*inputs)
        assert all(observed is source for observed, source in zip(actual, inputs, strict=True))


def test_nonprefix_mask_preserves_original_sort_and_gather_path():
    valid = torch.tensor([[True, False, True, False, True]])
    inputs = _inputs(valid)
    expected = _reference_frontpack(inputs)
    actual = _frontpack_admitted_bank(*inputs)

    _assert_equal(actual, expected)
    assert any(observed is not source for observed, source in zip(actual, inputs, strict=True))


def test_noncontiguous_inputs_preserve_original_gather_path():
    valid = torch.tensor([[True, True, True, False, False]])
    inputs = _inputs(valid, noncontiguous=True)
    assert any(not tensor.is_contiguous() for tensor in inputs)
    expected = _reference_frontpack(inputs)
    actual = _frontpack_admitted_bank(*inputs)

    _assert_equal(actual, expected)
    assert any(observed is not source for observed, source in zip(actual, inputs, strict=True))


def test_prefix_elision_preserves_gradients_of_identity_gather():
    valid = torch.tensor([[True, True, True, True, False]])
    fast_inputs = _inputs(valid, requires_grad=True)
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(tensor.requires_grad) for tensor in fast_inputs
    )

    fast_outputs = _frontpack_admitted_bank(*fast_inputs)
    reference_outputs = _reference_frontpack(reference_inputs)
    fast_loss = sum((index + 1) * tensor.square().sum() for index, tensor in enumerate(fast_outputs[:6]))
    reference_loss = sum(
        (index + 1) * tensor.square().sum()
        for index, tensor in enumerate(reference_outputs[:6])
    )
    fast_loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(fast_loss, reference_loss, rtol=0.0, atol=0.0)
    for fast, reference in zip(fast_inputs[:6], reference_inputs[:6], strict=True):
        torch.testing.assert_close(fast.grad, reference.grad, rtol=0.0, atol=0.0)


def test_compilation_keeps_value_oblivious_gather_path(monkeypatch):
    valid = torch.tensor([[True, True, True, False, False]])
    inputs = _inputs(valid)
    expected = _reference_frontpack(inputs)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    actual = _frontpack_admitted_bank(*inputs)

    _assert_equal(actual, expected)
    assert any(observed is not source for observed, source in zip(actual, inputs, strict=True))


def test_fake_tensor_keeps_value_oblivious_gather_path():
    from torch._subclasses.fake_tensor import FakeTensorMode

    valid = torch.tensor([[True, True, True, False, False]])
    real_inputs = _inputs(valid)
    with FakeTensorMode() as mode:
        inputs = tuple(mode.from_tensor(tensor) for tensor in real_inputs)
        actual = _frontpack_admitted_bank(*inputs)

    assert any(observed is not source for observed, source in zip(actual, inputs, strict=True))
