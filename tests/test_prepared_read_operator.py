import pytest
import torch

from dabsn.kernels.prepared_admitted import prepared_admitted_page
from dabsn.prepared_memory import _prepared_page_merge


def _arguments():
    torch.manual_seed(822)
    batch, queries, capacity, hidden = 2, 3, 5, 7
    return (
        torch.randn(batch, queries, hidden), torch.randn(batch, queries, 4),
        torch.arange(queries, dtype=torch.int64).expand(batch, -1),
        torch.randn(batch, capacity, hidden), torch.randn(batch, capacity, hidden),
        torch.arange(capacity, dtype=torch.int64).expand(batch, -1),
        torch.randn(batch, capacity, 4), torch.rand(batch, capacity),
        torch.randn(batch, capacity), torch.randn(batch, capacity),
        torch.ones(batch, capacity, dtype=torch.bool), torch.tensor(1.), torch.tensor(.5),
        *(tensor for _ in range(3) for tensor in (torch.full((batch, queries), -torch.inf),
                                                 torch.zeros(batch, queries), torch.zeros(batch, queries, hidden))),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_prepared_page_registered_schema_fake_and_compile(dtype):
    args = _arguments()
    args = tuple(value.to(dtype) if value.is_floating_point() and index < 13 else value
                 for index, value in enumerate(args))
    expected = _prepared_page_merge(*args)
    for actual, value in zip(prepared_admitted_page(*args), expected):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    torch.library.opcheck(prepared_admitted_page, args)
    compiled = torch.compile(prepared_admitted_page, backend="aot_eager", fullgraph=True)
    for actual, value in zip(compiled(*args), expected):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)


def test_prepared_backend_accepts_more_than_eight_query_lengths():
    from dabsn.kernels.prepared_admitted import _cuda_prepared_page

    # Exercise the actual compiler policy with CPU tensors. This checks capture
    # and CPU lowering, not CUDA kernel execution.
    for length in range(1, 11):
        args = list(_arguments())
        args[0] = torch.randn(2, length, 7)
        args[1] = torch.randn(2, length, 4)
        args[2] = torch.arange(length, dtype=torch.int64).expand(2, -1)
        args[13:] = [tensor for _ in range(3) for tensor in (
            torch.full((2, length), -torch.inf), torch.zeros(2, length), torch.zeros(2, length, 7))]
        expected = _prepared_page_merge(*args)
        for actual, value in zip(_cuda_prepared_page(*args), expected):
            torch.testing.assert_close(actual, value, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_single_page_fused_read_matches_three_component_oracle(dtype):
    from dabsn.kernels.prepared_admitted import prepared_admitted_single_page

    args = list(_arguments())
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    for index in range(13):
        if args[index].is_floating_point():
            args[index] = args[index].to(dtype if index != 7 else compute)
    args[2] = torch.tensor([-1, 1], dtype=torch.int64)  # Empty, causal, and successor boundaries.
    args[10][1, 3:] = False
    positions = args[2][:, None] + torch.arange(3)[None, :]
    promoted = [value.to(compute) if value.is_floating_point() else value for value in args]
    promoted[2] = positions
    oracle = _prepared_page_merge(*promoted)
    gains = (torch.tensor(.7, dtype=dtype), torch.tensor(-.4, dtype=dtype), torch.tensor(1.2, dtype=dtype))
    expected = sum(gain.to(compute) * oracle[index * 3 + 2] /
                   oracle[index * 3 + 1].clamp_min(1e-12).unsqueeze(-1)
                   for index, gain in enumerate(gains)).to(dtype)
    call_args = (*args[:13], *gains)
    actual = prepared_admitted_single_page(*call_args)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[0, 0], torch.zeros_like(actual[0, 0]))
    torch.library.opcheck(prepared_admitted_single_page, call_args)
    compiled = torch.compile(prepared_admitted_single_page, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(*call_args), expected, atol=2e-6, rtol=2e-5)
