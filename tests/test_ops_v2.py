import torch

from dabsn import stable_expert_permutation, weighted_scatter_add


def test_registered_operator_names_and_schemas_are_stable():
    assert torch.ops.dabsn.stable_expert_permutation.default
    assert torch.ops.dabsn.weighted_scatter_add.default
    assert "Tensor" in str(torch.ops.dabsn.weighted_scatter_add.default._schema)


def test_stable_permutation_opcheck_and_reference():
    indices = torch.tensor([[2, 0], [1, 2], [0, 1]], dtype=torch.int64)
    order, counts, offsets = stable_expert_permutation(indices, 3)
    flat = indices.flatten()
    assert flat[order].tolist() == sorted(flat.tolist())
    assert counts.tolist() == [2, 2, 2]
    assert offsets.tolist() == [0, 2, 4, 6]
    result = torch.library.opcheck(
        stable_expert_permutation,
        (indices, 3),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert all(value == "SUCCESS" for value in result.values())
    functional = torch.func.functionalize(stable_expert_permutation)(indices, 3)
    for observed, reference in zip(functional, (order, counts, offsets)):
        torch.testing.assert_close(observed, reference)


def test_weighted_scatter_opcheck_gradcheck_and_reference():
    values = torch.randn(8, 5, dtype=torch.double, requires_grad=True)
    items = torch.tensor([0, 2, 1, 0, 2, 1, 1, 0])
    weights = torch.randn(8, dtype=torch.double, requires_grad=True)
    expected = torch.zeros(3, 5, dtype=torch.double).index_add(
        0, items, values * weights.unsqueeze(-1)
    )
    actual = weighted_scatter_add(values, items, weights, 3)
    torch.testing.assert_close(actual, expected)
    result = torch.library.opcheck(
        weighted_scatter_add,
        (values, items, weights, 3),
    )
    assert all(value == "SUCCESS" for value in result.values())
    assert torch.autograd.gradcheck(
        weighted_scatter_add, (values, items, weights, 3), fast_mode=True
    )
    assert torch.autograd.gradgradcheck(
        weighted_scatter_add, (values, items, weights, 3), fast_mode=True
    )
    torch.testing.assert_close(
        torch.func.functionalize(weighted_scatter_add)(values, items, weights, 3),
        actual,
    )


def test_weighted_scatter_vmap_matches_loop():
    values = torch.randn(4, 8, 5)
    items = torch.randint(0, 3, (4, 8))
    weights = torch.randn(4, 8)
    actual = torch.vmap(weighted_scatter_add, in_dims=(0, 0, 0, None))(values, items, weights, 3)
    expected = torch.stack(
        [weighted_scatter_add(values[i], items[i], weights[i], 3) for i in range(4)]
    )
    torch.testing.assert_close(actual, expected)


def test_weighted_scatter_compile_fullgraph():
    compiled = torch.compile(weighted_scatter_add, backend="aot_eager", fullgraph=True)
    values = torch.randn(8, 5, requires_grad=True)
    items = torch.randint(0, 3, (8,))
    weights = torch.randn(8, requires_grad=True)
    output = compiled(values, items, weights, 3)
    output.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
