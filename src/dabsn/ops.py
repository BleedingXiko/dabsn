"""Registered PyTorch operator ABI used by DABSN 2.x components."""

from __future__ import annotations

import torch
from torch import Tensor


@torch.library.custom_op("dabsn::stable_expert_permutation", mutates_args=())
def stable_expert_permutation(
    expert_indices: Tensor, experts: int
) -> tuple[Tensor, Tensor, Tensor]:
    flat = expert_indices.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sorted_experts = flat.index_select(0, order)
    counts = torch.bincount(sorted_experts, minlength=int(experts))
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    return order, counts, offsets


@stable_expert_permutation.register_fake
def _stable_expert_permutation_fake(expert_indices: Tensor, experts: int):
    assignments = expert_indices.numel()
    order = expert_indices.new_empty((assignments,))
    counts = expert_indices.new_empty((experts,))
    offsets = expert_indices.new_empty((experts + 1,))
    return order, counts, offsets


@torch.library.register_vmap(stable_expert_permutation)
def _stable_expert_permutation_vmap(info, in_dims, expert_indices, experts):
    indices_dim, experts_dim = in_dims
    if experts_dim is not None:
        raise ValueError("experts is a scalar and cannot have a vmap dimension")
    if indices_dim is None:
        output = stable_expert_permutation(expert_indices, experts)
        return output, (None, None, None)
    values = expert_indices.movedim(indices_dim, 0)
    outputs = [stable_expert_permutation(value, experts) for value in values]
    return tuple(torch.stack(items) for items in zip(*outputs)), (0, 0, 0)


@torch.library.custom_op("dabsn::weighted_scatter_add", mutates_args=())
def weighted_scatter_add(
    values: Tensor, item_indices: Tensor, weights: Tensor, output_items: int
) -> Tensor:
    output = values.new_zeros((int(output_items), values.shape[-1]))
    return output.index_add(0, item_indices, values * weights.unsqueeze(-1))


@weighted_scatter_add.register_fake
def _weighted_scatter_add_fake(
    values: Tensor, item_indices: Tensor, weights: Tensor, output_items: int
):
    return values.new_empty((output_items, values.shape[-1]))


def _weighted_scatter_setup(ctx, inputs, output):
    values, item_indices, weights, output_items = inputs
    ctx.save_for_backward(values, item_indices, weights)


def _weighted_scatter_backward(ctx, grad_output):
    values, item_indices, weights = ctx.saved_tensors
    selected = grad_output.index_select(0, item_indices)
    grad_values = selected * weights.unsqueeze(-1)
    grad_weights = (selected * values).sum(dim=-1)
    return grad_values, None, grad_weights, None


torch.library.register_autograd(
    weighted_scatter_add,
    _weighted_scatter_backward,
    setup_context=_weighted_scatter_setup,
)


@torch.library.register_vmap(weighted_scatter_add)
def _weighted_scatter_add_vmap(info, in_dims, values, item_indices, weights, output_items):
    values_dim, items_dim, weights_dim, output_items_dim = in_dims
    if output_items_dim is not None:
        raise ValueError("output_items is a scalar and cannot have a vmap dimension")
    batch_size = info.batch_size

    def batched(value, dim, index):
        if dim is None:
            return value
        return value.movedim(dim, 0)[index]

    outputs = [
        weighted_scatter_add(
            batched(values, values_dim, index),
            batched(item_indices, items_dim, index),
            batched(weights, weights_dim, index),
            output_items,
        )
        for index in range(batch_size)
    ]
    return torch.stack(outputs), 0


# Autocast policy is explicit: preserve the active low-precision compute lane.
torch.library.register_autocast(weighted_scatter_add, "cuda", torch.float16)
torch.library.register_autocast(weighted_scatter_add, "cpu", torch.bfloat16)


__all__ = ["stable_expert_permutation", "weighted_scatter_add"]
