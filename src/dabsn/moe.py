"""Native dropless sparse mixture-of-experts component."""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence, cast

import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .components import ComponentOutput
from .ops import stable_expert_permutation, weighted_scatter_add


class RouterOutput(NamedTuple):
    expert_indices: Tensor
    expert_weights: Tensor
    balance_loss: Tensor
    probabilities: Tensor


class DroplessDispatch(NamedTuple):
    sorted_inputs: Tensor
    sorted_experts: Tensor
    sorted_items: Tensor
    sorted_weights: Tensor
    counts: Tensor
    offsets: Tensor
    assignments: Tensor


class SwitchTopKRouter(nn.Module):
    """Top-k softmax router with an explicit Switch-style balance term."""

    def __init__(
        self,
        hidden_dim: int,
        experts: int,
        top_k: int,
        *,
        balance_coefficient: float = 1.0,
    ) -> None:
        super().__init__()
        if not 1 <= int(top_k) <= int(experts):
            raise ValueError("top_k must be between one and experts")
        self.hidden_dim = int(hidden_dim)
        self.experts = int(experts)
        self.top_k = int(top_k)
        if float(balance_coefficient) < 0:
            raise ValueError("balance_coefficient must be non-negative")
        self.balance_coefficient = float(balance_coefficient)
        self.proj = nn.Linear(self.hidden_dim, self.experts, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=self.hidden_dim**-0.5)

    def forward(self, inputs: Tensor) -> RouterOutput:
        logits = self.proj(inputs)
        probabilities = torch.softmax(logits.float(), dim=-1)
        selected_probabilities, indices = torch.topk(probabilities, self.top_k, dim=-1, sorted=True)
        weights = selected_probabilities / selected_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(selected_probabilities.dtype).tiny)
        assignment_share = F.one_hot(indices, self.experts).float().mean(dim=(0, 1))
        probability_share = probabilities.mean(dim=0)
        balance = (
            self.balance_coefficient
            * self.experts
            * torch.sum(assignment_share * probability_share)
        )
        return RouterOutput(indices, weights.to(inputs.dtype), balance, probabilities)


class AuxLossFreeTopKRouter(nn.Module):
    """Top-k router whose per-expert selection bias updates after real steps."""

    def __init__(
        self,
        hidden_dim: int,
        experts: int,
        top_k: int,
        *,
        bias_update_rate: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if not 1 <= int(top_k) <= int(experts):
            raise ValueError("top_k must be between one and experts")
        if float(bias_update_rate) <= 0:
            raise ValueError("bias_update_rate must be positive")
        self.hidden_dim = int(hidden_dim)
        self.experts = int(experts)
        self.top_k = int(top_k)
        self.bias_update_rate = float(bias_update_rate)
        self.proj = nn.Linear(self.hidden_dim, self.experts, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=self.hidden_dim**-0.5)
        self.selection_bias: Tensor
        self._pending_counts: Tensor
        self.register_buffer("selection_bias", torch.zeros(self.experts))
        self.register_buffer("_pending_counts", torch.zeros(self.experts))
        self.process_group: object | None = None

    def forward(self, inputs: Tensor) -> RouterOutput:
        logits = self.proj(inputs)
        probabilities = torch.softmax(logits.float(), dim=-1)
        selection_logits = logits.float() + self.selection_bias
        indices = torch.topk(selection_logits, self.top_k, dim=-1, sorted=True).indices
        selected = torch.gather(probabilities, 1, indices)
        weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(selected.dtype).tiny
        )
        counts = torch.bincount(indices.reshape(-1), minlength=self.experts).float()
        self._pending_counts.add_(counts.detach())
        # The router deliberately contributes no optimization loss. The fixed
        # zero keeps the graph's declared result arity identical across policies.
        balance = logits.new_zeros((), dtype=torch.float32)
        return RouterOutput(indices, weights.to(inputs.dtype), balance, probabilities)

    @torch.no_grad()
    def post_optimizer_step(self) -> None:
        if torch_dist.is_available() and torch_dist.is_initialized():
            torch_dist.all_reduce(
                self._pending_counts,
                op=torch_dist.ReduceOp.SUM,
                group=self.process_group,
            )
        total = self._pending_counts.sum()
        target = total / self.experts
        direction = torch.sign(target - self._pending_counts)
        self.selection_bias.add_(direction * self.bias_update_rate)
        self._pending_counts.zero_()


def dropless_dispatch(
    inputs: Tensor,
    expert_indices: Tensor,
    expert_weights: Tensor,
    experts: int,
) -> DroplessDispatch:
    """Create exactly ``N*K`` stable assignments and contiguous expert ranges."""

    if inputs.dim() != 2:
        raise ValueError("dropless dispatch inputs must be [N, H]")
    if expert_indices.shape != expert_weights.shape or expert_indices.dim() != 2:
        raise ValueError("router indices and weights must have the same [N, K] shape")
    if expert_indices.shape[0] != inputs.shape[0]:
        raise ValueError("router item count does not match inputs")
    n, k = expert_indices.shape
    assignments = torch.arange(n, device=inputs.device).repeat_interleave(k)
    flat_experts = expert_indices.reshape(-1)
    flat_weights = expert_weights.reshape(-1)
    order, counts, offsets = stable_expert_permutation(expert_indices, int(experts))
    sorted_experts = flat_experts.index_select(0, order)
    sorted_items = assignments.index_select(0, order)
    sorted_weights = flat_weights.index_select(0, order)
    sorted_inputs = inputs.index_select(0, sorted_items)
    if sorted_inputs.shape[0] != n * k:
        raise RuntimeError("dropless dispatch violated exact N*K assignment conservation")
    return DroplessDispatch(
        sorted_inputs,
        sorted_experts,
        sorted_items,
        sorted_weights,
        counts,
        offsets,
        assignments,
    )


class GenericExpertGroup(nn.Module):
    """Architecture-neutral execution of arbitrary same-shape expert modules."""

    def __init__(self, experts: Sequence[nn.Module]) -> None:
        super().__init__()
        if not experts:
            raise ValueError("GenericExpertGroup requires at least one expert")
        self.experts = nn.ModuleList(experts)

    def forward_dispatched(
        self, inputs: Tensor, expert_ids: Tensor, offsets: Tensor | None = None
    ) -> Tensor:
        output = torch.zeros_like(inputs)
        for expert_index, expert in enumerate(self.experts):
            positions = torch.nonzero(expert_ids == expert_index, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            values = expert(inputs.index_select(0, positions))
            if values.shape != inputs.index_select(0, positions).shape:
                raise ValueError(
                    f"expert {expert_index} changed routed item shape from "
                    f"{tuple(inputs.index_select(0, positions).shape)} to {tuple(values.shape)}"
                )
            output.index_copy_(0, positions, values)
        return output

    def __len__(self) -> int:
        return len(self.experts)


class ReLU2MLPExpertGroup(nn.Module):
    """Grouped ReLU-squared MLP experts with deterministic reference execution."""

    def __init__(
        self,
        experts: int,
        hidden_dim: int,
        inner_dim: int,
        *,
        zero_output: bool = False,
        accumulation_dtype: torch.dtype = torch.float32,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if min(int(experts), int(hidden_dim), int(inner_dim)) <= 0:
            raise ValueError("experts, hidden_dim, and inner_dim must be positive")
        self.experts = int(experts)
        self.hidden_dim = int(hidden_dim)
        self.inner_dim = int(inner_dim)
        self.accumulation_dtype = accumulation_dtype
        if backend not in {"auto", "reference", "grouped"}:
            raise ValueError("expert backend must be auto, reference, or grouped")
        self.backend = backend
        # Native grouped-MM layout is [expert, input, output].
        self.w1 = nn.Parameter(torch.empty(self.experts, self.hidden_dim, self.inner_dim))
        self.w2 = nn.Parameter(torch.empty(self.experts, self.inner_dim, self.hidden_dim))
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        if zero_output:
            nn.init.zeros_(self.w2)
        else:
            nn.init.normal_(self.w2, mean=0.0, std=0.02)

    def forward_expert(self, expert: int, inputs: Tensor) -> Tensor:
        compute = inputs.to(self.accumulation_dtype)
        hidden = compute @ self.w1[expert].to(self.accumulation_dtype)
        hidden = F.relu(hidden).square()
        output = hidden @ self.w2[expert].to(self.accumulation_dtype)
        return output.to(inputs.dtype)

    def _grouped_runnable(self, inputs: Tensor, offsets: Tensor | None) -> bool:
        """What the eager ``torch._grouped_mm`` kernel actually accepts."""
        return (
            offsets is not None
            and inputs.dtype in {torch.float32, torch.float16, torch.bfloat16}
            and (not inputs.is_cuda or inputs.dtype == torch.bfloat16)
            and inputs.stride(-2) * inputs.element_size() % 16 == 0
            and self.w1.stride(-2) * self.w1.element_size() % 16 == 0
            and self.w2.stride(-2) * self.w2.element_size() % 16 == 0
        )

    def _grouped_supported(self, inputs: Tensor, offsets: Tensor | None) -> bool:
        """What ``auto`` is allowed to choose on its own -- a stricter question.

        ``torch._grouped_mm`` has two contracts that disagree: the eager kernel
        takes FP32 on CPU, while its meta/fake kernel refuses anything but BF16
        ("Expected inputs of BF16 type"). So an FP32 MoE that ``auto`` sent down
        the grouped path runs fine right up until something propagates fake
        tensors through it -- conformance checks, meta-device construction,
        full-graph compilation -- and then dies. A silent default has to be
        right in every mode the component will be evaluated in, not just the one
        it was tried in, so ``auto`` agrees with the narrower contract.

        This gives up nothing where it counts: BF16 is the training dtype on the
        accelerators that have a grouped kernel worth using, so the fused path
        still fires there. An explicit ``backend="grouped"`` is a different
        statement -- the caller has taken responsibility for the mode they run
        in -- and is still held only to what the eager kernel accepts.
        """
        return self._grouped_runnable(inputs, offsets) and inputs.dtype == torch.bfloat16

    def forward_grouped(self, inputs: Tensor, offsets: Tensor) -> Tensor:
        ends = offsets[1:].to(dtype=torch.int32)
        first = torch._grouped_mm(
            inputs,
            self.w1,
            offs=ends,
            out_dtype=self.accumulation_dtype,
        )
        hidden = F.relu(first).square()
        output = torch._grouped_mm(
            hidden,
            self.w2,
            offs=ends,
            out_dtype=self.accumulation_dtype,
        )
        return output.to(inputs.dtype)

    def forward_dispatched(
        self, inputs: Tensor, expert_ids: Tensor, offsets: Tensor | None = None
    ) -> Tensor:
        grouped = self.backend == "grouped" or (
            self.backend == "auto" and self._grouped_supported(inputs, offsets)
        )
        if grouped:
            if offsets is None:
                raise ValueError("grouped expert execution requires expert offsets")
            if not self._grouped_runnable(inputs, offsets):
                raise RuntimeError(
                    "grouped expert execution is unsupported for this dtype/stride; "
                    "select backend='reference' explicitly"
                )
            return self.forward_grouped(inputs, offsets)
        output = torch.zeros_like(inputs)
        for expert in range(self.experts):
            positions = torch.nonzero(expert_ids == expert, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            values = self.forward_expert(expert, inputs.index_select(0, positions))
            output.index_copy_(0, positions, values)
        return output

    def __len__(self) -> int:
        return self.experts


class ExpertParallelExpertGroup(nn.Module):
    """Shard an ordinary expert group evenly and exchange assignments by rank.

    The wrapper is deliberately architecture-neutral: the local group may be a
    grouped MLP implementation, a generic collection, or another conforming
    expert group.  Every input assignment is returned in its original order.
    Variable all-to-all split sizes make this path incompatible with CUDA graph
    capture unless a separately proven static communication plan is supplied.
    """

    def __init__(
        self,
        local_group: GenericExpertGroup | ReLU2MLPExpertGroup,
        *,
        process_group=None,
        world_size: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        self.local_group = local_group
        if process_group is None and torch_dist.is_initialized():
            process_group = torch_dist.group.WORLD
        self.process_group = process_group
        if world_size is None:
            if not torch_dist.is_initialized():
                raise RuntimeError("expert parallelism requires an initialized process group")
            world_size = torch_dist.get_world_size(process_group)
        if rank is None:
            if int(world_size) == 1 and not torch_dist.is_initialized():
                rank = 0
            else:
                rank = torch_dist.get_rank(process_group)
        if int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
            raise ValueError("invalid expert-parallel rank/world size")
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.local_experts = len(local_group)
        self.experts = self.local_experts * self.world_size

    def __len__(self) -> int:
        return self.experts

    def forward_dispatched(
        self,
        inputs: Tensor,
        expert_ids: Tensor,
        offsets: Tensor | None = None,
    ) -> Tensor:
        if expert_ids.dim() != 1 or expert_ids.shape[0] != inputs.shape[0]:
            raise ValueError("expert-parallel IDs must be one per dispatched input")
        if inputs.numel() == 0:
            return torch.empty_like(inputs)
        torch._assert_async(
            ((expert_ids >= 0) & (expert_ids < self.experts)).all(),
            "expert-parallel ID is outside the global expert range",
        )
        if self.world_size == 1:
            local_order = torch.argsort(expert_ids, stable=True)
            local_inverse = torch.empty_like(local_order)
            local_inverse.scatter_(
                0,
                local_order,
                torch.arange(local_order.numel(), device=local_order.device),
            )
            sorted_ids = expert_ids.index_select(0, local_order)
            local_offsets = torch.cat(
                [
                    sorted_ids.new_zeros(1),
                    torch.bincount(sorted_ids, minlength=self.local_experts).cumsum(0),
                ]
            )
            sorted_outputs = self.local_group.forward_dispatched(
                inputs.index_select(0, local_order), sorted_ids, local_offsets
            )
            return sorted_outputs.index_select(0, local_inverse)

        from torch.distributed import _functional_collectives as collectives

        destination = torch.div(
            expert_ids,
            self.local_experts,
            rounding_mode="floor",
        )
        order = torch.argsort(destination, stable=True)
        inverse = torch.empty_like(order)
        inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device))
        send_inputs = inputs.index_select(0, order).contiguous()
        send_local_ids = torch.remainder(
            expert_ids.index_select(0, order), self.local_experts
        ).contiguous()
        send_counts = torch.bincount(destination, minlength=self.world_size).to(torch.int64)
        receive_counts = collectives.all_to_all_single(
            send_counts,
            None,
            None,
            self.process_group,
        )
        input_splits = [int(value) for value in send_counts.tolist()]
        output_splits = [int(value) for value in receive_counts.tolist()]
        received_inputs = collectives.all_to_all_single(
            send_inputs,
            output_splits,
            input_splits,
            self.process_group,
        )
        received_ids = collectives.all_to_all_single(
            send_local_ids,
            output_splits,
            input_splits,
            self.process_group,
        )

        local_order = torch.argsort(received_ids, stable=True)
        local_inverse = torch.empty_like(local_order)
        local_inverse.scatter_(
            0,
            local_order,
            torch.arange(local_order.numel(), device=local_order.device),
        )
        sorted_local_ids = received_ids.index_select(0, local_order)
        local_counts = torch.bincount(sorted_local_ids, minlength=self.local_experts)
        local_offsets = torch.cat([local_counts.new_zeros(1), local_counts.cumsum(0)])
        sorted_outputs = self.local_group.forward_dispatched(
            received_inputs.index_select(0, local_order),
            sorted_local_ids,
            local_offsets,
        )
        received_outputs = sorted_outputs.index_select(0, local_inverse)
        returned = collectives.all_to_all_single(
            received_outputs,
            input_splits,
            output_splits,
            self.process_group,
        )
        return returned.index_select(0, inverse)


ROUTER_REPORT_NAMES = (
    "expert_counts",
    "expert_shares",
    "balance_entropy",
    "cold_experts",
    "busiest_share",
    "quietest_share",
    "selected_confidence",
    "output_norm_differentiation",
    "assignment_count",
    "drop_count",
)


class SparseMoEComponent(nn.Module):
    """Dropless sparse MoE over independently routed complete H-worlds."""

    def __init__(
        self,
        hidden_dim: int,
        router: nn.Module,
        expert_group: GenericExpertGroup | ReLU2MLPExpertGroup,
        *,
        residual: bool = False,
        normalization: nn.Module | None = None,
        routing_granularity: str = "individual_h",
    ) -> None:
        super().__init__()
        if routing_granularity != "individual_h":
            raise ValueError(
                "the built-in MoE supports explicit individual_h routing; "
                "structure-native routing belongs to a provider declaring that contract"
            )
        self.hidden_dim = int(hidden_dim)
        self.router = router
        self.expert_group: GenericExpertGroup | ReLU2MLPExpertGroup | ExpertParallelExpertGroup = (
            expert_group
        )
        self.experts = len(expert_group)
        self.residual = bool(residual)
        self.normalization = normalization if normalization is not None else nn.Identity()
        self.routing_granularity = routing_granularity
        self._expert_parallel_gradient_hooks: list[object] = []
        if getattr(router, "experts", None) != self.experts:
            raise ValueError("router and expert group disagree on expert count")

    def _compute(self, value: Tensor) -> ComponentOutput:
        if value.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"SparseMoEComponent expected H={self.hidden_dim}, received {value.shape[-1]}"
            )
        original_shape = value.shape
        residual_items = value.reshape(-1, self.hidden_dim)
        items = self.normalization(residual_items)
        routed = self.router(items)
        dispatch = dropless_dispatch(
            items,
            routed.expert_indices,
            routed.expert_weights,
            self.experts,
        )
        expert_values = self.expert_group.forward_dispatched(
            dispatch.sorted_inputs, dispatch.sorted_experts, dispatch.offsets
        )
        combined = weighted_scatter_add(
            expert_values,
            dispatch.sorted_items,
            dispatch.sorted_weights,
            items.shape[0],
        )
        if self.residual:
            combined = residual_items + combined

        counts = dispatch.counts
        shares = counts.float() / counts.sum().clamp_min(1)
        nonzero = shares.clamp_min(torch.finfo(shares.dtype).tiny)
        entropy = (
            shares.new_ones(())
            if self.experts == 1
            else -(shares * nonzero.log()).sum() / math.log(self.experts)
        )
        cold = (counts == 0).sum()
        selected_confidence = torch.gather(routed.probabilities, 1, routed.expert_indices).mean()
        norms = expert_values.float().norm(dim=-1)
        norm_sums = torch.zeros(self.experts, device=value.device).index_add_(
            0, dispatch.sorted_experts, norms
        )
        norm_means = norm_sums / counts.clamp_min(1)
        differentiation = norm_means.std(unbiased=False) / norm_means.mean().clamp_min(
            torch.finfo(norm_means.dtype).tiny
        )
        reports = (
            counts,
            shares,
            entropy,
            cold,
            shares.max(),
            shares.min(),
            selected_confidence,
            differentiation,
            counts.sum(),
            counts.new_zeros(()),
        )
        return ComponentOutput(
            combined.reshape(original_shape),
            (routed.balance_loss,),
            reports,
            (),
        )

    def forward(self, value: Tensor) -> Tensor:
        return cast(Tensor, self._compute(value).value)

    def forward_with_terms(self, value: Tensor) -> ComponentOutput:
        return self._compute(value)

    def post_optimizer_step(self) -> None:
        action = getattr(self.router, "post_optimizer_step", None)
        if action is not None:
            action()


__all__ = [
    "AuxLossFreeTopKRouter",
    "DroplessDispatch",
    "ExpertParallelExpertGroup",
    "GenericExpertGroup",
    "ROUTER_REPORT_NAMES",
    "ReLU2MLPExpertGroup",
    "RouterOutput",
    "SparseMoEComponent",
    "SwitchTopKRouter",
    "dropless_dispatch",
]
