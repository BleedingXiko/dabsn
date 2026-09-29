"""AdamW with a native SparseAdam boundary for Capacity row parameters."""
from __future__ import annotations

import weakref
from collections.abc import Iterable

import torch
from torch import nn


def unscale_training_gradients(optimizer, scaler):
    """Unscale local gradient storage when FSDP2 mixes DTensors and plain rows.

    GradScaler still owns its normal per-optimizer state and overflow flags.
    Temporary tensor views expose the same local storage to its public unscale
    method; the optimizer's real parameter groups are restored before stepping.
    """
    from torch.distributed.tensor import DTensor
    saved = []
    meshes = {}
    try:
        for group in optimizer.param_groups:
            parameters = group["params"]
            if not any(isinstance(p, DTensor) for p in parameters):
                continue
            local = []
            for parameter in parameters:
                if isinstance(parameter, DTensor):
                    meshes[id(parameter.device_mesh)] = parameter.device_mesh
                    view = parameter.to_local().detach()
                    gradient = parameter.grad
                    if gradient is not None:
                        view.grad = gradient.to_local() if isinstance(gradient, DTensor) else gradient
                    local.append(view)
                else:
                    local.append(parameter)
            saved.append((group, parameters))
            group["params"] = local
        scaler.unscale_(optimizer)
        if meshes and isinstance(scaler, torch.amp.GradScaler) and scaler.is_enabled():
            flags = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
            groups = set()
            for mesh in meshes.values():
                for dimension in range(mesh.ndim):
                    group = mesh.get_group(dimension)
                    if group in groups:
                        continue
                    groups.add(group)
                    for flag in flags.values():
                        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=group)
    finally:
        for group, parameters in saved:
            group["params"] = parameters


def capacity_sparse_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    from ..capacity import DABSNCapacitySubsystem

    seen = set()
    result = []
    for module in model.modules():
        if isinstance(module, DABSNCapacitySubsystem):
            for parameter in module.sparse_parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    result.append(parameter)
    return tuple(result)


def as_row_sparse(gradient: torch.Tensor, touched_rows=None) -> torch.Tensor:
    """Coalesce COO rows or explicitly convert the ordinary BPTT dense result."""
    if gradient.is_sparse:
        result = gradient.coalesce()
        if result.sparse_dim() != 1:
            raise ValueError("Capacity requires one sparse row dimension")
        return result
    present = gradient.flatten(start_dim=1).ne(0).any(dim=1) if gradient.ndim > 1 else gradient.ne(0)
    if touched_rows is not None:
        present = present | touched_rows
    rows = present.nonzero().flatten()
    return torch.sparse_coo_tensor(
        rows.unsqueeze(0), gradient.index_select(0, rows), gradient.shape,
        device=gradient.device, dtype=gradient.dtype, is_coalesced=True, check_invariants=False,
    )


class CapacityTrainingOptimizer(torch.optim.Optimizer):
    """One optimizer interface, with disjoint SparseAdam and AdamW groups.

    The standard optimizer state/param_groups schema lets schedulers, AMP and
    distributed checkpointing see one atomic update. Children share the actual
    group dictionaries and state mapping; there is no duplicated optimizer state.
    """

    def __init__(self, model, groups, *, adamw_options):
        super().__init__(groups, defaults={})
        self._model = weakref.ref(model)
        self._adamw_options = dict(adamw_options)
        self._bind_children()

    def _bind_children(self):
        dense = [g for g in self.param_groups if g["capacity_optimizer"] == "adamw"]
        sparse = [g for g in self.param_groups if g["capacity_optimizer"] == "sparse_adam"]
        self.dense_optimizer = torch.optim.AdamW(dense, **self._adamw_options) if dense else None
        self.sparse_optimizer = torch.optim.SparseAdam(sparse) if sparse else None
        for child in (self.dense_optimizer, self.sparse_optimizer):
            if child is not None:
                child.state = self.state

    def load_state_dict(self, state_dict):
        tags = [g.get("capacity_optimizer") for g in state_dict["param_groups"]]
        if tags != [g["capacity_optimizer"] for g in self.param_groups]:
            raise ValueError("Capacity optimizer checkpoint group ownership mismatch")
        super().load_state_dict(state_dict)
        self._bind_children()

    @torch.no_grad()
    def prepare_gradients(self):
        from ..kernels.capacity import finalize_capacity_gradients
        finalize_capacity_gradients(p for group in self.param_groups for p in group["params"])
        for group in self.param_groups:
            if group["capacity_optimizer"] == "sparse_adam":
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad = as_row_sparse(parameter.grad, getattr(parameter, "_capacity_touched_rows", None))
        model = self._model()
        if model is not None:
            from ..capacity_distributed import synchronize_capacity_gradients
            synchronize_capacity_gradients(model)

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter.__dict__.pop("_capacity_synced_gradient", None)
                touched = getattr(parameter, "_capacity_touched_rows", None)
                if touched is not None:
                    touched.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.prepare_gradients()
        for child in (self.dense_optimizer, self.sparse_optimizer):
            if child is not None:
                child.step()
        for group in self.param_groups:
            for parameter in group["params"]:
                touched = getattr(parameter, "_capacity_touched_rows", None)
                if touched is not None:
                    touched.zero_()
        return loss


def build_training_optimizer(
    model: nn.Module,
    params: Iterable | None = None,
    **adamw_options,
) -> torch.optim.Optimizer:
    """Build ordinary AdamW, or SparseAdam rows plus AdamW for other parameters.

    Caller options/groups are retained. SparseAdam does not apply weight decay;
    its groups inherit lr, betas, eps and maximize. Options particular to AdamW
    (such as fused/foreach) apply only to the dense parameter groups.
    """
    parameters = model.parameters() if params is None else params
    sparse_ids = {id(p) for p in capacity_sparse_parameters(model)}
    if not sparse_ids:
        optimizer = torch.optim.AdamW(parameters, **adamw_options)
        from ..capacity_distributed import capacity_modules, synchronize_capacity_gradients
        if any(getattr(cap, "_capacity_gain_replicated", False)
               for cap in capacity_modules(model, sparse_only=False)):
            model_ref = weakref.ref(model)
            def prepare_gradients():
                current = model_ref()
                if current is not None:
                    synchronize_capacity_gradients(current)
            optimizer.prepare_gradients = prepare_gradients
            optimizer.register_step_pre_hook(lambda *_: prepare_gradients())
            def clear_synchronization(*_):
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        p.__dict__.pop("_capacity_synced_gradient", None)
            optimizer.register_step_post_hook(clear_synchronization)
        return optimizer
    supplied = list(parameters)
    if not supplied or not isinstance(supplied[0], dict):
        supplied = [{"params": supplied}]
    # Let AdamW validate options and populate its established defaults first.
    template = torch.optim.AdamW(supplied, **adamw_options)
    groups = []
    for group in template.param_groups:
        rows = [p for p in group["params"] if id(p) in sparse_ids]
        dense = [p for p in group["params"] if id(p) not in sparse_ids]
        if dense:
            groups.append({**group, "params": dense, "capacity_optimizer": "adamw"})
        if rows:
            sparse_group = {k: group[k] for k in ("lr", "betas", "eps", "maximize")}
            sparse_group.update(params=rows, weight_decay=0.0, capacity_optimizer="sparse_adam")
            groups.append(sparse_group)
    return CapacityTrainingOptimizer(model, groups, adamw_options=adamw_options)


__all__ = ["build_training_optimizer", "CapacityTrainingOptimizer"]
