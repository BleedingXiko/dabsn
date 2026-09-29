"""Sparse row communication for Capacity, separate from dense reducers."""
from __future__ import annotations

import torch
from torch import nn
from contextlib import contextmanager


ROW_NAMES = ("key", "bias", "retention_logit", "value")


def capacity_modules(model, *, sparse_only=True):
    from .capacity import DABSNCapacitySubsystem
    return [module for module in model.modules()
            if isinstance(module, DABSNCapacitySubsystem) and (not sparse_only or module.mode == "sparse")]


@contextmanager
def exclude_capacity_parameters(model):
    saved = []
    for module in capacity_modules(model, sparse_only=False):
        for name in ((*ROW_NAMES, "gain") if module.mode == "sparse" else ("gain",)):
            saved.append((module, name, module._parameters.pop(name)))
    try:
        yield
    finally:
        for module, name, parameter in saved:
            module.register_parameter(name, parameter)


def register_replicated_capacity(model, group):
    """Exclude sparse row parameters from DDP's ordinary dense reducer."""
    rows = set()
    for module in capacity_modules(model):
        module._capacity_replica_group = group
        module._capacity_replicated = True
        rows.update(id(p) for p in module.sparse_parameters())
    ignored = set(getattr(model, "_ddp_params_and_buffers_to_ignore", ()))
    ignored.update(name for name, p in model.named_parameters() if id(p) in rows)
    model._ddp_params_and_buffers_to_ignore = ignored
    # DDP deliberately does not broadcast ignored tensors, so do it here.
    source = torch.distributed.get_global_rank(group, 0)
    for module in capacity_modules(model):
        for p in module.sparse_parameters():
            torch.distributed.broadcast(p.data, src=source, group=group)


def _union_rows(rows, group):
    world = torch.distributed.get_world_size(group)
    count = torch.tensor([rows.numel()], device=rows.device, dtype=torch.long)
    counts = [torch.empty_like(count) for _ in range(world)]
    torch.distributed.all_gather(counts, count, group=group)
    width = max(int(c.item()) for c in counts)
    if not width:
        return rows
    padded = torch.zeros(width, device=rows.device, dtype=torch.long)
    padded[:rows.numel()] = rows
    gathered = [torch.empty_like(padded) for _ in range(world)]
    torch.distributed.all_gather(gathered, padded, group=group)
    return torch.cat([r[:int(c.item())] for r, c in zip(gathered, counts)]).unique(sorted=True)


class _FetchRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, rows, partition, online):
        owned = (rows >= partition.start) & (rows < partition.stop)
        positions = owned.nonzero().flatten()
        local_rows = rows[owned] - partition.start
        result = parameter.new_zeros((rows.numel(), *parameter.shape[1:]))
        result[positions] = parameter[local_rows]
        torch.distributed.all_reduce(result, group=partition.group)
        ctx.save_for_backward(positions, local_rows)
        ctx.partition, ctx.shape, ctx.online = partition, parameter.shape, online
        ctx.parameter = parameter
        return result

    @staticmethod
    def backward(ctx, gradient):
        # Reduction is over the union of requested rows, never the full table.
        gradient = gradient.to_dense() if gradient.is_sparse else gradient.contiguous()
        torch.distributed.all_reduce(gradient, group=ctx.partition.group)
        gradient = gradient / ctx.partition.world
        positions, rows = ctx.saved_tensors
        values = gradient.index_select(0, positions)
        touched = getattr(ctx.parameter, "_capacity_touched_rows", None)
        if touched is not None:
            touched.index_fill_(0, rows, True)
        result = torch.sparse_coo_tensor(rows[None], values, ctx.shape,
                                        is_coalesced=True, check_invariants=False)
        return (result if ctx.online else result.to_dense()), None, None, None


class CapacityRowPartition:
    """Whole-row parameter ownership; recurrent state remains stream-local."""
    def __init__(self, size, group, axis):
        self.group, self.axis = group, axis
        self.world = torch.distributed.get_world_size(group)
        self.rank = torch.distributed.get_rank(group)
        width = (size + self.world - 1) // self.world
        self.sizes = tuple(max(0, min(width, size - r * width)) for r in range(self.world))
        self.start = sum(self.sizes[:self.rank])
        self.stop = self.start + self.sizes[self.rank]

    def _prepare(self, module, x, online):
        routes = module.route(x)
        rows = _union_rows(routes.unique(sorted=True), self.group)
        local = torch.searchsorted(rows, routes)
        parameters = [_FetchRows.apply(getattr(module, name), rows, self, online) for name in ROW_NAMES]
        return rows, local, parameters

    def forward_sequence(self, module, x, initial_features=None, return_state=False):
        from .kernels.capacity import capacity_backend, capacity_scan
        rows, local, parameters = self._prepare(module, x, False)
        state_dtype = torch.promote_types(x.dtype, torch.float32)
        features = (torch.zeros((x.shape[0], module.size), device=x.device, dtype=state_dtype)
                    if initial_features is None else initial_features.to(state_dtype))
        selected = features.index_select(1, rows)
        key, bias, logit, value = parameters
        backend = capacity_backend(x)
        module._last_capacity_backend = backend
        module._record_precision(x)
        if backend != "reference":
            output, final = capacity_scan(x, key, bias, logit, value, module.gain,
                                         selected, local, sparse=True)
        else:
            with torch.autocast(x.device.type, enabled=False):
                calc_x = x.to(state_dtype)
                outputs = []
                for t in range(x.shape[1]):
                    ids = local[:, t]
                    old = selected.gather(1, ids)
                    rho = logit[ids].to(state_dtype).sigmoid()
                    act = (torch.einsum("bd,bad->ba", calc_x[:, t], key[ids].to(state_dtype))
                           + bias[ids].to(state_dtype)).tanh()
                    new = rho * old + (1 - rho) * act
                    selected = selected.scatter(1, ids, new)
                    outputs.append(module.gain * torch.einsum("ba,bas->bs", new, value[ids].to(state_dtype)))
                output = (torch.stack(outputs, dim=1) if outputs else
                          calc_x.new_empty((x.shape[0], 0, module.state_dim))).to(x.dtype)
            final = selected
        final = features.index_copy(1, rows, final)
        return (output, final) if return_state else output

    def forward_online(self, module, x, recurrent, eligibility):
        from .capacity import _SparseOnlineCapacity
        from .kernels.capacity import capacity_backend, capacity_scan
        rows, local, parameters = self._prepare(module, x, True)
        features = recurrent["features"].index_select(1, rows)
        traces = [eligibility[name].index_select(1, rows) for name in ("key", "bias", "retention")]
        key, bias, logit, value = parameters
        backend = capacity_backend(x)
        module._last_capacity_backend = backend
        module._record_precision(x)
        if backend != "reference":
            output, final = capacity_scan(x, key, bias, logit, value, module.gain,
                                         features, local, eligibility=traces, sparse=True)
        else:
            with torch.autocast(x.device.type, enabled=False):
                calc_x = x.to(torch.promote_types(x.dtype, torch.float32))
                output, final = _SparseOnlineCapacity.apply(calc_x, key, bias, logit, value,
                                                           module.gain, features, *traces, local)
                output = output.to(x.dtype)
        recurrent["features"].index_copy_(1, rows, final)
        # This hook runs after the scan's analytic replay has updated its traces.
        if key.requires_grad:
            def carry_traces(gradient):
                for name, trace in zip(("key", "bias", "retention"), traces):
                    eligibility[name].index_copy_(1, rows, trace)
                return gradient
            key.register_hook(carry_traces)
        return output, dict(recurrent), dict(eligibility)


def shard_capacity_rows(module, group, *, axis="data"):
    if module.mode != "sparse":
        return None
    if getattr(module, "_row_partition", None) is not None:
        raise ValueError("Capacity rows are already partitioned")
    partition = CapacityRowPartition(module.size, group, axis)
    module._touched_rows = module._touched_rows[partition.start:partition.stop].clone()
    for name in ROW_NAMES:
        parameter = getattr(module, name)
        setattr(module, name, nn.Parameter(parameter.detach()[partition.start:partition.stop].clone(),
                                          requires_grad=parameter.requires_grad))
        getattr(module, name)._capacity_touched_rows = module._touched_rows
    module._row_partition = partition
    return partition


def gather_capacity_rows(tensor, partition):
    if tensor.shape[0] == sum(partition.sizes):
        return tensor
    width = max(partition.sizes)
    padded = tensor.new_zeros((width, *tensor.shape[1:]))
    padded[:tensor.shape[0]] = tensor
    gathered = [torch.empty_like(padded) for _ in partition.sizes]
    torch.distributed.all_gather(gathered, padded, group=partition.group)
    return torch.cat([value[:size] for value, size in zip(gathered, partition.sizes)])


def checkpoint_capacity_state(model, model_state, optimizer_state=None, *, full=False, restore=False):
    """Annotate whole rows for DCP resharding, or gather a portable full state."""
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard
    from .distributed import _normalize_distributed_state_key

    entries = {}
    provider_entries = {}
    parameters = {_normalize_distributed_state_key(name): parameter
                  for name, parameter in model.named_parameters()}
    for prefix, module in model.named_modules():
        for name, specification in getattr(module, "_dabsn_checkpoint_shards", {}).items():
            canonical = _normalize_distributed_state_key(f"{prefix}.{name}".lstrip("."))
            provider_entries[canonical] = specification
        partition = getattr(module, "_row_partition", None)
        if partition is None:
            continue
        for name in ROW_NAMES:
            entries[_normalize_distributed_state_key(f"{prefix}.{name}".lstrip("."))] = (module, partition)

    def convert(name, value):
        canonical = _normalize_distributed_state_key(name)
        record = entries.get(canonical)
        specification = provider_entries.get(canonical)
        if (record is None and specification is None) or not isinstance(value, torch.Tensor) or value.ndim == 0:
            return value
        if restore:
            local = value.to_local() if isinstance(value, DTensor) else value
            parameter = parameters.get(canonical)
            if isinstance(parameter, DTensor):
                from torch.distributed.tensor import distribute_tensor
                return distribute_tensor(local.to(parameter.device), parameter.device_mesh, parameter.placements)
            return local
        if specification is not None:
            parameter = parameters[canonical]
            # FSDP can own the data-axis subdivision of a provider's TP shard.
            # Reconstruct only that TP shard before annotating its global location.
            local = value.full_tensor() if isinstance(value, DTensor) else value
            mesh = DeviceMesh.from_group(specification["group"], parameter.device.type,
                                         mesh_dim_names=(specification["axis"],))
            shape = torch.Size(specification["shape"])
            stride = torch.empty(shape, device="meta").stride()
            annotated = DTensor.from_local(local.to(parameter.device), mesh,
                                            [Shard(specification["dim"])], shape=shape,
                                            stride=stride, run_check=False)
            return annotated.full_tensor().cpu() if full else annotated.cpu()
        module, partition = record
        if full:
            return gather_capacity_rows(value.to(module.key.device), partition).cpu()
        mesh = DeviceMesh.from_group(partition.group, module.key.device.type,
                                     mesh_dim_names=(partition.axis,))
        shape = (module.size, *value.shape[1:])
        stride = torch.empty(shape, device="meta").stride()
        return DTensor.from_local(value.to(module.key.device), mesh, [Shard(0)],
                                  shape=torch.Size(shape), stride=stride, run_check=False).cpu()

    if full:
        # FSDP full-state APIs can return an empty mapping on nonzero ranks.
        # Every rank must nevertheless participate in the ignored-row gather.
        for name, (module, partition) in entries.items():
            actual_name = next((key for key in model_state if _normalize_distributed_state_key(key) == name), None)
            value = getattr(module, name.rsplit(".", 1)[-1]).detach()
            complete = gather_capacity_rows(value, partition).cpu()
            if actual_name is not None:
                model_state[actual_name] = complete
    for name in list(model_state):
        if full:
            if optimizer_state is not None and _normalize_distributed_state_key(name) in provider_entries:
                model_state[name] = convert(name, model_state[name])
            continue
        model_state[name] = convert(name, model_state[name])
    if optimizer_state is not None:
        for name, states in optimizer_state.get("state", {}).items():
            for key in list(states):
                # SparseAdam's scalar step is an integer, and remains replicated.
                states[key] = convert(name, states[key])
    return model_state, optimizer_state


def reshardable_checkpoint_providers(model):
    """Advertise only DABSN block placements fully described by this runtime."""
    from .distributed import _graph_for_distributed
    graph = _graph_for_distributed(model)
    if graph is None:
        return []
    blocks = [binding for binding in graph.bindings if binding.provider_key == "dabsn:block"]
    if not blocks:
        return []
    for binding in blocks:
        receipt = binding.parallel_execution_receipt
        if receipt is None:
            continue
        described = set(getattr(binding.module, "_dabsn_checkpoint_shards", {}))
        for prefix, module in binding.module.named_modules():
            if getattr(module, "_row_partition", None) is not None:
                described.update(f"{prefix}.{name}".lstrip(".") for name in ROW_NAMES)
        sharded = {name for name, axes in receipt.parameter_sharding.items() if axes}
        if not sharded.issubset(described):
            return []
    return ["dabsn:block"]


def synchronize_capacity_gradients(model):
    """Synchronize registered replicated sparse rows once per gradient object."""
    from .runtime.optimizer import as_row_sparse

    for module in capacity_modules(model, sparse_only=False):
        parameter = module.gain
        if not getattr(module, "_capacity_gain_replicated", False):
            continue
        gradient = parameter.grad
        if gradient is not None and getattr(parameter, "_capacity_synced_gradient", None) is gradient:
            continue
        # The parameter may be unused locally but receive a gradient on a peer.
        gradient = torch.zeros_like(parameter) if gradient is None else gradient
        group = module._capacity_gain_replica_group
        torch.distributed.all_reduce(gradient, group=group)
        gradient.div_(torch.distributed.get_world_size(group))
        parameter.grad = gradient
        parameter._capacity_synced_gradient = gradient

    for module in model.modules():
        group = getattr(module, "_capacity_replica_group", None)
        if not getattr(module, "_capacity_replicated", False):
            continue
        for parameter in module.sparse_parameters():
            gradient = parameter.grad
            if gradient is not None and getattr(parameter, "_capacity_synced_gradient", None) is gradient:
                continue
            if gradient is None:
                gradient = torch.sparse_coo_tensor(
                    torch.empty((1, 0), dtype=torch.long, device=parameter.device),
                    parameter.new_empty((0, *parameter.shape[1:])), parameter.shape,
                )
            gradient = as_row_sparse(gradient, getattr(parameter, "_capacity_touched_rows", None))
            world = torch.distributed.get_world_size(group)
            count = torch.tensor([gradient._nnz()], dtype=torch.long, device=parameter.device)
            counts = [torch.empty_like(count) for _ in range(world)]
            torch.distributed.all_gather(counts, count, group=group)
            width = max(int(c.item()) for c in counts)
            if width:
                index = torch.zeros(width, dtype=torch.long, device=parameter.device)
                values = parameter.new_zeros((width, *parameter.shape[1:]))
                index[:gradient._nnz()] = gradient.indices()[0]
                values[:gradient._nnz()] = gradient.values()
                all_indices = [torch.empty_like(index) for _ in range(world)]
                all_values = [torch.empty_like(values) for _ in range(world)]
                torch.distributed.all_gather(all_indices, index, group=group)
                torch.distributed.all_gather(all_values, values, group=group)
                gradient = torch.sparse_coo_tensor(
                    torch.cat([v[:int(n.item())] for v, n in zip(all_indices, counts)]).unsqueeze(0),
                    torch.cat([v[:int(n.item())] for v, n in zip(all_values, counts)]) / world,
                    parameter.shape,
                ).coalesce()
            parameter.grad = gradient
            parameter._capacity_synced_gradient = gradient


def synchronize_capacity_overflow(model, optimizer, scaler):
    """Make AMP's skip decision identical across all Capacity row owners.

    GradScaler exposes no public cross-rank found-inf setter. Its per-optimizer
    flags are the same flags consumed by step(), including when callers already
    unscaled before clipping. ShardedGradScaler uses this representation too.
    """
    if (not torch.distributed.is_initialized() or not capacity_modules(model, sparse_only=False)
            or not isinstance(scaler, torch.amp.GradScaler) or not scaler.is_enabled()):
        return
    state = scaler._per_optimizer_states[id(optimizer)]
    if state["stage"].name == "READY":
        from .runtime.optimizer import unscale_training_gradients
        unscale_training_gradients(optimizer, scaler)
    flags = state["found_inf_per_device"]
    if not flags:
        return
    first = next(iter(flags.values()))
    consensus = torch.stack([flag.to(first.device) for flag in flags.values()]).amax()
    torch.distributed.all_reduce(consensus, op=torch.distributed.ReduceOp.MAX)
    for flag in flags.values():
        flag.copy_(consensus.to(flag.device))


__all__ = ["synchronize_capacity_gradients", "shard_capacity_rows"]
