"""Distributed DABSN training with DDP and FSDP parameter sharding."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, Sequence, TypedDict, cast

import torch
import torch.distributed as torch_dist
import torch.nn as nn
from torch import Tensor

from .checkpoint import (
    dabsn_config_dict,
    graph_config_dict,
    inspect_dabsn,
    save_dabsn_state,
)
from .components import (
    ComponentOutput,
    ParallelAxis,
    ParallelExecutionContext,
    ParallelExecutionReceipt,
    ParallelPlan,
    ParallelTopology,
    ResultDeclaration,
)
from .core import DABSNCore
from .events import EventCode, emit_event
from .graph import DABSNGraph
from .model import DABSNModel, DABSNSequenceLM, DABSNTaskModel

DABSNModelType = DABSNModel | DABSNTaskModel | DABSNSequenceLM
DABSNArtifactType = DABSNModelType | DABSNGraph


def parse_parallel_topology(declarations: Sequence[str] | None) -> ParallelTopology | None:
    """Parse repeatable ``name=size`` declarations without assigning axis semantics."""

    if not declarations:
        return None
    axes = []
    for declaration in declarations:
        if not isinstance(declaration, str) or declaration.count("=") != 1:
            raise ValueError("parallel axes must use the explicit name=size form")
        name, raw_size = (part.strip() for part in declaration.split("=", 1))
        if not name or not raw_size:
            raise ValueError("parallel axes must use the explicit name=size form")
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise ValueError(f"parallel-axis {name!r} size must be an integer") from exc
        axes.append(ParallelAxis(name, size))
    return ParallelTopology(tuple(axes))


class StatefulTrainingObject(Protocol):
    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class DistributedState:
    """The process-group and device assignment for one DABSN worker."""

    kind: str = "none"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    backend: str | None = None
    topology: ParallelTopology | None = None
    parallel_context: ParallelExecutionContext | None = None

    def __post_init__(self) -> None:
        topology = self.topology or ParallelTopology((ParallelAxis("data", self.world_size),))
        if topology.world_size != self.world_size:
            raise ValueError(
                f"parallel topology has {topology.world_size} workers, "
                f"distributed state has {self.world_size}"
            )
        object.__setattr__(self, "topology", topology)
        if self.parallel_context is not None:
            if self.parallel_context.topology != topology:
                raise ValueError("parallel execution context and distributed topology disagree")
            if self.parallel_context.rank != self.rank:
                raise ValueError("parallel execution context and distributed rank disagree")

    @property
    def enabled(self) -> bool:
        return self.kind != "none"

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def parameter_sharded(self) -> bool:
        return self.kind in {"fsdp", "fsdp2"}

    @property
    def gradient_sharded(self) -> bool:
        return self.kind in {"fsdp", "fsdp2"}

    @property
    def optimizer_sharded(self) -> bool:
        return self.kind in {"fsdp", "fsdp2"}

    @property
    def batch_parallel(self) -> bool:
        return self.kind in {"ddp", "fsdp", "fsdp2"} and self.data_world_size > 1

    @property
    def coordinate(self) -> Mapping[str, int]:
        assert self.topology is not None
        return self.topology.coordinate(self.rank)

    @property
    def data_world_size(self) -> int:
        assert self.topology is not None
        try:
            return self.topology.axis_size("data")
        except KeyError:
            return 1

    @property
    def data_rank(self) -> int:
        return int(self.coordinate.get("data", 0))

    @property
    def data_group(self):
        if self.parallel_context is None:
            return None
        return self.parallel_context.groups.get("data")

    def report(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "device": str(self.device),
            "backend": self.backend,
            "parameter_sharded": self.parameter_sharded,
            "gradient_sharded": self.gradient_sharded,
            "optimizer_sharded": self.optimizer_sharded,
            "batch_parallel": self.batch_parallel,
            "context_parallel": False,
            "topology": self.topology.to_dict() if self.topology is not None else None,
            "topology_fingerprint": (
                self.topology.fingerprint() if self.topology is not None else None
            ),
            "coordinate": dict(self.coordinate),
        }


def _axis_rank_groups(topology: ParallelTopology, axis_name: str) -> tuple[tuple[int, ...], ...]:
    names = tuple(axis.name for axis in topology.axes)
    axis_index = names.index(axis_name)
    fixed_axes = [axis for index, axis in enumerate(topology.axes) if index != axis_index]
    groups = []
    for fixed_values in product(*(range(axis.size) for axis in fixed_axes)):
        fixed = dict(zip((axis.name for axis in fixed_axes), fixed_values))
        ranks = tuple(
            rank
            for rank in range(topology.world_size)
            if all(topology.coordinate(rank)[name] == value for name, value in fixed.items())
        )
        groups.append(ranks)
    return tuple(sorted(groups))


def _build_parallel_execution_context(
    topology: ParallelTopology,
    *,
    rank: int,
    device: torch.device,
) -> ParallelExecutionContext:
    """Create every mesh-axis group in identical global order on every worker."""

    if topology.world_size <= rank:
        raise ValueError(f"rank {rank} is outside topology world size {topology.world_size}")
    resolved_groups: dict[str, object] = {}
    resolved_ranks: dict[str, tuple[int, ...]] = {}
    distributed = torch_dist.is_available() and torch_dist.is_initialized()
    for axis in topology.axes:
        for ranks in _axis_rank_groups(topology, axis.name):
            if distributed and len(ranks) == topology.world_size:
                group: object = torch_dist.group.WORLD
            elif distributed:
                group = torch_dist.new_group(ranks=list(ranks))
            else:
                group = None
            if rank in ranks:
                resolved_groups[axis.name] = group
                resolved_ranks[axis.name] = ranks
    return ParallelExecutionContext(
        topology=topology,
        rank=rank,
        device=device,
        coordinate=topology.coordinate(rank),
        groups=MappingProxyType(resolved_groups),
        group_ranks=MappingProxyType(resolved_ranks),
    )


class DABSNSequenceModule(nn.Module):
    """Expose full sequence outputs through ``forward`` for DDP and FSDP."""

    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        self.body = body

    @property
    def backbone(self):
        return getattr(self.body, "backbone", None)

    def forward(
        self,
        inputs: Tensor,
        positions: Tensor | None = None,
        *,
        with_terms: bool = False,
    ) -> Tensor | ComponentOutput:
        if with_terms:
            if positions is not None:
                raise ValueError("positions and with_terms are mutually exclusive")
            forward_with_terms = getattr(self.body, "forward_with_terms", None)
            if not callable(forward_with_terms):
                raise TypeError("prepared module does not implement forward_with_terms")
            return forward_with_terms(inputs)
        if positions is None:
            forward_sequence = getattr(self.body, "forward_sequence", None)
            return forward_sequence(inputs) if callable(forward_sequence) else self.body(inputs)
        forward_hidden = getattr(self.body, "forward_hidden", None)
        project_positions = getattr(self.body, "project_positions", None)
        if not callable(forward_hidden) or not callable(project_positions):
            raise TypeError("prepared module does not implement position-selective inference")
        hidden = forward_hidden(inputs)
        return project_positions(hidden, positions)

    def post_optimizer_step(self, *, step_applied: bool) -> None:
        action = getattr(self.body, "post_optimizer_step", None)
        if callable(action):
            action(step_applied=step_applied)


def _resolve_device(requested: str | torch.device | None) -> torch.device:
    if requested is None or str(requested) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def setup_distributed(
    kind: str = "none",
    requested_device: str | torch.device | None = "auto",
    *,
    backend: str | None = None,
    topology: ParallelTopology | None = None,
) -> DistributedState:
    """Initialize a ``torchrun`` worker from its standard environment variables."""

    normalized = (kind or "none").lower()
    if normalized not in {"none", "ddp", "fsdp", "fsdp2"}:
        raise ValueError("distributed kind must be none, ddp, fsdp, or fsdp2")
    requested = _resolve_device(requested_device)
    if normalized == "none":
        resolved_topology = topology or ParallelTopology((ParallelAxis("data", 1),))
        context = _build_parallel_execution_context(
            resolved_topology,
            rank=0,
            device=requested,
        )
        return DistributedState(
            device=requested,
            topology=resolved_topology,
            parallel_context=context,
        )
    if not torch_dist.is_available():
        raise RuntimeError("this PyTorch build does not provide torch.distributed")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError(
            f"distributed={normalized} requires torchrun with at least two workers; "
            "use torchrun --nproc-per-node=2 ..."
        )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA distributed execution was requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        selected_backend = backend or "nccl"
    else:
        if normalized in {"fsdp", "fsdp2"}:
            raise RuntimeError("DABSN FSDP requires CUDA; CPU distributed execution may use DDP")
        device = requested
        selected_backend = backend or "gloo"
    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend=selected_backend)
    resolved_topology = topology or ParallelTopology((ParallelAxis("data", world_size),))
    if resolved_topology.world_size != world_size:
        raise ValueError(
            f"parallel topology requires {resolved_topology.world_size} workers, "
            f"torchrun launched {world_size}"
        )
    context = _build_parallel_execution_context(
        resolved_topology,
        rank=rank,
        device=device,
    )
    return DistributedState(
        kind=normalized,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=selected_backend,
        topology=resolved_topology,
        parallel_context=context,
    )


def resolve_precision(precision: str | None, device: torch.device) -> str:
    """Resolve ``auto`` to bf16, fp16, or fp32 for the selected device."""

    normalized = (precision or "fp32").lower()
    aliases = {"float32": "fp32", "32": "fp32", "float16": "fp16", "half": "fp16", "16": "fp16"}
    normalized = aliases.get(normalized, normalized)
    if normalized == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16" if device.type == "cuda" else "fp32"
    if normalized not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16")
    if normalized == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA")
    if normalized == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not report bf16 support")
    return normalized


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _mixed_precision_policy(precision: str) -> Any:
    if precision not in {"fp16", "bf16"}:
        return None
    from torch.distributed.fsdp import MixedPrecision

    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def make_grad_scaler(state: DistributedState, precision: str):
    """Return the sharding-aware fp16 loss scaler, or ``None`` when unnecessary."""

    if precision != "fp16" or state.device.type != "cuda":
        return None
    if state.kind in {"fsdp", "fsdp2"}:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        return ShardedGradScaler()
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()


def _graph_for_distributed(module: nn.Module):
    if isinstance(module, DABSNGraph):
        return module
    direct = getattr(module, "graph", None)
    if isinstance(direct, DABSNGraph):
        return direct
    body = module.body if isinstance(module, DABSNSequenceModule) else module
    if isinstance(body, DABSNGraph):
        return body
    backbone = getattr(body, "backbone", None)
    graph = None if backbone is None else getattr(backbone, "graph", None)
    return graph if isinstance(graph, DABSNGraph) else None


def _component_fsdp_boundaries(module: nn.Module) -> tuple[nn.Module, ...]:
    """Resolve safe FSDP boundaries without inspecting provider architectures."""

    graph = _graph_for_distributed(module)
    if graph is not None and isinstance(graph.components, nn.ModuleList):
        boundaries = []
        for binding in graph.bindings:
            plan = binding.capabilities.parallel_plan or ParallelPlan()
            emit_event(
                EventCode.DISTRIBUTED_PLACEMENT,
                component_id=binding.component_id,
                kind="parallel-plan",
                plan=plan.to_dict(),
            )
            if plan.fsdp_boundary:
                boundaries.append(binding.module)
        return tuple(boundaries)

    body = module.body if isinstance(module, DABSNSequenceModule) else module
    backbone = getattr(body, "backbone", None)
    if backbone is None:
        return ()
    # Legacy 0.1.x graphs use non-owning execution views. Their historical
    # parameter-owning blocks remain the safe component-level boundaries.
    return tuple(getattr(backbone, "blocks", ())) + tuple(getattr(backbone, "middle_mlps", ()))


def _execute_component_parallel_plans(
    module: nn.Module,
    context: ParallelExecutionContext,
) -> None:
    """Execute provider-owned directives exactly once and reject partial handling."""

    graph = _graph_for_distributed(module)
    if graph is None:
        return
    from .graph import UnsupportedExecutionModeError

    failures = []
    for binding in graph.bindings:
        plan = binding.capabilities.parallel_plan
        if plan is None or not plan.directives:
            continue
        fingerprint = context.topology.fingerprint()
        if binding.parallelized_topology is not None:
            if binding.parallelized_topology != fingerprint:
                failures.append(
                    f"{binding.component_id}: already parallelized for a different topology"
                )
            continue
        missing_axes = sorted(
            {
                axis
                for directive in plan.directives
                for axis in directive.mesh_axes
                if axis not in context.coordinate
            }
        )
        if missing_axes:
            failures.append(f"{binding.component_id}: topology is missing axes {missing_axes}")
            continue
        executor = binding.parallel_executor
        if executor is None:
            failures.append(f"{binding.component_id}: provider installed no parallel executor")
            continue
        receipt = executor(binding.module, binding.config, plan, context)
        if not isinstance(receipt, ParallelExecutionReceipt):
            failures.append(
                f"{binding.component_id}: parallel executor returned "
                f"{type(receipt).__name__}, expected ParallelExecutionReceipt"
            )
            continue
        requested = {directive.directive_id for directive in plan.directives}
        consumed = set(receipt.consumed_directives)
        if consumed != requested:
            failures.append(
                f"{binding.component_id}: unconsumed={sorted(requested - consumed)} "
                f"unexpected={sorted(consumed - requested)}"
            )
            continue
        live_parameter_names = {name for name, _parameter in binding.module.named_parameters()}
        declared_parameter_names = set(receipt.parameter_sharding)
        if declared_parameter_names != live_parameter_names:
            failures.append(
                f"{binding.component_id}: parameter ownership missing="
                f"{sorted(live_parameter_names - declared_parameter_names)} unexpected="
                f"{sorted(declared_parameter_names - live_parameter_names)}"
            )
            continue
        unknown_parameter_axes = sorted(
            {
                axis
                for axes in receipt.parameter_sharding.values()
                for axis in axes
                if axis not in context.coordinate
            }
        )
        if unknown_parameter_axes:
            failures.append(
                f"{binding.component_id}: parameter ownership uses missing axes "
                f"{unknown_parameter_axes}"
            )
            continue
        binding.parallel_execution_receipt = receipt
        binding.parallelized_topology = fingerprint
        emit_event(
            EventCode.DISTRIBUTED_PLACEMENT,
            component_id=binding.component_id,
            kind="provider-parallel-plan-executed",
            topology=context.topology.to_dict(),
            directives=[directive.to_dict() for directive in plan.directives],
        )
    if failures:
        raise UnsupportedExecutionModeError(
            "parallel-plan directives were not executed completely: " + "; ".join(failures)
        )


def wrap_distributed(
    module: nn.Module,
    state: DistributedState,
    *,
    precision: str = "fp32",
) -> nn.Module:
    """Wrap a sequence-forward module in DDP or block-sharded FSDP."""

    if state.kind == "none":
        emit_event(
            EventCode.DISTRIBUTED_PLACEMENT,
            component_id=None,
            kind="none",
            world_size=1,
            device=str(state.device),
        )
        return module
    assert state.topology is not None
    context = state.parallel_context or _build_parallel_execution_context(
        state.topology,
        rank=state.rank,
        device=state.device,
    )
    object.__setattr__(module, "_dabsn_parallel_context", context)
    _execute_component_parallel_plans(module, context)
    try:
        data_group = cast(Any, context.group("data"))
        data_world_size = state.topology.axis_size("data")
    except KeyError:
        graph = _graph_for_distributed(module)
        provider_parallel = bool(
            graph is not None
            and any(binding.parallel_execution_receipt is not None for binding in graph.bindings)
        )
        if not provider_parallel:
            raise ValueError(
                f"distributed kind {state.kind!r} requires a named 'data' topology axis "
                "unless a provider parallel plan owns the topology"
            ) from None
        data_group = None
        data_world_size = 1
    if data_world_size == 1:
        emit_event(
            EventCode.DISTRIBUTED_PLACEMENT,
            component_id=None,
            kind="provider-only",
            world_size=state.world_size,
            data_world_size=1,
            device=str(state.device),
        )
        return module
    if state.kind == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [state.local_rank] if state.device.type == "cuda" else None
        # The canonical unified read retains parameters that are either used by
        # hard membership decisions or inactive in that read path. They do not
        # receive autograd gradients, so DDP must explicitly permit them.
        wrapped = DistributedDataParallel(
            module,
            device_ids=device_ids,
            find_unused_parameters=True,
            process_group=data_group,
        )
        emit_event(
            EventCode.DISTRIBUTED_PLACEMENT,
            component_id=None,
            kind="ddp",
            world_size=data_world_size,
            device=str(state.device),
        )
        return wrapped
    if state.kind == "fsdp2":
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        dtype = {
            "fp32": None,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[precision]
        policy = MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=dtype,
            output_dtype=dtype,
        )
        data_mesh = None
        if torch_dist.is_initialized():
            data_mesh = DeviceMesh.from_group(
                data_group,
                state.device.type,
                mesh=list(context.group_ranks["data"]),
                mesh_dim_names=("data",),
            )
        boundaries = _component_fsdp_boundaries(module)
        seen: set[int] = set()
        for boundary in boundaries:
            if id(boundary) in seen or not any(
                parameter.requires_grad for parameter in boundary.parameters()
            ):
                continue
            seen.add(id(boundary))
            fully_shard(
                boundary,
                mesh=data_mesh,
                mp_policy=policy,
                reshard_after_forward=True,
            )
        fully_shard(
            module,
            mesh=data_mesh,
            mp_policy=policy,
            reshard_after_forward=True,
        )
        emit_event(
            EventCode.DISTRIBUTED_PLACEMENT,
            component_id=None,
            kind="fsdp2",
            world_size=data_world_size,
            device=str(state.device),
            boundaries=len(seen),
        )
        return module

    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel,
        ShardingStrategy,
    )

    boundaries = _component_fsdp_boundaries(module)
    boundary_ids = {id(boundary) for boundary in boundaries}

    def component_auto_wrap_policy(module, recurse, nonwrapped_numel):
        del nonwrapped_numel
        return True if recurse else id(module) in boundary_ids

    fsdp_wrapped = FullyShardedDataParallel(
        module,
        device_id=state.device,
        mixed_precision=_mixed_precision_policy(precision),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=component_auto_wrap_policy,
        limit_all_gathers=True,
        forward_prefetch=True,
        # Prefetch the next block's all-gather while the current block's
        # gradients reduce, so at seq 2048 the recurrent backward overlaps
        # communication instead of stalling on it. BACKWARD_PRE issues the
        # gather earliest (more overlap, slightly higher peak memory) which is
        # the right trade with FULL_SHARD + block-level grad checkpointing.
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        sync_module_states=True,
        process_group=data_group,
    )
    emit_event(
        EventCode.DISTRIBUTED_PLACEMENT,
        component_id=None,
        kind="fsdp",
        world_size=data_world_size,
        device=str(state.device),
    )
    return fsdp_wrapped


def prepare_distributed_model(
    model: DABSNModelType,
    state: DistributedState,
    *,
    precision: str = "fp32",
) -> nn.Module:
    """Move a DABSN model to its worker device and expose sequence-forward training."""

    sequence_model = DABSNSequenceModule(model.to(state.device))
    return wrap_distributed(sequence_model, state, precision=precision)


def prepare_distributed_module(
    module: nn.Module,
    state: DistributedState,
    *,
    precision: str = "fp32",
) -> nn.Module:
    """Prepare any component graph or ordinary module for distributed execution."""

    prepared = DABSNSequenceModule(module.to(state.device))
    return wrap_distributed(prepared, state, precision=precision)


def unwrap_dabsn_artifact(model: nn.Module) -> DABSNArtifactType:
    """Return the raw portable model or graph from distributed wrappers."""

    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (DABSNModel, DABSNTaskModel, DABSNSequenceLM, DABSNGraph)):
            return current
        if isinstance(current, DABSNSequenceModule):
            current = current.body
            continue
        child = getattr(current, "module", None)
        if isinstance(child, nn.Module):
            current = child
            continue
        break
    raise TypeError("distributed wrapper does not contain a portable DABSN model or graph")


def unwrap_dabsn_model(model: nn.Module) -> DABSNModelType:
    """Return the underlying public DABSN model from DDP/FSDP shells."""

    artifact = unwrap_dabsn_artifact(model)
    if isinstance(artifact, DABSNGraph):
        raise TypeError("distributed wrapper contains a raw DABSN graph, not a model")
    return artifact


def _is_fsdp(model: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        return False
    return isinstance(model, FullyShardedDataParallel)


def full_model_state_dict(model: nn.Module, state: DistributedState) -> dict[str, Tensor]:
    """Materialize a portable full CPU model state on rank zero."""

    if state.kind == "fsdp2":
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        result = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                strict=True,
            ),
        )
        return {str(name): value for name, value in result.items() if isinstance(value, Tensor)}
    if state.kind != "fsdp":
        return model.state_dict()
    if not _is_fsdp(model):
        raise TypeError("state.kind is fsdp but model is not an FSDP root")
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel,
        StateDictType,
    )

    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        config,
    ):
        return model.state_dict()


def full_training_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    """Gather canonical full model/optimizer state for release-gate evidence."""

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    model_state, optimizer_state = get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            strict=True,
        ),
    )
    return {"model": model_state, "optimizer": optimizer_state}


def state_dict_digest(state: object) -> str:
    """Hash nested training state with names, types, shapes, and tensor bytes."""

    import hashlib
    import json

    digest = hashlib.sha256()

    def update(value: object) -> None:
        if isinstance(value, Tensor):
            tensor = value.detach().cpu().contiguous()
            header = {
                "kind": "tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
            digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            return
        if isinstance(value, dict):
            digest.update(b"dict{")
            for key in sorted(value, key=lambda item: str(item)):
                update(str(key))
                update(value[key])
            digest.update(b"}")
            return
        if isinstance(value, (list, tuple)):
            digest.update(b"list[")
            for item in value:
                update(item)
            digest.update(b"]")
            return
        if value is None or isinstance(value, (str, int, float, bool)):
            digest.update(
                json.dumps(
                    {"kind": type(value).__name__, "value": value},
                    sort_keys=True,
                ).encode("utf-8")
            )
            return
        raise TypeError(f"unsupported state digest value {type(value).__name__}")

    update(state)
    return digest.hexdigest()


def _normalize_distributed_state_key(name: str) -> str:
    parts = name.split(".")
    while parts and parts[0] in {"module", "_fsdp_wrapped_module"}:
        parts.pop(0)
    if parts and parts[0] == "body":
        parts.pop(0)
    parts = [part for part in parts if part != "_fsdp_wrapped_module"]
    return ".".join(parts)


def _portable_state_dict(model: nn.Module, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    unwrap_dabsn_artifact(model)
    portable: dict[str, Tensor] = {}
    for original, value in state_dict.items():
        normalized = _normalize_distributed_state_key(original)
        if not normalized:
            raise KeyError(f"distributed state key {original!r} has no portable DABSN name")
        if normalized in portable:
            raise KeyError(f"distributed state keys collide at portable DABSN name {normalized!r}")
        portable[normalized] = value.detach().cpu()
    return portable


def _provider_sharded_bindings(model: nn.Module):
    graph = _graph_for_distributed(model)
    if graph is None:
        return ()
    return tuple(
        binding
        for binding in graph.bindings
        if binding.parallelized_topology is not None
        and binding.capabilities.parallel_plan is not None
        and binding.capabilities.parallel_plan.directives
    )


def _consolidate_provider_state_dict(
    model: nn.Module,
    state_dict: dict[str, Tensor],
    state: DistributedState,
) -> dict[str, Tensor]:
    """Let each provider reconstruct only the component state it owns."""

    bindings = _provider_sharded_bindings(model)
    if not bindings:
        return _portable_state_dict(model, state_dict)
    raw = unwrap_dabsn_artifact(model)
    canonical = _portable_state_dict(model, state_dict)
    module_paths = {id(module): name for name, module in raw.named_modules()}
    context = state.parallel_context
    if context is None:
        if state.topology is None:
            raise RuntimeError("provider-sharded model has no parallel topology")
        context = _build_parallel_execution_context(
            state.topology,
            rank=state.rank,
            device=state.device,
        )
    for binding in bindings:
        consolidator = binding.parallel_state_consolidator
        if consolidator is None:
            raise RuntimeError(
                f"component {binding.component_id!r} is provider-sharded but its provider "
                "does not implement portable-state consolidation"
            )
        try:
            prefix = module_paths[id(binding.module)]
        except KeyError as exc:
            raise RuntimeError(
                f"component {binding.component_id!r} is not owned by the checkpoint model"
            ) from exc
        prefix = f"{prefix}." if prefix else ""
        local = {
            name[len(prefix) :]: tensor
            for name, tensor in canonical.items()
            if name.startswith(prefix)
        }
        if not local:
            raise RuntimeError(
                f"component {binding.component_id!r} has no state under namespace {prefix!r}"
            )
        consolidated = consolidator(binding.module, binding.config, local, context)
        if any(not isinstance(name, str) or not name for name in consolidated):
            raise RuntimeError(
                f"component {binding.component_id!r} returned an invalid state name"
            )
        if any(not isinstance(tensor, Tensor) for tensor in consolidated.values()):
            raise RuntimeError(
                f"component {binding.component_id!r} returned a non-tensor state value"
            )
        old_keys = {prefix + name for name in local}
        for old_key in old_keys:
            del canonical[old_key]
        new_keys = {prefix + name for name in consolidated}
        collisions = new_keys.intersection(canonical)
        if collisions:
            raise RuntimeError(
                f"component {binding.component_id!r} consolidated state collides with "
                f"existing names {sorted(collisions)}"
            )
        for name, tensor in consolidated.items():
            canonical[prefix + name] = tensor.detach().cpu()
    return canonical


def optimizer_checkpoint_path(path: str | Path) -> Path:
    return Path(str(path) + ".optimizer.pt")


def save_distributed_dabsn(
    model: nn.Module,
    path: str | Path,
    state: DistributedState,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int | None = None,
    extra: dict[str, object] | None = None,
    scaler: StatefulTrainingObject | None = None,
) -> Path | None:
    """Save one portable SafeTensors model and optional optimizer sidecar.

    This portable mode materializes the full FSDP model and optimizer on rank
    zero. Use :func:`save_sharded_training_checkpoint` for runs whose state
    cannot fit on one host.
    """

    destination = Path(path)
    raw = unwrap_dabsn_artifact(model)
    full_state = full_model_state_dict(model, state)
    provider_shards = _provider_sharded_bindings(model)
    if optimizer is not None and provider_shards:
        component_ids = [binding.component_id for binding in provider_shards]
        raise ValueError(
            "portable optimizer sidecars do not encode provider-owned parameter shards; "
            f"components={component_ids}. Save exact training state with "
            "save_sharded_training_checkpoint, then export portable model weights without "
            "an optimizer."
        )
    portable_state = _consolidate_provider_state_dict(model, full_state, state)
    checkpoint_extra = dict(extra or {})
    transaction = None
    if optimizer is not None:
        import uuid

        transaction = uuid.uuid4().hex if state.is_main else None
        if state.enabled:
            objects = [transaction]
            torch_dist.broadcast_object_list(objects, src=0, device=state.device)
            transaction = objects[0]
        checkpoint_extra["training_transaction"] = transaction
    if state.is_main:
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_dabsn_state(
            portable_state,
            graph_config_dict(raw) if isinstance(raw, DABSNGraph) else dabsn_config_dict(raw),
            destination,
            extra=checkpoint_extra,
            _model=raw,
        )
    if optimizer is not None:
        if state.kind == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel

            optimizer_state = FullyShardedDataParallel.full_optim_state_dict(
                model,
                optimizer,
                rank0_only=True,
            )
        else:
            optimizer_state = optimizer.state_dict()
        if state.is_main:
            sidecar = optimizer_checkpoint_path(destination)
            temporary = sidecar.with_name(sidecar.name + ".tmp")
            torch.save(
                {
                    "optimizer": optimizer_state,
                    "step": step,
                    "extra": checkpoint_extra,
                    "scaler": None if scaler is None else scaler.state_dict(),
                },
                temporary,
            )
            os.replace(temporary, sidecar)
    barrier(state)
    return destination if state.is_main else None


def _training_manifest_path(path: str | Path) -> Path:
    return Path(path) / "dabsn-training.json"


def _training_commit_path(path: str | Path) -> Path:
    return Path(path) / "COMMITTED.json"


def _committed_checkpoint_source(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        return destination
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        return backup
    return destination


def _checkpoint_file_digests(path: Path) -> dict[str, str]:
    import hashlib

    digests: dict[str, str] = {}
    excluded = {_training_manifest_path(path), _training_commit_path(path)}
    for source in sorted(item for item in path.rglob("*") if item.is_file()):
        if source in excluded:
            continue
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[str(source.relative_to(path))] = digest.hexdigest()
    return digests


def _fsync_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _gather_rng_tensor(local: Tensor, state: DistributedState) -> Tensor:
    collective_device = state.device if state.enabled else local.device
    value = local.to(collective_device)
    if not state.enabled:
        return value.unsqueeze(0).cpu()
    gathered = [torch.empty_like(value) for _ in range(state.world_size)]
    torch_dist.all_gather(gathered, value)
    return torch.stack(gathered).cpu()


def _distributed_rng_state(state: DistributedState) -> dict[str, Tensor]:
    cpu = _gather_rng_tensor(torch.get_rng_state(), state)
    if state.device.type == "cuda":
        cuda = _gather_rng_tensor(torch.cuda.get_rng_state(state.device), state)
    else:
        cuda = torch.empty((state.world_size, 0), dtype=torch.uint8)
    return {"cpu": cpu, "cuda": cuda}


def save_sharded_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    state: DistributedState,
    *,
    step: int,
    extra: dict[str, object] | None = None,
    scaler: StatefulTrainingObject | None = None,
    scheduler: StatefulTrainingObject | None = None,
) -> Path | None:
    """Save reshardable model and optimizer state without a rank-zero gather."""

    import json
    import shutil
    import uuid

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    destination = Path(path)
    transaction = uuid.uuid4().hex if state.is_main else None
    if state.enabled:
        objects = [transaction]
        torch_dist.broadcast_object_list(objects, src=0, device=state.device)
        transaction = objects[0]
    assert isinstance(transaction, str)
    staging = destination.with_name(destination.name + f".staging.{transaction}")
    backup = destination.with_name(destination.name + ".previous")
    if state.is_main:
        if not destination.exists() and backup.exists():
            os.replace(backup, destination)
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
    barrier(state)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=False)
    model_state, optimizer_state = get_state_dict(
        model,
        optimizer,
        options=options,
    )
    rng_state = _distributed_rng_state(state)
    checkpoint_state: dict[str, Any] = {
        "model": model_state,
        "optimizer": optimizer_state,
        "rng": rng_state,
    }
    if scaler is not None:
        checkpoint_state["scaler"] = scaler.state_dict()
    if scheduler is not None:
        checkpoint_state["scheduler"] = scheduler.state_dict()
    dcp.save(
        checkpoint_state,
        checkpoint_id=staging,
        no_dist=not state.enabled,
    )
    if state.is_main:
        raw = unwrap_dabsn_artifact(model)
        graph = raw if isinstance(raw, DABSNGraph) else getattr(raw, "graph", None)
        providers = []
        if graph is not None:
            providers = sorted(
                {
                    binding.provider_key
                    for binding in graph.bindings
                    if binding.provider_key is not None
                }
            )
        manifest = {
            "format": "dabsn-distributed-checkpoint",
            "version": 2,
            "transaction": transaction,
            "framework_version": "2.0.0",
            "torch_version": torch.__version__,
            "step": int(step),
            "config": (
                graph_config_dict(raw) if isinstance(raw, DABSNGraph) else dabsn_config_dict(raw)
            ),
            "extra": extra or {},
            "providers": providers,
            "optimizer": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
            "topology": state.report(),
            "scheduler": (extra or {}).get("scheduler"),
            "scaler": (extra or {}).get("amp_scaler"),
            "rng": (extra or {}).get("rng"),
            "loader_cursor": (extra or {}).get("loader_cursor"),
            "accumulation": (extra or {}).get("accumulation"),
            "rng_world_size": state.world_size,
            "rng_cpu_bytes": int(rng_state["cpu"].shape[1]),
            "rng_cuda_bytes": int(rng_state["cuda"].shape[1]),
            "has_scaler_state": scaler is not None,
            "has_scheduler_state": scheduler is not None,
        }
        manifest["digests"] = _checkpoint_file_digests(staging)
        manifest_path = _training_manifest_path(staging)
        temporary = manifest_path.with_suffix(".tmp")
        _fsync_text(temporary, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, manifest_path)
        _fsync_directory(staging)
        # Re-read and validate every shard before publishing the commit marker.
        observed = _checkpoint_file_digests(staging)
        if observed != manifest["digests"]:
            raise RuntimeError("distributed checkpoint shard digest changed during staging")
        commit = {
            "format": "dabsn-distributed-checkpoint-commit",
            "version": 1,
            "transaction": transaction,
            "manifest": "dabsn-training.json",
        }
        _fsync_text(
            _training_commit_path(staging),
            json.dumps(commit, sort_keys=True, separators=(",", ":")) + "\n",
        )
        _fsync_directory(staging)
    barrier(state)
    if state.is_main:
        shutil.rmtree(backup, ignore_errors=True)
        if destination.exists():
            os.replace(destination, backup)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        shutil.rmtree(backup, ignore_errors=True)
        _fsync_directory(destination.parent)
    barrier(state)
    return destination if state.is_main else None


def inspect_sharded_training_checkpoint(path: str | Path) -> dict[str, object]:
    """Read the non-tensor manifest for a DABSN distributed checkpoint."""

    import json

    destination = _committed_checkpoint_source(path)
    source = _training_manifest_path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"missing distributed checkpoint manifest: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != "dabsn-distributed-checkpoint" or payload.get("version") not in {
        1,
        2,
    }:
        raise ValueError(f"not a supported DABSN distributed checkpoint: {path}")
    if payload.get("version") == 2:
        commit_path = _training_commit_path(destination)
        if not commit_path.is_file():
            raise ValueError(f"distributed checkpoint is incomplete (no commit marker): {path}")
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        if commit.get("transaction") != payload.get("transaction"):
            raise ValueError("distributed checkpoint transaction IDs disagree")
        expected = payload.get("digests")
        observed = _checkpoint_file_digests(destination)
        if not isinstance(expected, dict) or observed != expected:
            raise ValueError("distributed checkpoint shard digest validation failed")
    return payload


def _validate_checkpoint_topology(
    manifest: Mapping[str, object],
    state: DistributedState,
    *,
    allow_topology_change: bool,
    restore_rng: bool,
) -> None:
    saved_world_size = manifest.get("rng_world_size")
    if (
        isinstance(saved_world_size, int)
        and saved_world_size != state.world_size
        and not allow_topology_change
    ):
        raise ValueError(
            f"checkpoint worker count {saved_world_size}, received {state.world_size}; "
            "pass allow_topology_change=True only for an intentional reshard restore"
        )
    saved_report = manifest.get("topology")
    saved_fingerprint = (
        saved_report.get("topology_fingerprint")
        if isinstance(saved_report, Mapping)
        else None
    )
    assert state.topology is not None
    current_fingerprint = state.topology.fingerprint()
    if not isinstance(saved_fingerprint, str):
        if isinstance(saved_world_size, int) and saved_world_size != state.world_size:
            raise ValueError(
                "checkpoint worker count "
                f"{saved_world_size}, received {state.world_size}; legacy checkpoints "
                "without a topology fingerprint cannot change worker count"
            )
        return
    if saved_fingerprint == current_fingerprint:
        return
    if not allow_topology_change:
        saved_topology = saved_report.get("topology") if isinstance(saved_report, Mapping) else None
        raise ValueError(
            "checkpoint parallel topology differs from the active topology: "
            f"saved={saved_topology!r} active={state.topology.to_dict()!r}. "
            "Pass allow_topology_change=True only for an intentional reshard restore."
        )
    if restore_rng:
        raise ValueError(
            "an intentional topology-change restore cannot exactly restore per-worker random "
            "number generator state; pass restore_rng=False"
        )
    providers = manifest.get("providers")
    if isinstance(providers, list) and providers:
        raise ValueError(
            "topology-change restore for provider-owned component shards is not implemented; "
            f"checkpoint providers={providers}. Restore with the recorded topology, export "
            "portable model weights, then initialize a new optimizer under the new topology."
        )


def load_sharded_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    state: DistributedState,
    *,
    scaler: StatefulTrainingObject | None = None,
    scheduler: StatefulTrainingObject | None = None,
    restore_rng: bool = True,
    allow_topology_change: bool = False,
) -> dict[str, object]:
    """Restore model and optimizer state, resharding for the active workers."""

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )

    manifest = inspect_sharded_training_checkpoint(path)
    _validate_checkpoint_topology(
        manifest,
        state,
        allow_topology_change=allow_topology_change,
        restore_rng=restore_rng,
    )
    options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=False)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    state_dict: dict[str, Any] = {
        "model": model_state,
        "optimizer": optimizer_state,
    }
    saved_world_size = manifest.get("rng_world_size")
    has_rng = isinstance(saved_world_size, int)
    if restore_rng and has_rng and saved_world_size != state.world_size:
        raise ValueError(
            "exact RNG restore requires the checkpoint worker count "
            f"{saved_world_size}, received {state.world_size}; use "
            "restore_rng=False only for an intentional topology-change restore"
        )
    if has_rng:
        assert isinstance(saved_world_size, int)
        state_dict["rng"] = {
            "cpu": torch.empty(
                (saved_world_size, int(cast(Any, manifest["rng_cpu_bytes"]))),
                dtype=torch.uint8,
            ),
            "cuda": torch.empty(
                (saved_world_size, int(cast(Any, manifest["rng_cuda_bytes"]))),
                dtype=torch.uint8,
            ),
        }
    if manifest.get("has_scaler_state"):
        if scaler is None:
            raise ValueError(
                "checkpoint contains scaler state; provide the matching scaler for exact resume"
            )
        state_dict["scaler"] = scaler.state_dict()
    if manifest.get("has_scheduler_state"):
        if scheduler is None:
            raise ValueError(
                "checkpoint contains scheduler state; provide the matching "
                "scheduler for exact resume"
            )
        state_dict["scheduler"] = scheduler.state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=_committed_checkpoint_source(path),
        planner=dcp.DefaultLoadPlanner(allow_partial_load=True),
        no_dist=not state.enabled,
    )
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state_dict["model"],
        optim_state_dict=state_dict["optimizer"],
        options=options,
    )
    if manifest.get("has_scaler_state"):
        assert scaler is not None
        scaler.load_state_dict(state_dict["scaler"])
    if manifest.get("has_scheduler_state"):
        assert scheduler is not None
        scheduler.load_state_dict(state_dict["scheduler"])
    if restore_rng and has_rng:
        rng = cast(dict[str, Tensor], state_dict["rng"])
        torch.set_rng_state(rng["cpu"][state.rank].cpu())
        if state.device.type == "cuda" and rng["cuda"].shape[1]:
            torch.cuda.set_rng_state(rng["cuda"][state.rank].cpu(), state.device)
    barrier(state)
    return manifest


def load_sharded_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    state: DistributedState,
    *,
    allow_topology_change: bool = False,
) -> dict[str, object]:
    """Restore only model tensors from a DABSN distributed checkpoint."""

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    manifest = inspect_sharded_training_checkpoint(path)
    _validate_checkpoint_topology(
        manifest,
        state,
        allow_topology_change=allow_topology_change,
        restore_rng=False,
    )
    options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=True)
    model_state = get_model_state_dict(model, options=options)
    state_dict = {"model": model_state}
    dcp.load(
        state_dict,
        checkpoint_id=_committed_checkpoint_source(path),
        no_dist=not state.enabled,
    )
    set_model_state_dict(model, state_dict["model"], options=options)
    barrier(state)
    return manifest


def load_distributed_optimizer(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    path: str | Path,
    state: DistributedState,
    *,
    scaler: StatefulTrainingObject | None = None,
) -> int | None:
    """Restore an optimizer sidecar, scattering a full FSDP state from rank zero."""

    source = optimizer_checkpoint_path(path)
    local_exists = source.is_file()
    exists = local_exists
    if state.enabled:
        flag = torch.tensor(int(local_exists if state.is_main else False), device=state.device)
        torch_dist.broadcast(flag, src=0)
        exists = bool(flag.item())
    if not exists:
        return None
    payload: dict[str, Any] | None = (
        torch.load(source, map_location="cpu", weights_only=False) if state.is_main else None
    )
    transactions_match = True
    if state.is_main:
        assert payload is not None
        model_extra = cast(dict[str, Any], inspect_dabsn(path)["extra"])
        model_transaction = model_extra.get("training_transaction")
        optimizer_transaction = payload.get("extra", {}).get("training_transaction")
        transactions_match = model_transaction == optimizer_transaction
    if state.enabled:
        flag = torch.tensor(int(transactions_match), device=state.device)
        torch_dist.broadcast(flag, src=0)
        transactions_match = bool(flag.item())
    if not transactions_match:
        raise ValueError("model and optimizer sidecar are from different checkpoint transactions")
    if state.kind == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel

        if state.is_main:
            assert payload is not None
            full_optimizer_state = payload["optimizer"]
        else:
            full_optimizer_state = None
        local_optimizer_state = FullyShardedDataParallel.scatter_full_optim_state_dict(
            full_optimizer_state,
            model,
            optim=optimizer,
        )
        optimizer.load_state_dict(local_optimizer_state)
    else:
        if state.kind == "ddp":
            all_have = torch.tensor(int(local_exists), device=state.device)
            torch_dist.all_reduce(all_have, op=torch_dist.ReduceOp.MIN)
            if bool(all_have.item()):
                if not state.is_main:
                    payload = torch.load(source, map_location="cpu", weights_only=False)
            else:
                objects = [payload]
                torch_dist.broadcast_object_list(objects, src=0, device=state.device)
                payload = cast(dict[str, Any], objects[0])
        assert payload is not None
        optimizer.load_state_dict(payload["optimizer"])
    if state.is_main:
        assert payload is not None
        scaler_state = payload.get("scaler")
        step = payload.get("step")
    else:
        scaler_state = None
        step = None
    if state.enabled:
        objects = [scaler_state]
        torch_dist.broadcast_object_list(objects, src=0, device=state.device)
        scaler_state = objects[0]
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    if state.enabled:
        value = torch.tensor(-1 if step is None else int(step), device=state.device)
        torch_dist.broadcast(value, src=0)
        step = None if int(value.item()) < 0 else int(value.item())
    return None if step is None else int(step)


def shard_batch(tensor: Tensor, state: DistributedState) -> Tensor:
    """Select this worker's contiguous share of a global batch."""

    if not state.batch_parallel:
        return tensor.to(state.device)
    workers = state.data_world_size
    if tensor.shape[0] < workers or tensor.shape[0] % workers:
        raise ValueError(
            f"global batch {tensor.shape[0]} must be divisible by data-axis size {workers}"
        )
    return tensor.chunk(workers, dim=0)[state.data_rank].to(state.device)


def no_sync_context(model: nn.Module, *, synchronize: bool):
    """Suppress DDP/FSDP gradient collectives for non-final accumulation microbatches."""

    if synchronize:
        return nullcontext()
    set_sync = getattr(model, "set_requires_gradient_sync", None)
    if callable(set_sync):

        @contextmanager
        def fsdp2_no_sync():
            set_sync(False)
            try:
                yield
            finally:
                set_sync(True)

        return fsdp2_no_sync()
    if not hasattr(model, "no_sync"):
        return nullcontext()
    return cast(Any, model).no_sync()


def clip_grad_norm(model: nn.Module, max_norm: float) -> Tensor:
    """Clip global FSDP gradients correctly, or ordinary gradients otherwise."""

    context = None
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        candidate = getattr(current, "_dabsn_parallel_context", None)
        if isinstance(candidate, ParallelExecutionContext):
            context = candidate
            break
        child = getattr(current, "module", None)
        if not isinstance(child, nn.Module):
            break
        current = child
    graph = _graph_for_distributed(model)
    provider_sharded = bool(
        graph is not None
        and any(binding.parallel_execution_receipt is not None for binding in graph.bindings)
    )
    if context is None or not provider_sharded or not torch_dist.is_initialized():
        if _is_fsdp(model):
            return cast(Any, model).clip_grad_norm_(float(max_norm))
        return torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))

    parameter_axes: dict[int, tuple[str, ...]] = {}
    assert graph is not None
    for binding in graph.bindings:
        receipt = binding.parallel_execution_receipt
        if receipt is None:
            continue
        for name, parameter in binding.module.named_parameters():
            parameter_axes[id(parameter)] = receipt.parameter_sharding[name]

    data_is_sharded = _is_fsdp(model) or type(model).__name__ == "FSDPModule"
    all_axes = tuple(axis.name for axis in context.topology.axes)
    local_squared = torch.zeros((), device=cast(Any, context.device), dtype=torch.float64)
    gradients = []
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        to_local = getattr(gradient, "to_local", None)
        local_gradient = cast(Tensor, to_local() if callable(to_local) else gradient)
        gradients.append(local_gradient)
        sharded_axes = set(parameter_axes.get(id(parameter), ()))
        if data_is_sharded and "data" in all_axes:
            sharded_axes.add("data")
        replicas = 1
        for axis in context.topology.axes:
            if axis.name not in sharded_axes:
                replicas *= axis.size
        local_squared += local_gradient.detach().double().square().sum() / replicas
    torch_dist.all_reduce(local_squared, op=torch_dist.ReduceOp.SUM)
    total_norm = local_squared.sqrt()
    coefficient = torch.clamp(
        torch.tensor(float(max_norm), device=total_norm.device, dtype=total_norm.dtype)
        / (total_norm + 1.0e-6),
        max=1.0,
    )
    for gradient in gradients:
        gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
    # The accumulation is FP64 on purpose -- squares summed across every rank
    # lose too much in FP32 -- but the RETURN must not advertise that. The
    # unsharded branch above hands back whatever `torch.nn.utils.clip_grad_norm_`
    # returns, which is the gradient dtype, and a function that changes its
    # return dtype depending on the topology breaks every caller that logs the
    # norm or compares a sharded run against a single-process one. Precision
    # where it matters, one dtype at the boundary.
    return total_norm.to(gradients[0].dtype) if gradients else total_norm


def reduce_declared_results(
    values: tuple[Tensor, ...],
    declarations: tuple[ResultDeclaration, ...],
    *,
    group=None,
) -> tuple[Tensor, ...]:
    """Apply provider-declared telemetry reductions outside compiled execution."""

    if len(values) != len(declarations):
        raise ValueError("result values and declarations have different arity")
    if not torch_dist.is_available() or not torch_dist.is_initialized():
        return values
    operations = {
        "sum": torch_dist.ReduceOp.SUM,
        "mean": torch_dist.ReduceOp.SUM,
        "max": torch_dist.ReduceOp.MAX,
        "min": torch_dist.ReduceOp.MIN,
    }
    world_size = torch_dist.get_world_size(group)
    reduced = []
    for value, declaration in zip(values, declarations):
        if declaration.distributed_owner != "framework" or declaration.reduction == "none":
            reduced.append(value)
            continue
        result = value.detach().clone()
        torch_dist.all_reduce(
            result,
            op=operations[declaration.reduction],
            group=group,
        )
        if declaration.reduction == "mean":
            result.div_(world_size)
        reduced.append(result)
    return tuple(reduced)


def barrier(state: DistributedState) -> None:
    if state.enabled and torch_dist.is_initialized():
        torch_dist.barrier()


def cleanup_distributed(state: DistributedState) -> None:
    if state.enabled and torch_dist.is_initialized():
        torch_dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Tensor parallelism over the hidden dimension
#
# Data parallelism replicates the model on every rank, so the largest model that
# can be trained is the largest that fits on ONE device. Above roughly 10B
# parameters no device holds a replica and the model itself has to be split.
#
# For this recurrence the natural axis is the hidden dimension. Every per-unit
# parameter (beta, k_*, r_*, log_kappa, ...) is [H] and shards cleanly; the input
# projections W/Wg are output-sharded; the recurrent matrices Ug/A are ROW
# sharded, so rank p holds the rows that produce its own units.
#
# The one thing that does not shard is the recurrent product's input: rank p's
# rows need every unit of y, not just the H/P it owns. So each step all-gathers
# y ([B,H/P] -> [B,H]) and that is the only collective in the step. The
# alternative -- column-sharding the recurrent matrices and all-reducing the
# [B,2H] product -- moves 2P times more bytes per step, which is why the gather
# is on y rather than on the product.
#
# State (budget, energy, saturation) stays sharded end to end, so activation
# memory divides by P as well as parameter memory.

_PER_UNIT_CORE_PARAMS = (
    "beta",
    "log_kappa",
    "logit_recover",
    "k_s",
    "k_y",
    "k_b",
    "k_n",
    "k_bias",
    "k_saturation",
    "r_s",
    "r_y",
    "r_b",
    "r_n",
    "r_bias",
    "r_saturation",
)
# Scalars describe the whole core rather than a unit, so every rank keeps them
# whole; sharding them would change the math, not just its placement.
_REPLICATED_CORE_PARAMS = (
    "logit_alpha",
    "log_lambda",
    "logit_saturation_decay",
    "logit_saturation_suppress",
)


class TensorParallelCoreShard(TypedDict):
    beta: Tensor
    log_kappa: Tensor
    logit_recover: Tensor
    k_s: Tensor
    k_y: Tensor
    k_b: Tensor
    k_n: Tensor
    k_bias: Tensor
    k_saturation: Tensor
    r_s: Tensor
    r_y: Tensor
    r_b: Tensor
    r_n: Tensor
    r_bias: Tensor
    r_saturation: Tensor
    logit_alpha: Tensor
    log_lambda: Tensor
    logit_saturation_decay: Tensor
    logit_saturation_suppress: Tensor
    W: Tensor
    Wg: Tensor
    Wg_bias: Tensor
    Ug: Tensor
    A: Tensor
    slice: slice


def hidden_shard(hidden_dim: int, rank: int, world_size: int) -> slice:
    """This rank's contiguous slice of the hidden dimension.

    The remainder is spread one unit at a time over the low ranks rather than
    piled onto the last one, so no rank carries a disproportionate tail when
    H is not a multiple of P -- which it frequently is not once H is chosen for
    the model rather than for the cluster.
    """
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside world size {world_size}")
    base, extra = divmod(int(hidden_dim), int(world_size))
    start = rank * base + min(rank, extra)
    return slice(start, start + base + (1 if rank < extra else 0))


def shard_core_tensor_parallel(
    core: DABSNCore,
    rank: int,
    world_size: int,
) -> TensorParallelCoreShard:
    """Materialise one rank's shard of a core's parameters.

    Returns plain tensors rather than a module: the shard is what a rank feeds
    to the scan, and keeping it as data makes the sharded and replicated paths
    comparable in a test without constructing a second module type.
    """
    cut = hidden_shard(core.hidden_dim, rank, world_size)
    shard: dict[str, Any] = {}
    for name in _PER_UNIT_CORE_PARAMS:
        shard[name] = getattr(core, name).detach()[cut].clone()
    for name in _REPLICATED_CORE_PARAMS:
        shard[name] = getattr(core, name).detach().clone()
    shard["W"] = core.W.weight.detach()[cut].clone()  # output-sharded
    shard["Wg"] = core.Wg.weight.detach()[cut].clone()
    shard["Wg_bias"] = core.Wg.bias.detach()[cut].clone()
    # Row-sharded: rank p produces only its own units, from all of y.
    shard["Ug"] = core.Ug.weight.detach()[cut].clone()
    shard["A"] = core.A.weight.detach()[cut].clone()
    shard["slice"] = cut
    return cast(TensorParallelCoreShard, shard)


class _AllGatherHidden(torch.autograd.Function):
    """All-gather the hidden dimension, and route the gradient back home.

    ``dist.all_gather`` writes into fresh buffers that autograd knows nothing
    about, so the naive version is silently wrong in the backward pass: rank p's
    units influence every OTHER rank's recurrence through the gathered ``y``,
    and that cross-rank term simply vanishes. The forward matches the unsharded
    model, every shape checks out, and the gradients are quietly incomplete --
    the model trains, just not the model you think.

    The backward of an all-gather is a reduce-scatter: sum each rank's slice of
    the incoming gradient across all ranks, and hand each rank its own slice.
    Implemented as all-reduce plus a slice, which is the same arithmetic and
    tolerates uneven shards.

    Shards are padded to a common width before the gather because collectives
    require equal-sized buffers, and H is routinely not a multiple of the world
    size. The padding is trimmed on the way out, so an uneven split is exact
    rather than merely supported.
    """

    @staticmethod
    def forward(ctx, y_local: Tensor, group):
        world = torch_dist.get_world_size(group)
        rank = torch_dist.get_rank(group)
        widths = [torch.zeros(1, dtype=torch.long, device=y_local.device) for _ in range(world)]
        torch_dist.all_gather(
            widths,
            torch.tensor([y_local.shape[-1]], dtype=torch.long, device=y_local.device),
            group=group,
        )
        sizes = [int(w.item()) for w in widths]
        pad = max(sizes)

        padded = y_local
        if y_local.shape[-1] < pad:
            padded = torch.nn.functional.pad(y_local, (0, pad - y_local.shape[-1]))
        buffers = [torch.empty_like(padded) for _ in range(world)]
        torch_dist.all_gather(buffers, padded.contiguous(), group=group)

        ctx.sizes, ctx.rank, ctx.group = sizes, rank, group
        return torch.cat([buf[..., :size] for buf, size in zip(buffers, sizes)], dim=-1)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # Every rank computed with the full gathered y, so every rank holds a
        # partial gradient for every unit. Summing them and taking this rank's
        # slice is the reduce-scatter that closes the loop.
        grad = grad_output.contiguous()
        torch_dist.all_reduce(grad, op=torch_dist.ReduceOp.SUM, group=ctx.group)
        start = sum(ctx.sizes[: ctx.rank])
        return grad[..., start : start + ctx.sizes[ctx.rank]], None


def tensor_parallel_core_scan(
    shard: TensorParallelCoreShard,
    inputs: Tensor,
    *,
    group=None,
) -> Tensor:
    """Run the recurrence with the hidden dimension split across ranks.

    ``inputs`` is the full [B,T,in] batch (replicated); the return is this
    rank's slice of the core's own trajectory, ``cat([y, budget])`` at
    [B,T,2*(H/P)], matching what an unsharded core emits. Reassembly is
    per-field -- concatenate every rank's ``y`` half, then every rank's
    ``budget`` half -- because the shards partition units, and a rank's two
    halves are not adjacent in the unsharded layout.
    """
    from .kernels.batched_runtime import _forward_step

    world = torch_dist.get_world_size(group) if torch_dist.is_initialized() else 1
    local_h = shard["W"].shape[0]
    batch, steps, _ = inputs.shape
    device, dtype = inputs.device, torch.float32

    wx = torch.nn.functional.linear(inputs, shard["W"])
    wgx = torch.nn.functional.linear(inputs, shard["Wg"], shard["Wg_bias"])
    recurrent = torch.cat([shard["Ug"], shard["A"]], dim=0)

    budget = torch.zeros(batch, local_h, device=device, dtype=dtype)
    energy = torch.ones(batch, local_h, device=device, dtype=dtype)
    saturation = torch.zeros(batch, local_h, device=device, dtype=dtype)

    outputs = []
    for t in range(steps):
        y_local = torch.tanh(wx[:, t].float() + budget)
        y_full = _AllGatherHidden.apply(y_local, group) if world > 1 else y_local
        step = _forward_step(
            wx[:, t],
            wgx[:, t],
            recurrent,
            budget,
            energy,
            saturation,
            shard["beta"],
            shard["log_kappa"],
            shard["logit_recover"],
            shard["k_s"],
            shard["k_y"],
            shard["k_b"],
            shard["k_n"],
            shard["k_bias"],
            shard["r_s"],
            shard["r_y"],
            shard["r_b"],
            shard["r_n"],
            shard["r_bias"],
            shard["logit_saturation_decay"].expand(local_h),
            shard["k_saturation"],
            shard["r_saturation"],
            shard["logit_alpha"].reshape(()),
            shard["log_lambda"].reshape(()),
            shard["logit_saturation_suppress"].reshape(()),
            y_full=y_full,
        )
        y, budget, energy, saturation = step[0], step[1], step[2], step[3]
        # Same trajectory element the unsharded core emits: y and the UPDATED
        # budget, in that order.
        outputs.append(torch.cat([y, budget], dim=-1))
    return torch.stack(outputs, dim=1)


def reassemble_tensor_parallel_trajectory(pieces: list[Tensor]) -> Tensor:
    """Rebuild an unsharded trajectory from per-rank slices, in rank order.

    A rank's output is ``cat([y_local, budget_local])``, so the halves have to
    be regrouped rather than simply concatenated: all ranks' ``y`` first, then
    all ranks' ``budget``. Getting this wrong produces a tensor of the right
    shape holding interleaved fields, which is exactly the sort of error that
    survives a shape assertion.
    """
    ys, budgets = [], []
    for piece in pieces:
        half = piece.shape[-1] // 2
        ys.append(piece[..., :half])
        budgets.append(piece[..., half:])
    return torch.cat(ys + budgets, dim=-1)
