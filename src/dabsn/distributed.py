"""Distributed DABSN training with DDP and FSDP parameter sharding."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as torch_dist
import torch.nn as nn
from torch import Tensor

from .checkpoint import dabsn_config_dict, inspect_dabsn, save_dabsn_state
from .model import DABSNBlock, DABSNModel, DABSNSequenceLM, DABSNTaskModel

DABSNModelType = DABSNModel | DABSNTaskModel | DABSNSequenceLM


@dataclass(frozen=True)
class DistributedState:
    """The process-group and device assignment for one DABSN worker."""

    kind: str = "none"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    backend: str | None = None

    @property
    def enabled(self) -> bool:
        return self.kind != "none"

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def parameter_sharded(self) -> bool:
        return self.kind == "fsdp"

    @property
    def gradient_sharded(self) -> bool:
        return self.kind == "fsdp"

    @property
    def optimizer_sharded(self) -> bool:
        return self.kind == "fsdp"

    @property
    def batch_parallel(self) -> bool:
        return self.kind in {"ddp", "fsdp"}

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
        }


class DABSNSequenceModule(nn.Module):
    """Expose full sequence outputs through ``forward`` for DDP and FSDP."""

    def __init__(self, body: DABSNModelType) -> None:
        super().__init__()
        self.body = body

    @property
    def backbone(self):
        return self.body.backbone

    def forward(self, inputs: Tensor, positions: Tensor | None = None) -> Tensor:
        if positions is None:
            return self.body.forward_sequence(inputs)
        hidden = self.body.forward_hidden(inputs)
        return self.body.project_positions(hidden, positions)


def _resolve_device(requested: str | torch.device | None) -> torch.device:
    if requested is None or str(requested) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def setup_distributed(
    kind: str = "none",
    requested_device: str | torch.device | None = "auto",
    *,
    backend: str | None = None,
) -> DistributedState:
    """Initialize a ``torchrun`` worker from its standard environment variables."""

    normalized = (kind or "none").lower()
    if normalized not in {"none", "ddp", "fsdp"}:
        raise ValueError("distributed kind must be none, ddp, or fsdp")
    requested = _resolve_device(requested_device)
    if normalized == "none":
        return DistributedState(device=requested)
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
        if normalized == "fsdp":
            raise RuntimeError("DABSN FSDP requires CUDA; CPU distributed execution may use DDP")
        device = requested
        selected_backend = backend or "gloo"
    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend=selected_backend)
    return DistributedState(
        kind=normalized,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=selected_backend,
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
    if state.kind == "fsdp":
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        return ShardedGradScaler()
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()


def wrap_distributed(
    module: nn.Module,
    state: DistributedState,
    *,
    precision: str = "fp32",
) -> nn.Module:
    """Wrap a sequence-forward module in DDP or block-sharded FSDP."""

    if state.kind == "none":
        return module
    if state.kind == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [state.local_rank] if state.device.type == "cuda" else None
        # The canonical unified read retains parameters that are either used by
        # hard membership decisions or inactive in that read path. They do not
        # receive autograd gradients, so DDP must explicitly permit them.
        return DistributedDataParallel(
            module,
            device_ids=device_ids,
            find_unused_parameters=True,
        )
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    return FullyShardedDataParallel(
        module,
        device_id=state.device,
        mixed_precision=_mixed_precision_policy(precision),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DABSNBlock},
        ),
        limit_all_gathers=True,
        forward_prefetch=True,
        use_orig_params=True,
        sync_module_states=True,
    )


def prepare_distributed_model(
    model: DABSNModelType,
    state: DistributedState,
    *,
    precision: str = "fp32",
) -> nn.Module:
    """Move a DABSN model to its worker device and expose sequence-forward training."""

    sequence_model = DABSNSequenceModule(model.to(state.device))
    return wrap_distributed(sequence_model, state, precision=precision)


def unwrap_dabsn_model(model: nn.Module) -> DABSNModelType:
    """Return the underlying public DABSN model from DDP/FSDP shells."""

    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (DABSNModel, DABSNTaskModel, DABSNSequenceLM)):
            return current
        if isinstance(current, DABSNSequenceModule):
            current = current.body
            continue
        child = getattr(current, "module", None)
        if isinstance(child, nn.Module):
            current = child
            continue
        break
    raise TypeError("distributed wrapper does not contain a DABSN model")


def _is_fsdp(model: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        return False
    return isinstance(model, FullyShardedDataParallel)


def full_model_state_dict(model: nn.Module, state: DistributedState) -> dict[str, Tensor]:
    """Materialize a portable full CPU model state on rank zero."""

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


def _normalize_distributed_state_key(name: str) -> str:
    parts = name.split(".")
    while parts and parts[0] in {"module", "_fsdp_wrapped_module"}:
        parts.pop(0)
    if parts and parts[0] == "body":
        parts.pop(0)
    parts = [part for part in parts if part != "_fsdp_wrapped_module"]
    return ".".join(parts)


def _portable_state_dict(model: nn.Module, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    unwrap_dabsn_model(model)
    portable: dict[str, Tensor] = {}
    for original, value in state_dict.items():
        normalized = _normalize_distributed_state_key(original)
        if not normalized:
            raise KeyError(f"distributed state key {original!r} has no portable DABSN name")
        if normalized in portable:
            raise KeyError(
                f"distributed state keys collide at portable DABSN name {normalized!r}"
            )
        portable[normalized] = value.detach().cpu()
    return portable


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
    scaler: object | None = None,
) -> Path | None:
    """Save one portable SafeTensors model and optional optimizer sidecar.

    This portable mode materializes the full FSDP model and optimizer on rank
    zero. Use :func:`save_sharded_training_checkpoint` for runs whose state
    cannot fit on one host.
    """

    destination = Path(path)
    raw = unwrap_dabsn_model(model)
    full_state = full_model_state_dict(model, state)
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
            _portable_state_dict(model, full_state),
            dabsn_config_dict(raw),
            destination,
            extra=checkpoint_extra,
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


def save_sharded_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    state: DistributedState,
    *,
    step: int,
    extra: dict[str, object] | None = None,
) -> Path | None:
    """Save reshardable model and optimizer state without a rank-zero gather."""

    import json
    import shutil
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    destination = Path(path)
    staging = destination.with_name(destination.name + ".tmp")
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
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=staging,
        no_dist=not state.enabled,
    )
    if state.is_main:
        raw = unwrap_dabsn_model(model)
        manifest = {
            "format": "dabsn-distributed-checkpoint",
            "version": 1,
            "step": int(step),
            "config": dabsn_config_dict(raw),
            "extra": extra or {},
        }
        manifest_path = _training_manifest_path(staging)
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    barrier(state)
    if state.is_main:
        shutil.rmtree(backup, ignore_errors=True)
        if destination.exists():
            os.replace(destination, backup)
        os.replace(staging, destination)
        shutil.rmtree(backup, ignore_errors=True)
    barrier(state)
    return destination if state.is_main else None


def inspect_sharded_training_checkpoint(path: str | Path) -> dict[str, object]:
    """Read the non-tensor manifest for a DABSN distributed checkpoint."""

    import json

    destination = Path(path)
    backup = destination.with_name(destination.name + ".previous")
    if not destination.exists() and backup.exists():
        try:
            os.replace(backup, destination)
        except FileNotFoundError:
            # Another worker recovered the same shared checkpoint first.
            pass
    source = _training_manifest_path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"missing distributed checkpoint manifest: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != "dabsn-distributed-checkpoint" or payload.get("version") != 1:
        raise ValueError(f"not a supported DABSN distributed checkpoint: {path}")
    return payload


def load_sharded_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    state: DistributedState,
) -> dict[str, object]:
    """Restore model and optimizer state, resharding for the active workers."""

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )

    manifest = inspect_sharded_training_checkpoint(path)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=False)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    state_dict = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(
        state_dict,
        checkpoint_id=Path(path),
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
    barrier(state)
    return manifest


def load_sharded_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    state: DistributedState,
) -> dict[str, object]:
    """Restore only model tensors from a DABSN distributed checkpoint."""

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    manifest = inspect_sharded_training_checkpoint(path)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=True)
    model_state = get_model_state_dict(model, options=options)
    state_dict = {"model": model_state}
    dcp.load(
        state_dict,
        checkpoint_id=Path(path),
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
    scaler: object | None = None,
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
    payload = torch.load(source, map_location="cpu", weights_only=False) if state.is_main else None
    transactions_match = True
    if state.is_main:
        model_transaction = inspect_dabsn(path)["extra"].get("training_transaction")
        optimizer_transaction = payload.get("extra", {}).get("training_transaction")
        transactions_match = model_transaction == optimizer_transaction
    if state.enabled:
        flag = torch.tensor(int(transactions_match), device=state.device)
        torch_dist.broadcast(flag, src=0)
        transactions_match = bool(flag.item())
    if not transactions_match:
        raise ValueError(
            "model and optimizer sidecar are from different checkpoint transactions"
        )
    if state.kind == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel

        full_optimizer_state = payload["optimizer"] if state.is_main else None
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
                payload = objects[0]
        optimizer.load_state_dict(payload["optimizer"])
    scaler_state = payload.get("scaler") if state.is_main else None
    if state.enabled:
        objects = [scaler_state]
        torch_dist.broadcast_object_list(objects, src=0, device=state.device)
        scaler_state = objects[0]
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    step = payload.get("step") if state.is_main else None
    if state.enabled:
        value = torch.tensor(-1 if step is None else int(step), device=state.device)
        torch_dist.broadcast(value, src=0)
        step = None if int(value.item()) < 0 else int(value.item())
    return None if step is None else int(step)


def shard_batch(tensor: Tensor, state: DistributedState) -> Tensor:
    """Select this worker's contiguous share of a global batch."""

    if not state.batch_parallel:
        return tensor.to(state.device)
    if tensor.shape[0] < state.world_size or tensor.shape[0] % state.world_size:
        raise ValueError(
            f"global batch {tensor.shape[0]} must be divisible by world size {state.world_size}"
        )
    return tensor.chunk(state.world_size, dim=0)[state.rank].to(state.device)


def no_sync_context(model: nn.Module, *, synchronize: bool):
    """Suppress DDP/FSDP gradient collectives for non-final accumulation microbatches."""

    if synchronize or not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


def clip_grad_norm(model: nn.Module, max_norm: float) -> Tensor:
    """Clip global FSDP gradients correctly, or ordinary gradients otherwise."""

    if _is_fsdp(model):
        return model.clip_grad_norm_(float(max_norm))
    return torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))


def barrier(state: DistributedState) -> None:
    if state.enabled and torch_dist.is_initialized():
        torch_dist.barrier()


def cleanup_distributed(state: DistributedState) -> None:
    if state.enabled and torch_dist.is_initialized():
        torch_dist.destroy_process_group()
