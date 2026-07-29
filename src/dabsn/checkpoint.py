"""Self-describing, non-pickle DABSN model checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .config import DABSNConfig, DABSNLayerSpec
from .model import (
    DABSNModel,
    DABSNSequenceLM,
    DABSNTaskModel,
    build_dabsn_from_config,
)

Model = DABSNModel | DABSNTaskModel | DABSNSequenceLM

CHECKPOINT_FORMAT = "dabsn-model"
CHECKPOINT_VERSION = 1


def _introspect_config(model: DABSNModel | DABSNTaskModel) -> DABSNConfig:
    if isinstance(model, DABSNTaskModel):
        body = model.body
        backbone = body.backbone
        return DABSNConfig(
            input_dim=int(model.raw_input_dim),
            out_dim=int(model.out_dim),
            hidden_dim=int(model.model_input_dim),
            input_adapter=model.input_adapter_kind,
            output_adapter=body.output_adapter_kind,
            layers=[spec.to_metadata() for spec in backbone.layer_specs],
            residual=bool(backbone.residual),
            mlp_ratio=backbone.mlp_ratio,
        )
    return DABSNConfig(
        input_dim=int(model.input_dim),
        out_dim=int(model.out_dim),
        hidden_dim=int(model.input_dim),
        input_adapter="identity",
        output_adapter=model.output_adapter_kind,
        layers=[spec.to_metadata() for spec in model.backbone.layer_specs],
        residual=bool(model.backbone.residual),
        mlp_ratio=model.backbone.mlp_ratio,
    )


def dabsn_config_dict(model: Model) -> dict[str, object]:
    """Return the complete architecture needed to reconstruct ``model``."""

    if isinstance(model, DABSNSequenceLM):
        return {
            "model_kind": "sequence_lm",
            "vocab": int(model.vocab),
            "hidden_dim": int(model.hidden_dim),
            "depth": int(model.depth),
            "state_dim": model.state_dim,
            "layers": [spec.to_metadata() for spec in model.layers],
            "tie_embeddings": bool(model.tie_embeddings),
            "grad_checkpoint": bool(model.backbone.grad_checkpoint),
            "residual": bool(model.residual),
            "mlp_ratio": model.mlp_ratio,
        }
    config = getattr(model, "_dabsn_config", None) or _introspect_config(model)
    data = {name: getattr(config, name) for name in DABSNConfig.__dataclass_fields__}
    data["layers"] = [
        spec.to_metadata() if isinstance(spec, DABSNLayerSpec) else dict(spec)
        for spec in config.layer_specs()
    ]
    data["model_kind"] = "task" if isinstance(model, DABSNTaskModel) else "model"
    data["grad_checkpoint"] = bool(model.backbone.grad_checkpoint)
    return data


def build_dabsn_from_checkpoint_config(config: Mapping[str, object]) -> Model:
    """Construct a model from clean checkpoint metadata without loading weights."""

    if config.get("model_kind") == "sequence_lm":
        return DABSNSequenceLM(
            vocab=int(config["vocab"]),
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            layers=config["layers"],
            state_dim=config.get("state_dim"),
            tie_embeddings=bool(config.get("tie_embeddings", False)),
            grad_checkpoint=bool(config.get("grad_checkpoint", False)),
            residual=bool(config.get("residual", False)),
            mlp_ratio=config.get("mlp_ratio"),
        )
    config_fields = {
        name: value
        for name, value in config.items()
        if name in DABSNConfig.__dataclass_fields__
    }
    return build_dabsn_from_config(
        DABSNConfig(**config_fields),
        grad_checkpoint=bool(config.get("grad_checkpoint", False)),
    )


def _storage_identity(tensor: Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    try:
        pointer = tensor.untyped_storage().data_ptr()
    except AttributeError:
        pointer = tensor.storage().data_ptr()
    return pointer, tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride())


def _deduplicate_state_dict(
    state_dict: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, str]]:
    saved: dict[str, Tensor] = {}
    shared: dict[str, str] = {}
    seen: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], str] = {}
    for name, tensor in state_dict.items():
        identity = _storage_identity(tensor)
        if identity in seen:
            shared[name] = seen[identity]
        else:
            seen[identity] = name
            saved[name] = tensor.detach().cpu().contiguous()
    return saved, shared


def save_dabsn_state(
    state_dict: Mapping[str, Tensor],
    config: Mapping[str, object],
    path: str | Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Atomically write a clean DABSN SafeTensors checkpoint."""

    from safetensors.torch import save_file

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors, shared = _deduplicate_state_dict(state_dict)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "version": str(CHECKPOINT_VERSION),
        "config": json.dumps(dict(config), sort_keys=True, separators=(",", ":")),
        "extra": json.dumps(dict(extra or {}), sort_keys=True, separators=(",", ":")),
        "shared": json.dumps(shared, sort_keys=True, separators=(",", ":")),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_dabsn(
    model: Model,
    path: str | Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Save ``model`` as a self-describing SafeTensors checkpoint."""

    save_dabsn_state(model.state_dict(), dabsn_config_dict(model), path, extra=extra)


def inspect_dabsn(path: str | Path) -> dict[str, object]:
    """Read clean checkpoint metadata without allocating model tensors."""

    from safetensors import safe_open

    source = Path(path)
    try:
        with safe_open(str(source), framework="pt") as checkpoint:
            metadata = checkpoint.metadata() or {}
    except Exception as exc:
        raise ValueError(f"{source} is not a readable SafeTensors checkpoint") from exc
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{source} is not a {CHECKPOINT_FORMAT} checkpoint")
    version = int(metadata.get("version", "0"))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported {CHECKPOINT_FORMAT} version {version}; expected {CHECKPOINT_VERSION}"
        )
    try:
        config = json.loads(metadata["config"])
        extra = json.loads(metadata.get("extra", "{}"))
        shared = json.loads(metadata.get("shared", "{}"))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} has invalid DABSN metadata") from exc
    if not isinstance(config, dict) or not isinstance(extra, dict) or not isinstance(shared, dict):
        raise ValueError(f"{source} has invalid DABSN metadata objects")
    return {
        "format": CHECKPOINT_FORMAT,
        "version": version,
        "config": config,
        "extra": extra,
        "shared": shared,
    }


def load_dabsn(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> Model:
    """Load only the clean DABSN SafeTensors format; no legacy pickle fallback."""

    from safetensors.torch import load_file

    source = Path(path)
    metadata = inspect_dabsn(source)
    device = torch.device("cpu" if map_location is None else map_location)
    model = build_dabsn_from_checkpoint_config(metadata["config"]).to(device)
    state_dict = load_file(str(source), device=str(device))
    for duplicate, original in metadata["shared"].items():
        if original not in state_dict:
            raise ValueError(
                f"checkpoint shared tensor {duplicate!r} refers to missing {original!r}"
            )
        state_dict[duplicate] = state_dict[original]
    model.load_state_dict(state_dict, strict=strict)
    return model


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "build_dabsn_from_checkpoint_config",
    "dabsn_config_dict",
    "inspect_dabsn",
    "load_dabsn",
    "save_dabsn",
    "save_dabsn_state",
]
