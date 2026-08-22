"""Self-describing, non-pickle DABSN model checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict, cast

import torch
from torch import Tensor

from .config import DABSNConfig, DABSNLayerSpec
from .events import EventCode, emit_event
from .graph import DABSNGraph
from .model import (
    DABSNModel,
    DABSNSequenceLM,
    DABSNTaskModel,
    build_dabsn_from_config,
)

Model = DABSNModel | DABSNTaskModel | DABSNSequenceLM
Artifact = Model | DABSNGraph

CHECKPOINT_FORMAT = "dabsn-model"
CHECKPOINT_VERSION = 2
MAX_METADATA_BYTES = 1_048_576
MAX_METADATA_DEPTH = 32
MAX_METADATA_NODES = 100_000
MAX_METADATA_STRING = 65_536


class CheckpointInspection(TypedDict):
    format: str
    version: int
    config: dict[str, object]
    extra: dict[str, object]
    shared: dict[str, str]
    manifest: dict[str, object]


def _int(value: object) -> int:
    return int(cast(Any, value))


def _float(value: object) -> float:
    return float(cast(Any, value))


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


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
            mlp_middle_depth=backbone.mlp_middle_depth,
            mlp_depth_index=backbone.mlp_depth_index,
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
        mlp_middle_depth=model.backbone.mlp_middle_depth,
        mlp_depth_index=model.backbone.mlp_depth_index,
    )


def dabsn_config_dict(model: Model) -> dict[str, object]:
    """Return the complete architecture needed to reconstruct ``model``."""

    if isinstance(model, DABSNSequenceLM):
        if model.block_name == "dabsn-graph":
            return {
                "model_kind": "sequence_graph",
                "vocab": int(model.vocab),
                "hidden_dim": int(model.hidden_dim),
                "tie_embeddings": bool(model.tie_embeddings),
                "grad_checkpoint": bool(model.backbone.grad_checkpoint),
            }
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
            "mlp_middle_depth": int(model.mlp_middle_depth),
            "mlp_depth_index": int(model.mlp_depth_index),
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


def graph_config_dict(graph: DABSNGraph) -> dict[str, object]:
    """Return the provider-only construction metadata for a raw component graph."""

    return {
        "model_kind": "graph",
        "components": [_spec_to_dict(spec) for spec in graph.component_specs(portable=True)],
        "input_contract": graph.input_contract.to_dict(),
        "output_contract": graph.output_contract.to_dict(),
        "require_world_builder": graph.require_world_builder,
    }


def build_graph_from_config(
    config: Mapping[str, object],
    *,
    trusted_providers: Sequence[str] = (),
) -> DABSNGraph:
    """Resolve a domain-neutral graph from clean ordered provider specifications."""

    from .components import ComponentSpec, ValueContract, component_registry
    from .providers import register_builtin_components

    raw_components = config.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("graph config requires a non-empty components list")
    register_builtin_components()
    component_registry.discover()
    for key in trusted_providers:
        component_registry.authorize(key)
    bindings = []
    for index, raw in enumerate(raw_components):
        if not isinstance(raw, Mapping):
            raise ValueError(f"graph components[{index}] must be an object")
        provider_key = raw.get("provider_key")
        component_id = raw.get("component_id")
        component_config = raw.get("config")
        if not isinstance(provider_key, str) or not provider_key:
            raise ValueError(f"graph components[{index}] requires provider_key")
        if not isinstance(component_id, str) or not component_id:
            raise ValueError(f"graph components[{index}] requires component_id")
        if not isinstance(component_config, Mapping):
            raise ValueError(f"graph components[{index}] config must be an object")
        bindings.append(
            component_registry.build(
                ComponentSpec(
                    component_id=component_id,
                    provider_key=provider_key,
                    config=dict(component_config),
                    component_abi_version=_int(raw.get("component_abi_version", 2)),
                    config_schema_version=_int(raw.get("config_schema_version", 1)),
                    provider_distribution=cast(str | None, raw.get("provider_distribution")),
                    provider_version=cast(str | None, raw.get("provider_version")),
                )
            )
        )
    raw_input = config.get("input_contract")
    input_contract = (
        None
        if raw_input is None
        else ValueContract.from_dict(_mapping(raw_input, "graph input contract"))
    )
    graph = DABSNGraph(
        bindings,
        input_contract=input_contract,
        require_world_builder=bool(config.get("require_world_builder", False)),
    )
    raw_output = config.get("output_contract")
    if raw_output is not None and graph.output_contract.to_dict() != dict(
        _mapping(raw_output, "graph output contract")
    ):
        raise ValueError("resolved graph output contract does not match configuration")
    return graph


def build_dabsn_from_checkpoint_config(
    config: Mapping[str, object],
    *,
    graph_components: Sequence[Mapping[str, object]] | None = None,
    trusted_providers: Sequence[str] = (),
) -> Model:
    """Construct a model from clean checkpoint metadata without loading weights."""

    if config.get("model_kind") == "sequence_graph":
        if graph_components is None:
            raise ValueError("sequence_graph checkpoint is missing its ordered graph")
        from .components import ComponentSpec, component_registry
        from .graph import DABSNGraph
        from .providers import register_builtin_components

        register_builtin_components()
        component_registry.discover()
        for key in trusted_providers:
            component_registry.authorize(key)
        bindings = []
        for raw in graph_components:
            bindings.append(
                component_registry.build(
                    ComponentSpec(
                        component_id=str(raw["component_id"]),
                        provider_key=str(raw["provider_key"]),
                        config=dict(_mapping(raw["config"], "component config")),
                        component_abi_version=_int(raw["component_abi_version"]),
                        config_schema_version=_int(raw["config_schema_version"]),
                        provider_distribution=cast(str | None, raw.get("provider_distribution")),
                        provider_version=cast(str | None, raw.get("provider_version")),
                    )
                )
            )
        graph = DABSNGraph(bindings, require_world_builder=True)
        model = DABSNSequenceLM.from_graph(
            graph,
            vocab=_int(config["vocab"]),
            tie_embeddings=bool(config.get("tie_embeddings", False)),
        )
        model.backbone.grad_checkpoint = bool(config.get("grad_checkpoint", False))
        model.backbone.graph.set_activation_checkpointing(model.backbone.grad_checkpoint)
        return model
    if config.get("model_kind") == "sequence_lm":
        state_dim = config.get("state_dim")
        ratio = config.get("mlp_ratio")
        return DABSNSequenceLM(
            vocab=_int(config["vocab"]),
            hidden_dim=_int(config["hidden_dim"]),
            depth=_int(config["depth"]),
            layers=cast(str | Sequence[object] | None, config["layers"]),
            state_dim=None if state_dim is None else _int(state_dim),
            tie_embeddings=bool(config.get("tie_embeddings", False)),
            grad_checkpoint=bool(config.get("grad_checkpoint", False)),
            residual=bool(config.get("residual", False)),
            mlp_ratio=None if ratio is None else _float(ratio),
            mlp_middle_depth=_int(config.get("mlp_middle_depth", 0)),
            mlp_depth_index=_int(config.get("mlp_depth_index", 0)),
        )
    state_dim = config.get("state_dim")
    task = config.get("task")
    ratio = config.get("mlp_ratio")
    layers = cast(list[DABSNLayerSpec | Mapping[str, object]], config.get("layers", []))
    return build_dabsn_from_config(
        DABSNConfig(
            input_dim=_int(config["input_dim"]),
            out_dim=_int(config["out_dim"]),
            hidden_dim=_int(config.get("hidden_dim", 512)),
            depth=_int(config.get("depth", 1)),
            geometry=str(config.get("geometry", "seq")),
            state_dim=None if state_dim is None else _int(state_dim),
            input_adapter=str(config.get("input_adapter", "identity")),
            output_adapter=str(config.get("output_adapter", "field")),
            task=None if task is None else str(task),
            layers=layers,
            residual=bool(config.get("residual", False)),
            mlp_ratio=None if ratio is None else _float(ratio),
            mlp_middle_depth=_int(config.get("mlp_middle_depth", 0)),
            mlp_depth_index=_int(config.get("mlp_depth_index", 0)),
        ),
        grad_checkpoint=bool(config.get("grad_checkpoint", False)),
    )


def _storage_identity(tensor: Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    try:
        pointer = tensor.untyped_storage().data_ptr()
    except AttributeError:
        pointer = tensor.storage().data_ptr()
    return int(pointer), int(tensor.storage_offset()), tuple(tensor.shape), tuple(tensor.stride())


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


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_metadata_value(
    value: object, *, depth: int = 0, nodes: list[int] | None = None
) -> None:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > MAX_METADATA_NODES:
        raise ValueError("DABSN metadata exceeds the node limit")
    if depth > MAX_METADATA_DEPTH:
        raise ValueError("DABSN metadata exceeds the nesting limit")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("DABSN metadata contains a non-finite number")
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_METADATA_STRING:
            raise ValueError("DABSN metadata string exceeds the size limit")
        return
    if isinstance(value, list):
        for item in value:
            _validate_metadata_value(item, depth=depth + 1, nodes=nodes)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("DABSN metadata object keys must be strings")
            _validate_metadata_value(key, depth=depth + 1, nodes=nodes)
            _validate_metadata_value(item, depth=depth + 1, nodes=nodes)
        return
    raise ValueError(f"DABSN metadata contains unsupported type {type(value).__name__}")


def _bounded_json_load(raw: str, label: str) -> object:
    if len(raw.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError(f"DABSN {label} metadata exceeds the size limit")
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid DABSN {label} JSON") from exc
    _validate_metadata_value(value)
    return value


def _legacy_graph_components(config: Mapping[str, object]) -> list[dict[str, object]]:
    layers = list(cast(Sequence[object], config.get("layers", ())))
    if not layers:
        return []
    width = _int(config.get("hidden_dim", config.get("input_dim", 0)))
    ratio = config.get("mlp_ratio")
    middle_depth = _int(config.get("mlp_middle_depth", 0))
    middle_index = _int(config.get("mlp_depth_index", 0))
    components: list[dict[str, object]] = []
    middle_width = None
    for index, raw in enumerate(layers):
        layer = dict(_mapping(raw, "legacy layer"))
        hidden = _int(layer["hidden_dim"])
        state = layer.get("state_dim")
        state = hidden if state is None else _int(state)
        components.append(
            {
                "component_id": f"dabsn.{index}",
                "provider_key": "dabsn:block",
                "provider_distribution": "dabsn",
                "provider_version": "2.0.0",
                "component_abi_version": 2,
                "config_schema_version": 1,
                "config": {
                    "input_dim": width,
                    "hidden_dim": hidden,
                    "state_dim": state,
                    "read_geometry": str(layer.get("read_geometry", "field")),
                    "residual": bool(config.get("residual", False)),
                },
            }
        )
        width = hidden
        if ratio is not None:
            components.append(
                {
                    "component_id": f"legacy-inline-mlp.{index}",
                    "provider_key": "dabsn:residual_mlp",
                    "provider_distribution": "dabsn",
                    "provider_version": "2.0.0",
                    "component_abi_version": 2,
                    "config_schema_version": 1,
                    "config": {"dim": hidden, "ratio": _float(ratio)},
                }
            )
        if index == middle_index:
            middle_width = hidden
            for mlp_index in range(middle_depth):
                components.append(
                    {
                        "component_id": f"legacy-middle-mlp.{mlp_index}",
                        "provider_key": "dabsn:residual_mlp",
                        "provider_distribution": "dabsn",
                        "provider_version": "2.0.0",
                        "component_abi_version": 2,
                        "config_schema_version": 1,
                        "config": {"dim": middle_width, "ratio": _float(ratio)},
                    }
                )
    return components


def _spec_to_dict(spec) -> dict[str, object]:
    return {
        "component_id": spec.component_id,
        "provider_key": spec.provider_key,
        "provider_distribution": spec.provider_distribution,
        "provider_version": spec.provider_version,
        "component_abi_version": spec.component_abi_version,
        "config_schema_version": spec.config_schema_version,
        "config": dict(spec.config),
    }


def _manifest(
    config: Mapping[str, object],
    state_dict: Mapping[str, Tensor],
    shared: Mapping[str, str],
    *,
    model: Artifact | None = None,
) -> dict[str, object]:
    if isinstance(model, DABSNGraph):
        graph_components = [
            _spec_to_dict(spec) for spec in model.component_specs(portable=True)
        ]
        contract_fingerprint = _graph_contract_fingerprint(model)
        assert contract_fingerprint is not None
        construction = "raw-graph"
    elif model is not None and getattr(model, "block_name", None) == "dabsn-graph":
        graph_components = [
            _spec_to_dict(spec) for spec in cast(Any, model).graph.component_specs(portable=True)
        ]
        contract_fingerprint = _graph_contract_fingerprint(model)
        assert contract_fingerprint is not None
        construction = "graph"
    else:
        graph_components = _legacy_graph_components(config)
        contract_fingerprint = _graph_contract_fingerprint(model) if model is not None else None
        if contract_fingerprint is None:
            contract_fingerprint = hashlib.sha256(
                _canonical_json(graph_components).encode("utf-8")
            ).hexdigest()
        construction = "legacy-wrapper"
    memory_owners = [
        component["component_id"]
        for component in graph_components
        if component["provider_key"] == "dabsn:block"
    ]
    manifest = {
        "format": CHECKPOINT_FORMAT,
        "schema_version": CHECKPOINT_VERSION,
        "construction": construction,
        "framework_version": "2.0.0",
        "graph": graph_components,
        "contract_fingerprint": contract_fingerprint,
        "parameter_namespace_map": {name: name for name in state_dict},
        "tied_tensors": dict(shared),
        "dabsn_memory_owners": memory_owners,
    }
    _validate_metadata_value(manifest)
    return manifest


def _graph_contract_fingerprint(model: Artifact) -> str | None:
    graph = model if isinstance(model, DABSNGraph) else getattr(model, "graph", None)
    if graph is None:
        return None
    return hashlib.sha256(
        _canonical_json(
            {
                "input": graph.input_contract.to_dict(),
                "output": graph.output_contract.to_dict(),
            }
        ).encode("utf-8")
    ).hexdigest()


def save_dabsn_state(
    state_dict: Mapping[str, Tensor],
    config: Mapping[str, object],
    path: str | Path,
    *,
    extra: Mapping[str, object] | None = None,
    _model: Artifact | None = None,
) -> None:
    """Validate, fsync, and atomically write a portable SafeTensors-v2 file."""

    from safetensors.torch import save_file

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors, shared = _deduplicate_state_dict(state_dict)
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint tensor names must be non-empty strings")
        if not isinstance(tensor, Tensor):
            raise TypeError(f"checkpoint value {name!r} is not a tensor")
    config_data = dict(config)
    extra_data = dict(extra or {})
    _validate_metadata_value(config_data)
    _validate_metadata_value(extra_data)
    manifest = _manifest(config_data, state_dict, shared, model=_model)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "version": str(CHECKPOINT_VERSION),
        "config": _canonical_json(config_data),
        "extra": _canonical_json(extra_data),
        "shared": _canonical_json(shared),
        "manifest": _canonical_json(manifest),
    }
    for label, raw in metadata.items():
        if len(raw.encode("utf-8")) > MAX_METADATA_BYTES:
            raise ValueError(f"DABSN {label} metadata exceeds the size limit")
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        emit_event(
            EventCode.CHECKPOINT_TRANSACTION,
            component_id=None,
            phase="staging",
            destination=str(destination),
            temporary=str(temporary),
        )
        save_file(tensors, str(temporary), metadata=metadata)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            # Some filesystems do not support directory fsync. The file itself
            # is still flushed before atomic replacement.
            pass
        emit_event(
            EventCode.CHECKPOINT_TRANSACTION,
            component_id=None,
            phase="committed",
            destination=str(destination),
        )
    finally:
        temporary.unlink(missing_ok=True)


def save_dabsn(
    model: Model,
    path: str | Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Save ``model`` as a self-describing SafeTensors checkpoint."""

    save_dabsn_state(
        model.state_dict(),
        dabsn_config_dict(model),
        path,
        extra=extra,
        _model=model,
    )


def save_graph(
    graph: DABSNGraph,
    path: str | Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Save a domain-neutral component graph as a portable SafeTensors artifact."""

    save_dabsn_state(
        graph.state_dict(),
        graph_config_dict(graph),
        path,
        extra=extra,
        _model=graph,
    )


def inspect_dabsn(path: str | Path) -> CheckpointInspection:
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
    if version not in {1, CHECKPOINT_VERSION}:
        raise ValueError(f"unsupported {CHECKPOINT_FORMAT} version {version}; supported: 1 and 2")
    try:
        config = _bounded_json_load(metadata["config"], "config")
        extra = _bounded_json_load(metadata.get("extra", "{}"), "extra")
        shared = _bounded_json_load(metadata.get("shared", "{}"), "shared")
        manifest = _bounded_json_load(metadata["manifest"], "manifest") if version == 2 else None
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} has invalid DABSN metadata") from exc
    if not isinstance(config, dict) or not isinstance(extra, dict) or not isinstance(shared, dict):
        raise ValueError(f"{source} has invalid DABSN metadata objects")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in shared.items()):
        raise ValueError(f"{source} has invalid shared-tensor metadata")
    if version == 1:
        manifest = _manifest(config, {}, shared)
        manifest["schema_version"] = 1
        manifest["construction"] = "v1-migration-view"
    if not isinstance(manifest, dict):
        raise ValueError(f"{source} has invalid DABSN manifest")
    if manifest.get("schema_version") != version:
        raise ValueError(f"{source} metadata versions disagree")
    return {
        "format": CHECKPOINT_FORMAT,
        "version": version,
        "config": cast(dict[str, object], config),
        "extra": cast(dict[str, object], extra),
        "shared": cast(dict[str, str], shared),
        "manifest": cast(dict[str, object], manifest),
    }


def load_dabsn(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    strict: bool = True,
    trusted_providers: Sequence[str] = (),
) -> Model:
    """Load only the clean DABSN SafeTensors format; no legacy pickle fallback."""

    from safetensors.torch import load_file

    source = Path(path)
    metadata = inspect_dabsn(source)
    if metadata["config"].get("model_kind") == "graph":
        raise TypeError("raw graph artifact requires load_graph(), not load_dabsn()")
    device = torch.device("cpu" if map_location is None else map_location)
    raw_graph = metadata["manifest"].get("graph")
    graph_components = (
        None if raw_graph is None else cast(Sequence[Mapping[str, object]], raw_graph)
    )
    model = build_dabsn_from_checkpoint_config(
        metadata["config"],
        graph_components=graph_components,
        trusted_providers=trusted_providers,
    ).to(device)
    state_dict = load_file(str(source), device=str(device))
    # Drop parameters removed from the model but present in older checkpoints,
    # so legacy files still load under strict=True.
    for stale in [
        k for k in state_dict if k.rsplit(".", 1)[-1] in {"long_decay", "adm_train_band"}
    ]:
        del state_dict[stale]
    for duplicate, original in metadata["shared"].items():
        if original not in state_dict:
            raise ValueError(
                f"checkpoint shared tensor {duplicate!r} refers to missing {original!r}"
            )
        state_dict[duplicate] = state_dict[original]
    namespace = metadata["manifest"].get("parameter_namespace_map", {})
    if metadata["version"] == 2:
        if not isinstance(namespace, dict):
            raise ValueError("checkpoint parameter namespace map is invalid")
        missing_names = set(namespace) - set(state_dict)
        if missing_names:
            raise ValueError(
                f"checkpoint namespace refers to missing tensors: {sorted(missing_names)}"
            )
        state_dict = {str(namespace.get(name, name)): tensor for name, tensor in state_dict.items()}
        expected_fingerprint = metadata["manifest"].get("contract_fingerprint")
        actual_fingerprint = _graph_contract_fingerprint(model)
        if actual_fingerprint is not None and actual_fingerprint != expected_fingerprint:
            raise ValueError("reconstructed graph contract fingerprint does not match checkpoint")
    model.load_state_dict(state_dict, strict=strict)
    return model


def load_graph(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    strict: bool = True,
    trusted_providers: Sequence[str] = (),
) -> DABSNGraph:
    """Reconstruct a domain-neutral graph from clean provider metadata and tensors."""

    from safetensors.torch import load_file

    source = Path(path)
    metadata = inspect_dabsn(source)
    config = metadata["config"]
    if config.get("model_kind") != "graph":
        raise TypeError("model artifact requires load_dabsn(), not load_graph()")
    raw_components = metadata["manifest"].get("graph")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("raw graph checkpoint is missing its ordered component specs")
    graph = build_graph_from_config(
        {**config, "components": raw_components},
        trusted_providers=trusted_providers,
    )
    device = torch.device("cpu" if map_location is None else map_location)
    graph = graph.to(device)
    expected_fingerprint = metadata["manifest"].get("contract_fingerprint")
    if _graph_contract_fingerprint(graph) != expected_fingerprint:
        raise ValueError("reconstructed graph contract fingerprint does not match checkpoint")
    state_dict = load_file(str(source), device=str(device))
    for duplicate, original in metadata["shared"].items():
        if original not in state_dict:
            raise ValueError(
                f"checkpoint shared tensor {duplicate!r} refers to missing {original!r}"
            )
        state_dict[duplicate] = state_dict[original]
    namespace = metadata["manifest"].get("parameter_namespace_map", {})
    if not isinstance(namespace, dict):
        raise ValueError("checkpoint parameter namespace map is invalid")
    missing_names = set(namespace) - set(state_dict)
    if missing_names:
        raise ValueError(
            f"checkpoint namespace refers to missing tensors: {sorted(missing_names)}"
        )
    remapped = {str(namespace.get(name, name)): tensor for name, tensor in state_dict.items()}
    graph.load_state_dict(remapped, strict=strict)
    return graph


def artifact_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrate_dabsn_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
    trusted_providers: Sequence[str] = (),
) -> str:
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("checkpoint migration never rewrites an artifact in place")
    metadata = inspect_dabsn(source_path)
    model = load_dabsn(
        source_path,
        map_location=map_location,
        trusted_providers=trusted_providers,
    )
    save_dabsn(model, destination_path, extra=metadata["extra"])
    return artifact_digest(destination_path)


def import_prototype_moe_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> str:
    """Import the historical ``tools.dabsn_moe_source.load_moe`` layout.

    The source remains untouched.  Only the known single-MoE-after-each-DABSN
    prototype schema is accepted; ambiguous variants fail with a named error.
    """

    from safetensors.torch import load_file

    from .components import ComponentSpec, component_registry
    from .graph import DABSNGraph
    from .providers import register_builtin_components

    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("prototype MoE import never rewrites an artifact in place")
    metadata = inspect_dabsn(source_path)
    config = metadata["config"]
    experts = _int(config.get("moe_experts") or 0)
    if experts <= 0:
        raise ValueError("source is not a prototype load_moe checkpoint")
    if config.get("model_kind") != "sequence_lm":
        raise ValueError("prototype MoE import supports sequence_lm artifacts only")
    if config.get("mlp_ratio") is not None or _int(config.get("mlp_middle_depth", 0)):
        raise ValueError("prototype MoE plus legacy MLP variants require a named migration")
    if bool(config.get("residual", False)):
        raise ValueError("prototype MoE with an inner DABSN residual is ambiguous")

    top_k = _int(config["moe_top_k"])
    ratio = _float(config.get("moe_ratio") or 8.0)
    aux_coefficient = _float(config.get("moe_aux_coeff") or 0.0)
    stack_residual = bool(config.get("moe_stack_residual", True))
    layers = list(cast(Sequence[Mapping[str, object]], config["layers"]))
    width = _int(config["hidden_dim"])
    register_builtin_components()
    bindings = []
    for index, raw_layer in enumerate(layers):
        layer = dict(raw_layer)
        hidden = _int(layer["hidden_dim"])
        state = _int(layer.get("state_dim") or hidden)
        bindings.append(
            component_registry.build(
                ComponentSpec(
                    f"dabsn.{index}",
                    "dabsn:block",
                    {
                        "input_dim": width,
                        "hidden_dim": hidden,
                        "state_dim": state,
                        "read_geometry": str(layer.get("read_geometry", "field")),
                        "residual": stack_residual,
                    },
                )
            )
        )
        bindings.append(
            component_registry.build(
                ComponentSpec(
                    f"prototype-moe.{index}",
                    "dabsn:sparse_moe",
                    {
                        "hidden_dim": hidden,
                        "experts": experts,
                        "top_k": top_k,
                        "inner_dim": int(round(ratio * hidden)),
                        "router": "switch",
                        # The native router normalizes top-k assignment share to
                        # one; the prototype's f_i summed to K.
                        "balance_coefficient": aux_coefficient * top_k,
                        "normalization": "rmsnorm",
                        "residual": True,
                        "routing_granularity": "individual_h",
                        "backend": "auto",
                        "zero_output": False,
                    },
                )
            )
        )
        width = hidden
    graph = DABSNGraph(bindings, require_world_builder=True)
    model = DABSNSequenceLM.from_graph(
        graph,
        vocab=_int(config["vocab"]),
        tie_embeddings=bool(config.get("tie_embeddings", False)),
    ).to(torch.device("cpu" if map_location is None else map_location))

    source_state = load_file(
        str(source_path),
        device=str(torch.device("cpu" if map_location is None else map_location)),
    )
    for duplicate, original in metadata["shared"].items():
        source_state[duplicate] = source_state[original]
    converted: dict[str, Tensor] = {}
    consumed: set[str] = set()

    def copy(source_name: str, target_name: str, *, transpose: bool = False) -> None:
        if source_name not in source_state:
            raise ValueError(f"prototype MoE tensor is missing: {source_name}")
        value = source_state[source_name]
        converted[target_name] = value.transpose(-2, -1) if transpose else value
        consumed.add(source_name)

    copy("embed.weight", "embed.weight")
    copy("readout.weight", "readout.weight")
    copy("readout.bias", "readout.bias")
    target_state = model.state_dict()
    for index, raw_layer in enumerate(layers):
        native_dabsn = f"backbone.graph.components.{2 * index}."
        prototype_dabsn = f"backbone.blocks.{index}.dabsn."
        for target_name in target_state:
            if not target_name.startswith(native_dabsn):
                continue
            suffix = target_name[len(native_dabsn) :]
            if suffix == "residual_skip.weight":
                copy(
                    f"backbone.blocks.{index}.stack_skip.weight",
                    target_name,
                )
            else:
                copy(prototype_dabsn + suffix, target_name)

        native_moe = f"backbone.graph.components.{2 * index + 1}."
        prototype_moe = f"backbone.blocks.{index}.moe."
        copy(prototype_moe + "norm.weight", native_moe + "normalization.weight")
        copy(prototype_moe + "router.weight", native_moe + "router.proj.weight")
        first = []
        second = []
        for expert in range(experts):
            first_name = prototype_moe + f"experts.{expert}.fc1.weight"
            second_name = prototype_moe + f"experts.{expert}.fc2.weight"
            first.append(source_state[first_name].transpose(-2, -1))
            second.append(source_state[second_name].transpose(-2, -1))
            consumed.update((first_name, second_name))
        converted[native_moe + "expert_group.w1"] = torch.stack(first)
        converted[native_moe + "expert_group.w2"] = torch.stack(second)

    missing = set(target_state) - set(converted)
    unexpected = set(source_state) - consumed
    if missing or unexpected:
        raise ValueError(
            "prototype MoE namespace mismatch "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    model.load_state_dict(converted, strict=True)
    save_dabsn(
        model,
        destination_path,
        extra={
            "migration": "prototype-load_moe-to-native-v2",
            "source_sha256": artifact_digest(source_path),
        },
    )
    return artifact_digest(destination_path)


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "MAX_METADATA_BYTES",
    "artifact_digest",
    "build_dabsn_from_checkpoint_config",
    "dabsn_config_dict",
    "inspect_dabsn",
    "import_prototype_moe_checkpoint",
    "load_dabsn",
    "migrate_dabsn_checkpoint",
    "save_dabsn",
    "save_dabsn_state",
]
