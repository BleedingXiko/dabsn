"""Immutable public DABSN construction contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

DABSN_ARCH = "dabsn"


@dataclass(frozen=True)
class DABSNLayerSpec:
    """Width and read geometry for one canonical DABSN block."""

    hidden_dim: int
    state_dim: int | None = None
    read_geometry: str = "field"

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.state_dim is not None and self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.read_geometry not in {"seq", "field", "hybrid"}:
            raise ValueError("read_geometry must be seq, field, or hybrid")

    @property
    def resolved_state_dim(self) -> int:
        return self.hidden_dim if self.state_dim is None else self.state_dim

    def to_metadata(self) -> dict[str, object]:
        return {
            "hidden_dim": int(self.hidden_dim),
            "state_dim": None if self.state_dim is None else int(self.state_dim),
            "read_geometry": self.read_geometry,
        }


def parse_dabsn_layer_specs(spec: str | Sequence[object] | None) -> list[DABSNLayerSpec]:
    """Parse compact, JSON, file-backed, or mapping layer specifications."""

    if spec is None:
        return []
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return []
        if text.startswith("@"):
            return parse_dabsn_layer_specs(Path(text[1:]).read_text(encoding="utf-8"))
        if text[0] in "[{":
            payload = json.loads(text)
            if isinstance(payload, Mapping):
                payload = payload.get("layers", payload)
            return parse_dabsn_layer_specs(payload)
        parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
        layers: list[DABSNLayerSpec] = []
        for part in parts:
            fields = [item.strip() for item in part.split(":")]
            if len(fields) not in {2, 3}:
                raise ValueError(
                    "compact DABSN layer specs must be geometry:hidden[:state], "
                    f"got {part!r}"
                )
            layers.append(
                DABSNLayerSpec(
                    hidden_dim=int(fields[1]),
                    state_dim=int(fields[2]) if len(fields) == 3 and fields[2] else None,
                    read_geometry=fields[0],
                )
            )
        return layers

    layers = []
    for item in spec:
        if isinstance(item, DABSNLayerSpec):
            layers.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ValueError(
                f"DABSN layer spec entries must be mappings, got {type(item).__name__}"
            )
        geometry = item.get("geometry", item.get("read_geometry", "field"))
        hidden = item.get("hidden", item.get("hidden_dim"))
        if hidden is None:
            raise ValueError(f"DABSN layer spec missing hidden/hidden_dim: {item!r}")
        state = item.get(
            "state",
            item.get("state_dim", item.get("bias_state", item.get("bias_state_dim"))),
        )
        layers.append(
            DABSNLayerSpec(
                hidden_dim=int(hidden),
                state_dim=None if state is None else int(state),
                read_geometry=str(geometry),
            )
        )
    return layers


def resolve_layer_geometries(
    layer_geometries: str | Sequence[str] | None,
    depth: int,
) -> list[str]:
    """Repeat a geometry pattern to one entry per block."""

    if layer_geometries is None or layer_geometries == "":
        pattern = ["seq"]
    elif isinstance(layer_geometries, str):
        pattern = [part.strip() for part in layer_geometries.replace(",", " ").split() if part.strip()]
    else:
        pattern = [str(part).strip() for part in layer_geometries if str(part).strip()]
    if not pattern:
        raise ValueError("layer_geometries must contain at least one of: seq, field, hybrid")
    bad = [geometry for geometry in pattern if geometry not in {"seq", "field", "hybrid"}]
    if bad:
        raise ValueError(f"unknown DABSN layer geometries: {bad}")
    if int(depth) <= 0:
        raise ValueError("depth must be positive")
    return [pattern[index % len(pattern)] for index in range(int(depth))]


def resolve_dabsn_layers(
    *,
    layers: str | Sequence[object] | None = None,
    hidden_dim: int,
    depth: int,
    layer_geometries: str | Sequence[str] | None = None,
    state_dim: int | None = None,
) -> list[DABSNLayerSpec]:
    parsed = parse_dabsn_layer_specs(layers)
    if parsed:
        return parsed
    return [
        DABSNLayerSpec(
            hidden_dim=int(hidden_dim),
            state_dim=state_dim,
            read_geometry=geometry,
        )
        for geometry in resolve_layer_geometries(layer_geometries, depth)
    ]


def coerce_layer_specs(
    specs: Sequence[DABSNLayerSpec | Mapping[str, object]],
) -> list[DABSNLayerSpec]:
    out: list[DABSNLayerSpec] = []
    for spec in specs:
        if isinstance(spec, DABSNLayerSpec):
            out.append(spec)
        else:
            out.extend(parse_dabsn_layer_specs([spec]))
    if not out:
        raise ValueError("DABSN needs at least one layer spec")
    return out


@dataclass(frozen=True)
class DABSNConfig:
    """Complete public construction config for ordinary PyTorch DABSN."""

    input_dim: int
    out_dim: int
    hidden_dim: int = 512
    depth: int = 1
    geometry: str = "seq"
    state_dim: int | None = None
    input_adapter: str = "identity"
    output_adapter: str = "field"
    task: str | None = None
    layers: list[DABSNLayerSpec | Mapping[str, object]] = field(default_factory=list)
    residual: bool = False
    mlp_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.out_dim <= 0:
            raise ValueError("input_dim and out_dim must be positive")
        if self.hidden_dim <= 0 or self.depth <= 0:
            raise ValueError("hidden_dim and depth must be positive")
        if self.geometry not in {"seq", "field", "hybrid"}:
            raise ValueError("geometry must be seq, field, or hybrid")
        if self.mlp_ratio is not None and float(self.mlp_ratio) <= 0:
            raise ValueError("mlp_ratio must be positive or None")

    def layer_specs(self) -> list[DABSNLayerSpec]:
        if self.layers:
            return coerce_layer_specs(self.layers)
        return [
            DABSNLayerSpec(
                hidden_dim=self.hidden_dim,
                state_dim=self.state_dim,
                read_geometry=self.geometry,
            )
            for _ in range(self.depth)
        ]

    def metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["arch"] = DABSN_ARCH
        data["layers"] = [spec.to_metadata() for spec in self.layer_specs()]
        return data


@dataclass(frozen=True)
class DABSNPretrainConfig:
    """Large-run next-token pretraining configuration."""

    corpus_bin: str | None = None
    corpus_text: str | None = None
    corpus_dtype: str = "uint16"
    vocab: int = 50_257
    hidden_dim: int = 512
    depth: int = 1
    layers: str | Sequence[object] | None = None
    layer_geometries: str | Sequence[str] | None = ("seq", "field", "hybrid")
    state_dim: int | None = None
    tie_embeddings: bool = True
    residual: bool = False
    mlp_ratio: float | None = None
    train_context: int = 2048
    eval_contexts: tuple[int, ...] = ()
    steps: int = 16_000
    token_budget: int = 0
    batch_size: int = 16
    eval_batch_size: int = 8
    val_batches: int = 8
    val_fraction: float = 0.005
    learning_rate: float = 6e-4
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    clip_grad_norm: float = 1.0
    seed: int = 0
    precision: str = "auto"
    distributed: str = "fsdp"
    grad_checkpoint: bool = True
    grad_accum_steps: int = 4
    cuda_graph: bool = False
    checkpoint_every: int = 1000
    log_every: int = 50
    val_every: int = 0
    long_scan_chunk: int = 8192
    # Element budget (rows*V) for one logit chunk in the fused linear+CE loss.
    # 0 keeps the default (2**24); the loss auto-engages chunking only above it,
    # so small models keep the exact single-shot path. Mirrors env
    # DABSN_LOSS_CHUNK_SCORES; block-time chunking (Phase 6) uses block_chunk_t.
    loss_chunk_scores: int = 0
    # Block-time activation chunking: 0 = auto (engage under memory pressure),
    # -1 = force off, >0 = explicit chunk width. Mirrors env DABSN_BLOCK_CHUNK_T.
    block_chunk_t: int = 0

    def __post_init__(self) -> None:
        if bool(self.corpus_bin) == bool(self.corpus_text):
            raise ValueError("set exactly one of corpus_bin or corpus_text")
        if self.corpus_dtype not in {"uint8", "uint16", "uint32", "int32", "int64"}:
            raise ValueError("corpus_dtype must be uint8, uint16, uint32, int32, or int64")
        positive = {
            "vocab": self.vocab,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "train_context": self.train_context,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
        }
        bad = [name for name, value in positive.items() if int(value) <= 0]
        if bad:
            raise ValueError(f"pretraining values must be positive: {bad}")
        if self.token_budget < 0 or self.warmup_steps < 0:
            raise ValueError("token_budget and warmup_steps must be non-negative")
        if not 0.0 <= self.val_fraction < 0.5:
            raise ValueError("val_fraction must be in [0, 0.5)")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.distributed not in {"none", "ddp", "fsdp"}:
            raise ValueError("distributed must be none, ddp, or fsdp")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16, or bf16")
        if self.mlp_ratio is not None and float(self.mlp_ratio) <= 0:
            raise ValueError("mlp_ratio must be positive or None")
        if any(int(context) <= 0 for context in self.eval_contexts):
            raise ValueError("eval_contexts must contain positive lengths")
        if self.checkpoint_every < 0 or self.log_every < 0 or self.val_every < 0:
            raise ValueError("checkpoint_every, log_every, and val_every must be non-negative")
        if self.checkpoint_every and self.checkpoint_every % self.grad_accum_steps:
            raise ValueError(
                "checkpoint_every must be divisible by grad_accum_steps so exact resume "
                "never loses partially accumulated gradients"
            )

    def layer_specs(self) -> list[DABSNLayerSpec]:
        return resolve_dabsn_layers(
            layers=self.layers,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            layer_geometries=self.layer_geometries,
            state_dim=self.state_dim,
        )

    def metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["layers"] = [spec.to_metadata() for spec in self.layer_specs()]
        data["eval_contexts"] = list(self.eval_contexts)
        data["objective"] = "next-token"
        return data
