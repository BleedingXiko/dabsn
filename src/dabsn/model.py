"""DABSN blocks, backbones, task models, and language-model wrappers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .adapters import build_input_adapter, build_output_head
from .config import (
    DABSNConfig,
    DABSNLayerSpec,
    coerce_layer_specs,
    resolve_dabsn_layers,
)
from .core import DABSNCore
from .read import DABSNRead, _stream_is_capturing


class _MLPRMSNorm(nn.Module):
    """Scale-only normalization used exclusively by the optional MLP branch."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, inputs: Tensor) -> Tensor:
        scale = torch.rsqrt(inputs.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (inputs.float() * scale).to(inputs.dtype) * self.weight


class DABSNBlock(nn.Module):
    """One canonical core plus admitted/permanent/long read."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        read_geometry: str,
        *,
        residual: bool = False,
        mlp_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.read_geometry = read_geometry
        self.core = DABSNCore(input_dim=input_dim, hidden_dim=state_dim)
        self.read = DABSNRead(hidden_dim=state_dim, read_geometry=read_geometry)
        self.read_gain = nn.Parameter(torch.full((1,), 0.01))
        self.field_read_gain = (
            nn.Parameter(torch.full((1,), 1.0)) if read_geometry == "field" else None
        )
        self.state_to_hidden = (
            nn.Identity()
            if state_dim == hidden_dim
            else nn.Linear(state_dim, hidden_dim)
        )
        self.residual = bool(residual)
        self.residual_skip = None
        if self.residual and input_dim != hidden_dim:
            self.residual_skip = nn.Linear(input_dim, hidden_dim, bias=False)
            nn.init.normal_(
                self.residual_skip.weight,
                mean=0.0,
                std=(1.0 / input_dim) ** 0.5,
            )

        if mlp_ratio is not None and float(mlp_ratio) <= 0:
            raise ValueError("mlp_ratio must be positive or None")
        self.mlp_ratio = None if mlp_ratio is None else float(mlp_ratio)
        self.mlp_norm = None
        self.mlp_fc1 = None
        self.mlp_fc2 = None
        if self.mlp_ratio is not None:
            inner_dim = int(round(self.mlp_ratio * hidden_dim))
            if inner_dim <= 0:
                raise ValueError("mlp_ratio is too small to produce a non-empty MLP")
            self.mlp_norm = _MLPRMSNorm(hidden_dim)
            self.mlp_fc1 = nn.Linear(hidden_dim, inner_dim, bias=False)
            self.mlp_fc2 = nn.Linear(inner_dim, hidden_dim, bias=False)
            nn.init.normal_(self.mlp_fc1.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mlp_fc2.weight)
        self.last_trace: dict[str, object] = {}
        self.last_signals: dict[str, Tensor] = {}

    def _finish_block(self, inputs: Tensor, dabsn_output: Tensor) -> Tensor:
        """Apply the stack residual, then the optional post-DABSN MLP residual."""

        hidden = dabsn_output
        if self.residual:
            skip = inputs if self.residual_skip is None else self.residual_skip(inputs)
            hidden = skip + hidden
        if self.mlp_fc2 is not None:
            branch = self.mlp_fc2(
                F.relu(self.mlp_fc1(self.mlp_norm(hidden))).square()
            )
            hidden = hidden + branch
        return hidden

    def _resolve_block_chunk_t(self, inputs: Tensor) -> int:
        """Decide the time-chunk width for the core scan.

        ``block_chunk_t``/``DABSN_BLOCK_CHUNK_T``: ``-1`` forces off, ``>0`` is an
        explicit width, ``0`` (default) is auto -- engage only under real device
        memory pressure. On CPU (or when free memory is unknown) auto stays off,
        so CPU behavior and the release gate are unchanged.

        Chunking is off under CUDA-graph capture at any setting: the per-chunk
        ``checkpoint`` recompute is not capturable, and a graph already bounds
        activation memory by replaying one fixed allocation. ``_capture_no_chunk``
        (set by ``runtime.graph``) covers the warmup iterations too, so warmup and
        capture run the same structure.
        """
        T = int(inputs.shape[1])
        raw = os.environ.get("DABSN_BLOCK_CHUNK_T")
        setting = int(raw) if raw not in (None, "") else 0
        if getattr(self, "_capture_no_chunk", False) or _stream_is_capturing():
            if setting > 0:
                from .runtime.dispatch import warn_routing_once

                warn_routing_once(
                    "block_chunk",
                    "DABSN_BLOCK_CHUNK_T is set but CUDA-graph capture cannot "
                    "replay checkpoint recompute; chunking is off for graphed "
                    "steps (the graph bounds activation memory itself)",
                    chunk_t=setting,
                )
            return 0
        if setting < 0:
            return 0  # forced off
        if setting > 0:
            return min(setting, T)
        # auto: decided ONCE per execution shape and remembered. Re-deciding per
        # call would read whatever free memory happens to be around at that
        # moment, so the same shape could chunk on one step and not the next --
        # nondeterministic structure, and a warmup/capture mismatch waiting to
        # happen. The first answer for a shape is the answer for that shape.
        B = int(inputs.shape[0])
        H = int(self.state_dim)
        key = (B, T, H, inputs.dtype)
        cache = self.__dict__.setdefault("_block_chunk_auto", {})
        if key in cache:
            return cache[key]
        if not inputs.is_cuda:
            cache[key] = 0
            return 0
        try:
            free_bytes, _total = torch.cuda.mem_get_info(inputs.device)
        except Exception:
            cache[key] = 0
            return 0
        elem = inputs.element_size()
        # ~8 full-T [B,T,H] tapes are the core's activation working set.
        est = 8 * B * T * H * elem
        decision = 0
        if est > 0.5 * float(free_bytes):
            # Target roughly a quarter of the footprint per chunk.
            target = max(1, int(T * (0.5 * free_bytes) / est) // 4)
            decision = max(1, min(target, T))
        cache[key] = decision
        return decision

    def _chunked_core(self, inputs: Tensor, chunk_t: int):
        """Run the core scan in exact time-chunks, carrying (budget, energy,
        saturation) across chunk boundaries and assembling the full-T public
        tapes. Each chunk is wrapped in ``torch.utils.checkpoint`` while training
        so only one chunk's activation graph is live at a time -- the core's
        activation working set drops from O(T) to O(chunk_t). Exactness is the
        same carried-state parity the release gate pins (chunked == full).
        """
        B, T, _ = inputs.shape
        carry = self.core.initial_state(B, device=inputs.device, dtype=inputs.dtype)
        tapes: list[list[Tensor]] = [[] for _ in range(7)]
        use_ckpt = self.training and torch.is_grad_enabled()
        for t0 in range(0, T, chunk_t):
            t1 = min(T, t0 + chunk_t)
            chunk = inputs[:, t0:t1]

            def run(chunk_in, b_in, e_in, c_in):
                result, final = self.core.forward_from_state(
                    chunk_in,
                    initial_state=(b_in, e_in, c_in),
                    return_writes=True,
                    return_cocktail=True,
                    return_final_state=True,
                )
                return (*result, *final)

            if use_ckpt:
                outs = checkpoint(run, chunk, carry[0], carry[1], carry[2], use_reentrant=False)
            else:
                outs = run(chunk, carry[0], carry[1], carry[2])
            for i in range(7):
                tapes[i].append(outs[i])
            carry = (outs[7], outs[8], outs[9])
        return tuple(torch.cat(parts, dim=1) for parts in tapes)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 4:
            batch, cells, steps, dim = inputs.shape
            inputs = inputs.reshape(batch, cells * steps, dim)
        chunk_t = self._resolve_block_chunk_t(inputs)
        if chunk_t and chunk_t < int(inputs.shape[1]):
            trajectory, novelty, plasticity, expression, write, energy, saturation = (
                self._chunked_core(inputs, chunk_t)
            )
        else:
            trajectory, novelty, plasticity, expression, write, energy, saturation = self.core(
                inputs,
                return_writes=True,
                return_cocktail=True,
            )
        y, budget = trajectory.split(self.state_dim, dim=-1)
        read = self.read(
            y,
            budget,
            expression,
            write,
            novelty,
            plasticity,
            energy,
            saturation,
            field_shape=None,
        )
        dabsn_output = self.state_to_hidden(y + self.read_gain * read)
        output = self._finish_block(inputs, dabsn_output)
        trace_enabled = os.environ.get("DABSN_COLLECT_TRACES", "0") == "1" or not inputs.is_cuda
        if trace_enabled:
            self.last_signals = {
                "ay": expression.detach(),
                "write": write.detach(),
                "y": y.detach(),
                "b": budget.detach(),
                "read": read.detach(),
                "novelty": novelty.detach(),
                "plasticity": plasticity.detach(),
                "energy": energy.detach(),
                "saturation": saturation.detach(),
            }
        else:
            self.last_signals = {}
        self.last_trace = {
            "read_geometry": self.read_geometry,
            "admitted_n_max": self.read.last_n_max,
            "hybrid_gate": self.read.last_hybrid_gate,
            "seq_norm": self.read.last_seq_norm,
            "field_norm": self.read.last_field_norm,
            "field_neighbors": self.read.last_field_neighbors,
            "admit_gate_mean": self.read.last_admit_gate_mean,
            "core_path": "seq_scan",
            "long_scan_norm": self.read.last_long_scan_norm,
            "read_contract": self.read.last_read_contract,
            "read_gain": float(self.read_gain.detach().cpu()) if trace_enabled else None,
        }
        return output


class DABSNBackbone(nn.Module):
    """Stacked DABSN body with explicit width and geometry per block."""

    def __init__(
        self,
        input_dim: int,
        layers: Sequence[DABSNLayerSpec | Mapping[str, object]],
        grad_checkpoint: bool = False,
        *,
        residual: bool = False,
        mlp_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer_specs = coerce_layer_specs(layers)
        self.grad_checkpoint = grad_checkpoint
        self.residual = bool(residual)
        self.mlp_ratio = None if mlp_ratio is None else float(mlp_ratio)
        blocks: list[DABSNBlock] = []
        width = input_dim
        for spec in self.layer_specs:
            blocks.append(
                DABSNBlock(
                    input_dim=width,
                    hidden_dim=spec.hidden_dim,
                    state_dim=spec.resolved_state_dim,
                    read_geometry=spec.read_geometry,
                    residual=self.residual,
                    mlp_ratio=self.mlp_ratio,
                )
            )
            width = spec.hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = width

    @torch.compiler.disable
    def forward(self, inputs: Tensor) -> Tensor:
        """Preserve gradients through every custom-autograd block in a compiled stack."""

        field_rank = inputs.dim() in (4, 5)
        if inputs.dim() == 5:
            batch, height, width, steps, dim = inputs.shape
            hidden = inputs.reshape(batch, height * width, steps, dim)
        elif inputs.dim() == 4:
            batch, height, width, dim = inputs.shape
            hidden = inputs.reshape(batch, height * width, dim)
        else:
            hidden = inputs
        # Checkpoint recompute cannot be captured into a CUDA graph: it re-runs
        # forward during backward against capture-pool addresses that no longer
        # hold those activations at replay, which surfaces as an illegal memory
        # access at the next sync. `runtime.graph` already pins it off across the
        # whole warmup+capture window; this is the guard for a capture taken
        # without that helper, so the step degrades to plain (correct, heavier)
        # activations instead of corrupting the recorded graph.
        recompute = self.grad_checkpoint and self.training and not _stream_is_capturing()
        for block in self.blocks:
            if recompute:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        if field_rank and hidden.dim() > 2:
            return hidden.reshape(batch, height, width, -1)
        return hidden

    def read_traces(self) -> list[dict[str, object]]:
        return [block.last_trace for block in self.blocks]

    def signal_traces(self) -> list[dict[str, Tensor]]:
        return [block.last_signals for block in self.blocks]

    def set_hybrid_gate_override(self, value: float | None) -> None:
        for block in self.blocks:
            if block.read_geometry == "hybrid":
                block.read.hybrid_gate_override = value


class DABSNModel(nn.Module):
    """DABSN backbone plus a registered per-position output head."""

    is_dabsn = True

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        layers: Sequence[DABSNLayerSpec | Mapping[str, object]],
        output_adapter: str = "field",
        grad_checkpoint: bool = False,
        residual: bool = False,
        mlp_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.output_adapter_kind = output_adapter
        self.backbone = DABSNBackbone(
            input_dim,
            layers,
            grad_checkpoint,
            residual=residual,
            mlp_ratio=mlp_ratio,
        )
        self.output_adapter = build_output_head(
            output_adapter,
            self.backbone.output_dim,
            out_dim,
        )

    def forward_hidden(self, inputs: Tensor) -> Tensor:
        return self.backbone(inputs)

    def project_sequence(self, hidden: Tensor) -> Tensor:
        return self.output_adapter(hidden)

    def project_positions(self, hidden: Tensor, positions: Tensor) -> Tensor:
        if hidden.dim() != 3:
            raise ValueError("project_positions expects hidden states shaped [B, T, H]")
        if positions.dim() == 1:
            selected = hidden[:, positions, :]
        elif positions.dim() == 2:
            selected = torch.gather(
                hidden,
                1,
                positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
            )
        else:
            raise ValueError("positions must be [K] or [B, K]")
        return self.output_adapter(selected)

    def forward_sequence(self, inputs: Tensor) -> Tensor:
        return self.project_sequence(self.forward_hidden(inputs))

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.forward_sequence(inputs)
        return output if output.dim() < 3 else output[:, -1, :]

    def read_traces(self) -> list[dict[str, object]]:
        return self.backbone.read_traces()

    def signal_traces(self) -> list[dict[str, Tensor]]:
        return self.backbone.signal_traces()

    @property
    def last_signals(self) -> dict[str, Tensor]:
        traces = self.signal_traces()
        return traces[-1] if traces else {}

    def set_hybrid_gate_override(self, value: float | None) -> None:
        self.backbone.set_hybrid_gate_override(value)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class DABSNTaskModel(nn.Module):
    """Input adapter plus canonical DABSN model."""

    is_dabsn = True

    def __init__(
        self,
        raw_input_dim: int,
        model_input_dim: int,
        out_dim: int,
        layers: Sequence[DABSNLayerSpec | Mapping[str, object]],
        input_adapter: str = "identity",
        output_adapter: str = "field",
        grad_checkpoint: bool = False,
        residual: bool = False,
        mlp_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.raw_input_dim = int(raw_input_dim)
        self.model_input_dim = int(model_input_dim)
        self.out_dim = int(out_dim)
        self.input_adapter_kind = input_adapter
        self.output_adapter_kind = output_adapter
        self.input_adapter = build_input_adapter(
            input_adapter,
            self.raw_input_dim,
            self.model_input_dim,
        )
        self.body = DABSNModel(
            input_dim=int(getattr(self.input_adapter, "output_dim", self.model_input_dim)),
            out_dim=self.out_dim,
            layers=layers,
            output_adapter=output_adapter,
            grad_checkpoint=grad_checkpoint,
            residual=residual,
            mlp_ratio=mlp_ratio,
        )

    @property
    def backbone(self) -> DABSNBackbone:
        return self.body.backbone

    def forward_hidden(self, inputs: Tensor) -> Tensor:
        return self.body.forward_hidden(self.input_adapter(inputs))

    def project_sequence(self, hidden: Tensor) -> Tensor:
        return self.body.project_sequence(hidden)

    def project_positions(self, hidden: Tensor, positions: Tensor) -> Tensor:
        return self.body.project_positions(hidden, positions)

    def forward_sequence(self, inputs: Tensor) -> Tensor:
        return self.project_sequence(self.forward_hidden(inputs))

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.forward_sequence(inputs)
        return output if output.dim() < 3 else output[:, -1, :]

    def read_traces(self) -> list[dict[str, object]]:
        return self.body.read_traces()

    def signal_traces(self) -> list[dict[str, Tensor]]:
        return self.body.signal_traces()

    @property
    def last_signals(self) -> dict[str, Tensor]:
        return self.body.last_signals

    def set_hybrid_gate_override(self, value: float | None) -> None:
        self.body.set_hybrid_gate_override(value)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class DABSNSequenceLM(nn.Module):
    """Token embedding/readout wrapper around the canonical backbone."""

    is_dabsn_sequence_lm = True

    def __init__(
        self,
        vocab: int,
        hidden_dim: int,
        depth: int,
        layer_geometries: str | Sequence[str] | None = None,
        *,
        layers: str | Sequence[object] | None = None,
        state_dim: int | None = None,
        tie_embeddings: bool = False,
        grad_checkpoint: bool = False,
        residual: bool = False,
        mlp_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.block_name = "dabsn"
        self.vocab = int(vocab)
        self.hidden_dim = int(hidden_dim)
        self.state_dim = None if state_dim is None else int(state_dim)
        self.tie_embeddings = bool(tie_embeddings)
        self.residual = bool(residual)
        self.mlp_ratio = None if mlp_ratio is None else float(mlp_ratio)
        self.layers = resolve_dabsn_layers(
            layers=layers,
            hidden_dim=self.hidden_dim,
            depth=int(depth),
            layer_geometries=layer_geometries,
            state_dim=self.state_dim,
        )
        self.depth = len(self.layers)
        self.layer_geometries = [spec.read_geometry for spec in self.layers]
        self.embed = nn.Embedding(self.vocab, self.hidden_dim)
        self.backbone = DABSNBackbone(
            input_dim=self.hidden_dim,
            layers=self.layers,
            grad_checkpoint=grad_checkpoint,
            residual=self.residual,
            mlp_ratio=self.mlp_ratio,
        )
        self.readout = nn.Linear(self.backbone.output_dim, self.vocab)
        if self.tie_embeddings and self.backbone.output_dim == self.hidden_dim:
            self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.readout.bias)

    def forward_hidden(self, ids: Tensor) -> Tensor:
        return self.backbone(self.embed(ids))

    def project_sequence(self, hidden: Tensor) -> Tensor:
        return self.readout(hidden)

    def project_positions(self, hidden: Tensor, positions: Tensor) -> Tensor:
        if hidden.dim() != 3:
            raise ValueError("project_positions expects hidden states shaped [B, T, H]")
        if positions.dim() == 1:
            selected = hidden[:, positions, :]
        elif positions.dim() == 2:
            selected = torch.gather(
                hidden,
                1,
                positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
            )
        else:
            raise ValueError("positions must be [K] or [B, K]")
        return self.readout(selected)

    def forward_sequence(self, ids: Tensor) -> Tensor:
        return self.project_sequence(self.forward_hidden(ids))

    def forward(self, ids: Tensor) -> Tensor:
        return self.forward_sequence(ids)[:, -1, :]

    def read_traces(self) -> list[dict[str, object]]:
        return self.backbone.read_traces()

    def signal_traces(self) -> list[dict[str, Tensor]]:
        return self.backbone.signal_traces()

    def set_hybrid_gate_override(self, value: float | None) -> None:
        self.backbone.set_hybrid_gate_override(value)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_dabsn(
    input_dim: int,
    out_dim: int,
    hidden_dim: int,
    read_geometry: str = "field",
    depth: int = 1,
    state_dim: int | None = None,
    output_adapter: str = "field",
    grad_checkpoint: bool = False,
    residual: bool = False,
    mlp_ratio: float | None = None,
) -> DABSNModel:
    config = DABSNConfig(
        input_dim=input_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        geometry=read_geometry,
        state_dim=state_dim,
        output_adapter=output_adapter,
        residual=residual,
        mlp_ratio=mlp_ratio,
    )
    return build_dabsn_from_config(config, grad_checkpoint=grad_checkpoint)


def build_dabsn_from_config(
    config: DABSNConfig | Mapping[str, object],
    *,
    grad_checkpoint: bool = False,
) -> DABSNModel | DABSNTaskModel:
    if not isinstance(config, DABSNConfig):
        config = DABSNConfig(**dict(config))
    layers = config.layer_specs()
    if config.input_adapter not in {"identity", "tensor"}:
        model: DABSNModel | DABSNTaskModel = DABSNTaskModel(
            raw_input_dim=config.input_dim,
            model_input_dim=config.hidden_dim,
            out_dim=config.out_dim,
            layers=layers,
            input_adapter=config.input_adapter,
            output_adapter=config.output_adapter,
            grad_checkpoint=grad_checkpoint,
            residual=config.residual,
            mlp_ratio=config.mlp_ratio,
        )
    else:
        model = DABSNModel(
            input_dim=config.input_dim,
            out_dim=config.out_dim,
            layers=layers,
            output_adapter=config.output_adapter,
            grad_checkpoint=grad_checkpoint,
            residual=config.residual,
            mlp_ratio=config.mlp_ratio,
        )
    model._dabsn_config = config
    return model


def dabsn_adamw_param_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    embedded = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
    }
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or id(parameter) in embedded:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, object]] = [
        {"params": decay, "weight_decay": float(weight_decay)}
    ]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups
