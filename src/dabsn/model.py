"""DABSN blocks, backbones, task models, and language-model wrappers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
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
from .read import DABSNRead


class DABSNBlock(nn.Module):
    """One canonical core plus admitted/permanent/long read."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        read_geometry: str,
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
        self.last_trace: dict[str, object] = {}
        self.last_signals: dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 4:
            batch, cells, steps, dim = inputs.shape
            inputs = inputs.reshape(batch, cells * steps, dim)
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
        output = self.state_to_hidden(y + self.read_gain * read)
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
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer_specs = coerce_layer_specs(layers)
        self.grad_checkpoint = grad_checkpoint
        blocks: list[DABSNBlock] = []
        width = input_dim
        for spec in self.layer_specs:
            blocks.append(
                DABSNBlock(
                    input_dim=width,
                    hidden_dim=spec.hidden_dim,
                    state_dim=spec.resolved_state_dim,
                    read_geometry=spec.read_geometry,
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
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
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
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.output_adapter_kind = output_adapter
        self.backbone = DABSNBackbone(input_dim, layers, grad_checkpoint)
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
    ) -> None:
        super().__init__()
        self.block_name = "dabsn"
        self.vocab = int(vocab)
        self.hidden_dim = int(hidden_dim)
        self.state_dim = None if state_dim is None else int(state_dim)
        self.tie_embeddings = bool(tie_embeddings)
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
) -> DABSNModel:
    config = DABSNConfig(
        input_dim=input_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        geometry=read_geometry,
        state_dim=state_dim,
        output_adapter=output_adapter,
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
        )
    else:
        model = DABSNModel(
            input_dim=config.input_dim,
            out_dim=config.out_dim,
            layers=layers,
            output_adapter=config.output_adapter,
            grad_checkpoint=grad_checkpoint,
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
