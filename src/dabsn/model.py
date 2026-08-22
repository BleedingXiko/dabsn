"""DABSN blocks, backbones, task models, and language-model wrappers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch._subclasses.fake_tensor import is_fake
from torch.utils.checkpoint import checkpoint

from .adapters import build_input_adapter, build_output_head
from .components import (
    AxisContract,
    ComponentCapabilities,
    ComponentContract,
    ComponentOutput,
    ValueContract,
    bind_module,
)
from .config import (
    DABSNConfig,
    DABSNLayerSpec,
    coerce_layer_specs,
    resolve_dabsn_layers,
)
from .core import DABSNCore
from .graph import DABSNGraph
from .read import DABSNRead, _stream_is_capturing


class MLPRMSNorm(nn.Module):
    """Scale-only normalization used exclusively by the optional MLP branch."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, inputs: Tensor) -> Tensor:
        scale = torch.rsqrt(inputs.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (inputs.float() * scale).to(inputs.dtype) * self.weight


class ResidualMLPComponent(nn.Module):
    """Standalone copy of the canonical post-DABSN residual MLP branch."""

    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        inner_dim = int(round(float(ratio) * int(dim)))
        if inner_dim <= 0:
            raise ValueError("mlp_ratio is too small to produce a non-empty MLP")
        self.dim = int(dim)
        self.ratio = float(ratio)
        self.norm = MLPRMSNorm(self.dim)
        self.fc1 = nn.Linear(self.dim, inner_dim, bias=False)
        self.fc2 = nn.Linear(inner_dim, self.dim, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        branch = self.fc2(F.relu(self.fc1(self.norm(inputs))).square())
        return inputs + branch


# Private aliases remain import-compatible with the 0.1.x implementation while
# the public component names are used by all new composition code.
_MLPRMSNorm = MLPRMSNorm
_ResidualMLP = ResidualMLPComponent


def _sequence_world_contract(width: int) -> ValueContract:
    return ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("experience", "T", dynamic=True),
        AxisContract("world", int(width)),
    )


class _DABSNWorldView(nn.Module):
    """Non-owning graph view of the DABSN computation in a legacy block."""

    def __init__(self, block: DABSNBlock) -> None:
        super().__init__()
        self.block: DABSNBlock
        object.__setattr__(self, "block", block)

    def forward(self, inputs: Tensor) -> Tensor:
        # Route through nn.Module.__call__ so user forward hooks retain their
        # 0.1.x behavior while the private static flag selects only the DABSN
        # component for ordered graph execution.
        return self.block(inputs, _dabsn_only=True)


class _InlineMLPView(nn.Module):
    """Non-owning explicit component view of a legacy block's MLP modules."""

    def __init__(self, block: DABSNBlock) -> None:
        super().__init__()
        self.block: DABSNBlock
        object.__setattr__(self, "block", block)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block.forward_mlp(inputs)


class DABSNBlock(nn.Module):
    """One canonical core plus admitted/permanent/long read."""

    input_dim: int
    hidden_dim: int
    state_dim: int
    core: DABSNCore
    read: DABSNRead
    read_gain: Tensor
    field_read_gain: Tensor | None
    state_to_hidden: nn.Module
    residual_skip: nn.Linear | None
    mlp_ratio: float | None
    mlp_norm: MLPRMSNorm | None
    mlp_fc1: nn.Linear | None
    mlp_fc2: nn.Linear | None

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
            nn.Identity() if state_dim == hidden_dim else nn.Linear(state_dim, hidden_dim)
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
            self.mlp_norm = MLPRMSNorm(hidden_dim)
            self.mlp_fc1 = nn.Linear(hidden_dim, inner_dim, bias=False)
            self.mlp_fc2 = nn.Linear(inner_dim, hidden_dim, bias=False)
            nn.init.normal_(self.mlp_fc1.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mlp_fc2.weight)
        self.last_trace: dict[str, object] = {}
        self.last_signals: dict[str, Tensor] = {}

    def _finish_block(self, inputs: Tensor, dabsn_output: Tensor) -> Tensor:
        """Apply only the DABSN stack residual."""

        hidden = dabsn_output
        if self.residual:
            skip = inputs if self.residual_skip is None else self.residual_skip(inputs)
            hidden = skip + hidden
        return hidden

    def forward_mlp(self, hidden: Tensor) -> Tensor:
        """Compatibility execution for the translated inline MLP component."""

        if self.mlp_fc2 is None:
            return hidden
        assert self.mlp_fc1 is not None and self.mlp_norm is not None
        branch = self.mlp_fc2(F.relu(self.mlp_fc1(self.mlp_norm(hidden))).square())
        return hidden + branch

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
        # Under compilation, off -- for the same reason it is off under capture,
        # and before anything host-side is touched. Everything below this line
        # reads process state rather than tensors: an environment variable, live
        # free memory, a Python dict cache. None of that belongs inside a traced
        # graph, the compiler bounds the activation footprint itself, and a
        # structural decision made mid-trace would only produce recompiles.
        if torch.compiler.is_compiling():
            return 0
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
        # Plain attribute rather than `self.__dict__.setdefault(...)`: reaching
        # into `__dict__` is untraceable ("Dynamo does not know how to trace
        # method setdefault"), and it is the same lazy init either way.
        cache = getattr(self, "_block_chunk_auto", None)
        if cache is None:
            cache = {}
            self._block_chunk_auto = cache
        if key in cache:
            return cache[key]
        if not inputs.is_cuda:
            cache[key] = 0
            return 0
        try:
            free_bytes, _total = torch.cuda.mem_get_info(inputs.device)
        except Exception:  # noqa: BLE001 - any driver refusal means "don't chunk"
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

    def forward_world(self, inputs: Tensor) -> Tensor:
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
        compiling = torch.compiler.is_compiling()
        fake_tensor = not compiling and is_fake(inputs)
        trace_enabled = (
            not compiling
            and not fake_tensor
            and (os.environ.get("DABSN_COLLECT_TRACES", "0") == "1" or not inputs.is_cuda)
        )
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

    def forward(self, inputs: Tensor, _dabsn_only: bool = False) -> Tensor:
        """0.1.x wrapper: DABSN world transform followed by translated MLP."""

        hidden = self.forward_world(inputs)
        return hidden if _dabsn_only else self.forward_mlp(hidden)


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
        mlp_middle_depth: int = 0,
        mlp_depth_index: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer_specs = coerce_layer_specs(layers)
        self.grad_checkpoint = grad_checkpoint
        self.residual = bool(residual)
        self.mlp_ratio = None if mlp_ratio is None else float(mlp_ratio)
        self.mlp_middle_depth = int(mlp_middle_depth)
        self.mlp_depth_index = int(mlp_depth_index)
        if self.mlp_middle_depth < 0:
            raise ValueError("mlp_middle_depth must be non-negative")
        if self.mlp_middle_depth:
            if self.mlp_ratio is None:
                raise ValueError("mlp_middle_depth requires mlp_ratio")
            if len(self.layer_specs) < 2:
                raise ValueError("mlp_middle_depth requires at least two DABSN blocks")
            if not 0 <= self.mlp_depth_index < len(self.layer_specs) - 1:
                raise ValueError("mlp_depth_index must select a DABSN block before the final block")
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
        if self.mlp_middle_depth:
            assert self.mlp_ratio is not None
            middle_width = self.layer_specs[self.mlp_depth_index].hidden_dim
            self.middle_mlps = nn.ModuleList(
                ResidualMLPComponent(middle_width, self.mlp_ratio)
                for _ in range(self.mlp_middle_depth)
            )
        else:
            self.middle_mlps = nn.ModuleList()
        self.output_dim = width

        ordered = []
        middle_index = 0
        for index, block in enumerate(blocks):
            input_contract = _sequence_world_contract(block.input_dim)
            output_contract = _sequence_world_contract(block.hidden_dim)
            ordered.append(
                bind_module(
                    f"dabsn.{index}",
                    _DABSNWorldView(block),
                    ComponentContract(input_contract, output_contract),
                    capabilities=ComponentCapabilities(
                        eager=True,
                        activation_checkpoint=True,
                        amp_fp32=True,
                        amp_bf16=True,
                        amp_fp16=True,
                        distributed=True,
                        world_builder=index == 0,
                        dabsn_memory_owner=True,
                    ),
                )
            )
            if block.mlp_ratio is not None:
                ordered.append(
                    bind_module(
                        f"legacy-inline-mlp.{index}",
                        _InlineMLPView(block),
                        ComponentContract(output_contract, output_contract),
                        capabilities=ComponentCapabilities(
                            eager=True,
                            compile_fullgraph=True,
                            dynamic_shapes=True,
                            export=True,
                            cuda_graph=True,
                            activation_checkpoint=True,
                            amp_fp32=True,
                            amp_bf16=True,
                            amp_fp16=True,
                            distributed=True,
                        ),
                    )
                )
            if index == self.mlp_depth_index:
                for mlp in self.middle_mlps:
                    ordered.append(
                        bind_module(
                            f"legacy-middle-mlp.{middle_index}",
                            mlp,
                            ComponentContract(output_contract, output_contract),
                            capabilities=ComponentCapabilities(
                                eager=True,
                                compile_fullgraph=True,
                                dynamic_shapes=True,
                                export=True,
                                cuda_graph=True,
                                activation_checkpoint=True,
                                amp_fp32=True,
                                amp_bf16=True,
                                amp_fp16=True,
                                distributed=True,
                            ),
                        )
                    )
                    middle_index += 1
        self.graph = DABSNGraph(
            ordered,
            input_contract=_sequence_world_contract(self.input_dim),
            require_world_builder=True,
            register_modules=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the ordered component graph without introducing a graph break."""

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
        if recompute:
            for component in self.graph.components:
                hidden = checkpoint(component, hidden, use_reentrant=False)
        else:
            hidden = self.graph.forward_sequence(hidden)
        if field_rank and hidden.dim() > 2:
            return hidden.reshape(batch, height, width, -1)
        return hidden

    def forward_with_terms(self, inputs: Tensor) -> ComponentOutput:
        field_rank = inputs.dim() in (4, 5)
        if inputs.dim() == 5:
            batch, height, width, steps, dim = inputs.shape
            hidden = inputs.reshape(batch, height * width, steps, dim)
        elif inputs.dim() == 4:
            batch, height, width, dim = inputs.shape
            hidden = inputs.reshape(batch, height * width, dim)
        else:
            hidden = inputs
        result = self.graph.forward_with_terms(hidden)
        value = cast(Tensor, result.value)
        if field_rank and value.dim() > 2:
            value = value.reshape(batch, height, width, -1)
        return ComponentOutput(value, result.loss_terms, result.reports, result.next_state)

    def post_optimizer_step(self, *, step_applied: bool) -> None:
        self.graph.post_optimizer_step(step_applied=step_applied)

    def read_traces(self) -> list[dict[str, object]]:
        blocks = cast(list[DABSNBlock], list(self.blocks))
        return [block.last_trace for block in blocks]

    def signal_traces(self) -> list[dict[str, Tensor]]:
        blocks = cast(list[DABSNBlock], list(self.blocks))
        return [block.last_signals for block in blocks]

    def set_hybrid_gate_override(self, value: float | None) -> None:
        blocks = cast(list[DABSNBlock], list(self.blocks))
        for block in blocks:
            if block.read_geometry == "hybrid":
                block.read.hybrid_gate_override = value


class DABSNModel(nn.Module):
    """DABSN backbone plus a registered per-position output head."""

    is_dabsn = True
    _dabsn_config: DABSNConfig

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        layers: Sequence[DABSNLayerSpec | Mapping[str, object]],
        output_adapter: str = "field",
        grad_checkpoint: bool = False,
        residual: bool = False,
        mlp_ratio: float | None = None,
        mlp_middle_depth: int = 0,
        mlp_depth_index: int = 0,
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
            mlp_middle_depth=mlp_middle_depth,
            mlp_depth_index=mlp_depth_index,
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

    def forward_with_terms(self, inputs: Tensor) -> ComponentOutput:
        result = self.backbone.forward_with_terms(inputs)
        return ComponentOutput(
            self.project_sequence(cast(Tensor, result.value)),
            result.loss_terms,
            result.reports,
            result.next_state,
        )

    def post_optimizer_step(self, *, step_applied: bool) -> None:
        self.backbone.post_optimizer_step(step_applied=step_applied)

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
    _dabsn_config: DABSNConfig

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
        mlp_middle_depth: int = 0,
        mlp_depth_index: int = 0,
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
            mlp_middle_depth=mlp_middle_depth,
            mlp_depth_index=mlp_depth_index,
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

    def forward_with_terms(self, inputs: Tensor) -> ComponentOutput:
        result = self.body.forward_with_terms(self.input_adapter(inputs))
        return result

    def post_optimizer_step(self, *, step_applied: bool) -> None:
        self.body.post_optimizer_step(step_applied=step_applied)

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


class _GraphBackbone(nn.Module):
    """Backbone owner used by first-class graph language models."""

    def __init__(self, graph: DABSNGraph, output_dim: int) -> None:
        super().__init__()
        self.graph = graph
        self.output_dim = int(output_dim)
        self.grad_checkpoint = False

    def forward(self, inputs: Tensor) -> Tensor:
        return self.graph.forward_sequence(inputs)

    def forward_with_terms(self, inputs: Tensor):
        return self.graph.forward_with_terms(inputs)

    def post_optimizer_step(self, *, step_applied: bool) -> None:
        self.graph.post_optimizer_step(step_applied=step_applied)

    def read_traces(self) -> list[dict[str, object]]:
        traces: list[dict[str, object]] = []
        for binding in self.graph.bindings:
            trace = getattr(binding.module, "last_trace", None)
            if isinstance(trace, dict):
                traces.append(cast(dict[str, object], trace))
        return traces

    def signal_traces(self) -> list[dict[str, Tensor]]:
        traces: list[dict[str, Tensor]] = []
        for binding in self.graph.bindings:
            signals = getattr(binding.module, "last_signals", None)
            if isinstance(signals, dict):
                traces.append(cast(dict[str, Tensor], signals))
        return traces

    def set_hybrid_gate_override(self, value: float | None) -> None:
        for binding in self.graph.bindings:
            action = getattr(binding.module, "set_hybrid_gate_override", None)
            if callable(action):
                action(value)


def _world_width(contract: ValueContract) -> int:
    if len(contract.leaves) != 1:
        raise ValueError("language-model graph endpoints must be one tensor leaf")
    matches = [axis for axis in contract.leaves[0].axes if axis.name == "world"]
    if len(matches) != 1 or not isinstance(matches[0].size, int):
        raise ValueError("language-model graph endpoints require one static world axis")
    return matches[0].size


class DABSNSequenceLM(nn.Module):
    """Token embedding/readout wrapper around the canonical backbone."""

    is_dabsn_sequence_lm = True
    backbone: DABSNBackbone | _GraphBackbone

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
        mlp_middle_depth: int = 0,
        mlp_depth_index: int = 0,
    ) -> None:
        super().__init__()
        self.block_name = "dabsn"
        self.vocab = int(vocab)
        self.hidden_dim = int(hidden_dim)
        self.state_dim = None if state_dim is None else int(state_dim)
        self.tie_embeddings = bool(tie_embeddings)
        self.residual = bool(residual)
        self.mlp_ratio = None if mlp_ratio is None else float(mlp_ratio)
        self.mlp_middle_depth = int(mlp_middle_depth)
        self.mlp_depth_index = int(mlp_depth_index)
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
            mlp_middle_depth=self.mlp_middle_depth,
            mlp_depth_index=self.mlp_depth_index,
        )
        self.readout = nn.Linear(self.backbone.output_dim, self.vocab)
        if self.tie_embeddings and self.backbone.output_dim == self.hidden_dim:
            self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.readout.bias)

    @classmethod
    def from_graph(
        cls,
        graph: DABSNGraph,
        *,
        vocab: int,
        tie_embeddings: bool = False,
    ) -> DABSNSequenceLM:
        """Construct a language model whose only body is ``graph``.

        Token IDs are embedded once, the first DABSN component consumes that
        representation and builds the world, and downstream components receive
        only the preceding component's world value.
        """

        if not graph.bindings[0].capabilities.world_builder:
            raise ValueError("a language-model graph must begin with DABSN")
        input_width = _world_width(graph.input_contract)
        output_width = _world_width(graph.output_contract)
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.block_name = "dabsn-graph"
        model.vocab = int(vocab)
        model.hidden_dim = input_width
        model.state_dim = None
        model.tie_embeddings = bool(tie_embeddings)
        model.residual = False
        model.mlp_ratio = None
        model.mlp_middle_depth = 0
        model.mlp_depth_index = 0
        model.layers = []
        model.depth = graph.dabsn_memory_count
        model.layer_geometries = []
        model.embed = nn.Embedding(model.vocab, input_width)
        model.backbone = _GraphBackbone(graph, output_width)
        model.readout = nn.Linear(output_width, model.vocab)
        if model.tie_embeddings:
            if input_width != output_width:
                raise ValueError("tied embeddings require equal graph input/output world widths")
            model.readout.weight = model.embed.weight
        nn.init.normal_(model.embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(model.readout.bias)
        return model

    @property
    def graph(self) -> DABSNGraph:
        return self.backbone.graph

    def forward_with_terms(self, ids: Tensor):
        result = self.backbone.forward_with_terms(self.embed(ids))
        return type(result)(
            self.readout(result.value),
            result.loss_terms,
            result.reports,
            result.next_state,
        )

    def post_optimizer_step(self, *, step_applied: bool) -> None:
        self.backbone.post_optimizer_step(step_applied=step_applied)

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
    mlp_middle_depth: int = 0,
    mlp_depth_index: int = 0,
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
        mlp_middle_depth=mlp_middle_depth,
        mlp_depth_index=mlp_depth_index,
    )
    model = build_dabsn_from_config(config, grad_checkpoint=grad_checkpoint)
    if not isinstance(model, DABSNModel):
        raise RuntimeError("identity-input DABSN construction returned a task model")
    return model


def build_dabsn_from_config(
    config: DABSNConfig | Mapping[str, object],
    *,
    grad_checkpoint: bool = False,
) -> DABSNModel | DABSNTaskModel:
    if not isinstance(config, DABSNConfig):
        config = DABSNConfig(**cast(Any, dict(config)))
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
            mlp_middle_depth=config.mlp_middle_depth,
            mlp_depth_index=config.mlp_depth_index,
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
            mlp_middle_depth=config.mlp_middle_depth,
            mlp_depth_index=config.mlp_depth_index,
        )
    model._dabsn_config = config
    return model


def dabsn_adamw_param_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    embedded = {id(module.weight) for module in model.modules() if isinstance(module, nn.Embedding)}
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or id(parameter) in embedded:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, object]] = [{"params": decay, "weight_decay": float(weight_decay)}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups
