"""One world per sequence, routed to MLP and attention experts, all declared.

This module replaces the hand-written ``moe.py`` of the 24-expert run. Nothing
here reaches into ``dabsn``: the components are ordinary providers registered at
runtime, and the architecture is then a list of provider specifications that
``dabsn`` resolves, runs, checkpoints, and restores by itself.

The body
--------
``dabsn:block``                     H=2048, bank S=4096, seq, stack residual.
``dabsn-world-experts:final_field``  [B, T, H] -> [B, 1, H]. Keeps the last world
                                    DABSN built and drops the rest, so nothing
                                    downstream has an experience axis to reason
                                    about.
``dabsn:sparse_moe``                top-4 of 24, RMSNorm on the branch input,
                                    residual add: ``z = z + moe(norm(z))``.

The experts
-----------
Every expert takes the world as ``[N, H]`` and returns ``[N, H]``. That shared
contract is what lets one router feed two different kinds of thing:

    MLP expert        [N, H] -> H -> 4H -> H, relu()^2. Never sees D.
    attention expert  [N, H] -> lift each coordinate to D -> [N, H, D]
                             -> standard pre-norm blocks over the H coordinates
                             -> project back -> [N, H]

D exists only inside the attention expert; the router never learns the families
differ. H is the attention sequence and D is d_model, exactly as in any
transformer, with the difference that H is a world rather than a timeline --
which is why attention over it is unmasked. Causality belongs to the DABSN scan
and is finished before the router sees anything. A learned coordinate identity
is added at the lift, because the H coordinates are unordered but not
interchangeable.

An attention expert is one block, not a stack. Its width stays modest so most
of its work is the mixing rather than a fat pointwise transform bolted to every
latent spot; the model's total parameter count is held by choosing how many
experts of each kind there are, not by inflating the width of one.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dabsn import (
    AxisContract,
    AxisEffect,
    ComponentCapabilities,
    ComponentContract,
    ComponentSpec,
    DABSNGraph,
    DABSNSequenceLM,
    MLPRMSNorm,
    ValueContract,
    component_registry,
)

# The report NAMES are the component's observability contract; they live beside
# the component rather than in the top-level namespace.
from dabsn.moe import ROUTER_REPORT_NAMES

MLP_EXPERT_KEY = "dabsn-world-experts:mlp"
ATTENTION_EXPERT_KEY = "dabsn-world-experts:attention"
FINAL_FIELD_KEY = "dabsn-world-experts:final_field"
EXPERT_PROVIDER_KEYS = (MLP_EXPERT_KEY, ATTENTION_EXPERT_KEY, FINAL_FIELD_KEY)
EXPERT_DISTRIBUTION = "dabsn-world-experts"
EXPERT_DISTRIBUTION_VERSION = "1.0.0"


def _world(width: int) -> ValueContract:
    """The value a world-building component emits: [batch, experience, world]."""
    return ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("experience", "T", dynamic=True),
        AxisContract("world", int(width)),
    )


def _routed_item(width: int) -> ValueContract:
    """The contract every expert must preserve: N independent H-worlds.

    ``dabsn:sparse_moe`` refuses any expert whose input or output contract is
    not exactly this, which is what makes a mixed-family expert matrix safe to
    declare -- an expert that quietly changed the item shape would be rejected
    at build time, not discovered as a wrong-shaped gradient later.
    """
    return ValueContract.tensor(
        AxisContract("dabsn:routed_item", "N", dynamic=True),
        AxisContract("world", int(width)),
    )


class ReLU2MLPExpert(nn.Module):
    """The fat relu-squared MLP expert, unchanged from the 24-expert run."""

    def __init__(self, width: int, inner: int, *, zero_output: bool = True) -> None:
        super().__init__()
        self.width = int(width)
        self.inner = int(inner)
        self.fc1 = nn.Linear(self.width, self.inner, bias=False)
        self.fc2 = nn.Linear(self.inner, self.width, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        # Zero-init fc2: the branch contributes exactly nothing at step 0, so
        # the model starts as pure DABSN. fc1's gradient is therefore also
        # exactly zero on the first step -- it flows through fc2 -- and becomes
        # ordinary as soon as fc2 leaves zero. That is expected, not a fault.
        if zero_output:
            nn.init.zeros_(self.fc2.weight)
        else:
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)

    def forward(self, items: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(items)).square())


class WorldAttentionExpert(nn.Module):
    """One attention block whose sequence is the world itself.

    The routed item is ``[N, H]``. Every one of the H coordinates is lifted into
    ``d_model`` channels, giving ``[N, H, D]`` -- the shape attention takes
    everywhere, with H sitting where T normally sits. Nothing is reshaped and
    nothing is factored: H keeps its own structure and each coordinate becomes a
    position.

    One attention over the H coordinates, one ratio-``ffn_ratio`` feed-forward,
    and back to ``[N, H]``. No stack: this is a single block, so the expert costs
    what one attention layer costs and nothing more.

    Attention is unmasked -- the coordinates are a world, not an order -- but a
    learned coordinate identity is added at the lift, because they are unordered
    without being interchangeable. Causality is the DABSN scan's job and is
    finished before the router sees anything.
    """

    def __init__(
        self,
        width: int,
        d_model: int,
        heads: int,
        ffn_ratio: float,
        *,
        zero_output: bool = True,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError(f"d_model {d_model} must divide by heads {heads}")
        self.width = int(width)
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.head_dim = self.d_model // self.heads
        self.ffn_ratio = float(ffn_ratio)
        inner = int(round(self.ffn_ratio * self.d_model))
        self.lift = nn.Linear(1, self.d_model)
        self.coordinate = nn.Parameter(torch.randn(1, self.width, self.d_model) * 0.02)
        self.attention_norm = MLPRMSNorm(self.d_model)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.ffn_norm = MLPRMSNorm(self.d_model)
        self.fc1 = nn.Linear(self.d_model, inner, bias=False)
        self.fc2 = nn.Linear(inner, self.d_model, bias=False)
        self.final_norm = MLPRMSNorm(self.d_model)
        self.project = nn.Linear(self.d_model, 1, bias=False)
        nn.init.normal_(self.lift.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lift.bias)
        for linear in (self.qkv, self.proj, self.fc1, self.fc2):
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        if zero_output:
            nn.init.zeros_(self.project.weight)
        else:
            nn.init.normal_(self.project.weight, mean=0.0, std=0.02)

    def forward(self, items: Tensor) -> Tensor:
        rows = items.shape[0]
        value = self.lift(items.unsqueeze(-1)) + self.coordinate
        query, key, mixed = self.qkv(self.attention_norm(value)).chunk(3, dim=-1)
        shape = (rows, self.width, self.heads, self.head_dim)
        query = query.reshape(shape).transpose(1, 2)
        key = key.reshape(shape).transpose(1, 2)
        mixed = mixed.reshape(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, mixed)
        attended = attended.transpose(1, 2).reshape(rows, self.width, self.d_model)
        value = value + self.proj(attended)
        value = value + self.fc2(F.relu(self.fc1(self.ffn_norm(value))).square())
        return self.project(self.final_norm(value)).squeeze(-1)


def attention_expert_params(width: int, d_model: int, ffn_ratio: float) -> int:
    """Exact parameter count, without building the module."""
    inner = int(round(float(ffn_ratio) * int(d_model)))
    return (
        int(width) * d_model            # coordinate identity
        + 2 * d_model                   # lift weight + bias
        + 4 * d_model * d_model         # qkv and output projection
        + 2 * d_model * inner           # feed-forward
        + 3 * d_model                   # three norms
        + d_model                       # readout projection
    )


class FinalFieldComponent(nn.Module):
    """Reduce the experience axis to the world DABSN finished building.

    DABSN emits a world per position, `[B, T, H]`, each one causal. This keeps
    the last of them and drops the rest, so everything downstream sees exactly
    one world per sequence. The experience axis survives with length one rather
    than being squeezed away, because that is what makes this an ordinary
    component: the value it hands on is still a `[batch, experience, world]`
    tensor and any world-consuming component can take it unchanged.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)

    def forward(self, value: Tensor) -> Tensor:
        if value.dim() != 3:
            raise ValueError(f"final-field reduction expects [B, T, H], got {tuple(value.shape)}")
        return value[:, -1:, :]


class FinalFieldProvider:
    provider_key = FINAL_FIELD_KEY
    component_abi_version = 2
    config_schema_version = 1
    capabilities = ComponentCapabilities(
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
        deterministic=True,
    )

    def validate_config(self, config: Mapping[str, object]) -> None:
        unknown = set(config) - {"width"}
        if unknown:
            raise ValueError(f"unknown {self.provider_key} fields: {sorted(unknown)}")
        if int(config["width"]) <= 0:  # type: ignore[arg-type]
            raise ValueError("width must be positive")

    def contract(self, config: Mapping[str, object]) -> ComponentContract:
        width = int(config["width"])  # type: ignore[arg-type]
        return ComponentContract(
            _world(width),
            ValueContract.tensor(
                AxisContract("batch", "B", dynamic=True),
                # The experience axis survives but is no longer the source
                # length, so it carries no size constraint and declares the
                # collapse it performed.
                AxisContract("experience", None, dynamic=True, effect=AxisEffect.COLLAPSE),
                AxisContract("world", width),
            ),
        )

    def build(self, config: Mapping[str, object], context):
        del context
        return FinalFieldComponent(int(config["width"]))  # type: ignore[arg-type]

    def migrate_config(self, old_version: int, config: Mapping[str, object]):
        if old_version != self.config_schema_version:
            raise ValueError(f"no {self.provider_key} migration from schema {old_version}")
        return dict(config)


class _ExpertProvider:
    """Shared provider surface: routed-item contract in, routed-item contract out."""

    component_abi_version = 2
    config_schema_version = 1

    def contract(self, config: Mapping[str, object]) -> ComponentContract:
        item = _routed_item(int(config["width"]))  # type: ignore[arg-type]
        return ComponentContract(item, item)

    def migrate_config(self, old_version: int, config: Mapping[str, object]):
        if old_version != self.config_schema_version:
            raise ValueError(f"no {self.provider_key} migration from schema {old_version}")
        return dict(config)


class ReLU2MLPExpertProvider(_ExpertProvider):
    provider_key = MLP_EXPERT_KEY
    capabilities = ComponentCapabilities(
        eager=True,
        compile_fullgraph=True,
        dynamic_shapes=True,
        activation_checkpoint=True,
        amp_fp32=True,
        amp_bf16=True,
        amp_fp16=True,
        distributed=True,
        deterministic=True,
    )

    def validate_config(self, config: Mapping[str, object]) -> None:
        unknown = set(config) - {"width", "inner", "zero_output"}
        if unknown:
            raise ValueError(f"unknown {self.provider_key} fields: {sorted(unknown)}")
        if int(config["width"]) <= 0 or int(config["inner"]) <= 0:  # type: ignore[arg-type]
            raise ValueError("width and inner must be positive")

    def build(self, config: Mapping[str, object], context):
        del context
        return ReLU2MLPExpert(
            int(config["width"]),  # type: ignore[arg-type]
            int(config["inner"]),  # type: ignore[arg-type]
            zero_output=bool(config.get("zero_output", True)),
        )


class WorldAttentionExpertProvider(_ExpertProvider):
    provider_key = ATTENTION_EXPERT_KEY
    capabilities = ComponentCapabilities(
        eager=True,
        compile_fullgraph=True,
        dynamic_shapes=True,
        activation_checkpoint=True,
        amp_fp32=True,
        amp_bf16=True,
        amp_fp16=True,
        distributed=True,
        # Fused attention kernels choose their split strategy from the shapes
        # they are handed, so bit-exact repeats are not promised here.
        deterministic=False,
    )

    def validate_config(self, config: Mapping[str, object]) -> None:
        unknown = set(config) - {"width", "d_model", "heads", "ffn_ratio", "zero_output"}
        if unknown:
            raise ValueError(f"unknown {self.provider_key} fields: {sorted(unknown)}")
        width = int(config["width"])  # type: ignore[arg-type]
        d_model = int(config["d_model"])  # type: ignore[arg-type]
        heads = int(config["heads"])  # type: ignore[arg-type]
        if min(width, d_model, heads) <= 0:
            raise ValueError("width, d_model, and heads must be positive")
        if float(config["ffn_ratio"]) <= 0:  # type: ignore[arg-type]
            raise ValueError("ffn_ratio must be positive")
        if d_model % heads:
            raise ValueError("d_model must divide by heads")

    def build(self, config: Mapping[str, object], context):
        del context
        return WorldAttentionExpert(
            int(config["width"]),  # type: ignore[arg-type]
            int(config["d_model"]),  # type: ignore[arg-type]
            int(config["heads"]),  # type: ignore[arg-type]
            float(config["ffn_ratio"]),  # type: ignore[arg-type]
            zero_output=bool(config.get("zero_output", True)),
        )


def register_expert_providers() -> tuple[str, ...]:
    """Register both expert providers once; safe to call repeatedly."""
    for provider in (
        ReLU2MLPExpertProvider(),
        WorldAttentionExpertProvider(),
        FinalFieldProvider(),
    ):
        if provider.provider_key not in component_registry.keys():
            component_registry.register(
                provider,
                trusted=True,
                distribution=EXPERT_DISTRIBUTION,
                version=EXPERT_DISTRIBUTION_VERSION,
            )
        else:
            component_registry.authorize(provider.provider_key)
    return EXPERT_PROVIDER_KEYS


def _expert_spec(provider_key: str, config: dict) -> dict:
    return {
        "provider_key": provider_key,
        "provider_distribution": EXPERT_DISTRIBUTION,
        "provider_version": EXPERT_DISTRIBUTION_VERSION,
        "component_abi_version": 2,
        "config_schema_version": 1,
        "config": config,
    }


def expert_specs(
    *,
    width: int,
    experts: int,
    attention_experts: int,
    mlp_ratio: float,
    attention_d_model: int,
    attention_ffn_ratio: float,
    zero_output: bool = True,
) -> list[dict]:
    """Build the expert matrix: MLP experts first, attention experts last.

    Blocked rather than interleaved so a routing log reads directly as
    "how much traffic went to each family" without re-deriving indices.

    """
    if not 0 <= attention_experts <= experts:
        raise ValueError(f"attention_experts must be in [0, {experts}]")
    inner = int(round(float(mlp_ratio) * int(width)))
    if inner <= 0:
        raise ValueError("mlp_ratio is too small for a non-empty expert")
    specs: list[dict] = []
    for index in range(int(experts)):
        if index >= int(experts) - int(attention_experts):
            specs.append(
                _expert_spec(
                    ATTENTION_EXPERT_KEY,
                    {
                        "width": int(width),
                        "d_model": int(attention_d_model),
                        # Attention always splits its width into 64-wide groups
                        # that run side by side. Derived, never configured.
                        "heads": max(1, int(attention_d_model) // 64),
                        "ffn_ratio": float(attention_ffn_ratio),
                        "zero_output": bool(zero_output),
                    },
                )
            )
        else:
            specs.append(
                _expert_spec(
                    MLP_EXPERT_KEY,
                    {
                        "width": int(width),
                        "inner": int(inner),
                        "zero_output": bool(zero_output),
                    },
                )
            )
    return specs


def build_moe_lm(
    *,
    vocab: int,
    hidden_dim: int,
    state_dim: int,
    read_geometry: str = "seq",
    stack_residual: bool = True,
    tie_embeddings: bool = True,
    final_field: bool = True,
    experts: int,
    top_k: int,
    attention_experts: int,
    mlp_ratio: float,
    attention_d_model: int,
    attention_ffn_ratio: float = 4.0,
    router: str = "switch",
    balance_coefficient: float = 0.01,
    bias_update_rate: float = 1.0e-3,
    zero_output: bool = True,
) -> DABSNSequenceLM:
    """Resolve the whole body from provider specifications and wrap it as an LM.

    ``final_field=True`` puts the collapse between DABSN and the MoE::

        tokens -> DABSN scan -> [B, T, H]
               -> final field -> [B, 1, H]      one world per sequence
               -> sparse MoE  -> [B, 1, H]      routed once
               -> readout     -> [B, 1, vocab]  one prediction per sequence

    ``final_field=False`` leaves it out and the MoE routes every world DABSN
    built::

        tokens -> DABSN scan -> [B, T, H]
               -> sparse MoE  -> [B, T, H]      routed per position
               -> readout     -> [B, T, vocab]  a prediction at every position

    Same components, same experts, same router -- the graph is one node longer
    or shorter. Nothing downstream branches on which one you picked: the loss
    reads the model's own output shape (see ``aligned_targets``), and the
    experts never had an experience axis to care about in the first place.
    """

    register_expert_providers()
    block = component_registry.build(
        ComponentSpec(
            "dabsn.0",
            "dabsn:block",
            {
                "input_dim": int(hidden_dim),
                "hidden_dim": int(hidden_dim),
                "state_dim": int(state_dim),
                "read_geometry": str(read_geometry),
                "residual": bool(stack_residual),
            },
        )
    )
    field = (
        component_registry.build(
            ComponentSpec("field.0", FINAL_FIELD_KEY, {"width": int(hidden_dim)})
        )
        if final_field
        else None
    )
    moe_config: dict = {
        "hidden_dim": int(hidden_dim),
        "experts": int(experts),
        "top_k": int(top_k),
        "router": str(router),
        "normalization": "rmsnorm",
        "residual": True,
        "routing_granularity": "individual_h",
        "expert_specs": expert_specs(
            width=int(hidden_dim),
            experts=int(experts),
            attention_experts=int(attention_experts),
            mlp_ratio=float(mlp_ratio),
            attention_d_model=int(attention_d_model),
            attention_ffn_ratio=float(attention_ffn_ratio),
            zero_output=bool(zero_output),
        ),
    }
    if router == "switch":
        moe_config["balance_coefficient"] = float(balance_coefficient)
    else:
        moe_config["bias_update_rate"] = float(bias_update_rate)
    moe = component_registry.build(ComponentSpec("moe.0", "dabsn:sparse_moe", moe_config))
    ordered = [block, field, moe] if field is not None else [block, moe]
    graph = DABSNGraph(ordered, require_world_builder=True)
    return DABSNSequenceLM.from_graph(graph, vocab=int(vocab), tie_embeddings=bool(tie_embeddings))


def moe_component(model: DABSNSequenceLM):
    """The single sparse-MoE component of a graph-built model, or None."""
    for binding in model.graph.bindings:
        if binding.provider_key == "dabsn:sparse_moe":
            return binding.module
    return None


def dabsn_blocks(model: DABSNSequenceLM) -> list[nn.Module]:
    """Every DABSN block, from the legacy backbone or from a component graph."""
    blocks = getattr(model.backbone, "blocks", None)
    if blocks is not None:
        return list(blocks)
    return [
        binding.module
        for binding in model.graph.bindings
        if getattr(binding.module, "read_gain", None) is not None
    ]


def moe_params(model: DABSNSequenceLM) -> int:
    """Parameters belonging to the MoE branch: router, norm, and every expert."""
    branch = moe_component(model)
    if branch is None:
        return 0
    return sum(parameter.numel() for parameter in branch.parameters())


def branch_summary(model: DABSNSequenceLM) -> dict:
    """Per-family parameter accounting, active cost, and the parity delta."""
    branch = moe_component(model)
    if branch is None:
        return {}
    experts = list(branch.expert_group.experts)
    families: dict[str, dict] = {}
    for index, expert in enumerate(experts):
        name = "attention" if isinstance(expert, WorldAttentionExpert) else "mlp"
        entry = families.setdefault(name, {"count": 0, "params": 0, "indices": []})
        entry["count"] += 1
        entry["params"] += sum(p.numel() for p in expert.parameters())
        entry["indices"].append(index)
    for entry in families.values():
        entry["params_each"] = entry["params"] // max(1, entry["count"])
        entry["indices"] = [min(entry["indices"]), max(entry["indices"])]
    stored = sum(entry["params"] for entry in families.values())
    mean_expert = stored / max(1, len(experts))
    return {
        "experts": len(experts),
        "top_k": int(branch.router.top_k),
        "families": families,
        "router_params": sum(p.numel() for p in branch.router.parameters()),
        "norm_params": sum(p.numel() for p in branch.normalization.parameters()),
        "stored_expert_params": stored,
        "expected_active_expert_params": int(round(mean_expert * int(branch.router.top_k))),
        "sparsity": len(experts) / max(1, int(branch.router.top_k)),
    }


def aligned_targets(logits: Tensor, targets: Tensor) -> Tensor:
    """Match a target tensor to whatever the model actually predicted.

    A per-position model returns one prediction for every experience step and
    scores every shifted target. A one-world-per-sequence model returns a
    single prediction -- the token that follows the whole window -- so only the
    last target applies. Driven by the returned shape rather than a flag, so
    both model kinds go through the identical loss code.
    """
    return targets[:, -1:] if logits.shape[1] == 1 else targets


class RouterTelemetry:
    """Accumulate the graph's declared router reports across microbatches.

    ``dabsn:sparse_moe`` declares ten reports by name; the graph hands them back
    in ``ComponentOutput.reports`` in ``ROUTER_REPORT_NAMES`` order. Nothing here
    reaches into the component's internals -- this is the framework's own
    observability surface, read the way a component consumer is meant to read it.
    """

    def __init__(self, experts: int, attention_experts: int) -> None:
        self.experts = int(experts)
        self.attention_experts = int(attention_experts)
        self.mlp_experts = self.experts - self.attention_experts
        self._counts = torch.zeros(self.experts, dtype=torch.float64)
        self._confidence = 0.0
        self._differentiation = 0.0
        self._assignments = 0.0
        self._forwards = 0

    def observe(self, reports: Sequence[Tensor]) -> None:
        if len(reports) != len(ROUTER_REPORT_NAMES):
            raise RuntimeError(
                f"expected {len(ROUTER_REPORT_NAMES)} router reports, got {len(reports)}; "
                "the graph holds more than one reporting component"
            )
        named = dict(zip(ROUTER_REPORT_NAMES, reports))
        with torch.no_grad():
            assignments = float(named["assignment_count"].detach().cpu())
            self._counts += named["expert_counts"].detach().double().cpu()
            self._confidence += float(named["selected_confidence"].detach().cpu()) * assignments
            self._differentiation += (
                float(named["output_norm_differentiation"].detach().cpu()) * assignments
            )
            self._assignments += assignments
            self._forwards += 1

    def pop(self) -> dict | None:
        """Interval statistics since the last call, then reset."""
        if not self._forwards or self._assignments <= 0:
            return None
        counts = self._counts
        total = float(counts.sum())
        shares = (counts / max(total, 1.0)).tolist()
        uniform = 1.0 / self.experts
        entropy = 0.0
        for share in shares:
            if share > 0:
                entropy -= share * math.log(share)
        stats = {
            "experts": self.experts,
            "forwards": self._forwards,
            "balance": entropy / math.log(self.experts) if self.experts > 1 else 1.0,
            "cold": int(sum(1 for share in shares if share == 0.0)),
            "max_frac": max(shares),
            "min_frac": min(shares),
            "uniform": uniform,
            "conf": self._confidence / self._assignments,
            "spread": self._differentiation / self._assignments,
            "mlp_frac": float(sum(shares[: self.mlp_experts])),
            "attention_frac": float(sum(shares[self.mlp_experts :])),
            "mlp_experts": self.mlp_experts,
            "attention_experts": self.attention_experts,
            "shares": [round(share, 5) for share in shares],
        }
        self._counts = torch.zeros(self.experts, dtype=torch.float64)
        self._confidence = 0.0
        self._differentiation = 0.0
        self._assignments = 0.0
        self._forwards = 0
        return stats


class MoEForward:
    """Logits for the training loop; declared loss terms and reports on the side.

    The loop wants a callable returning logits, and the MoE's load-balance term
    is a declared graph result rather than something stashed on a module. This
    adapter calls the authoritative ``forward_with_terms`` path once, hands the
    loop its logits, and keeps the balance term for the loss and the router
    reports for the log line.
    """

    def __init__(
        self,
        model: DABSNSequenceLM,
        telemetry: RouterTelemetry,
        *,
        compile_forward: bool = True,
    ) -> None:
        self.model = model
        self.telemetry = telemetry
        self._aux: Tensor | None = None
        self._call = (
            torch.compile(model.forward_with_terms, dynamic=False)
            if compile_forward
            else model.forward_with_terms
        )

    def __call__(self, ids: Tensor) -> Tensor:
        result = self._call(ids)
        self._aux = (
            None
            if not result.loss_terms
            else torch.stack([term.float() for term in result.loss_terms]).sum()
        )
        if result.reports:
            self.telemetry.observe(result.reports)
        return result.value

    def pop_aux(self) -> Tensor | None:
        """The declared balance term for the forward just run, consumed once.

        The coefficient is already inside the router's term, so the trainer adds
        this as-is; multiplying again would double-apply it.
        """
        aux, self._aux = self._aux, None
        return aux


__all__ = [
    "ATTENTION_EXPERT_KEY",
    "aligned_targets",
    "FINAL_FIELD_KEY",
    "FinalFieldComponent",
    "FinalFieldProvider",
    "attention_expert_params",
    "EXPERT_DISTRIBUTION",
    "EXPERT_DISTRIBUTION_VERSION",
    "EXPERT_PROVIDER_KEYS",
    "MLP_EXPERT_KEY",
    "MoEForward",
    "ReLU2MLPExpert",
    "ReLU2MLPExpertProvider",
    "RouterTelemetry",
    "WorldAttentionExpert",
    "WorldAttentionExpertProvider",
    "branch_summary",
    "build_moe_lm",
    "dabsn_blocks",
    "expert_specs",
    "moe_component",
    "moe_params",
    "register_expert_providers",
]
