"""External providers: this package contains no DABSN source modifications."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
from torch import Tensor

from dabsn.components import (
    AxisContract,
    BuildContext,
    ComponentCapabilities,
    ComponentContract,
    ValueContract,
)


@torch.library.custom_op("dabsn_fixture::scale", mutates_args=())
def fixture_scale(value: Tensor, scale: float) -> Tensor:
    return value * scale


@fixture_scale.register_fake
def _fixture_scale_fake(value: Tensor, scale: float):
    return torch.empty_like(value)


def _scale_setup(ctx, inputs, output):
    _, scale = inputs
    ctx.scale = scale


def _scale_backward(ctx, grad):
    return grad * ctx.scale, None


torch.library.register_autograd(fixture_scale, _scale_backward, setup_context=_scale_setup)


def _world(width: int) -> ValueContract:
    return ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("experience", "T", dynamic=True),
        AxisContract("world", width),
    )


def _expert_item(width: int) -> ValueContract:
    return ValueContract.tensor(
        AxisContract("dabsn:routed_item", "N", dynamic=True),
        AxisContract("world", width),
    )


class _BaseProvider:
    component_abi_version = 2
    config_schema_version = 1
    capabilities = ComponentCapabilities(
        eager=True,
        compile_fullgraph=True,
        dynamic_shapes=True,
        export=True,
        amp_fp32=True,
        amp_bf16=True,
        amp_fp16=True,
        deterministic=False,
    )

    def validate_config(self, config: Mapping[str, object]) -> None:
        if int(config["width"]) <= 0:
            raise ValueError("width must be positive")

    def contract(self, config: Mapping[str, object]) -> ComponentContract:
        world = _world(int(config["width"]))
        return ComponentContract(world, world)

    def migrate_config(self, old_version: int, config: Mapping[str, object]):
        if old_version != 1:
            raise ValueError(f"no migration from {old_version}")
        return dict(config)


class ScaleModule(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, value: Tensor) -> Tensor:
        return fixture_scale(value, self.scale)


class ScaleProvider(_BaseProvider):
    provider_key = "fixture.scale"

    def build(self, config: Mapping[str, object], context: BuildContext):
        return ScaleModule(float(config.get("scale", 1.0)))


class HAttentionModule(nn.Module):
    """Each complete H-world becomes attention's native sequence."""

    def __init__(self, width: int, latent: int, heads: int):
        super().__init__()
        self.width = width
        self.to_latent = nn.Linear(1, latent)
        self.attention = nn.MultiheadAttention(latent, heads, batch_first=True)
        self.to_world = nn.Linear(latent, 1)

    def forward(self, value: Tensor) -> Tensor:
        shape = value.shape
        worlds = value.reshape(-1, self.width, 1)
        latent = self.to_latent(worlds)
        mixed, _ = self.attention(latent, latent, latent, need_weights=False)
        return (worlds + self.to_world(mixed)).reshape(shape)


class HAttentionProvider(_BaseProvider):
    provider_key = "fixture.h_attention"

    def validate_config(self, config):
        super().validate_config(config)
        if int(config["latent"]) % int(config["heads"]):
            raise ValueError("latent must be divisible by heads")

    def build(self, config, context):
        return HAttentionModule(int(config["width"]), int(config["latent"]), int(config["heads"]))


class CNNModule(nn.Module):
    """Structure-native temporal CNN retaining both T and H."""

    def __init__(self, width: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel, padding=kernel // 2)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.conv(value.transpose(1, 2)).transpose(1, 2)


class CNNProvider(_BaseProvider):
    provider_key = "fixture.cnn"

    def build(self, config, context):
        return CNNModule(int(config["width"]), int(config.get("kernel", 3)))


class TransformerProvider(_BaseProvider):
    provider_key = "fixture.transformer"

    def validate_config(self, config):
        super().validate_config(config)
        if int(config["width"]) % int(config.get("heads", 2)):
            raise ValueError("width must be divisible by heads")

    def build(self, config, context):
        return nn.TransformerEncoderLayer(
            d_model=int(config["width"]),
            nhead=int(config.get("heads", 2)),
            dim_feedforward=int(config.get("inner", int(config["width"]) * 4)),
            batch_first=True,
        )


class WorldTransformerExpert(nn.Module):
    """A complete H-world is the transformer's native sequence."""

    def __init__(self, width: int, latent: int, heads: int, inner: int):
        super().__init__()
        self.width = int(width)
        self.to_latent = nn.Linear(1, latent)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=latent,
            nhead=heads,
            dim_feedforward=inner,
            batch_first=True,
        )
        self.to_world = nn.Linear(latent, 1)

    def forward(self, value: Tensor) -> Tensor:
        worlds = value.reshape(-1, self.width, 1)
        latent = self.to_latent(worlds)
        transformed = self.transformer(latent)
        return self.to_world(transformed).reshape_as(value)


class WorldTransformerExpertProvider(_BaseProvider):
    provider_key = "fixture.world_transformer_expert"

    def validate_config(self, config):
        super().validate_config(config)
        if int(config["latent"]) % int(config["heads"]):
            raise ValueError("latent must be divisible by heads")

    def contract(self, config):
        value = _expert_item(int(config["width"]))
        return ComponentContract(value, value)

    def build(self, config, context):
        return WorldTransformerExpert(
            int(config["width"]),
            int(config["latent"]),
            int(config["heads"]),
            int(config["inner"]),
        )
