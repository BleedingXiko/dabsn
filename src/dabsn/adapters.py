"""Built-in DABSN adapters and registration APIs for task-specific modules."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from torch import Tensor


class IdentityInputAdapter(nn.Module):
    """Already-vectorized sequence or field input."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.output_dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.float()


class LinearInputAdapter(nn.Module):
    """Continuous features projected to the DABSN input width."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.proj(inputs.float())


class ByteInputAdapter(nn.Module):
    """Byte ids ``[B,T]`` to vectors ``[B,T,D]``."""

    def __init__(self, output_dim: int, vocab_size: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.emb = nn.Embedding(vocab_size, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.emb(inputs.long())


class LinearOutputHead(nn.Module):
    """Per-position linear output head."""

    def __init__(self, input_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, out_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.proj(hidden)


class ByteOutputHead(LinearOutputHead):
    def __init__(self, input_dim: int, out_dim: int = 256) -> None:
        del out_dim
        super().__init__(input_dim, 256)


InputBuilder = Callable[[int, int | None], nn.Module]
OutputBuilder = Callable[[int, int], nn.Module]

_INPUT_ADAPTERS: dict[str, InputBuilder] = {
    "identity": lambda input_dim, output_dim: IdentityInputAdapter(
        input_dim if output_dim is None else output_dim
    ),
    "tensor": lambda input_dim, output_dim: IdentityInputAdapter(
        input_dim if output_dim is None else output_dim
    ),
    "linear": lambda input_dim, output_dim: LinearInputAdapter(
        input_dim,
        input_dim if output_dim is None else output_dim,
    ),
    "byte": lambda input_dim, output_dim: ByteInputAdapter(
        input_dim if output_dim is None else output_dim
    ),
}

_OUTPUT_HEADS: dict[str, OutputBuilder] = {
    "linear": LinearOutputHead,
    "field": LinearOutputHead,
    "token": LinearOutputHead,
    "byte": ByteOutputHead,
}


def register_input_adapter(kind: str, builder: InputBuilder) -> None:
    """Register an input-adapter builder under a case-insensitive name."""
    _INPUT_ADAPTERS[kind.lower()] = builder


def register_output_head(kind: str, builder: OutputBuilder) -> None:
    """Register an output-head builder under a case-insensitive name."""
    _OUTPUT_HEADS[kind.lower()] = builder


def build_input_adapter(
    kind: str,
    input_dim: int,
    output_dim: int | None = None,
) -> nn.Module:
    """Construct a registered input adapter."""
    normalized = kind.lower()
    if normalized not in _INPUT_ADAPTERS:
        raise ValueError(f"unknown DABSN input adapter: {normalized}")
    return _INPUT_ADAPTERS[normalized](input_dim, output_dim)


def build_output_head(kind: str, input_dim: int, out_dim: int) -> nn.Module:
    """Construct a registered output head."""
    normalized = kind.lower()
    if normalized not in _OUTPUT_HEADS:
        raise ValueError(f"unknown DABSN output head: {normalized}")
    return _OUTPUT_HEADS[normalized](input_dim, out_dim)
