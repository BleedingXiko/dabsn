"""Explicit gradient buffers for replay-safe microbatch accumulation.

Each CUDA-graph replay is treated as an independent single backward. Parameter
gradients are cleared before the replay, copied into persistent accumulation
buffers afterwards, and only the completed sum is installed for the optimizer.
Correctness therefore depends only on single-backward graph parity; it never
depends on autograd accumulating across graph replays or CUDA streams.
"""

from __future__ import annotations

import torch
from torch import nn


class ManualGradientAccumulator:
    """Accumulate independent microbatch gradients outside autograd.

    DABSN training keeps FP32 master parameters, so the default FP32 buffers
    preserve the same accumulation precision as ordinary parameter gradients.
    The implementation is task and shape agnostic and works with eager or
    CUDA-graphed callables.
    """

    def __init__(self, module: nn.Module, *, dtype: torch.dtype = torch.float32):
        self.module = module
        self.dtype = dtype
        self.parameters = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        self.buffers = {
            parameter: torch.zeros_like(
                parameter,
                dtype=dtype,
                memory_format=torch.preserve_format,
            )
            for parameter in self.parameters
        }
        self.active: set[nn.Parameter] = set()
        self.microbatches = 0

    def begin_microbatch(self) -> None:
        """Give the next backward a clean parameter-gradient destination."""

        self.module.zero_grad(set_to_none=True)

    @torch.no_grad()
    def add_microbatch(self, *, scale: float = 1.0) -> None:
        """Add one completed backward to persistent buffers, then clear it."""

        found = False
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            self.buffers[parameter].add_(gradient.detach().to(self.dtype), alpha=float(scale))
            self.active.add(parameter)
            found = True
        if not found:
            raise RuntimeError("manual gradient accumulation found no parameter gradients")
        self.microbatches += 1
        self.module.zero_grad(set_to_none=True)

    @torch.no_grad()
    def install(self) -> None:
        """Expose the completed accumulation as ``parameter.grad`` for stepping."""

        if self.microbatches < 1:
            raise RuntimeError("cannot install an empty manual gradient accumulation")
        self.module.zero_grad(set_to_none=True)
        for parameter in self.active:
            buffer = self.buffers[parameter]
            parameter.grad = (
                buffer if buffer.dtype == parameter.dtype else buffer.to(parameter.dtype)
            )

    @torch.no_grad()
    def reset(self) -> None:
        """Clear installed gradients and buffers after an optimizer step."""

        self.module.zero_grad(set_to_none=True)
        for parameter in self.active:
            self.buffers[parameter].zero_()
        self.active.clear()
        self.microbatches = 0


__all__ = ["ManualGradientAccumulator"]
