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
        from .optimizer import capacity_sparse_parameters
        self.sparse_parameters = set(capacity_sparse_parameters(module))
        self.buffers = {
            parameter: torch.zeros_like(
                parameter,
                dtype=dtype,
                memory_format=torch.preserve_format,
            )
            for parameter in self.parameters if parameter not in self.sparse_parameters
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
        from ..kernels.capacity import finalize_capacity_gradients
        finalize_capacity_gradients(self.module)
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            # TensorIterator converts the source while adding into the FP32
            # destination. A separate .to() duplicates the entire gradient.
            if parameter in self.sparse_parameters or gradient.is_sparse:
                from .optimizer import as_row_sparse
                contribution = as_row_sparse(gradient.detach(), getattr(parameter, "_capacity_touched_rows", None)).to(self.dtype) * float(scale)
                if parameter in self.active:
                    contribution = self.buffers[parameter] + contribution
                self.buffers[parameter] = contribution.coalesce()
            else:
                self.buffers[parameter].add_(gradient.detach(), alpha=float(scale))
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
            touched = getattr(parameter, "_capacity_touched_rows", None)
            if touched is not None:
                touched.zero_()
            if self.buffers[parameter].is_sparse:
                del self.buffers[parameter]
            else:
                self.buffers[parameter].zero_()
        self.active.clear()
        self.microbatches = 0


__all__ = ["ManualGradientAccumulator"]
