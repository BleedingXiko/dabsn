"""DABSN memory as a native sidecar for arbitrary host sequence models.

``DABSNHostMemory`` deliberately instantiates no ``DABSNCore`` and no
``DABSNBlock``. A host architecture (Transformer attention, SSM, CNN, MLP, ...)
creates the latent experience stream; the canonical ``DABSNRead`` owns
admission, the admitted content/permanent/induction reads, and LONG
prediction-error memory, unchanged.

The only DABSN-behaviour override is the initial admission window (1.5 instead
of the native-core default softplus(4) ~= 4.02), calibrated for host latents.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .read import DABSNRead

__all__ = ["DABSNHostMemory", "HostMemoryComponent"]


class _HostMemoryRead(DABSNRead):
    """Accelerated host-memory read. Kernel flags live on this subclass only,
    so enabling them can never activate DABSNCore scans or the native read
    used by ordinary DABSN blocks."""


class _HostMemoryReferenceRead(DABSNRead):
    """Exact reference host-memory read; never receives kernel flags.

    The flags are pinned off here, so enabling kernels on ``DABSNRead`` (or any
    other subclass) can never reach the reference read through inheritance."""

    _cuda_native_enabled = False
    _cuda_native_required = False
    _long_native_enabled = False
    _long_native_required = False
    _cpu_native_read_enabled = False
    _cpu_native_required = False


class DABSNHostMemory(nn.Module):
    """Canonical non-core DABSN memory for a host latent stream ``[B,T,H]``.

    Tiny feed-forward adapters produce the tensors the canonical read expects:

      expression -> query/content address
      write      -> plasticity * remembered host latent
      novelty/plasticity/energy/saturation -> four scalar tags widened as views

    ``forward`` returns the memory branch only (``y``/``budget`` are zero), never
    an implicit residual of the host latent. Hosts add it where they choose.
    ``use_long=False`` exists only as an ablation/control.
    """

    def __init__(
        self,
        dim: int,
        *,
        read_geometry: Literal["seq", "field", "hybrid"] = "seq",
        admit_window: float = 1.5,
        use_long: bool = True,
        kernels: Literal["auto", "reference"] = "auto",
    ) -> None:
        super().__init__()
        if int(dim) <= 0:
            raise ValueError("dim must be positive")
        if not float(admit_window) > 0:
            raise ValueError("admit_window must be > 0")
        if kernels not in {"auto", "reference"}:
            raise ValueError("kernels must be 'auto' or 'reference'")

        self.dim = int(dim)
        self.admit_window = float(admit_window)
        self.use_long = bool(use_long)
        read_type = _HostMemoryRead if kernels == "auto" else _HostMemoryReferenceRead
        self.read = read_type(self.dim, read_geometry=read_geometry)

        # DABSN parameterizes the positive admission margin through softplus.
        raw_window = math.log(math.expm1(self.admit_window))
        with torch.no_grad():
            self.read.admit_window.fill_(raw_window)

        # The host creates experience. Feed-forward adapters, never recurrent.
        self.write_tags = nn.Linear(self.dim, 4)
        self.query_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.write_proj = nn.Linear(self.dim, self.dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.query_proj.weight)
            nn.init.eye_(self.write_proj.weight)

        if not self.use_long:
            with torch.no_grad():
                self.read.long_gain.zero_()
            self.read.long_gain.requires_grad_(False)

        self._kernel_mode = kernels
        self._kernel_init: dict[str, object] = {}
        if kernels == "auto":
            self._enable_memory_kernels()

    def _enable_memory_kernels(self) -> None:
        """Enable only memory/read acceleration; never DABSNCore scans."""
        _HostMemoryRead._long_native_enabled = True
        _HostMemoryRead._long_native_required = False
        self._kernel_init["long"] = True
        try:
            from .kernels.cpu import enable_native_cpu_read

            self._kernel_init["cpu_admitted_read"] = bool(enable_native_cpu_read(required=False))
        except Exception as exc:  # the reference path remains exact
            self._kernel_init["cpu_admitted_read"] = False
            self._kernel_init["cpu_error"] = f"{type(exc).__name__}: {exc}"
        try:
            from .kernels.triton import triton_available

            cuda_memory = bool(torch.cuda.is_available() and triton_available())
            if cuda_memory:
                # Register only the compact-flash read operator. Avoid
                # enable_triton_kernels(), which also toggles DABSNCore scans.
                from .kernels.compact_flash import register_compact_flash_cuda
                from .kernels.triton import _runtime

                register_compact_flash_cuda(_runtime())
            _HostMemoryRead._cuda_native_enabled = cuda_memory
            _HostMemoryRead._cuda_native_required = False
            self._kernel_init["cuda_admitted_read"] = cuda_memory
        except Exception as exc:
            self._kernel_init["cuda_admitted_read"] = False
            self._kernel_init["cuda_error"] = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _widen(scalar: Tensor, dim: int) -> Tensor:
        # DABSNRead consumes these four fields only via mean(dim=-1): a stride-0
        # view supplies the [B,T,H] interface without four H-wide copies.
        return scalar.unsqueeze(-1).expand(-1, -1, dim)

    def signals_from_host(self, host_latents: Tensor) -> dict[str, Tensor]:
        if host_latents.dim() != 3 or host_latents.shape[-1] != self.dim:
            raise ValueError(
                f"expected host_latents [B,T,{self.dim}], got {tuple(host_latents.shape)}"
            )
        tags = torch.sigmoid(self.write_tags(host_latents))
        novelty_s, plasticity_s, energy_s, saturation_s = tags.unbind(dim=-1)
        expression = self.query_proj(host_latents)
        plasticity = self._widen(plasticity_s, self.dim)
        # Same write contract as DABSNCore: write = plasticity * expression; the
        # remembered basis may differ from the query basis. No recurrence here.
        write = plasticity * self.write_proj(host_latents)
        zeros = torch.zeros_like(host_latents)
        return {
            # Core-state channels are zero: the branch is memory-only.
            "y": zeros,
            "budget": zeros,
            "expression": expression,
            "write": write,
            "novelty": self._widen(novelty_s, self.dim),
            "plasticity": plasticity,
            "energy": self._widen(energy_s, self.dim),
            "saturation": self._widen(saturation_s, self.dim),
        }

    def forward(self, host_latents: Tensor) -> Tensor:
        # The public canonical read call: admission + content/permanent/
        # induction + LONG. No hook, copied equation, DABSNCore or DABSNBlock.
        return self.read(**self.signals_from_host(host_latents))

    @property
    def admitted_bank(self) -> int | None:
        """Packed admitted width of the latest call (batch maximum)."""
        return self.read.last_n_max

    @property
    def read_contract(self) -> dict[str, object]:
        return dict(self.read.last_read_contract)

    @property
    def kernel_status(self) -> dict[str, object]:
        status = dict(self._kernel_init)
        status.update({
            "kernel_mode": self._kernel_mode,
            "last_three_way_backend": getattr(self.read, "_last_three_way_backend", "eager_reference"),
            "last_long_backend": getattr(self.read, "_last_long_backend", None),
            "core_scan_instantiated": False,
        })
        return status


class HostMemoryComponent(nn.Module):
    """Graph component: ``x + DABSNHostMemory(x)`` on the world stream.

    The residual lives here, at the composition boundary, exactly as
    ``ResidualMLPComponent`` adds its branch. ``DABSNHostMemory`` itself stays
    memory-only.
    """

    def __init__(self, dim: int, **memory_config) -> None:
        super().__init__()
        self.memory = DABSNHostMemory(dim, **memory_config)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.memory(inputs)
