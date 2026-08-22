"""Explicit DABSN kernel selection, status, and primitive map."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .admitted import admitted_three_way_read
from .batched_runtime import core_scan_batched
from .compact_admitted import (
    admitted_three_way_read_compact_dense,
    admitted_three_way_read_compact_dense_bmm,
)
from .compact_flash import admitted_three_way_read_compact_flash
from .local_field import local_field_gather, local_field_gather_stats
from .long import linear_recurrence
from .permanent import (
    permanent_delta_read,
    permanent_delta_scan,
    permanent_runtime_status,
)

_ACTIVE_BACKEND = "reference"


@dataclass(frozen=True)
class Primitive:
    name: str
    reference: str
    cpu: str
    cuda: str
    backward: str
    geometries: tuple[str, ...]
    validation: str


PRIMITIVES = (
    Primitive(
        "core_scan",
        "DABSNCore.forward / DABSNCore.forward_from_state",
        "C++/OpenMP",
        "Triton",
        "C++ reverse scan / Triton fused reverse scan",
        ("seq", "field", "hybrid"),
        "reference/native full and carried-state forward/backward parity",
    ),
    Primitive(
        "admitted_three_way_read",
        "DABSNRead._three_way_read",
        "ATen BLAS or forced C++/OpenMP",
        "registered compact dense / registered compact flash Triton",
        "registered ATen/C++ analytic or compact Triton backward",
        ("seq", "field", "hybrid"),
        "reference/native forward and backward parity",
    ),
    Primitive(
        "permanent_delta_read",
        "permanent_delta_scan",
        "C++/OpenMP",
        "Triton",
        "C++ reverse recurrence / Triton reverse recurrence",
        ("seq",),
        "CPU forward and backward comparison",
    ),
    Primitive(
        "recurrent_long_read",
        "DABSNRead._long_scan_from_state",
        "C++/OpenMP linear recurrence",
        "Triton linear recurrence",
        "C++/Triton analytic reverse recurrence",
        ("seq", "field", "hybrid"),
        "CPU model-path forward and backward comparison",
    ),
    Primitive(
        "local_field_gather",
        "inputs[:, patch]",
        "C++/OpenMP gather/scatter",
        "Triton gather/scatter",
        "C++/Triton scatter-add",
        ("field", "hybrid"),
        "CPU and CUDA gather/scatter parity",
    ),
)


def primitive_map() -> list[dict[str, object]]:
    return [asdict(primitive) for primitive in PRIMITIVES]


def enable(
    backend: str = "auto",
    *,
    required: bool = False,
) -> dict[str, object]:
    """Enable one explicit backend family.

    ``required=True`` refuses a missing extension when the backend is enabled.
    Installed class-level hooks remain device-safe: a later tensor on the other
    device uses the eager implementation instead of inheriting a stale
    process-wide ``required`` flag.
    """

    global _ACTIVE_BACKEND
    normalized = backend.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized not in {"reference", "cpu", "cuda"}:
        raise ValueError("backend must be one of: auto, reference, cpu, cuda")
    if _ACTIVE_BACKEND not in {"reference", normalized}:
        raise RuntimeError(
            f"DABSN backend already selected as {_ACTIVE_BACKEND}; "
            "select a different backend in a fresh process"
        )
    if normalized == "reference":
        _ACTIVE_BACKEND = normalized
        return status()
    if normalized == "cpu":
        from .cpu import enable_native_cpu_kernels
        from .local_field import enable_native_cpu_local_field
        from .long import enable_fused_long_read
        from .permanent import enable_native_cpu_permanent

        results = enable_native_cpu_kernels(required=required)
        results["permanent_delta_scan"] = enable_native_cpu_permanent(required=required)
        results["local_field_gather"] = enable_native_cpu_local_field(required=required)
        enable_fused_long_read(required_cuda=False)
        if required and not all(results.values()):
            raise RuntimeError(f"required native CPU primitives unavailable: {results}")
    else:
        from .long import enable_fused_long_read
        from .triton import enable_triton_kernels

        results = enable_triton_kernels(required=required)
        enable_fused_long_read(required_cuda=required)
        if required and not all(results.values()):
            raise RuntimeError(f"required Triton primitives unavailable: {results}")
    _ACTIVE_BACKEND = normalized
    return status()


def status() -> dict[str, object]:
    from .long import long_runtime_status
    from .triton import triton_status

    cpu_status: dict[str, object]
    try:
        from .cpu import native_cpu_status

        cpu_status = native_cpu_status()
    except Exception as exc:
        cpu_status = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "active_backend": _ACTIVE_BACKEND,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu": cpu_status,
        "cuda": triton_status(),
        "long_memory": long_runtime_status(),
        "permanent": permanent_runtime_status(),
        "local_field": local_field_gather_stats(),
        "primitives": primitive_map(),
    }


__all__ = [
    "PRIMITIVES",
    "Primitive",
    "admitted_three_way_read",
    "admitted_three_way_read_compact_dense",
    "admitted_three_way_read_compact_dense_bmm",
    "admitted_three_way_read_compact_flash",
    "core_scan_batched",
    "enable",
    "linear_recurrence",
    "local_field_gather",
    "local_field_gather_stats",
    "permanent_delta_read",
    "permanent_delta_scan",
    "primitive_map",
    "status",
]
