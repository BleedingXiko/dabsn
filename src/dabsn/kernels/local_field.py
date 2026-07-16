"""Triton gather/scatter for adapter-side 2D local field patches."""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
    _TRITON_IMPORT_ERROR = None
except Exception as exc:
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = exc

_GATHER_HITS = 0
_GATHER_FALLBACKS = 0
_CPU_NATIVE_ENABLED = False
_CPU_NATIVE_REQUIRED = False
_CPU_GATHER_HITS = 0


def enable_native_cpu_local_field(*, required: bool = False) -> bool:
    """Enable the C++/OpenMP local-field gather for CPU tensors."""

    global _CPU_NATIVE_ENABLED, _CPU_NATIVE_REQUIRED
    from .cpu import _load_ext

    extension = _load_ext(required=required)
    _CPU_NATIVE_ENABLED = extension is not None
    _CPU_NATIVE_REQUIRED = bool(required)
    return _CPU_NATIVE_ENABLED


def local_field_gather_stats() -> dict[str, object]:
    return {
        "triton_available": bool(_HAS_TRITON),
        "cpu_native_enabled": bool(_CPU_NATIVE_ENABLED),
        "cpu_hits": int(_CPU_GATHER_HITS),
        "hits": int(_GATHER_HITS),
        "fallbacks": int(_GATHER_FALLBACKS),
        "import_error": (
            None if _TRITON_IMPORT_ERROR is None else repr(_TRITON_IMPORT_ERROR)
        ),
    }


if _HAS_TRITON:

    @triton.jit
    def _local_field_gather_fwd(
        x,
        patch,
        out,
        total: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        d: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        feature = offsets % d
        temporary = offsets // d
        neighbor = temporary % k
        cell = (temporary // k) % n
        batch = temporary // (k * n)
        source_cell = tl.load(
            patch + cell * k + neighbor,
            mask=mask,
            other=0,
        )
        values = tl.load(
            x + (batch * n + source_cell) * d + feature,
            mask=mask,
            other=0.0,
        )
        tl.store(out + offsets, values, mask=mask)

    @triton.jit
    def _local_field_gather_bwd(
        grad_out,
        patch,
        grad_x,
        total: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        d: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        feature = offsets % d
        temporary = offsets // d
        neighbor = temporary % k
        cell = (temporary // k) % n
        batch = temporary // (k * n)
        source_cell = tl.load(
            patch + cell * k + neighbor,
            mask=mask,
            other=0,
        )
        values = tl.load(grad_out + offsets, mask=mask, other=0.0)
        tl.atomic_add(
            grad_x + (batch * n + source_cell) * d + feature,
            values,
            sem="relaxed",
            mask=mask,
        )


class _LocalFieldGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, patch: Tensor) -> Tensor:
        if not (_HAS_TRITON and inputs.is_cuda and patch.is_cuda):
            raise RuntimeError("local-field Triton gather requires CUDA tensors")
        if inputs.dim() != 3 or patch.dim() != 2:
            raise ValueError(
                f"expected inputs [B,N,D], patch [N,K], got "
                f"{tuple(inputs.shape)} and {tuple(patch.shape)}"
            )
        batch, cells, dim = inputs.shape
        patch_cells, neighbors = patch.shape
        if int(patch_cells) != int(cells):
            raise ValueError(
                f"patch index N={patch_cells} does not match input N={cells}"
            )
        contiguous_inputs = inputs.contiguous()
        contiguous_patch = patch.to(
            device=inputs.device,
            dtype=torch.long,
        ).contiguous()
        output = torch.empty(
            (batch, cells, neighbors, dim),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        total = int(batch) * int(cells) * int(neighbors) * int(dim)
        block = 256
        grid = (triton.cdiv(total, block),)
        _local_field_gather_fwd[grid](
            contiguous_inputs,
            contiguous_patch,
            output,
            total,
            int(cells),
            int(neighbors),
            int(dim),
            BLOCK=block,
        )
        ctx.save_for_backward(contiguous_patch)
        ctx.shape = (int(batch), int(cells), int(dim), int(neighbors))
        return output

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (patch,) = ctx.saved_tensors
        batch, cells, dim, neighbors = ctx.shape
        grad = torch.zeros(
            (batch, cells, dim),
            device=grad_out.device,
            dtype=grad_out.dtype,
        )
        contiguous_grad = grad_out.contiguous()
        total = int(batch) * int(cells) * int(neighbors) * int(dim)
        block = 256
        grid = (triton.cdiv(total, block),)
        _local_field_gather_bwd[grid](
            contiguous_grad,
            patch,
            grad,
            total,
            int(cells),
            int(neighbors),
            int(dim),
            BLOCK=block,
        )
        return grad, None


class _LocalFieldGatherCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, patch: Tensor) -> Tensor:
        from .cpu import _load_ext

        extension = _load_ext(required=_CPU_NATIVE_REQUIRED)
        if extension is None:
            raise RuntimeError("native CPU local-field gather is unavailable")
        contiguous_inputs = inputs.float().contiguous()
        contiguous_patch = patch.to(device="cpu", dtype=torch.long).contiguous()
        ctx.save_for_backward(contiguous_patch)
        ctx.cells = int(inputs.shape[1])
        ctx.input_dtype = inputs.dtype
        return extension.local_field_gather_cpu(
            contiguous_inputs, contiguous_patch
        ).to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        from .cpu import _load_ext

        (patch,) = ctx.saved_tensors
        extension = _load_ext(required=True)
        grad = extension.local_field_gather_bwd_cpu(
            grad_out.float().contiguous(), patch, ctx.cells
        )
        return grad.to(ctx.input_dtype), None


def local_field_gather(inputs: Tensor, patch: Tensor) -> tuple[Tensor, str]:
    """Gather ``inputs[:, patch]`` with an explicit backend label."""

    global _GATHER_HITS, _GATHER_FALLBACKS, _CPU_GATHER_HITS
    if _HAS_TRITON and inputs.is_cuda:
        _GATHER_HITS += 1
        return (
            _LocalFieldGather.apply(inputs, patch),
            "triton_patch_gather_then_shared_core",
        )
    if _CPU_NATIVE_ENABLED and not inputs.is_cuda:
        _CPU_GATHER_HITS += 1
        return (
            _LocalFieldGatherCPU.apply(inputs, patch),
            "cpu_native_cpp_patch_gather_then_shared_core",
        )
    _GATHER_FALLBACKS += 1
    return inputs[:, patch], "torch_indexing_then_shared_core"
