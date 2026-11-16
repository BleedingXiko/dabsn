"""GPU architecture checks shared by the Triton kernels."""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def _capability(index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(index)


def bf16_kernels_supported(device: torch.device) -> bool:
    """Whether Triton kernels may take bf16 operands on this CUDA device.

    Pre-Ampere GPUs (Turing T4, sm_75) have no bf16 tensor cores, and Triton
    3.2 (the cuda-turing pin) cannot lower bf16 for them at all: the kernels
    fail to compile. Callers fall back to FP32 operands or another path.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    return _capability(index) >= (8, 0)
