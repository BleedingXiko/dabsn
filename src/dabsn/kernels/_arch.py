"""GPU architecture checks and device placement shared by the Triton kernels."""

from __future__ import annotations

import dataclasses
import functools
import itertools

import torch


def _cuda_device(value, depth: int = 0):
    """The device of the first CUDA tensor reachable from ``value``, or None."""
    if isinstance(value, torch.Tensor):
        return value.device if value.is_cuda else None
    if depth >= 2:
        return None
    if isinstance(value, (tuple, list)):
        items = value
    elif isinstance(value, dict):
        items = value.values()
    elif isinstance(value, torch.nn.Module):
        items = itertools.islice(value.parameters(), 1)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        items = (getattr(value, field.name) for field in dataclasses.fields(value))
    else:
        return None
    for item in items:
        device = _cuda_device(item, depth + 1)
        if device is not None:
            return device
    return None


def on_tensor_device(fn):
    """Run ``fn`` with its CUDA tensors' device as the current device.

    Triton launches on the *current* CUDA device and stream, not on the device
    of the pointers it is given. A model placed with ``model.to("cuda:1")``
    while the current device is still ``cuda:0`` would otherwise launch every
    kernel on the wrong GPU and fail ("Pointer argument cannot be accessed
    from Triton"). The guard is a no-op when the devices already agree, which
    is the case under ``torch.cuda.set_device`` and inside autograd's
    per-device backward threads.
    """

    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        if torch.compiler.is_compiling():
            return fn(*args, **kwargs)
        device = None
        for value in itertools.chain(args, kwargs.values()):
            device = _cuda_device(value)
            if device is not None:
                break
        if device is None or device.index is None or device.index == torch.cuda.current_device():
            return fn(*args, **kwargs)
        with torch.cuda.device(device):
            return fn(*args, **kwargs)

    return guarded


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
