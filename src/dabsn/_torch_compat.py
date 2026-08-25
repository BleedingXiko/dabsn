"""Compatibility shims for PyTorch APIs used by DABSN's registered operators."""

from __future__ import annotations

import collections.abc
from typing import Any

import torch
from torch import Tensor


def _cast_autocast_inputs(value: Any, device_type: str, dtype: torch.dtype) -> Any:
    """Mirror the input-casting policy used by ``torch.library.register_autocast``."""

    if isinstance(value, Tensor):
        eligible = (
            value.is_floating_point()
            and value.device.type == device_type
            and value.dtype is not torch.float64
        )
        return value.to(dtype) if eligible else value
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, collections.abc.Iterable):
        cast = (_cast_autocast_inputs(item, device_type, dtype) for item in value)
        if isinstance(value, (list, tuple)):
            return type(value)(cast)
        return cast
    return value


def register_custom_op_autocast(
    op: Any,
    device_type: str,
    cast_inputs: torch.dtype,
) -> Any:
    """Register a custom-op autocast rule on both old and new PyTorch.

    ``torch.library.register_autocast`` was added after PyTorch 2.6, while
    DABSN's ``cuda-turing`` extra deliberately pins PyTorch 2.6.  The older
    dispatcher already supports AutocastCPU/AutocastCUDA kernels, so reproduce
    the upstream rule there: cast eligible inputs, exclude both autocast keys,
    and redispatch to the custom operator's ordinary backend.

    DABSN passes ``CustomOpDef`` objects here.  Access to their dispatcher
    handles is intentionally isolated in this compatibility module.
    """

    if device_type not in {"cpu", "cuda"}:
        raise ValueError(f"device_type must be 'cpu' or 'cuda'; got {device_type!r}")

    native = getattr(torch.library, "register_autocast", None)
    if native is not None:
        return native(op, device_type, cast_inputs)

    library = getattr(op, "_lib", None)
    op_name = getattr(op, "_name", None)
    overload = getattr(op, "_opoverload", None)
    if library is None or not isinstance(op_name, str) or overload is None:
        raise TypeError(
            "PyTorch without torch.library.register_autocast requires a "
            "torch.library.custom_op CustomOpDef"
        )

    autocast_keys = torch._C.DispatchKeySet(
        torch._C.DispatchKey.AutocastCPU
    ) | torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCUDA)

    def kernel(_keyset, *args, **kwargs):
        if kwargs:
            raise AssertionError("custom-op autocast rules require positional arguments")
        with torch._C._ExcludeDispatchKeyGuard(autocast_keys):
            cast_args = _cast_autocast_inputs(args, device_type, cast_inputs)
            return overload(*cast_args)

    dispatch_key = "AutocastCUDA" if device_type == "cuda" else "AutocastCPU"
    library.impl(op_name, kernel, dispatch_key, with_keyset=True)
    return kernel


__all__ = ["register_custom_op_autocast"]
