"""Bounded, shape-specialized full-graph compilation for online credit.

Dynamo attaches its guard cache to a Python code object, not to the result of
``torch.compile``. Separate wrappers around the same function still exhaust
that shared cache after eight ordinary shapes. Each execution signature here
owns a distinct code object; no process-wide Dynamo limits are modified.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, is_dataclass
from types import FunctionType

import torch


_COMPILED = OrderedDict()
_MAX_COMPILED_SIGNATURES = 64


def strict_inductor_backend(graph, example_inputs):
    """Do not let Inductor's SkipFrame escape into Dynamo's eager fallback."""
    from torch._dynamo.exc import SkipFrame

    try:
        return torch._inductor.compile(graph, example_inputs)
    except SkipFrame as error:
        raise RuntimeError(
            "DABSN requires compiled execution here; Inductor declined the graph "
            f"and eager fallback is forbidden: {error}"
        ) from error


def require_native_online_dtype(device: torch.device, dtype: torch.dtype) -> None:
    if device.type == "cuda" and dtype == torch.bfloat16:
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise RuntimeError(
                "Compiled online credit requires native BF16 hardware (CUDA "
                "compute capability 8.0 or newer). This GPU only emulates BF16; "
                "use FP32 or FP16. No eager replay fallback is permitted."
            )


def _signature(value, bf16_devices: set, tensor_ids: dict):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bfloat16:
            bf16_devices.add(value.device)
        return (
            # Dynamo also guards duplicate-input identities. Normalize aliases
            # by encounter order, never retain live tensors or their addresses.
            "tensor", tensor_ids.setdefault(id(value), len(tensor_ids)),
            tuple(value.shape), tuple(value.stride()), value.dtype,
            value.device, value.requires_grad,
        )
    if isinstance(value, torch.nn.Module):
        return (
            type(value), value.training,
            tuple((name, _signature(tensor, bf16_devices, tensor_ids)) for name, tensor in value.named_parameters()),
            tuple((name, _signature(tensor, bf16_devices, tensor_ids)) for name, tensor in value.named_buffers()),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple(
            (field.name, _signature(getattr(value, field.name), bf16_devices, tensor_ids))
            for field in fields(value)
        ))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_signature(item, bf16_devices, tensor_ids) for item in value))
    if isinstance(value, dict):
        return (dict, tuple((key, _signature(item, bf16_devices, tensor_ids)) for key, item in value.items()))
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device)):
        return (type(value), value)
    raise TypeError(f"Unsupported online compilation argument: {type(value).__name__}")


def call_online_compiled(function, *args):
    bf16_devices = set()
    signature = _signature(args, bf16_devices, {})
    key = (
        function, signature, torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
        torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"),
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        for device in bf16_devices:
            require_native_online_dtype(device, torch.bfloat16)
        isolated = FunctionType(
            function.__code__.replace(), function.__globals__, function.__name__,
            function.__defaults__, function.__closure__,
        )
        isolated.__kwdefaults__ = function.__kwdefaults__
        compiled = torch.compile(isolated, backend=strict_inductor_backend, fullgraph=True, dynamic=False)
        _COMPILED[key] = compiled
        if len(_COMPILED) > _MAX_COMPILED_SIGNATURES:
            _COMPILED.popitem(last=False)
    else:
        _COMPILED.move_to_end(key)
    return compiled(*args)
