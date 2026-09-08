"""Batched forward tapes for the online temporal-credit core boundary.

Online replay consumes FP32 intermediate values even with BF16 parameter and
activation storage. The ordinary batched training scan rounds its recurrent
GEMM inputs and saved tapes to the activation dtype; that policy cannot be
reused here. This path shares its pointwise recurrence but retains FP32 GEMMs
and tapes, without Dynamo compilation, autotuning, or CUDA graph capture.
FP32 matrix multiplication follows the caller's existing PyTorch precision
policy, including TF32 when enabled; this helper never changes global flags.
"""

from __future__ import annotations

import torch

from .batched_runtime import CoreScanOutputs, _forward_step


def validate_online_forward_dtypes(core, inputs, initial_state) -> None:
    """Reject unsupported precision before constructing a narrowed FP32 tape."""
    supported = {torch.float16, torch.bfloat16, torch.float32}
    if any(value.dtype not in supported for value in (inputs, *initial_state, *core.parameters())):
        raise TypeError(
            "native online forward requires FP16, BF16, or FP32 inputs, parameters, and state"
        )


@torch.no_grad()
def batched_online_forward_tapes(core, projected, gate_projected, initial_state) -> CoreScanOutputs:
    """Advance every stream together, preserving precise online replay values.

    Returns the same eleven tensors as the persistent precise-tape scan. Only
    scratch storage is promoted; the module's parameters are never converted
    in place. The caller retains responsibility for public output casting.
    """
    validate_online_forward_dtypes(core, projected, initial_state)
    if gate_projected.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("native online forward requires FP16, BF16, or FP32 gate projections")
    batch, steps, hidden = projected.shape
    if gate_projected.shape != projected.shape:
        raise ValueError("input and gate projections must have matching shapes")
    if len(initial_state) != 3 or any(value.shape != (batch, hidden) for value in initial_state):
        raise ValueError(f"initial_state tensors must have shape {(batch, hidden)}")
    if core.Ug.weight.shape != (hidden, hidden) or core.A.weight.shape != (hidden, hidden):
        raise ValueError("recurrent matrices must match the projected state width")

    # Autocast must not round y or the recurrent product: these are the values
    # differentiated by online replay. Input projections have already run with
    # the caller's activation policy before entering this boundary.
    with torch.autocast(device_type=projected.device.type, enabled=False):
        recurrent = torch.empty((2 * hidden, hidden), device=projected.device, dtype=torch.float32)
        recurrent[:hidden].copy_(core.Ug.weight)
        recurrent[hidden:].copy_(core.A.weight)
        params = tuple(
            value.float()
            for value in (
                core.beta,
                core.log_kappa,
                core.logit_recover,
                core.k_s,
                core.k_y,
                core.k_b,
                core.k_n,
                core.k_bias,
                core.r_s,
                core.r_y,
                core.r_b,
                core.r_n,
                core.r_bias,
                core.logit_saturation_decay,
                core.k_saturation,
                core.r_saturation,
                core.logit_alpha,
                core.log_lambda,
                core.logit_saturation_suppress,
            )
        )
        budget, energy, saturation = (
            value.to(device=projected.device, dtype=torch.float32) for value in initial_state
        )
        trajectory = torch.empty(
            (batch, steps, 2 * hidden), device=projected.device, dtype=torch.float32
        )
        tapes = tuple(
            torch.empty((batch, steps, hidden), device=projected.device, dtype=torch.float32)
            for _ in range(7)
        )
        if projected.device.type == "cuda":
            from .online_forward_triton import scan_batched_online_cuda

            final = scan_batched_online_cuda(
                projected.contiguous(),
                gate_projected.contiguous(),
                recurrent,
                params,
                (budget, energy, saturation),
                (trajectory, *tapes),
            )
            return (trajectory, *tapes, *final)
        for step in range(steps):
            (
                y,
                budget,
                next_energy,
                saturation,
                novelty,
                plasticity,
                expression,
                write,
                previous_energy,
                gate,
            ) = _forward_step(
                projected[:, step],
                gate_projected[:, step],
                recurrent,
                budget,
                energy,
                saturation,
                *params,
            )
            trajectory[:, step, :hidden] = y
            trajectory[:, step, hidden:] = budget
            for destination, value in zip(
                tapes, (novelty, plasticity, expression, write, previous_energy, saturation, gate)
            ):
                destination[:, step] = value
            energy = next_energy
    return (trajectory, *tapes, budget, energy, saturation)
