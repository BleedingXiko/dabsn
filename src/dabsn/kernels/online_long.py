"""Bounded long-read online forward and replay without Dynamo compilation."""

from __future__ import annotations

import torch


def _parameters(read):
    return (read.logit_retain, read.logit_expect_retain,
            read.logit_retention_decay, read.logit_retention_strength)


def _validate_native(read, values, eligibility):
    supported = {torch.float16, torch.bfloat16, torch.float32}
    if any(value.dtype not in supported for value in (*values, *_parameters(read))):
        raise TypeError("native long-read requires FP16, BF16, or FP32 inputs, parameters, and state")
    if eligibility.traces.dtype != torch.float32:
        raise TypeError("native long-read replay requires FP32 eligibility")
    if any(value.device != values[0].device for value in (*values, *_parameters(read), eligibility.traces)):
        raise ValueError("native long-read inputs, parameters, state, and eligibility must share a device")
    from .online_compile import require_native_online_dtype
    for value in values:
        require_native_online_dtype(value.device, value.dtype)


def _rates(read, dtype=torch.float32):
    return torch.sigmoid(torch.stack([value.detach().float().reshape(())
                                      for value in _parameters(read)])).to(dtype)


@torch.no_grad()
def long_online_forward(read, write, plasticity, novelty, initial_state, eligibility_state):
    """Use native CUDA channel scans, or the canonical portable CPU path."""
    from dabsn.temporal_credit import LongReadOnlineTape, long_read_online_forward

    if write.device.type != "cuda":
        return long_read_online_forward(read, write, plasticity, novelty, initial_state, eligibility_state)
    if write.ndim != 3 or plasticity.shape != write.shape or novelty.shape != write.shape:
        raise ValueError("long-read online inputs must share shape [B,T,H]")
    batch, steps, hidden = write.shape
    if steps < 1 or len(initial_state) != 3 or any(value.shape != (batch, hidden) for value in initial_state):
        raise ValueError("long-read requires a nonempty chunk and three initial states shaped [B,H]")
    if eligibility_state.batch_size != batch or eligibility_state.hidden_dim != hidden:
        raise ValueError("long-read eligibility dimensions do not match inputs")
    _validate_native(read, (write, plasticity, novelty, *initial_state), eligibility_state)
    from .online_long_triton import launch_long_forward

    output = torch.empty_like(write, memory_format=torch.contiguous_format)
    saved = tuple(torch.empty_like(output) for _ in range(3))
    final = tuple(torch.empty((batch, hidden), dtype=write.dtype, device=write.device) for _ in range(3))
    launch_long_forward(write.contiguous(), plasticity.contiguous(), _rates(read, write.dtype),
                        tuple(value.to(write.dtype).contiguous() for value in initial_state), output, saved, final)
    tape = LongReadOnlineTape(write=write.detach(), plasticity=plasticity.detach(), novelty=novelty.detach(),
                              long=saved[0], expected=saved[1], retention=saved[2])
    return output, tape, final


@torch.no_grad()
def long_online_replay(read, tape, output_gradient, eligibility_state):
    """Scan the 3x4 trace forward and bounded input credit backward on CUDA."""
    from dabsn.temporal_credit import LongReadEligibilityState, long_read_online_replay_analytic

    if tape.write.device.type != "cuda":
        return long_read_online_replay_analytic(read, tape, output_gradient, eligibility_state)
    if output_gradient.shape != tape.write.shape or tape.steps < 1:
        raise ValueError("long-read output gradient must match a nonempty tape")
    if eligibility_state.batch_size != tape.batch_size or eligibility_state.hidden_dim != tape.hidden_dim:
        raise ValueError("long-read eligibility dimensions do not match tape")
    _validate_native(read, (tape.write, tape.plasticity, tape.novelty, tape.long, tape.expected,
                            tape.retention, output_gradient), eligibility_state)
    from .online_long_triton import launch_long_replay

    traces = torch.empty_like(eligibility_state.traces)
    partials = torch.empty((tape.batch_size * tape.hidden_dim, 4), dtype=torch.float32, device=tape.write.device)
    write_gradient = torch.empty_like(tape.write, memory_format=torch.contiguous_format)
    plasticity_gradient = torch.empty_like(write_gradient)
    launch_long_replay(tape, _rates(read), _rates(read, tape.write.dtype), output_gradient.contiguous(),
                       eligibility_state.traces, traces, partials, write_gradient, plasticity_gradient)
    reduced = partials.sum(dim=0)
    gradients = {parameter: reduced[index].to(parameter.dtype).reshape_as(parameter)
                 for index, parameter in enumerate(_parameters(read))}
    return LongReadEligibilityState(traces), gradients, (
        write_gradient, plasticity_gradient, torch.zeros_like(tape.novelty),
    )
