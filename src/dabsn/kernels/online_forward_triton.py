"""Direct pointwise kernels for the precise batched online forward scan."""

import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(value):
    return 2.0 * tl.sigmoid(2.0 * value) - 1.0


@triton.jit
def _softplus(value):
    return tl.where(value > 20.0, value, tl.log(1.0 + tl.exp(value)))


@triton.jit
def _prepare_y(
    Projected,
    Budget,
    Y,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    step,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < B * H
    row, h = index // H, index % H
    wx = tl.load(Projected + (row * T + step) * H + h, mask, 0).to(tl.float32)
    budget = tl.load(Budget + index, mask, 0)
    tl.store(Y + index, _tanh(wx + budget), mask)


@triton.jit
def _finish_step(
    GateProjected,
    Y,
    Recurrent,
    Params,
    Budget,
    Energy,
    Saturation,
    Trajectory,
    Novelty,
    Plasticity,
    Expression,
    Write,
    EnergyTape,
    SaturationTape,
    GateTape,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    step,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < B * H
    row, h = index // H, index % H
    slot = (row * T + step) * H + h
    y = tl.load(Y + index, mask, 0)
    budget = tl.load(Budget + index, mask, 0)
    energy = tl.load(Energy + index, mask, 0)
    saturation = tl.load(Saturation + index, mask, 0)
    recurrence = tl.load(Recurrent + row * (2 * H) + h, mask, 0)
    expression = tl.load(Recurrent + row * (2 * H) + H + h, mask, 0)
    gate_projection = tl.load(GateProjected + slot, mask, 0).to(tl.float32)
    gate = tl.minimum(tl.maximum((gate_projection + recurrence + 3.0) / 6.0, 0.0), 1.0)
    novelty = _tanh(tl.abs(expression - budget))
    decay = tl.sigmoid(tl.load(Params + 13 * H + h, mask, 0))
    next_saturation = decay * saturation + (1.0 - decay) * (novelty * (1.0 - energy))
    suppress = tl.sigmoid(tl.load(Params + 18 * H + h, mask, 0))
    effective = novelty * (1.0 - suppress * next_saturation)
    tanh_budget = _tanh(budget)
    k_signal = (
        tl.load(Params + 3 * H + h, mask, 0) * gate
        + tl.load(Params + 4 * H + h, mask, 0) * y
        + tl.load(Params + 5 * H + h, mask, 0) * tanh_budget
        + tl.load(Params + 6 * H + h, mask, 0) * effective
        + tl.load(Params + 7 * H + h, mask, 0)
        + tl.load(Params + 14 * H + h, mask, 0) * next_saturation
    )
    r_signal = (
        tl.load(Params + 8 * H + h, mask, 0) * gate
        + tl.load(Params + 9 * H + h, mask, 0) * y
        + tl.load(Params + 10 * H + h, mask, 0) * tanh_budget
        + tl.load(Params + 11 * H + h, mask, 0) * effective
        + tl.load(Params + 12 * H + h, mask, 0)
        + tl.load(Params + 15 * H + h, mask, 0) * next_saturation
    )
    write_cost = _softplus(tl.load(Params + H + h, mask, 0)) * tl.exp(0.5 * _tanh(k_signal))
    recovery = tl.minimum(
        tl.maximum(
            tl.sigmoid(tl.load(Params + 2 * H + h, mask, 0)) * tl.exp(0.5 * _tanh(r_signal)), 0.0
        ),
        1.0,
    )
    plasticity = gate * energy
    write = plasticity * expression
    next_budget = (
        (1.0 - tl.sigmoid(tl.load(Params + 16 * H + h, mask, 0))) * budget
        + tl.load(Params + h, mask, 0)
        + _softplus(tl.load(Params + 17 * H + h, mask, 0)) * write
    )
    next_energy = tl.minimum(
        tl.maximum(energy + recovery * (1.0 - energy) - write_cost * plasticity, 0.0), 1.0
    )
    tl.store(Trajectory + (row * T + step) * (2 * H) + h, y, mask)
    tl.store(Trajectory + (row * T + step) * (2 * H) + H + h, next_budget, mask)
    tl.store(Novelty + slot, novelty, mask)
    tl.store(Plasticity + slot, plasticity, mask)
    tl.store(Expression + slot, expression, mask)
    tl.store(Write + slot, write, mask)
    tl.store(EnergyTape + slot, energy, mask)
    tl.store(SaturationTape + slot, next_saturation, mask)
    tl.store(GateTape + slot, gate, mask)
    tl.store(Budget + index, next_budget, mask)
    tl.store(Energy + index, next_energy, mask)
    tl.store(Saturation + index, next_saturation, mask)


def scan_batched_online_cuda(projected, gate_projected, recurrent, params, state, outputs):
    """Two pointwise launches and one batched FP32 GEMM per recurrent step."""
    batch, steps, hidden = projected.shape
    budget, energy, saturation = (value.clone().contiguous() for value in state)
    packed = torch.stack(tuple(value.expand(hidden) for value in params)).contiguous()
    y = torch.empty((batch, hidden), device=projected.device, dtype=torch.float32)
    recurrence = torch.empty((batch, 2 * hidden), device=projected.device, dtype=torch.float32)
    grid = (triton.cdiv(batch * hidden, 256),)
    for step in range(steps):
        _prepare_y[grid](projected, budget, y, batch, steps, hidden, step, 256)
        torch.mm(y, recurrent.T, out=recurrence)
        _finish_step[grid](
            gate_projected,
            y,
            recurrence,
            packed,
            budget,
            energy,
            saturation,
            *outputs,
            batch,
            steps,
            hidden,
            step,
            256,
            enable_fp_fusion=False,
        )
    return budget, energy, saturation
