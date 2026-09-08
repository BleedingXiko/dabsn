"""Direct Triton channel scans for online long-read credit.

The temporal loop is a runtime loop, so code size does not grow with the chunk.
Eligibility is FP32; recurrent state and input gradients retain storage dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _long_forward_kernel(W, P, Rates, L0, E0, R0, Out, LT, ET, RT, LF, EF, RF,
                         B: tl.constexpr, H: tl.constexpr, T, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < B * H
    base = (lane // H) * T * H + lane % H
    long = tl.load(L0 + lane, valid, 0).to(tl.float32)
    expected = tl.load(E0 + lane, valid, 0).to(tl.float32)
    retention = tl.load(R0 + lane, valid, 0).to(tl.float32)
    a = tl.load(Rates).to(tl.float32)
    b = tl.load(Rates + 1).to(tl.float32)
    c = tl.load(Rates + 2).to(tl.float32)
    s = tl.load(Rates + 3).to(tl.float32)
    for step in range(0, T):
        offset = base + step * H
        tl.store(LT + offset, long, valid)
        tl.store(ET + offset, expected, valid)
        tl.store(RT + offset, retention, valid)
        write = tl.load(W + offset, valid, 0).to(tl.float32)
        plasticity = tl.load(P + offset, valid, 0).to(tl.float32)
        error = 2.0 * tl.sigmoid(2.0 * tl.abs(write - expected)) - 1.0
        next_expected = b * expected + (1.0 - b) * write
        next_retention = c * retention + (1.0 - c) * (1.0 - error)
        effective = a + (1.0 - a) * s * next_retention
        long = effective * long + (1.0 - effective) * (plasticity * error * write)
        long = long.to(Out.dtype.element_ty).to(tl.float32)
        expected = next_expected.to(Out.dtype.element_ty).to(tl.float32)
        retention = next_retention.to(Out.dtype.element_ty).to(tl.float32)
        tl.store(Out + offset, long, valid)
    tl.store(LF + lane, long, valid)
    tl.store(EF + lane, expected, valid)
    tl.store(RF + lane, retention, valid)


@triton.jit
def _long_replay_kernel(W, P, LT, ET, RT, Rates, InputRates, G, Initial, Final,
                        Partial, GW, GP, B: tl.constexpr, H: tl.constexpr, T,
                        BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < B * H
    base = (lane // H) * T * H + lane % H
    parameter = tl.arange(0, 4)
    trace_offset = lane[:, None] * 12 + parameter[None, :]
    trace_long = tl.load(Initial + trace_offset, valid[:, None], 0).to(tl.float32)
    trace_expected = tl.load(Initial + trace_offset + 4, valid[:, None], 0).to(tl.float32)
    trace_retention = tl.load(Initial + trace_offset + 8, valid[:, None], 0).to(tl.float32)
    gradient = tl.zeros((BLOCK, 4), tl.float32)
    a = tl.load(Rates).to(tl.float32)
    b = tl.load(Rates + 1).to(tl.float32)
    c = tl.load(Rates + 2).to(tl.float32)
    s = tl.load(Rates + 3).to(tl.float32)
    for step in range(0, T):
        offset = base + step * H
        long = tl.load(LT + offset, valid, 0).to(tl.float32)
        expected = tl.load(ET + offset, valid, 0).to(tl.float32)
        retention = tl.load(RT + offset, valid, 0).to(tl.float32)
        write = tl.load(W + offset, valid, 0).to(tl.float32)
        plasticity = tl.load(P + offset, valid, 0).to(tl.float32)
        difference = write - expected
        error = 2.0 * tl.sigmoid(2.0 * tl.abs(difference)) - 1.0
        sign = tl.where(difference > 0, 1.0, tl.where(difference < 0, -1.0, 0.0))
        slope = (1.0 - error * error) * sign
        next_retention = c * retention + (1.0 - c) * (1.0 - error)
        effective = a + (1.0 - a) * s * next_retention
        candidate = plasticity * error * write
        gap = long - candidate
        retention_expected = (1.0 - c) * slope
        retention_direct = c * (1.0 - c) * (retention - (1.0 - error))
        j_long_expected = gap * (1.0 - a) * s * retention_expected - (1.0 - effective) * plasticity * write * slope
        j_long_retention = gap * (1.0 - a) * s * c
        direct_long_a = gap * (1.0 - s * next_retention) * a * (1.0 - a)
        direct_long_c = gap * (1.0 - a) * s * retention_direct
        direct_long_s = gap * (1.0 - a) * next_retention * s * (1.0 - s)
        direct_long = tl.where(parameter[None, :] == 0, direct_long_a[:, None], 0.0)
        direct_long += tl.where(parameter[None, :] == 2, direct_long_c[:, None], 0.0)
        direct_long += tl.where(parameter[None, :] == 3, direct_long_s[:, None], 0.0)
        next_trace_long = (effective[:, None] * trace_long + j_long_expected[:, None] * trace_expected
                           + j_long_retention[:, None] * trace_retention + direct_long)
        next_trace_expected = b * trace_expected + tl.where(
            parameter[None, :] == 1, (b * (1.0 - b) * (expected - write))[:, None], 0.0)
        next_trace_retention = (retention_expected[:, None] * trace_expected + c * trace_retention
                                + tl.where(parameter[None, :] == 2, retention_direct[:, None], 0.0))
        trace_long = next_trace_long
        trace_expected = next_trace_expected
        trace_retention = next_trace_retention
        gradient += tl.load(G + offset, valid, 0).to(tl.float32)[:, None] * trace_long
    tl.store(Final + trace_offset, trace_long, valid[:, None])
    tl.store(Final + trace_offset + 4, trace_expected, valid[:, None])
    tl.store(Final + trace_offset + 8, trace_retention, valid[:, None])
    tl.store(Partial + lane[:, None] * 4 + parameter[None, :], gradient, valid[:, None])

    # Historical input graphs are not retained. This reverse loop is confined
    # to the current chunk and computes its ordinary input credit exactly.
    a = tl.load(InputRates).to(tl.float32)
    b = tl.load(InputRates + 1).to(tl.float32)
    c = tl.load(InputRates + 2).to(tl.float32)
    s = tl.load(InputRates + 3).to(tl.float32)
    carry_long = tl.zeros((BLOCK,), tl.float32)
    carry_expected = tl.zeros((BLOCK,), tl.float32)
    carry_retention = tl.zeros((BLOCK,), tl.float32)
    for reverse in range(0, T):
        offset = base + (T - 1 - reverse) * H
        long = tl.load(LT + offset, valid, 0).to(tl.float32)
        expected = tl.load(ET + offset, valid, 0).to(tl.float32)
        retention = tl.load(RT + offset, valid, 0).to(tl.float32)
        write = tl.load(W + offset, valid, 0).to(tl.float32)
        plasticity = tl.load(P + offset, valid, 0).to(tl.float32)
        difference = write - expected
        error = 2.0 * tl.sigmoid(2.0 * tl.abs(difference)) - 1.0
        salience = plasticity * error
        next_retention = c * retention + (1.0 - c) * (1.0 - error)
        effective = a + (1.0 - a) * s * next_retention
        candidate = salience * write
        gradient_long = tl.load(G + offset, valid, 0).to(tl.float32) + carry_long
        gradient_effective = gradient_long * (long - candidate)
        gradient_candidate = gradient_long * (1.0 - effective)
        gradient_retention = carry_retention + gradient_effective * (1.0 - a) * s
        gradient_error = -gradient_retention * (1.0 - c) + gradient_candidate * write * plasticity
        sign = tl.where(difference > 0, 1.0, tl.where(difference < 0, -1.0, 0.0))
        gradient_difference = gradient_error * (1.0 - error * error) * sign
        gradient_write = gradient_candidate * salience + carry_expected * (1.0 - b) + gradient_difference
        gradient_plasticity = gradient_candidate * write * error
        tl.store(GW + offset, gradient_write, valid)
        tl.store(GP + offset, gradient_plasticity, valid)
        carry_long = (gradient_long * effective).to(W.dtype.element_ty).to(tl.float32)
        carry_expected = (carry_expected * b - gradient_difference).to(W.dtype.element_ty).to(tl.float32)
        carry_retention = (gradient_retention * c).to(W.dtype.element_ty).to(tl.float32)


def launch_long_forward(write, plasticity, rates, state, output, saved, final):
    batch, steps, hidden = write.shape
    _long_forward_kernel[(triton.cdiv(batch * hidden, 128),)](
        write, plasticity, rates, *state, output, *saved, *final,
        batch, hidden, steps, 128, enable_fp_fusion=False,
    )


def launch_long_replay(tape, rates, input_rates, gradient, initial, final, partials, write_gradient, plasticity_gradient):
    batch, steps, hidden = tape.write.shape
    _long_replay_kernel[(triton.cdiv(batch * hidden, 64),)](
        tape.write.contiguous(), tape.plasticity.contiguous(), tape.long.contiguous(),
        tape.expected.contiguous(), tape.retention.contiguous(), rates, input_rates,
        gradient, initial, final, partials, write_gradient, plasticity_gradient,
        batch, hidden, steps, 64, enable_fp_fusion=False,
    )
