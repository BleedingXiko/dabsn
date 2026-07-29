"""Tensor-core batched training runtime for the DABSN recurrent core.

The persistent Triton scan is ideal for latency-oriented, small-batch work, but
its one-program-per-sequence layout turns the two recurrent matrices into GEMV
traffic.  Language-model pretraining has a large device batch and needs GEMM:
this custom autograd function advances every sequence in the batch together at
each recurrent step and computes the reverse scan the same way.  Only the
pointwise step is compiled; a complete DABSN block or stack is never wrapped in
``torch.compile``.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import Tensor

from ..read import _stream_is_capturing


def _forward_step(
    wx: Tensor,
    wgx: Tensor,
    recurrent: Tensor,
    budget: Tensor,
    energy: Tensor,
    saturation: Tensor,
    beta: Tensor,
    log_kappa: Tensor,
    logit_recover: Tensor,
    k_s: Tensor,
    k_y: Tensor,
    k_b: Tensor,
    k_n: Tensor,
    k_bias: Tensor,
    r_s: Tensor,
    r_y: Tensor,
    r_b: Tensor,
    r_n: Tensor,
    r_bias: Tensor,
    saturation_decay: Tensor,
    k_saturation: Tensor,
    r_saturation: Tensor,
    logit_alpha: Tensor,
    log_lambda: Tensor,
    logit_saturation_suppress: Tensor,
    y_full: Tensor | None = None,
):
    """One recurrence step over the complete device batch.

    ``y_full`` exists for tensor parallelism. When the hidden dimension is
    sharded across ranks, this rank owns ``H/P`` units of state and the matching
    rows of the recurrent matrix -- but the recurrent product needs every unit
    of ``y``, not just the local ones. The caller all-gathers ``y`` and passes
    it here, so the sharded and unsharded paths remain the same function and the
    recurrence has exactly one definition. Left ``None`` (every single-device
    path) the locally computed ``y`` is used and nothing changes.
    """

    # The recurrence runs in the STATE's dtype, which the caller chooses -- not a
    # hardcoded FP32. Per step this function does one GEMM and ~30 elementwise
    # ops on [B,H], so its cost is memory traffic on those tensors, not FLOPs;
    # running them at the activation dtype halves that traffic. Measured drift
    # of a BF16 state chain against an FP32 one is 0.16% at T=512 and does not
    # compound (0.13% at T=32), against a parity bar of rtol 2e-2.
    #
    # Parameters stay FP32 masters for AdamW; they are [H] and cast per step at
    # negligible cost. Passing an FP32 state gives byte-for-byte the previous
    # behaviour, so this is a caller policy, not a change of arithmetic here.
    dt = budget.dtype
    y = torch.tanh(wx.to(dt) + budget)
    y_in = y if y_full is None else y_full
    recurrent_out = F.linear(y_in.to(recurrent.dtype), recurrent).to(dt)
    gate_recurrence, expression = recurrent_out.chunk(2, dim=-1)
    gate = F.hardsigmoid(wgx.to(dt) + gate_recurrence)
    novelty = torch.tanh((expression - budget).abs())

    decay = torch.sigmoid(saturation_decay.to(dt))
    stress = novelty * (1.0 - energy)
    saturation = decay * saturation + (1.0 - decay) * stress
    suppress = torch.sigmoid(logit_saturation_suppress.to(dt))
    novelty_effective = novelty * (1.0 - suppress * saturation)

    tanh_budget = torch.tanh(budget)
    k_signal = (
        k_s.to(dt) * gate
        + k_y.to(dt) * y
        + k_b.to(dt) * tanh_budget
        + k_n.to(dt) * novelty_effective
        + k_bias.to(dt)
        + k_saturation.to(dt) * saturation
    )
    r_signal = (
        r_s.to(dt) * gate
        + r_y.to(dt) * y
        + r_b.to(dt) * tanh_budget
        + r_n.to(dt) * novelty_effective
        + r_bias.to(dt)
        + r_saturation.to(dt) * saturation
    )
    write_cost = F.softplus(log_kappa.to(dt)) * torch.exp(0.5 * torch.tanh(k_signal))
    recovery = torch.clamp(
        torch.sigmoid(logit_recover.to(dt)) * torch.exp(0.5 * torch.tanh(r_signal)),
        0.0,
        1.0,
    )
    plasticity = gate * energy
    write = plasticity * expression
    next_budget = (
        (1.0 - torch.sigmoid(logit_alpha.to(dt))) * budget
        + beta.to(dt)
        + F.softplus(log_lambda.to(dt)) * write
    )
    next_energy = torch.clamp(
        energy + recovery * (1.0 - energy) - write_cost * plasticity,
        0.0,
        1.0,
    )
    return (
        y,
        next_budget,
        next_energy,
        saturation,
        novelty,
        plasticity,
        expression,
        write,
        energy,
        gate,
    )


# One compiled artifact per (step, grad-mode, dtype). Dynamo keys its cache on
# tensor properties, and `requires_grad` is one of them: sharing a single
# artifact between the grad-off forward and the grad-on forward made every
# transition between them burn a cache slot, so a run that legitimately
# alternates (a capacity probe, an eval inside training) hit the recompile limit
# on grad-mode churn rather than on real shape variety. Keyed properly, each
# mode keeps its own artifact and neither evicts the other.
def _state_dtype(activation_dtype: torch.dtype) -> torch.dtype:
    """The recurrent state chain is FP32. It is not a free choice.

    A BF16 state would halve the memory traffic of the ~30 elementwise ops per
    step, which is where this step's time actually goes -- worth roughly 2x. The
    forward tolerates it: 0.16% max relative drift at T=512, and it does not
    compound (0.13% at T=32).

    The backward does not. With a BF16 chain,
    `test_batched_bf16_scratch_matches_fp32_eager` fails -- one gradient element
    in ninety moves 10.3% against a 5% bar. The reverse scan differentiates
    through those same 30 elementwise ops per step and amplifies what the
    forward absorbs, so a forward-only drift measurement does not predict it.

    Hence FP32, with no flag: a switch whose correct setting depends on a
    measurement nobody has taken is not a choice worth offering. Claiming the 2x
    requires measuring the BACKWARD at a real width on real hardware first.
    """
    return torch.float32


_COMPILED_STEPS: dict[tuple, object] = {}
_RECOMPILE_BUDGET_RAISED = False


def _raise_recompile_budget() -> None:
    """Give the step room for the shapes a real sweep visits.

    Dynamo's default recompile limit is 8. This function is compiled per
    execution shape by design (`dynamic=False` is what makes the generated
    kernel worth having), so 8 is exhausted by an ordinary batch sweep. The
    limit exists to catch pathological recompilation in user code; here the
    recompilation is intended and bounded by how many shapes the run actually
    visits.
    """
    global _RECOMPILE_BUDGET_RAISED
    if _RECOMPILE_BUDGET_RAISED:
        return
    _RECOMPILE_BUDGET_RAISED = True
    budget = int(os.environ.get("DABSN_STEP_RECOMPILE_LIMIT", "128"))
    try:
        from torch._dynamo import config as _dynamo_config
        for field in ("recompile_limit", "cache_size_limit"):
            if hasattr(_dynamo_config, field):
                setattr(_dynamo_config, field, max(getattr(_dynamo_config, field), budget))
        for field in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
            if hasattr(_dynamo_config, field):
                setattr(_dynamo_config, field, max(getattr(_dynamo_config, field), budget * 8))
    except Exception:
        pass


def _step_compile_enabled(tensor: Tensor) -> bool:
    return tensor.is_cuda and os.environ.get("DABSN_BATCHED_STEP_COMPILE", "1") == "1"


def _compiled_step(fn, which: str, sample: Tensor):
    key = (which, torch.is_grad_enabled(), sample.dtype, sample.device.type)
    compiled = _COMPILED_STEPS.get(key)
    if compiled is None:
        _raise_recompile_budget()
        compiled = torch.compile(fn, dynamic=False, fullgraph=True)
        _COMPILED_STEPS[key] = compiled
    return compiled


def _step_compile_failed(exc: BaseException, which: str) -> RuntimeError:
    """Compilation failure is a hard failure, not a quiet downgrade.

    Falling back to the interpreter here used to keep the run alive at roughly
    a hundredth of the throughput, and said so in a warning nobody reads mid-run.
    That is the worst outcome: the run still costs money and hours, and the
    number it produces is not the number the framework can do. Every path that
    matters is compiled or captured, so if compilation breaks, the run stops and
    says exactly how to proceed.
    """
    return RuntimeError(
        f"torch.compile of the {which} step failed: {type(exc).__name__}: {exc}\n"
        "This step is never run through the interpreter, because that costs "
        "roughly 100x and would silently make every timing meaningless.\n"
        "Fixes, in order:\n"
        "  1. If this is a recompile-limit error, the run is visiting more "
        "shapes than the budget allows. Raise DABSN_STEP_RECOMPILE_LIMIT "
        "(default 128), or hold the shape fixed across the sweep.\n"
        "  2. If Inductor itself failed to compile the step, set "
        "DABSN_BATCHED_STEP_COMPILE=0 to run the step through the interpreter "
        "ON PURPOSE. The math is identical; the throughput is not, so any "
        "number produced that way must not be reported as a kernel result.\n"
        "  3. Chunk-level CUDA graph capture (DABSN_SCAN_GRAPH=1, the default) "
        "removes the per-step dispatch cost independently of Inductor and is "
        "the larger win of the two."
    )


def _run_forward_step(*args):
    if not _step_compile_enabled(args[0]):
        return _forward_step(*args)
    try:
        return _compiled_step(_forward_step, "forward", args[0])(*args)
    except Exception as exc:  # noqa: BLE001
        raise _step_compile_failed(exc, "forward") from exc


def _backward_step(
    y: Tensor,
    b_prev: Tensor,
    e_prev: Tensor,
    c_prev: Tensor,
    gate: Tensor,
    novelty: Tensor,
    plasticity: Tensor,
    expression: Tensor,
    c_t: Tensor,
    gy0: Tensor,
    gb_new: Tensor,
    ge_new: Tensor,
    gc_new: Tensor,
    gnov0: Tensor,
    gp0: Tensor,
    gay0: Tensor,
    gwrite: Tensor,
    ge_out: Tensor,
    recurrent_backward: Tensor,
    log_kappa: Tensor,
    logit_recover: Tensor,
    k_s: Tensor,
    k_y: Tensor,
    k_b: Tensor,
    k_n: Tensor,
    k_bias: Tensor,
    r_s: Tensor,
    r_y: Tensor,
    r_b: Tensor,
    r_n: Tensor,
    r_bias: Tensor,
    decay: Tensor,
    k_c: Tensor,
    r_c: Tensor,
    alpha: Tensor,
    lam: Tensor,
    sig_lam: Tensor,
    suppress: Tensor,
    acc_beta: Tensor,
    acc_log_kappa: Tensor,
    acc_logit_recover: Tensor,
    acc_k_s: Tensor,
    acc_k_y: Tensor,
    acc_k_b: Tensor,
    acc_k_n: Tensor,
    acc_k_bias: Tensor,
    acc_r_s: Tensor,
    acc_r_y: Tensor,
    acc_r_b: Tensor,
    acc_r_n: Tensor,
    acc_r_bias: Tensor,
    acc_decay: Tensor,
    acc_k_c: Tensor,
    acc_r_c: Tensor,
    acc_alpha: Tensor,
    acc_lambda: Tensor,
    acc_suppress: Tensor,
):
    """One exact reverse-recurrence step over the complete device batch."""

    kappa = F.softplus(log_kappa)
    recover = torch.sigmoid(logit_recover)
    tanh_b = torch.tanh(b_prev)
    novelty_effective = novelty * (1.0 - suppress * c_t)
    k_signal = (
        k_s * gate + k_y * y + k_b * tanh_b
        + k_n * novelty_effective + k_bias + k_c * c_t
    )
    r_signal = (
        r_s * gate + r_y * y + r_b * tanh_b
        + r_n * novelty_effective + r_bias + r_c * c_t
    )
    tanh_k, tanh_r = torch.tanh(k_signal), torch.tanh(r_signal)
    exp_k, exp_r = torch.exp(0.5 * tanh_k), torch.exp(0.5 * tanh_r)
    write_cost = kappa * exp_k
    recovery_pre = recover * exp_r
    recovery = torch.clamp(recovery_pre, 0.0, 1.0)
    recovery_clamp_grad = ((recovery_pre >= 0.0) & (recovery_pre <= 1.0)).float()
    energy_pre = e_prev + recovery * (1.0 - e_prev) - write_cost * plasticity
    ge_pre = ge_new * ((energy_pre >= 0.0) & (energy_pre <= 1.0)).float()
    g_bprev_state = gb_new * (1.0 - alpha)
    g_eprev_state = ge_pre * (1.0 - recovery)
    gp = gp0 + gb_new * lam * expression - ge_pre * write_cost
    gay = gay0 + gb_new * lam * plasticity
    g_write_cost = -ge_pre * plasticity
    g_recovery = ge_pre * (1.0 - e_prev)

    d_expression_main = gay + gwrite * plasticity
    gp = gp + gwrite * expression
    g_k_signal = g_write_cost * write_cost * 0.5 * (1.0 - tanh_k.square())
    g_recovery_pre = g_recovery * recovery_clamp_grad
    g_r_signal = g_recovery_pre * recovery_pre * 0.5 * (1.0 - tanh_r.square())
    g_gate_energy = g_k_signal * k_s + g_r_signal * r_s
    gy_energy = g_k_signal * k_y + g_r_signal * r_y
    gb_energy = (g_k_signal * k_b + g_r_signal * r_b) * (1.0 - tanh_b.square())
    gnov_effective = g_k_signal * k_n + g_r_signal * r_n
    gc_energy = g_k_signal * k_c + g_r_signal * r_c

    dgate_pre = (
        (gp * e_prev + g_gate_energy)
        * (((gate > 0.0) & (gate < 1.0)).float() / 6.0)
    )
    ge_gate = gp * gate

    stress = novelty * (1.0 - e_prev)
    c_new = decay * c_prev + (1.0 - decay) * stress
    g_c_new = gc_new + gc_energy - gnov_effective * novelty * suppress
    g_stress = g_c_new * (1.0 - decay)
    g_novelty = (
        gnov_effective * (1.0 - suppress * c_new)
        + g_stress * (1.0 - e_prev)
    )
    g_e_prev_cort = -g_stress * novelty
    g_c_prev = g_c_new * decay
    diff = expression - b_prev
    novelty_check = torch.tanh(diff.abs())
    d_novelty = (gnov0 + g_novelty) * (1.0 - novelty_check.square()) * torch.sign(diff)
    d_expression = d_expression_main + d_novelty

    recurrent_adjoint = (
        torch.cat((dgate_pre, d_expression), dim=-1).to(recurrent_backward.dtype)
        @ recurrent_backward
    ).float()
    gy = gy0 + gy_energy + recurrent_adjoint
    gpre = gy * (1.0 - y.square())
    next_gb = g_bprev_state + gb_energy - d_novelty + gpre
    next_ge = g_eprev_state + ge_gate + g_e_prev_cort + ge_out
    next_gc = g_c_prev

    parameter_parts = (
        gb_new.sum(dim=0),
        (g_write_cost * exp_k * torch.sigmoid(log_kappa)).sum(dim=0),
        (g_recovery_pre * exp_r * recover * (1.0 - recover)).sum(dim=0),
        (g_k_signal * gate).sum(dim=0),
        (g_k_signal * y).sum(dim=0),
        (g_k_signal * tanh_b).sum(dim=0),
        (g_k_signal * novelty_effective).sum(dim=0),
        g_k_signal.sum(dim=0),
        (g_r_signal * gate).sum(dim=0),
        (g_r_signal * y).sum(dim=0),
        (g_r_signal * tanh_b).sum(dim=0),
        (g_r_signal * novelty_effective).sum(dim=0),
        g_r_signal.sum(dim=0),
        (g_c_new * (c_prev - stress) * decay * (1.0 - decay)).sum(dim=0),
        (g_k_signal * c_t).sum(dim=0),
        (g_r_signal * c_t).sum(dim=0),
    )
    scalar_parts = (
        (gb_new * (-b_prev) * alpha * (1.0 - alpha)).sum(),
        (gb_new * (plasticity * expression) * sig_lam).sum(),
        (gnov_effective * novelty * (-c_new) * suppress * (1.0 - suppress)).sum(),
    )
    return (
        gpre,
        dgate_pre,
        d_expression,
        next_gb,
        next_ge,
        next_gc,
        acc_beta + parameter_parts[0],
        acc_log_kappa + parameter_parts[1],
        acc_logit_recover + parameter_parts[2],
        acc_k_s + parameter_parts[3],
        acc_k_y + parameter_parts[4],
        acc_k_b + parameter_parts[5],
        acc_k_n + parameter_parts[6],
        acc_k_bias + parameter_parts[7],
        acc_r_s + parameter_parts[8],
        acc_r_y + parameter_parts[9],
        acc_r_b + parameter_parts[10],
        acc_r_n + parameter_parts[11],
        acc_r_bias + parameter_parts[12],
        acc_decay + parameter_parts[13],
        acc_k_c + parameter_parts[14],
        acc_r_c + parameter_parts[15],
        acc_alpha + scalar_parts[0],
        acc_lambda + scalar_parts[1],
        acc_suppress + scalar_parts[2],
    )


def _run_backward_step(*args):
    if not _step_compile_enabled(args[0]):
        return _backward_step(*args)
    try:
        return _compiled_step(_backward_step, "backward", args[0])(*args)
    except Exception as exc:  # noqa: BLE001
        raise _step_compile_failed(exc, "backward") from exc


class _ReverseCarry:
    """Mutable carry for the reverse scan: state grads plus grad accumulators.

    The reverse loop threads three [B,H] state gradients and 19 parameter
    accumulators from step to step. Cloning on entry keeps the caller's tensors
    untouched (the same aliasing trap `_ScanState` documents), and holding them
    in fixed storage is what lets a captured chunk pick up where the last replay
    left off.
    """

    __slots__ = ("gb", "ge", "gc", "param_acc", "scalar_acc")

    def __init__(self, gb: Tensor, ge: Tensor, gc: Tensor,
                 param_acc: list, scalar_acc: list):
        self.gb = gb.clone()
        self.ge = ge.clone()
        self.gc = gc.clone()
        self.param_acc = [t.clone() for t in param_acc]
        self.scalar_acc = [t.clone() for t in scalar_acc]

    def tensors(self) -> list:
        return [self.gb, self.ge, self.gc, *self.param_acc, *self.scalar_acc]


def _backward_chunk(
    reads: tuple, grads: tuple, outs: tuple, prev: tuple,
    carry: _ReverseCarry, recurrent: Tensor, consts: tuple,
    H: int, count: int, offset: int = 0,
) -> None:
    """Run ``count`` reverse steps over locally-indexed slices.

    ``prev`` supplies the previous-step budget and saturation already aligned to
    the local index, so this body has no boundary branch: the driver decides what
    step 0's predecessor is (the initial state, or the tape row before the
    chunk). Carry and accumulators are read from and written back through
    ``carry``'s tensors, so consecutive chunks chain through fixed storage --
    which is what a CUDA graph needs in order to replay this body.

    Single definition of the reverse scan math: the eager path and the captured
    path both run it.
    """
    U, e_tape, c_tape, s_tape, plasticity, expression = reads
    gU, gnov, gp_out, gay_out, gwrite, ge_out, gc_out = grads
    gWx, gWgx, day_tape = outs
    b_prev_src, c_prev_src = prev
    gbn, gen, gcn = carry.gb, carry.ge, carry.gc
    param_acc = list(carry.param_acc)
    scalar_acc = list(carry.scalar_acc)

    for step in range(count - 1, -1, -1):
        slot = offset + step
        y = U[:, slot, :H].float()
        b_prev = b_prev_src[:, step].float()
        e_prev = e_tape[:, slot].float()
        c_prev = c_prev_src[:, step].float()
        gate = s_tape[:, slot].float()
        p = plasticity[:, slot].float()
        ay = expression[:, slot].float()
        # Recompute novelty from the saved expression + prev-budget instead of
        # reading a saved novelty tape (Phase 5): novelty is a pure function
        # tanh(|expression - budget|), so this is bit-identical in FP32 (release
        # gate stays 0.0) and within tolerance in bf16, and it lets novelty drop
        # out of the backward context entirely.
        nov = torch.tanh((ay - b_prev).abs())
        c_t = c_tape[:, slot].float()

        gb_new = gU[:, slot, H:] + gbn
        gc_new = gcn + gc_out[:, slot]
        step_outputs = _run_backward_step(
            y, b_prev, e_prev, c_prev, gate, nov, p, ay, c_t,
            gU[:, slot, :H], gb_new, gen, gc_new,
            gnov[:, slot], gp_out[:, slot], gay_out[:, slot],
            gwrite[:, slot], ge_out[:, slot],
            recurrent, *consts,
            *param_acc, *scalar_acc,
        )
        gpre, dpre, d_ay, gbn, gen, gcn = step_outputs[:6]
        gWx[:, slot] = gpre
        gWgx[:, slot] = dpre
        day_tape[:, slot] = d_ay
        param_acc = list(step_outputs[6:22])
        scalar_acc = list(step_outputs[22:25])

    carry.gb.copy_(gbn)
    carry.ge.copy_(gen)
    carry.gc.copy_(gcn)
    for dst, src in zip(carry.param_acc, param_acc):
        dst.copy_(src)
    for dst, src in zip(carry.scalar_acc, scalar_acc):
        dst.copy_(src)


def _prev_slices(U, c_tape, initial_b, initial_c, H, start, count, out=None):
    """Previous-step budget/saturation for local steps ``0..count``.

    Step ``t`` reads the state as it stood at ``t-1``; for ``t == 0`` that is the
    scan's initial state. Materializing it here keeps the boundary case out of
    the loop body, which is what allows one body to serve both the eager and the
    captured path.
    """
    budget = U[:, :, H:]
    if start >= 1:
        b_src = budget[:, start - 1:start + count - 1]
        c_src = c_tape[:, start - 1:start + count - 1]
        if out is None:
            return b_src, c_src
        out[0].copy_(b_src)
        out[1].copy_(c_src)
        return out
    b_dst, c_dst = out if out is not None else (
        torch.empty((U.shape[0], count, H), device=U.device, dtype=U.dtype),
        torch.empty((U.shape[0], count, H), device=U.device, dtype=c_tape.dtype),
    )
    b_dst[:, 0].copy_(initial_b)
    c_dst[:, 0].copy_(initial_c)
    if count > 1:
        b_dst[:, 1:].copy_(budget[:, :count - 1])
        c_dst[:, 1:].copy_(c_tape[:, :count - 1])
    return b_dst, c_dst


_REVERSE_GRAPHS: dict[tuple, dict] = {}


def _reverse_chunk_width(B: int, T: int, H: int, elem: int, device=None) -> int:
    """Steps per capture for the reverse scan.

    The reverse stages more per step than the forward -- eight activation-dtype
    reads, eight FP32 output grads, three activation-dtype writes -- so its width
    is derived from that mix rather than reusing the forward's number.
    """
    per_step = B * H * (8 * elem + 8 * 4 + 3 * elem)
    return int(max(1, min(_scan_stage_bytes(device) // max(1, per_step), T)))


def _reverse_capture_width(B: int, T: int, H: int, elem: int, device=None) -> int:
    """Adaptive reverse width that always leaves the initial-state boundary eager.

    ``_reverse_chunk_width`` is a memory-capacity calculation and may legitimately
    return the complete sequence.  Reverse replay cannot include position zero:
    that position reads the caller-owned initial state rather than a predecessor
    in the staged tape.  Derive the capturable tail from the live sequence
    length instead of rejecting capture whenever the capacity happens to cover
    all ``T`` positions.
    """
    if T <= 1:
        return 0
    return min(_reverse_chunk_width(B, T, H, elem, device), T - 1)


def _reverse_into_grads(*, reads, grads, outs, initial, carry, recurrent,
                        consts, B, T, H, dtype, device) -> None:
    """Run the whole reverse scan, replaying a captured chunk when possible.

    Mirrors ``_scan_into_tapes``: the reverse loop issues one launch per timestep
    per layer and is dominated by the cost of asking for work rather than the
    work itself. Chunks are replayed from the top down; the final chunk -- the
    one containing step 0, whose predecessor is the scan's initial state -- runs
    uncaptured, as does everything when capture cannot help or be trusted.
    """
    U, _e, c_tape, _s, _p, _x = reads
    initial_b, initial_c = initial

    def run_eager() -> None:
        prev = _prev_slices(U, c_tape, initial_b, initial_c, H, 0, T)
        _backward_chunk(reads, grads, outs, prev, carry, recurrent, consts,
                        H, T, 0)

    if (
        not U.is_cuda
        or T < _SCAN_GRAPH_MIN_STEPS
        or torch.is_grad_enabled()
        or _stream_is_capturing()
        or os.environ.get("DABSN_SCAN_GRAPH", "1") != "1"
    ):
        run_eager()
        return

    chunk = _reverse_capture_width(B, T, H, U.element_size(), device)
    # Step 0's chunk is handled eagerly, so a capture needs a full chunk above it.
    if chunk < _SCAN_GRAPH_MIN_STEPS:
        run_eager()
        return

    try:
        entry = _reverse_graph_entry(B, chunk, H, U, c_tape, recurrent,
                                     consts, carry, reads, grads, outs, dtype, device)
    except Exception as exc:  # noqa: BLE001 - identical math runs below
        _disable_scan_graph(exc)
        run_eager()
        return

    entry["recurrent"].copy_(recurrent)
    for dst, src in zip(entry["consts"], consts):
        dst.copy_(src)
    for dst, src in zip(entry["carry"], carry.tensors()):
        dst.copy_(src)

    start = T - chunk
    while start >= 1:
        for dst, src in zip(entry["reads"], reads):
            dst.copy_(src[:, start:start + chunk])
        for dst, src in zip(entry["grads"], grads):
            dst.copy_(src[:, start:start + chunk])
        _prev_slices(U, c_tape, initial_b, initial_c, H, start, chunk,
                     out=entry["prev"])
        entry["graph"].replay()
        for real, staged in zip(outs, entry["outs"]):
            real[:, start:start + chunk].copy_(staged)
        start -= chunk

    for dst, src in zip(carry.tensors(), entry["carry"]):
        dst.copy_(src)
    remaining = start + chunk
    if remaining > 0:
        prev = _prev_slices(U, c_tape, initial_b, initial_c, H, 0, remaining)
        _backward_chunk(reads, grads, outs, prev, carry, recurrent, consts,
                        H, remaining, 0)


def _reverse_graph_entry(B, chunk, H, U, c_tape, recurrent, consts, carry,
                         reads, grads, outs, dtype, device) -> dict:
    """Static buffers plus the captured reverse chunk, built once per shape."""
    key = (B, chunk, H, dtype, device.index, len(consts))
    entry = _REVERSE_GRAPHS.get(key)
    if entry is not None:
        return entry

    def like(src):
        return torch.empty((B, chunk, *src.shape[2:]), device=device, dtype=src.dtype)

    entry = {
        "reads": tuple(like(t) for t in reads),
        "grads": tuple(like(t) for t in grads),
        "outs": tuple(like(t) for t in outs),
        "prev": [
            torch.empty((B, chunk, H), device=device, dtype=U.dtype),
            torch.empty((B, chunk, H), device=device, dtype=c_tape.dtype),
        ],
        "recurrent": torch.empty_like(recurrent),
        "consts": tuple(torch.empty_like(c) for c in consts),
        "carry": [t.clone() for t in carry.tensors()],
    }
    entry["recurrent"].copy_(recurrent)
    for dst, src in zip(entry["consts"], consts):
        dst.copy_(src)

    staged_carry = _ReverseCarry(
        entry["carry"][0], entry["carry"][1], entry["carry"][2],
        entry["carry"][3:19], entry["carry"][19:22],
    )
    entry["carry"] = staged_carry.tensors()

    def body() -> None:
        _backward_chunk(
            entry["reads"], entry["grads"], entry["outs"], entry["prev"],
            staged_carry, entry["recurrent"], entry["consts"], H, chunk, 0,
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    entry["graph"] = graph
    _REVERSE_GRAPHS[key] = entry
    return entry


class _ScanState:
    """Mutable carry for the forward scan (budget, energy, saturation)."""

    __slots__ = ("budget", "energy", "saturation")

    def __init__(self, budget: Tensor, energy: Tensor, saturation: Tensor):
        # Clone: the scan writes the carry back through these tensors so chunks
        # chain through fixed storage, and `initial_b.float()` returns the
        # CALLER's tensor unchanged when it is already fp32 -- writing through it
        # would silently mutate the caller's initial state.
        self.budget = budget.clone()
        self.energy = energy.clone()
        self.saturation = saturation.clone()


def _forward_chunk(
    Wx_c: Tensor, Wgx_c: Tensor, recurrent: Tensor,
    b_io: Tensor, e_io: Tensor, c_io: Tensor,
    tapes: tuple, params: tuple, H: int, count: int, offset: int = 0,
) -> None:
    """Advance ``count`` steps, writing tapes at ``offset .. offset+count``.

    The carry is read from ``b_io/e_io/c_io`` at entry and written back into
    those same tensors at exit, so consecutive calls chain through fixed storage.
    That is what lets a CUDA graph replay this body: the addresses a capture
    records stay valid, and each replay picks up exactly where the last left off.

    This is the single definition of the forward scan math -- the eager path and
    the captured path both run this function, so there is no second
    implementation that could drift.
    """
    budget, energy, saturation = b_io, e_io, c_io
    for step in range(count):
        (
            y, budget, next_energy, saturation,
            novelty, plasticity, expression, write, energy_before, gate,
        ) = _run_forward_step(
            Wx_c[:, step], Wgx_c[:, step], recurrent,
            budget, energy, saturation, *params,
        )
        energy = next_energy
        slot = offset + step
        tapes[0][:, slot, :H] = y
        tapes[0][:, slot, H:] = budget
        tapes[1][:, slot] = novelty
        tapes[2][:, slot] = plasticity
        tapes[3][:, slot] = expression
        tapes[4][:, slot] = write
        tapes[5][:, slot] = energy_before
        tapes[6][:, slot] = saturation
        tapes[7][:, slot] = gate
    b_io.copy_(budget)
    e_io.copy_(energy)
    c_io.copy_(saturation)


# Captured-chunk cache, keyed by execution shape. A capture is only valid for the
# exact buffers it recorded, so the buffers live in the cache alongside it.
_SCAN_GRAPHS: dict[tuple, dict] = {}
# Static staging for one chunk costs ~10 [B,chunk,H] buffers. This budget used
# to be a flat 512 MiB, and that turned capture OFF exactly where it was needed
# most: the per-step footprint grows with batch, so the number of steps that fit
# a FIXED budget shrinks as the batch grows. At 2048h/seq-512 on an A100 the
# reverse width fell from 37 steps at batch 128 to 18 at batch 256 -- under the
# 32-step floor -- so the whole reverse scan dropped back to one launch at a
# time. That is the entire reason batch 256 measured SLOWER than batch 128
# (3,797 vs 5,499 tok/s) on a step that is otherwise more efficient at 256.
#
# The budget is therefore a fraction of the device rather than a constant. It is
# derived from TOTAL memory, never free memory: total is a static device
# property, so reading it is safe inside a capture window, whereas live free
# memory is exactly the kind of host state that made a recorded graph misalign
# and fault. Deterministic per device, and it scales from a 16 GiB T4 to an
# 80 GiB A100 without a hardcoded number.
_SCAN_GRAPH_STAGE_FRACTION = 0.02
_SCAN_GRAPH_STAGE_FLOOR = 512 << 20
_SCAN_GRAPH_MIN_STEPS = 32
_STAGE_BYTES_BY_DEVICE: dict[int, int] = {}


def _scan_stage_bytes(device=None) -> int:
    """Bytes of static staging one captured chunk may hold, per device."""
    try:
        index = torch.cuda.current_device() if device is None else torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
    except Exception:
        return _SCAN_GRAPH_STAGE_FLOOR
    cached = _STAGE_BYTES_BY_DEVICE.get(index)
    if cached is not None:
        return cached
    try:
        total = torch.cuda.get_device_properties(index).total_memory
    except Exception:
        total = 0
    budget = max(_SCAN_GRAPH_STAGE_FLOOR, int(total * _SCAN_GRAPH_STAGE_FRACTION))
    _STAGE_BYTES_BY_DEVICE[index] = budget
    return budget


def _scan_chunk_width(B: int, T: int, H: int, elem: int, device=None) -> int:
    """Steps to record per capture, from the staging budget -- never a fixed T."""
    per_step = 10 * B * H * elem
    width = _scan_stage_bytes(device) // max(1, per_step)
    return int(max(1, min(width, T)))


def _scan_into_tapes(Wx, Wgx, recurrent, state, tapes, params, H) -> None:
    """Run the whole scan, replaying a captured chunk when that is possible.

    The scan issues one launch per timestep per layer. At a real training shape
    that is tens of thousands of Python-level dispatches per step, each costing
    far more than the ~14us of arithmetic it carries -- the loop is dominated by
    the cost of *asking* for work, not by the work. A recurrence cannot be
    parallelized across time, but those launches can be recorded once and
    replayed, which is what CUDA graphs are for.

    Capture is skipped when it cannot help or cannot be trusted: on CPU, inside
    an enclosing capture, when grad is live (the tapes would need autograd
    plumbing a graph cannot record), or when the scan is too short to amortize
    the staging copies. In every one of those cases -- and if capture itself
    fails -- the SAME ``_forward_chunk`` runs uncaptured. The math is one
    function; capture only changes how its launches are issued.
    """
    B, T, _ = Wx.shape
    eager = lambda: _forward_chunk(  # noqa: E731
        Wx, Wgx, recurrent, state.budget, state.energy, state.saturation,
        tapes, params, H, T, 0,
    )
    if (
        not Wx.is_cuda
        or T < _SCAN_GRAPH_MIN_STEPS
        or torch.is_grad_enabled()
        or _stream_is_capturing()
        or os.environ.get("DABSN_SCAN_GRAPH", "1") != "1"
    ):
        eager()
        return

    chunk = _scan_chunk_width(B, T, H, Wx.element_size(), Wx.device)
    if chunk < _SCAN_GRAPH_MIN_STEPS:
        eager()
        return

    try:
        entry = _scan_graph_entry(B, chunk, H, Wx, recurrent, params)
    except Exception as exc:  # noqa: BLE001 - identical math runs below
        _disable_scan_graph(exc)
        eager()
        return

    # Parameters and the recurrent matrix change every optimizer step, so their
    # current values must land in the buffers the capture recorded.
    entry["recurrent"].copy_(recurrent)
    for dst, src in zip(entry["params"], params):
        dst.copy_(src)
    entry["b"].copy_(state.budget)
    entry["e"].copy_(state.energy)
    entry["c"].copy_(state.saturation)

    done = 0
    while T - done >= chunk:
        entry["wx"].copy_(Wx[:, done:done + chunk])
        entry["wgx"].copy_(Wgx[:, done:done + chunk])
        entry["graph"].replay()
        for real, staged in zip(tapes, entry["tapes"]):
            real[:, done:done + chunk].copy_(staged)
        done += chunk

    state.budget = entry["b"].clone()
    state.energy = entry["e"].clone()
    state.saturation = entry["c"].clone()
    if done < T:
        # Tail shorter than one capture: same function, uncaptured.
        _forward_chunk(
            Wx[:, done:], Wgx[:, done:], recurrent,
            state.budget, state.energy, state.saturation,
            tapes, params, H, T - done, done,
        )


def _disable_scan_graph(exc: BaseException) -> None:
    os.environ["DABSN_SCAN_GRAPH"] = "0"
    from dabsn.runtime.dispatch import warn_routing_once

    warn_routing_once(
        "scan_graph",
        f"CUDA-graph capture of the core scan failed ({type(exc).__name__}); "
        "running the same scan uncaptured for the rest of this process. The math "
        "is identical -- only per-step launch overhead differs.",
        reason=str(exc)[:200],
    )


def _scan_graph_entry(B: int, chunk: int, H: int, Wx: Tensor,
                      recurrent: Tensor, params: tuple) -> dict:
    """Build (once per shape) the static buffers and the captured chunk."""
    key = (B, chunk, H, Wx.dtype, Wx.device.index, len(params))
    entry = _SCAN_GRAPHS.get(key)
    if entry is not None:
        return entry

    dev, dt = Wx.device, Wx.dtype
    f32 = torch.float32
    entry = {
        "wx": torch.empty((B, chunk, H), device=dev, dtype=dt),
        "wgx": torch.empty((B, chunk, H), device=dev, dtype=dt),
        "recurrent": torch.empty_like(recurrent),
        "b": torch.zeros((B, H), device=dev, dtype=f32),
        "e": torch.ones((B, H), device=dev, dtype=f32),
        "c": torch.zeros((B, H), device=dev, dtype=f32),
        "params": tuple(torch.empty_like(p) for p in params),
        "tapes": (
            torch.empty((B, chunk, 2 * H), device=dev, dtype=dt),
            *(torch.empty((B, chunk, H), device=dev, dtype=dt) for _ in range(7)),
        ),
    }
    entry["recurrent"].copy_(recurrent)
    for dst, src in zip(entry["params"], params):
        dst.copy_(src)

    def body() -> None:
        _forward_chunk(
            entry["wx"], entry["wgx"], entry["recurrent"],
            entry["b"], entry["e"], entry["c"],
            entry["tapes"], entry["params"], H, chunk, 0,
        )

    # Warm up on a side stream so the capture records steady-state work only:
    # first-call autotuning, lazy module init, and allocator growth must all
    # happen BEFORE the recording or they get baked into the graph.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    entry["graph"] = graph
    _SCAN_GRAPHS[key] = entry
    return entry


def _batched_forward_tapes(
    Wx, Wgx, Ug, A,
    beta, log_kappa, logit_recover,
    k_s, k_y, k_b, k_n, k_bias,
    r_s, r_y, r_b, r_n, r_bias,
    logit_c_decay, k_c, r_c,
    logit_alpha, log_lambda, logit_c_suppress,
    initial_b, initial_e, initial_c,
) -> tuple[Tensor, ...]:
    """Advance the whole batch through T with per-step tensor-core GEMMs.

    Returns exactly the tapes the reverse recurrence consumes, in the order
    ``(U, novelty, plasticity, expression, write, e_tape, c_tape, s_tape,
    final_b, final_e, final_c)``. This is the single source of truth for the
    forward math shared by the eager batched Function and the fused-kernel
    Function's CPU/large-H fallback, so both save an identical backward context.
    """

    B, T, H = Wx.shape
    # Parameters remain FP32 master weights for AdamW.  The recurrent GEMM must
    # use the live activation compute dtype (BF16 on Ampere/Hopper, FP16 on
    # Turing when selected) or this tensor-core backend silently degenerates
    # into TF32/FP32.  Pointwise state remains FP32.
    recurrent = torch.cat((Ug, A), dim=0).to(Wx.dtype).contiguous()
    # The state chain carries the activation dtype, so the ~30 elementwise ops
    # per step move half the bytes they did in FP32. `_forward_step` derives its
    # working precision from these tensors, so this one line is the policy.
    state_dt = _state_dtype(Wx.dtype)
    budget = initial_b.to(state_dt)
    energy = initial_e.to(state_dt)
    saturation = initial_c.to(state_dt)

    # Preallocate the output tapes in the activation dtype and write each step in
    # place. This replaces eight Python lists + a full-T `torch.stack(...).to()`
    # per tape: stacking first built a transient FP32 [B,T,H] for every tape and
    # then cast it, doubling the tape working set at the boundary. Writing an
    # FP32 step row into a preallocated activation-dtype tape casts each element
    # once (RTNE), which is bit-identical to stacking in FP32 and casting the
    # whole tape at the end. U packs (y, budget) along the last axis exactly as
    # the old `torch.cat((y, budget))` row did.
    dt = Wx.dtype
    U = torch.empty((B, T, 2 * H), device=Wx.device, dtype=dt)
    novelty_t = torch.empty((B, T, H), device=Wx.device, dtype=dt)
    plasticity_t = torch.empty_like(novelty_t)
    expression_t = torch.empty_like(novelty_t)
    write_t = torch.empty_like(novelty_t)
    energy_t = torch.empty_like(novelty_t)
    saturation_t = torch.empty_like(novelty_t)
    gate_t = torch.empty_like(novelty_t)

    tapes = (U, novelty_t, plasticity_t, expression_t, write_t,
             energy_t, saturation_t, gate_t)
    params = (
        beta, log_kappa, logit_recover,
        k_s, k_y, k_b, k_n, k_bias,
        r_s, r_y, r_b, r_n, r_bias,
        logit_c_decay, k_c, r_c,
        logit_alpha, log_lambda, logit_c_suppress,
    )
    state = _ScanState(budget, energy, saturation)
    _scan_into_tapes(Wx, Wgx, recurrent, state, tapes, params, H)
    budget, energy, saturation = state.budget, state.energy, state.saturation

    return (
        U,
        novelty_t,
        plasticity_t,
        expression_t,
        write_t,
        energy_t,
        saturation_t,
        gate_t,
        budget.to(Wx.dtype),
        energy.to(Wx.dtype),
        saturation.to(Wx.dtype),
    )


class DABSNCoreScanBatched(torch.autograd.Function):
    """Memory-bounded batched-GEMM scan with an explicit reverse recurrence."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        Wx: Tensor,
        Wgx: Tensor,
        Ug: Tensor,
        A: Tensor,
        beta: Tensor,
        log_kappa: Tensor,
        logit_recover: Tensor,
        k_s: Tensor,
        k_y: Tensor,
        k_b: Tensor,
        k_n: Tensor,
        k_bias: Tensor,
        r_s: Tensor,
        r_y: Tensor,
        r_b: Tensor,
        r_n: Tensor,
        r_bias: Tensor,
        logit_c_decay: Tensor,
        k_c: Tensor,
        r_c: Tensor,
        logit_alpha: Tensor,
        log_lambda: Tensor,
        logit_c_suppress: Tensor,
        initial_b: Tensor,
        initial_e: Tensor,
        initial_c: Tensor,
        return_tape: bool,
        return_final_state: bool,
    ) -> tuple[Tensor, ...]:
        B, T, H = Wx.shape
        if Wgx.shape != Wx.shape or Ug.shape != (H, H) or A.shape != (H, H):
            raise ValueError("batched DABSN core received incompatible tensor shapes")
        ctx.return_tape = bool(return_tape)
        ctx.return_final_state = bool(return_final_state)
        (
            U, novelty, plasticity, expression, write,
            e_tape, c_tape, s_tape, final_b, final_e, final_c,
        ) = _batched_forward_tapes(
            Wx, Wgx, Ug, A,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha, log_lambda, logit_c_suppress,
            initial_b, initial_e, initial_c,
        )
        ctx.save_for_backward(
            Wx, Wgx, Ug, A,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha, log_lambda, logit_c_suppress,
            initial_b, initial_e, initial_c,
            U, plasticity, expression,
            e_tape, c_tape, s_tape,
            final_b, final_e, final_c,
        )
        public = [U, novelty, plasticity, expression, write]
        if ctx.return_tape:
            public.extend((e_tape, c_tape))
        if ctx.return_final_state:
            public.extend((final_b, final_e, final_c))
        return tuple(public)

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        (
            Wx, Wgx, Ug, A,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha, log_lambda, logit_c_suppress,
            initial_b, initial_e, initial_c,
            U, plasticity, expression,
            e_tape, c_tape, s_tape,
            final_b, final_e, final_c,
        ) = ctx.saved_tensors
        # novelty and write are no longer saved (Phase 5 / 2d); plasticity has
        # the same [B,T,H] shape and serves as the zeros_like template.
        B, T, H = plasticity.shape

        def public_grad(index: int, value: Tensor) -> Tensor:
            if index >= len(grad_outputs) or grad_outputs[index] is None:
                return torch.zeros_like(value, dtype=torch.float32)
            return grad_outputs[index].float()

        gU = public_grad(0, U)
        gnov = public_grad(1, plasticity)
        gp_out = public_grad(2, plasticity)
        gay_out = public_grad(3, expression)
        gwrite = public_grad(4, plasticity)
        if ctx.return_tape:
            ge_out = public_grad(5, e_tape)
            gc_out = public_grad(6, c_tape)
            final_offset = 7
        else:
            ge_out = torch.zeros_like(e_tape, dtype=torch.float32)
            gc_out = torch.zeros_like(c_tape, dtype=torch.float32)
            final_offset = 5
        if ctx.return_final_state:
            gbn = public_grad(final_offset, final_b)
            gen = public_grad(final_offset + 1, final_e)
            gcn = public_grad(final_offset + 2, final_c)
        else:
            gbn = torch.zeros_like(final_b, dtype=torch.float32)
            gen = torch.zeros_like(final_e, dtype=torch.float32)
            gcn = torch.zeros_like(final_c, dtype=torch.float32)

        recurrent_backward = torch.cat((Ug, A), dim=0).to(Wx.dtype).contiguous()
        lk, lr = log_kappa.float(), logit_recover.float()
        ks, ky, kb, kn, kbias = (x.float() for x in (k_s, k_y, k_b, k_n, k_bias))
        rs, ry, rb, rn, rbias = (x.float() for x in (r_s, r_y, r_b, r_n, r_bias))
        decay = torch.sigmoid(logit_c_decay.float())
        kc, rc = k_c.float(), r_c.float()
        alpha = torch.sigmoid(logit_alpha.float())
        lam, sig_lam = F.softplus(log_lambda.float()), torch.sigmoid(log_lambda.float())
        suppress = torch.sigmoid(logit_c_suppress.float())

        # Backward activation scratch. These three [B,T,H] tapes dominate the
        # reverse-pass working set, so they are allocated directly in the input
        # activation dtype instead of FP32: each element is written once from an
        # FP32 step result, and casting at store (RTNE per element) is
        # bit-identical to accumulating in FP32 and casting the whole tape at the
        # end -- which is exactly what the returns below already did. The
        # recurrent-GEMM contraction (dpre/d_ay @ y) also ran in Wx.dtype, so
        # feeding it Wx.dtype tapes changes nothing. `dpre_tape` used to be a
        # third [B,T,H] tape holding a byte-for-byte copy of `gWgx` (both stored
        # the same `dpre`); it is gone, and the GEMM reads `gWgx` directly.
        # gWgx doubles as grad_Wgx (returned via .to(Wgx.dtype)) and as the Ug
        # half of the recurrent GEMM (needs Wx.dtype), so those dtypes must match
        # or storing in one would double-round for the other.
        if Wgx.dtype != Wx.dtype:
            raise RuntimeError(
                "DABSNCoreScanBatched backward requires Wx and Wgx to share a "
                f"dtype (got Wx={Wx.dtype}, Wgx={Wgx.dtype}); the batched core "
                "always projects both inputs in the block activation dtype."
            )
        gWx = torch.empty((B, T, H), device=Wx.device, dtype=Wx.dtype)
        gWgx = torch.empty_like(gWx)
        day_tape = torch.empty_like(gWx)
        param_acc = [torch.zeros(H, device=Wx.device, dtype=torch.float32) for _ in range(16)]
        scalar_acc = [torch.zeros((), device=Wx.device, dtype=torch.float32) for _ in range(3)]

        carry = _ReverseCarry(gbn, gen, gcn, param_acc, scalar_acc)
        _reverse_into_grads(
            reads=(U, e_tape, c_tape, s_tape, plasticity, expression),
            grads=(gU, gnov, gp_out, gay_out, gwrite, ge_out, gc_out),
            outs=(gWx, gWgx, day_tape),
            initial=(initial_b, initial_c),
            carry=carry,
            recurrent=recurrent_backward,
            consts=(lk, lr, ks, ky, kb, kn, kbias, rs, ry, rb, rn, rbias,
                    decay, kc, rc, alpha, lam, sig_lam, suppress),
            B=B, T=T, H=H, dtype=Wx.dtype, device=Wx.device,
        )
        gbn, gen, gcn = carry.gb, carry.ge, carry.gc
        param_acc, scalar_acc = carry.param_acc, carry.scalar_acc

        y_flat = U[:, :, :H].reshape(B * T, H).to(Wx.dtype)
        # gWgx holds the gate-pre grad (dpre, the Ug half); day_tape holds the
        # expression grad (d_ay, the A half). Both are already Wx.dtype, so the
        # cat feeds the contraction with no extra cast.
        grad_recurrent = (
            torch.cat((gWgx, day_tape), dim=-1)
            .reshape(B * T, 2 * H)
            .T
            @ y_flat
        ).float()
        grad_Ug, grad_A = grad_recurrent.split(H, dim=0)
        return (
            gWx.to(Wx.dtype), gWgx.to(Wgx.dtype), grad_Ug.to(Ug.dtype), grad_A.to(A.dtype),
            *(value.to(beta.dtype) for value in param_acc[:1]),
            param_acc[1].to(log_kappa.dtype), param_acc[2].to(logit_recover.dtype),
            *(value.to(k_s.dtype) for value in param_acc[3:8]),
            *(value.to(r_s.dtype) for value in param_acc[8:13]),
            param_acc[13].to(logit_c_decay.dtype),
            param_acc[14].to(k_c.dtype), param_acc[15].to(r_c.dtype),
            scalar_acc[0].reshape_as(logit_alpha).to(logit_alpha.dtype),
            scalar_acc[1].reshape_as(log_lambda).to(log_lambda.dtype),
            scalar_acc[2].reshape_as(logit_c_suppress).to(logit_c_suppress.dtype),
            gbn.to(initial_b.dtype), gen.to(initial_e.dtype), gcn.to(initial_c.dtype),
            None, None,
        )


def dabsn_core_scan_batched(
    Wx: Tensor,
    Wgx: Tensor,
    Ug: Tensor,
    A: Tensor,
    beta: Tensor,
    log_kappa: Tensor,
    logit_recover: Tensor,
    k_s: Tensor,
    k_y: Tensor,
    k_b: Tensor,
    k_n: Tensor,
    k_bias: Tensor,
    r_s: Tensor,
    r_y: Tensor,
    r_b: Tensor,
    r_n: Tensor,
    r_bias: Tensor,
    logit_c_decay: Tensor,
    k_c: Tensor,
    r_c: Tensor,
    logit_alpha: Tensor,
    log_lambda: Tensor,
    logit_c_suppress: Tensor,
    *,
    return_tape: bool = False,
    initial_state: tuple[Tensor, Tensor, Tensor] | None = None,
    return_final_state: bool = False,
) -> tuple[Tensor, ...]:
    B, _, H = Wx.shape
    if initial_state is None:
        # Carried state stays FP32: the forward immediately floats it
        # (`budget = initial_b.float()`), so an activation-dtype carry would only
        # round the chunk-boundary state for nothing. FP32 here makes a chunked
        # scan bit-exact against the full scan on GPU (0,1,0 are exact in both
        # dtypes, so default init is unchanged).
        initial_state = (
            torch.zeros((B, H), device=Wx.device, dtype=torch.float32),
            torch.ones((B, H), device=Wx.device, dtype=torch.float32),
            torch.zeros((B, H), device=Wx.device, dtype=torch.float32),
        )
    initial_b, initial_e, initial_c = (
        value.to(device=Wx.device).contiguous() for value in initial_state
    )
    return DABSNCoreScanBatched.apply(
        Wx, Wgx, Ug, A,
        beta, log_kappa, logit_recover,
        k_s, k_y, k_b, k_n, k_bias,
        r_s, r_y, r_b, r_n, r_bias,
        logit_c_decay, k_c, r_c,
        logit_alpha, log_lambda, logit_c_suppress,
        initial_b, initial_e, initial_c,
        bool(return_tape), bool(return_final_state),
    )


# ---------------------------------------------------------------------------
# batched_fused: single-launch tiled tensor-core scan (Phase B)
# ---------------------------------------------------------------------------
# The batched scan above issues one tensor-core GEMM per recurrent step but
# advances the T loop in Python: every step round-trips the carried
# budget/energy/saturation state through HBM. `batched_fused` replaces the
# forward scan with a SINGLE Triton launch whose programs each own a BLOCK_B
# tile of sequences, loop T on-chip carrying state in registers, and contract
# [Ug;A] with `tl.dot` (tensor cores) -- no Python loop, no per-step HBM state
# round-trip. It writes exactly the tapes the reverse recurrence needs, so the
# backward reuses the already-certified batched reverse unchanged. The Triton
# forward is GPU-only; on CPU / when Triton is unavailable / for widths past one
# tile it falls back to the identical `_batched_forward_tapes` math, so results
# are always exact and the Function's tape/backward wiring is CPU-provable.
#
# `select_core_backend` never returns `batched_fused` from `auto`: per the plan
# it is reachable only by explicit request until `tools/train_scale_gate.py`
# certifies the Triton forward bit-parity on real hardware (the gate already
# includes it in its parity matrix).


# The fused forward K-tiles the recurrent contraction, so shared memory is
# bounded by BLOCK_K and there is no matrix-width SMEM limit. The real ceiling is
# the register file: a program carries the full [BLOCK_B, H] budget/energy/
# saturation state across T, so H is capped where those tiles stop fitting in
# registers (~256 for BLOCK_B=16 on A100/H100). Wider cores -- a 1B model is
# H~2048 -- run the batched tensor-core GEMM scan, which is cuBLAS-optimal at
# that width and, under CUDA graphs, is the transformer-competitive scale path;
# the fused single-tile kernel is the small/medium-H win, not the scale path.
def _fused_max_h(device=None, batch: int = 16) -> int:
    """Width ceiling for the single-launch fused scan, derived from the device.

    Never a hardcoded model width: the fused scan keeps three fp32 ``[tile, H]``
    state tiles live across the whole T loop, so the ceiling falls out of the
    register file (see ``triton_runtime._fused_max_hidden``). On CPU or without
    Triton the question is moot and the env value (or 256) answers it.
    """
    env = os.environ.get("DABSN_FUSED_CORE_MAX_H")
    if env not in (None, ""):
        return int(env)
    try:
        from .triton_runtime import _fused_batch_tile, _fused_max_hidden

        return _fused_max_hidden(device, _fused_batch_tile(int(batch)))
    except Exception:
        return 256


def _fused_forward_available(Wx: Tensor) -> bool:
    if not Wx.is_cuda or os.environ.get("DABSN_FUSED_CORE_DISABLE", "0") == "1":
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return int(Wx.shape[-1]) <= _fused_max_h(Wx.device, int(Wx.shape[0]))


class DABSNCoreScanBatchedFused(torch.autograd.Function):
    """Single-launch fused-forward scan; shares the batched reverse recurrence."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        Wx, Wgx, Ug, A,
        beta, log_kappa, logit_recover,
        k_s, k_y, k_b, k_n, k_bias,
        r_s, r_y, r_b, r_n, r_bias,
        logit_c_decay, k_c, r_c,
        logit_alpha, log_lambda, logit_c_suppress,
        initial_b, initial_e, initial_c,
        return_tape, return_final_state,
    ) -> tuple[Tensor, ...]:
        B, T, H = Wx.shape
        if Wgx.shape != Wx.shape or Ug.shape != (H, H) or A.shape != (H, H):
            raise ValueError("fused DABSN core received incompatible tensor shapes")
        ctx.return_tape = bool(return_tape)
        ctx.return_final_state = bool(return_final_state)
        # Whether the Triton fused forward ran (vs the eager tape fallback)
        # decides which reverse the backward uses: the single-launch Triton
        # backward when on GPU+Triton, else the certified eager reverse. The
        # forward writes an identical tape contract either way, so both reverses
        # consume the same saved context.
        used_triton = _fused_forward_available(Wx)
        ctx.used_triton = bool(used_triton)
        if not used_triton:
            # The caller reached the fused Function but the single-launch Triton
            # forward cannot run: say why, once, so the drop to the eager tape is
            # never silent (the math is identical; only the throughput differs).
            from dabsn.runtime.dispatch import warn_routing_once

            if not Wx.is_cuda:
                reason = "input is not on CUDA"
            elif os.environ.get("DABSN_FUSED_CORE_DISABLE", "0") == "1":
                reason = "DABSN_FUSED_CORE_DISABLE=1"
            elif int(Wx.shape[-1]) > _fused_max_h(Wx.device, int(Wx.shape[0])):
                cap = _fused_max_h(Wx.device, int(Wx.shape[0]))
                reason = (
                    f"H={int(Wx.shape[-1])} exceeds the device's fused-scan width "
                    f"ceiling {cap} (register file holds the carried state across T)"
                )
            else:
                reason = "triton import failed"
            warn_routing_once(
                "core_scan_fused",
                f"fused forward unavailable ({reason}); using eager batched tape",
                hidden=int(Wx.shape[-1]),
            )
        if used_triton:
            from .triton_runtime import (
                dabsn_core_scan_batched_fused_forward,
                is_triton_out_of_resources,
            )

            try:
                tapes = dabsn_core_scan_batched_fused_forward(
                    Wx, Wgx, Ug, A,
                    beta, log_kappa, logit_recover,
                    k_s, k_y, k_b, k_n, k_bias,
                    r_s, r_y, r_b, r_n, r_bias,
                    logit_c_decay, k_c, r_c,
                    logit_alpha, log_lambda, logit_c_suppress,
                    initial_b, initial_e, initial_c,
                )
            except Exception as exc:
                if not is_triton_out_of_resources(exc):
                    raise
                ctx.used_triton = False
                from dabsn.runtime.dispatch import warn_routing_once

                warn_routing_once(
                    "core_scan_fused",
                    "fused forward exceeds this device's launch resources; "
                    "using eager batched tape",
                    hidden=int(Wx.shape[-1]),
                )
                tapes = _batched_forward_tapes(
                    Wx, Wgx, Ug, A,
                    beta, log_kappa, logit_recover,
                    k_s, k_y, k_b, k_n, k_bias,
                    r_s, r_y, r_b, r_n, r_bias,
                    logit_c_decay, k_c, r_c,
                    logit_alpha, log_lambda, logit_c_suppress,
                    initial_b, initial_e, initial_c,
                )
        else:
            tapes = _batched_forward_tapes(
                Wx, Wgx, Ug, A,
                beta, log_kappa, logit_recover,
                k_s, k_y, k_b, k_n, k_bias,
                r_s, r_y, r_b, r_n, r_bias,
                logit_c_decay, k_c, r_c,
                logit_alpha, log_lambda, logit_c_suppress,
                initial_b, initial_e, initial_c,
            )
        (U, novelty, plasticity, expression, write,
         e_tape, c_tape, s_tape, final_b, final_e, final_c) = tapes
        ctx.save_for_backward(
            Wx, Wgx, Ug, A,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha, log_lambda, logit_c_suppress,
            initial_b, initial_e, initial_c,
            U, plasticity, expression,
            e_tape, c_tape, s_tape,
            final_b, final_e, final_c,
        )
        public = [U, novelty, plasticity, expression, write]
        if ctx.return_tape:
            public.extend((e_tape, c_tape))
        if ctx.return_final_state:
            public.extend((final_b, final_e, final_c))
        return tuple(public)

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        # Off the GPU/Triton path (CPU, large H, resource fallback, or Triton
        # disabled) the tapes came from the eager forward, so use the certified
        # eager reverse. The single-launch Triton reverse remains opt-in until
        # it has parity across supported GPU architectures.
        use_triton_bwd = (
            getattr(ctx, "used_triton", False)
            and os.environ.get("DABSN_FUSED_CORE_TRITON_BWD", "0") == "1"
        )
        if not use_triton_bwd:
            # If the Triton forward ran but the single-launch reverse is disabled,
            # the backward drops to the Python per-step reverse -- correct, but a
            # real throughput cliff on GPU. Announce that once.
            if getattr(ctx, "used_triton", False):
                from dabsn.runtime.dispatch import warn_routing_once

                warn_routing_once(
                    "core_scan_fused",
                    "fused Triton forward ran but DABSN_FUSED_CORE_TRITON_BWD=0; "
                    "backward uses the slower Python per-step reverse",
                )
            return DABSNCoreScanBatched.backward(ctx, *grad_outputs)

        # Single-launch reverse: reuse the already-certified fused backward
        # (one-tile / K-tiled chunked) that the persistent Triton path uses.
        # The fused forward writes exactly the tape set it consumes, so this is
        # bit-parity with the eager reverse (the GPU parity test asserts it),
        # with no Python T-loop -- the graph-free / consumer-GPU throughput win.
        from .triton_runtime import _dabsn_core_fused_backward

        (
            Wx, Wgx, Ug, A,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha, log_lambda, logit_c_suppress,
            initial_b, initial_e, initial_c,
            U, plasticity, expression,
            e_tape, c_tape, s_tape,
            final_b, final_e, final_c,
        ) = ctx.saved_tensors

        gU, gnov_out, gp_out, gay_out, gwrite_out = (
            torch.zeros_like(out) if grad is None else grad.contiguous()
            # novelty and write are not saved (Phase 5 / 2d); plasticity (same
            # [B,T,H]) is the zeros_like shape template for their public grads.
            for out, grad in zip((U, plasticity, plasticity, expression, plasticity), grad_outputs)
        )
        ge_out = (
            torch.zeros_like(e_tape)
            if not ctx.return_tape or grad_outputs[5] is None
            else grad_outputs[5].contiguous()
        )
        gc_out = (
            torch.zeros_like(c_tape)
            if not ctx.return_tape or grad_outputs[6] is None
            else grad_outputs[6].contiguous()
        )
        final_offset = 7 if ctx.return_tape else 5
        if ctx.return_final_state:
            gfinal_b = (
                torch.zeros_like(final_b)
                if grad_outputs[final_offset] is None
                else grad_outputs[final_offset].contiguous()
            )
            gfinal_e = (
                torch.zeros_like(final_e)
                if grad_outputs[final_offset + 1] is None
                else grad_outputs[final_offset + 1].contiguous()
            )
            gfinal_c = (
                torch.zeros_like(final_c)
                if grad_outputs[final_offset + 2] is None
                else grad_outputs[final_offset + 2].contiguous()
            )
        else:
            gfinal_b = torch.zeros_like(final_b)
            gfinal_e = torch.zeros_like(final_e)
            gfinal_c = torch.zeros_like(final_c)

        (
            gWx, gWgx, grad_A, grad_Ug, gparam, gscal,
            ginitial_b, ginitial_e, ginitial_c,
        ) = _dabsn_core_fused_backward(
            U, plasticity, expression, e_tape, c_tape, s_tape,
            initial_b, initial_c,
            A, Ug,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha, log_lambda, logit_c_suppress,
            gU, gnov_out, gp_out, gay_out, gwrite_out, ge_out, gc_out,
            gfinal_b, gfinal_e, gfinal_c,
        )
        return (
            gWx.to(Wx.dtype), gWgx.to(Wgx.dtype),
            grad_Ug.to(Ug.dtype), grad_A.to(A.dtype),
            gparam[0].to(beta.dtype),
            gparam[1].to(log_kappa.dtype),
            gparam[2].to(logit_recover.dtype),
            gparam[3].to(k_s.dtype),
            gparam[4].to(k_y.dtype),
            gparam[5].to(k_b.dtype),
            gparam[6].to(k_n.dtype),
            gparam[7].to(k_bias.dtype),
            gparam[8].to(r_s.dtype),
            gparam[9].to(r_y.dtype),
            gparam[10].to(r_b.dtype),
            gparam[11].to(r_n.dtype),
            gparam[12].to(r_bias.dtype),
            gparam[13].to(logit_c_decay.dtype),
            gparam[14].to(k_c.dtype),
            gparam[15].to(r_c.dtype),
            gscal[0].reshape_as(logit_alpha).to(logit_alpha.dtype),
            gscal[1].reshape_as(log_lambda).to(log_lambda.dtype),
            gscal[2].reshape_as(logit_c_suppress).to(logit_c_suppress.dtype),
            ginitial_b.to(initial_b.dtype),
            ginitial_e.to(initial_e.dtype),
            ginitial_c.to(initial_c.dtype),
            None, None,
        )


def dabsn_core_scan_batched_fused(
    Wx: Tensor,
    Wgx: Tensor,
    Ug: Tensor,
    A: Tensor,
    beta: Tensor,
    log_kappa: Tensor,
    logit_recover: Tensor,
    k_s: Tensor,
    k_y: Tensor,
    k_b: Tensor,
    k_n: Tensor,
    k_bias: Tensor,
    r_s: Tensor,
    r_y: Tensor,
    r_b: Tensor,
    r_n: Tensor,
    r_bias: Tensor,
    logit_c_decay: Tensor,
    k_c: Tensor,
    r_c: Tensor,
    logit_alpha: Tensor,
    log_lambda: Tensor,
    logit_c_suppress: Tensor,
    *,
    return_tape: bool = False,
    initial_state: tuple[Tensor, Tensor, Tensor] | None = None,
    return_final_state: bool = False,
) -> tuple[Tensor, ...]:
    """Single-launch fused-forward core scan (see module note on `batched_fused`)."""

    B, _, H = Wx.shape
    if initial_state is None:
        # FP32 carried state (see the batched wrapper): the forward floats it, so
        # this avoids a needless chunk-boundary round and keeps chunked == full
        # exact on GPU.
        initial_state = (
            torch.zeros((B, H), device=Wx.device, dtype=torch.float32),
            torch.ones((B, H), device=Wx.device, dtype=torch.float32),
            torch.zeros((B, H), device=Wx.device, dtype=torch.float32),
        )
    initial_b, initial_e, initial_c = (
        value.to(device=Wx.device).contiguous() for value in initial_state
    )
    return DABSNCoreScanBatchedFused.apply(
        Wx, Wgx, Ug, A,
        beta, log_kappa, logit_recover,
        k_s, k_y, k_b, k_n, k_bias,
        r_s, r_y, r_b, r_n, r_bias,
        logit_c_decay, k_c, r_c,
        logit_alpha, log_lambda, logit_c_suppress,
        initial_b, initial_e, initial_c,
        bool(return_tape), bool(return_final_state),
    )
