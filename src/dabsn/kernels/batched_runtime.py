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
):
    """One recurrence step over the complete device batch."""

    y = torch.tanh(wx.float() + budget)
    # BF16/FP16 GEMM is the reason this backend exists. The pointwise recurrence
    # remains FP32, matching the arithmetic used inside the Triton scan.
    recurrent_out = F.linear(y.to(recurrent.dtype), recurrent).float()
    gate_recurrence, expression = recurrent_out.chunk(2, dim=-1)
    gate = F.hardsigmoid(wgx.float() + gate_recurrence)
    novelty = torch.tanh((expression - budget).abs())

    decay = torch.sigmoid(saturation_decay.float())
    stress = novelty * (1.0 - energy)
    saturation = decay * saturation + (1.0 - decay) * stress
    suppress = torch.sigmoid(logit_saturation_suppress.float())
    novelty_effective = novelty * (1.0 - suppress * saturation)

    tanh_budget = torch.tanh(budget)
    k_signal = (
        k_s.float() * gate
        + k_y.float() * y
        + k_b.float() * tanh_budget
        + k_n.float() * novelty_effective
        + k_bias.float()
        + k_saturation.float() * saturation
    )
    r_signal = (
        r_s.float() * gate
        + r_y.float() * y
        + r_b.float() * tanh_budget
        + r_n.float() * novelty_effective
        + r_bias.float()
        + r_saturation.float() * saturation
    )
    write_cost = F.softplus(log_kappa.float()) * torch.exp(0.5 * torch.tanh(k_signal))
    recovery = torch.clamp(
        torch.sigmoid(logit_recover.float()) * torch.exp(0.5 * torch.tanh(r_signal)),
        0.0,
        1.0,
    )
    plasticity = gate * energy
    write = plasticity * expression
    next_budget = (
        (1.0 - torch.sigmoid(logit_alpha.float())) * budget
        + beta.float()
        + F.softplus(log_lambda.float()) * write
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


_COMPILED_FORWARD_STEP = None
_COMPILED_BACKWARD_STEP = None


def _run_forward_step(*args):
    global _COMPILED_FORWARD_STEP
    use_compile = args[0].is_cuda and os.environ.get("DABSN_BATCHED_STEP_COMPILE", "1") == "1"
    if not use_compile:
        return _forward_step(*args)
    if _COMPILED_FORWARD_STEP is None:
        _COMPILED_FORWARD_STEP = torch.compile(
            _forward_step,
            dynamic=False,
            fullgraph=True,
        )
    return _COMPILED_FORWARD_STEP(*args)


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
    global _COMPILED_BACKWARD_STEP
    use_compile = args[0].is_cuda and os.environ.get("DABSN_BATCHED_STEP_COMPILE", "1") == "1"
    if not use_compile:
        return _backward_step(*args)
    if _COMPILED_BACKWARD_STEP is None:
        _COMPILED_BACKWARD_STEP = torch.compile(
            _backward_step,
            dynamic=False,
            fullgraph=True,
        )
    return _COMPILED_BACKWARD_STEP(*args)


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
    budget = initial_b.float()
    energy = initial_e.float()
    saturation = initial_c.float()
    u_rows, novelty_rows, plasticity_rows = [], [], []
    expression_rows, write_rows, energy_rows = [], [], []
    saturation_rows, gate_rows = [], []
    for step in range(T):
        (
            y,
            budget,
            next_energy,
            saturation,
            novelty,
            plasticity,
            expression,
            write,
            energy_before,
            gate,
        ) = _run_forward_step(
            Wx[:, step], Wgx[:, step], recurrent,
            budget, energy, saturation,
            beta, log_kappa, logit_recover,
            k_s, k_y, k_b, k_n, k_bias,
            r_s, r_y, r_b, r_n, r_bias,
            logit_c_decay, k_c, r_c,
            logit_alpha, log_lambda, logit_c_suppress,
        )
        energy = next_energy
        u_rows.append(torch.cat((y, budget), dim=-1))
        novelty_rows.append(novelty)
        plasticity_rows.append(plasticity)
        expression_rows.append(expression)
        write_rows.append(write)
        energy_rows.append(energy_before)
        saturation_rows.append(saturation)
        gate_rows.append(gate)

    def stack(rows):
        return torch.stack(rows, dim=1).to(Wx.dtype)

    return (
        stack(u_rows),
        stack(novelty_rows),
        stack(plasticity_rows),
        stack(expression_rows),
        stack(write_rows),
        stack(energy_rows),
        stack(saturation_rows),
        stack(gate_rows),
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
            U, novelty, plasticity, expression, write,
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
            U, novelty, plasticity, expression, write,
            e_tape, c_tape, s_tape,
            final_b, final_e, final_c,
        ) = ctx.saved_tensors
        B, T, H = novelty.shape

        def public_grad(index: int, value: Tensor) -> Tensor:
            if index >= len(grad_outputs) or grad_outputs[index] is None:
                return torch.zeros_like(value, dtype=torch.float32)
            return grad_outputs[index].float()

        gU = public_grad(0, U)
        gnov = public_grad(1, novelty)
        gp_out = public_grad(2, plasticity)
        gay_out = public_grad(3, expression)
        gwrite = public_grad(4, write)
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

        gWx = torch.empty((B, T, H), device=Wx.device, dtype=torch.float32)
        gWgx = torch.empty_like(gWx)
        day_tape = torch.empty_like(gWx)
        dpre_tape = torch.empty_like(gWx)
        param_acc = [torch.zeros(H, device=Wx.device, dtype=torch.float32) for _ in range(16)]
        scalar_acc = [torch.zeros((), device=Wx.device, dtype=torch.float32) for _ in range(3)]

        for t in range(T - 1, -1, -1):
            y = U[:, t, :H].float()
            b_prev = initial_b.float() if t == 0 else U[:, t - 1, H:].float()
            e_prev = e_tape[:, t].float()
            c_prev = initial_c.float() if t == 0 else c_tape[:, t - 1].float()
            gate = s_tape[:, t].float()
            nov = novelty[:, t].float()
            p = plasticity[:, t].float()
            ay = expression[:, t].float()
            c_t = c_tape[:, t].float()

            gb_new = gU[:, t, H:] + gbn
            gc_new = gcn + gc_out[:, t]
            step_outputs = _run_backward_step(
                y, b_prev, e_prev, c_prev, gate, nov, p, ay, c_t,
                gU[:, t, :H], gb_new, gen, gc_new,
                gnov[:, t], gp_out[:, t], gay_out[:, t], gwrite[:, t], ge_out[:, t],
                recurrent_backward,
                lk, lr,
                ks, ky, kb, kn, kbias,
                rs, ry, rb, rn, rbias,
                decay, kc, rc, alpha, lam, sig_lam, suppress,
                *param_acc, *scalar_acc,
            )
            gpre, dpre, d_ay, gbn, gen, gcn = step_outputs[:6]
            gWx[:, t] = gpre
            gWgx[:, t] = dpre
            day_tape[:, t] = d_ay
            dpre_tape[:, t] = dpre
            param_acc = list(step_outputs[6:22])
            scalar_acc = list(step_outputs[22:25])

        y_flat = U[:, :, :H].reshape(B * T, H).to(Wx.dtype)
        grad_recurrent = (
            torch.cat((dpre_tape, day_tape), dim=-1)
            .reshape(B * T, 2 * H)
            .to(Wx.dtype)
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
        initial_state = (
            torch.zeros((B, H), device=Wx.device, dtype=Wx.dtype),
            torch.ones((B, H), device=Wx.device, dtype=Wx.dtype),
            torch.zeros((B, H), device=Wx.device, dtype=Wx.dtype),
        )
    initial_b, initial_e, initial_c = (
        value.to(device=Wx.device, dtype=Wx.dtype).contiguous() for value in initial_state
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
_FUSED_MAX_H = int(os.environ.get("DABSN_FUSED_CORE_MAX_H", "256"))


def _fused_forward_available(Wx: Tensor) -> bool:
    if not Wx.is_cuda or os.environ.get("DABSN_FUSED_CORE_DISABLE", "0") == "1":
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return int(Wx.shape[-1]) <= _FUSED_MAX_H


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
        if used_triton:
            from .triton_runtime import dabsn_core_scan_batched_fused_forward

            tapes = dabsn_core_scan_batched_fused_forward(
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
            U, novelty, plasticity, expression, write,
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
        # Off the GPU/Triton path (CPU, large H, or Triton disabled) the tapes
        # came from the eager forward, so use the certified eager reverse. This
        # is also the escape hatch: DABSN_FUSED_CORE_TRITON_BWD=0 forces it.
        use_triton_bwd = (
            getattr(ctx, "used_triton", False)
            and os.environ.get("DABSN_FUSED_CORE_TRITON_BWD", "1") == "1"
        )
        if not use_triton_bwd:
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
            U, novelty, plasticity, expression, write,
            e_tape, c_tape, s_tape,
            final_b, final_e, final_c,
        ) = ctx.saved_tensors

        gU, gnov_out, gp_out, gay_out, gwrite_out = (
            torch.zeros_like(out) if grad is None else grad.contiguous()
            for out, grad in zip((U, novelty, plasticity, expression, write), grad_outputs)
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
            U, novelty, plasticity, expression, e_tape, c_tape, s_tape,
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
        initial_state = (
            torch.zeros((B, H), device=Wx.device, dtype=Wx.dtype),
            torch.ones((B, H), device=Wx.device, dtype=Wx.dtype),
            torch.zeros((B, H), device=Wx.device, dtype=Wx.dtype),
        )
    initial_b, initial_e, initial_c = (
        value.to(device=Wx.device, dtype=Wx.dtype).contiguous() for value in initial_state
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
