"""Native small-state eligibility scans, without graph compilation.

The scan retains the three-state, same-channel Jacobian of the supplied
hybrid eligibility rule. Large matrix traces are contracted separately by
BLAS; they are never loaded or advanced here.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def core_trace_scan_reference(core, tape, incoming, eligibility):
    from ..temporal_credit import _core_trace_scan_reference
    return _core_trace_scan_reference(core, tape, incoming, eligibility)


if triton is not None:
    @triton.jit
    def _tanh(x):
        return 2.0 * tl.sigmoid(2.0 * x) - 1.0

    @triton.jit
    def _softplus(x):
        return tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))

    @triton.jit
    def _param(P, s, S: tl.constexpr, index: tl.constexpr):
        return tl.load(P + index * S + s, s < S, other=0)[:, None]

    @triton.jit
    def _scan(
        Budget, Energy, Saturation, Y, Expression, GatePre, Incoming,
        Params, Globals, ADiag, UgDiag, Local, Global, NextLocal, NextGlobal,
        LocalGrad, GlobalGrad, Jbb,
        S: tl.constexpr, T, BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0)
        s = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        d = tl.arange(0, 32)[None, :]
        valid = s[:, None] < S
        local_mask = valid & (d >= 3) & (d < 19)
        global_mask = valid & (d >= 19) & (d < 23)
        local_base = ((b * S + s[:, None]) * 3) * 16 + d - 3
        global_base = ((b * S + s[:, None]) * 3) * 4 + d - 19
        eb = tl.load(Local + local_base, local_mask, other=0) + tl.load(Global + global_base, global_mask, other=0)
        ee = tl.load(Local + local_base + 16, local_mask, other=0) + tl.load(Global + global_base + 4, global_mask, other=0)
        es = tl.load(Local + local_base + 32, local_mask, other=0) + tl.load(Global + global_base + 8, global_mask, other=0)
        grad = tl.full((BLOCK, 32), 0, tl.float32)
        db = (d == 0).to(tl.float32)
        de = (d == 1).to(tl.float32)
        ds = (d == 2).to(tl.float32)
        a_diag = tl.load(ADiag + s, s < S, other=0)[:, None]
        ug_diag = tl.load(UgDiag + s, s < S, other=0)[:, None]
        kappa = _param(Params, s, S, 2)
        recover = tl.sigmoid(_param(Params, s, S, 3))
        ks = _param(Params, s, S, 4)
        ky = _param(Params, s, S, 5)
        kb = _param(Params, s, S, 6)
        kn = _param(Params, s, S, 7)
        kbias = _param(Params, s, S, 8)
        rs = _param(Params, s, S, 9)
        ry = _param(Params, s, S, 10)
        rb = _param(Params, s, S, 11)
        rn = _param(Params, s, S, 12)
        rbias = _param(Params, s, S, 13)
        ksat = _param(Params, s, S, 14)
        rsat = _param(Params, s, S, 15)
        alpha = tl.sigmoid(tl.load(Globals))
        loglam = tl.load(Globals + 1)
        lam = _softplus(loglam)
        decay = tl.sigmoid(tl.load(Globals + 2))
        suppress = tl.sigmoid(tl.load(Globals + 3))
        dalpha = alpha * (1 - alpha) * (d == 19)
        dlam = tl.sigmoid(loglam) * (d == 20)
        ddecay = decay * (1 - decay) * (d == 21)
        dsuppress = suppress * (1 - suppress) * (d == 22)
        for t in range(T):
            offset = (b * T + t) * S + s
            budget = tl.load(Budget + offset, s < S, other=0)[:, None]
            energy = tl.load(Energy + offset, s < S, other=0)[:, None]
            saturation = tl.load(Saturation + offset, s < S, other=0)[:, None]
            y = tl.load(Y + offset, s < S, other=0)[:, None]
            expression = tl.load(Expression + offset, s < S, other=0)[:, None]
            pre = tl.load(GatePre + offset, s < S, other=0)[:, None]
            ib = tl.load(Incoming + offset * 3, s < S, other=0)[:, None]
            ie = tl.load(Incoming + offset * 3 + 1, s < S, other=0)[:, None]
            iz = tl.load(Incoming + offset * 3 + 2, s < S, other=0)[:, None]
            grad += ib * eb + ie * ee + iz * es
            dy = (1 - y * y) * db
            dexpression = dy * a_diag
            gate = tl.minimum(tl.maximum((pre + 3) / 6, 0), 1)
            dgate = ((pre > -3) & (pre < 3)).to(tl.float32) / 6 * (dy * ug_diag + (d == 3))
            delta = expression - budget
            novelty = _tanh(tl.abs(delta))
            sign = tl.where(delta > 0, 1.0, tl.where(delta < 0, -1.0, 0.0))
            dnovelty = (1 - novelty * novelty) * sign * (dexpression - db)
            next_sat = decay * saturation + (1 - decay) * novelty * (1 - energy)
            dnext_sat = decay * ds + (1 - decay) * ((1 - energy) * dnovelty - novelty * de) + (saturation - novelty * (1 - energy)) * ddecay
            neff = novelty * (1 - suppress * next_sat)
            dneff = dnovelty * (1 - suppress * next_sat) - novelty * (suppress * dnext_sat + next_sat * dsuppress)
            tb = _tanh(budget)
            dtb = (1 - tb * tb) * db
            k = ks * gate + ky * y + kb * tb + kn * neff + kbias + ksat * next_sat
            dk = ks * dgate + gate * (d == 7) + ky * dy + y * (d == 8) + kb * dtb + tb * (d == 9) + kn * dneff + neff * (d == 10) + (d == 11) + ksat * dnext_sat + next_sat * (d == 17)
            r = rs * gate + ry * y + rb * tb + rn * neff + rbias + rsat * next_sat
            dr = rs * dgate + gate * (d == 12) + ry * dy + y * (d == 13) + rb * dtb + tb * (d == 14) + rn * dneff + neff * (d == 15) + (d == 16) + rsat * dnext_sat + next_sat * (d == 18)
            tk = _tanh(k)
            tr = _tanh(r)
            ek = tl.exp(.5 * tk)
            er = tl.exp(.5 * tr)
            cost = _softplus(kappa) * ek
            dcost = tl.sigmoid(kappa) * (d == 5) * ek + cost * .5 * (1 - tk * tk) * dk
            raw_recovery = recover * er
            recovery = tl.minimum(tl.maximum(raw_recovery, 0), 1)
            drecovery = (recover * (1 - recover) * (d == 6) * er + raw_recovery * .5 * (1 - tr * tr) * dr) * ((raw_recovery > 0) & (raw_recovery < 1))
            plasticity = gate * energy
            dplasticity = gate * de + energy * dgate
            dnext_b = (1 - alpha) * db - budget * dalpha + (d == 4) + lam * (dplasticity * expression + plasticity * dexpression) + plasticity * expression * dlam
            raw_energy = energy + recovery * (1 - energy) - cost * plasticity
            dnext_e = (de + drecovery * (1 - energy) - recovery * de - dcost * plasticity - cost * dplasticity) * ((raw_energy > 0) & (raw_energy < 1))
            jb0 = tl.sum(tl.where(d == 0, dnext_b, 0), 1)[:, None]
            jb1 = tl.sum(tl.where(d == 1, dnext_b, 0), 1)[:, None]
            jb2 = tl.sum(tl.where(d == 2, dnext_b, 0), 1)[:, None]
            je0 = tl.sum(tl.where(d == 0, dnext_e, 0), 1)[:, None]
            je1 = tl.sum(tl.where(d == 1, dnext_e, 0), 1)[:, None]
            je2 = tl.sum(tl.where(d == 2, dnext_e, 0), 1)[:, None]
            js0 = tl.sum(tl.where(d == 0, dnext_sat, 0), 1)[:, None]
            js1 = tl.sum(tl.where(d == 1, dnext_sat, 0), 1)[:, None]
            js2 = tl.sum(tl.where(d == 2, dnext_sat, 0), 1)[:, None]
            nb = jb0 * eb + jb1 * ee + jb2 * es + dnext_b
            ne = je0 * eb + je1 * ee + je2 * es + dnext_e
            ns = js0 * eb + js1 * ee + js2 * es + dnext_sat
            eb, ee, es = nb, ne, ns
            tl.store(Jbb + offset, tl.sum(tl.where(d == 0, dnext_b, 0), 1), s < S)
        tl.store(NextLocal + local_base, eb, local_mask)
        tl.store(NextLocal + local_base + 16, ee, local_mask)
        tl.store(NextLocal + local_base + 32, es, local_mask)
        tl.store(NextGlobal + global_base, eb, global_mask)
        tl.store(NextGlobal + global_base + 4, ee, global_mask)
        tl.store(NextGlobal + global_base + 8, es, global_mask)
        tl.store(LocalGrad + (b * S + s[:, None]) * 16 + d - 3, grad, local_mask)
        tl.store(GlobalGrad + (b * S + s[:, None]) * 4 + d - 19, grad, global_mask)

    @triton.jit
    def _weights(Jbb, Incoming, Boundary, Multiplier, Future, Suffix,
                 S: tl.constexpr, T, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        s = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        carry = tl.full((BLOCK,), 0, tl.float32)
        product = tl.full((BLOCK,), 1, tl.float32)
        for reverse_t in range(T):
            t = T - 1 - reverse_t
            offset = (b * T + t) * S + s
            tl.store(Future + offset, carry, s < S)
            tl.store(Suffix + offset, product, s < S)
            j = tl.load(Jbb + offset, s < S, other=0)
            incoming = tl.load(Incoming + offset, s < S, other=0)
            carry = incoming + j * carry
            product *= j
        tl.store(Boundary + b * S + s, carry, s < S)
        tl.store(Multiplier + b * S + s, product, s < S)


def _require_cuda_fp32(value, name):
    if triton is None or value.device.type != 'cuda':
        raise RuntimeError(f'{name} requires CUDA and Triton')
    if value.dtype != torch.float32:
        raise TypeError(f'{name} requires FP32 eligibility arithmetic')


def core_trace_scan(core, tape, incoming, eligibility):
    """Return final local/global traces, their gradients, and budget Jacobians."""
    _require_cuda_fp32(eligibility.local, 'core_trace_scan')
    batch, steps, state, _ = incoming.shape
    if incoming.shape != (tape.batch_size, tape.inputs.shape[1], tape.state_dim, 3):
        raise ValueError('incoming must have shape [batch, time, state, 3]')
    names = ('beta', 'log_kappa', 'logit_recover', 'k_s', 'k_y', 'k_b',
             'k_n', 'k_bias', 'r_s', 'r_y', 'r_b', 'r_n', 'r_bias',
             'k_saturation', 'r_saturation')
    params = torch.stack([core.Wg.bias.detach().float().reshape(-1)] +
                         [getattr(core, name).detach().float().reshape(-1) for name in names])
    globals_ = torch.stack([getattr(core, name).detach().float().reshape(()) for name in
                           ('logit_alpha', 'log_lambda', 'logit_saturation_decay',
                            'logit_saturation_suppress')])
    next_local = torch.empty_like(eligibility.local)
    next_global = torch.empty_like(eligibility.global_)
    local_grad = torch.empty((batch, state, 16), device=incoming.device, dtype=torch.float32)
    global_grad = torch.empty((batch, state, 4), device=incoming.device, dtype=torch.float32)
    jbb = torch.empty((batch, steps, state), device=incoming.device, dtype=torch.float32)
    tapes = [getattr(tape, name).float().contiguous() for name in
             ('budget', 'energy', 'saturation', 'y', 'expression', 'gate_preactivation')]
    _scan[(batch, triton.cdiv(state, 4))](
        *tapes, incoming.float().contiguous(), params, globals_,
        core.A.weight.detach().diagonal().float().contiguous(),
        core.Ug.weight.detach().diagonal().float().contiguous(),
        eligibility.local.contiguous(), eligibility.global_.contiguous(),
        next_local, next_global, local_grad, global_grad, jbb,
        state, steps, 4, num_warps=4, enable_fp_fusion=False,
    )
    return next_local, next_global, local_grad.sum(0), global_grad.sum((0, 1)), jbb


def budget_trace_weights(jbb, incoming_budget):
    """Reverse scalar scan for exact chunk contractions of matrix traces."""
    _require_cuda_fp32(jbb, 'budget_trace_weights')
    if jbb.shape != incoming_budget.shape or jbb.ndim != 3:
        raise ValueError('budget weights require matching [batch, time, state] tensors')
    batch, steps, state = jbb.shape
    boundary = torch.empty((batch, state), device=jbb.device, dtype=jbb.dtype)
    multiplier = torch.empty_like(boundary)
    future = torch.empty_like(jbb, memory_format=torch.contiguous_format)
    suffix = torch.empty_like(future)
    _weights[(batch, triton.cdiv(state, 128))](
        jbb.contiguous(), incoming_budget.float().contiguous(), boundary,
        multiplier, future, suffix, state, steps, 128,
        num_warps=4, enable_fp_fusion=False,
    )
    return boundary, multiplier, future, suffix
