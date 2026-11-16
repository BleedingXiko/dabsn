"""The native action: the core's credit normalized by the core's own E, paced by E.

One definition, shared by every study that applies it (conjunction, stream-normalized):

    s_theta = column RMS of the core's E for core parameter theta (Eq. response-rms)
    eta_t   = mean of s_theta over the core's parameters
    direction  d_theta = g_theta * eta_t / max(s_theta, floor)     (credit shaped by E)
    pace       the core's action is d rescaled to coordinate RMS eta_t: its size is set by
               E's amplitude, not by the size of the error signal. The core's credit is
               K^T l, and the column normalization removes K's scales but passes |l| through,
               so without the pace a spike in l at a learning transition becomes a step of
               the same size (measured: ~1000x the median step, after which training is dead).
    everything else  unchanged: the read and the readout keep their own online credit;
                     the core is not responsible for them and its statistics never scale them

floor = max(relative_floor * eta_t, absolute_floor). The caller then applies plain SGD at one
nominal rate, so the core's per-coordinate step has RMS rate * eta_t. No optimizer state is
read or kept, nothing is clipped.
"""
from __future__ import annotations

import torch

import credit
import harness


def normalize(core, parameters, eligibility, relative_floor=0.0, absolute_floor=0.0):
    """Shape the core's gradients by E and pace them by E, in place; return the statistics."""
    scales, mean = credit.column_scales(core, eligibility)
    floor = max(float(relative_floor) * float(mean), float(absolute_floor))
    with torch.no_grad():
        shaped = []
        for p in parameters:
            if p.grad is None:
                continue
            if p in scales:
                p.grad.mul_(mean / scales[p].clamp_min(floor))
                shaped.append(p)
            harness.finite(p.grad, "nonfinite native action")
        squares = sum(float(p.grad.double().square().sum()) for p in shaped)
        count = sum(p.grad.numel() for p in shaped)
        direction_rms = (squares / count) ** 0.5 if count else 0.0
        if direction_rms > 0:
            for p in shaped:
                p.grad.mul_(float(mean) / direction_rms)
    return {"global_response_scale": float(mean), "floor": floor, "direction_rms_before_pace": direction_rms,
            "common_coordinate_denominator": 3 * eligibility.batch_size * eligibility.state_dim}
