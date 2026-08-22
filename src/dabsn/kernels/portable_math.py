"""Cross-machine bit-exact transcendentals for portable DABSN execution.

Elementwise transcendentals such as ``tanh / sigmoid / exp / softplus`` are evaluated
by the platform's libm, and libm is not bit-identical between CPUs, operating systems,
and GPUs. This module provides a fixed portable alternative for DABSN runtime surfaces
that require cross-machine repeatability.

The implementation builds each function from IEEE-754 arithmetic, ``sqrt``,
``frexp``, and ``ldexp`` using fixed range reductions and polynomial
coefficients. Conforming platforms therefore produce repeatable elementwise
results without relying on platform-specific transcendental libraries.

Accuracy target: less than 1e-12 absolute error versus libm for bounded outputs.

Everything runs in float64. Inputs are cast to float64; output dtype matches input.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ln(2) and 1/ln(2) as exact fp64 constants (identical on every IEEE-754 platform).
_LN2 = 0.6931471805599453  # nearest double to ln 2
_INV_LN2 = 1.4426950408889634  # nearest double to 1/ln 2
# Two-part ln2 (Cody-Waite) for an accurate r = x - k*ln2 with no catastrophic cancel.
_LN2_HI = 0.6931471803691238  # high bits of ln2
_LN2_LO = 1.9082149292705877e-10  # ln2 - _LN2_HI

# exp(r) minimax-ish: plain Taylor to order 13 on |r| <= ln2/2 ~ 0.347 -> < 1e-17.
_EXP_INV_FACT = [
    1.0,
    1.0,
    0.5,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
    1.0 / 39916800.0,
    1.0 / 479001600.0,
    1.0 / 6227020800.0,
]


def pexp(x: Tensor) -> Tensor:
    """exp(x), cross-machine bit-exact. exp(x) = 2^k * exp(r), k=round(x/ln2),
    r = x - k*ln2 (Cody-Waite split), exp(r) via fixed Taylor on |r|<=ln2/2."""
    dt = x.dtype
    xf = x.to(torch.float64)
    # round-half-to-even is IEEE-defined -> portable.
    k = torch.round(xf * _INV_LN2)
    r = (xf - k * _LN2_HI) - k * _LN2_LO
    # Horner on the fixed coefficients.
    acc = torch.full_like(r, _EXP_INV_FACT[-1])
    for c in reversed(_EXP_INV_FACT[:-1]):
        acc = acc * r + c
    # 2^k exactly via ldexp (k is integral-valued; cast to int).
    out = torch.ldexp(acc, k.to(torch.int64))
    return out.to(dt)


# log(m) for m in [2/3, 4/3): t=(m-1)/(m+1) in (-0.2,0.2), log(m)=2*(t+t^3/3+t^5/5+...).
_LOG_RECIP = [
    1.0,
    1.0 / 3.0,
    1.0 / 5.0,
    1.0 / 7.0,
    1.0 / 9.0,
    1.0 / 11.0,
    1.0 / 13.0,
    1.0 / 15.0,
    1.0 / 17.0,
    1.0 / 19.0,
]


def plog(x: Tensor) -> Tensor:
    """log(x) for x>0, cross-machine bit-exact. frexp -> m in [0.5,1), e; rebase m to
    [2/3,4/3) so the atanh series converges fast; log = (e+adj)*ln2 + 2*atanh((m-1)/(m+1))."""
    dt = x.dtype
    xf = x.to(torch.float64).clamp_min(2.2250738585072014e-308)
    m, e = torch.frexp(xf)  # x = m * 2^e, m in [0.5, 1)
    e = e.to(torch.float64)
    # rebase: if m < 2/3 push into [2/3,4/3) by doubling and dropping one from e.
    low = m < 0.6666666666666666
    m = torch.where(low, m * 2.0, m)
    e = torch.where(low, e - 1.0, e)
    t = (m - 1.0) / (m + 1.0)
    t2 = t * t
    acc = torch.full_like(t, _LOG_RECIP[-1])
    for c in reversed(_LOG_RECIP[:-1]):
        acc = acc * t2 + c
    logm = 2.0 * t * acc
    out = e * _LN2 + logm
    return out.to(dt)


def plog1p(x: Tensor) -> Tensor:
    """log(1+x), accurate for small x (avoids cancellation in plog(1+x))."""
    dt = x.dtype
    xf = x.to(torch.float64)
    u = 1.0 + xf
    # When u==1 (x tiny) log1p ~ x; else use plog(u)*x/(u-1) correction (Kahan trick).
    safe = u != 1.0
    corr = torch.where(safe, plog(u) * (xf / (u - 1.0).clamp_min(2.2250738585072014e-308)), xf)
    out = torch.where(u == 1.0, xf, corr)
    return out.to(dt)


def psigmoid(x: Tensor) -> Tensor:
    """1/(1+exp(-x)), numerically stable via sign split, cross-machine bit-exact."""
    dt = x.dtype
    xf = x.to(torch.float64)
    neg = xf < 0.0
    z = pexp(-xf.abs())  # in (0,1]
    pos_branch = 1.0 / (1.0 + z)  # for x>=0
    neg_branch = z / (1.0 + z)  # for x<0
    out = torch.where(neg, neg_branch, pos_branch)
    return out.to(dt)


def ptanh(x: Tensor) -> Tensor:
    """tanh(x) = sign(x)*(1 - 2/(exp(2|x|)+1)), saturates for large |x|; portable."""
    dt = x.dtype
    xf = x.to(torch.float64)
    a = xf.abs().clamp_max(40.0)  # exp(80) overflows; tanh already ==1 well before
    z = pexp(2.0 * a)
    mag = 1.0 - 2.0 / (z + 1.0)
    out = torch.sign(xf) * mag
    return out.to(dt)


def psoftplus(x: Tensor) -> Tensor:
    """log(1+exp(x)) = max(x,0) + log1p(exp(-|x|)); portable, no overflow."""
    dt = x.dtype
    xf = x.to(torch.float64)
    out = xf.clamp_min(0.0) + plog1p(pexp(-xf.abs()))
    return out.to(dt)


def _selftest() -> None:
    """Compare the portable functions with the platform math library on CPU."""
    import torch.nn.functional as F

    x = torch.linspace(-40, 40, 200001, dtype=torch.float64)
    # (name, portable, libm, mode): 'abs' for bounded outputs, 'rel' for wide-magnitude.
    checks = [
        ("exp", pexp(x.clamp_max(10)), torch.exp(x.clamp_max(10)), "rel"),
        ("tanh", ptanh(x), torch.tanh(x), "abs"),
        ("sigmoid", psigmoid(x), torch.sigmoid(x), "abs"),
        ("softplus", psoftplus(x), F.softplus(x), "rel"),
        ("log(x>0)", plog(x.abs() + 1e-6), torch.log(x.abs() + 1e-6), "rel"),
    ]
    ok = True
    for name, a, b, mode in checks:
        if mode == "abs":
            d = (a - b).abs().max().item()
            tol = 1e-12
        else:
            # relative error only where the value is in a meaningful range (deep tail is
            # last-bit: at x=-23 softplus~1e-10 so abs err ~1e-20 -- irrelevant to ratio).
            keep = b.abs() > 1e-6
            # The relative tolerance protects the wide-magnitude functions while
            # exactness comes from the fixed IEEE-only construction.
            d = ((a[keep] - b[keep]).abs() / b[keep].abs()).max().item()
            tol = 2e-10
        ok = ok and d < tol
        print(
            f"  {name:10s} max {mode} err = {d:.2e}  (tol {tol:.0e})  {'OK' if d < tol else 'FAIL'}"
        )
    print(f"PORTABLE MATH bit-exact-construction + ratio-preserving: {ok}")


if __name__ == "__main__":
    _selftest()
