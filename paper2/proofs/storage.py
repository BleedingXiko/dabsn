"""History-independent storage of the B-channel eligibility, analytically.

For input = state = S and one batch lane: P = 4S^2 + 16S + 4 core parameters;
the B-channel eligibility holds n = 4S^2 + 60S scalars (one budget trace per
recurrent-matrix entry, three-state traces for local and shared controls),
against 3SP for the full state-parameter influence. The native oracle study
checks n against the framework's actual CoreEligibilityState allocation.
"""
from fractions import Fraction as Q


def check():
    counts = []
    for state in (16, 64, 256, 1024, 4096):
        p = 4 * state * state + 16 * state + 4
        n = 4 * state * state + 60 * state
        assert n == p + 44 * state - 4
        counts.append({"input": state, "state": state, "batch": 1, "parameters": p,
                       "eligibility_scalars": n, "full_influence_scalars": 3 * state * p,
                       "eligibility_mib_fp32": float(Q(4 * n, 2 ** 20)),
                       "full_influence_gib_fp32": float(Q(12 * state * p, 2 ** 30)),
                       "kind": "analytical_formula"})
    return {"analytical_counts": counts,
            "scope": "formula; native allocation is verified by the oracle study"}
