"""Omitted cross-channel influence of the B-channel matrix trace, exactly.

Scalar DABSN construction with tanh(beta)=1/2. The B-channel trace carries the
budget-to-budget path of a recurrent-matrix sensitivity; the exact sensitivity
adds the budget-to-energy path. Their difference is the loss-weighted residual.
Two input signs give identical forward values and opposite historical credit.
"""
from fractions import Fraction as Q


def check():
    e, y = Q(19, 20), Q(1, 2)
    jbb = Q(1, 2) + e * Q(1, 2) * (1 - y * y)
    jbe = Q(1, 2) * y
    db, de = Q(1, 2), -Q(1, 40)
    online = jbb * db
    exact = online + jbe * de
    assert (jbb, online, exact, exact - online) == (Q(137, 160), Q(137, 320), Q(27, 64), -Q(1, 160))
    assert jbe * de == exact - online
    histories = [{"input": s, "boundary_b": "log(3)/2", "boundary_energy": str(e),
                  "trace": str(s * db), "online_credit": str(s * online),
                  "exact_credit": str(s * exact)} for s in (-1, 1)]
    return {"scalar": {"online": str(online), "full": str(exact), "difference": str(exact - online)},
            "indistinguishable_histories": histories,
            "scope": "exact arithmetic for the stated scalar recurrence; no framework execution"}
