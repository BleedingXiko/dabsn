"""Composition across detached boundaries: fixed-parameter credit maps are
affine, and affine maps compose independently of how a stream is chunked."""
from fractions import Fraction as Q


def check():
    maps = [(Q(2, 3), Q(1, 7)), (Q(-3, 5), Q(2, 9)), (Q(7, 8), Q(-1, 4))]

    def compose(second, first):
        a, b = second
        c, d = first
        return a * c, a * d + b

    left = compose(maps[2], compose(maps[1], maps[0]))
    right = compose(compose(maps[2], maps[1]), maps[0])
    assert left == right
    return {"affine_example": [str(v) for v in left],
            "scope": "exact composition identity; chunk-partition invariance of native credit is an oracle study"}
