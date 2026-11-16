"""Exact rational algebra shared by the proofs. Python standard library only."""
from fractions import Fraction as Q


class Poly:
    """Polynomial in two indeterminates (c, n) with rational coefficients."""

    def __init__(self, value=0):
        if isinstance(value, dict):
            self.terms = {k: Q(v) for k, v in value.items() if v}
        else:
            self.terms = {(0, 0): Q(value)} if value else {}

    def __add__(self, other):
        other = as_poly(other)
        terms = self.terms.copy()
        for key, value in other.terms.items():
            terms[key] = terms.get(key, Q(0)) + value
        return Poly(terms)

    __radd__ = __add__

    def __neg__(self):
        return Poly({k: -v for k, v in self.terms.items()})

    def __sub__(self, other):
        return self + -as_poly(other)

    def __rsub__(self, other):
        return as_poly(other) + -self

    def __mul__(self, other):
        other = as_poly(other)
        terms = {}
        for (a, b), x in self.terms.items():
            for (c, d), y in other.terms.items():
                key = (a + c, b + d)
                terms[key] = terms.get(key, Q(0)) + x * y
        return Poly(terms)

    __rmul__ = __mul__

    def __eq__(self, other):
        return self.terms == as_poly(other).terms


def as_poly(value):
    return value if isinstance(value, Poly) else Poly(value)


def transpose(a):
    return [list(column) for column in zip(*a)]


def multiply(a, b):
    bt = transpose(b)
    return [[sum((x * y for x, y in zip(row, column)), Q(0)) for column in bt] for row in a]


def add(a, b, sign=1):
    return [[x + sign * y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def eye(n, scale=Q(1)):
    return [[scale if i == j else Q(0) for j in range(n)] for i in range(n)]


def solve(a, b):
    """Dense Gauss-Jordan elimination with exact rational pivoting."""
    n = len(a)
    augmented = [list(row) + list(rhs) for row, rhs in zip(a, b)]
    for column in range(n):
        pivot = next(i for i in range(column, n) if augmented[i][column])
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [v / divisor for v in augmented[column]]
        for row in range(n):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [v - factor * w for v, w in zip(augmented[row], augmented[column])]
    return [row[n:] for row in augmented]
