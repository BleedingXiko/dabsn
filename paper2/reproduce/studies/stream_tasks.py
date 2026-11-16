"""Standard stream benchmarks for the stream studies (standard library only).

Each lane is one uninterrupted stream with its own keyed random source; a stream is
a pure function of (benchmark, seed, lane, length) and never depends on a model.
"""
from __future__ import annotations

import hashlib
import math
import random


def stream_rng(benchmark, seed, lane, attempt=0):
    key = f'paper2-stream-v1/{benchmark}/{seed}/{lane}/{attempt}'.encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:16], 'big'))


def memory_capacity_stream(seed, lane, length):
    """Jaeger (2001) short-term memory capacity: u(t) ~ U[-1, 1] i.i.d.; the target for
    delay k at step t is u(t - k). Returns the input sequence."""
    if not all(isinstance(v, int) and v >= 0 for v in (seed, lane, length)):
        raise ValueError('seed, lane and length must be nonnegative integers')
    rng = stream_rng('memory_capacity', seed, lane)
    return [rng.uniform(-1.0, 1.0) for _ in range(length)]


NARMA_ORDER = 10
NARMA_BOUND = 1.0


def narma10_stream(seed, lane, length):
    """NARMA-10 (Atiya and Parlos, 2000): u(t) ~ U[0, 0.5] i.i.d. and
    y(t+1) = 0.3 y(t) + 0.05 y(t) sum_{i=0..9} y(t-i) + 1.5 u(t-9) u(t) + 0.1.
    The system can diverge on long streams; as in standard practice a divergent
    series is regenerated: the lane takes the first attempt whose series stays
    within |y| <= NARMA_BOUND for the whole stream. The rule is fixed in advance
    and independent of any model. Returns (inputs u(t), targets y(t+1), attempt)."""
    if not all(isinstance(v, int) and v >= 0 for v in (seed, lane, length)):
        raise ValueError('seed, lane and length must be nonnegative integers')
    for attempt in range(1000):
        rng = stream_rng('narma10', seed, lane, attempt)
        u_window = [0.0] * NARMA_ORDER; y_window = [0.0] * NARMA_ORDER
        inputs = []; targets = []
        for _ in range(length):
            u_window = u_window[1:] + [rng.uniform(0.0, 0.5)]
            y = y_window[-1]
            following = 0.3 * y + 0.05 * y * sum(y_window) + 1.5 * u_window[0] * u_window[-1] + 0.1
            if not math.isfinite(following) or abs(following) > NARMA_BOUND:
                break
            y_window = y_window[1:] + [following]
            inputs.append(u_window[-1]); targets.append(following)
        else:
            return inputs, targets, attempt
    raise RuntimeError('No bounded NARMA-10 series within 1000 attempts')
