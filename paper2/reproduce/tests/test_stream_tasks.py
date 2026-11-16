import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studies import stream_tasks  # noqa: E402


def test_memory_capacity_stream_is_deterministic_and_uniform():
    a = stream_tasks.memory_capacity_stream(100, 3, 4096)
    assert a == stream_tasks.memory_capacity_stream(100, 3, 4096)
    assert a != stream_tasks.memory_capacity_stream(100, 4, 4096)
    assert all(-1.0 <= v <= 1.0 for v in a) and abs(sum(a) / len(a)) < 0.05


def test_narma10_recurrence_and_bound():
    u, y, attempt = stream_tasks.narma10_stream(100, 0, 20000)
    assert len(u) == len(y) == 20000 and attempt >= 0
    assert all(abs(v) <= stream_tasks.NARMA_BOUND for v in y)
    uw = [0.0] * 10; yw = [0.0] * 10
    for t in range(200):
        uw = uw[1:] + [u[t]]
        following = 0.3 * yw[-1] + 0.05 * yw[-1] * sum(yw) + 1.5 * uw[0] * uw[-1] + 0.1
        assert following == y[t]
        yw = yw[1:] + [following]
