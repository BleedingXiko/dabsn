"""The central study's standard task generators (Hochreiter and Schmidhuber, 1997)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studies import delayed_tasks as dt  # noqa: E402


def test_adding_definition():
    x, y, _ = dt.delayed_batch("adding", 3, 0, 200, 64, 64, "cpu")
    assert x.shape == (64, 200, 64) and y.shape == (64, 1)
    assert torch.count_nonzero(x[..., 2:]) == 0
    markers = x[..., 1]
    assert torch.equal(markers.sum(1), torch.full((64,), 2.0))
    assert torch.equal(markers[:, :100].sum(1), torch.ones(64))
    total = (x[..., 0] * markers).sum(1, keepdim=True)
    torch.testing.assert_close(total, y)
    assert float(x[..., 0].min()) >= 0 and float(x[..., 0].max()) <= 1


def test_adding_baseline_is_one_sixth():
    _, y, _ = dt.delayed_batch("adding", 5, 0, 64, 4096, 2, "cpu")
    assert abs(float((y - 1).square().mean()) - 1 / 6) < 0.01


def test_temporal_order_definition():
    for length in (128, 512, 2048):
        x, y, _ = dt.delayed_batch("temporal_order", 4, 0, length, 32, 64, "cpu")
        codes = x[..., :8].argmax(-1)
        assert torch.equal(x[..., :8].sum(-1), torch.ones(32, length))
        assert torch.count_nonzero(x[..., 8:]) == 0
        assert (codes[:, 0] == 4).all() and (codes[:, -1] == 5).all()
        for lane in range(32):
            where = torch.nonzero(codes[lane] >= 6).flatten().tolist()
            assert len(where) == 2
            t1, t2 = where
            assert round(0.1 * length) <= t1 <= round(0.2 * length)
            assert round(0.5 * length) <= t2 <= round(0.6 * length)
            label = 2 * int(codes[lane, t1] - 6) + int(codes[lane, t2] - 6)
            assert label == int(y[lane])


def test_classes_balanced():
    _, y, _ = dt.delayed_batch("temporal_order", 9, 0, 128, 4000, 8, "cpu")
    counts = torch.bincount(y, minlength=4).float() / 4000
    assert (counts - 0.25).abs().max() < 0.03


def test_episode_purity_and_stability():
    for task in dt.TASKS:
        whole, yw, ids = dt.delayed_batch(task, 71, 10, 96, 6, 16, "cpu")
        for lane in range(6):
            alone, ya, _ = dt.delayed_batch(task, 71, 10 + lane, 96, 1, 16, "cpu")
            assert torch.equal(whole[lane:lane + 1], alone) and torch.equal(yw[lane:lane + 1], ya)
        assert ids[0] == (71, 10)
        other, _, _ = dt.delayed_batch(task, 72, 10, 96, 6, 16, "cpu")
        assert not torch.equal(other, whole)
