"""The central learning study's tasks: two standard long-range credit benchmarks.

Both come from Hochreiter and Schmidhuber (1997), the benchmarks introduced to test
credit over long time lags, and give one output at the final step.

adding          Experiment 4, in the standard form of Arjovsky, Shah and Bengio (2016):
                two input channels per step, a value drawn from U[0, 1] and a marker;
                exactly two markers, one at a uniform position in the first half of the
                sequence and one in the second half; the target is the sum of the two
                marked values. Predicting the constant 1 gives MSE 1/6, the standard
                baseline; scores are reported as MSE over 1/6 (normalized MSE).
temporal_order  Experiment 6a: symbols a, b, c, d as noise, B first and E last, and X or Y
                at one position in [0.1 T, 0.2 T] and one in [0.5 T, 0.6 T] (the original's
                positions 10-20 and 50-60 at length 100, scaled to the horizon T); the class
                is the ordered pair (XX, XY, YX, YY); chance 1/4.

Task channels occupy the first input coordinates of the model input; the rest are zero.
Every episode is a pure function of (task, seed, episode index, length) with its own
generator, so an episode is identical alone or inside any batch, and matched methods
see identical episodes.
"""
from __future__ import annotations

import hashlib

import torch
from torch.nn import functional as F

import harness
import models

TASKS = ("temporal_order", "adding")
ADDING_BASELINE_MSE = 1.0 / 6.0
SYMBOLS = ("a", "b", "c", "d", "B", "E", "X", "Y")


def channels(task):
    """Task input channels; the remaining model inputs are zero."""
    return 2 if task == "adding" else len(SYMBOLS)


def output_dim(task):
    return 4 if task == "temporal_order" else 1


def objective(task):
    if task not in TASKS:
        raise ValueError(f"Unknown central task: {task}")
    return "classification" if task == "temporal_order" else "regression"


def _generator(task, seed, index, length):
    key = f"{task}:{int(seed)}:{int(index)}:{int(length)}".encode()
    g = torch.Generator(device="cpu")
    g.manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF)
    return g


def _adding(g, length):
    values = torch.rand(length, generator=g)
    half = length // 2
    first = int(torch.randint(0, half, (1,), generator=g))
    second = int(torch.randint(half, length, (1,), generator=g))
    x = torch.zeros(length, 2)
    x[:, 0] = values
    x[first, 1] = 1.0; x[second, 1] = 1.0
    return x, torch.tensor([float(values[first] + values[second])])


def _temporal_order(g, length):
    codes = torch.randint(0, 4, (length,), generator=g)          # a, b, c, d
    codes[0] = 4; codes[-1] = 5                                    # B ... E
    lo1, hi1 = max(1, round(0.1 * length)), max(1, round(0.2 * length))
    lo2, hi2 = round(0.5 * length), round(0.6 * length)
    t1 = int(torch.randint(lo1, hi1 + 1, (1,), generator=g))
    t2 = int(torch.randint(lo2, hi2 + 1, (1,), generator=g))
    first = int(torch.randint(0, 2, (1,), generator=g)); second = int(torch.randint(0, 2, (1,), generator=g))
    codes[t1] = 6 + first; codes[t2] = 6 + second                  # X or Y
    return F.one_hot(codes, len(SYMBOLS)).float(), torch.tensor(2 * first + second)


def delayed_batch(task, seed, start, length, batch_size, inputs, device):
    """(x [B, T, inputs], targets, episode identities); the same contract as the study expects."""
    if task not in TASKS:
        raise ValueError(f"Unknown central task: {task}")
    xs = []; ys = []; ids = []
    for lane in range(batch_size):
        g = _generator(task, seed, start + lane, length)
        x, y = _adding(g, length) if task == "adding" else _temporal_order(g, length)
        padded = torch.zeros(length, inputs)
        padded[:, :x.shape[1]] = x
        xs.append(padded); ys.append(y); ids.append((seed, start + lane))
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device) if task == "adding" else torch.stack(ys).long().to(device)
    return x, y, ids


@torch.no_grad()
def delayed_evaluate(model, task, seed, length, count, batch_size, inputs, chunk, device, start_index=0,
                     input_policy="full"):
    if count % batch_size:
        raise ValueError("Evaluation count must divide batch")
    if input_policy not in ("full", "final_chunk_only"):
        raise ValueError("Unknown evaluation input policy")
    summed = 0.0; correct = 0
    for start in range(0, count, batch_size):
        x, y, _ = delayed_batch(task, seed, start_index + start, length, batch_size, inputs, device)
        if input_policy == "final_chunk_only":
            x = x.clone(); x[:, :-chunk] = 0
        logits = models.delayed_predict(model, x, chunk).float()
        harness.finite(logits, "nonfinite evaluation output")
        if objective(task) == "classification":
            summed += float(F.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(-1) == y).sum())
        else:
            summed += float(F.mse_loss(logits, y.float(), reduction="sum"))
    if objective(task) == "classification":
        return {"examples": count, "loss": summed / count, "correct": correct, "accuracy": correct / count}
    mse = summed / count
    return {"examples": count, "loss": mse, "mse": mse, "normalized_mse": mse / ADDING_BASELINE_MSE}
