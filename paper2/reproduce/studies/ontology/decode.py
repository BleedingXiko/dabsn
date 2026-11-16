"""Bounded, held-out probes of what the core learning state exposes.

The full terminal state-parameter Jacobian is materialized only for a tiny
three-channel core. Decoder outcomes measure accessibility under the declared
probe; they do not establish universal information inclusion or sufficiency.
"""
from __future__ import annotations

import json
import math

import reference
import torch
from dabsn.temporal_credit import CoreSignalGrads, CoreEligibilityState
from dabsn.temporal_credit import core_online_forward_reference
from dabsn.temporal_credit import core_online_replay_reference, initial_core_eligibility


TRAIN_CORES = (0, 1, 2)
TEST_CORES = (3, 4)
TRAIN_LENGTHS = (32, 64)
TEST_LENGTHS = (32, 64, 128)
EXAMPLES_PER_CORE_AND_LENGTH = 24
NOISE_STD = 0.1
RIDGE = 0.01
FEATURE_BUDGETS = (9, 48)


def episode(core_seed, index, length):
    generator = torch.Generator(device="cpu").manual_seed(
        70001 + 1009 * core_seed + 97 * index + length)
    x = torch.randn(1, length, 2, generator=generator,
                    dtype=torch.float64) * NOISE_STD
    position = int(torch.randint(1, length - 1, (), generator=generator))
    sign = 1 if bool(torch.randint(0, 2, (), generator=generator)) else -1
    x[0, position, 0] += sign
    return x, position / (length - 1), sign


def state_and_eligibility(core, x):
    state = core.initial_state(1, device="cpu", dtype=x.dtype)
    eligibility = initial_core_eligibility(core, 1, device="cpu", dtype=x.dtype)
    forward, tape = core_online_forward_reference(core, x, state, eligibility)
    eligibility, _, _ = core_online_replay_reference(
        core, tape, CoreSignalGrads(), eligibility)
    return forward.final_state, eligibility


def terminal_jacobian(core, x):
    initial = core.initial_state(1, device="cpu", dtype=x.dtype)
    _, state = core._reference_forward_from_state(x, initial_state=initial,
                                                  return_final_state=True)
    flattened = torch.stack(state, dim=-1).reshape(-1)
    parameters = tuple(core.parameters())
    rows = []
    for index in range(flattened.numel()):
        gradients = torch.autograd.grad(flattened[index], parameters,
            retain_graph=index + 1 < flattened.numel(), allow_unused=True)
        rows.append(torch.cat([(value if value is not None else torch.zeros_like(parameter))
                               .reshape(-1) for value, parameter in zip(gradients, parameters)]))
    return torch.stack(rows).reshape(-1).detach()


def collect(core_seed, index, length):
    core = reference.core(2000 + core_seed, inputs=2, state=3, dtype=torch.float64)
    x, position, sign = episode(core_seed, index, length)
    state, eligibility = state_and_eligibility(core, x)
    features = {
        "state": torch.stack(state, dim=-1).reshape(-1).detach(),
        "local_E": eligibility.local.reshape(-1).detach(),
        "complete_E": torch.cat([value.reshape(-1) for value in
                                 eligibility.tensors()]).detach(),
        "terminal_H": terminal_jacobian(core, x),
    }
    return {"core_seed": core_seed, "index": index, "length": length,
            "event_position": position, "event_sign": sign,
            "features": features,
            "local_by_channel": eligibility.local[0].reshape(3, -1).detach(),
            "state_by_channel": torch.stack(state, dim=-1)[0].detach()}


def project(name, matrix, budget):
    width = matrix.shape[1]
    if name == "state":
        if width != 9:
            raise AssertionError("State probe dimension changed")
        return torch.nn.functional.pad(matrix, (0, budget - width)) if budget > width else matrix
    seed = {"local_E": 1401, "complete_E": 1402, "terminal_H": 1403}[name]
    generator = torch.Generator(device="cpu").manual_seed(seed + 100 * budget)
    projection = torch.randn(width, budget, generator=generator,
                             dtype=torch.float64) / math.sqrt(width)
    return matrix @ projection


def fit_ridge(x, y):
    mean_x = x.mean(0)
    scale_x = x.std(0, unbiased=False).clamp_min(1e-8)
    mean_y = y.mean(0)
    standardized = (x - mean_x) / scale_x
    gram = standardized.T @ standardized
    weights = torch.linalg.solve(
        gram + RIDGE * x.shape[0] * torch.eye(x.shape[1], dtype=x.dtype),
        standardized.T @ (y - mean_y))
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("Nonfinite probe fit")
    return mean_x, scale_x, mean_y, weights


def predict(fit, x):
    mean_x, scale_x, mean_y, weights = fit
    return ((x - mean_x) / scale_x) @ weights + mean_y


def r2(prediction, target, baseline):
    residual = float((prediction - target).square().sum())
    denominator = float((target - baseline).square().sum())
    return 1 - residual / denominator if denominator > 0 else None


def auc(scores, labels):
    positive = [float(score) for score, label in zip(scores, labels) if label > 0]
    negative = [float(score) for score, label in zip(scores, labels) if label < 0]
    if not positive or not negative:
        return None
    wins = sum((1. if a > b else 0.5 if a == b else 0.)
               for a in positive for b in negative)
    return wins / (len(positive) * len(negative))


def decode_event(train, test):
    result = []
    y_train = torch.tensor([[row["event_position"], row["event_sign"]]
                            for row in train], dtype=torch.float64)
    y_test = torch.tensor([[row["event_position"], row["event_sign"]]
                           for row in test], dtype=torch.float64)
    for name in ("state", "local_E", "complete_E", "terminal_H"):
        train_matrix = torch.stack([row["features"][name] for row in train])
        test_matrix = torch.stack([row["features"][name] for row in test])
        for budget in FEATURE_BUDGETS:
            x_train = project(name, train_matrix, budget)
            x_test = project(name, test_matrix, budget)
            fit = fit_ridge(x_train, y_train)
            prediction = predict(fit, x_test)
            for length in TEST_LENGTHS:
                mask = torch.tensor([row["length"] == length for row in test])
                p = prediction[mask]; y = y_test[mask]
                result.append({"representation": name, "feature_budget": budget,
                               "effective_feature_rank_at_most": min(train_matrix.shape[1], budget),
                               "test_length": length,
                               "held_out_cores": list(TEST_CORES),
                               "examples": int(mask.sum()),
                               "time_r2": r2(p[:, 0], y[:, 0], y_train[:, 0].mean()),
                               "sign_accuracy": float(((p[:, 1] >= 0) == (y[:, 1] > 0)).double().mean()),
                               "sign_auc": auc(p[:, 1].tolist(), y[:, 1].tolist()),
                               "training_examples": len(train),
                               "baseline_time_prediction": float(y_train[:, 0].mean())})
    return result


def reconstruction_fit(train):
    x = torch.cat([row["local_by_channel"] for row in train])
    y = torch.cat([row["state_by_channel"] for row in train])
    if x.shape[1] != 48 or y.shape[1] != 3:
        raise AssertionError("Local eligibility channel schema changed")
    return fit_ridge(x, y)


def reconstruction_metrics(fit, train, test):
    training_mean = torch.cat([row["state_by_channel"] for row in train]).mean(0)
    result = []
    for length in TEST_LENGTHS:
        selected = [row for row in test if row["length"] == length]
        x = torch.cat([row["local_by_channel"] for row in selected])
        y = torch.cat([row["state_by_channel"] for row in selected])
        p = predict(fit, x)
        result.append({"test_length": length, "held_out_cores": list(TEST_CORES),
                       "channel_examples": x.shape[0],
                       "component_r2": {name: r2(p[:, index], y[:, index],
                                               training_mean[index])
                                        for index, name in enumerate(("budget", "energy", "saturation"))},
                       "component_rmse": {name: float((p[:, index] - y[:, index])
                                                       .square().mean().sqrt())
                                          for index, name in enumerate(("budget", "energy", "saturation"))},
                       "constant_predictor": training_mean.tolist()})
    return result


def recursive_rollout(fit, core_seed, length):
    core = reference.core(2000 + core_seed, inputs=2, state=3, dtype=torch.float64)
    x, _, _ = episode(core_seed, 9000 + length, length)
    true_state = core.initial_state(1, dtype=x.dtype)
    approximate_state = core.initial_state(1, dtype=x.dtype)
    true_E = initial_core_eligibility(core, 1, device="cpu", dtype=x.dtype)
    approximate_E = initial_core_eligibility(core, 1, device="cpu", dtype=x.dtype)
    errors = []
    failure_step = None
    for step in range(length):
        true_forward, true_tape = core_online_forward_reference(
            core, x[:, step:step + 1], true_state, true_E)
        true_E, _, _ = core_online_replay_reference(
            core, true_tape, CoreSignalGrads(), true_E)
        true_state = true_forward.final_state
        predicted_forward, predicted_tape = core_online_forward_reference(
            core, x[:, step:step + 1], approximate_state, approximate_E)
        approximate_E, _, _ = core_online_replay_reference(
            core, predicted_tape, CoreSignalGrads(), approximate_E)
        predicted = predict(fit, approximate_E.local[0].reshape(3, -1))
        approximate_state = tuple(predicted[:, component].unsqueeze(0).detach()
                                  for component in range(3))
        truth = torch.stack(true_state, dim=-1)
        estimate = torch.stack(approximate_state, dim=-1)
        error = float(torch.linalg.vector_norm(estimate - truth))
        if not math.isfinite(error):
            errors.append(None)
            failure_step = step + 1
            break
        errors.append(error)
        if error > 100:
            failure_step = step + 1
            break
    finite_errors = [value for value in errors if value is not None]
    return {"core_seed": core_seed, "length": length,
            "steps_completed": len(errors), "failure_step": failure_step,
            "final_state_absolute_l2_error": errors[-1],
            "maximum_state_absolute_l2_error": max(finite_errors) if finite_errors else None,
            "checkpoint_errors": {str(step): errors[step - 1] for step in
                                  (1, 8, 32, 64, 128) if step <= len(errors)},
            "scope": "unclipped E-to-state linear decoder recursively replaces all three state channels"}


def run():
    train = [collect(core_seed, index, length)
             for core_seed in TRAIN_CORES for length in TRAIN_LENGTHS
             for index in range(EXAMPLES_PER_CORE_AND_LENGTH)]
    test = [collect(core_seed, index + 10000, length)
            for core_seed in TEST_CORES for length in TEST_LENGTHS
            for index in range(EXAMPLES_PER_CORE_AND_LENGTH)]
    fit = reconstruction_fit(train)
    return {"event_decoding": decode_event(train, test),
            "state_reconstruction": reconstruction_metrics(fit, train, test),
            "recursive_rollout": [recursive_rollout(fit, core_seed, length)
                                  for core_seed in TEST_CORES for length in TEST_LENGTHS]}
