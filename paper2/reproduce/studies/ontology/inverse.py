"""Bounded inverse-credit readout versus true and replayed state response.

The target is scaled after solving so every intervention has the same small
parameter norm. This is a local descriptive test, not a controller claim.
"""
from __future__ import annotations

import copy
import json
import math

import reference
from credit import embedded_operator
import torch
from dabsn.temporal_credit import CoreSignalGrads
from dabsn.temporal_credit import core_online_forward_reference
from dabsn.temporal_credit import core_online_replay_reference, initial_core_eligibility


SEEDS = (0, 1, 2, 3, 4)
LENGTHS = (8, 32, 64)
PARAMETER_STEP_NORM = 0.001
RIDGE_FRACTION = 0.01


def state_vector(core, x, initial):
    _, state = core._reference_forward_from_state(
        x, initial_state=initial, return_final_state=True)
    return torch.stack(state, dim=-1).reshape(-1)


def true_jacobian(core, x, initial):
    state = state_vector(core, x, initial)
    parameters = tuple(core.parameters())
    rows = []
    for coordinate in range(state.numel()):
        derivatives = torch.autograd.grad(state[coordinate], parameters,
            retain_graph=coordinate + 1 < state.numel(), allow_unused=True)
        rows.append(torch.cat([(gradient if gradient is not None else
            torch.zeros_like(parameter)).reshape(-1)
            for gradient, parameter in zip(derivatives, parameters)]))
    return torch.stack(rows).detach()


def one(seed, length):
    core = reference.core(5000 + seed, inputs=2, state=3, dtype=torch.float64)
    initial = tuple(value.detach() for value in core.initial_state(1, dtype=torch.float64))
    generator = torch.Generator(device="cpu").manual_seed(60001 + seed * 1009 + length)
    x = torch.randn(1, length, 2, generator=generator, dtype=torch.float64) * .1
    x[:, length // 4, 0] += 1.
    eligibility = initial_core_eligibility(core, 1, device="cpu", dtype=torch.float64)
    forward, tape = core_online_forward_reference(core, x, initial, eligibility)
    eligibility, _, _ = core_online_replay_reference(
        core, tape, CoreSignalGrads(), eligibility)
    K, _, _ = embedded_operator(core, eligibility)
    H = true_jacobian(core, x, initial)
    if K.shape != H.shape or K.shape[0] != 9:
        raise AssertionError("Inverse-probe operators have different coordinates")
    requested = torch.randn(9, generator=generator, dtype=torch.float64)
    requested /= torch.linalg.vector_norm(requested)
    gram = K @ K.T
    ridge = max(1e-10, RIDGE_FRACTION * float(torch.trace(gram)) / K.shape[0])
    raw_step = -K.T @ torch.linalg.solve(gram + ridge * torch.eye(9,
        dtype=torch.float64), requested)
    raw_norm = float(torch.linalg.vector_norm(raw_step))
    if not math.isfinite(raw_norm) or raw_norm <= 0:
        raise FloatingPointError("Inverse readout produced zero or nonfinite step")
    step = raw_step * (PARAMETER_STEP_NORM / raw_norm)
    target = requested * (PARAMETER_STEP_NORM / raw_norm)
    predicted_K = K @ step
    predicted_H = H @ step
    before = state_vector(core, x, initial).detach()
    updated = copy.deepcopy(core)
    cursor = 0
    with torch.no_grad():
        for parameter in updated.parameters():
            width = parameter.numel()
            parameter.add_(step[cursor:cursor + width].reshape_as(parameter))
            cursor += width
    if cursor != step.numel():
        raise AssertionError("Inverse parameter order changed")
    actual = (state_vector(updated, x, initial).detach() - before)
    if not all(bool(torch.isfinite(value).all()) for value in
               (predicted_K, predicted_H, actual)):
        raise FloatingPointError("Nonfinite inverse response")
    denominator = float(torch.linalg.vector_norm(target))
    return {"seed": seed, "length": length,
        "operator_shape": list(K.shape), "ridge": ridge,
        "parameter_step_l2": float(torch.linalg.vector_norm(step)),
        "requested_state_change_l2_after_step_scaling": denominator,
        "regularized_K_target_relative_l2": float(torch.linalg.vector_norm(
            predicted_K + target)) / denominator,
        "true_H_target_relative_l2": float(torch.linalg.vector_norm(
            predicted_H + target)) / denominator,
        "actual_replay_target_relative_l2": float(torch.linalg.vector_norm(
            actual + target)) / denominator,
        "actual_vs_true_linear_relative_l2": float(torch.linalg.vector_norm(
            actual - predicted_H)) / max(float(torch.linalg.vector_norm(predicted_H)), 1e-30),
        "K_vs_H_relative_l2": float(torch.linalg.vector_norm(
            predicted_K - predicted_H)) / max(float(torch.linalg.vector_norm(predicted_H)), 1e-30),
        "scope": "one fixed history and fixed initial state; equal small parameter norm; three-state terminal response"}
