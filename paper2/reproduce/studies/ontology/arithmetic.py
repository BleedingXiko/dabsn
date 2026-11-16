"""Bounded event-edit probe for the implemented core eligibility state.

All edited states are evaluated at one frozen physical recurrent state and with
the same subsequent inputs and prescribed learning signals. The experiment
measures representation and actual parameter/forward effects separately. It
does not interpret an eligibility edit as a valid physical history.
"""
from __future__ import annotations

import copy
import json
import math

import reference
import torch
from dabsn.temporal_credit import CoreEligibilityState, CoreSignalGrads
from dabsn.temporal_credit import core_online_forward_reference
from dabsn.temporal_credit import core_online_replay_reference, initial_core_eligibility


LENGTHS = (32, 64, 128)
SEEDS = (0, 1, 2, 3, 4)
AMPLITUDES = (0.5, 1.0)
NOISE_STD = 0.1
STEP_NORM = 0.001


def family(background, event_a, event_b, length):
    """Four matched histories and a wrong-time event on one background."""
    first, second, wrong = length // 8, length // 4, length // 2
    if len({first, second, wrong}) != 3:
        raise ValueError("Event positions must differ")
    histories = {}
    for name, additions in {
        "empty": (), "a": ((first, 0, event_a),),
        "b": ((second, 1, event_b),),
        "ab": ((first, 0, event_a), (second, 1, event_b)),
        "wrong_a": ((wrong, 0, event_a),),
    }.items():
        x = background.clone()
        for position, channel, magnitude in additions:
            x[:, position, channel] += magnitude
        histories[name] = x
    return histories


def trace(core, sequence):
    state = core.initial_state(sequence.shape[0], device=sequence.device,
                               dtype=sequence.dtype)
    eligibility = initial_core_eligibility(core, sequence.shape[0],
        device=sequence.device, dtype=sequence.dtype)
    output, tape = core_online_forward_reference(core, sequence, state, eligibility)
    eligibility, _, _ = core_online_replay_reference(
        core, tape, CoreSignalGrads(), eligibility)
    return output.final_state, eligibility


def edit(*weighted_states):
    terms = [state.tensors() for _, state in weighted_states]
    tensors = [sum((weight * values[index] for (weight, _), values in
                    zip(weighted_states, terms))).contiguous()
               for index in range(6)]
    return CoreEligibilityState(*tensors)


def vector(state):
    return torch.cat([value.reshape(-1) for value in state.tensors()])


def relative_distance(first, second):
    return float(torch.linalg.vector_norm(first - second) /
                 torch.linalg.vector_norm(second).clamp_min(1e-30))


def cosine(first, second):
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    return float(torch.dot(first, second) / denominator) if float(denominator) > 0 else None


def suffix_readout(core, state, eligibility, x, signals):
    output, tape = core_online_forward_reference(core, x, state, eligibility)
    _, gradients, _ = core_online_replay_reference(core, tape, signals, eligibility)
    return output.trajectory, gradients, reference.flat(core, gradients)


def actual_update_response(core, gradients, evaluation, baseline):
    updated = copy.deepcopy(core)
    g = reference.flat(core, gradients)
    norm = torch.linalg.vector_norm(g)
    if not bool(torch.isfinite(norm)) or float(norm) == 0:
        raise FloatingPointError("Zero or nonfinite prescribed-signal credit")
    factor = STEP_NORM / float(norm)
    actual_step_squared = 0.0
    with torch.no_grad():
        for original, changed in zip(core.parameters(), updated.parameters()):
            before = changed.detach().clone()
            changed.add_(gradients[original], alpha=-factor)
            actual_step_squared += float((changed - before).square().sum())
        observed = updated._reference_forward_from_state(evaluation)[0]
    difference = observed - baseline
    return {"parameter_step_l2": math.sqrt(actual_step_squared),
            "actual_forward_change_l2": float(torch.linalg.vector_norm(difference)),
            "actual_forward_change": difference.reshape(-1)}


def one(seed, length, amplitude):
    core = reference.core(1000 + seed, inputs=2, state=3, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(
        83001 + 1009 * seed + 37 * length + int(100 * amplitude))
    background = torch.randn(1, length, 2, generator=generator,
                             dtype=torch.float64) * NOISE_STD
    histories = family(background, amplitude, -amplitude, length)
    outputs = {name: trace(core, x) for name, x in histories.items()}
    states = {name: values[1] for name, values in outputs.items()}
    empty, a, b, ab, wrong = (states[key] for key in
                             ("empty", "a", "b", "ab", "wrong_a"))
    proposed = {
        "actual_ab": ab,
        "additive_mix": edit((1, a), (1, b), (-1, empty)),
        "actual_b": b,
        "matched_subtract_a": edit((1, ab), (-1, a), (1, empty)),
        "mismatched_subtract_a": edit((1, ab), (-1, wrong), (1, empty)),
        "no_edit": ab,
    }
    suffix = torch.randn(1, 4, 2, generator=generator, dtype=torch.float64) * NOISE_STD
    physical_state = outputs["ab"][0]
    reference_output, _ = core_online_forward_reference(core, suffix, physical_state, ab)
    signals = reference.signals(reference_output, 91001 + seed)
    readouts = {}
    maximum_forward_difference = 0.0
    for name, eligibility in proposed.items():
        forward, gradient_map, credit = suffix_readout(
            core, physical_state, eligibility, suffix, signals)
        maximum_forward_difference = max(maximum_forward_difference,
            reference.maxdiff(forward, reference_output.trajectory))
        readouts[name] = (gradient_map, credit)
    if maximum_forward_difference > 1e-12:
        raise AssertionError("Eligibility edit altered frozen pre-update forward computation")
    evaluation = torch.randn(1, 8, 2, generator=generator,
                             dtype=torch.float64) * NOISE_STD
    with torch.no_grad():
        baseline = core._reference_forward_from_state(evaluation)[0]
    changes = {name: actual_update_response(core, gradient_map, evaluation, baseline)
               for name, (gradient_map, _) in readouts.items()}

    ab_delta = vector(ab) - vector(empty)
    mixed_delta = vector(proposed["additive_mix"]) - vector(empty)
    b_delta = vector(b) - vector(empty)
    def comparison(name, target):
        probe = vector(proposed[name]) - vector(empty)
        goal = vector(target) - vector(empty)
        probe_credit = readouts[name][1]
        goal_credit = readouts["actual_ab" if target is ab else "actual_b"][1]
        probe_forward = changes[name]["actual_forward_change"]
        goal_forward = changes["actual_ab" if target is ab else "actual_b"][
            "actual_forward_change"]
        return {"trace_relative_l2": relative_distance(probe, goal),
                "trace_cosine": cosine(probe, goal),
                "credit_relative_l2": relative_distance(probe_credit, goal_credit),
                "credit_cosine": cosine(probe_credit, goal_credit),
                "actual_equal_step_forward_relative_l2": relative_distance(
                    probe_forward, goal_forward),
                "actual_equal_step_forward_cosine": cosine(probe_forward, goal_forward),
                "actual_forward_change_l2": changes[name]["actual_forward_change_l2"]}
    result = {"seed": seed, "length": length, "event_amplitude": amplitude,
              "max_pre_update_forward_difference": maximum_forward_difference,
              "combined_from_separate": comparison("additive_mix", ab),
              "remove_a_matched": comparison("matched_subtract_a", b),
              "remove_a_mismatched": comparison("mismatched_subtract_a", b),
              "remove_a_no_edit": comparison("no_edit", b),
              "event_positions": {"a": length // 8, "b": length // 4,
                                  "mismatched_a": length // 2},
              "interpretation": "descriptive finite-core edit; true response after equal-norm parameter steps"}
    for value in (ab_delta, mixed_delta, b_delta):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite event representation")
    return result
