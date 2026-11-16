"""Bounded characterization of E: the seven readouts.

Three probes on microscopic FP64 cores (reference equations, CPU):

- decode: held-out decoding of event time and sign from the state, local E,
  complete E and the full terminal Jacobian at matched feature budgets; state
  reconstruction from local E; recursive E-to-state rollout.
- arithmetic: event edits of E (sums, differences, wrong-time events) at one
  frozen recurrent state, with their actual parameter and forward effects.
- inverse: inverse readouts of E against parameter steps of fixed norm.

Decoder outcomes measure accessibility under the declared probe; they do not
establish universal information inclusion or sufficiency.
"""
from __future__ import annotations

from studies.ontology import arithmetic, decode, inverse

NEEDS_NATIVE = False
FP64_REFERENCE = True  # reference equations in float64; the compute policy does not apply
DEVICE = "cpu"  # FP64 reference equations; the result does not depend on the accelerator


def declared(protocol):
    """The probe constants in the matrix must equal the constants in the probe code."""
    d = protocol["decode"]
    checks = {
        "decode": (decode.TRAIN_CORES, decode.TEST_CORES, decode.TRAIN_LENGTHS, decode.TEST_LENGTHS,
                   decode.EXAMPLES_PER_CORE_AND_LENGTH, decode.NOISE_STD, decode.RIDGE, decode.FEATURE_BUDGETS)
                  == (tuple(d["train_cores"]), tuple(d["test_cores"]), tuple(d["train_lengths"]),
                      tuple(d["test_lengths"]), d["examples_per_core_and_length"], d["noise_std"],
                      d["ridge"], tuple(d["feature_budgets"])),
        "arithmetic": (arithmetic.SEEDS, arithmetic.LENGTHS, arithmetic.AMPLITUDES, arithmetic.NOISE_STD,
                       arithmetic.STEP_NORM)
                      == (tuple(protocol["arithmetic"]["seeds"]), tuple(protocol["arithmetic"]["lengths"]),
                          tuple(protocol["arithmetic"]["amplitudes"]), protocol["arithmetic"]["noise_std"],
                          protocol["arithmetic"]["step_norm"]),
        "inverse": (inverse.SEEDS, inverse.LENGTHS, inverse.PARAMETER_STEP_NORM, inverse.RIDGE_FRACTION)
                   == (tuple(protocol["inverse"]["seeds"]), tuple(protocol["inverse"]["lengths"]),
                       protocol["inverse"]["parameter_step_norm"], protocol["inverse"]["ridge_fraction"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Ontology probe constants differ from the matrix: {checks}")


def jobs(protocol):
    declared(protocol)
    return [{"probe": name} for name in ("decode", "arithmetic", "inverse")]


def identifier(job):
    return job["probe"]


def run(job, ctx):
    declared(ctx.protocol)
    if job["probe"] == "decode":
        return decode.run()
    if job["probe"] == "arithmetic":
        return {"results": [arithmetic.one(seed, length, amplitude) for seed in arithmetic.SEEDS
                            for length in arithmetic.LENGTHS for amplitude in arithmetic.AMPLITUDES]}
    return {"results": [inverse.one(seed, length) for seed in inverse.SEEDS for length in inverse.LENGTHS]}
