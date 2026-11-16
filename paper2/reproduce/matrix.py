"""The paper-2 reproduction matrix: every study, its jobs, and their order.

Standard library only. ``matrix_protocol.json`` holds every declared value.
Stages run in order. A stage may need a selection made from an earlier stage
(development learning rates); the selection is recorded with its inputs.

``smoke_protocol`` shrinks counts so the whole pipeline executes end to end in
minutes. Smoke results are flagged in every record and never mix with the matrix.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATRIX = json.loads((HERE / "matrix_protocol.json").read_text())

# (stage, study module, protocol key, job builder, selection it needs, selection it produces)
#  job builder: module attribute name; called with (protocol) or (protocol, selection)
STAGES = [
    ("preflight", "preflight", "execution", "jobs", None, None),
    ("oracles", "oracles", "oracles", "jobs", None, None),
    ("action", "action", "action", "jobs", None, None),
    ("continuation", "continuation", "continuation", "jobs", None, None),
    ("directions", "directions", "directions", "jobs", None, None),
    ("ontology", "ontology", "ontology", "jobs", None, None),
    ("scaling", "scaling", "scaling", "jobs", None, None),
    ("resource", "resource", "resource", "jobs", None, None),
    ("continuous", "continuous", "continuous", "jobs", None, None),
    ("delayed-development", "delayed", "delayed", "development_jobs", None, "delayed_rates"),
    ("delayed", "delayed", "delayed", "main_jobs", "delayed_rates", None),
    ("interventions", "interventions", "interventions", "jobs", "delayed_rates", None),
    ("ablation", "delayed", "ablation", "ablation_jobs", "delayed_rates", None),
    ("sustained", "sustained", "sustained", "jobs", None, None),
    ("alignment", "alignment", "alignment", "jobs", None, None),
    ("stream-window", "stream_window", "stream_window", "jobs", None, None),
    ("stream-normalized-development", "stream_normalized", "stream_normalized", "development_jobs", None,
     "stream_native_settings"),
    ("stream-normalized", "stream_normalized", "stream_normalized", "main_jobs", "stream_native_settings", None),
    ("conjunction-development", "conjunction", "conjunction", "development_jobs", None, "conjunction_settings"),
    ("conjunction", "conjunction", "conjunction", "main_jobs", "conjunction_settings", None),
    ("continuity", "continuity", "continuity", "jobs", None, None),
    ("audit", "audit", "continuity", "jobs", None, None),
]

# Results of these stages are read by later stages under their study name.
RECORD_STUDY = {"delayed-development": "delayed", "delayed": "delayed", "ablation": "delayed",
                "conjunction-development": "conjunction",
                "stream-normalized-development": "stream-normalized"}


def record_study(stage):
    return RECORD_STUDY.get(stage, stage)


def stage(name):
    for entry in STAGES:
        if entry[0] == name:
            return entry
    raise KeyError(name)


def protocol_for(name, matrix):
    """The protocol a stage's jobs and runs see. Ablation runs inside the delayed protocol."""
    _, _, key, _, _, _ = stage(name)
    if name == "ablation":
        return matrix["delayed"]
    return matrix[key]


def smoke_protocol(matrix=None):
    """Small counts on every axis, two seeds wherever the analysis forms intervals; the same code paths."""
    m = copy.deepcopy(matrix or MATRIX)
    m["smoke"] = True
    s = m["scaling"]
    s.update(seeds=[0], widths=[32, 256], lengths=[64], warmup=1, repetitions=1)
    r = m["resource"]
    r.update(shapes=[[64, 256]], lengths=[512], warmup=1, measurements=1)
    m["alignment"].update(seeds=[100, 101], windows=2)
    m["stream_window"].update(seeds=[100, 101])
    m["stream_normalized"].update(seeds=[100, 101])
    m["stream_normalized"]["development"].update(updates=6, rate_grid=[1.0, 3.0], relative_floor_grid=[0.1])
    m["conjunction"].update(seeds=[0, 1], evaluate_every=2, evaluation_batches=1)
    for key in ("continuous", "sustained"):
        m[key].update(seeds=[100, 101], warmup_chunks=2, updates=6, checkpoint_every=3, report_every=3,
                      score_window_updates=4, tbptt_window_chunks=3)
    d = m["delayed"]
    d["development"].update(seeds=[10], updates=4, evaluation_steps=[0, 2, 4], checkpoint_steps=[2, 4],
                            evaluation_examples_per_horizon=64)
    d["main"].update(seeds=[100, 101], updates=6, evaluation_steps=[0, 3, 6], checkpoint_steps=[3, 6],
                     train_lengths_round_robin=[128, 512], development_evaluation_examples_per_horizon=64,
                     final_evaluation_lengths=[128, 512], final_evaluation_examples_per_horizon=64)
    i = m["interventions"]
    i.update(training_seeds=[100, 101], checkpoint_steps=[0, 6], lengths=[128], paired_episodes_per_condition=8)
    ab = m["ablation"]
    ab["shape"].update(seeds=[100, 101], declared_training_jobs=16)
    ct = m["continuity"]
    ct.update(seeds=[0, 1], a5_heldout_seeds=[1, 2])
    for task in ct["tasks"].values():
        task.update(steps=3, checkpoint_every=2, warmup_steps=1, diagnostic_every=2, val_batches=1,
                    eval_lengths=task["eval_lengths"][:2])
        if "curriculum_steps" in task:
            task["curriculum_steps"] = 2
    m["action"].update(seeds=[0, 1], teacher_relative_perturbations=[0.01], repeated_actions=2)
    m["continuation"].update(seeds=[0, 1])
    m["directions"].update(calibration_seeds=[20, 21], test_seeds=[100, 101])
    return m


def summary(matrix=None, selections=None):
    """Job counts per stage (stages that need a selection use the first candidate)."""
    import importlib
    import sys
    sys.path.insert(0, str(HERE))
    m = matrix or MATRIX
    selections = dict(selections or {})
    selections.setdefault("delayed_rates", {t: m["delayed"]["development"]["learning_rates"][0]
                                            for t in m["delayed"]["tasks"]})
    sn = m["stream_normalized"]["development"]
    selections.setdefault("stream_native_settings", {
        study: {"rate": sn["rate_grid"][0], "relative_floor": sn["relative_floor_grid"][0]}
        for study in m["stream_normalized"]["studies"]})
    selections.setdefault("conjunction_settings", {
        task: {"rate": m["conjunction"]["rate_grid"][0], "relative_floor": m["conjunction"]["relative_floor_grid"][0]}
        for task in m["conjunction"]["tasks"]})
    counts = {}
    for name, module, key, builder, needs, _ in STAGES:
        mod = importlib.import_module(f"studies.{module}")
        function = getattr(mod, builder)
        if name == "ablation":
            jobs = function(m["ablation"], selections[needs])
        elif needs:
            jobs = function(m[key], selections[needs])
        else:
            jobs = function(m[key])
        counts[name] = len(jobs)
    return counts
