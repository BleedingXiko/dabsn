"""The declared grid: counts, unique identifiers, and verbatim protocol values."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import matrix  # noqa: E402

EXPECTED = {"preflight": 15, "oracles": 9, "action": 15, "continuation": 5, "directions": 1,
            "ontology": 3, "scaling": 108, "resource": 54, "continuous": 30,
            "delayed-development": 16, "delayed": 30, "interventions": 180, "ablation": 24,
            "sustained": 30, "alignment": 10, "stream-window": 10, "stream-normalized-development": 20, "stream-normalized": 10, "conjunction-development": 24, "conjunction": 27, "continuity": 31, "audit": 4}


def test_job_counts():
    assert matrix.summary() == EXPECTED


def test_identifiers_unique():
    import importlib
    rates = {"delayed_rates": {"selected_rates": {"temporal_order": 1e-3, "adding": 3e-3}},
             "conjunction_settings": {"selected_rates": {t: {"rate": 0.03, "relative_floor": 1.0}
                                                         for t in ("copy", "mqar", "keyvalue")}},
             "stream_native_settings": {"selected_rates": {"continuous": {"rate": 1.0, "relative_floor": 0.1},
                                                           "sustained": {"rate": 1.0, "relative_floor": 0.1}}}}
    import run
    for name, *_ in matrix.STAGES:
        module, jobs = run.stage_jobs(name, matrix.MATRIX, rates)
        names = [module.identifier(job) for job in jobs]
        assert len(names) == len(set(names)), name


def test_smoke_runs_every_stage():
    counts = matrix.summary(matrix.smoke_protocol())
    assert all(count > 0 for count in counts.values()), counts


def test_interventions_fork_from_saved_checkpoints():
    m = matrix.MATRIX
    assert set(m["interventions"]["checkpoint_steps"]) - {0} <= set(m["delayed"]["main"]["checkpoint_steps"])
    s = matrix.smoke_protocol()
    assert set(s["interventions"]["checkpoint_steps"]) - {0} <= set(s["delayed"]["main"]["checkpoint_steps"])


def test_declared_values():
    m = matrix.MATRIX
    d = m["delayed"]
    assert d["main"]["methods"] == ["online", "frozen_core", "bptt"] and d["main"]["online_chunk"] == "episode"
    assert d["main"]["seeds"] == [100, 101, 102, 103, 104] and d["main"]["updates"] == 4500 and d["main"]["batch"] == 32
    assert d["development"]["learning_rates"] == [0.001, 0.003]
    assert d["tasks"] == ["temporal_order", "adding"] and "association" not in m
    assert m["interventions"]["tasks"] == d["tasks"] and m["ablation"]["shape"]["tasks"] == d["tasks"]
    for key, benchmark in (("continuous", "memory_capacity"), ("sustained", "narma10")):
        s = m[key]
        assert s["benchmark"] == benchmark and s["methods"] == ["carry", "reset", "permute", "frozen_core", "tbptt", "carry_normalized"]
        assert s["exchange_lane_permutation"] == [2, 0, 1, 7, 3, 4, 5, 6] and s["chunk"] == 8
        assert s["updates"] == 16384 and s["state_dim"] == 256 and s["learning_rate"] == 0.003
    assert m["continuity"]["a5_frozen_online_lr"] == 0.006 and m["continuity"]["a5_heldout_seeds"] == [1, 2, 3]
    assert m["interventions"]["conditions"] == ["carry", "zero", "permute_lanes", "negate_w", "erase_local"]
    assert "credit_rule" not in json.dumps(m)
