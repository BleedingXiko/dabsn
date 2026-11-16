"""Declare the paper 3 experiment matrix and write ``matrix_protocol.json``.

    python paper3/reproduce/matrix.py            # (re)writes the protocol

The protocol is the contract: ``run.py`` executes exactly the entries listed
there, in order. Every entry is self-describing (task, model, resolved model
spec, training and evaluation settings, seed, learning rate or selection rule).

Learning-rate selection. Each (task, model) is trained at every rate in
``lrs`` with seed 0 (phase ``sweep``) and scored on in-distribution validation
data only. The ``main`` phase then trains seeds 1 and 2 at the selected rate and
evaluates all three seeds on the test axes; seed 0 at the selected rate reuses
the sweep checkpoint, because it is the identical training run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from models import model_spec
from tasks import make_task

HERE = Path(__file__).resolve().parent

# Synthetic data version: the published generators (see tasks.py).
DATA = "standard_v1"

SYNTHETIC = {
    # The vocabulary is each task's own (Zoology 8192, copying 10, selective
    # copying 16, passkey word list); see ``_size``.
    "size": {"d_model": 128, "n_layers": 4, "n_heads": 4,
             "mlp_mult": 4, "window": 64, "segment": 64},
    # Train to convergence: constant rate after warmup; stop once held-out
    # validation accuracy (never test data) reaches solved_acc at solved_checks
    # consecutive checks, or at the step cap. Constant rate makes a cap
    # extension exact. Every check is logged as the run's learning curve.
    #
    # Budgets are in tokens; steps = tokens / (batch * train_len). Batch 64 is
    # the measured throughput optimum on the L4 (larger batches were slower).
    "train": {"train_len": 256, "batch": 64,
              "tokens": 30000 * 64 * 256, "warmup_tokens": 200 * 64 * 256,
              "val_every_tokens": 1000 * 64 * 256,
              "schedule": "constant", "min_lr_ratio": 1.0, "weight_decay": 0.1,
              "betas": [0.9, 0.95], "grad_clip": 1.0,
              "val_check_seqs": 128, "solved_acc": 0.99, "solved_checks": 2},
    "lrs": [3e-4, 1e-3, 3e-3],
    # Edge extension: if the best rate is at an edge of the grid and has not
    # solved the task, one more rate past that edge is trained (seed 0) and
    # joins the selection; otherwise it is skipped. The effective range is
    # Zoology's sweep range, 1e-4 to 1e-2.
    "lr_extend": {"low": 1e-4, "high": 1e-2},
    "seeds": [0, 1, 2],
    # Training loads. MQAR: Zoology's three length-256 training configurations
    # (16, 32, 64 pairs, equal shares). Copying: Arjovsky's 10 tokens.
    # Selective copying: Mamba's 16 tokens. Passkey: M&J's key in [1, 50000].
    "tasks": {
        "mqar": {"loads": [16, 32, 64]},
        "passkey": {"loads": [5]},
        "copy": {"loads": [10]},
        "selective_copy": {"loads": [16]},
    },
    # The single memory sits at the front (before the first block) in both
    # memory models; the parameter-matched control sits at the same place.
    "models": ["nope", "rope", "alibi", "swa", "infini", "nope_ctrl_front",
               "nope_mem_front", "swa_mem_front"],
    "eval": {
        "distance_multipliers": [1, 2, 4, 8, 16, 32, 64],
        "load_multipliers": [1, 2, 4, 8],
        "val_seqs": 256,
        "test_seqs": 128,
        "eval_tokens_per_batch": 131072,
        "mech_samples": 8,
    },
}

# Where the single memory sits. Front is the main grid's placement; on MQAR
# both memory models (and the matched control) are also trained in the
# middle, and NoPE + memory at the back.
PLACEMENT = {"tasks": ["mqar"],
             "models": ["nope_ctrl", "nope_mem", "nope_mem_back", "swa_mem"]}

# Content-dependence demonstration. Memory models are trained on key-value
# recall with the query after the data. Each trained model (every seed) is then
# tested on the same sequences with the query also shown before the data:
# accuracy, admitted bank and read work per sample, query late and query early.
MECHANISM = {
    "tasks": ["kv_recall"],
    "loads": [16, 32, 64],
    "models": ["nope_mem_front", "swa_mem_front"],
    "evals": ["val", "distance", "mech"],
    "query_early": {"samples": 128},
}

# Passkey far beyond the evaluation grid. Evaluation only, on the passkey
# checkpoints of the main phase at the selected rate: every seed for the
# memory models, seed 0 for the baselines.
EXTREME = {
    "task": "passkey",
    "models": ["swa_mem_front", "nope_mem_front", "infini", "swa", "nope"],
    "all_seeds": ["swa_mem_front", "nope_mem_front"],
    "lengths": [16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    "samples": 10,
    "max_seconds_per_length": 1800,
}

LM = {
    "dataset": "enwik8",
    "size": {"vocab": 256, "d_model": 256, "n_layers": 6, "n_heads": 8,
             "mlp_mult": 4, "window": 128, "segment": 128},
    "train": {"train_len": 512, "batch": 32, "steps": 20000, "warmup": 500,
              "schedule": "cosine", "min_lr_ratio": 0.1, "weight_decay": 0.1,
              "betas": [0.9, 0.95], "grad_clip": 1.0, "val_every": 1000,
              "val_check_seqs": 16},
    "sweep_steps": 4000,
    "lrs": [3e-4, 1e-3, 2e-3],
    "lr_extend": {"low": 1e-4, "high": 4e-3},
    "seeds": [0, 1, 2],
    "models": ["nope", "rope", "alibi", "swa", "infini", "nope_ctrl_front",
               "nope_mem_front", "swa_mem_front"],
    "eval": {
        "val_seqs": 64,
        "position_multiplier": 32,
        "position_seqs": 16,
        "passkey_multipliers": [1, 2, 4, 8, 16, 32],
        "passkey_samples": 32,
        "passkey_format": "mohtashami_jaggi_2023",
        "eval_tokens_per_batch": 65536,
    },
}

# RoPE with YaRN applied properly: each converged RoPE model is fine-tuned at
# ``scale`` x the training length with YaRN scaling (tokens per step held
# fixed), then evaluated on the same grid with that scaling built in.
YARN_FT = {"scale": 4, "tokens": 2000 * 16 * 1024, "warmup_tokens": 100 * 16 * 1024,
           "val_every_tokens": 500 * 16 * 1024}

# A larger tier on MQAR, with its own sweep and seeds.
SCALE = {
    "size": {"d_model": 256, "n_layers": 8, "n_heads": 8,
             "mlp_mult": 4, "window": 64, "segment": 64},
    "tag": "d256L8",
    "batch": 64,
    "tasks": ["mqar"],
    "models": ["nope", "rope", "alibi", "swa", "infini", "nope_mem_front", "swa_mem_front"],
}

# Run last: the redundancy ablation (two memories), on MQAR.
OPTIONAL = {"tasks": ["mqar"], "models": ["nope_mem_x2"]}

EVAL_SEED = 20261110
LEARNED_THRESHOLD = 0.9


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]


def _lr_tag(lr) -> str:
    return f"{lr:g}"


def _size(task: str, base: dict) -> dict:
    return {**base, "vocab": make_task(task).vocab}


MAIN_EVALS = ["val", "distance", "load", "mech"]


def _extensions(base_ids, mk, rates, metric, solved):
    """Conditional sweep entries past each edge of the declared grid.

    Each runs only if, once every earlier sweep entry has finished, the best
    rate is the edge on its side and (for accuracy) has not reached ``solved``.
    """
    out, after = [], list(base_ids)
    for direction in ("low", "high"):
        e = mk(rates[direction])
        e["extend"] = {"after": list(after), "direction": direction, "metric": metric,
                       "solved": solved}
        out.append(e)
        after.append(e["id"])
    return out


def _synthetic_entries(task: str, loads, models, phase_prefix="", size=None,
                       size_tag=None, batch=None, evals=None):
    s = SYNTHETIC
    sweep, main = [], []
    for model in models:
        spec = model_spec(model, _size(task, size or s["size"]))
        common = {
            "kind": "synthetic",
            "task": task,
            "loads": list(loads),
            "data": DATA,
            "model": model,
            "spec": spec,
            "train": dict(s["train"]) if batch is None else {**s["train"], "batch": batch},
            "eval": dict(s["eval"]),
        }
        if size_tag:
            common["size_tag"] = size_tag

        def mk(lr, common=common, model=model):
            return {**common, "id": f"{phase_prefix}sweep__{task}__{model}__lr{_lr_tag(lr)}",
                    "phase": f"{phase_prefix}sweep", "seed": 0, "lr": lr, "evals": ["val"]}

        base = [mk(lr) for lr in s["lrs"]]
        ext = _extensions([e["id"] for e in base], mk, s["lr_extend"], "val_acc",
                          s["train"]["solved_acc"])
        sweep += base + ext
        ids = [e["id"] for e in base + ext]
        for seed in s["seeds"]:
            main.append({**common, "id": f"{phase_prefix}main__{task}__{model}__s{seed}",
                         "phase": f"{phase_prefix}main", "seed": seed, "lr": "best",
                         "select_from": ids, "select_metric": "val_acc",
                         "evals": list(evals or MAIN_EVALS)})
    return sweep + main


def _yarn_ft_entries(task: str, loads):
    s, y = SYNTHETIC, YARN_FT
    tl = s["train"]["train_len"]
    spec = {**model_spec("rope", _size(task, s["size"])),
            "rope_scaling": {"orig_len": tl, "min_scale": y["scale"]}}
    train = {k: v for k, v in s["train"].items() if k not in {"solved_acc", "solved_checks"}}
    train.update({"train_len": tl * y["scale"], "batch_div": y["scale"], "tokens": y["tokens"],
                  "warmup_tokens": y["warmup_tokens"],
                  "val_every_tokens": y["val_every_tokens"]})
    return [{
        "kind": "synthetic", "task": task, "loads": list(loads), "data": DATA,
        "model": "rope_yarn_ft", "spec": spec, "train": train, "eval": dict(s["eval"]),
        "eval_train_len": tl, "id": f"main__{task}__rope_yarn_ft__s{seed}", "phase": "main",
        "seed": seed, "lr": "init", "init_from": f"main__{task}__rope__s{seed}",
        "evals": ["val", "distance", "load"],
    } for seed in s["seeds"]]


def build() -> dict:
    entries: list[dict] = []
    tasks = SYNTHETIC["tasks"]
    # 1. Main synthetic grid, task by task (MQAR first).
    for task, cfg in tasks.items():
        entries += _synthetic_entries(task, cfg["loads"], SYNTHETIC["models"])
        if task in PLACEMENT["tasks"]:
            entries += _synthetic_entries(task, cfg["loads"], PLACEMENT["models"])
        entries += _yarn_ft_entries(task, cfg["loads"])
    # 2. Mechanism.
    for task in MECHANISM["tasks"]:
        entries += _synthetic_entries(task, MECHANISM["loads"], MECHANISM["models"],
                                      evals=MECHANISM["evals"])
    # Query early: evaluation only, on the trained query-late checkpoints.
    for model in MECHANISM["models"]:
        for seed in SYNTHETIC["seeds"]:
            entries.append({
                "id": f"qearly__kv_recall__{model}__s{seed}",
                "phase": "mechanism",
                "kind": "qearly",
                "task": "kv_recall",
                "loads": list(MECHANISM["loads"]),
                "model": model,
                "seed": seed,
                "train_from": f"main__kv_recall__{model}__s{seed}",
                "query_early": {
                    "distance_multipliers": list(SYNTHETIC["eval"]["distance_multipliers"]),
                    **MECHANISM["query_early"]},
            })

    # 3. Language model.
    lm_ids: dict[str, list[str]] = {}
    for model in LM["models"]:
        spec = model_spec(model, LM["size"])
        common = {"kind": "lm", "task": LM["dataset"], "model": model, "spec": spec,
                  "eval": dict(LM["eval"])}

        def mk(lr, common=common, model=model):
            return {**common, "id": f"lm_sweep__{model}__lr{_lr_tag(lr)}", "phase": "lm_sweep",
                    "seed": 0, "lr": lr, "train": {**LM["train"], "steps": LM["sweep_steps"]},
                    "evals": ["val"]}

        base = [mk(lr) for lr in LM["lrs"]]
        ext = _extensions([e["id"] for e in base], mk, LM["lr_extend"], "val_loss", None)
        entries += base + ext
        lm_ids[model] = [e["id"] for e in base + ext]
    for model in LM["models"]:
        spec = model_spec(model, LM["size"])
        for seed in LM["seeds"]:
            entries.append({"kind": "lm", "task": LM["dataset"], "model": model, "spec": spec,
                            "eval": dict(LM["eval"]), "train": dict(LM["train"]),
                            "id": f"lm_main__{model}__s{seed}", "phase": "lm_main",
                            "seed": seed, "lr": "best", "select_from": lm_ids[model],
                            "select_metric": "val_loss",
                            "evals": ["val", "position", "passkey"]})

    # 4. Extreme-length passkey (evaluation only).
    pk = tasks["passkey"]
    for model in EXTREME["models"]:
        seeds = SYNTHETIC["seeds"] if model in EXTREME["all_seeds"] else [0]
        for seed in seeds:
            entries.append({
                "id": f"extreme__passkey__{model}__s{seed}",
                "phase": "extreme",
                "kind": "extreme",
                "task": "passkey",
                "loads": pk["loads"],
                "model": model,
                "seed": seed,
                "train_from": f"main__passkey__{model}__s{seed}",
                "extreme": dict(EXTREME),
            })

    # 5. Scale tier.
    for task in SCALE["tasks"]:
        entries += _synthetic_entries(task, tasks[task]["loads"], SCALE["models"],
                                      phase_prefix="scale_", size=SCALE["size"],
                                      size_tag=SCALE["tag"], batch=SCALE["batch"])

    # 6. Optional ablation.
    for task in OPTIONAL["tasks"]:
        entries += _synthetic_entries(task, tasks[task]["loads"], OPTIONAL["models"],
                                      phase_prefix="opt_")

    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate run ids"
    for e in entries:
        if e["kind"] == "synthetic":
            task = make_task(e["task"], e["loads"])
            # every training sequence must fit its declared length
            need = task.content_len(max(e["loads"]))
            assert need <= e["train"]["train_len"], (e["id"], need)
            assert e["spec"]["vocab"] == task.vocab, e["id"]
        if e.get("init_from"):
            assert e["init_from"] in ids, e["id"]
        if e.get("train_from"):
            assert e["train_from"] in ids, e["id"]
    protocol = {
        "name": "paper3_matrix",
        "eval_seed": EVAL_SEED,
        "learned_threshold": LEARNED_THRESHOLD,
        "selection": "highest in-distribution validation accuracy (synthetic) or lowest "
                     "validation loss (LM), seed 0, among the declared rates plus a "
                     "conditional rate past the best edge (run only when the best rate "
                     "is at that edge and, for accuracy, below the solved threshold)",
        "phases": ["sweep", "main", "mechanism", "lm_sweep", "lm_main", "extreme", "scale_sweep",
                   "scale_main", "opt_sweep", "opt_main"],
        "config": {"synthetic": SYNTHETIC, "data": DATA, "placement": PLACEMENT,
                   "yarn_ft": YARN_FT, "mechanism": MECHANISM, "extreme": EXTREME,
                   "scale": SCALE, "lm": LM, "optional": OPTIONAL},
        "runs": entries,
    }
    protocol["protocol_hash"] = _hash({k: v for k, v in protocol.items()})
    return protocol


def main():
    protocol = build()
    path = HERE / "matrix_protocol.json"
    path.write_text(json.dumps(protocol, indent=1) + "\n")
    by_phase: dict[str, int] = {}
    for e in protocol["runs"]:
        by_phase[e["phase"]] = by_phase.get(e["phase"], 0) + 1
    print(f"wrote {path} ({len(protocol['runs'])} entries)")
    for k, v in by_phase.items():
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
