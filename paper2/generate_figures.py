"""Regenerate the paper-2 figures from the run records.

    python generate_figures.py [results_all.json.gz]   (needs matplotlib and numpy)

The records are flattened into per-study tables in a temporary directory by
reproduce/export_results.py, and every plotted value is read from those tables;
lines are seed means and bands are one sample standard deviation across seeds.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "reproduce"))
from collect import results_directory  # noqa: E402
from export_results import export  # noqa: E402

DATA = None  # the tables exported from the records, set in __main__
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})
STYLE = {
    "online": dict(color="#c0392b", lw=2.2, zorder=5, label="online credit (E)"),
    "carry": dict(color="#c0392b", lw=2.2, zorder=5, label="carried E"),
    "bptt": dict(color="#2980b9", lw=1.8, zorder=3, label="BPTT"),
    "tbptt": dict(color="#2980b9", lw=1.6, zorder=3, ls="--", label="TBPTT (64 steps)"),
    "frozen_core": dict(color="#7f8c8d", lw=1.4, zorder=2, label="frozen core"),
    "reset": dict(color="#e67e22", lw=1.4, zorder=2, label="E erased"),
    "permute": dict(color="#8e44ad", lw=1.4, zorder=2, label="E exchanged"),
    "native": dict(color="#27ae60", lw=1.6, zorder=4, label="carried E, native action (core)"),
}


def read(name):
    path = DATA / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def by_method(rows, x_key, y_key, select=lambda row: True, method_key="job.method"):
    values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if select(row) and row.get(y_key) not in (None, ""):
            values[row[method_key]][float(row[x_key])].append(float(row[y_key]))
    out = {}
    for method, series in values.items():
        xs = np.array(sorted(series))
        means = np.array([np.mean(series[x]) for x in xs])
        stds = np.array([np.std(series[x], ddof=1) if len(series[x]) > 1 else 0.0 for x in xs])
        out[method] = xs, means, stds
    return out


def draw(axis, data, order, log_x=True):
    for method in order:
        if method not in data:
            continue
        xs, means, stds = data[method]
        style = dict(STYLE.get(method, dict(color="gray", lw=1, label=method)))
        axis.plot(xs, means, marker="o", ms=3, **style)
        if stds.any():
            axis.fill_between(xs, means - stds, means + stds, color=style["color"], alpha=.15, lw=0)
    if log_x:
        axis.set_xscale("log", base=2)


def save_continuity():
    rows = read("continuity__evaluation.csv")
    if rows is None:
        return None
    tasks = [("copy", "Copy (vocab 64), trained at 64", 1 / 64, 64), ("mqar", "MQAR, trained at 64", 1 / 16, 64),
             ("keyvalue", "Key-value recall, trained at 64", 1 / 16, 64), ("a5", "A5/60, trained at 256", 1 / 60, 256)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))
    for axis, (task, title, chance, trained) in zip(axes, tasks):
        # A5: held-out seeds only (seed 0 is the cohort's development seed).
        data = by_method(rows, "eval_length", "eval_accuracy",
                         lambda row: row["job.task"] == task and not (task == "a5" and row["job.seed"] == "0"))
        draw(axis, data, ["online", "bptt", "frozen_core"])
        axis.axhline(chance, color="gray", ls=":", lw=1); axis.axvline(trained, color="gray", ls="--", lw=.8, alpha=.6)
        axis.set_ylim(-.03, 1.05); axis.set_xlabel("evaluation length"); axis.set_title(title)
    # A5 BPTT is the architecture paper's published result on the same model and task
    # (1.000 at every length 128..16384, two seeds; Rosdahl 2026, Table "A5/60 accuracy vs. evaluation
    # length"), drawn as a reference; it is not a record of this reproduction.
    a5 = axes[3]
    lengths = [128 * 2 ** i for i in range(8)]
    a5.plot(lengths, [1.0] * len(lengths), color=STYLE["bptt"]["color"], lw=1.8, marker="o", ms=3,
            label="BPTT (architecture paper)")
    a5.legend(loc="lower left", frameon=False)
    axes[0].set_ylabel("held-out accuracy"); axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "fig1_continuity.pdf"); plt.close(fig)
    return "fig1_continuity.pdf"


def save_delayed():
    rows = read("delayed.csv")
    if rows is None:
        return None
    rows = [row for row in rows if row.get("job.phase") == "main"]
    if not rows:
        return None
    lengths = sorted({int(k.split(".")[1]) for k in rows[0] if k.startswith("final_by_length.")})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for axis, (task, metric, title, ylabel) in zip(axes, [
            ("temporal_order", "accuracy", "Temporal order", "held-out accuracy"),
            ("adding", "normalized_mse", "Adding", "MSE / (1/6)")]):
        long_rows = [{"job.method": row["job.method"], "length": length,
                      "value": row.get(f"final_by_length.{length}.{metric}")}
                     for row in rows if row["job.task"] == task for length in lengths]
        draw(axis, by_method(long_rows, "length", "value"), ["online", "bptt", "frozen_core"])
        axis.set_xlabel("episode length (steps)"); axis.set_ylabel(ylabel); axis.set_title(title)
        if metric == "accuracy":
            axis.axhline(1 / 4, color="gray", ls=":", lw=1); axis.set_ylim(-.03, 1.05)
        else:
            axis.axhline(1.0, color="gray", ls=":", lw=1)
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "fig2_delayed.pdf"); plt.close(fig)
    return "fig2_delayed.pdf"


def save_streams():
    capacity = read("continuous.csv"); narma = read("sustained.csv")
    if capacity is None and narma is None:
        return None
    order = ["carry", "native", "tbptt", "reset", "permute", "frozen_core"]
    # The stream studies' records also hold a superseded normalized arm; the native action on
    # the core comes from the stream-normalized stage's main runs.
    native = [dict(row, **{"job.method": "native"}) for row in (read("stream-normalized.csv") or [])
              if row.get("job.phase") == "main"]
    if capacity is not None:
        capacity = [row for row in capacity if row["job.method"] != "carry_normalized"]
        capacity += [row for row in native if row.get("job.study") == "continuous"]
    if narma is not None:
        narma = [row for row in narma if row["job.method"] != "carry_normalized"]
        narma += [row for row in native if row.get("job.study") == "sustained"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
    if capacity is not None:
        delays = sorted(int(k.rsplit(".", 1)[1]) for k in capacity[0] if k.startswith("final.memory_capacity_by_delay."))
        long_rows = [{"job.method": row["job.method"], "delay": k, "value": row.get(f"final.memory_capacity_by_delay.{k}")}
                     for row in capacity for k in delays]
        draw(left, by_method(long_rows, "delay", "value"), order)
        left.axvline(8, color="gray", ls="--", lw=.8, alpha=.6)
        left.set_xlabel("delay k (steps); dashed: chunk length"); left.set_ylabel("memory capacity MC$_k$")
        left.set_title("Memory capacity on uninterrupted streams"); left.set_ylim(-.03, 1.05)
        left.legend(loc="upper right", frameon=False, fontsize=7)
    if narma is not None:
        scores = defaultdict(list)
        for row in narma:
            scores[row["job.method"]].append(float(row["final.nmse"]))
        names = [m for m in order if m in scores]
        means = [np.mean(scores[m]) for m in names]
        stds = [np.std(scores[m], ddof=1) if len(scores[m]) > 1 else 0 for m in names]
        right.bar(range(len(names)), means, yerr=stds, color=[STYLE[m]["color"] for m in names], alpha=.85)
        right.set_xticks(range(len(names)), [STYLE[m]["label"] for m in names], rotation=35, ha="right", fontsize=7)
        right.set_ylabel("normalized MSE (lower is better)"); right.set_title("NARMA-10 on uninterrupted streams")
    fig.tight_layout(); fig.savefig(FIGS / "fig3_streams.pdf"); plt.close(fig)
    return "fig3_streams.pdf"


if __name__ == "__main__":
    keep = []
    records = results_directory(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results_all.json.gz", keep)
    tables = tempfile.TemporaryDirectory()
    DATA = Path(tables.name)
    export(records, DATA)
    for make in (save_continuity, save_delayed, save_streams):
        name = make()
        print(name or f"{make.__name__}: tables not available yet")
