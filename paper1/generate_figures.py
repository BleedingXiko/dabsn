"""Regenerate the three paper-1 figures from the bundled result CSVs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "ancillary_results"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})
STYLE = {
    "dabsn_seq": dict(color="#c0392b", lw=2.2, zorder=5, label="DABSN (seq)"),
    "dabsn_field": dict(color="#e67e22", lw=1.6, zorder=4, label="DABSN (field)"),
    "dabsn_hybrid": dict(color="#d4a017", lw=1.4, zorder=4, label="DABSN (hybrid)"),
    "transformer_rope": dict(color="#2980b9", lw=1.8, zorder=3, label="Transformer (RoPE)"),
    "transformer_rope_causal": dict(color="#2980b9", lw=1.8, zorder=3, label="Transformer (RoPE)"),
    "transformer": dict(color="#7fb3d5", lw=1.4, zorder=2, label="Transformer (abs-PE)"),
    "gru": dict(color="#27ae60", lw=1.4, zorder=2, label="GRU"),
    "lstm": dict(color="#8e44ad", lw=1.4, zorder=2, label="LSTM"),
    "dabsn_seq_zero_read": dict(color="#7f8c8d", lw=1.2, zorder=1, label="DABSN, read zeroed"),
}


def series(path: Path, length_column: str, *, a5: bool = False):
    values = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if a5:
                if row["status"] != "ok":
                    continue
                name = row["model"]
                if name == "dabsn_seq" and row["condition"] == "zero_dabsn_read":
                    name = "dabsn_seq_zero_read"
                elif row["condition"] != "intact":
                    continue
            else:
                name = row["model"]
            values[name][int(row[length_column])].append(float(row["eval_acc"]))
    output = {}
    for name, by_length in values.items():
        lengths = np.array(sorted(by_length))
        means = np.array([np.mean(by_length[length]) for length in lengths])
        stds = np.array([np.std(by_length[length], ddof=1) if len(by_length[length]) > 1 else 0 for length in lengths])
        output[name] = lengths, means, stds
    return output


def panel(axis, data, chance, train_length, title, order=None):
    for name in order or data:
        if name not in data:
            continue
        lengths, means, stds = data[name]
        style = STYLE.get(name, dict(color="gray", lw=1, label=name))
        axis.plot(lengths, means, marker="o", ms=3, **style)
        if stds.any():
            axis.fill_between(lengths, means - stds, means + stds, color=style["color"], alpha=.15, lw=0)
    axis.axhline(chance, color="gray", ls=":", lw=1)
    axis.axvline(train_length, color="gray", ls="--", lw=.8, alpha=.6)
    axis.set_xscale("log", base=2); axis.set_ylim(-.03, 1.05)
    axis.set_xlabel("evaluation length"); axis.set_title(title)


def save_both_gaps():
    copy = series(DATA / "copy_length.csv", "eval_str_len")
    a5 = series(DATA / "a5_word_results.csv", "eval_seq_len", a5=True)
    order = ["dabsn_seq", "dabsn_field", "dabsn_hybrid", "transformer_rope", "transformer", "gru", "lstm"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 2.7))
    panel(left, copy, 1 / 64, 64, "Copy (vocab 64), 3 seeds", order); left.set_ylabel("accuracy")
    panel(right, a5, 1 / 60, 256, "A5 word problem (train length 256)", ["dabsn_seq", "lstm", "transformer_rope_causal", "dabsn_seq_zero_read"])
    left.legend(loc="center right", frameon=False, fontsize=7); right.legend(loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "fig1_both_gaps.pdf"); plt.close(fig)


def save_recall():
    fig, (left, right) = plt.subplots(1, 2, figsize=(7, 2.7))
    panel(left, series(DATA / "mqar_length.csv", "eval_seq_len"), 1 / 16, 64, "MQAR, 3 seeds"); left.set_ylabel("accuracy")
    panel(right, series(DATA / "keyvalue_length.csv", "eval_seq_len"), 1 / 16, 64, "Key-value recall, 3 seeds")
    left.legend(loc="center left", frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig2_recall_50x.pdf"); plt.close(fig)


def _curve(rows, model, condition):
    selected = sorted((int(row["eval_seq_len"]), float(row["eval_acc"])) for row in rows if row["model"] == model and row["condition"] == condition and row["status"] == "ok")
    return np.array([x for x, _ in selected]), np.array([y for _, y in selected])


def save_a5_ablation():
    def read(name):
        with (DATA / name).open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    seed0, seed1 = read("a5_word_results.csv"), read("a5_seed1_linear_ablation.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for axis, rows, model_name, condition, style, label in [
        (left, seed0, "dabsn_seq", "intact", "-o", "DABSN (seq), seed 0 (headline)"),
        (left, seed1, "dabsn_seq", "intact", "--s", "DABSN (seq), seed 1 (new)"),
        (right, seed1, "dabsn_seq", "intact", "-o", "DABSN (seq), intact"),
        (right, seed1, "dabsn_seq_linear", "intact", "-^", "linear-state ablation (TC$^0$)"),
        (right, seed1, "dabsn_seq", "zero_dabsn_read", "-D", "read zeroed"),
    ]:
        x, y = _curve(rows, model_name, condition); axis.plot(x, y, style, lw=2, label=label)
    left.set_title("A5/60 headline, two seeds"); left.set_ylabel("accuracy")
    right.set_title("Mechanism: nonlinear state is necessary")
    for axis in (left, right):
        axis.axhline(1 / 60, ls=":", color="gray", lw=1); axis.set_xscale("log", base=2)
        axis.set_xlabel("evaluation length"); axis.set_ylim(-.03, 1.05); axis.legend(fontsize=8, loc="center left"); axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIGS / "fig3_a5_seed_ablation.pdf", bbox_inches="tight"); fig.savefig(FIGS / "fig3_a5_seed_ablation.png", dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    save_both_gaps(); save_recall(); save_a5_ablation()
    print("wrote", *sorted(str(path) for path in FIGS.glob("fig*")), sep="\n")
