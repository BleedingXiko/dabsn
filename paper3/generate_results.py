"""Build every table, figure and in-text number of paper 3 from the run records.

    python generate_results.py [results_all.json]     (needs numpy and matplotlib)

The input is the file written by ``reproduce/run.py collect``. Outputs:
    tables/*.tex      LaTeX tables included by main.tex
    tables/numbers.tex  macros for every number quoted in the text
    figs/*.pdf        figures

Accuracies are seed means over seeds 0-2 with one sample standard deviation;
a model is marked as not having learned a task when its mean validation
accuracy at the training length is below the protocol's learned threshold.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results_all.json"
TABLES, FIGS = HERE / "tables", HERE / "figs"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

PROTOCOL = json.loads((HERE / "reproduce" / "matrix_protocol.json").read_text())
LEARNED = float(PROTOCOL["learned_threshold"])
RESULTS = json.loads(SRC.read_text())["results"]
DIST = [1, 2, 4, 8, 16, 32, 64]
LOAD = [1, 2, 4, 8]

NAMES = {
    "nope": "NoPE", "rope": "RoPE", "rope/pi": "RoPE + PI", "rope/yarn": "RoPE + YaRN",
    "rope_yarn_ft": "RoPE + YaRN (fine-tuned)", "alibi": "ALiBi", "swa": "Sliding window",
    "infini": "Infini-attention", "nope_ctrl_front": "NoPE + matched MLP (front)",
    "nope_ctrl": "NoPE + matched MLP (middle)",
    "nope_mem_front": "NoPE + memory (front)", "swa_mem_front": "Sliding window + memory (front)",
    "nope_mem": "NoPE + memory (middle)", "swa_mem": "Sliding window + memory (middle)",
    "nope_mem_back": "NoPE + memory (back)", "nope_mem_x2": "NoPE + two memories (front, back)",
}
MEM = {"nope_mem_front", "swa_mem_front", "nope_mem", "swa_mem", "nope_mem_back", "nope_mem_x2"}
BASELINE_ORDER = ["nope", "rope", "rope/pi", "rope/yarn", "rope_yarn_ft", "alibi", "swa", "infini",
                  "nope_ctrl_front", "nope_mem_front", "swa_mem_front"]

# ----------------------------------------------------------------------
# Index the records

runs = defaultdict(list)  # (phase, task, model) -> [record]
for r in RESULTS:
    e = r["entry"]
    if r.get("skipped"):
        continue
    runs[(e["phase"], e["task"], e["model"])].append(r)
for v in runs.values():
    v.sort(key=lambda r: r["entry"]["seed"])


def series(phase, task, row, key_fn):
    """Per-seed values of an eval item; ``row`` may carry a test-time variant."""
    model, _, variant = row.partition("/")
    variant = variant or "base"
    out = []
    for r in runs.get((phase, task, model), []):
        item = r["eval"].get(key_fn(variant))
        if item is None and variant != "base":
            item = r["eval"].get(key_fn("base"))  # variants exist only beyond the train length
        if item is not None and item.get("status") == "ok":
            out.append(item["acc"])
    return out


def ms(xs):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return None, None
    return 100 * xs.mean(), (100 * xs.std(ddof=1) if len(xs) > 1 else 0.0)


def learned(phase, task, row):
    m, s = ms(series(phase, task, row, lambda v: "val/1/base"))
    return m is not None and m >= 100 * LEARNED


def cell(xs, bold=False):
    m, s = ms(xs)
    if m is None:
        return "--"
    txt = f"{m:.1f}"
    if bold:
        txt = r"\textbf{" + txt + "}"
    return txt + (r"{\scriptsize$\pm$" + f"{s:.1f}" + "}" if s is not None and len(xs) > 1 else "")


def write(name, text):
    (TABLES / name).write_text(text)


NUM: dict[str, str] = {}


def num(name, value, fmt="{:.1f}"):
    NUM[name] = fmt.format(value) if not isinstance(value, str) else value


# ----------------------------------------------------------------------
# Synthetic accuracy tables

def synth_table(phase, task, rows, caption, label, dist_cols=(1, 4, 16, 64), load_cols=(2, 8)):
    have_load = any(series(phase, task, r, lambda v: f"load/1/{v}") for r in rows)
    cols = [("val", None)] + [("distance", m) for m in dist_cols]
    if have_load:
        cols += [("load", f) for f in load_cols]
    vals = {}
    for row in rows:
        for axis, x in cols:
            key = (lambda v: "val/1/base") if axis == "val" else (lambda v, a=axis, x=x: f"{a}/{x}/{v}")
            vals[(row, axis, x)] = series(phase, task, row, key)
    best = {}
    for axis, x in cols:
        if axis == "val":
            continue
        means = [ms(vals[(r, axis, x)])[0] or -1 for r in rows if learned(phase, task, r)]
        best[(axis, x)] = max(means) if means else None
    head = ["Model", "val"] + [rf"{m}$\times$" for _, m in cols[1:1 + len(dist_cols)]]
    if have_load:
        head += [rf"{f}$\times$" for f in load_cols]
    spec = "l" + "r" * (len(head) - 1)
    lines = [r"\begin{table}\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    group = r" & \multicolumn{1}{c}{train} & \multicolumn{" + str(len(dist_cols)) + r"}{c}{distance}"
    if have_load:
        group += r" & \multicolumn{" + str(len(load_cols)) + r"}{c}{load}"
    lines += [group + r" \\", " & ".join(head) + r" \\", r"\midrule"]
    for row in rows:
        if not runs.get((phase, task, row.split("/")[0])):
            continue
        ok = learned(phase, task, row)
        name = NAMES.get(row, row) + ("" if ok else r"$^\dagger$")
        if row.split("/")[0] in MEM:
            name = r"\textbf{" + name + "}"
        cells = []
        for axis, x in cols:
            xs = vals[(row, axis, x)]
            m = ms(xs)[0]
            b = axis != "val" and ok and best.get((axis, x)) is not None and m is not None \
                and round(m, 1) == round(best[(axis, x)], 1)
            cells.append(cell(xs, bold=b))
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    write(label.split(":")[1] + ".tex", "\n".join(lines))


DAG = (r"Accuracy (\%), mean $\pm$ standard deviation over three seeds. Distance: filler "
       r"between the facts and their use at $k\times$ the training length; load: $k\times$ "
       r"the largest training load with the filler at its training size. The best "
       r"model that learned the task is in bold; $^\dagger$ marks models below "
       rf"{int(100 * LEARNED)}\% at the training length, which are not compared beyond it.")

synth_table("main", "mqar", BASELINE_ORDER,
            r"MQAR~\citep{arora2024zoology}, trained at 256 tokens. " + DAG, "tab:mqar")
synth_table("main", "passkey", BASELINE_ORDER,
            r"Passkey retrieval~\citep{mohtashami2023landmark}, trained at 256 tokens. " + DAG,
            "tab:passkey")
synth_table("main", "copy", BASELINE_ORDER,
            r"Copying~\citep{arjovsky2016unitary}, trained at 256 tokens. " + DAG, "tab:copy",
            dist_cols=(1, 2, 4, 16, 64))
synth_table("main", "selective_copy", BASELINE_ORDER,
            r"Selective copying~\citep{gu2023mamba}, trained at 256 tokens. " + DAG,
            "tab:selcopy", dist_cols=(1, 2, 4, 16, 64))
synth_table("scale_main", "mqar", ["nope", "rope", "rope/pi", "rope/yarn", "alibi", "swa", "infini",
                                   "nope_mem_front", "swa_mem_front"],
            r"MQAR with width 256 and 8 layers, with its own learning-rate sweep and seeds. "
            + DAG, "tab:scale")
synth_table("main", "mqar", ["nope_ctrl_front", "nope_mem_front", "swa_mem_front", "nope_ctrl",
                             "nope_mem", "swa_mem", "nope_mem_back"],
            r"Placement of the single memory on MQAR, with the parameter-matched control at "
            r"the same place. " + DAG, "tab:placement")


# ----------------------------------------------------------------------
# Optional ablation: two memories

def x2_row():
    rs = runs.get(("opt_main", "mqar", "nope_mem_x2"), [])
    return {m: series("opt_main", "mqar", "nope_mem_x2", lambda v, m=m: f"distance/{m}/{v}")
            for m in DIST}, rs


x2, _ = x2_row()
num("xTwoDistSixtyFour", ms(x2[64])[0])

# ----------------------------------------------------------------------
# Extreme length

EXT_LEN = [16384, 32768, 65536, 131072, 262144, 524288, 1048576]
ext = defaultdict(dict)  # model -> seed -> {length: (status, acc, seconds)}
for r in RESULTS:
    e = r["entry"]
    if e["phase"] != "extreme":
        continue
    for L in EXT_LEN:
        it = r["eval"].get(f"extreme/{L}/base")
        if it is None:
            continue
        ext[e["model"]][e["seed"]] = ext[e["model"]].get(e["seed"], {})
        ext[e["model"]][e["seed"]][L] = (it["status"], it.get("acc"), it.get("seconds"),
                                         it.get("seq_acc"))

EXT_ORDER = ["nope_mem_front", "swa_mem_front", "nope", "swa", "infini"]
lines = [r"\begin{table}\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Model & Seed & 16K & 32K & 64K & 128K & 256K & 512K & 1M \\", r"\midrule"]
for model in EXT_ORDER:
    for seed in sorted(ext.get(model, {})):
        row = ext[model][seed]
        cells = []
        for L in EXT_LEN:
            st = row.get(L)
            if st is None:
                cells.append(r"--\rlap{$^\ast$}" if model in ("swa", "swa_mem_front", "infini") else "--")
            elif st[0] == "ok":
                cells.append(f"{100 * st[1]:.0f}")
            else:
                cells.append(st[0].replace("_", " "))
        name = NAMES[model]
        if model in MEM:
            name = r"\textbf{" + name + "}"
        lines.append(f"{name} & {seed} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Passkey retrieval far beyond the 256-token training length: digit "
          r"accuracy (\%) over 10 prompts per length (chance 10\%), on the passkey "
          r"checkpoints of the main phase (every seed for the memory models, seed 0 for the "
          r"baselines). $^\ast$The reference sliding-window and segment attention kernels "
          r"used here exceed a CUDA launch limit at $2^{20}$ tokens (one call with "
          r"$2^{16}$ attention blocks); no result exists for those models at that length.}",
          r"\label{tab:extreme}", r"\end{table}", ""]
write("extreme.tex", "\n".join(lines))

nm = ext["nope_mem_front"]
num("extMemOneMMin", min(100 * nm[s][1048576][1] for s in nm), "{:.0f}")
num("extMemOneMMax", max(100 * nm[s][1048576][1] for s in nm), "{:.0f}")
num("extMemOneMSeeds", ", ".join(f"{100 * nm[s][1048576][1]:.0f}" for s in sorted(nm)), "{}")
num("extNopeOneM", 100 * ext["nope"][0][1048576][1], "{:.0f}")
num("extMemOneMSec", np.mean([nm[s][1048576][2] for s in nm]) / 10, "{:.0f}")
base_max = max(100 * v[1] for m in ("nope", "swa", "infini") for s in ext[m].values()
               for v in s.values() if v[0] == "ok")
num("extBaselineMax", base_max, "{:.0f}")
sm = ext["swa_mem_front"]
num("extSwaMemHalfM", ", ".join(f"{100 * sm[s][524288][1]:.0f}" for s in sorted(sm)), "{}")

fig, ax = plt.subplots(figsize=(5.2, 3.0))
colors = {"nope_mem_front": "#c0392b", "swa_mem_front": "#e67e22", "nope": "#2c3e50",
          "swa": "#7f8c8d", "infini": "#8e44ad"}
for model in EXT_ORDER:
    seeds = ext.get(model, {})
    xs = [L for L in EXT_LEN if all(seeds[s].get(L, ("x",))[0] == "ok" for s in seeds)]
    if not xs:
        continue
    ys = [100 * np.mean([seeds[s][L][1] for s in seeds]) for L in xs]
    ax.plot(xs, ys, marker="o", ms=3, color=colors[model], lw=2 if model in MEM else 1.3,
            label=NAMES[model] + ("" if len(seeds) > 1 else " (seed 0)"))
ax.axhline(10, color="#bbb", lw=0.8, ls=":")
ax.set_xscale("log", base=2)
ax.set_xticks(EXT_LEN, ["16K", "32K", "64K", "128K", "256K", "512K", "1M"])
ax.set_ylim(0, 103)
ax.set_xlabel("sequence length (training length 256)")
ax.set_ylabel("passkey digit accuracy (%)")
ax.legend(frameon=False, fontsize=7, loc="center left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIGS / "extreme.pdf")
plt.close(fig)

# ----------------------------------------------------------------------
# Distance curves (MQAR, passkey)

STY = {"nope": ("#2c3e50", "-"), "rope": ("#2980b9", "-"), "rope/yarn": ("#2980b9", "--"),
       "rope_yarn_ft": ("#16a085", "-"), "alibi": ("#27ae60", "-"), "swa": ("#7f8c8d", "-"),
       "infini": ("#8e44ad", "-"), "nope_ctrl_front": ("#95a5a6", "--"),
       "nope_mem_front": ("#c0392b", "-"), "swa_mem_front": ("#e67e22", "-")}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, task, title in zip(axes, ("mqar", "passkey"), ("MQAR", "Passkey")):
    for row, (c, ls) in STY.items():
        if not learned("main", task, row):
            continue
        ys = [ms(series("main", task, row, lambda v, m=m: f"distance/{m}/{v}"))[0] for m in DIST]
        ax.plot(DIST, ys, color=c, ls=ls, marker="o", ms=2.5,
                lw=2.2 if row in MEM else 1.2, label=NAMES[row])
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIST, [f"{m}" for m in DIST])
    ax.set_xlabel(r"evaluation length / training length")
    ax.set_title(title)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("accuracy (%)")
axes[0].set_ylim(0, 103)
h, lab = axes[1].get_legend_handles_labels()
h0, lab0 = axes[0].get_legend_handles_labels()
seen = dict(zip(lab0 + lab, h0 + h))
fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=4, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(FIGS / "distance.pdf")
plt.close(fig)


def mean_at(phase, task, row, axis, x):
    return ms(series(phase, task, row, lambda v: f"{axis}/{x}/{v}"))[0]


for task, tag in (("mqar", "Mqar"), ("passkey", "Passkey")):
    num(f"mem{tag}DistSixtyFour", mean_at("main", task, "nope_mem_front", "distance", 64))
    num(f"swaMem{tag}DistSixtyFour", mean_at("main", task, "swa_mem_front", "distance", 64))
    num(f"mem{tag}LoadEight", mean_at("main", task, "nope_mem_front", "load", 8))
    for row, rtag in (("rope", "Rope"), ("rope/yarn", "RopeYarn"), ("rope_yarn_ft", "YarnFt"),
                      ("alibi", "Alibi"), ("nope", "Nope"), ("nope_ctrl_front", "Ctrl")):
        num(f"{rtag}{tag}DistSixtyFour", mean_at("main", task, row, "distance", 64))
        num(f"{rtag}{tag}LoadEight", mean_at("main", task, row, "load", 8))
num("nopeScaleVal", mean_at("scale_main", "mqar", "nope", "val", 1))
num("nopeScaleDistSixtyFour", mean_at("scale_main", "mqar", "nope", "distance", 64))
num("memScaleDistSixtyFour", mean_at("scale_main", "mqar", "nope_mem_front", "distance", 64))
num("swaMemScaleDistSixtyFour", mean_at("scale_main", "mqar", "swa_mem_front", "distance", 64))
num("memScaleLoadEight", mean_at("scale_main", "mqar", "nope_mem_front", "load", 8))
num("bestBaseScaleDistSixtyFour", max(mean_at("scale_main", "mqar", r, "distance", 64)
                                      for r in ("nope", "rope", "rope/pi", "rope/yarn", "alibi")))
num("bestBaseScaleLoadEight", max(mean_at("scale_main", "mqar", r, "load", 8)
                                  for r in ("nope", "rope", "rope/pi", "rope/yarn", "alibi")))
for row, rtag in (("nope_mem", "Middle"), ("swa_mem", "SwaMiddle"), ("nope_mem_back", "Back"),
                  ("nope_ctrl", "CtrlMiddle")):
    num(f"place{rtag}DistSixtyFour", mean_at("main", "mqar", row, "distance", 64))
    num(f"place{rtag}LoadEight", mean_at("main", "mqar", row, "load", 8))
for task, tag in (("copy", "Copy"), ("selective_copy", "Sel")):
    num(f"mem{tag}DistTwo", mean_at("main", task, "nope_mem_front", "distance", 2))
    num(f"infini{tag}DistTwo", mean_at("main", task, "infini", "distance", 2))
    num(f"alibi{tag}DistTwo", mean_at("main", task, "alibi", "distance", 2))
    num(f"mem{tag}Val", mean_at("main", task, "nope_mem_front", "val", 1))
    num(f"swaMem{tag}Val", mean_at("main", task, "swa_mem_front", "val", 1))
num("bestSelDistTwo", max(mean_at("main", "selective_copy", r, "distance", 2)
                          for r in BASELINE_ORDER if learned("main", "selective_copy", r)))


def solve_steps(phase, task, model):
    rs = runs.get((phase, task, model), [])
    return [r["train"]["steps_done"] for r in rs], [r["train"]["stop_reason"] for r in rs]


st, why = solve_steps("main", "mqar", "nope_mem_front")
num("frontSolveSteps", f"{min(st):,}--{max(st):,}" if min(st) != max(st) else f"{st[0]:,}", "{}")
st2, _ = solve_steps("main", "mqar", "nope_mem")
num("middleSolveSteps", f"{min(st2):,}--{max(st2):,}", "{}")

# ----------------------------------------------------------------------
# Language model

LM_ORDER = ["nope", "rope", "alibi", "swa", "infini", "nope_ctrl_front", "nope_mem_front",
            "swa_mem_front"]
lm = {m: runs.get(("lm_main", "enwik8", m), []) for m in LM_ORDER}
tl_lm = PROTOCOL["config"]["lm"]["train"]["train_len"]
edges = None
lines = [r"\begin{table}\centering\small", r"\setlength{\tabcolsep}{3.5pt}"]
rows_txt = []
for m in LM_ORDER:
    rs = lm[m]
    if not rs:
        continue
    bpc = [r["eval"]["val"]["bpc"] for r in rs]
    variants = sorted({k.split("/", 1)[1] for k in rs[0]["eval"] if k.startswith("position/")})
    for v in variants:
        b = np.array([[x["loss"] for x in r["eval"][f"position/{v}"]["buckets"]] for r in rs])
        edges = [(x["from"], x["to"]) for x in rs[0]["eval"][f"position/{v}"]["buckets"]]
        pk = [np.mean([r["eval"].get(f"passkey/{k}/{v}", {}).get("acc", np.nan) for r in rs])
              for k in (1, 4, 32)]
        label = NAMES[m] + ({"pi": " + PI", "yarn": " + YaRN"}.get(v, ""))
        if m in MEM:
            label = r"\textbf{" + label + "}"
        bits = b.mean(0) / np.log(2)
        rows_txt.append(label + " & " + (f"{np.mean(bpc):.3f}" if v == "base" else "")
                        + " & " + " & ".join(f"{x:.2f}" for x in bits)
                        + " & " + " & ".join(f"{100 * x:.0f}" for x in pk) + r" \\")
def _k(n):
    return f"{n // 1024}K" if n >= 1024 else str(n)


heads = [f"{_k(a)}--{_k(b)}" for a, b in edges]
lines += [r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{l r " + "r" * len(edges) + " rrr}", r"\toprule",
          r" & val & \multicolumn{" + str(len(edges)) + r"}{c}{test loss by position (bits/char)}"
          r" & \multicolumn{3}{c}{passkey (\%)} \\",
          r"Model & bpc & " + " & ".join(heads) + r" & $1\times$ & $4\times$ & $32\times$ \\",
          r"\midrule"] + rows_txt
lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
          rf"\caption{{Character-level enwik8, trained at {tl_lm} characters; means over three "
          r"seeds. Validation bits per character at the training length; test loss averaged "
          r"over position ranges of 16 sequences of $32\times$ the training length; passkey "
          r"digit accuracy on the exact prompt of~\citet{mohtashami2023landmark} at "
          r"$k\times$ the training length (32 prompts each, chance 10\%).}",
          r"\label{tab:lm}", r"\end{table}", ""]
write("lm.tex", "\n".join(lines))


def lm_bpc(m):
    return np.mean([r["eval"]["val"]["bpc"] for r in lm[m]])


def lm_bucket(m, i, v="base"):
    return np.mean([r["eval"][f"position/{v}"]["buckets"][i]["loss"] for r in lm[m]]) / np.log(2)


num("lmMemBpc", lm_bpc("nope_mem_front"), "{:.3f}")
num("lmSwaMemBpc", lm_bpc("swa_mem_front"), "{:.3f}")
num("lmAlibiBpc", lm_bpc("alibi"), "{:.3f}")
num("lmRopeBpc", lm_bpc("rope"), "{:.3f}")
num("lmNopeBpc", lm_bpc("nope"), "{:.2f}")
num("lmCtrlBpc", lm_bpc("nope_ctrl_front"), "{:.2f}")
last = len(edges) - 1
for m, tag in (("nope_mem_front", "Mem"), ("swa_mem_front", "SwaMem"), ("alibi", "Alibi"),
               ("nope_ctrl_front", "Ctrl"), ("infini", "Infini"), ("swa", "Swa")):
    num(f"lm{tag}First", lm_bucket(m, 0), "{:.2f}")
    num(f"lm{tag}Last", lm_bucket(m, last), "{:.2f}")
num("lmRopeYarnLast", lm_bucket("rope", last, "yarn"), "{:.2f}")
num("lmRopeLast", lm_bucket("rope", last), "{:.2f}")
num("lmMemPkOne", 100 * np.mean([r["eval"]["passkey/1/base"]["acc"] for r in lm["nope_mem_front"]]),
    "{:.0f}")

fig, ax = plt.subplots(figsize=(5.2, 3.0))
LMSTY = {"rope": ("#2980b9", "-", "base"), "alibi": ("#27ae60", "-", "base"),
         "swa": ("#7f8c8d", "-", "base"), "infini": ("#8e44ad", "-", "base"),
         "nope_ctrl_front": ("#95a5a6", "--", "base"), "nope_mem_front": ("#c0392b", "-", "base"),
         "swa_mem_front": ("#e67e22", "-", "base")}
for m, (c, ls, v) in LMSTY.items():
    rs = lm[m]
    pos = np.array(rs[0]["eval"][f"position/{v}"]["curve_pos"]) + 1
    y = np.mean([r["eval"][f"position/{v}"]["curve_loss"] for r in rs], axis=0) / np.log(2)
    ax.plot(pos, y, color=c, ls=ls, lw=2 if m in MEM else 1.1, label=NAMES[m])
ax.axvline(tl_lm, color="#bbb", lw=0.8, ls=":")
ax.set_xscale("log", base=2)
ax.set_xlim(16, None)
ax.set_ylim(0.5, 5)
ax.set_xlabel("position (training length 512)")
ax.set_ylabel("test loss (bits/char)")
ax.legend(frameon=False, fontsize=7, ncol=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIGS / "lm_position.pdf")
plt.close(fig)

# ----------------------------------------------------------------------
# Admitted bank (mechanism records, batch 1)

TASKS = [("mqar", "MQAR"), ("passkey", "Passkey"), ("copy", "Copying"),
         ("selective_copy", "Selective copying")]
lines = [r"\begin{table}\centering\small", r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Task & Model & $1\times$ & $4\times$ & $16\times$ & $64\times$ \\", r"\midrule"]
frac = {}
for task, tname in TASKS:
    for m in ("nope_mem_front", "swa_mem_front"):
        rs = runs.get(("main", task, m), [])
        cells = []
        for k in (1, 4, 16, 64):
            per_seed = []
            for r in rs:
                it = r["eval"].get(f"mech/{k}/base")
                if it and it.get("mech"):
                    per_seed.append(np.mean([s[0]["admitted_bank"] for s in it["mech"]]) / it["length"])
            frac[(task, m, k)] = np.mean(per_seed)
            cells.append(f"{np.mean(per_seed):.2f}")
        lines.append(f"{tname} & {NAMES[m]} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Admitted bank as a fraction of the sequence length, at $k\times$ the "
          r"training length (8 sequences per point, read one at a time; mean over seeds).}",
          r"\label{tab:bank}", r"\end{table}", ""]
write("bank.tex", "\n".join(lines))
num("bankPasskeyMem", frac[("passkey", "nope_mem_front", 64)], "{:.2f}")
num("bankPasskeySwaMem", frac[("passkey", "swa_mem_front", 64)], "{:.2f}")

# ----------------------------------------------------------------------
# Selected learning rates and sizes (appendix)

lines = [r"\begin{table}\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Model & Params & MQAR & Passkey & Copying & Sel.\ copying & enwik8 \\", r"\midrule"]
for m in ["nope", "rope", "alibi", "swa", "infini", "nope_ctrl_front", "nope_mem_front",
          "swa_mem_front", "nope_ctrl", "nope_mem", "swa_mem", "nope_mem_back", "rope_yarn_ft"]:
    p = next((r["params"] for k, v in runs.items() if k[0] == "main" and k[2] == m for r in v), None)
    cells = []
    for task in ("mqar", "passkey", "copy", "selective_copy"):
        rs = runs.get(("main", task, m), [])
        cells.append(f"{rs[0]['lr']:g}" if rs else "--")
    rs = lm.get(m) or []
    cells.append(f"{rs[0]['lr']:g}" if rs else "--")
    lines.append(f"{NAMES[m]} & {p / 1e6:.2f}M & " + " & ".join(cells) + r" \\" if p else "")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Parameter counts (MQAR vocabulary) and the learning rate selected for each "
          r"task by in-distribution validation. The fine-tuned YaRN model uses the rate selected "
          r"for its RoPE source.}", r"\label{tab:lr}", r"\end{table}", ""]
write("lr.tex", "\n".join(lines))

mem_p = next(r["params"] for r in runs[("main", "mqar", "nope_mem_front")])
nope_p = next(r["params"] for r in runs[("main", "mqar", "nope")])
ctrl_p = next(r["params"] for r in runs[("main", "mqar", "nope_ctrl_front")])
num("memParams", f"{(mem_p - nope_p):,}", "{}")
num("ctrlParams", f"{(ctrl_p - nope_p):,}", "{}")
num("memParamFrac", 100 * (mem_p - nope_p) / nope_p)
num("runsTotal", f"{len(RESULTS)}", "{}")

# ----------------------------------------------------------------------

(TABLES / "numbers.tex").write_text(
    "% Generated by generate_results.py from the run records; do not edit.\n"
    + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(NUM.items())))
print(f"{len(NUM)} numbers, tables in {TABLES}, figures in {FIGS}")
