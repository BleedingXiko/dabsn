"""Tables for the paper from a results directory. Standard library only.

    python analysis.py ../results_all.json.gz          # writes ../results/analysis/ (results directory: RESULTS/analysis/)
    python analysis.py ../results_all.json.gz --paper  # also writes ../results/generated.tex for main.tex

Every declared job must have passed. A failure, a missing job, a mixed precision,
or mixed reproduction source within one study stops the analysis and is named.
Intervals are descriptive two-sided Student t intervals over independent training
(or input) seeds, without multiplicity adjustment.
"""
from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matrix  # noqa: E402

T975 = {1: 12.706204736174698, 2: 4.302652729749464, 3: 3.182446305284263, 4: 2.7764451051977934,
        5: 2.570581835636314, 6: 2.4469118511449697, 7: 2.364624251592785, 8: 2.306004135204166,
        9: 2.2621571627982053, 10: 2.2281388519649385}


def interval(values):
    values = list(values)
    if len(values) < 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError(f"interval needs at least two finite values: {values}")
    mean = statistics.mean(values)
    half = T975[len(values) - 1] * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half, "n": len(values),
            "seed_values": values}


class Records:
    def __init__(self, results: Path, m: dict, selections: dict):
        self.results = results; self.m = m; self.selections = selections
        self.precisions = {}; self.sources = {}

    def get(self, stage, job):
        import importlib
        module = importlib.import_module(f"studies.{matrix.stage(stage)[1]}")
        directory = self.results / matrix.record_study(stage) / module.identifier(job)
        attempts = sorted(directory.glob("attempt-*.json"))
        if not attempts:
            raise ValueError(f"{stage}: {module.identifier(job)} has not run")
        record = json.loads(attempts[-1].read_text())
        if record["status"] != "passed":
            raise ValueError(f"{stage}: {module.identifier(job)} ended {record['status']}: {record.get('error')}")
        if record["job"] != job:
            raise ValueError(f"{stage}: {module.identifier(job)} recorded a different job")
        self.precisions.setdefault(stage, set()).add(record["precision"]["compute"])
        self.sources.setdefault(stage, set()).add(record["reproduction_source_sha256"])
        return record

    def jobs(self, stage):
        import run
        _, jobs = run.stage_jobs(stage, self.m, self.selections)
        if jobs is None:
            raise ValueError(f"{stage} needs a selection that has not been made")
        return jobs

    def all(self, stage):
        return [(job, self.get(stage, job)) for job in self.jobs(stage)]


CLASSIFICATION_TASKS = ("temporal_order",)


def metric_name(task):
    return "accuracy" if task in CLASSIFICATION_TASKS else "normalized_mse"


def better(task, a, b):
    """Positive when a is better than b."""
    return a - b if task in CLASSIFICATION_TASKS else b - a


# ------------------------------------------------------------- studies

def delayed(r):
    p = r.m["delayed"]; main = p["main"]; rates = r.selections["delayed_rates"]["selected_rates"]
    rows = []; differences = []; diagnostic = []
    for task in p["tasks"]:
        initial = {}
        for length in main["final_evaluation_lengths"]:
            by_method = {}
            for method in main["methods"]:
                full, occluded = [], []
                for seed in main["seeds"]:
                    job = {"phase": "main", "task": task, "seed": seed, "method": method, "lr": rates[task]}
                    result = r.get("delayed", job)["result"]
                    initial.setdefault(seed, set()).add(result["initial_parameters_sha256"])
                    full.append(result["final_by_length"][str(length)][metric_name(task)])
                    occluded.append(result["final_chunk_only_by_length"][str(length)][metric_name(task)])
                by_method[method] = full
                rows.append({"task": task, "length": length, "method": method,
                             "metric": metric_name(task), **interval(full)})
                diagnostic.append({"task": task, "length": length, "method": method,
                                   "metric": "full-input advantage over final-chunk-only input",
                                   **interval([better(task, a, b) for a, b in zip(full, occluded)])})
            for method in main["methods"]:
                if method != "online":
                    differences.append({"task": task, "length": length, "comparison": f"online_vs_{method}",
                                        "positive_means": "online better",
                                        **interval([better(task, a, b) for a, b in
                                                    zip(by_method["online"], by_method[method])])})
        if any(len(v) != 1 for v in initial.values()):
            raise ValueError("Matched delayed methods did not start from identical parameters")
    return {"rows": rows, "paired_differences": differences, "final_chunk_input_diagnostic": diagnostic,
            "selected_rates": rates, "rate_selection": r.selections["delayed_rates"]}


def ablation(r):
    p = r.m["ablation"]; rates = r.selections["delayed_rates"]["selected_rates"]
    lengths = r.m["delayed"]["main"]["final_evaluation_lengths"]
    groups = [("chunk", p["chunk"]["task"], p["chunk"]["input_dim"], p["chunk"]["state_dim"], chunk,
               p["chunk"]["seeds"], "reset") for chunk in p["chunk"]["chunks"]]
    groups += [("shape", task, shape[0], shape[1], p["shape"]["chunk"], p["shape"]["seeds"], "bptt")
               for task in p["shape"]["tasks"] for shape in p["shape"]["shapes"]]
    rows = []
    for study, task, inputs, state, chunk, seeds, comparator in groups:
        for length in lengths:
            scores = {}
            for method in ("online", comparator):
                scores[method] = [r.get("ablation", {"phase": "ablation", "study": study, "task": task, "seed": seed,
                                                     "method": method, "lr": rates[task], "input_dim": inputs,
                                                     "state_dim": state, "chunk": chunk})["result"][
                                      "final_by_length"][str(length)][metric_name(task)] for seed in seeds]
                rows.append({"study": study, "task": task, "input_dim": inputs, "state_dim": state, "chunk": chunk,
                             "length": length, "method": method, "metric": metric_name(task),
                             **interval(scores[method])})
            rows.append({"study": study, "task": task, "input_dim": inputs, "state_dim": state, "chunk": chunk,
                         "length": length, "method": f"online_minus_{comparator}",
                         "metric": "paired advantage; positive favors online",
                         **interval([better(task, a, b) for a, b in zip(scores["online"], scores[comparator])])})
    return {"rows": rows}


def interventions(r):
    """Each edit of E against the carried branch at a frozen checkpoint: core credit cosine and norm
    ratio (the edit's direct effect on credit) and post-update held-out loss. Per seed, batches are
    averaged; ``pooled`` also averages the evaluation lengths."""
    p = r.m["interventions"]; rates = r.selections["delayed_rates"]["selected_rates"]
    metrics = {"core credit cosine to carry": lambda b, c: b["core_credit_against_carry"]["core_cosine"],
               "core credit norm ratio to carry": lambda b, c: b["core_credit_against_carry"]["core_norm_ratio"],
               "post-update loss minus carry": lambda b, c: (b["true_optimizer_post_update"]["loss"]
                                                             - c["true_optimizer_post_update"]["loss"])}

    def per_seed(task, suffix, step, lengths, condition, metric):
        values = []
        for seed in p["training_seeds"]:
            per_batch = []
            for length in lengths:
                for batch in r.get("interventions", {"task": task, "seed": seed, "checkpoint_step": step,
                                                     "length": length, "boundary_suffix": suffix,
                                                     "source_lr": rates[task]})["result"]["batches"]:
                    indexed = {branch["condition"]: branch for branch in batch["conditions"]}
                    value = metrics[metric](indexed[condition], indexed["carry"])
                    if value is not None:
                        per_batch.append(value)
            if not per_batch:
                raise ValueError(f"interventions: {task} b{suffix} c{step} {condition} has no defined {metric}")
            values.append(statistics.mean(per_batch))
        return values

    rows, pooled = [], []
    for task, suffix, step in itertools.product(p["tasks"], p["boundary_suffixes"], p["checkpoint_steps"]):
        for condition in [c for c in p["conditions"] if c != "carry"]:
            key = {"task": task, "boundary_suffix": suffix, "checkpoint_step": step, "condition": condition}
            for metric in metrics:
                for length in p["lengths"]:
                    rows.append({**key, "length": length, "metric": metric,
                                 **interval(per_seed(task, suffix, step, [length], condition, metric))})
                pooled.append({**key, "metric": metric, **interval(per_seed(task, suffix, step, p["lengths"], condition, metric))})
    isolated = []
    for task, step in itertools.product(p["tasks"], p["checkpoint_steps"]):
        for metric in ("negate_w_cosine_to_carry", "zero_norm_ratio_to_carry"):
            values = []
            for seed in p["training_seeds"]:
                per_batch = [entry["isolated_w_pathway"][metric]
                             for length in p["lengths"]
                             for entry in r.get("interventions", {"task": task, "seed": seed, "checkpoint_step": step,
                                                                  "length": length, "boundary_suffix": 1,
                                                                  "source_lr": rates[task]})["result"]["immediate_only"]
                             if entry["isolated_w_pathway"][metric] is not None]
                if not per_batch:
                    raise ValueError(f"interventions: {task} c{step} has no defined isolated {metric}")
                values.append(statistics.mean(per_batch))
            isolated.append({"task": task, "checkpoint_step": step, "metric": metric, **interval(values)})
    return {"rows": rows, "pooled": pooled, "isolated_w_pathway": isolated}


# The stream studies' reported arms. Their records also hold a superseded normalized arm
# (carry_normalized); the paper reports the native action from the stream-normalized stage.
STREAM_ARMS = ("carry", "reset", "permute", "frozen_core", "tbptt")


def _stream(r, study):
    """Both stream benchmarks: every reported arm by seed, and paired contrasts against carried
    E (positive favors carry: higher memory capacity, lower normalized MSE)."""
    p = dict(r.m[study]); p["methods"] = [m for m in p["methods"] if m in STREAM_ARMS]
    capacity = p["benchmark"] == "memory_capacity"

    def final(method, seed):
        record = r.get(study, {"phase": "main", "method": method, "seed": seed})["result"]
        return record, record["final"]

    for seed in p["seeds"]:
        if len({final(m, seed)[0]["initial_parameters_sha256"] for m in p["methods"]}) != 1:
            raise ValueError(f"Matched {study} arms did not start from identical parameters")
    metric = "memory_capacity_total" if capacity else "nmse"
    rows = []; contrasts = []
    scores = {m: [final(m, seed)[1][metric] for seed in p["seeds"]] for m in p["methods"]}
    for method in p["methods"]:
        rows.append({"method": method, "metric": metric, **interval(scores[method])})
        if capacity:
            for k in p["delays"]:
                rows.append({"method": method, "metric": f"memory_capacity_delay_{k}",
                             **interval([final(method, seed)[1]["memory_capacity_by_delay"][str(k)]
                                         for seed in p["seeds"]])})
    for method in p["methods"]:
        if method == "carry":
            continue
        signed = [(a - b) if capacity else (b - a) for a, b in zip(scores["carry"], scores[method])]
        contrasts.append({"contrast": f"carry versus {method}", "metric": metric,
                          "positive_favors": "carry", **interval(signed)})
    bound = []
    for method in p["methods"]:
        for seed in p["seeds"]:
            record = final(method, seed)[0]
            b = record.get("eligibility_bound") or {}
            t = record.get("transport_diagnostic") or {}
            bound.append({"method": method, "seed": seed,
                          "rescaling_events": b.get("rescaling_events"),
                          "bounded_units": len(b.get("bounded_units") or []),
                          "peak_unit_trace_norm": b.get("peak_unit_trace_norm"),
                          "units_with_expanding_trace": t.get("units_with_expanding_trace"),
                          "fastest_unit_trace_log_growth_per_step": t.get("fastest_unit_trace_log_growth_per_step"),
                          "exact_state_jacobian_log_growth_per_step": t.get("exact_state_jacobian_log_growth_per_step")})
    return {"benchmark": p["benchmark"], "rows": rows, "paired_contrasts": contrasts,
            "eligibility_bound_and_transport": bound,
            "frozen_core_bitwise_unchanged": all(final("frozen_core", seed)[0]["frozen_core_bitwise_unchanged"]
                                                  for seed in p["seeds"]) if "frozen_core" in p["methods"] else None}


def alignment(r):
    """Online versus exact credit per parameter family at trained stream weights."""
    p = r.m["alignment"]; rows = []
    for study in p["studies"]:
        records = [r.get("alignment", {"study": study, "seed": seed})["result"] for seed in p["seeds"]]
        for family in sorted(records[0]["by_family"]):
            cos = [x["by_family"][family]["mean_cosine"] for x in records]
            ratio = [x["by_family"][family]["mean_norm_ratio"] for x in records]
            if any(v is None for v in cos + ratio):
                continue
            rows.append({"study": study, "family": family, "metric": "cosine", **interval(cos)})
            rows.append({"study": study, "family": family, "metric": "norm ratio online/exact", **interval(ratio)})
    return {"rows": rows, "window_steps": records[0]["window_steps"]}


def stream_window(r):
    """Horizon-matched pairs: one-chunk exact BPTT versus erased E (same reach), and the
    declared truncated window versus carried E. Positive favors exact credit."""
    p = r.m["stream_window"]; rows = []
    for study in p["studies"]:
        capacity = r.m[study]["benchmark"] == "memory_capacity"
        metric = "memory_capacity_total" if capacity else "nmse"
        def score(stage, job):
            return r.get(stage, job)["result"]["final"][metric]
        exact_short = [score("stream-window", {"phase": "main", "method": "tbptt", "seed": seed, "study": study,
                                               "window_chunks": p["window_chunks"]}) for seed in p["seeds"]]
        erased = [score(study, {"phase": "main", "method": "reset", "seed": seed}) for seed in p["seeds"]]
        exact_long = [score(study, {"phase": "main", "method": "tbptt", "seed": seed}) for seed in p["seeds"]]
        carried = [score(study, {"phase": "main", "method": "carry", "seed": seed}) for seed in p["seeds"]]
        sign = 1 if capacity else -1
        rows.append({"study": study, "reach": f"{p['window_chunks']} chunk", "arm": "exact BPTT",
                     "metric": metric, **interval(exact_short)})
        rows.append({"study": study, "reach": f"{p['window_chunks']} chunk", "arm": "erased E",
                     "metric": metric, **interval(erased)})
        rows.append({"study": study, "reach": f"{p['window_chunks']} chunk", "arm": "exact minus E (positive favors exact)",
                     "metric": metric, **interval([sign * (a - b) for a, b in zip(exact_short, erased)])})
        long_window = r.m[study]["tbptt_window_chunks"]
        rows.append({"study": study, "reach": f"{long_window} chunks vs unbounded", "arm": "exact BPTT minus carried E "
                     "(positive favors exact)", "metric": metric,
                     **interval([sign * (a - b) for a, b in zip(exact_long, carried)])})
    return {"rows": rows}


def stream_normalized(r):
    """The native action on the core (readout under the stream arms' Adam regime) against
    carried E with Adam on every parameter. A diverged seed is reported as diverged, never
    dropped."""
    p = r.m["stream_normalized"]; rows = []
    chosen = r.selections["stream_native_settings"]["selected_rates"]
    for study in p["studies"]:
        capacity = r.m[study]["benchmark"] == "memory_capacity"
        metric = "memory_capacity_total" if capacity else "nmse"
        native = [r.get("stream-normalized", {"phase": "main", "method": "carry_normalized", "seed": seed,
                                              "study": study, **chosen[study]})["result"] for seed in p["seeds"]]
        diverged = [x.get("diverged", False) for x in native]
        adam = [r.get(study, {"phase": "main", "method": "carry", "seed": seed})["result"]["final"][metric]
                for seed in p["seeds"]]
        sign = 1 if capacity else -1
        rows.append({"study": study, "arm": "carried E with Adam", "metric": metric, **interval(adam)})
        if any(diverged):
            rows.append({"study": study, "arm": "native action on the core", "metric": metric,
                         "diverged_seeds": [s for s, d in zip(p["seeds"], diverged) if d]})
            continue
        values = [x["final"][metric] for x in native]
        rows.append({"study": study, "arm": "native action on the core", "metric": metric, **interval(values)})
        rows.append({"study": study, "arm": "native minus Adam (positive favors native)", "metric": metric,
                     **interval([sign * (a - b) for a, b in zip(values, adam)])})
    return {"rows": rows, "native_settings": chosen, "selection": r.selections["stream_native_settings"]}


def conjunction(r):
    """The core driven by E (credit, action, pace) against AdamW with online credit and with BPTT;
    the read and readout share the AdamW regime in every arm."""
    p = r.m["conjunction"]; rows = []; efficiency = []
    chosen = r.selections["conjunction_settings"]["selected_rates"]

    def job(task, seed, arm):
        out = {"phase": "main", "task": task, "seed": seed, "arm": arm}
        if arm == "native":
            out.update(rate=chosen[task]["rate"], relative_floor=chosen[task]["relative_floor"])
        return out
    for task in p["tasks"]:
        records = {arm: [r.get("conjunction", job(task, seed, arm))["result"]
                         for seed in p["seeds"]] for arm in p["arms"]}
        lengths = [e["eval_length"] for e in records[p["arms"][0]][0]["evaluation"]]
        for length in lengths:
            scores = {arm: [next(e["eval_accuracy"] for e in x["evaluation"] if e["eval_length"] == length)
                            for x in records[arm]] for arm in p["arms"]}
            for arm in p["arms"]:
                rows.append({"task": task, "length": length, "arm": arm, **interval(scores[arm])})
            for other in ("adamw_online", "adamw_bptt"):
                rows.append({"task": task, "length": length, "arm": f"native_minus_{other}",
                             **interval([a - b for a, b in zip(scores["native"], scores[other])])})
        for arm in p["arms"]:
            for seed, x in zip(p["seeds"], records[arm]):
                reached = next((row["step"] for row in x["rows"] if row["accuracy"] >= 0.99), None)
                efficiency.append({"task": task, "arm": arm, "seed": seed, "first_update_at_0_99": reached})
    return {"rows": rows, "data_efficiency": efficiency, "native_settings": chosen,
            "selection": r.selections["conjunction_settings"]}


def continuous(r):
    return _stream(r, "continuous")


def sustained(r):
    return _stream(r, "sustained")


def continuity(r):
    p = r.m["continuity"]; margin = p["equivalence_accuracy_margin_absolute"]; rows = []
    for task in p["suite_tasks"]:
        for length in p["tasks"][task]["eval_lengths"]:
            scores = {}
            for method in [*p["methods"], *p.get("control_methods", [])]:
                scores[method] = [next(row["eval_accuracy"] for row in r.get(
                    "continuity", {"task": task, "seed": seed, "method": method})["result"]["evaluation"]
                    if row["eval_length"] == length) for seed in p["seeds"]]
                rows.append({"task": task, "length": length, "method": method, "metric": "held-out accuracy",
                             **interval(scores[method])})
            if "frozen_core" in scores:
                rows.append({"task": task, "length": length, "method": "online_minus_frozen_core",
                             "metric": "paired accuracy difference; positive: core learning credited by E helps",
                             **interval([a - b for a, b in zip(scores["online"], scores["frozen_core"])])})
            estimate = interval([a - b for a, b in zip(scores["online"], scores["bptt"])])
            rows.append({"task": task, "length": length, "method": "online_minus_bptt",
                         "metric": "paired accuracy difference", "declared_margin": -margin,
                         "no_loss_larger_than_margin_supported": estimate["ci95_low"] > -margin, **estimate})
    cohort = []
    for seed in [p["a5_development_seed"], *p["a5_heldout_seeds"]]:
        result = r.get("continuity", {"task": "a5", "seed": seed, "method": "online"})["result"]
        cohort.append({"seed": seed, "role": result["role"], "lr": result["configuration"]["lr"],
                       "exact_accuracy_at_every_length": result["a5_exact_reproduction_passed"],
                       "evaluation": result["evaluation"]})
    heldout = [row for row in cohort if row["role"] == "heldout"]
    a5_rows = []
    if len(heldout) >= 2:
        for index, length in enumerate(p["tasks"]["a5"]["eval_lengths"]):
            a5_rows.append({"length": length, **interval([row["evaluation"][index]["eval_accuracy"] for row in heldout])})
    return {"rows": rows, "a5_cohort": cohort, "a5_heldout_accuracy": a5_rows,
            "equivalence_rule": p["equivalence_interpretation"]}


def resource(r):
    rows = []
    for job, record in r.all("resource"):
        result = record["result"]
        rows.append({**job, "median_update_seconds": result["median_update_seconds"],
                     "eligibility_bytes_formula": result["eligibility_bytes_formula"],
                     "peak_allocated_bytes_max": max((row["peak_allocated_bytes"] or 0) for row in result["replicates"])})
    return {"rows": rows}


def scaling(r):
    rows = []
    for job, record in r.all("scaling"):
        result = record["result"]
        rows.append({**job, "median_seconds": result["median_seconds"], "tokens_per_second": result["tokens_per_second"],
                     "eligibility_bytes_formula": result["eligibility_bytes_formula"],
                     "peak_allocated_bytes_max": max((v or 0) for v in result["peak_allocated_bytes"])})
    return {"rows": rows}


def action(r):
    p = r.m["action"]; values = {}
    for seed, relative in itertools.product(p["seeds"], p["teacher_relative_perturbations"]):
        case = r.get("action", {"seed": seed, "teacher_relative_perturbation": relative})["result"]
        for row in case["rows"]:
            values[(seed, relative, row["method"], row["action"])] = row
    rows = []
    for relative, method, step in itertools.product(p["teacher_relative_perturbations"], p["methods"],
                                                    range(1, p["repeated_actions"] + 1)):
        rows.append({"teacher_relative_perturbation": relative, "method": method, "action": step,
                     **{metric: interval([values[(seed, relative, method, step)][metric] for seed in p["seeds"]])
                        for metric in ("l2_error_reduction_from_initial", "error_ratio_to_initial",
                                       "parameter_displacement_l2")}})
    return {"rows": rows, "independent_unit": "core and input seed"}


def continuation_like(rows_by_key, controls, seeds, goals):
    summaries = []; comparisons = []
    for control in controls:
        observed = [statistics.mean(rows_by_key[(seed, goal, control)]["pre_action_mse_last_half"] for goal in goals)
                    for seed in seeds]
        summaries.append({"control": control, **interval(observed)})
        if control != "retained":
            comparisons.append({"contrast": f"retained versus {control}", "positive_favors": "retained",
                                **interval([statistics.mean(rows_by_key[(seed, goal, control)]["pre_action_mse_last_half"]
                                                            - rows_by_key[(seed, goal, "retained")]["pre_action_mse_last_half"]
                                                            for goal in goals) for seed in seeds])})
    return {"rows": summaries, "paired_comparisons": comparisons}


def continuation(r):
    p = r.m["continuation"]; rows = {}
    for seed in p["seeds"]:
        for row in r.get("continuation", {"seed": seed})["result"]["rows"]:
            rows[(row["seed"], row["objective"], row["control"])] = row
    return continuation_like(rows, p["controls"], p["seeds"], p["objectives"])


def directions(r):
    p = r.m["directions"]
    result = r.get("directions", r.jobs("directions")[0])["result"]
    rows = {(row["seed"], row["angle_pi"], row["control"]): row for row in result["rows"]}
    return {**continuation_like(rows, p["controls"], p["test_seeds"], p["test_angles_pi"]),
            "calibration": result["calibration"]}


def exact(r):
    out = {}
    for stage in ("oracles", "ontology"):
        out[stage] = {list(job.values())[0]: record["result"] for job, record in r.all(stage)}
    return out


SUMMARIES = {"oracles+ontology": exact, "action": action, "continuation": continuation, "directions": directions,
             "scaling": scaling, "resource": resource, "continuous": continuous,
             "delayed": delayed, "interventions": interventions, "ablation": ablation, "sustained": sustained,
             "alignment": alignment, "stream-window": stream_window, "stream-normalized": stream_normalized, "conjunction": conjunction,
             "continuity": continuity}


# ---------------------------------------------------------------- LaTeX

def fmt(x, digits=3):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if abs(x) >= 1e4 or (x != 0 and abs(x) < 1e-3):
        mantissa, exponent = f"{x:.{digits - 1}e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{x:.{digits}f}"


def ci(row):
    return f"{fmt(row['mean'])} [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"


def pivot(rows, key, column, label):
    """Rows keyed by key(r) with one column per column(r) (first-seen order): (header, body)."""
    keys = list(dict.fromkeys(key(r) for r in rows)); columns = list(dict.fromkeys(column(r) for r in rows))
    cells = {(key(r), column(r)): ci(r) for r in rows}
    return ([*label, *(str(c).replace("_", " ") for c in columns)],
            [(*(k if isinstance(k, tuple) else (k,)), *(cells.get((k, c), "--") for c in columns)) for k in keys])


def table(caption, header, rows):
    columns = "l" * len(header)

    def cell(value):
        text = str(value)
        return text if "$" in text or "\\" in text else text.replace("_", " ")
    body = "\n".join(" & ".join(cell(c) for c in row) + r" \\" for row in rows)
    return (f"\\begin{{table}}[tbp]\\centering\\small\\caption{{{caption}}}\n"
            f"\\begin{{adjustbox}}{{max width=\\linewidth}}\\begin{{tabular}}{{{columns}}}\\toprule\n"
            + " & ".join(header) + r" \\ \midrule" + "\n" + body
            + "\n\\bottomrule\\end{tabular}\\end{adjustbox}\\end{table}\n")


def latex(summary):
    out = ["% Generated by reproduce/analysis.py from passed records. Do not edit.\n"]
    studies = summary["studies"]
    run = summary["run"]
    out.append(f"All numbers below come from one results directory: device {run['device']}, "
               f"{run['precision']} core compute, framework tree \\texttt{{{run['framework_tree_sha256'][:12]}}}. "
               "Brackets are descriptive 95\\% Student intervals over independent seeds.\n\n")
    if "delayed" in studies:
        d = studies["delayed"]
        out.append(table("Central delayed learning: final held-out score (accuracy for the temporal order problem, "
                         "MSE over 1/6 for the adding problem).",
                         ["task", "length", "method", "mean [95\\% CI]"],
                         [(r["task"].replace("_", " "), r["length"], r["method"], ci(r)) for r in d["rows"]]))
        out.append(table("Paired differences, online minus comparator (positive favors online).",
                         ["task", "length", "comparison", "mean [95\\% CI]"],
                         [(r["task"].replace("_", " "), r["length"], r["comparison"].replace("_", " "), ci(r))
                          for r in d["paired_differences"]]))
    if "ablation" in studies:
        out.append(table("Shapes: final held-out score of online credit and BPTT at $(I,S)=(64,512)$ and "
                         "$(128,256)$, one chunk per episode (accuracy for the temporal order problem, MSE over "
                         "1/6 for the adding problem; paired advantage positive favors online).",
                         ["task", "$(I,S)$", "length", "method", "mean [95\\% CI]"],
                         [(r["task"].replace("_", " "), f"({r['input_dim']},{r['state_dim']})", r["length"],
                           r["method"].replace("_", " "), ci(r)) for r in studies["ablation"]["rows"]]))
    if "interventions" in studies:
        pooled = {(r["task"], r["boundary_suffix"], r["checkpoint_step"], r["condition"], r["metric"]): r
                  for r in studies["interventions"]["pooled"]}
        keys = list(dict.fromkeys(k[:4] for k in pooled))
        out.append(table("Frozen-checkpoint edits of $E$ at a boundary 1 or 32 steps before the query, against "
                         "the carried branch: the core's credit (cosine and norm ratio) and held-out loss after one "
                         "true update (positive favors carry); seeds averaged over evaluation lengths.",
                         ["task", "boundary", "update", "edit", "credit cosine", "credit norm ratio",
                          "post-update loss $-$ carry"],
                         [(k[0].replace("_", " "), k[1], k[2], k[3].replace("_", " "),
                           ci(pooled[(*k, "core credit cosine to carry")]),
                           ci(pooled[(*k, "core credit norm ratio to carry")]),
                           ci(pooled[(*k, "post-update loss minus carry")])) for k in keys]))
        iso = {(r["task"], r["checkpoint_step"], r["metric"]): r for r in studies["interventions"]["isolated_w_pathway"]}
        out.append(table("Erased isolated historical pathway at the checkpoint (1-step boundary, final input "
                         "zeroed): $W$'s credit is purely historical.",
                         ["task", "update", "cosine, negated $W$ trace to carry", "norm ratio, erased $E$ to carry"],
                         [(t.replace("_", " "), u, ci(iso[(t, u, "negate_w_cosine_to_carry")]),
                           ci(iso[(t, u, "zero_norm_ratio_to_carry")]))
                          for t, u in dict.fromkeys((k[0], k[1]) for k in iso)]))
    for study, caption in (("continuous", "Memory capacity on uninterrupted streams (stock core, no read)."),
                           ("sustained", "NARMA-10 on uninterrupted streams (stock core, no read): normalized MSE.")):
        if study in studies:
            header, body = pivot(studies[study]["rows"], lambda r: r["metric"].replace("_", " "),
                                 lambda r: r["method"], ["metric"])
            out.append(table(caption + " Mean [95\\% CI] by arm.", header, body))
            carried = [r for r in studies[study]["eligibility_bound_and_transport"]
                       if r["exact_state_jacobian_log_growth_per_step"] is not None]
            if carried:
                out.append(table(caption.split(" (")[0] + ": per-unit bound of carried $E$ and the transport "
                                 "diagnostic at final weights (log-growth per step; Jacobian rate $<0$: full "
                                 "dynamics contract).",
                                 ["arm", "seed", "rescalings", "units", "expanding units", "fastest unit rate",
                                  "Jacobian rate"],
                                 [(r["method"].replace("_", " "), r["seed"], r["rescaling_events"], r["bounded_units"],
                                   r["units_with_expanding_trace"], fmt(r["fastest_unit_trace_log_growth_per_step"], 4),
                                   fmt(r["exact_state_jacobian_log_growth_per_step"], 4)) for r in carried]))
            out.append(table(caption.split(" (")[0] + ": paired contrasts (positive favors carried $E$).",
                             ["contrast", "mean [95\\% CI]"],
                             [(r["contrast"].replace("_", " "), ci(r)) for r in studies[study]["paired_contrasts"]]))
    if "conjunction" in studies:
        c = studies["conjunction"]
        header, body = pivot(c["rows"], lambda x: (x["task"], x["length"]), lambda x: x["arm"], ["task", "length"])
        out.append(table("The conjunction: learning driven entirely by $E$ (native) against AdamW with online "
                         "credit and with BPTT; held-out accuracy by evaluation length, mean [95\\% CI] by arm.",
                         header, body))
        out.append(table("Data efficiency: first scheduled update reaching 0.99 held-out accuracy at the "
                         "training length (-- never).", ["task", "arm", "seed", "update"],
                         [(x["task"], x["arm"].replace("_", " "), x["seed"],
                           "--" if x["first_update_at_0_99"] is None else x["first_update_at_0_99"])
                          for x in c["data_efficiency"]]))
    if "stream-normalized" in studies:
        out.append(table("Moment-free normalized action on the stream benchmarks.",
                         ["benchmark", "arm", "mean [95\\% CI]"],
                         [(r["study"], r["arm"], ci(r)) for r in studies["stream-normalized"]["rows"]]))
    if "stream-window" in studies:
        out.append(table("Stream benchmarks at matched credit reach.",
                         ["benchmark", "reach", "arm", "mean [95\\% CI]"],
                         [(r["study"], r["reach"], r["arm"], ci(r)) for r in studies["stream-window"]["rows"]]))
    if "alignment" in studies:
        out.append(table("Online versus exact credit at trained stream weights, by parameter family "
                         f"(windows of {studies['alignment']['window_steps']} steps).",
                         ["benchmark", "family", "metric", "mean [95\\% CI]"],
                         [(r["study"], r["family"].replace("_", " "), r["metric"], ci(r))
                          for r in studies["alignment"]["rows"]]))
    if "continuity" in studies:
        c = studies["continuity"]
        out.append(table("Continuity with paper 1: online minus BPTT accuracy.",
                         ["task", "length", "mean [95\\% CI]", "no loss $>$ 2 pts supported"],
                         [(r["task"], r["length"], ci(r), fmt(r["no_loss_larger_than_margin_supported"]))
                          for r in c["rows"] if r["method"] == "online_minus_bptt"]))
        out.append(table("A5/60 frozen-rate cohort (online).", ["seed", "role", "exact at every length"],
                         [(r["seed"], r["role"], fmt(r["exact_accuracy_at_every_length"])) for r in c["a5_cohort"]]))
        development = next(r for r in c["a5_cohort"] if r["role"] != "heldout")
        out.append(table("A5/60 accuracy by evaluation length under online credit (trained at 256).",
                         ["length", "held-out seeds, mean [95\\% CI]", f"development seed {development['seed']}"],
                         [(r["length"], ci(r), fmt(e["eval_accuracy"]))
                          for r, e in zip(c["a5_heldout_accuracy"], development["evaluation"])]))
    for name, caption in (("continuation", "Objective-conditioned continuation: last-half pre-action MSE."),
                          ("directions", "Held-out objective directions: last-half pre-action MSE.")):
        if name in studies:
            out.append(table(caption, ["control", "mean [95\\% CI]"],
                             [(r["control"].replace("_", " "), ci(r)) for r in studies[name]["rows"]]))
    if "action" in studies:
        out.append(table("Teacher matching by direct action: $\\ell_2$ error reduction after each action.",
                         ["perturbation", "method", "action", "mean [95\\% CI]"],
                         [(r["teacher_relative_perturbation"], r["method"].replace("_", " "), r["action"],
                           ci(r["l2_error_reduction_from_initial"])) for r in studies["action"]["rows"]]))
    if summary["incomplete"]:
        out.append("\\paragraph{Not yet complete.} " + "; ".join(
            f"{k.replace('_', ' ')}" for k in sorted(summary["incomplete"])) + ".\n")
    return "".join(out)


def main():
    from collect import results_directory
    keep = []  # results_all.json.gz is expanded into a temporary directory that lives until exit
    results = results_directory(Path(sys.argv[1]).resolve(), keep)
    run_info = json.loads((results / "run.json").read_text())
    m = matrix.smoke_protocol() if run_info["smoke"] else matrix.MATRIX
    selections = json.loads((results / "selections.json").read_text()) if (results / "selections.json").exists() else {}
    r = Records(results, m, selections)
    summary = {"run": run_info, "studies": {}, "incomplete": {}}
    for name, function in SUMMARIES.items():
        try:
            summary["studies"][name] = function(r)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            summary["incomplete"][name] = f"{type(exc).__name__}: {exc}"
    for stage, values in r.precisions.items():
        if len(values) != 1:
            raise SystemExit(f"{stage} mixes compute precisions {values}")
    summary["reproduction_source_sha256_by_stage"] = {k: sorted(v) for k, v in r.sources.items()}
    mixed = sorted(stage for stage, values in r.sources.items() if len(values) != 1)
    if mixed:
        message = f"Stages mix reproduction sources (code or protocol changed between jobs): {mixed}"
        if not run_info["smoke"]:
            raise SystemExit(message + "; rerun those stages with one source")
        print("smoke only, allowed: " + message)
    out = results / "analysis" if Path(sys.argv[1]).is_dir() else HERE.parent / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "generated.tex").write_text(latex(summary))
    if "--paper" in sys.argv:
        if run_info["smoke"]:
            raise SystemExit("Refusing to put smoke results into the paper")
        (HERE.parent / "results").mkdir(exist_ok=True)
        (HERE.parent / "results" / "generated.tex").write_text(latex(summary))
    print(f"wrote {out / 'summary.json'}; complete: {sorted(summary['studies'])}")
    for name, reason in summary["incomplete"].items():
        print(f"  incomplete {name}: {reason}")


if __name__ == "__main__":
    main()
