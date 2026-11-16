"""The native action on the two stream benchmarks (studies/native_action.py).

    core      s_theta = column RMS of the core's E, eta_t = mean_theta s_theta;
              core credit g_theta <- g_theta * eta_t / max(s_theta, floor * eta_t);
              plain SGD at one nominal rate; no momentum, no moments, no clipping, no state.
    readout   not the core's responsibility: it keeps its own credit under exactly the
              regime of the stream study's other arms (Adam at the declared rate and betas,
              clipping at the declared norm over its own gradient), so this arm differs from
              the carry arm only in the core's action.

The rule's two constants (the core's nominal rate and the relative floor) are chosen per
benchmark on a development seed that is never a main seed, over a declared grid, at a
shorter declared budget. A run whose loss becomes non-finite is recorded as diverged (a
measured outcome of that setting), not as a failed job.

stream.train is used unchanged: its normalization hook is pointed at this rule for the
duration of this stage's job only. The readout's optimizer state is saved in every
checkpoint, so a resumed run continues exactly.
"""
from __future__ import annotations

import itertools
import math

import torch
from torch.nn.utils import clip_grad_norm_

import harness
from studies import native_action, stream

NEEDS_NATIVE = True


def development_jobs(protocol):
    d = protocol["development"]
    return [{"phase": "development", "method": "carry_normalized", "seed": d["seed"], "study": study,
             "rate": rate, "relative_floor": floor}
            for study, rate, floor in itertools.product(protocol["studies"], d["rate_grid"],
                                                        d["relative_floor_grid"])]


def main_jobs(protocol, chosen):
    return [{"phase": "main", "method": "carry_normalized", "seed": seed, "study": study,
             "rate": chosen[study]["rate"], "relative_floor": chosen[study]["relative_floor"]}
            for study, seed in itertools.product(protocol["studies"], protocol["seeds"])]


def identifier(job):
    return (f"{job['phase']}-{job['study']}-native-s{job['seed']}"
            f"-r{job['rate']:g}-f{job['relative_floor']:g}")


def _quality(study_protocol, result):
    """Higher is better; a diverged run is worst."""
    if result.get("diverged"):
        return -math.inf
    final = result["final"]
    if study_protocol["benchmark"] == "memory_capacity":
        return final["memory_capacity_total"]
    return -final["nmse"]


def select_rates(protocol, records, matrix=None):
    from matrix import MATRIX
    matrix = matrix or MATRIX
    required = {identifier(job): job for job in development_jobs(protocol)}
    if set(records) != set(required):
        raise ValueError("Stream native development grid incomplete or contains extra jobs")
    selected = {}; table = {}
    for study in protocol["studies"]:
        scores = {}
        for name, job in required.items():
            if job["study"] != study:
                continue
            record = records[name]
            if record["status"] != "passed" or record["job"] != job:
                raise ValueError(f"Invalid stream native development result {name}")
            scores[(job["rate"], job["relative_floor"])] = _quality(matrix[study], record["result"])
        best = max(scores, key=lambda k: (scores[k], -k[0], k[1]))
        if scores[best] == -math.inf:
            raise ValueError(f"Every native setting diverged on {study}")
        selected[study] = {"rate": best[0], "relative_floor": best[1]}
        table[study] = {f"rate {k[0]:g}, floor {k[1]:g}": (None if v == -math.inf else v)
                        for k, v in scores.items()}
    return {"selected_rates": selected, "development_quality_by_setting": table,
            "selection": protocol["development"]["selection"]}


def configuration(matrix, study, protocol, job):
    """The stream study's declared configuration; the core's nominal rate from the job; no
    global clipping (the readout's clipping is applied to its own gradient below)."""
    cfg = dict(matrix[study])
    cfg["normalized_learning_rate"] = job["rate"]
    cfg["clip_norm"] = float("inf")
    cfg["methods"] = ["carry_normalized"]
    if job["phase"] == "development":
        cfg["updates"] = protocol["development"]["updates"]
    return cfg


def _finite_or_none(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    return value


def _guarded_transport(inherited):
    """The stream transport diagnostic, recording a vanished Jacobian (0/0 in its power
    iteration) as measured rather than as a non-finite number."""
    def run(*args, **kwargs):
        out = inherited(*args, **kwargs)
        sigma = out.get("exact_state_jacobian_top_singular_value")
        degenerate = isinstance(sigma, float) and sigma != sigma
        out = _finite_or_none(out)
        out["jacobian_degenerate"] = degenerate
        return out
    return run


class _Readout:
    """The readout's own credit under the stream arms' Adam regime, with its state carried
    through every checkpoint."""

    def __init__(self, study_cfg, saved):
        self.cfg = study_cfg; self.saved = saved; self.optimizer = None; self.norm = None

    def step(self, model):
        head = list(model.head.parameters())
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(head, lr=self.cfg["learning_rate"],
                                              betas=tuple(self.cfg["betas"]), weight_decay=0.0)
            if self.saved is not None:
                harness.restore_optimizer_state(self.optimizer, self.saved["readout_optimizer"])
        self.norm = float(clip_grad_norm_(head, self.cfg["clip_norm"]))
        self.optimizer.step()
        for p in head:
            p.grad = None  # the stream loop's plain SGD then moves the core only

    def state(self):
        return None if self.optimizer is None else self.optimizer.state_dict()


def run(job, ctx):
    study_cfg = ctx.matrix[job["study"]]
    cfg = configuration(ctx.matrix, job["study"], ctx.protocol, job)
    inner = harness.Context(job["study"], job, cfg, ctx.device, ctx.directory, ctx.results,
                            ctx.resume, None, ctx.smoke, ctx.identity["reproduction_source_sha256"])
    inner.matrix = ctx.matrix; inner.deadline = ctx.deadline; inner.started = ctx.started
    saved = inner.latest() if ctx.resume else None
    readout = _Readout(study_cfg, saved)
    checkpoint = inner.checkpoint
    inner.checkpoint = lambda name, payload, **kw: checkpoint(
        name, {**payload, "readout_optimizer": readout.state()}, **kw)

    def hook(_cfg, model, eligibility):
        action = native_action.normalize(model.core, list(model.core.parameters()), eligibility,
                                         relative_floor=job["relative_floor"])
        readout.step(model)
        return {**action, "readout_preclip_norm": readout.norm}

    inherited, inherited_transport = stream.normalize_credit, stream.transport_diagnostic
    stream.normalize_credit = hook
    stream.transport_diagnostic = _guarded_transport(inherited_transport)
    try:
        result = stream.run({k: v for k, v in job.items() if k not in ("study", "rate", "relative_floor")},
                            inner)
        result["diverged"] = False
    except FloatingPointError as error:
        result = {"diverged": True, "error": str(error), "final": None}
    finally:
        stream.normalize_credit, stream.transport_diagnostic = inherited, inherited_transport
    result["rule"] = {"definition": "studies/native_action.py on the core; the readout under the stream "
                                    "arms' Adam regime with its own clipping",
                      "core_nominal_rate": job["rate"], "relative_floor": job["relative_floor"],
                      "core_clipping": "none", "core_optimizer_state": "none"}
    return result
