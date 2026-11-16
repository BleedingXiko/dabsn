"""The conjunction: learning driven entirely by E, against the standard stack.

The paper introduces online credit (E), an action read from E without optimizer state
(Eq. normalized-action), and pace set by E itself (the action amplitude evolves with
the mean column RMS of E, with no moment estimate and no time-indexed schedule). This
study runs them together on the standard paper-1 benchmarks with the full DABSN model:

    native        online credit, one chunk per episode. The core takes the native action
                  (studies/native_action.py): direction from its credit normalized by its own
                  E (s_theta the column RMS of E, eta_t its mean; core credit times
                  eta_t / max(s_theta, floor * eta_t)), pace from E (the direction rescaled to
                  coordinate RMS eta_t), plain SGD at one nominal rate, no optimizer state, no
                  schedule, no clipping. The read
                  and the readout are not the core's responsibility: they keep their own
                  online credit under exactly the regime of the AdamW arms (AdamW at the
                  paper-1 rate and schedule, clipping at the declared norm over their own
                  gradients), so native and adamw_online differ only in the core's action.
    adamw_online  online credit with AdamW at the paper-1 rate, warmup and schedule, and
                  the declared clipping (continuity.one_update, unchanged)
    adamw_bptt    full BPTT with the same AdamW regime (continuity.one_update, unchanged)

Same model factory, initialization, training data (identical random draws), budgets and
final evaluation as the continuity study; held-out accuracy at the training length is also
recorded on a fixed schedule so data efficiency is compared on equal terms. Evaluation
saves and restores the random state, so the AdamW arms reproduce the continuity records.
"""
from __future__ import annotations

import itertools
import math
import random
import time

import torch
from torch.nn import functional as F

import harness
import models
from dabsn.reproductions import common as paper1
from dabsn.temporal_credit import finalize_online_temporal_state
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from studies import continuity, native_action

NEEDS_NATIVE = True
ARMS = ("native", "adamw_online", "adamw_bptt")


def development_jobs(protocol):
    """The native rule's two constants on a development seed that is never a main seed."""
    return [{"phase": "development", "task": task, "seed": protocol["development_seed"], "arm": "native",
             "rate": rate, "relative_floor": floor}
            for task, rate, floor in itertools.product(protocol["tasks"], protocol["rate_grid"],
                                                       protocol["relative_floor_grid"])]


def main_jobs(protocol, selected):
    out = []
    for task, seed, arm in itertools.product(protocol["tasks"], protocol["seeds"], protocol["arms"]):
        job = {"phase": "main", "task": task, "seed": seed, "arm": arm}
        if arm == "native":
            job.update(rate=selected[task]["rate"], relative_floor=selected[task]["relative_floor"])
        out.append(job)
    return out


def identifier(job):
    base = f"{job['phase']}-{job['task']}-s{job['seed']}-{job['arm']}"
    if job["arm"] == "native":
        base += f"-r{job['rate']:g}-f{job['relative_floor']:g}"
    return base


def select_rates(protocol, records):
    """The native core's two constants per task, as the AdamW arms have per-task paper-1 rates:
    the highest mean final held-out accuracy over the paper-1 evaluation lengths (the
    evaluation the main study reports) on the development seed; ties choose the smaller rate,
    then the larger floor."""
    required = {identifier(job): job for job in development_jobs(protocol)}
    if set(records) != set(required):
        raise ValueError("Conjunction development grid incomplete or contains extra jobs")
    selected = {}; table = {}
    for task in protocol["tasks"]:
        scores = {}
        for name, job in required.items():
            if job["task"] != task:
                continue
            record = records[name]
            if record["status"] != "passed" or record["job"] != job:
                raise ValueError(f"Invalid conjunction development result {name}")
            evaluation = record["result"]["evaluation"]
            scores[(job["rate"], job["relative_floor"])] = (
                sum(e["eval_accuracy"] for e in evaluation) / len(evaluation))
        best = max(scores, key=lambda k: (scores[k], -k[0], k[1]))
        selected[task] = {"rate": best[0], "relative_floor": best[1]}
        table[task] = {f"rate {k[0]:g}, floor {k[1]:g}": v for k, v in scores.items()}
    return {"selected_rates": selected, "mean_final_accuracy_over_lengths_by_setting": table,
            "selection": protocol["selection"]}


def core_parameters(net):
    view = models._NetView(net)
    return list(view.core.parameters())


def other_parameters(net):
    core = {id(p) for p in core_parameters(net)}
    return [p for p in net.parameters() if id(p) not in core]


def native_update(net, x, targets, rate, relative_floor, other_optimizer, clip_norm):
    """One episode of online credit. The core: the native action (credit times eta_t /
    max(s_theta, relative_floor * eta_t) from its own E, paced to coordinate RMS eta_t), plain
    SGD at the nominal rate, no state, no clipping. The read and the readout: their own online credit under the AdamW arms'
    regime (other_optimizer, clipping at clip_norm over their own gradients)."""
    view = models._NetView(net)
    valid = int((targets != -100).sum())
    loss_fn = continuity.objective(x.shape[1], valid)
    state = models.block_online_state(view, x.shape[0], x.shape[1], x.device)
    net.zero_grad(set_to_none=True)
    with harness.autocast(x.device):
        output, pending = net.forward_sequence_online(x, state)
    loss = loss_fn(output, targets)
    harness.finite(loss, "nonfinite native loss")
    loss.backward()
    state = finalize_online_temporal_state(pending)
    eligibility = state.layers[0].core_eligibility
    core = core_parameters(net)
    action = native_action.normalize(view.core, core, eligibility, relative_floor=relative_floor)
    mean = action["global_response_scale"]
    core_norm = action["direction_rms_before_pace"]
    with torch.no_grad():
        for p in core:
            if p.grad is not None:
                p.add_(p.grad, alpha=-rate)
    other_norm = float(clip_grad_norm(other_parameters(net), clip_norm))
    other_optimizer.step()
    net.post_optimizer_step(step_applied=True)
    return float(loss.detach()), float(mean), core_norm, other_norm


def accuracy_at(net, task, cfg, seed, length, batches, device, table, identity):
    """Held-out accuracy at one length, in its own seed namespace, leaving training RNG intact."""
    saved = (random.getstate(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    paper1.seed_all(970000 + 10000 * seed + length)
    net.eval(); correct = 0.0; total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, targets = continuity.task_batch(task, cfg, continuity.evaluation_batch_size(task, cfg, length),
                                               length, device, table, identity)
            with harness.autocast(device):
                logits = net.forward_sequence(x)
            valid = targets != -100
            correct += float(((logits.argmax(-1) == targets) & valid).sum()); total += float(valid.sum())
    net.train()
    random.setstate(saved[0]); torch.set_rng_state(saved[1])
    if saved[2] is not None:
        torch.cuda.set_rng_state_all(saved[2])
    return correct / total


def run(job, ctx):
    protocol = ctx.protocol; device = ctx.device; dev = torch.device(device)
    continuity.PROTOCOL = ctx.matrix["continuity"]
    task, seed, arm = job["task"], job["seed"], job["arm"]
    cfg = continuity.configuration_for({"task": task, "seed": seed, "method": "online"})
    paper1.seed_all(seed)
    net = continuity.model_for(task, cfg, dev)
    initial_parameters_sha256 = harness.parameter_digest(net)
    P = continuity.PROTOCOL
    # native: the AdamW regime holds only the read and readout; the core has no optimizer.
    optimizer = torch.optim.AdamW(other_parameters(net) if arm == "native" else net.parameters(),
                                  lr=cfg["lr"], betas=tuple(P["betas"]), eps=P["eps"],
                                  weight_decay=P["weight_decay"])
    table, identity = (None, None)
    rows = []; start = 0; last_loss = None
    saved = ctx.latest() if ctx.resume else None
    if saved is not None:
        net.load_state_dict(saved["model"])
        harness.restore_optimizer_state(optimizer, saved["optimizer"])
        random.setstate(saved["python_random"]); torch.set_rng_state(saved["torch_random"].cpu().clone())
        if device == "cuda":
            torch.cuda.set_rng_state_all([s.cpu().clone() for s in saved["cuda_random"]])
        start = saved["completed"]; rows = saved["rows"]; last_loss = saved["last_training_loss"]

    def payload(step):
        return {"completed": step, "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "python_random": random.getstate(), "torch_random": torch.get_rng_state(),
                "cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "rows": rows, "last_training_loss": last_loss}

    every = protocol["evaluate_every"]; train_length = cfg["train_length"]
    if start == 0:
        rows.append({"step": 0, "accuracy": accuracy_at(net, task, cfg, seed, train_length,
                                                        protocol["evaluation_batches"], dev, table, identity)})
    started = time.monotonic(); net.train()
    for step in range(start + 1, cfg["steps"] + 1):
        length = continuity.train_length(task, cfg, step)
        x, targets = continuity.task_batch(task, cfg, cfg["batch"], length, dev, table, identity)
        extra = {}
        for group in optimizer.param_groups:
            group["lr"] = continuity.scheduled_lr(cfg, step)
        if arm == "native":
            last_loss, amplitude, core_norm, other_norm = native_update(
                net, x, targets, job["rate"], job["relative_floor"], optimizer, P["clip_norm"])
            extra = {"amplitude": amplitude, "core_direction_rms_before_pace": core_norm, "read_readout_preclip_norm": other_norm}
        else:
            last_loss = continuity.one_update(net, optimizer, "online" if arm == "adamw_online" else "bptt",
                                              x, targets)
        continuity.original_probe_draw(task, cfg, step, dev, table, identity)
        if not math.isfinite(last_loss):
            raise FloatingPointError("Nonfinite training loss")
        if step % every == 0 or step == cfg["steps"]:
            rows.append({"step": step, "training_loss": last_loss, **extra,
                         "accuracy": accuracy_at(net, task, cfg, seed, train_length,
                                                 protocol["evaluation_batches"], dev, table, identity)})
        if step % cfg["checkpoint_every"] == 0 or step == cfg["steps"]:
            ctx.checkpoint(ctx.step_name(step), payload(step))
        harness.maybe_stop(ctx, lambda: ctx.checkpoint("recovery", payload(step), replace=True))
    evaluation = continuity.evaluate(net, task, cfg, seed, dev, table, identity)
    return {"task": task, "seed": seed, "arm": arm, "rows": rows, "evaluation": evaluation,
            "training_updates": cfg["steps"], "last_training_loss": last_loss,
            "initial_parameters_sha256": initial_parameters_sha256,
            "parameter_count": sum(p.numel() for p in net.parameters()),
            "training_seconds": time.monotonic() - started,
            "regime": (f"native: core by E credit, the normalized action and pace from E, nominal rate {job['rate']:g}, floor "
                       f"{job['relative_floor']:g} x eta_t, no core optimizer state, schedule or clipping; read and "
                       f"readout by their own credit under the AdamW arms' regime (lr {cfg['lr']}, schedule, clipping)"
                       if arm == "native" else
                       f"{arm}: continuity regime unchanged (AdamW lr {cfg['lr']}, schedule, clipping)"),
            "precision": harness.precision_record(device), "checkpoint_sha256": ctx.checkpoint_hashes()}
