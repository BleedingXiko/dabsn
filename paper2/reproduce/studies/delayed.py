"""Matched episodic learning on two delayed tasks (the central learning study).

Tasks: selective retrieval (classification) and conditional computation
(regression). Methods: B-channel online credit (one temporal chunk per episode,
the framework's train_step default), the same online path with the core frozen
(core gradients dropped; read and readout train), and full BPTT, all from
identical initial parameters on identical episodes. Recurrent state and E reset
between episodes.

Phases:
- development: two seeds, online and BPTT, both candidate rates.
  One common rate per task is selected from all development runs.
- main: five seeds, all three methods, the selected rate, with
  scientific checkpoints that the intervention study forks from.
- ablation: temporal chunk size and two other shapes, at the selected rates.
"""
from __future__ import annotations

import itertools
import math
import time

import torch

import harness
import models
from studies import delayed_tasks

NEEDS_NATIVE = True


# ------------------------------------------------------------------ jobs

def development_jobs(protocol):
    cfg = protocol["development"]
    return [{"phase": "development", "task": task, "seed": seed, "method": method, "lr": lr}
            for task, seed, method, lr in itertools.product(
                protocol["tasks"], cfg["seeds"], cfg["methods"], cfg["learning_rates"])]


def main_jobs(protocol, rates):
    cfg = protocol["main"]
    return [{"phase": "main", "task": task, "seed": seed, "method": method, "lr": rates[task]}
            for task, seed, method in itertools.product(protocol["tasks"], cfg["seeds"], cfg["methods"])]


def ablation_jobs(ablation, rates):
    cfg = ablation["chunk"]
    result = [{"phase": "ablation", "study": "chunk", "task": cfg["task"], "seed": seed,
               "method": method, "lr": rates[cfg["task"]], "input_dim": cfg["input_dim"],
               "state_dim": cfg["state_dim"], "chunk": chunk}
              for seed, method, chunk in itertools.product(cfg["seeds"], cfg["methods"], cfg["chunks"])]
    cfg = ablation["shape"]
    result += [{"phase": "ablation", "study": "shape", "task": task, "seed": seed,
                "method": method, "lr": rates[task], "input_dim": shape[0],
                "state_dim": shape[1], "chunk": cfg["chunk"]}
               for task, seed, method, shape in itertools.product(
                   cfg["tasks"], cfg["seeds"], cfg["methods"], cfg["shapes"])]
    expected = ablation["chunk"]["declared_training_jobs"] + ablation["shape"]["declared_training_jobs"]
    if len(result) != expected:
        raise ValueError("Ablation grid count changed")
    return result


def identifier(job):
    if job["phase"] == "ablation":
        return (f"ablation-{job['study']}-{job['task']}-s{job['seed']}-{job['method']}"
                f"-i{job['input_dim']}-h{job['state_dim']}-c{job['chunk']}-lr{job['lr']:.4g}")
    return f"{job['phase']}-{job['task']}-s{job['seed']}-{job['method']}-lr{job['lr']:.4g}"


def select_rates(protocol, records):
    """One common rate per task: lowest mean final development loss over all horizons,
    both seeds and both methods. Ties choose the lower rate."""
    required = {identifier(job): job for job in development_jobs(protocol)}
    if set(records) != set(required):
        raise ValueError("Development grid incomplete or contains extra jobs")
    selected = {}; scores_by_task = {}
    for task in protocol["tasks"]:
        scores = {}
        for lr in protocol["development"]["learning_rates"]:
            losses = []
            for name, job in required.items():
                if job["task"] != task or job["lr"] != lr:
                    continue
                record = records[name]
                if record["status"] != "passed" or record["job"] != job:
                    raise ValueError(f"Invalid development result {name}")
                rows = record["result"]["development_rows"]
                if [row["step"] for row in rows] != protocol["development"]["evaluation_steps"]:
                    raise ValueError("Missing development evaluations")
                horizon_rows = rows[-1]["by_length"]
                if set(horizon_rows) != {str(v) for v in protocol["main"]["train_lengths_round_robin"]}:
                    raise ValueError("Missing development horizon")
                for metric in horizon_rows.values():
                    if not harness.nonnegative_finite(metric["loss"]):
                        raise ValueError("Invalid development loss")
                    losses.append(metric["loss"])
            scores[lr] = sum(losses) / len(losses)
        selected[task] = min(scores, key=lambda rate: (scores[rate], rate))
        scores_by_task[task] = {str(rate): scores[rate] for rate in scores}
    return {"selected_rates": selected, "mean_validation_loss_by_rate": scores_by_task,
            "selection": protocol["development"]["selection"]}


# ------------------------------------------------------------- training

def run(job, ctx):
    protocol = ctx.protocol
    central = protocol["main"]
    device = ctx.device
    task = job["task"]; method = job["method"]
    inputs = job.get("input_dim", central["input_dim"])
    state = job.get("state_dim", central["state_dim"])
    batch_size = central["batch"]
    # Training chunk: one chunk per episode unless an ablation job names one. The
    # declared 32-step chunk remains the final-chunk-only diagnostic window.
    train_chunk = job.get("chunk", central["online_chunk"]); chunk = central["chunk"]
    model = models.Block(inputs, state, delayed_tasks.output_dim(task), job["seed"], device)
    initial_parameters_sha256 = harness.parameter_digest(model)
    frozen = models.FrozenCore(model) if method == "frozen_core" else None
    optimizer = torch.optim.Adam(frozen.trainable(model) if frozen else model.parameters(), lr=job["lr"],
                                 betas=tuple(central["betas"]), weight_decay=central["weight_decay"])
    rows = []; diagnostics = []; completed = 0
    if ctx.resume:
        saved = ctx.latest()
        if saved is not None:
            model.load_state_dict(saved["model"])
            harness.restore_optimizer_state(optimizer, saved["optimizer"])
            rows = saved["rows"]; diagnostics = saved["diagnostics"]; completed = saved["completed"]
    development = job["phase"] == "development"
    updates = protocol["development"]["updates"] if development else central["updates"]
    horizons = central["train_lengths_round_robin"]
    evaluation_steps = protocol["development" if development else "main"]["evaluation_steps"]
    checkpoint_steps = protocol["development" if development else "main"]["checkpoint_steps"]
    evaluation_count = (protocol["development"]["evaluation_examples_per_horizon"] if development
                        else central["development_evaluation_examples_per_horizon"])
    evaluation_batch = protocol["execution"]["evaluation_batch"]
    proof_x = torch.zeros(batch_size, chunk, inputs, device=device, dtype=torch.float32)
    projection_precision = harness.assert_projection_precision(model, proof_x)
    del proof_x

    def evaluate_all():
        return {str(length): delayed_tasks.delayed_evaluate(model, task, 730021, length, evaluation_count,
                                                     evaluation_batch, inputs, chunk, device)
                for length in horizons}

    def payload():
        return {"completed": completed, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rows": rows, "diagnostics": diagnostics}

    started = time.monotonic()
    for step in range(completed, updates + 1):
        if step in evaluation_steps and (not rows or rows[-1]["step"] != step):
            rows.append({"step": step, "by_length": evaluate_all()})
        if step == updates:
            break
        length = horizons[step % len(horizons)]
        x, y, episode_ids = delayed_tasks.delayed_batch(task, 17003 + job["seed"], step * batch_size, length,
                                                 batch_size, inputs, device)
        loss, norm, _, _, _ = models.update(model, optimizer, x, y, method, train_chunk,
                                            central["clip_norm"], delayed_tasks.objective(task), frozen)
        if step + 1 in evaluation_steps or step + 1 in checkpoint_steps:
            diagnostics.append({"step": step + 1, "length": length, "training_loss": loss,
                                "pre_clip_norm": norm, "first_episode_id": episode_ids[0],
                                "last_episode_id": episode_ids[-1]})
        completed = step + 1
        if completed in checkpoint_steps:
            # The evaluation for this completed update belongs to its checkpoint.
            if completed in evaluation_steps:
                rows.append({"step": completed, "by_length": evaluate_all()})
            ctx.checkpoint(ctx.step_name(completed), payload())
        harness.maybe_stop(ctx, lambda: ctx.checkpoint("recovery", payload(), replace=True))

    final = {}; final_chunk_only = {}
    if not development:
        for length in central["final_evaluation_lengths"]:
            for policy, target in (("full", final), ("final_chunk_only", final_chunk_only)):
                target[str(length)] = delayed_tasks.delayed_evaluate(
                    model, task, 930021, length, central["final_evaluation_examples_per_horizon"],
                    evaluation_batch, inputs, chunk, device, input_policy=policy)
    return {"updates": updates, "precision": harness.precision_record(device),
            "projection_precision": projection_precision, "development_rows": rows,
            "final_by_length": final, "final_chunk_only_by_length": final_chunk_only,
            "diagnostics": diagnostics, "parameter_count": sum(p.numel() for p in model.parameters()),
            "backend": getattr(model.core, "_last_core_backend", None),
            "initial_parameters_sha256": initial_parameters_sha256,
            "checkpoint_sha256": ctx.checkpoint_hashes(),
            "training_and_evaluation_seconds": time.monotonic() - started,
            "objective": delayed_tasks.objective(task), "training_chunk": train_chunk,
            "frozen_core_bitwise_unchanged": frozen.verify() if frozen else None}
