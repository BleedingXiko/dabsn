"""Horizon-matched truncated BPTT on the two stream benchmarks.

The stream studies compare arms whose credit reaches different distances back:
E erased after every update reaches one chunk; truncated BPTT reaches its window;
carried E reaches indefinitely. To separate how far credit reaches from how exact it
is, this stage runs truncated BPTT whose window is exactly one chunk, the reach of
erased E, with everything else identical to the stream study it belongs to (same
streams, seeds, model, rate, updates and scoring, through stream.train unchanged).
The matched pairs are then: one chunk, exact BPTT against erased E; long reach,
the declared truncated window against carried E.
"""
from __future__ import annotations

import itertools

import harness
from studies import stream

NEEDS_NATIVE = True


def jobs(protocol):
    return [{"phase": "main", "method": "tbptt", "seed": seed, "study": study,
             "window_chunks": protocol["window_chunks"]}
            for study, seed in itertools.product(protocol["studies"], protocol["seeds"])]


def identifier(job):
    return f"{job['study']}-tbptt-w{job['window_chunks']}-s{job['seed']}"


def configuration(matrix, study, window_chunks):
    """The stream study's own declared configuration with only the window changed."""
    cfg = dict(matrix[study])
    cfg["tbptt_window_chunks"] = window_chunks
    cfg["methods"] = ["tbptt"]
    return cfg


def run(job, ctx):
    window = ctx.protocol["window_chunks"]
    cfg = configuration(ctx.matrix, job["study"], window)
    inner = harness.Context(job["study"], job, cfg, ctx.device, ctx.directory, ctx.results,
                            ctx.resume, None, ctx.smoke, ctx.identity["reproduction_source_sha256"])
    inner.matrix = ctx.matrix; inner.deadline = ctx.deadline; inner.started = ctx.started
    result = stream.run({k: v for k, v in job.items() if k not in ("study", "window_chunks")}, inner)
    result["window_chunks"] = window
    result["window_steps"] = window * cfg["chunk"]
    result["effective_configuration"] = {k: cfg[k] for k in (
        "benchmark", "chunk", "updates", "learning_rate", "state_dim", "batch", "seeds", "tbptt_window_chunks")}
    return result
