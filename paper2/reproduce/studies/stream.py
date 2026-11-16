"""Learning on uninterrupted streams with the stock DABSN core (shared by two studies).

Each lane is one stream that never restarts. The model is the published DABSNCore at
its default initialization with an FP32 linear readout at every step; there is no read,
so everything the model carries from the past lives in the recurrent state, and
everything that teaches it to carry it arrives through E. The stream is processed in
chunks; one optimizer update follows every chunk, and every prediction scored is made
before the update that uses its target (prequential).

Arms, all from identical parameters on identical streams:
    carry        E carried across every chunk boundary (the online method)
    reset        E erased after every update
    permute      E exchanged across lanes after every update (declared permutation)
    frozen_core  the online computation with the core's gradients, which arrive only
                 through E, dropped before clipping; the readout learns, the core is
                 verified bitwise unchanged
    tbptt        truncated BPTT: the last W chunks recomputed from their entering state
                 with the current weights before every update (window W declared)
    carry_normalized
                 carried E with the moment-free normalized action: each core parameter
                 coordinate's credit is scaled by the global mean response scale over its
                 own response scale read from E (embedded column L2 over sqrt(3BS), floored),
                 the readout's by the global mean, then clipped and applied by plain SGD;
                 no persistent optimizer state

Carried E is bounded per unit (``eligibility_unit_ceiling``). A unit whose own transport
expands grows its trace without bound on a stream that never resets, because E carries no
cross-unit coupling; the bound rescales such a unit's traces back to the ceiling, exactly as
global clipping bounds the action in every arm, and the event is recorded. Every rescaling is
counted and the largest per-unit trace norm is reported, so a run that needed the bound is
visible in its record rather than silently different.

Benchmarks are standard and defined in stream_tasks.py: memory capacity (Jaeger 2001) and
NARMA-10 (Atiya and Parlos 2000).
"""
from __future__ import annotations

import io
import itertools
import math
import time

import torch
from dabsn.distributed import clip_grad_norm
from dabsn.temporal_credit import OnlineCoreFrame, initial_core_eligibility

import credit
import harness
import models
from studies import stream_tasks

NEEDS_NATIVE = True
METHODS = ("carry", "reset", "permute", "frozen_core", "tbptt", "carry_normalized")


# ------------------------------------------------------------------ jobs

def validate(protocol):
    if tuple(protocol["methods"]) != tuple(m for m in METHODS if m in protocol["methods"]):
        raise ValueError("Unknown or misordered stream method")
    permutation = protocol["exchange_lane_permutation"]
    if sorted(permutation) != list(range(protocol["batch"])) or any(i == p for i, p in enumerate(permutation)):
        raise ValueError("Exchange permutation must move every lane")


def jobs(protocol):
    validate(protocol)
    return [{"phase": "main", "method": method, "seed": seed}
            for method, seed in itertools.product(protocol["methods"], protocol["seeds"])]


def preflight_jobs(protocol):
    """Two real updates plus exact checkpoint recovery for every arm."""
    validate(protocol)
    return [{"phase": "preflight", "method": method, "seed": protocol["seeds"][0]} for method in protocol["methods"]]


def identifier(job):
    return f"{job['phase']}-{job['method']}-s{job['seed']}"


# ---------------------------------------------------------------- streams

class Streams:
    """Inputs, targets and target masks for every lane, as [B, T, ...] tensors."""

    def __init__(self, protocol, seed):
        self.chunk = protocol["chunk"]
        # The transport diagnostic reads the run's own next inputs, so the stream carries
        # its window beyond the last update.
        total = (protocol["warmup_chunks"] + protocol["updates"]
                 + protocol["transport_diagnostic_chunks"]) * self.chunk
        benchmark = protocol["benchmark"]
        lanes = range(protocol["batch"])
        self.regenerated = []
        if benchmark == "memory_capacity":
            delays = protocol["delays"]
            u = torch.tensor([stream_tasks.memory_capacity_stream(seed, lane, total) for lane in lanes])
            self.inputs = u.unsqueeze(-1)
            targets = torch.zeros(u.shape[0], total, len(delays))
            mask = torch.zeros(u.shape[0], total, len(delays), dtype=torch.bool)
            for j, k in enumerate(delays):
                if k < total:  # a delay longer than the stream has no target yet
                    targets[:, k:, j] = u[:, :total - k]
                    mask[:, k:, j] = True
            self.targets, self.mask = targets, mask
        elif benchmark == "narma10":
            series = [stream_tasks.narma10_stream(seed, lane, total) for lane in lanes]
            self.inputs = torch.tensor([s[0] for s in series]).unsqueeze(-1)
            self.targets = torch.tensor([s[1] for s in series]).unsqueeze(-1)
            self.mask = torch.ones_like(self.targets, dtype=torch.bool)
            self.regenerated = [s[2] for s in series]
        else:
            raise ValueError(f"Unknown stream benchmark: {benchmark}")

    def get(self, index, device):
        a, b = index * self.chunk, (index + 1) * self.chunk
        return (self.inputs[:, a:b].to(device), self.targets[:, a:b].to(device), self.mask[:, a:b].to(device))


def score(protocol, predictions, targets, masks):
    """Benchmark score on prequential predictions: per-delay memory capacity (squared
    correlation) and their sum, or normalized MSE (MSE over target variance)."""
    p = torch.cat(predictions, 1).double(); t = torch.cat(targets, 1).double(); m = torch.cat(masks, 1)
    if protocol["benchmark"] == "memory_capacity":
        per_delay = {}
        for j, k in enumerate(protocol["delays"]):
            a, b = p[..., j][m[..., j]], t[..., j][m[..., j]]
            if a.numel() < 2:
                per_delay[str(k)] = None  # no target yet at this delay
                continue
            a, b = a - a.mean(), b - b.mean()
            denominator = float((a.square().sum() * b.square().sum()).sqrt())
            per_delay[str(k)] = 0.0 if denominator == 0 else float((a * b).sum()) ** 2 / denominator ** 2
        return {"memory_capacity_by_delay": per_delay,
                "memory_capacity_total": sum(v for v in per_delay.values() if v is not None)}
    error = float((p - t).square()[m].mean()); variance = float(t[m].var())
    return {"nmse": error / variance, "mse": error, "target_variance": variance}


# --------------------------------------------------------------- training

def normalize_credit(cfg, model, eligibility):
    """The moment-free normalized action, in place on the gradients (see the docstring)."""
    scales, mean = credit.column_scales(model.core, eligibility)
    floor = cfg["normalized_action"]["floor"]
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            if p in scales:
                p.grad.mul_(mean / scales[p].clamp_min(floor))
            else:
                p.grad.mul_(mean)
            harness.finite(p.grad, "Nonfinite normalized stream credit")
    return {"global_response_scale": float(mean),
            "common_coordinate_denominator": 3 * eligibility.batch_size * eligibility.state_dim}


def unit_trace_norms(eligibility):
    """Per-unit L2 norm of all traces owned by that unit, over lanes."""
    e = eligibility
    squared = e.local.square().sum(dim=(0, 2, 3)) + e.global_.square().sum(dim=(0, 2, 3))
    for value in (e.w_budget, e.a_budget, e.wg_budget, e.ug_budget):
        squared = squared + value.square().sum(dim=tuple(d for d in range(value.dim()) if d != 1))
    return squared.sqrt()


def bound_eligibility(eligibility, ceiling):
    """Rescale any unit whose trace norm exceeds the ceiling back to it, and report.

    E omits cross-unit coupling by construction, so a unit whose own transport expands
    grows without bound on a stream that never resets even when the full dynamics
    contract. This bounds the carried object the way global clipping bounds the action:
    direction within the unit is preserved, other units are untouched."""
    norms = unit_trace_norms(eligibility)
    largest = float(norms.max())
    exceeded = norms > ceiling
    count = int(exceeded.sum())
    if count == 0:
        return eligibility, largest, 0, []
    factor = torch.where(exceeded, ceiling / norms.clamp_min(1e-30), torch.ones_like(norms))
    def scale(value):
        shape = [1] * value.dim(); shape[1] = -1
        return (value * factor.reshape(shape)).contiguous()
    units = [int(i) for i in torch.nonzero(exceeded).flatten()[:8]]
    return credit.rebuild(scale(v) for v in eligibility.tensors()), largest, count, units


def transport_diagnostic(cfg, core, state, streams, index, device):
    """Does E's per-unit transport expand where the full dynamics contract?

    Eq. (residual) omits cross-unit transport $(J_t-D_t)K_t$. On a stream that never
    resets, a unit whose own transport $D_t$ expands grows its trace without bound even
    where the full Jacobian $J_t$ contracts, because the term that would have stabilized
    it is exactly the omitted one. This measures both at the run's own final weights and
    inputs: per-unit trace growth from a fresh E, and the top singular value of the exact
    state Jacobian over the same window (power iteration on J^T J; J v by central
    difference of the exact forward, J^T u by the exact backward)."""
    chunks = cfg["transport_diagnostic_chunks"]
    with torch.no_grad():
        traces = []
        advancing, trace = state, initial_core_eligibility(
            core, cfg["batch"], device=device, dtype=torch.float32)
    for c in range(chunks):
        x, _, _ = streams.get(index + c, device)
        frame = OnlineCoreFrame(trace)
        out, advancing = core.forward_from_state(x, initial_state=advancing, return_final_state=True,
                                                 temporal_credit="online", eligibility_state=trace,
                                                 online_frame=frame)
        (out[0].float().sum() * 0).backward()
        trace = frame.resulting_eligibility
        advancing = tuple(v.detach() for v in advancing)
        if c in (chunks // 4 - 1, chunks - 1):
            traces.append(unit_trace_norms(trace).detach())
    core.zero_grad(set_to_none=True)
    steps = (chunks - chunks // 4) * cfg["chunk"]
    growth = (traces[1].clamp_min(1e-30).log() - traces[0].clamp_min(1e-30).log()) / steps
    expanding = int((growth > 1e-3).sum())
    top = int(growth.argmax())

    def final_state(xs, start_state):
        _, end = core.forward_from_state(xs, initial_state=start_state, return_final_state=True,
                                         temporal_credit="bptt")
        return torch.cat([v.reshape(v.shape[0], -1) for v in end], 1)

    def split(flat):
        out = []; i = 0
        for v in state:
            n = v[0].numel(); out.append(flat[:, i:i + n].reshape_as(v)); i += n
        return tuple(out)

    xs = torch.cat([streams.get(index + c, device)[0] for c in range(chunks)], 1)
    v = torch.randn(state[0].shape[0], sum(t[0].numel() for t in state), device=device)
    v = v / v.norm()
    sigma = 0.0
    for _ in range(cfg["transport_diagnostic_iterations"]):
        epsilon = 1e-3
        with torch.no_grad():
            plus = final_state(xs, tuple(a + epsilon * b for a, b in zip(state, split(v))))
            minus = final_state(xs, tuple(a - epsilon * b for a, b in zip(state, split(v))))
        jv = (plus - minus) / (2 * epsilon)
        sigma = float(jv.norm())
        start = tuple(a.detach().clone().requires_grad_(True) for a in state)
        gradients = torch.autograd.grad(final_state(xs, start), start, grad_outputs=jv)
        jtjv = torch.cat([t.reshape(t.shape[0], -1) for t in gradients], 1)
        v = jtjv / jtjv.norm()
    core.zero_grad(set_to_none=True)
    jacobian_steps = chunks * cfg["chunk"]
    return {"window_steps": jacobian_steps,
            "units_with_expanding_trace": expanding, "units": int(growth.numel()),
            "fastest_unit": top,
            "fastest_unit_trace_log_growth_per_step": float(growth[top]),
            "median_unit_trace_log_growth_per_step": float(growth.median()),
            "exact_state_jacobian_top_singular_value": sigma,
            "exact_state_jacobian_log_growth_per_step": math.log(max(sigma, 1e-300)) / jacobian_steps,
            "reading": ("positive unit growth with a negative Jacobian rate is the omitted "
                        "cross-unit transport of Eq. (residual): E expands where the full "
                        "dynamics contract")}


def train(job, ctx, saved=None):
    cfg = ctx.protocol; device = ctx.device; method = job["method"]
    preflight = job["phase"] == "preflight"
    updates = 2 if preflight else cfg["updates"]
    warmup = cfg["warmup_chunks"]; window = cfg["tbptt_window_chunks"]
    outputs_dim = len(cfg["delays"]) if cfg["benchmark"] == "memory_capacity" else 1
    model = models.Model(1, cfg["state_dim"], outputs_dim, job["seed"], device)
    core = model.core
    initial_parameters_sha256 = harness.parameter_digest(model)
    frozen = models.FrozenCore(model) if method == "frozen_core" else None
    normalized = method == "carry_normalized"
    optimizer = None if normalized else torch.optim.Adam(
        frozen.trainable(model) if frozen else model.parameters(),
        lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), weight_decay=0.0)
    streams = Streams(cfg, job["seed"])
    batch = cfg["batch"]
    permutation = torch.tensor(cfg["exchange_lane_permutation"], device=device, dtype=torch.long)
    state = core.initial_state(batch, device=device, dtype=torch.float32)
    eligibility = initial_core_eligibility(core, batch, device=device, dtype=torch.float32)
    entering = state; buffer = []            # tbptt: state entering the window, and its chunks
    rows = []; recent = []; completed = 0
    peak_unit_trace = 0.0; bound_events = 0; bounded_unit_ids = set(); first_bound = None
    largest_unit_trace = 0.0
    if saved is not None:
        completed = saved["completed"]
        model.load_state_dict(saved["model"])
        if optimizer is not None:
            harness.restore_optimizer_state(optimizer, saved["optimizer"])
        elif saved["optimizer"] is not None:
            raise ValueError("Unexpected persistent optimizer state")
        state = tuple(v.to(device) for v in saved["state"])
        eligibility = credit.rebuild(v.to(device) for v in saved["eligibility"])
        entering = tuple(v.to(device) for v in saved["entering"])
        # Window inputs and scored predictions live on the CPU; a checkpoint loaded onto
        # the device must not move them.
        buffer = [chunk.cpu() for chunk in saved["buffer"]]
        rows = saved["rows"]; recent = [tuple(t.cpu() for t in r) for r in saved["recent"]]
        peak_unit_trace = saved["peak_unit_trace"]; bound_events = saved["bound_events"]
        bounded_unit_ids = set(saved["bounded_unit_ids"]); first_bound = saved["first_bound"]

    def payload():
        return {"completed": completed, "model": model.state_dict(),
                "optimizer": None if optimizer is None else optimizer.state_dict(),
                "state": list(state), "eligibility": list(eligibility.tensors()), "entering": list(entering),
                "buffer": buffer, "rows": rows, "recent": [list(r) for r in recent],
                "peak_unit_trace": peak_unit_trace, "bound_events": bound_events,
                "bounded_unit_ids": sorted(bounded_unit_ids), "first_bound": first_bound}

    started = time.monotonic(); first_boundary = None
    for index in range(warmup + completed if saved is not None else 0, warmup + updates):
        tick = time.monotonic()
        x, target, mask = streams.get(index, device)
        feedback = index >= warmup
        model.zero_grad(set_to_none=True)
        if method == "tbptt":
            buffer.append(x.cpu())
            chunks = [c.to(device) for c in buffer]
            with harness.autocast(device):
                head, after_first = core.forward_from_state(chunks[0], initial_state=entering,
                                                            return_final_state=True, temporal_credit="bptt")
                trajectory = head[0]
                if len(chunks) > 1:
                    rest, state = core.forward_from_state(torch.cat(chunks[1:], 1), initial_state=after_first,
                                                          return_final_state=True, temporal_credit="bptt")
                    trajectory = rest[0]
                else:
                    state = after_first
            prediction = model.head(torch.tanh(trajectory[:, -x.shape[1]:, core.hidden_dim:])).float()
            if len(buffer) == window:
                entering = tuple(v.detach().float() for v in after_first); buffer = buffer[1:]
            frame = None
        else:
            frame = OnlineCoreFrame(eligibility)
            with harness.autocast(device):
                out, state = core.forward_from_state(x, initial_state=state, return_final_state=True,
                                                     temporal_credit="online", eligibility_state=eligibility,
                                                     online_frame=frame)
            trajectory = out[0]
            prediction = model.head(torch.tanh(trajectory[:, :, core.hidden_dim:])).float()
        if feedback:
            loss = (prediction - target).square()[mask].mean()
            harness.finite(loss, "nonfinite stream loss")
            loss.backward()
        else:
            (trajectory.float().sum() * 0).backward()
        if frame is not None:
            if not frame.replayed or frame.resulting_eligibility is None:
                raise RuntimeError("online eligibility failed to advance")
            eligibility = frame.resulting_eligibility
            eligibility, largest_unit_trace, bounded_units, which_units = bound_eligibility(
                eligibility, cfg["eligibility_unit_ceiling"])
            peak_unit_trace = max(peak_unit_trace, largest_unit_trace)
            if bounded_units:
                bound_events += 1
                bounded_unit_ids.update(which_units)
                if first_bound is None:
                    first_bound = {"update": completed + 1, "units": which_units,
                                   "unit_trace_norm_before_bound": largest_unit_trace}
        state = tuple(v.detach().float() for v in state)
        if not feedback:
            model.zero_grad(set_to_none=True)
            continue
        if frozen is not None:
            frozen.drop_gradients()
        action = None
        if normalized:
            action = normalize_credit(cfg, model, eligibility)
        norm = float(clip_grad_norm(model, cfg["clip_norm"]))
        if not math.isfinite(norm):
            raise FloatingPointError("Nonfinite stream credit")
        if normalized:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-cfg["normalized_learning_rate"])
        else:
            optimizer.step()
        completed += 1
        if method == "reset":
            eligibility = initial_core_eligibility(core, batch, device=device, dtype=torch.float32)
        elif method == "permute":
            eligibility = credit.rebuild(v.index_select(0, permutation) for v in eligibility.tensors())
        recent.append((prediction.detach().cpu(), target.cpu(), mask.cpu()))
        recent = recent[-cfg["score_window_updates"]:]
        if completed % cfg["report_every"] == 0 or completed == updates:
            rows.append({"update": completed, "pre_clip_norm": norm, "step_seconds": time.monotonic() - tick,
                         "largest_unit_trace_norm": largest_unit_trace, "bound_events": bound_events,
                         **({"normalized_action": action} if action is not None else {}),
                         **score(cfg, *zip(*recent))})
        if preflight:
            if completed == 1 and saved is None:
                stream = io.BytesIO(); torch.save(payload(), stream); first_boundary = stream.getvalue()
            continue
        if completed % cfg["checkpoint_every"] == 0 or completed == updates:
            ctx.checkpoint(ctx.step_name(completed), payload())
        harness.maybe_stop(ctx, lambda: ctx.checkpoint("recovery", payload(), replace=True))
    transport = None
    if not preflight and method in ("carry", "carry_normalized"):
        transport = transport_diagnostic(cfg, core, state, streams, warmup + completed, device)
    result = {"benchmark": cfg["benchmark"], "completed_updates": completed, "rows": rows,
              "final": score(cfg, *zip(*recent)), "precision": harness.precision_record(device),
              "architecture": "stock DABSNCore, FP32 linear readout at every step; no read",
              "initial_parameters_sha256": initial_parameters_sha256,
              "regenerated_series_attempts": streams.regenerated,
              "frozen_core_bitwise_unchanged": frozen.verify() if frozen else None,
              "eligibility_scalars": credit.scalars(eligibility),
              "transport_diagnostic": transport,
              "eligibility_bound": {
                  "unit_ceiling": cfg["eligibility_unit_ceiling"],
                  "peak_unit_trace_norm": peak_unit_trace,
                  "rescaling_events": bound_events,
                  "bounded_units": sorted(bounded_unit_ids),
                  "first_event": first_bound,
                  "scope": "carried E only; no effect on any arm that never reaches the ceiling"},
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "training_seconds": time.monotonic() - started,
              "checkpoint_sha256": ctx.checkpoint_hashes()}
    if preflight and saved is None:
        boundary = torch.load(io.BytesIO(first_boundary), map_location=device, weights_only=True)
        replayed = train(job, ctx, boundary)[1]
        # Rows are logs (they hold wall-clock times); everything else is state.
        error = harness.compare_recovery({k: v for k, v in payload().items() if k != "rows"},
                                         {k: v for k, v in replayed.items() if k != "rows"})
        if error["max_scaled_error"] > 2e-5:
            raise AssertionError(f"Next update after recovery differs: {error}")
        result["checkpoint_recovery"] = {**error, "next_update_scaled_tolerance": 2e-5}
    return result, payload()


def run(job, ctx):
    saved = ctx.latest() if ctx.resume and job["phase"] != "preflight" else None
    return train(job, ctx, saved)[0]
