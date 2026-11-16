"""Where online credit departs from exact credit on uninterrupted streams.

At the final weights of every carried-E run of both stream studies, over consecutive
windows of the run's own continuing stream, two gradients of the same loss (the window's
last chunk) are taken from the same entering state:

    exact   BPTT through the whole window (what truncated BPTT uses)
    online  the online computation through the same window, E started fresh at the window
            so both see the same horizon

and compared per parameter family: cosine and norm ratio. The readout's gradient involves
no temporal credit and must agree exactly (cosine 1): a built-in check. Eq. (residual)
names what online credit omits: cross-unit transport (J_t - D_t) K_t and, for the large
matrices, the energy and saturation pathways. This measures how much that omission costs
each family's direction.
"""
from __future__ import annotations

import itertools
import math

import torch
from dabsn.temporal_credit import OnlineCoreFrame, initial_core_eligibility

import credit
import harness
import models
from studies import stream

NEEDS_NATIVE = True


def jobs(protocol):
    return [{"study": study, "seed": seed}
            for study, seed in itertools.product(protocol["studies"], protocol["seeds"])]


def identifier(job):
    return f"{job['study']}-carry-s{job['seed']}"


def _loss(prediction, target, mask):
    return (prediction - target).square()[mask].mean()


def _family_gradients(model):
    out = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        out.setdefault(credit.family(name), []).append(p.grad.detach().double().flatten())
    return {k: torch.cat(v) for k, v in out.items()}


def _compare(exact, online):
    rows = {}
    for name in sorted(exact):
        a = exact[name]; b = online.get(name, torch.zeros_like(a))
        na = float(a.norm()); nb = float(b.norm())
        rows[name] = {"cosine": float((a @ b) / (na * nb)) if na > 0 and nb > 0 else None,
                      "norm_ratio_online_over_exact": nb / na if na > 0 else None}
    a = torch.cat([exact[k] for k in sorted(exact)])
    b = torch.cat([online.get(k, torch.zeros_like(exact[k])) for k in sorted(exact)])
    rows["all"] = {"cosine": float((a @ b) / (a.norm() * b.norm())),
                   "norm_ratio_online_over_exact": float(b.norm() / a.norm())}
    return rows


def run(job, ctx):
    cfg = ctx.matrix[job["study"]]; device = ctx.device; protocol = ctx.protocol
    source = {"phase": "main", "method": "carry", "seed": job["seed"]}
    directory, record = ctx.dependency(job["study"], stream.identifier(source))
    checkpoint = directory / ctx.step_name(cfg["updates"])
    checkpoint = checkpoint.with_suffix(".pt")
    if harness.sha256(checkpoint) != record["result"]["checkpoint_sha256"][checkpoint.name]:
        raise ValueError("Source checkpoint checksum differs")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    outputs_dim = len(cfg["delays"]) if cfg["benchmark"] == "memory_capacity" else 1
    model = models.Model(1, cfg["state_dim"], outputs_dim, job["seed"], device)
    model.load_state_dict(saved["model"])
    core = model.core
    streams = stream.Streams(cfg, job["seed"])
    state = tuple(v.to(device) for v in saved["state"])
    window = protocol["window_chunks"]
    start = cfg["warmup_chunks"] + cfg["updates"]
    if start + protocol["windows"] * window > streams.inputs.shape[1] // cfg["chunk"]:
        raise ValueError("Alignment windows exceed the generated stream")
    windows = []
    for w in range(protocol["windows"]):
        first = start + w * window
        chunks = [streams.get(first + c, device) for c in range(window)]
        x = torch.cat([c[0] for c in chunks], 1)
        target, mask = chunks[-1][1], chunks[-1][2]
        last = chunks[-1][0].shape[1]

        # exact: BPTT through the window from the entering state
        model.zero_grad(set_to_none=True)
        with harness.autocast(device):
            out, _ = core.forward_from_state(x, initial_state=state, return_final_state=True,
                                             temporal_credit="bptt")
        prediction = model.head(torch.tanh(out[0][:, -last:, core.hidden_dim:])).float()
        _loss(prediction, target, mask).backward()
        exact = _family_gradients(model)

        # online: same forward, credit through E started fresh at the window
        model.zero_grad(set_to_none=True)
        e = initial_core_eligibility(core, cfg["batch"], device=device, dtype=torch.float32)
        advancing = state
        for c, (xc, tc_, mc) in enumerate(chunks):
            frame = OnlineCoreFrame(e)
            with harness.autocast(device):
                out, advancing = core.forward_from_state(xc, initial_state=advancing, return_final_state=True,
                                                         temporal_credit="online", eligibility_state=e,
                                                         online_frame=frame)
            if c == window - 1:
                prediction_online = model.head(torch.tanh(out[0][:, :, core.hidden_dim:])).float()
                _loss(prediction_online, tc_, mc).backward()
            else:
                (out[0].float().sum() * 0).backward()
            e = frame.resulting_eligibility
            advancing = tuple(v.detach() for v in advancing)
        if not torch.allclose(prediction_online, prediction, rtol=1e-5, atol=1e-6):
            raise AssertionError("Online and ordinary forwards differ on the alignment window")
        online = _family_gradients(model)
        model.zero_grad(set_to_none=True)
        windows.append(_compare(exact, online))
        state = tuple(v.detach() for v in advancing)

    families = sorted(windows[0])
    summary = {}
    for name in families:
        cos = [w[name]["cosine"] for w in windows if w[name]["cosine"] is not None]
        ratio = [w[name]["norm_ratio_online_over_exact"] for w in windows
                 if w[name]["norm_ratio_online_over_exact"] is not None]
        summary[name] = {"mean_cosine": sum(cos) / len(cos) if cos else None,
                         "min_cosine": min(cos) if cos else None,
                         "mean_norm_ratio": sum(ratio) / len(ratio) if ratio else None}
    readout = summary.get("readout", {}).get("min_cosine")
    if readout is None or readout < 1 - 1e-4:
        raise AssertionError(f"Readout gradients must agree exactly; min cosine {readout}")
    return {"benchmark": cfg["benchmark"], "window_steps": window * cfg["chunk"],
            "windows": windows, "by_family": summary, "source_checkpoint": checkpoint.name,
            "source_checkpoint_sha256": harness.sha256(checkpoint),
            "precision": harness.precision_record(device)}
