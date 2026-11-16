"""Resumable executor for the paper 3 matrix.

    python run.py run    --out DIR [--phases sweep,main] [--only REGEX] [--max-hours H]
    python run.py probe  --out DIR      # measured throughput of every variant
    python run.py plan   --out DIR      # remaining work projected from the probe
    python run.py status --out DIR
    python run.py collect --out DIR     # all results -> DIR/results_all.json

Resume contract. Every artifact is written atomically (temporary file, then
rename), so an interrupted process leaves either the previous or the new
version, never a torn file. Training data is a pure function of
(seed, step), and checkpoints hold the model, optimizer, scaler and step, so a
resumed run continues the identical schedule; at most ``--ckpt-minutes`` of
work repeats. Evaluation is saved item by item. Re-running the same command
skips finished work and continues where it stopped.

Layout under --out:
    train/<train_id>/ckpt.pt, final.pt, train.json, log.jsonl
    runs/<run_id>/eval_partial.json, result.json, error.json
    probe/<gpu>.json, data/enwik8.*
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tasks as T  # noqa: E402
from models import EVAL_VARIANTS, Transformer, param_groups  # noqa: E402

MAX_ATTEMPTS = 2


# ----------------------------------------------------------------------
# Atomic, retrying IO (Google Drive's FUSE mount can fail transiently)

def _retry(fn, *a, tries=5, **kw):
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError):
            raise
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2.0 * (i + 1))


def _atomic(path: Path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    def go():
        writer(tmp)
        os.replace(tmp, path)

    _retry(go)


def clean_tmp(directory: Path):
    """Remove temporary files torn by a process killed mid-write."""
    if directory.exists():
        for p in directory.glob(".*.tmp"):
            try:
                p.unlink()
            except OSError:
                pass


def write_json(path: Path, obj):
    def w(p):
        with open(p, "w") as f:
            json.dump(obj, f, indent=1, default=_jsonable)
            f.flush()
            os.fsync(f.fileno())
    _atomic(path, w)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_json(path: Path, default=None):
    try:
        return _retry(lambda: json.loads(path.read_text()))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def save_torch(path: Path, obj):
    _atomic(path, lambda p: torch.save(obj, p))


def load_torch(path: Path):
    return _retry(torch.load, path, map_location="cpu", weights_only=False)


def append_line(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    _retry(lambda: path.open("a").write(json.dumps(obj, default=_jsonable) + "\n"))


# ----------------------------------------------------------------------
# Environment

def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def gpu_name(device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def framework_hash() -> str:
    import dabsn
    root = Path(dabsn.__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.suffix in {".py", ".cpp", ".cu", ".cuh"}:
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def environment(device) -> dict:
    import dabsn
    return {
        "gpu": gpu_name(device),
        "torch": torch.__version__,
        "dabsn": getattr(dabsn, "__version__", "?"),
        "dabsn_source": framework_hash(),
        "host": socket.gethostname(),
    }


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# Data

def make_task(entry):
    return T.make_task(entry["task"], tuple(entry["loads"]))


ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def enwik8(out: Path) -> dict[str, np.ndarray]:
    d = out / "data"
    paths = {k: d / f"enwik8.{k}.bin" for k in ("train", "val", "test")}
    if not all(p.exists() for p in paths.values()):
        import io
        import urllib.request
        import zipfile
        print(f"downloading {ENWIK8_URL}", flush=True)
        raw = _retry(lambda: urllib.request.urlopen(ENWIK8_URL, timeout=120).read())
        data = zipfile.ZipFile(io.BytesIO(raw)).read("enwik8")
        assert len(data) == 100_000_000, len(data)
        arr = np.frombuffer(data, dtype=np.uint8)
        splits = {"train": arr[:90_000_000], "val": arr[90_000_000:95_000_000],
                  "test": arr[95_000_000:]}
        for k, v in splits.items():
            _atomic(paths[k], lambda p, v=v: v.tofile(p))
    return {k: np.fromfile(p, dtype=np.uint8) for k, p in paths.items()}


_POOL = None


def data_pool() -> ProcessPoolExecutor:
    """Worker processes for synthetic batch generation (one per spare core).

    Processes, not threads: generation is Python-heavy and threads share one
    interpreter lock with the training loop. Batches are the same pure function
    of (seed, step), so the data is bit-identical to single-process generation.
    """
    global _POOL
    if _POOL is None or getattr(_POOL, "_broken", False):
        workers = min(8, max(1, (os.cpu_count() or 2) - 1))
        _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        _POOL.workers = workers
        atexit.register(_POOL.shutdown, wait=False, cancel_futures=True)
    return _POOL


def lm_batch(data: np.ndarray, seed: int, step: int, batch: int, length: int):
    rng = np.random.default_rng([seed, T.SPLITS["train"], step])
    off = rng.integers(0, len(data) - length - 1, size=batch)
    idx = off[:, None] + np.arange(length + 1)[None, :]
    seq = data[idx].astype(np.int64)
    return seq[:, :-1], seq[:, 1:]


# ----------------------------------------------------------------------
# Model / optimizer

def memory_kernels(model) -> list[dict]:
    """Kernel status of every DABSN memory insert after its latest forward."""
    out = []
    for key, ins in model.inserts.items():
        if ins.kind == "dabsn":
            st = dict(ins.branch.kernel_status)
            c = ins.branch.read_contract
            st["read_strategy"] = c.get("read_strategy")
            st["kernel_backend"] = c.get("kernel_backend")
            out.append({"at": int(key), **st})
    return out


def require_memory_kernels(model, ctx) -> list[dict]:
    """On CUDA with kernels='auto', the DABSN read must run on its compact
    CUDA kernel. Fail loudly instead of silently using the reference path."""
    report = memory_kernels(model)
    if ctx.device.type == "cuda" and ctx.kernels == "auto":
        for st in report:
            if not st.get("cuda_admitted_read") or \
                    st.get("read_strategy") != "compact_flash_admitted_bank":
                raise RuntimeError(f"DABSN read kernels are not active: {st}")
    return report


def build_model(entry, device, kernels):
    torch.manual_seed(int(entry["seed"]))
    return Transformer(entry["spec"], memory_kernels=kernels).to(device)


def lr_at(step: int, cfg: dict, lr: float) -> float:
    warm = int(cfg["warmup"])
    if step < warm:
        return lr * (step + 1) / warm
    if cfg.get("schedule", "cosine") == "constant":
        return lr
    prog = (step - warm) / max(1, int(cfg["steps"]) - warm)
    floor = float(cfg["min_lr_ratio"])
    return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def concrete(entry, ctx):
    """Resolve token budgets and the automatic batch into a concrete schedule."""
    tr = entry.get("train")
    if not tr or "tokens" not in tr or tr.get("resolved"):
        return entry
    tr = dict(tr)
    b = int(tr["batch"]) // int(tr.get("batch_div", 1))
    per = b * int(tr["train_len"])
    tr.update(batch=b, steps=max(1, int(tr["tokens"]) // per),
              warmup=max(1, int(tr["warmup_tokens"]) // per),
              val_every=max(1, int(tr["val_every_tokens"]) // per), resolved=True)
    return {**entry, "train": tr}


def train_identity(entry, lr) -> dict:
    ident = {"kind": entry["kind"], "task": entry["task"], "model": entry["model"],
             "spec": entry["spec"], "train": entry["train"], "lr": float(lr),
             "seed": int(entry["seed"])}
    if entry["kind"] == "synthetic":
        ident["loads"] = list(entry["loads"])
        ident["data"] = entry.get("data")
    if entry.get("init_train"):
        ident["init_train"] = entry["init_train"]
    return ident


def train_id(entry, lr) -> str:
    ident = train_identity(entry, lr)
    return f"{entry['task']}__{entry['model']}__lr{lr:g}__s{entry['seed']}__{_hash(ident)}"


class Deadline(Exception):
    pass


def forward_loss(model, x, y, dt):
    with torch.autocast(device_type=x.device.type, dtype=dt, enabled=dt is not None):
        h = model.hidden(x)
    sel = y != T.IGNORE
    logits = model.head(h[sel]).float()
    return F.cross_entropy(logits, y[sel])


def train(entry, lr, ctx) -> Path:
    """Train (or resume) the run's model; returns the directory with final.pt."""
    tdir = ctx.out / "train" / train_id(entry, lr)
    final = tdir / "final.pt"
    if final.exists():
        return tdir
    clean_tmp(tdir)
    cfg = entry["train"]
    device, dt = ctx.device, ctx.amp
    model = build_model(entry, device, ctx.kernels)
    opt = torch.optim.AdamW(param_groups(model, float(cfg["weight_decay"])), lr=lr,
                            betas=tuple(cfg["betas"]), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=dt == torch.float16)
    step, elapsed, tokens = 0, 0.0, 0
    streak = 0
    stop_reason = "budget"
    kernel_report = None
    ckpt = tdir / "ckpt.pt"
    if ckpt.exists():
        state = load_torch(ckpt)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        step, elapsed, tokens = state["step"], state["elapsed"], state["tokens"]
        streak = state.get("solved_streak", 0)
        print(f"    resume {tdir.name} at step {step}", flush=True)
    elif entry.get("init_train"):
        init = load_torch(ctx.out / "train" / entry["init_train"] / "final.pt")
        model.load_state_dict(init["model"])
        print(f"    initialized from {entry['init_train']}", flush=True)
    write_json(tdir / "identity.json", train_identity(entry, lr))

    steps, batch, length = int(cfg["steps"]), int(cfg["batch"]), int(cfg["train_len"])
    if entry["kind"] == "synthetic":
        task = make_task(entry)
        pool = data_pool()
        depth = 2 * pool.workers
        submit = lambda s: pool.submit(T.train_batch, task, int(entry["seed"]), s,  # noqa: E731
                                       batch, length)
    else:
        data = ctx.lm_data()["train"]
        pool = ThreadPoolExecutor(max_workers=2)
        depth = 4
        submit = lambda s: pool.submit(lm_batch, data, int(entry["seed"]), s,  # noqa: E731
                                       batch, length)

    def save(s):
        save_torch(ckpt, {"model": model.state_dict(), "opt": opt.state_dict(),
                          "scaler": scaler.state_dict(), "step": s, "elapsed": elapsed,
                          "tokens": tokens, "solved_streak": streak})

    model.train()
    ahead = {s: submit(s) for s in range(step, min(steps, step + depth))}
    last_ckpt = time.time()
    t0 = time.time()
    seg_start, seg_tokens, loss_n = time.time(), 0, 0
    loss_sum = torch.zeros((), device=device)
    diverged = False
    try:
        while step < steps:
            x_np, y_np = ahead.pop(step).result()
            nxt = step + depth
            if nxt < steps:
                ahead[nxt] = submit(nxt)
            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            y = torch.from_numpy(y_np).to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg, lr)
            loss = forward_loss(model, x, y, dt)
            if kernel_report is None:
                kernel_report = require_memory_kernels(model, ctx)
                if kernel_report:
                    print(f"    memory kernels: {kernel_report}", flush=True)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            step += 1
            tokens += x.numel()
            seg_tokens += x.numel()
            loss_sum += loss.detach()
            loss_n += 1
            if step % ctx.log_every == 0 or step == steps:
                if not math.isfinite(float(loss_sum)):
                    diverged = True
                    stop_reason = "diverged"
                    print(f"    non-finite loss by step {step}; stopping as diverged", flush=True)
                    break
                now = time.time()
                rate = seg_tokens / max(now - seg_start, 1e-9)
                rec = {"step": step, "loss": float(loss_sum) / max(loss_n, 1),
                       "lr": opt.param_groups[0]["lr"], "tok_per_s": rate}
                append_line(tdir / "log.jsonl", rec)
                print(f"    step {step}/{steps} loss {rec['loss']:.4f} {rate:,.0f} tok/s",
                      flush=True)
                seg_start, seg_tokens, loss_n = now, 0, 0
                loss_sum.zero_()
            val_every = int(cfg.get("val_every", 0))
            if val_every and step % val_every == 0 and step < steps:
                v = quick_val(entry, model, ctx)
                append_line(tdir / "log.jsonl", {"step": step, **v})
                print(f"    step {step} " + " ".join(f"{k} {x:.4f}" for k, x in v.items()),
                      flush=True)
                if "solved_acc" in cfg:
                    streak = streak + 1 if v["val_acc"] >= float(cfg["solved_acc"]) else 0
                    if streak >= int(cfg["solved_checks"]):
                        stop_reason = "solved"
                        print(f"    solved at step {step}", flush=True)
                        break
            if time.time() - last_ckpt > ctx.ckpt_seconds:
                elapsed += time.time() - t0
                t0 = time.time()
                save(step)
                last_ckpt = time.time()
            if ctx.past_deadline():
                elapsed += time.time() - t0
                save(step)
                raise Deadline()
    finally:
        for f in ahead.values():
            f.cancel()
        if isinstance(pool, ThreadPoolExecutor):
            pool.shutdown(wait=False, cancel_futures=True)
    elapsed += time.time() - t0
    save_torch(final, {"model": model.state_dict()})
    write_json(tdir / "train.json", {
        "steps_done": step, "steps": steps, "diverged": diverged, "stop_reason": stop_reason, "train_seconds": elapsed,
        "tokens": tokens, "tok_per_s": tokens / max(elapsed, 1e-9), "lr": lr,
        "memory_kernels": kernel_report,
        "env": ctx.env,
    })
    if ckpt.exists():
        ckpt.unlink()
    return tdir


def quick_val(entry, model, ctx) -> dict:
    """Held-out validation during training (learning curve and stopping).

    Uses validation-split indices disjoint from the final validation set, and
    never test data."""
    cfg = entry["train"]
    tl, n = int(cfg["train_len"]), int(cfg["val_check_seqs"])
    model.eval()
    try:
        if entry["kind"] == "synthetic":
            task = make_task(entry)

            def sample(i):
                j = 1_000_000 + i
                rng = np.random.default_rng([ctx.eval_seed, 1, 99, j])
                return T.eval_sample(task, ctx.eval_seed, "val", "val", 1, j, tl,
                                     task.train_load(rng), "random")

            r = _batched_eval(model, sample, n, tl, ctx, variant="base", train_len=tl,
                              budget=entry["eval"]["eval_tokens_per_batch"])
            acc = r.get("acc", 0.0)
            return {"val_acc": acc if math.isfinite(acc) else 0.0}
        data = ctx.lm_data()["val"]
        offs = np.linspace(len(data) // 2, len(data) - tl - 2, n).astype(np.int64)
        seq = data[offs[:, None] + np.arange(tl + 1)[None, :]].astype(np.int64)
        x = torch.from_numpy(seq[:, :-1]).to(ctx.device)
        y = torch.from_numpy(seq[:, 1:]).to(ctx.device)
        with torch.no_grad():
            h = lm_hidden(model, x, ctx, "base", tl)
            loss = F.cross_entropy(model.head(h).float().flatten(0, 1), y.flatten())
        return {"val_loss": float(loss)}
    finally:
        model.train()


def load_final(entry, tdir: Path, ctx):
    model = build_model(entry, ctx.device, ctx.kernels)
    model.load_state_dict(load_torch(tdir / "final.pt")["model"])
    model.eval()
    return model


# ----------------------------------------------------------------------
# Synthetic evaluation

def _batched_eval(model, samples_fn, n, length, ctx, *, variant, train_len, budget,
                  batch=None, collect_depth=False, collect_mech=False):
    """Accuracy on n generated sequences. Halves the batch on OOM."""
    B = batch or max(1, budget // length)
    correct = total = seq_ok = 0
    depth_hit = np.zeros(10)
    depth_n = np.zeros(10)
    mech = []
    i = 0
    t0 = time.time()
    while i < n:
        b = min(B, n - i)
        samples = [samples_fn(i + j) for j in range(b)]
        x = torch.from_numpy(np.stack([s.x for s in samples])).to(ctx.device)
        y = torch.from_numpy(np.stack([s.y for s in samples])).to(ctx.device)
        try:
            with torch.no_grad(), torch.autocast(device_type=x.device.type, dtype=ctx.amp,
                                                 enabled=ctx.amp is not None):
                h = model.hidden(x, rope_mode=variant, train_len=train_len)
                sel = y != T.IGNORE
                pred = model.head(h[sel]).argmax(-1)
        except Exception as exc:  # noqa: BLE001
            if not is_oom(exc):
                raise
            del exc
            free_cuda()
            if B == 1:
                return {"status": "oom", "length": length, "done": i}
            B = max(1, B // 2)
            continue
        hit = (pred == y[sel])
        rows = sel.nonzero()[:, 0]
        per_row = torch.zeros(b, device=x.device).index_add_(0, rows, hit.float())
        need = sel.sum(1).float()
        correct += int(hit.sum())
        total += int(sel.sum())
        seq_ok += int((per_row == need).sum())
        if collect_depth:
            for j, s in enumerate(samples):
                k = min(9, int(s.meta["depth"] * 10))
                depth_n[k] += 1
                depth_hit[k] += float(per_row[j] / need[j])
        if collect_mech:
            mech.append(model.memory_stats())
        i += b
    out = {"status": "ok", "acc": correct / max(total, 1), "seq_acc": seq_ok / max(n, 1),
           "targets": total, "seqs": n, "length": length, "batch": B,
           "seconds": time.time() - t0}
    if collect_depth:
        out["depth_acc"] = [float(h / c) if c else None for h, c in zip(depth_hit, depth_n)]
    if collect_mech:
        out["mech"] = mech
    return out


def eval_train_len(entry) -> int:
    return int(entry.get("eval_train_len", entry["train"]["train_len"]))


def synthetic_items(entry, task) -> list[tuple]:
    ev, tl = entry["eval"], eval_train_len(entry)
    hi = max(entry["loads"])
    variants = EVAL_VARIANTS.get(entry["model"], ["base"])
    items = []
    for kind in entry["evals"]:
        if kind == "val":
            items.append(("val", 1, "base", tl, None))
        elif kind == "distance":
            for m in ev["distance_multipliers"]:
                for v in (variants if m > 1 else ["base"]):
                    items.append(("distance", m, v, tl * m, hi))
        elif kind == "load":
            for f in ev["load_multipliers"]:
                load = hi * f
                length = task.load_axis_length(tl, load)
                for v in (variants if length > tl else ["base"]):
                    items.append(("load", f, v, length, load))
        elif kind == "mech":
            if entry["spec"]["insert"] == "dabsn":
                for m in ev["distance_multipliers"]:
                    items.append(("mech", m, "base", tl * m, hi))
    return items


def eval_synthetic(entry, model, rdir: Path, ctx):
    task = make_task(entry)
    ev, tl = entry["eval"], eval_train_len(entry)
    partial_path = rdir / "eval_partial.json"
    partial = read_json(partial_path, {}) or {}
    for axis, value, variant, length, load in synthetic_items(entry, task):
        key = f"{axis}/{value}/{variant}"
        if key in partial:
            continue
        split = "val" if axis == "val" else "test"
        placement = "random" if axis == "val" else "max_gap"
        n = {"val": ev["val_seqs"], "mech": ev["mech_samples"]}.get(axis, ev["test_seqs"])

        def sample(i, axis=axis, value=value, length=length, load=load, split=split,
                   placement=placement):
            rng_load = load
            if rng_load is None:  # validation mirrors the training load distribution
                rng = np.random.default_rng([ctx.eval_seed, 1, 99, i])
                rng_load = task.train_load(rng)
            return T.eval_sample(task, ctx.eval_seed, split, axis, value, i, length,
                                 rng_load, placement)

        res = _batched_eval(model, sample, n, length, ctx, variant=variant, train_len=tl,
                            budget=ctx.eval_tokens or ev["eval_tokens_per_batch"],
                            batch=1 if axis == "mech" else None,
                            collect_depth=entry["task"] == "passkey" and axis != "mech",
                            collect_mech=axis == "mech")
        res.update({"axis": axis, "value": value, "variant": variant, "load": load})
        partial[key] = res
        write_json(partial_path, partial)
        if res["status"] == "ok":
            print(f"    {key:22s} T={length:<6d} acc {res['acc']:.3f} "
                  f"({res['seconds']:.0f}s)", flush=True)
        else:
            print(f"    {key:22s} T={length:<6d} {res['status']}", flush=True)
        ctx.check_deadline()
    return partial


def eval_extreme(entry, model, rdir: Path, ctx):
    task = make_task(entry)
    ex = entry["extreme"]
    hi = max(entry["loads"])
    partial_path = rdir / "eval_partial.json"
    partial = read_json(partial_path, {}) or {}
    tl = ctx.train_len_of(entry["train_from"])
    variants = EVAL_VARIANTS.get(entry["model"], ["base"])
    for variant in variants:
        stop = False
        for length in ex["lengths"]:
            key = f"extreme/{length}/{variant}"
            if key in partial:
                stop = partial[key]["status"] != "ok"
                continue
            if stop:
                partial[key] = {"status": "skipped_after_previous", "length": length}
                write_json(partial_path, partial)
                continue
            hits, n_done, t0 = [], 0, time.time()
            status = "ok"
            for i in range(ex["samples"]):
                s = T.eval_sample(task, ctx.eval_seed, "test", "extreme", length, i,
                                  length, hi, "max_gap")
                r = _batched_eval(model, lambda _i, s=s: s, 1, length, ctx, variant=variant,
                                  train_len=tl, budget=length, batch=1)
                if r["status"] != "ok":
                    status = r["status"]
                    break
                hits.append({"acc": r["acc"], "seq_acc": r["seq_acc"],
                             "depth": s.meta["depth"], "seconds": r["seconds"]})
                n_done += 1
                spent = time.time() - t0
                if spent / n_done * ex["samples"] > ex["max_seconds_per_length"] and \
                        n_done < ex["samples"]:
                    status = "too_slow"
                    break
                ctx.check_deadline()
            res = {"status": status, "length": length, "samples": hits, "done": n_done,
                   "seconds": time.time() - t0}
            if hits:
                res["acc"] = float(np.mean([h["acc"] for h in hits]))
                res["seq_acc"] = float(np.mean([h["seq_acc"] for h in hits]))
            partial[key] = res
            write_json(partial_path, partial)
            print(f"    {key:28s} {status} acc {res.get('acc', float('nan')):.3f} "
                  f"({res['seconds']:.0f}s)", flush=True)
            stop = status != "ok"
    return partial


def eval_qearly(entry, model, rdir: Path, ctx):
    """A model trained with the query last, tested with the query also shown
    before the data. At each distance both versions of the same sequences are
    run one at a time (batch 1), so accuracy, the admitted bank and the read
    work are recorded per sample, query late and query early. The query-late
    sequences are the ones of the trained model's ``mech`` evaluation."""
    task = make_task(entry)
    q = entry["query_early"]
    hi = max(entry["loads"])
    tl = ctx.train_len_of(entry["train_from"])
    partial_path = rdir / "eval_partial.json"
    partial = read_json(partial_path, {}) or {}
    for m in q["distance_multipliers"]:
        length = tl * m
        for cond in ("late", "early"):
            key = f"{cond}/{m}"
            if key in partial:
                continue

            def sample(i, m=m, length=length, cond=cond):
                s = T.eval_sample(task, ctx.eval_seed, "test", "mech", m, i, length, hi,
                                  "max_gap")
                return task.query_first(s) if cond == "early" else s

            res = _batched_eval(model, sample, int(q["samples"]), length, ctx, variant="base",
                                train_len=tl, budget=length, batch=1, collect_mech=True)
            res.update({"axis": "query", "query": cond, "value": m, "load": hi})
            partial[key] = res
            write_json(partial_path, partial)
            if res["status"] == "ok":
                bank = [s[0]["admitted_bank"] for s in res["mech"]]
                print(f"    {key:10s} T={length:<6d} acc {res['acc']:.3f}  admitted bank "
                      f"mean {np.mean(bank):.1f} ({res['seconds']:.0f}s)", flush=True)
            else:
                print(f"    {key:10s} T={length:<6d} {res['status']}", flush=True)
            ctx.check_deadline()
    return partial


# ----------------------------------------------------------------------
# Language-model evaluation

def lm_hidden(model, x, ctx, variant, train_len):
    with torch.autocast(device_type=x.device.type, dtype=ctx.amp, enabled=ctx.amp is not None):
        return model.hidden(x, rope_mode=variant, train_len=train_len)


def eval_lm(entry, model, rdir: Path, ctx):
    ev, tl = entry["eval"], entry["train"]["train_len"]
    data = ctx.lm_data()
    partial_path = rdir / "eval_partial.json"
    partial = read_json(partial_path, {}) or {}
    variants = EVAL_VARIANTS.get(entry["model"], ["base"])

    @torch.no_grad()
    def token_losses(seq: np.ndarray, variant, length):
        x = torch.from_numpy(seq[:, :-1].astype(np.int64)).to(ctx.device)
        y = torch.from_numpy(seq[:, 1:].astype(np.int64)).to(ctx.device)
        h = lm_hidden(model, x, ctx, variant, tl)
        out = torch.empty(y.shape, device=ctx.device)
        for s in range(0, length, 4096):
            e = min(length, s + 4096)
            logits = model.head(h[:, s:e]).float()
            out[:, s:e] = F.cross_entropy(logits.transpose(1, 2), y[:, s:e], reduction="none")
        return out

    for kind in entry["evals"]:
        if kind == "val" and "val" not in partial:
            vdata = data["val"]
            n = ev["val_seqs"]
            offs = np.linspace(0, len(vdata) - tl - 2, n).astype(np.int64)
            tot, t0 = 0.0, time.time()
            B = max(1, ev["eval_tokens_per_batch"] // tl)
            for s in range(0, n, B):
                o = offs[s:s + B]
                seq = vdata[o[:, None] + np.arange(tl + 1)[None, :]]
                tot += float(token_losses(seq, "base", tl).mean()) * len(o)
            partial["val"] = {"status": "ok", "loss": tot / n, "bpc": tot / n / math.log(2),
                              "seconds": time.time() - t0}
            write_json(partial_path, partial)
            print(f"    val loss {partial['val']['loss']:.4f}", flush=True)
        elif kind == "position":
            L = tl * ev["position_multiplier"]
            tdata = data["test"]
            offs = np.linspace(0, len(tdata) - L - 2, ev["position_seqs"]).astype(np.int64)
            for variant in variants:
                key = f"position/{variant}"
                if key in partial:
                    continue
                sums = np.zeros(L)
                t0 = time.time()
                status = "ok"
                for o in offs:
                    seq = tdata[o + np.arange(L + 1)][None, :]
                    try:
                        sums += token_losses(seq, variant, L)[0].double().cpu().numpy()
                    except Exception as exc:  # noqa: BLE001
                        if not is_oom(exc):
                            raise
                        free_cuda()
                        status = "oom"
                        break
                mean = sums / len(offs)
                edges = [0] + [tl * 2 ** k for k in range(int(math.log2(ev["position_multiplier"])) + 1)]
                buckets = [{"from": a, "to": b, "loss": float(mean[a:b].mean())}
                           for a, b in zip(edges[:-1], edges[1:])]
                fine = np.unique(np.geomspace(1, L, 200).astype(int)) - 1
                partial[key] = {"status": status, "length": L, "buckets": buckets,
                                "curve_pos": fine.tolist(),
                                "curve_loss": [float(mean[max(0, p - 31):p + 1].mean())
                                               for p in fine],
                                "seconds": time.time() - t0}
                write_json(partial_path, partial)
                print(f"    {key} " + " ".join(f"{b['loss']:.3f}" for b in buckets), flush=True)
                ctx.check_deadline()
        elif kind == "passkey":
            for variant in variants:
                for m in ev["passkey_multipliers"]:
                    key = f"passkey/{m}/{variant}"
                    if key in partial:
                        continue
                    partial[key] = lm_passkey(model, tl * m, ev, variant, tl, ctx)
                    write_json(partial_path, partial)
                    print(f"    {key} acc {partial[key].get('acc', float('nan')):.3f}",
                          flush=True)
                    ctx.check_deadline()
    return partial


MJ_TASK = ("There is an important info hidden inside a lot of irrelevant text. Find it and "
           "memorize them. I will quiz you about the important information there.")
MJ_GARBAGE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
MJ_QUESTION = "What is the pass key? The pass key is"


def mj_passkey_prompt(rng, length: int) -> tuple[bytes, bytes]:
    """The passkey prompt of Mohtashami & Jaggi (2023, ``generate_prompt``),
    sized to exactly ``length`` characters including the answer. The answer is
    the key after one space; only its digits are scored."""
    key = str(int(rng.integers(1, 50001))).encode()
    info = b"The pass key is " + key + b". Remember it. " + key + b" is the pass key."
    fixed = len(MJ_TASK) + len(info) + len(MJ_QUESTION) + 4 + 1 + len(key)
    n_garbage = length - fixed
    if n_garbage < 0:
        raise ValueError("passkey prompt does not fit")
    pre = int(rng.integers(0, n_garbage + 1))
    inf = " ".join([MJ_GARBAGE] * (n_garbage // len(MJ_GARBAGE) + 2)).encode()
    prompt = b"\n".join([MJ_TASK.encode(), inf[:pre], info, inf[:n_garbage - pre],
                         MJ_QUESTION.encode()]) + b" "
    return prompt, key


def lm_passkey(model, length: int, ev, variant, tl, ctx):
    hits, exact, t0 = [], [], time.time()
    for i in range(ev["passkey_samples"]):
        rng = np.random.default_rng([ctx.eval_seed, 2, 7, length, i])
        prompt, ans = mj_passkey_prompt(rng, length)
        arr = np.frombuffer(prompt + ans, dtype=np.uint8).astype(np.int64)
        assert len(arr) == length
        x = torch.from_numpy(arr[:-1])[None].to(ctx.device)
        tgt_pos = torch.arange(len(arr) - 1 - len(ans), len(arr) - 1, device=ctx.device)
        try:
            with torch.no_grad():
                h = lm_hidden(model, x, ctx, variant, tl)
                pred = model.head(h[0, tgt_pos]).argmax(-1).cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            if not is_oom(exc):
                raise
            free_cuda()
            return {"status": "oom", "length": length}
        ok = pred == arr[-len(ans):]
        hits.append(float(ok.mean()))
        exact.append(float(ok.all()))
    return {"status": "ok", "length": length, "acc": float(np.mean(hits)),
            "exact": float(np.mean(exact)), "seconds": time.time() - t0}


# ----------------------------------------------------------------------
# Orchestration

class Ctx:
    def __init__(self, args, protocol):
        self.out = Path(args.out)
        self.device = pick_device(args.device)
        self.amp = amp_dtype(self.device) if not args.no_amp else None
        if self.device.type == "cuda":
            # TF32 for the fp32 matmuls (memory branch, output head).
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.kernels = args.memory_kernels
        self.ckpt_seconds = 60.0 * args.ckpt_minutes
        self.log_every = args.log_every
        self.deadline = time.time() + 3600 * args.max_hours if args.max_hours else None
        self.protocol = protocol
        self.by_id = {e["id"]: e for e in protocol["runs"]}
        self.eval_seed = int(protocol["eval_seed"])
        self.eval_tokens = int(args.eval_tokens) if args.eval_tokens else None
        self.retry_errors = bool(getattr(args, "retry_errors", False))
        self._lm = None
        self.env = environment(self.device)

    def past_deadline(self):
        return self.deadline is not None and time.time() > self.deadline

    def check_deadline(self):
        if self.past_deadline():
            raise Deadline()

    def lm_data(self):
        if self._lm is None:
            self._lm = enwik8(self.out)
        return self._lm

    def train_len_of(self, run_id):
        return int(self.by_id[run_id]["train"]["train_len"])


def entry_dir(out: Path, entry) -> Path:
    """Results are keyed by the exact declared entry: a changed protocol entry
    can never be mistaken for a finished one."""
    return Path(out) / "runs" / f"{entry['id']}__{_hash(entry)}"


def run_dir(ctx, entry) -> Path:
    return entry_dir(ctx.out, entry)


def _outcome(ctx, rid):
    """('result', res) | ('excluded', None) | ('pending', None) for a run id.

    Skipped edge extensions and runs that failed MAX_ATTEMPTS times are
    excluded from selection instead of blocking it forever."""
    rdir = run_dir(ctx, ctx.by_id[rid])
    res = read_json(rdir / "result.json")
    if res is not None:
        return ("excluded", None) if res.get("skipped") else ("result", res)
    err = read_json(rdir / "error.json", {}) or {}
    if err.get("attempts", 0) >= MAX_ATTEMPTS and not ctx.retry_errors:
        return "excluded", None
    return "pending", None


def _score(res, metric) -> float:
    ev = res["eval"]
    if metric == "val_acc":
        v = ev.get("val/1/base", {}).get("acc")
        return -1.0 if v is None or not math.isfinite(v) else v
    v = ev.get("val", {}).get("loss")
    return -1e9 if v is None or not math.isfinite(v) else -v


def _selection(ctx, ids, metric):
    """[(score, -lr, lr)] over finished runs, or None while any is pending.
    max() picks the best score, ties going to the smaller rate."""
    scores = []
    for rid in ids:
        state, res = _outcome(ctx, rid)
        if state == "pending":
            return None
        if state == "result":
            scores.append((_score(res, metric), -float(res["lr"]), float(res["lr"])))
    return scores


def extension_needed(entry, ctx):
    """For a conditional edge-extension sweep entry: True (train it), False
    (skip it), or None while earlier sweep entries are unfinished."""
    x = entry["extend"]
    scores = _selection(ctx, x["after"], x["metric"])
    if scores is None:
        return None
    if not scores:
        return False
    best = max(scores)
    rates = [r for _, _, r in scores]
    edge = min(rates) if x["direction"] == "low" else max(rates)
    if best[2] != edge:
        return False
    if x["metric"] == "val_acc" and x.get("solved") is not None and best[0] >= x["solved"]:
        return False
    return True


def resolve_lr(entry, ctx):
    """Learning rate of an entry, or None while its selection is incomplete."""
    if entry["lr"] == "init":  # fine-tune at the source run's selected rate
        src = ctx.by_id[entry["init_from"]]
        if not (run_dir(ctx, src) / "result.json").exists():
            return None
        return resolve_lr(src, ctx)
    if entry["lr"] != "best":
        return float(entry["lr"])
    scores = _selection(ctx, entry["select_from"], entry["select_metric"])
    if not scores:
        return None
    return max(scores)[2]


def execute(entry, ctx) -> str:
    rdir = run_dir(ctx, entry)
    if (rdir / "result.json").exists():
        return "done"
    err = read_json(rdir / "error.json", {}) or {}
    if err.get("attempts", 0) >= MAX_ATTEMPTS and not ctx.retry_errors:
        return "error"
    if entry.get("extend"):
        need = extension_needed(entry, ctx)
        if need is None:
            return "blocked"
        if not need:
            write_json(rdir / "result.json", {
                "id": entry["id"], "entry": entry, "lr": float(entry["lr"]), "skipped": True,
                "reason": "best rate is not at this edge of the grid, or it solved the task",
                "eval": {}, "protocol_hash": ctx.protocol["protocol_hash"],
                "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"\n== {entry['id']}  edge extension not needed; skipped", flush=True)
            return "skipped"
        print(f"\n   best rate is at the {entry['extend']['direction']} edge and unsolved; "
              f"extending the sweep", flush=True)
    if entry["kind"] in ("extreme", "qearly"):
        src = ctx.by_id[entry["train_from"]]
        lr = resolve_lr(src, ctx)
        if lr is None or not (run_dir(ctx, src) / "result.json").exists():
            return "blocked"
        train_src = concrete({**src, "seed": entry["seed"]}, ctx)
    else:
        lr = resolve_lr(entry, ctx)
        if lr is None:
            return "blocked"
        train_src = concrete(entry, ctx)
        if entry.get("init_from"):
            src = concrete(ctx.by_id[entry["init_from"]], ctx)
            train_src = {**train_src, "init_train": train_id(src, resolve_lr(src, ctx))}
    print(f"\n== {entry['id']}  (lr {lr:g})", flush=True)
    clean_tmp(rdir)
    t0 = time.time()
    try:
        tdir = train(train_src, lr, ctx)
        model = load_final(train_src, tdir, ctx)
        if entry["kind"] == "synthetic":
            ev = eval_synthetic(entry, model, rdir, ctx)
        elif entry["kind"] == "extreme":
            ev = eval_extreme(entry, model, rdir, ctx)
        elif entry["kind"] == "qearly":
            ev = eval_qearly(entry, model, rdir, ctx)
        else:
            ev = eval_lm(entry, model, rdir, ctx)
        result = {
            "id": entry["id"], "entry": entry, "lr": lr, "train_id": tdir.name,
            "train": read_json(tdir / "train.json"), "eval": ev,
            "params": sum(p.numel() for p in model.parameters()),
            "env": ctx.env, "protocol_hash": ctx.protocol["protocol_hash"],
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(rdir / "result.json", result)
        print(f"   done in {time.time() - t0:.0f}s", flush=True)
        return "done"
    except Deadline:
        raise
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        write_json(rdir / "error.json", {"attempts": err.get("attempts", 0) + 1,
                                          "error": f"{type(exc).__name__}: {exc}",
                                          "traceback": tb})
        free_cuda()
        return "error"


def select_entries(protocol, args):
    phases = set(args.phases.split(",")) if args.phases else None
    pat = re.compile(args.only) if args.only else None
    out = []
    for e in protocol["runs"]:
        if phases and e["phase"] not in phases:
            continue
        if pat and not pat.search(e["id"]):
            continue
        out.append(e)
    return out


def cmd_run(args, protocol):
    ctx = Ctx(args, protocol)
    print(f"device {ctx.env['gpu']}  amp {ctx.amp}  dabsn {ctx.env['dabsn']} "
          f"({ctx.env['dabsn_source']})  out {ctx.out}", flush=True)
    entries = select_entries(protocol, args)
    if not args.no_probe and ctx.device.type == "cuda":
        probe(ctx, entries)
        plan(ctx, entries)
    counts: dict[str, int] = {}
    try:
        while True:
            counts = {}
            for e in entries:
                st = execute(e, ctx)
                counts[st] = counts.get(st, 0) + 1
            # Another pass picks up entries unblocked by work done in this pass.
            if counts.get("blocked", 0) == 0 or not progressed_since(ctx, entries):
                break
    except Deadline:
        print("\nstopped at --max-hours; re-run the same command to continue", flush=True)
        return
    print("\nsummary " + json.dumps(counts), flush=True)


def progressed_since(ctx, entries):
    """True if some blocked entry can now be resolved."""
    for e in entries:
        if (run_dir(ctx, e) / "result.json").exists():
            continue
        if e["kind"] in ("extreme", "qearly"):
            src = ctx.by_id[e["train_from"]]
            if (run_dir(ctx, src) / "result.json").exists():
                return True
        elif e["lr"] in ("best", "init") and resolve_lr(e, ctx) is not None:
            err = read_json(run_dir(ctx, e) / "error.json", {}) or {}
            if err.get("attempts", 0) < MAX_ATTEMPTS:
                return True
        elif e.get("extend") and extension_needed(e, ctx) is not None:
            err = read_json(run_dir(ctx, e) / "error.json", {}) or {}
            if err.get("attempts", 0) < MAX_ATTEMPTS:
                return True
    return False


# ----------------------------------------------------------------------
# Timing probe and projection

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def probe_key(e) -> str:
    key = f"{e['kind']}/{e['model']}"
    if e.get("size_tag"):
        key += f"/{e['size_tag']}"
    if e.get("train"):
        key += f"/b{e['train']['batch']}"
    return key


def probe_variants(entries, ctx):
    """One representative entry per (kind, model): cost depends on size and
    architecture, not on which synthetic task supplies the tokens."""
    seen, reps = set(), []
    for e in entries:
        e = concrete(e, ctx)
        if e["kind"] in ("extreme", "qearly"):
            continue
        k = probe_key(e)
        if k not in seen:
            seen.add(k)
            reps.append(e)
    return reps


def probe(ctx, entries, warm=5, timed=20):
    path = ctx.out / "probe" / f"{slug(ctx.env['gpu'])}.json"
    data = read_json(path, {}) or {}
    for e in probe_variants(entries, ctx):
        key = probe_key(e)
        if key in data:
            continue
        print(f"probe {key}", flush=True)
        entry = concrete({**e, "seed": 0}, ctx)
        cfg = entry["train"]
        model = build_model(entry, ctx.device, ctx.kernels)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        B, L = int(cfg["batch"]), int(cfg["train_len"])
        if e["kind"] == "synthetic":
            task = make_task(e)
            get = lambda s: T.train_batch(task, 0, s, B, L)  # noqa: E731
        else:
            tr = ctx.lm_data()["train"]
            get = lambda s: lm_batch(tr, 0, s, B, L)  # noqa: E731
        batches = [get(s) for s in range(warm + timed)]
        rec = {}
        try:
            for s, (xn, yn) in enumerate(batches):
                if s == warm:
                    _sync(ctx.device)
                    t0 = time.time()
                    if ctx.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                x, y = torch.from_numpy(xn).to(ctx.device), torch.from_numpy(yn).to(ctx.device)
                loss = forward_loss(model, x, y, ctx.amp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            _sync(ctx.device)
            dt = time.time() - t0
            rec["train_steps_per_s"] = timed / dt
            rec["train_tok_per_s"] = timed * B * L / dt
            rec["memory_kernels"] = require_memory_kernels(model, ctx)
            if rec["memory_kernels"]:
                print(f"  memory kernels: {rec['memory_kernels']}", flush=True)
            if ctx.device.type == "cuda":
                rec["train_peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        except Exception as exc:  # noqa: BLE001
            if not is_oom(exc):
                raise
            free_cuda()
            rec["train_status"] = "oom"
        del opt
        model.eval()
        # Evaluation forward cost per sequence at each length on the grid.
        if e["kind"] == "synthetic":
            lengths = [L * m for m in e["eval"]["distance_multipliers"]]
        else:
            lengths = [L * m for m in e["eval"]["passkey_multipliers"]]
        rec["eval_seconds_per_seq"] = {}
        for length in lengths:
            nb = max(1, (ctx.eval_tokens or (e["eval"]["eval_tokens_per_batch"])) // length)
            x = torch.randint(0, entry["spec"]["vocab"], (nb, length), device=ctx.device)
            try:
                for rep in range(2):
                    _sync(ctx.device)
                    t0 = time.time()
                    with torch.no_grad(), torch.autocast(device_type=x.device.type,
                                                         dtype=ctx.amp,
                                                         enabled=ctx.amp is not None):
                        model.hidden(x, train_len=L)
                    _sync(ctx.device)
                rec["eval_seconds_per_seq"][str(length)] = (time.time() - t0) / nb
            except Exception as exc:  # noqa: BLE001
                if not is_oom(exc):
                    raise
                free_cuda()
                rec["eval_seconds_per_seq"][str(length)] = None
        del model
        free_cuda()
        data[key] = rec
        write_json(path, data)
        print(f"  {json.dumps(rec)}", flush=True)
    return data


def _eval_seconds(rec, length):
    table = {int(k): v for k, v in rec.get("eval_seconds_per_seq", {}).items() if v}
    if not table:
        return 0.0
    if length in table:
        return table[length]
    near = min(table, key=lambda k: abs(math.log(k / length)))
    return table[near] * length / near


def plan(ctx, entries):
    path = ctx.out / "probe" / f"{slug(ctx.env['gpu'])}.json"
    data = read_json(path, {}) or {}
    if not data:
        print("no probe for this GPU; run `probe` first")
        return
    per_phase: dict[str, float] = {}
    trained = set()
    for raw in entries:
        if (run_dir(ctx, raw) / "result.json").exists():
            continue
        e = concrete(raw, ctx)
        if e["kind"] == "extreme":
            ex = e["extreme"]
            per_phase["extreme (upper bound)"] = per_phase.get("extreme (upper bound)", 0) + \
                ex["max_seconds_per_length"] * len(ex["lengths"]) * \
                len(EVAL_VARIANTS.get(e["model"], ["base"]))
            continue
        if e["kind"] == "qearly":
            src = concrete(ctx.by_id[e["train_from"]], ctx)
            rec = data.get(probe_key(src), {})
            q, tl = e["query_early"], int(src["train"]["train_len"])
            per_phase[e["phase"]] = per_phase.get(e["phase"], 0.0) + sum(
                2 * int(q["samples"]) * _eval_seconds(rec, tl * m)
                for m in q["distance_multipliers"])
            continue
        rec = data.get(probe_key(e), {})
        sps = rec.get("train_steps_per_s")
        secs = 0.0
        lr_key = e["lr"] if e["lr"] != "best" else "best"
        tkey = (e["task"], e["model"], lr_key, e["seed"], json.dumps(e["train"], sort_keys=True))
        if sps and tkey not in trained and not (e["lr"] == "best" and e["seed"] == 0):
            secs += int(e["train"]["steps"]) / sps
            trained.add(tkey)
        if e["kind"] == "synthetic":
            task = make_task(e)
            for axis, value, variant, length, load in synthetic_items(e, task):
                n = {"val": e["eval"]["val_seqs"], "mech": e["eval"]["mech_samples"]}.get(
                    axis, e["eval"]["test_seqs"])
                secs += n * _eval_seconds(rec, length)
        key = e["phase"] + (" (LR edge, only if needed)" if e.get("extend") else "")
        per_phase[key] = per_phase.get(key, 0.0) + secs
    total = sum(per_phase.values())
    cond = sum(v for k, v in per_phase.items() if "if needed" in k)
    print(f"\nremaining work projected from the measured probe on {ctx.env['gpu']}"
          " (training at the full step cap; runs that solve stop earlier):")
    for k, v in per_phase.items():
        print(f"  {k:40s} {v / 3600:7.2f} h")
    print(f"  {'total without conditional runs':40s} {(total - cond) / 3600:7.2f} h")
    print(f"  {'total upper bound':40s} {total / 3600:7.2f} h\n", flush=True)


def cmd_status(args, protocol):
    out = Path(args.out)
    rows: dict[str, dict[str, int]] = {}
    for e in select_entries(protocol, args):
        rdir = entry_dir(out, e)
        if (rdir / "result.json").exists():
            st = "skipped" if (read_json(rdir / "result.json") or {}).get("skipped") else "done"
        elif (rdir / "error.json").exists():
            st = "error"
        elif (rdir / "eval_partial.json").exists():
            st = "evaluating"
        else:
            st = "pending"
        rows.setdefault(e["phase"], {}).setdefault(st, 0)
        rows[e["phase"]][st] += 1
    for ph, c in rows.items():
        print(f"{ph:10s} " + "  ".join(f"{k} {v}" for k, v in sorted(c.items())))
    active = [p.parent.name for p in (out / "train").glob("*/ckpt.pt")]
    if active:
        print("training in progress (checkpointed): " + ", ".join(active))


def cmd_collect(args, protocol):
    out = Path(args.out)
    results = []
    for e in protocol["runs"]:
        rdir = entry_dir(out, e)
        r = read_json(rdir / "result.json")
        if r is not None:
            results.append(r)
            continue
        # A run that failed partway keeps every evaluation item it finished;
        # report those, marked partial, together with the error.
        partial = read_json(rdir / "eval_partial.json")
        if partial:
            err = read_json(rdir / "error.json", {}) or {}
            results.append({"id": e["id"], "entry": e, "partial": True, "eval": partial,
                            "error": err.get("error"), "attempts": err.get("attempts")})
    write_json(out / "results_all.json", {"protocol_hash": protocol["protocol_hash"],
                                          "results": results})
    print(f"{len(results)} results -> {out / 'results_all.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "probe", "plan", "status", "collect"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default=str(HERE / "matrix_protocol.json"))
    ap.add_argument("--phases", default=None, help="comma-separated phase names")
    ap.add_argument("--only", default=None, help="regex on run ids")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--memory-kernels", default="auto", choices=["auto", "reference"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--ckpt-minutes", type=float, default=5.0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--eval-tokens", type=int, default=None,
                    help="override tokens per evaluation batch")
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args(argv)
    protocol = json.loads(Path(args.protocol).read_text())
    if args.command == "run":
        cmd_run(args, protocol)
    elif args.command == "status":
        cmd_status(args, protocol)
    elif args.command == "collect":
        cmd_collect(args, protocol)
    else:
        ctx = Ctx(args, protocol)
        entries = select_entries(protocol, args)
        if args.command == "probe":
            probe(ctx, entries)
        plan(ctx, entries)


if __name__ == "__main__":
    main()
