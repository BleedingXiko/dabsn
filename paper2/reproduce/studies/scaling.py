"""Measured update time and memory for online credit, TBPTT and BPTT across width and length.

The eligibility byte count is the B-channel formula for input = state = S:
B * (4*S^2 + 60*S) scalars of 4 bytes.
"""
from __future__ import annotations

import itertools
import statistics
import time

import torch

import models

NEEDS_NATIVE = True


def jobs(protocol):
    return [{"width":width,"length":length,"seed":seed,"method":method}
            for width,length,seed,method in itertools.product(
                protocol["widths"],protocol["lengths"],protocol["seeds"],protocol["methods"])]


def identifier(job):
    return f"w{job['width']}-t{job['length']}-s{job['seed']}-{job['method']}"


def run(job, ctx):
    device=ctx.device
    runtime=ctx.runtime
    cfg=ctx.protocol; s=job["width"]; length=job["length"]
    model=models.Model(s,s,4,job["seed"],device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg["lr"])
    g=torch.Generator().manual_seed(8000+job["seed"])
    x=torch.randn(cfg["batch"],length,s,generator=g).to(device)*.1
    y=torch.arange(cfg["batch"],device=device)%4
    chunk=length if job["method"]=="bptt" else cfg["chunk"]
    times=[]; peaks=[]; reserves=[]
    first=None
    for i in range(cfg["warmup"]+cfg["repetitions"]):
        opt.zero_grad(set_to_none=True)
        runtime.synchronize()
        runtime.reset_peak()
        started=time.perf_counter()
        models.update(model,opt,x,y,job["method"],chunk,1.)
        runtime.synchronize()
        elapsed=time.perf_counter()-started
        if first is None: first=elapsed
        if i>=cfg["warmup"]:
            allocated,reserved=runtime.peaks()
            times.append(elapsed); peaks.append(allocated)
            reserves.append(reserved)
    n=2*s*s+2*s*s+60*s
    return {"config":cfg,"seconds":times,"median_seconds":statistics.median(times),
            "tokens_per_second":cfg["batch"]*length/statistics.median(times),
            "peak_allocated_bytes":peaks,"peak_reserved_bytes":reserves,
            "first_update_seconds":first,"eligibility_bytes_formula":cfg["batch"]*n*4 if job["method"]=="online" else 0,
            "backend":getattr(model.core,"_last_core_backend",None)}
