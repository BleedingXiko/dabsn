"""Run the whole paper-2 matrix, resumably. Rerun the same command to continue.

    python run.py --device cuda --results /path/to/results            # the full matrix
    python run.py --device cuda --results /tmp/smoke --smoke          # every code path, small counts
    python run.py --device cuda --results R --stages oracles,delayed  # chosen stages only
    python run.py --results R --status                                # progress only

Each job runs in its own process (job.py). A job that is interrupted resumes from
its newest checkpoint; a passed job is never rerun; every attempt is kept. The
device and compute precision are fixed on the first run of a results directory.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matrix  # noqa: E402  (standard library only)

OUT_OF_MEMORY_TOLERATED = {"scaling", "resource"}  # an observed resource limit is a result there


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def latest_attempt(directory):
    attempts = sorted(Path(directory).glob("attempt-*.json"))
    return (attempts[-1], read(attempts[-1])) if attempts else (None, None)


def stage_jobs(name, m, selections):
    _, module_name, key, builder, needs, _ = matrix.stage(name)
    module = importlib.import_module(f"studies.{module_name}")
    function = getattr(module, builder)
    if needs and needs not in selections:
        return module, None
    if name == "ablation":
        return module, function(m["ablation"], selections[needs]["selected_rates"])
    if needs:
        return module, function(m[key], selections[needs]["selected_rates"])
    return module, function(m[key])


def job_directory(results, name, module, job):
    return results / matrix.record_study(name) / module.identifier(job)


def select(name, m, results, selections):
    """Make the selection this stage produces, once every job it reads has passed."""
    _, module_name, key, _, _, produces = matrix.stage(name)
    if not produces or produces in selections:
        return
    module, jobs = stage_jobs(name, m, selections)
    records = {}; inputs = {}
    for job in jobs:
        path, record = latest_attempt(job_directory(results, name, module, job))
        if record is None or record["status"] != "passed":
            return
        records[module.identifier(job)] = record
        inputs[module.identifier(job)] = path.name
    made = {**module.select_rates(m[key], records), "inputs": inputs}
    # Several runners may share one results directory: merge with the file as it is now,
    # so a selection another runner wrote is never overwritten by a stale in-memory copy.
    path = results / "selections.json"
    on_disk = read(path) if path.exists() else {}
    on_disk.setdefault(produces, made)
    write(path, on_disk)
    selections.update(on_disk)
    print(f"selected {produces}: {selections[produces]['selected_rates']}", flush=True)


def resolve_run(results, device, smoke, precision):
    """Fix device and compute precision for this results directory."""
    run_path = results / "run.json"
    env = {**os.environ, "DABSN_PAPER2_PRECISION": precision}
    probe = subprocess.run([sys.executable, "-c",
                            "import sys; sys.path.insert(0, %r); import harness; "
                            "print(harness.selected_precision(%r)); print(harness.FRAMEWORK_SHA256)"
                            % (str(HERE), device)], env=env, capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(probe.stderr.strip() or "Could not resolve the compute precision")
    resolved, framework = probe.stdout.split()[-2:]
    current = {"device": device, "precision": resolved, "smoke": smoke, "framework_tree_sha256": framework}
    if run_path.exists():
        pinned = read(run_path)
        if any(pinned[key] != current[key] for key in ("device", "smoke", "framework_tree_sha256")):
            raise SystemExit(f"This results directory was started with {pinned}; now {current}")
        if pinned["precision"] != resolved:
            raise SystemExit(f"This results directory uses {pinned['precision']} compute; this machine resolves "
                             f"{resolved}. Use --precision {pinned['precision']} or a GPU that supports it.")
        return pinned
    results.mkdir(parents=True, exist_ok=True)
    write(run_path, {**current, "started_unix": time.time()})
    return current


def status(results, m, selections):
    rows = []
    for name, *_ in matrix.STAGES:
        module, jobs = stage_jobs(name, m, selections)
        if jobs is None:
            rows.append((name, "waiting for selection", "", ""))
            continue
        counts = {}
        for job in jobs:
            _, record = latest_attempt(job_directory(results, name, module, job))
            key = "not run" if record is None else record["status"]
            counts[key] = counts.get(key, 0) + 1
        rows.append((name, len(jobs), counts.get("passed", 0),
                     ", ".join(f"{k} {v}" for k, v in sorted(counts.items()) if k != "passed")))
    width = max(len(r[0]) for r in rows)
    for name, total, passed, other in rows:
        print(f"{name:<{width}}  {passed}/{total} passed  {other}")


def archive(results):
    target = results.with_name(results.name + ".zip")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(results.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                z.write(path, str(path.relative_to(results.parent)))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stages", help="comma-separated stage names (default: all, in order)")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                        help="fp32 (the paper). bf16 is not a matched policy in this framework: the "
                             "online-credit forward always computes in FP32, so BPTT/TBPTT would run a "
                             "different forward precision; the preflight refuses it.")
    parser.add_argument("--max-hours", type=float, help="stop cleanly at a saved boundary after this long")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--archive", action="store_true", help="zip the results directory")
    args = parser.parse_args()
    results = args.results.resolve()
    smoke = args.smoke or (results / "run.json").exists() and read(results / "run.json").get("smoke", False)
    m = matrix.smoke_protocol() if smoke else matrix.MATRIX
    selections = read(results / "selections.json") if (results / "selections.json").exists() else {}
    if args.status:
        status(results, m, selections); return
    if args.archive:
        print(archive(results)); return
    if args.device is None:
        parser.error("--device is required to run")
    pinned = resolve_run(results, args.device, smoke, args.precision)
    names = [s[0] for s in matrix.STAGES]
    chosen = names if not args.stages else args.stages.split(",")
    unknown = sorted(set(chosen) - set(names))
    if unknown:
        parser.error(f"unknown stages {unknown}; stages are {names}")
    deadline = None if args.max_hours is None else time.monotonic() + 3600 * args.max_hours
    env = {**os.environ, "DABSN_PAPER2_PRECISION": pinned["precision"], "PYTHONDONTWRITEBYTECODE": "1"}
    process = None

    def forward(signum, frame):
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)  # the job saves its boundary and exits with 3
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    for name in names:
        if name not in chosen:
            continue
        module, jobs = stage_jobs(name, m, selections)
        if jobs is None:
            print(f"[{name}] waiting: needs the {matrix.stage(name)[4]} selection; run its development stage")
            continue
        for job in jobs:
            directory = job_directory(results, name, module, job)
            _, record = latest_attempt(directory)
            if record is not None and record["status"] == "passed":
                continue
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 60:
                print("time budget reached; rerun the same command to continue", flush=True)
                write(results / "manifest.json", {"stopped": "time budget", "unix": time.time()})
                return
            directory.mkdir(parents=True, exist_ok=True)
            log = directory / f"log-{len(list(directory.glob('log-*.txt'))) + 1:04d}.txt"
            command = [sys.executable, str(HERE / "job.py"), "--stage", name, "--job", json.dumps(job),
                       "--device", args.device, "--results", str(results)]
            if remaining is not None:
                command += ["--max-seconds", str(remaining - 60)]
            if smoke:
                command.append("--smoke")
            print(f"[{name}] {module.identifier(job)}", flush=True)
            with log.open("w") as stream:
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                code = process.wait()
            _, record = latest_attempt(directory)
            outcome = record["status"] if record else "no record"
            if code == 3:
                print(f"stopped at a saved boundary in {directory}; rerun to resume", flush=True)
                return
            if code != 0:
                if outcome == "out_of_memory" and matrix.record_study(name) in OUT_OF_MEMORY_TOLERATED:
                    print(f"  out of memory (recorded as a result)", flush=True)
                    continue
                print(f"  {outcome}: see {log}", flush=True)
                raise SystemExit(1)
        select(name, m, results, selections)
    write(results / "manifest.json", {"finished_unix": time.time(), "stages": chosen})
    status(results, m, selections)


if __name__ == "__main__":
    main()
