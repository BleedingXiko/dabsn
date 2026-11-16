"""Run one job of the matrix in this process. ``run.py`` calls this once per job.

Exit status: 0 passed, 3 stopped at a saved boundary (rerun resumes), 1 failed.
Every attempt is kept as attempt-NNNN.json next to the job's checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness  # noqa: E402  (verifies the framework lock before torch is imported)
import matrix  # noqa: E402


SHARED_SOURCES = ("harness.py", "lock.py", "credit.py", "models.py", "tasks.py", "reference.py", "job.py")


def study_files(module_name: str) -> list[Path]:
    """The study's own files and the studies it imports, transitively."""
    seen, pending, result = set(), [module_name], []
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        package = HERE / "studies" / name
        paths = sorted(package.glob("*.py")) if package.is_dir() else [HERE / "studies" / f"{name}.py"]
        for path in paths:
            result.append(path)
            for line in path.read_text().splitlines():
                if line.startswith("from studies import ") or line.startswith("from studies."):
                    rest = line.split("import", 1)[1] if line.startswith("from studies import") else line.split(".")[1].split()[0]
                    for item in rest.replace("(", " ").replace(")", " ").split(","):
                        token = item.split(" as ")[0].strip().split()[0] if item.strip() else ""
                        if token and token != "studies":
                            pending.append(token.split(".")[0])
    return sorted(set(result))


def source_digest(module_name: str, protocol: dict) -> str:
    """Identity of everything that can change this study's numbers."""
    checksum = hashlib.sha256()
    for path in [HERE / name for name in SHARED_SOURCES] + study_files(module_name):
        checksum.update(path.relative_to(HERE).as_posix().encode())
        checksum.update(path.read_bytes())
    checksum.update(harness.canonical(protocol).encode())
    return checksum.hexdigest()


def effective_matrix(smoke: bool) -> dict:
    return matrix.smoke_protocol() if smoke else matrix.MATRIX


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--job", required=True, help="the job as JSON")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    name, module_name, _, _, _, _ = matrix.stage(args.stage)
    module = importlib.import_module(f"studies.{module_name}")
    job = json.loads(args.job)
    m = effective_matrix(args.smoke)
    protocol = matrix.protocol_for(name, m)
    device = getattr(module, "DEVICE", args.device)
    override = getattr(module, "precision_for", lambda job: None)(job)
    if getattr(module, "FP64_REFERENCE", False):
        override = "fp32"  # no autocast is used; the study computes in float64
    if override is not None:
        os.environ["DABSN_PAPER2_PRECISION"] = override
    directory = args.results / matrix.record_study(name) / module.identifier(job)
    directory.mkdir(parents=True, exist_ok=True)
    attempts = sorted(directory.glob("attempt-*.json"))
    if attempts and json.loads(attempts[-1].read_text())["status"] == "passed":
        return 0
    resume = bool(attempts)
    attempt_path = directory / f"attempt-{len(attempts) + 1:04d}.json"
    ctx = harness.Context(matrix.record_study(name), job, protocol, device, directory, args.results,
                          resume, args.max_seconds, args.smoke, source_digest(module_name, protocol))
    ctx.matrix = m
    record = {"schema": 1, "stage": name, "study": matrix.record_study(name), "job": job,
              "device": device, "smoke": args.smoke, "status": "running",
              "precision": harness.precision_record(device), "provenance": harness.provenance(device),
              "reproduction_source_sha256": ctx.identity["reproduction_source_sha256"],
              "protocol_sha256": harness.digest_value(protocol),
              "numerics": ("float64 reference equations" if getattr(module, "FP64_REFERENCE", False)
                           else "compute policy above")}
    harness.write_record(attempt_path, record)
    harness.install_stop_handlers()
    code = 1
    try:
        record["hardware"] = ctx.runtime.configure()
        if device == "cuda" and record["hardware"]["free_memory_fraction_at_start"] < m["delayed"]["execution"][
                "minimum_free_gpu_memory_fraction"]:
            raise RuntimeError("CUDA free-memory fraction below the declared minimum")
        if getattr(module, "NEEDS_NATIVE", False):
            from dabsn.kernels import enable
            record["native_backend"] = repr(enable(device, required=True))
        record["result"] = module.run(job, ctx)
        json.dumps(record["result"], allow_nan=False)
        record["status"] = "passed"
        code = 0
    except harness.Stopped as stop:
        record["status"] = "stopped"; record["error"] = str(stop)
        code = 3
    except BaseException as exc:  # noqa: BLE001  (every outcome is recorded)
        out_of_memory = "OutOfMemoryError" in type(exc).__name__
        record["status"] = "out_of_memory" if out_of_memory else "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        code = 1
    finally:
        record["wall_seconds"] = ctx.elapsed()
        record["finished_unix"] = time.time()
        harness.write_record(attempt_path, record)
        print(json.dumps({"stage": name, "job": job, "status": record["status"],
                          "wall_seconds": round(record["wall_seconds"], 1)}), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
