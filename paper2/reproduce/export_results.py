"""Export the matrix's passed records to the paper's machine-readable result tables.

    python3 paper2/reproduce/export_results.py RESULTS --out DIRECTORY

Standard library only. For every study directory in RESULTS, the newest attempt of
every job that passed is read and written as:

- <study>.csv           one row per job: the job's fields and every scalar of its result;
- <study>__<series>.csv one row per element of every list of records inside a result
                        (learning curves, per-condition rows, per-delay rows ...), with
                        the job's fields and the element's index.

Lists of numbers up to LIST_LIMIT long are written in one cell, separated by ';'.
Longer numeric arrays (raw per-history vectors) stay in the raw JSON records only.
Failed or superseded attempts are never exported; every attempt stays in the raw
records. The run identity (device, precision, framework tree) is written to run.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LIST_LIMIT = 64
# Arms whose records exist but are not part of the paper: the stream studies' superseded
# normalized arm (the native action is the stream-normalized stage).
SUPERSEDED_ARMS = {"continuous": {"carry_normalized"}, "sustained": {"carry_normalized"}}
SKIP = {"checkpoint_sha256", "projection_precision", "config", "configuration", "regime"}


def latest_passed(directory: Path):
    attempts = sorted(directory.glob("attempt-*.json"))
    if not attempts:
        return None
    record = json.loads(attempts[-1].read_text())
    return record if record.get("status") == "passed" else None


def scalar(value):
    return value is None or isinstance(value, (bool, int, float, str))


def flatten(value, prefix, out, series, path):
    """Scalars into out; lists of dicts into series[path]; short numeric lists joined."""
    if scalar(value):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in SKIP:
                continue
            flatten(item, f"{prefix}.{key}" if prefix else str(key), out, series, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            series.setdefault(path, []).append(value)
        elif all(scalar(item) for item in value) and len(value) <= LIST_LIMIT:
            out[prefix] = ";".join("" if item is None else str(item) for item in value)


def job_columns(job):
    flat = {}
    flatten(job, "", flat, {}, "")
    return {f"job.{k}": v for k, v in flat.items()}


def write(path: Path, rows: list[dict]):
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export(results: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    written = []
    run = json.loads((results / "run.json").read_text())
    write(out / "run.csv", [{k: (json.dumps(v) if not scalar(v) else v) for k, v in run.items()}])
    written.append("run.csv")
    for study in sorted(p for p in results.iterdir() if p.is_dir()):
        rows = []; series = {}
        for job_dir in sorted(p for p in study.iterdir() if p.is_dir()):
            record = latest_passed(job_dir)
            if record is None or (study.name in SUPERSEDED_ARMS
                                  and record["job"].get("method") in SUPERSEDED_ARMS[study.name]):
                continue
            identity = {"job.identifier": job_dir.name, **job_columns(record["job"]),
                        "stage": record.get("stage"), "reproduction_source_sha256": record.get("reproduction_source_sha256")}
            flat = {}; nested = {}
            flatten(record.get("result", {}), "", flat, nested, "")
            rows.append({**identity, **flat})
            for name, lists in nested.items():
                for items in lists:
                    for index, item in enumerate(items):
                        element = {}
                        flatten(item, "", element, {}, "")
                        series.setdefault(name, []).append({**identity, "index": index, **element})
        if rows:
            write(out / f"{study.name}.csv", rows); written.append(f"{study.name}.csv")
        for name, items in sorted(series.items()):
            filename = f"{study.name}__{name.replace('.', '_')}.csv"
            write(out / filename, items); written.append(filename)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results_all.json.gz or a results directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from collect import results_directory
    keep = []
    for name in export(results_directory(args.results, keep), args.out):
        print(name)


if __name__ == "__main__":
    main()
