"""Collect a results directory into one file: every job's latest record, with the selections.

    python collect.py RESULTS [--out ../results_all.json.gz]

The file is gzip-compressed JSON (written with a zero timestamp, so equal records give equal
bytes); a path without .gz is written and read as plain JSON.

``expand`` writes a collected file back out as a results directory (each job's latest record,
its regime record, and the selections, run and manifest files), so the analysis and the table
export read results_all.json.gz and a live results directory the same way.

One entry per declared-or-run job directory: its stage study, identifier, number of attempts and
the latest attempt record (status, job, provenance, result). A continuity job's regime record is
included with it. Checkpoints and logs are not results and are not collected.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path


def read_collected(path: Path) -> dict:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def write_collected(data: dict, path: Path) -> None:
    text = json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".writing")
    if path.suffix == ".gz":
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as handle:
            handle.write(text.encode("utf-8"))
        temporary.write_bytes(buffer.getvalue())
    else:
        temporary.write_text(text)
    temporary.replace(path)


def collect(results: Path) -> dict:
    out = {"selections": {}, "run": None, "manifest": None, "results": []}
    for name in ("selections", "run", "manifest"):
        path = results / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    out["selections"] = out["selections"] or {}
    for study in sorted(p for p in results.iterdir() if p.is_dir()):
        for job in sorted(p for p in study.iterdir() if p.is_dir()):
            attempts = sorted(job.glob("attempt-*.json"))
            if not attempts:
                continue
            entry = {"study": study.name, "id": job.name, "attempts": len(attempts),
                     "record": json.loads(attempts[-1].read_text())}
            regime = job / "regime.json"
            if regime.exists():
                entry["regime"] = json.loads(regime.read_text())
            out["results"].append(entry)
    return out


def expand(collected: Path, directory: Path) -> Path:
    data = read_collected(collected)
    directory = Path(directory)
    for name in ("selections", "run", "manifest"):
        if data.get(name) is not None:
            (directory / f"{name}.json").write_text(json.dumps(data[name]))
    for entry in data["results"]:
        job = directory / entry["study"] / entry["id"]
        job.mkdir(parents=True, exist_ok=True)
        (job / f"attempt-{entry['attempts']:04d}.json").write_text(json.dumps(entry["record"]))
        if "regime" in entry:
            (job / "regime.json").write_text(json.dumps(entry["regime"]))
    return directory


def results_directory(path: Path, keep):
    """A results directory for PATH: the directory itself, or a collected file expanded into a
    temporary directory (KEEP holds the TemporaryDirectory so it lives as long as the caller)."""
    import tempfile
    path = Path(path)
    if path.is_dir():
        return path
    keep.append(tempfile.TemporaryDirectory())
    return expand(path, Path(keep[-1].name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results_all.json.gz")
    args = parser.parse_args()
    data = collect(args.results)
    write_collected(data, args.out)
    print(f"collected {len(data['results'])} jobs into {args.out}")


if __name__ == "__main__":
    main()
