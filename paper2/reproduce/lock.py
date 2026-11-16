"""The framework lock: SHA-256 of every source file of the installed DABSN package.

Standard library only. ``python lock.py PACKAGE_DIR`` writes ../framework.lock.json
from a DABSN source package directory (for example ../../src/dabsn).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SUFFIXES = (".py", ".cpp")


def files(package: Path) -> dict[str, str]:
    result = {}
    for path in sorted(Path(package).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in SUFFIXES:
            result[path.relative_to(package).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def tree_digest(listing: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(listing, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


if __name__ == "__main__":
    package = Path(sys.argv[1]).resolve()
    listing = files(package)
    if "temporal_credit.py" not in listing:
        raise SystemExit(f"{package} is not a DABSN package directory")
    target = Path(__file__).resolve().parents[1] / "framework.lock.json"
    target.write_text(json.dumps({"schema": 1, "framework": "dabsn", "tree": "DABSN 1.0 working tree",
                                  "source_sha256": listing, "tree_sha256": tree_digest(listing)},
                                 indent=2, sort_keys=True) + "\n")
    print(f"wrote {target}: {len(listing)} files, tree {tree_digest(listing)[:16]}")
