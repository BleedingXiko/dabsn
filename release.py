#!/usr/bin/env python3
"""Non-interactive DABSN build and optional PyPI upload helper."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INIT_FILE = ROOT / "src" / "dabsn" / "__init__.py"
CITATION_FILE = ROOT / "CITATION.cff"
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def read_version() -> str:
    match = re.search(
        r'^__version__ = ["\']([^"\']+)["\']$',
        INIT_FILE.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"Could not find __version__ in {INIT_FILE}")
    return match.group(1)


def write_version(version: str) -> None:
    init_text = INIT_FILE.read_text(encoding="utf-8")
    INIT_FILE.write_text(
        re.sub(
            r'^__version__ = ["\'][^"\']+["\']$',
            f'__version__ = "{version}"',
            init_text,
            flags=re.MULTILINE,
        ),
        encoding="utf-8",
    )
    citation = CITATION_FILE.read_text(encoding="utf-8")
    citation = re.sub(r"^version: .*$", f"version: {version}", citation, flags=re.MULTILINE)
    citation = re.sub(
        r"^date-released: .*$",
        f"date-released: {dt.date.today().isoformat()}",
        citation,
        flags=re.MULTILINE,
    )
    CITATION_FILE.write_text(citation, encoding="utf-8")


def bump_version(version: str, bump: str) -> str:
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise RuntimeError(f"Cannot bump non-semver version {version!r}")
    major, minor, patch = map(int, parts)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def clean_build_outputs() -> None:
    shutil.rmtree(DIST_DIR, ignore_errors=True)
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    for egg_info in (ROOT / "src").glob("*.egg-info"):
        shutil.rmtree(egg_info, ignore_errors=True)


def distributions() -> list[Path]:
    files = sorted((*DIST_DIR.glob("*.whl"), *DIST_DIR.glob("*.tar.gz")))
    if len(files) != 2:
        raise RuntimeError(f"Expected one wheel and one sdist, found: {files}")
    return files


def write_checksums(files: list[Path]) -> Path:
    output = DIST_DIR / "SHA256SUMS.txt"
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    version = parser.add_mutually_exclusive_group(required=True)
    version.add_argument(
        "--release",
        help="Version bump (patch/minor/major) or an explicit x.y.z version.",
    )
    version.add_argument(
        "--keep-version",
        action="store_true",
        help="Build the current version without changing metadata.",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Regenerate the figures and rebuild paper1/main.pdf before packaging.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload after all local gates pass. The default is build-only.",
    )
    parser.add_argument("--testpypi", action="store_true")
    parser.add_argument("--repository", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.testpypi or args.repository or args.skip_existing) and not args.upload:
        raise SystemExit("--testpypi, --repository, and --skip-existing require --upload")

    current = read_version()
    target = current
    if args.release:
        target = bump_version(current, args.release) if args.release in {"major", "minor", "patch"} else args.release
        if not re.fullmatch(r"\d+\.\d+\.\d+", target):
            raise SystemExit("--release must be patch/minor/major or an explicit x.y.z version")
    if target != current:
        print(f"Updating version: {current} -> {target}")
        write_version(target)
    else:
        print(f"Keeping version: {current}")

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "build",
            "check-wheel-contents",
            "pytest",
            "setuptools",
            "twine",
            "wheel",
        ]
    )
    # A reused release environment may contain an older wheel with the same
    # version. Install the current checkout explicitly before running tests so
    # a green result always describes the source about to be packaged.
    run([sys.executable, "-m", "pip", "install", "--editable", "."])
    if not args.skip_tests:
        run([sys.executable, "-m", "pytest"])
    if args.paper:
        run([sys.executable, "-m", "pip", "install", "--upgrade", ".[paper]"])
        run([sys.executable, "generate_figures.py"], cwd=ROOT / "paper1")
        run(["make"], cwd=ROOT / "paper1")

    clean_build_outputs()
    run([sys.executable, "-m", "build"])
    files = distributions()
    run([sys.executable, "-m", "twine", "check", *map(str, files)])
    wheel = next(path for path in files if path.suffix == ".whl")
    checker = shutil.which("check-wheel-contents")
    if checker is None:
        candidate = Path(sys.executable).with_name("check-wheel-contents")
        windows_candidate = candidate.with_suffix(".exe")
        checker = str(windows_candidate if windows_candidate.exists() else candidate)
    run([checker, str(wheel)])
    checksums = write_checksums(files)
    print(f"\nChecksums: {checksums}")

    if not args.upload:
        print("DABSN RELEASE BUILD: PASS (nothing uploaded)")
        return 0
    if not (ROOT / "LICENSE").is_file():
        raise SystemExit(
            "Upload refused: choose the public license and add LICENSE before publishing."
        )
    command = [sys.executable, "-m", "twine", "upload"]
    if args.testpypi:
        command.extend(["--repository", "testpypi"])
    elif args.repository:
        command.extend(["--repository", args.repository])
    if args.skip_existing:
        command.append("--skip-existing")
    command.extend(map(str, files))
    run(command)
    print(f"DABSN {read_version()} UPLOAD: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
