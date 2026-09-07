from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from tools.build_sprint_gate import build_snapshot
from tools.sprint_gate import overall_verdict, validate_snapshot


def test_public_feature_docs_do_not_expose_private_sprint_workflow():
    root = Path(__file__).resolve().parents[1]
    for relative in ("README.md", "docs/online-temporal-credit-and-prepared-memory.md"):
        content = (root / relative).read_text().lower()
        for private_marker in ("kaggle", "sprint", "superpowers", "ensurepip", "evidence ledger"):
            assert private_marker not in content, (relative, private_marker)


def test_release_verdict_never_promotes_skips_or_failed_performance():
    assert overall_verdict([{"name": "tests", "returncode": 0}], gpu=True, required_reads="PASS") == "PASS"
    assert overall_verdict([{"name": "tests", "returncode": 0}], gpu=True, required_reads="UNVERIFIED_REQUIRED_SHAPES") == "INCOMPLETE"
    assert overall_verdict([{"name": "tests", "returncode": 0}], gpu=False, required_reads="PASS") == "INCOMPLETE"
    assert overall_verdict([{"name": "tests", "returncode": 1}], gpu=True, required_reads="PASS") == "FAIL"
    assert overall_verdict([{"name": "tests", "returncode": 0}], gpu=True, required_reads="FAIL_PERFORMANCE") == "FAIL"
    assert overall_verdict([], gpu=True, required_reads="PASS") == "INCOMPLETE"


def test_framework_verdict_does_not_invent_a_required_bank_capacity():
    rows = [{"name": "framework-tests", "returncode": 0},
            {"name": "large-read-diagnostic", "returncode": 2, "diagnostic": True}]
    assert overall_verdict(rows, gpu=True) == "PASS"
    assert overall_verdict(rows, gpu=False) == "INCOMPLETE"
    # An actual correctness failure remains blocking, including in a diagnostic.
    rows[-1]["failed"] = True
    assert overall_verdict(rows, gpu=True) == "FAIL"


def test_embedded_snapshot_has_exact_framework_sources_and_no_research_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = build_snapshot(root)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        assert "src/dabsn/temporal_credit.py" in names
        assert "src/dabsn/kernels/cpu_runtime.cpp" in names
        assert "tools/sprint_gate.py" in names
        assert "tests/fixtures/dabsn_external_fixture/pyproject.toml" in names
        assert not any(name.endswith((".ipynb", ".safetensors", ".pt")) for name in names)
        assert not any("5x768" in name.lower() for name in names)
        archive.extractall(tmp_path)
    manifest = validate_snapshot(tmp_path)
    for relative, digest in manifest["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
    (tmp_path / "src/dabsn/temporal_credit.py").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        validate_snapshot(tmp_path)


def test_snapshot_validation_refuses_paths_outside_root(tmp_path):
    (tmp_path / "SPRINT_MANIFEST.json").write_text(json.dumps({"files": {"../outside": "bad"}}))
    with pytest.raises(ValueError, match="path"):
        validate_snapshot(tmp_path)


def test_kaggle_bootstrap_uses_current_python_without_ensurepip(tmp_path):
    import os
    import subprocess
    import sys

    # Reproduce Kaggle's unavailable ensurepip while retaining a usable Python
    # package environment. The real bootstrap must still return a usable child.
    program = r'''
import importlib.abc, sys, pathlib, subprocess, os
class NoEnsurepip(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("venv", "ensurepip"):
            raise RuntimeError("Kaggle has no working ensurepip")
sys.meta_path.insert(0, NoEnsurepip())
from tools.kaggle_sprint_bootstrap import prepare_environment
os.environ["PATH"] = "/nonexistent"
python, env = prepare_environment(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), install_missing=False)
assert python == sys.executable
assert not (pathlib.Path(sys.argv[2]) / "env").exists()
subprocess.run([python, "-c", "import torch, pytest, build, safetensors; print(torch.__version__)"], env=env, check=True)
subprocess.run(["ninja", "--version"], env=env, check=True)
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", program, str(root), str(tmp_path)],
                            cwd=root, env=dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
