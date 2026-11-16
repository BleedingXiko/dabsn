from __future__ import annotations

import hashlib
import base64
import io
import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from tools.build_sprint_gate import CELL_HEADER, CELL_FOOTER, build_snapshot
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
    assert overall_verdict([{"name": "capacity", "returncode": 2}], gpu=True,
                           capacity="INCOMPLETE") == "INCOMPLETE"
    assert overall_verdict([{"name": "capacity", "returncode": 1}], gpu=True,
                           capacity="FAIL") == "FAIL"


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
        assert "src/dabsn/kernels/capacity_cpu.cpp" in names
        assert "src/dabsn/kernels/capacity.py" in names
        assert "tools/sprint_gate.py" in names
        assert "tools/capacity_gate.py" in names
        assert "tools/build_kaggle_dual_t4_gate.py" in names
        assert "tools/kaggle_dual_t4_gate_source.py" in names
        assert "tools/tests/test_capacity_gate.py" in names
        assert "tests/test_capacity_memory.py" in names
        assert "docs/capacity.md" in names
        assert "tests/fixtures/dabsn_external_fixture/pyproject.toml" in names
        assert not any(name.endswith((".ipynb", ".safetensors", ".pt")) for name in names)
        assert not any("5x768" in name.lower() for name in names)
        archive.extractall(tmp_path)
    manifest = validate_snapshot(tmp_path)
    for relative, digest in manifest["files"].items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == digest
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


@pytest.mark.parametrize("kaggle_working_exists", [False, True], ids=["local", "kaggle"])
def test_one_cell_retains_json_and_zip_when_runner_fails_early(
    tmp_path, monkeypatch, kaggle_working_exists
):
    payload_file = io.BytesIO()
    with zipfile.ZipFile(payload_file, "w") as archive:
        archive.writestr("tools/kaggle_sprint_bootstrap.py",
                         "import os, sys\ndef prepare_environment(source, workspace):\n"
                         "    return sys.executable, os.environ.copy()\n")
    payload = payload_file.getvalue()

    class FailedJob:
        def wait(self):
            return 17

    current_directory = tmp_path / "cwd"
    current_directory.mkdir()
    kaggle_directory = tmp_path / "kaggle-working"
    kaggle_directory.mkdir()
    real_is_dir = Path.is_dir
    real_mkdtemp = tempfile.mkdtemp

    def controlled_is_dir(path):
        if path == Path("/kaggle/working"):
            return kaggle_working_exists
        return real_is_dir(path)

    def isolated_mkdtemp(*, prefix, dir):
        # Keep the cell's real location selection, but never write into the
        # host's /kaggle/working when these tests themselves run on Kaggle.
        if dir == Path("/kaggle/working"):
            dir = kaggle_directory
        return real_mkdtemp(prefix=prefix, dir=dir)

    monkeypatch.chdir(current_directory)
    monkeypatch.setattr(Path, "is_dir", controlled_is_dir)
    monkeypatch.setattr(tempfile, "mkdtemp", isolated_mkdtemp)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: FailedJob())
    namespace = {"SNAPSHOT_B64": base64.b64encode(payload).decode(),
                 "SNAPSHOT_SHA256": hashlib.sha256(payload).hexdigest()}
    with pytest.raises(RuntimeError, match="exited 17"):
        exec(compile(CELL_HEADER + CELL_FOOTER, "sprint-cell-footer", "exec"), namespace)
    expected_directory = kaggle_directory if kaggle_working_exists else current_directory
    workspaces = list(expected_directory.glob("dabsn-sprint-*"))
    assert len(workspaces) == 1
    report = json.loads((workspaces[0] / "results/SPRINT_RESULT.json").read_text())
    assert report["verdict"] == "FAIL"
    assert "exited 17" in report["error"]
    with zipfile.ZipFile(workspaces[0] / "DABSN_SPRINT_RESULTS.zip") as archive:
        assert "SPRINT_RESULT.json" in archive.namelist()
