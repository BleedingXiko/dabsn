import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="CUDA and Triton are required for the native CUDA release gate",
)
def test_required_cuda_runtime_release_gate(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "release_gate.py"),
            "--backend",
            "cuda",
            "--report",
            str(tmp_path / "cuda-report.json"),
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DABSN RELEASE GATE: PASS" in result.stdout
