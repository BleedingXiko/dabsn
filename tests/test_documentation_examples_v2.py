from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _run_example(name: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout


def test_minimal_training_example_executes_a_real_update():
    output = _run_example("minimal_train.py")
    assert "loss" in output and "parameters" in output


def test_custom_adapter_example_executes_a_real_update():
    output = _run_example("custom_adapter.py")
    assert "output_shape" in output and "loss" in output and "parameters" in output


def test_capacity_example_executes_a_real_update():
    output = _run_example("capacity_train.py")
    assert "loss" in output and "parameters" in output


def test_field2d_local_adapter_executes_the_documented_shape():
    namespace = runpy.run_path(str(ROOT / "examples" / "field2d_local_adapter.py"))
    adapter = namespace["Field2DLocalInputAdapter"](torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]))
    output = adapter(torch.randn(2, 2, 2, 3))
    assert output.shape == (2, 4, 2, 3)
