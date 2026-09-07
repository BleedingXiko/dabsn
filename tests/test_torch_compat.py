import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import Tensor

from dabsn._torch_compat import register_custom_op_autocast


def test_import_succeeds_without_native_register_autocast():
    root = Path(__file__).resolve().parents[1]
    source = root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "delattr(torch.library, 'register_autocast'); "
                "import dabsn; "
                "assert dabsn.__version__; "
                "assert callable(dabsn.DABSNCore); "
                "print('DABSN import succeeded')"
            ),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DABSN import succeeded" in result.stdout


def test_backported_autocast_rule_casts_and_redispatches(monkeypatch):
    monkeypatch.delattr(torch.library, "register_autocast", raising=False)

    @torch.library.custom_op("dabsn_test::torch26_autocast", mutates_args=())
    def torch26_autocast(value: Tensor) -> Tensor:
        return value.clone()

    register_custom_op_autocast(torch26_autocast, "cpu", torch.bfloat16)

    value = torch.randn(8, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = torch26_autocast(value)
    assert output.dtype == torch.bfloat16
