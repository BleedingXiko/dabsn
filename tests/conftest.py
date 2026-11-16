import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_float32_matmul_precision():
    """Keep PyTorch's process-wide FP32 matmul policy per test.

    ``dabsn.pretrain`` and the CLI set ``high`` (TF32) for the whole process, as
    a training run should; without this, FP32 parity tests that run later in the
    same session silently compare under TF32.
    """
    previous = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(previous)
