"""The capture-safe (full front-packed bank) read must equal the compact read.

CUDA-graph capture forbids the host ``.item()`` the compact read uses to size
the admitted bank.  The capture-safe path keeps the bank full width and lets the
kernel bound work by an on-device count, so it must produce identical results --
the admitted *set* is unchanged, only the buffer width differs.  These CPU tests
pin that equivalence (forward and gradients) so the CUDA path can rely on it.
"""

import copy

import pytest
import torch

from dabsn.read import DABSNRead


def _inputs(batch, steps, hidden, *, seed):
    torch.manual_seed(seed)
    return dict(
        y=torch.randn(batch, steps, hidden),
        budget=torch.randn(batch, steps, hidden),
        expression=torch.randn(batch, steps, hidden, requires_grad=True),
        write=torch.randn(batch, steps, hidden, requires_grad=True),
        novelty=torch.rand(batch, steps, hidden),
        plasticity=torch.rand(batch, steps, hidden),
        energy=torch.rand(batch, steps, hidden),
        saturation=torch.rand(batch, steps, hidden),
    )


def _clone_inputs(inputs):
    return {
        key: (value.detach().clone().requires_grad_(value.requires_grad) if torch.is_tensor(value) else value)
        for key, value in inputs.items()
    }


@pytest.mark.parametrize("geometry", ["seq", "field", "hybrid"])
@pytest.mark.parametrize("admit_window", [4.0, -6.0])
def test_full_bank_read_matches_compact(geometry, admit_window):
    # admit_window=-6.0 makes admission genuinely sparse (n_max << seq_len), the
    # case where full-width vs compact banks actually differ in shape.
    hidden = 8
    compact = DABSNRead(hidden, geometry)
    with torch.no_grad():
        compact.admit_window.fill_(admit_window)
    full = copy.deepcopy(compact)
    full._capture_safe_bank = True

    a = _inputs(3, 16, hidden, seed=7)
    b = _clone_inputs(a)
    out_compact = compact(**a)
    out_full = full(**b)
    torch.testing.assert_close(out_full, out_compact, rtol=1e-5, atol=1e-6)

    out_compact.square().mean().backward()
    out_full.square().mean().backward()
    torch.testing.assert_close(b["expression"].grad, a["expression"].grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(b["write"].grad, a["write"].grad, rtol=1e-5, atol=1e-6)
    for (_, pc), (_, pf) in zip(compact.named_parameters(), full.named_parameters()):
        if pc.grad is not None:
            torch.testing.assert_close(pf.grad, pc.grad, rtol=1e-5, atol=1e-6)


def test_capture_safe_bank_removes_host_item_dependency():
    # With the flag on, the reported bank width is the full sequence length --
    # a static value, i.e. no data-dependent host sync sizing the bank.
    read = DABSNRead(8, "seq")
    read._capture_safe_bank = True
    inputs = _inputs(2, 12, 8, seed=1)
    read(**inputs)
    assert read.last_n_max == 12
