"""Query-time chunking of the admitted read is exact.

Long-context training stays on tensor cores by TILING the read along the
query-time axis (each tile's score tensor fits under the dense-BMM cap). That is
only valid if query rows are independent given the bank -- i.e. reading query
positions in chunks equals reading them all at once. This CPU test pins that
property on the reference read (which runs anywhere), so the GPU dense-BMM
tiling in the dispatch rests on a proven invariant. The GPU-gated companion
(`tests/test_batched_runtime.py`) certifies the actual `dense_bmm` offset math.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest
import torch

from dabsn.read import DABSNRead


def test_gather_before_normalize_is_bit_identical():
    # Phase 2c liveness reorder: the read normalizes keys AFTER the admitted
    # gather. F.normalize is row-wise and the gather selects whole rows, so
    # normalize(gather(write)) must equal gather(normalize(write)) bit-for-bit,
    # in both the forward value and the write gradient -- otherwise the memory
    # win would silently change the math.
    torch.manual_seed(0)
    B, T, H, n = 2, 7, 5, 4
    write_a = torch.randn(B, T, H, dtype=torch.float64, requires_grad=True)
    write_b = write_a.detach().clone().requires_grad_(True)
    # A fixed admitted index set (any per-row selection of positions).
    idx = torch.stack([torch.randperm(T)[:n] for _ in range(B)])
    gather_hidden = idx.unsqueeze(-1).expand(-1, -1, H)

    # new order: gather then normalize
    keys_new = torch.nn.functional.normalize(
        torch.gather(write_a, 1, gather_hidden), dim=-1
    )
    # old order: normalize then gather
    keys_old = torch.gather(
        torch.nn.functional.normalize(write_b, dim=-1), 1, gather_hidden
    )
    assert torch.equal(keys_new, keys_old)

    grad = torch.randn_like(keys_new)
    keys_new.backward(grad)
    keys_old.backward(grad)
    assert torch.equal(write_a.grad, write_b.grad)


def _ref(read, query, mk, wm, nwm, rc, mc, kbias, adm, scale, allow, ind):
    return read._three_way_read(
        query, mk, wm, nwm, rc, mc, kbias, adm, scale,
        allow, ind, allow.any(dim=-1), ind.any(dim=-1),
    )


@pytest.mark.parametrize("geometry", ["seq", "field"])
def test_reference_read_is_query_chunkable(geometry):
    torch.manual_seed(3)
    B, T, N, H = 2, 12, 5, 7
    read = DABSNRead(H, geometry)
    query = torch.randn(B, T, H)
    mk, wm, nwm = (torch.randn(B, N, H) for _ in range(3))
    rc = torch.randn(B, T, 4)
    mc = torch.randn(B, N, 4)
    kbias = torch.randn(B, N)
    adm = torch.randn(B, N)
    scale = torch.tensor(1.2)
    bank_idx = torch.tensor([[0, 1, 3, 4, 5], [0, 2, 3, 4, 5]])
    bank_valid = torch.tensor(
        [[True, True, False, True, True], [True, True, True, False, True]]
    )
    qpos = torch.arange(T).view(1, T, 1)
    allow = bank_valid.unsqueeze(1) & (bank_idx.unsqueeze(1) <= qpos)
    ind = bank_valid.unsqueeze(1) & (bank_idx.unsqueeze(1) < qpos)

    full = _ref(read, query, mk, wm, nwm, rc, mc, kbias, adm, scale, allow, ind)
    for chunk in (1, 3, 5, T):
        pieces = []
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            pieces.append(_ref(
                read, query[:, t0:t1], mk, wm, nwm, rc[:, t0:t1], mc,
                kbias, adm, scale, allow[:, t0:t1], ind[:, t0:t1],
            ))
        chunked = torch.cat(pieces, dim=1)
        # Query rows are independent given the bank, so tiling is mathematically
        # exact; the only gap is float rounding from differing reduction shapes.
        torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-6)


def test_inference_read_respects_the_score_budget(monkeypatch):
    """Forward-only reads must tile queries exactly like the training path.

    The density dispatcher chooses dense-vs-flash on how FULL the bank is; it has
    no notion of how big the resulting [B,T,N] score tensor would be. Without a
    budget here a large batch or long context OOMs in the forward pass on
    multi-GiB score allocations while the training path -- which tiles -- runs the
    same shapes fine. Query rows are independent given the bank, so tiling is
    bit-identical and the budget must apply in both grad modes.
    """
    from dabsn.kernels import triton as kt

    src = inspect.getsource(kt.cuda_three_way_read)
    grad_split = src.index("if torch.is_grad_enabled():")
    infer_half = src[grad_split:]
    # The budget helper must be defined BEFORE the grad split (shared by both),
    # and the forward-only half must actually consult it.
    assert "_query_chunk_t" in src[:grad_split], (
        "the score budget must be computed before the grad-mode split so the "
        "forward-only path can use it too"
    )
    assert "_query_chunk_t" in infer_half and "_chunked_dense" in infer_half, (
        "the forward-only read path must tile queries under the score budget; "
        "without it a large-batch forward allocates the full [B,T,N] scores"
    )


def test_read_weight_normalization_allocates_no_extra_full_tensors():
    """The [B,T,N] weight tensor is the largest thing in the read.

    Building it as where(elig, nan_to_num(w), zeros_like(w)) allocated three more
    tensors of that size on top of the masked-fill copy and the softmax output --
    five live [B,T,N] buffers per read, three reads per call, which is what made
    a batch-256 forward OOM.

    The economy has to be bought without writing the softmax output, because
    autograd saves that exact buffer. So the ineligible rows are zeroed on the
    [B,T,H] read vector instead, and masking uses the dtype's finite minimum so
    no NaN is created that would need a repair pass. That keeps two [B,T,N]
    buffers AND keeps backward legal; this test pins both halves.
    """
    # Read the source from disk: triton_runtime imports triton at module scope,
    # so it cannot be imported on a CPU-only box -- but the property under test
    # is textual and must hold everywhere the file ships.
    import dabsn.kernels as _k

    path = pathlib.Path(_k.__file__).with_name("triton_runtime.py")
    code = "\n".join(
        line for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    # No third/fourth/fifth [B,T,N] buffer for the eligibility fixup.
    assert "torch.where(elig" not in code
    assert "torch.zeros_like(weights)" not in code
    assert "torch.nan_to_num(w), torch.zeros_like(w)" not in code
    # The zeroing lands on the read vector, which bmm does not save.
    assert ".float().mul_(elig)" in code
    assert "torch.bmm(weights, values).mul_(elig.unsqueeze(-1))" in code
    # Finite-minimum masking, so there is no NaN to repair after the softmax.
    assert code.count("torch.finfo(scores.dtype).min") >= 1
    assert "neg = torch.finfo(scores.dtype).min" in code
