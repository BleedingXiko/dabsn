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

import pytest
import torch

from dabsn.read import DABSNRead


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
