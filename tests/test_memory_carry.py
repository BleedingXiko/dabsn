"""Carried memory: chunked ingest must equal one dense pass, and survive a file.

The gate is exactness. ``forward_query_bank`` had no caller anywhere in this
repo before ``dabsn.memory``, so every claim about it is pinned here.
"""

import pytest
import torch

from dabsn.memory import (
    BANK_FIELDS,
    DABSNMemory,
    embed_in_checkpoint,
    extract_from_checkpoint,
    forward_with_memory,
    ingest,
    load,
    memory_cost,
    save,
)
from dabsn.model import DABSNLayerSpec, DABSNSequenceLM

VOCAB, BATCH, TOKENS = 61, 2, 48


def build_model(geometry: str = "seq", seed: int = 0) -> DABSNSequenceLM:
    torch.manual_seed(seed)
    return DABSNSequenceLM(
        vocab=VOCAB,
        hidden_dim=16,
        depth=2,
        layers=[
            DABSNLayerSpec(hidden_dim=16, state_dim=16, read_geometry=geometry),
            DABSNLayerSpec(hidden_dim=16, state_dim=16, read_geometry=geometry),
        ],
    ).eval()


def token_ids(seed: int = 3, tokens: int = TOKENS) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (BATCH, tokens), generator=generator)


def make_selective(model: DABSNSequenceLM) -> None:
    """Tighten the admit window so the bank is sparse, not the full sequence."""
    with torch.no_grad():
        for block in model.backbone.blocks:
            block.read.admit_window.fill_(-3.0)


@pytest.mark.parametrize("chunk_size", [None, 1, 7, 16, 48])
def test_chunked_ingest_matches_dense_forward(chunk_size):
    model, ids = build_model(), token_ids()
    with torch.no_grad():
        reference = model.forward_sequence(ids)
    logits, memory = forward_with_memory(model, ids, chunk_size=chunk_size, extend=True)
    assert logits.shape == reference.shape
    assert torch.allclose(logits, reference, atol=1e-5, rtol=1e-4)
    assert memory.position == TOKENS
    assert memory.entries == TOKENS


@pytest.mark.parametrize("chunk_size", [1, 7, 16, 48])
def test_residual_mlp_width_changes_match_dense_forward(chunk_size):
    torch.manual_seed(17)
    model = DABSNSequenceLM(
        vocab=VOCAB,
        hidden_dim=16,
        depth=2,
        layers=[
            DABSNLayerSpec(hidden_dim=19, state_dim=17, read_geometry="seq"),
            DABSNLayerSpec(hidden_dim=13, state_dim=15, read_geometry="seq"),
        ],
        residual=True,
        mlp_ratio=2.0,
    ).eval()
    for block in model.backbone.blocks:
        with torch.no_grad():
            block.mlp_fc2.weight.normal_(mean=0.0, std=0.02)
    ids = token_ids()
    with torch.no_grad():
        reference = model.forward_sequence(ids)
    logits, memory = forward_with_memory(model, ids, chunk_size=chunk_size, extend=True)
    assert torch.allclose(logits, reference, atol=1e-5, rtol=1e-4)
    assert memory.position == TOKENS


def test_exact_when_admission_is_sparse():
    """The dense-bank case can hide index bugs; force a selective bank."""
    model, ids = build_model(), token_ids()
    make_selective(model)
    with torch.no_grad():
        reference = model.forward_sequence(ids)
    assert model.backbone.blocks[0].read.last_n_max < TOKENS, "bank was not sparse"
    logits, _ = forward_with_memory(model, ids, chunk_size=8, extend=True)
    assert torch.allclose(logits, reference, atol=1e-5, rtol=1e-4)


def test_split_ingest_then_query_matches_one_pass():
    """Ingest a corpus, then score a continuation against the carried bank."""
    model = build_model()
    ids = token_ids(tokens=TOKENS)
    context, query = ids[:, :32], ids[:, 32:]
    with torch.no_grad():
        reference = model.forward_sequence(ids)[:, 32:]
    memory = ingest(model, context, chunk_size=8)
    assert memory.entries == 32
    logits, unchanged = forward_with_memory(model, query, memory)
    assert torch.allclose(logits, reference, atol=1e-5, rtol=1e-4)
    assert unchanged is memory, "a non-extending query must not grow the bank"
    assert unchanged.entries == 32


def test_incremental_decode_matches_one_pass():
    """The generation path: one token per step, memory growing as it goes."""
    model = build_model()
    ids = token_ids()
    context, tail = ids[:, :32], ids[:, 32:]
    with torch.no_grad():
        reference = model.forward_sequence(ids)[:, 32:]

    memory = ingest(model, context, chunk_size=8)
    steps = []
    for index in range(tail.shape[1]):
        logits, memory = forward_with_memory(
            model, tail[:, index : index + 1], memory, extend=True
        )
        steps.append(logits)

    assert torch.allclose(torch.cat(steps, dim=1), reference, atol=1e-5, rtol=1e-4)
    assert memory.entries == TOKENS, "each decoded token must join the bank"


def test_ingest_is_resumable_in_pieces():
    model = build_model()
    ids = token_ids()
    whole = ingest(model, ids, chunk_size=16)
    piece = ingest(model, ids[:, :16], chunk_size=16)
    piece = ingest(model, ids[:, 16:32], piece, chunk_size=16)
    piece = ingest(model, ids[:, 32:], piece, chunk_size=16)
    assert piece.position == whole.position
    for left, right in zip(whole.layers, piece.layers):
        for name in BANK_FIELDS:
            assert torch.allclose(left.bank[name], right.bank[name], atol=1e-6)


def test_save_load_roundtrip_reproduces_logits(tmp_path):
    model = build_model()
    context, query = token_ids()[:, :32], token_ids(seed=9, tokens=8)
    memory = ingest(model, context)
    before, _ = forward_with_memory(model, query, memory)

    path = save(memory, tmp_path / "corpus.dmem")
    assert path.exists()
    restored = load(path, model=model)

    assert restored.position == memory.position
    assert restored.entries == memory.entries
    after, _ = forward_with_memory(model, query, restored)
    assert torch.equal(before, after)


def test_saved_memory_only_stores_scored_fields(tmp_path):
    model = build_model()
    memory = ingest(model, token_ids())
    payload = torch.load(save(memory, tmp_path / "m.dmem"), weights_only=False)
    assert set(payload["layers"][0]["bank"]) == set(BANK_FIELDS)


def test_embed_and_extract_from_checkpoint(tmp_path):
    model = build_model()
    memory = ingest(model, token_ids()[:, :24])
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "config": {"vocab": VOCAB}}, checkpoint)

    embed_in_checkpoint(memory, checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    assert "state_dict" in payload, "embedding must not disturb the checkpoint"

    restored = extract_from_checkpoint(checkpoint, model=model)
    assert restored is not None
    assert restored.entries == memory.entries
    query = token_ids(seed=11, tokens=6)
    assert torch.equal(
        forward_with_memory(model, query, memory)[0],
        forward_with_memory(model, query, restored)[0],
    )


def test_extract_returns_none_without_memory(tmp_path):
    checkpoint = tmp_path / "bare.pt"
    torch.save({"state_dict": {}}, checkpoint)
    assert extract_from_checkpoint(checkpoint) is None


def test_rejects_memory_from_another_architecture(tmp_path):
    memory = ingest(build_model(), token_ids())
    other = DABSNSequenceLM(vocab=VOCAB, hidden_dim=16, depth=3).eval()
    with pytest.raises(ValueError, match="different architecture"):
        forward_with_memory(other, token_ids(tokens=4), memory)


@pytest.mark.parametrize(
    ("residual", "mlp_ratio"),
    [(True, None), (False, 2.0), (True, 2.0)],
)
def test_rejects_memory_with_different_residual_mlp_architecture(
    residual, mlp_ratio
):
    memory = ingest(build_model(), token_ids())
    other = DABSNSequenceLM(
        vocab=VOCAB,
        hidden_dim=16,
        depth=2,
        layers=[
            DABSNLayerSpec(16, 16, "seq"),
            DABSNLayerSpec(16, 16, "seq"),
        ],
        residual=residual,
        mlp_ratio=mlp_ratio,
    ).eval()
    with pytest.raises(ValueError, match="different architecture"):
        forward_with_memory(other, token_ids(tokens=4), memory)


def test_legacy_memory_fingerprint_means_features_disabled():
    model = build_model()
    memory = ingest(model, token_ids())
    memory.fingerprint.pop("residual")
    memory.fingerprint.pop("mlp_ratio")
    forward_with_memory(model, token_ids(tokens=4), memory)


def test_rejects_batch_mismatch():
    model = build_model()
    memory = ingest(model, token_ids())
    with pytest.raises(ValueError, match="cannot be rebatched"):
        forward_with_memory(model, torch.zeros(1, 4, dtype=torch.long), memory)


def test_rejects_non_seq_geometry():
    model = build_model(geometry="field")
    with pytest.raises(NotImplementedError, match="seq read geometry"):
        ingest(model, token_ids(tokens=8))


def test_memory_cost_matches_reality():
    model = build_model()
    memory = ingest(model, token_ids())
    predicted = memory_cost(model, TOKENS, batch_size=BATCH)["total_bytes"]
    banked = sum(
        t.numel() * t.element_size()
        for layer in memory.layers
        for t in layer.bank.values()
    )
    assert predicted == banked


def test_empty_memory_reports_zero():
    assert DABSNMemory().entries == 0
    assert DABSNMemory().nbytes() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity gate")
def test_cuda_carry_matches_cpu_and_dense():
    """The compact CUDA read takes ``_compact_bank_idx = bank_idx - query_start``,
    which is NEGATIVE for bank entries before the query window -- the carry case
    -- and a ``total_T`` equal to the query length while the bank is longer.
    Nothing exercised that until carry existed. This is that gate.
    """
    model = build_model().cuda()
    ids = token_ids().cuda()
    make_selective(model)
    with torch.no_grad():
        reference = model.forward_sequence(ids)

    logits, _ = forward_with_memory(model, ids, chunk_size=8, extend=True)
    assert torch.allclose(logits, reference, atol=1e-3, rtol=1e-2)

    memory = ingest(model, ids[:, :32], chunk_size=8)
    split, _ = forward_with_memory(model, ids[:, 32:], memory)
    assert torch.allclose(split, reference[:, 32:], atol=1e-3, rtol=1e-2)

    cpu_logits, _ = forward_with_memory(
        build_model().cpu(), ids.cpu(), chunk_size=8, extend=True
    )
    assert torch.allclose(logits.cpu(), cpu_logits, atol=1e-2, rtol=1e-2)
