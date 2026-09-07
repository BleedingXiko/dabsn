"""Exactness contract for the paged carried-memory inference representation."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from dabsn.memory import (
    DABSNMemory,
    LayerMemory,
    PreparedDABSNMemory,
    forward_with_memory,
    fingerprint_of,
    ingest,
    load_prepared,
    prepare_for_inference,
    save_prepared,
)
from dabsn.model import DABSNLayerSpec, DABSNSequenceLM
from dabsn.prepared_memory import prepared_memory_bytes, prepared_three_way_read


def _model(*, seed: int = 0) -> DABSNSequenceLM:
    torch.manual_seed(seed)
    return DABSNSequenceLM(
        vocab=43,
        hidden_dim=8,
        depth=1,
        layers=[DABSNLayerSpec(hidden_dim=8, state_dim=8, read_geometry="seq")],
    ).eval()


def _ids(tokens: int, *, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 43, (2, tokens), generator=generator)


def _valid_rows(prepared: PreparedDABSNMemory, layer: int = 0):
    bank = prepared.layers[layer].bank
    rows = []
    for lane in range(prepared.batch_size):
        lane_rows = []
        for page in bank.pages:
            for slot in torch.nonzero(page.valid[lane], as_tuple=False).flatten().tolist():
                lane_rows.append(
                    (
                        int(page.position[lane, slot]),
                        page.write[lane, slot],
                        page.successor_write[lane, slot],
                    )
                )
        rows.append(lane_rows)
    return rows


def test_prepare_matches_existing_prefix_admission_and_does_not_mutate_source():
    model = _model()
    raw = ingest(model, _ids(19), chunk_size=5)
    source = copy.deepcopy(raw)
    prepared = prepare_for_inference(model, raw, page_size=4)

    block = model.backbone.blocks[0]
    bank = raw.layers[0].bank
    hidden = bank["write"].shape[-1]
    admission = (
        block.read.short_read.adm_p_gain * torch.log1p(bank["plasticity_mean"] * hidden)
        + block.read.short_read.adm_novelty_gain * bank["novelty_mean"]
        + block.read.short_read.adm_saturation_gain * bank["saturation_mean"]
        + block.read.short_read.adm_energy_gain * bank["energy_mean"]
        + block.read.short_read.adm_bias
    )
    member = admission >= torch.cummax(admission, dim=1).values - torch.nn.functional.softplus(
        block.read.admit_window
    )

    assert torch.equal(prepared.layers[0].bank.entries, member.sum(dim=1))
    for lane, rows in enumerate(_valid_rows(prepared)):
        assert [position for position, _, _ in rows] == torch.nonzero(
            member[lane], as_tuple=False
        ).flatten().tolist()
    for name, value in raw.layers[0].bank.items():
        assert torch.equal(value, source.layers[0].bank[name])


def test_prepared_query_matches_legacy_query_bank():
    model = _model()
    context, query = _ids(23)[:, :17], _ids(6, seed=7)
    raw = ingest(model, context, chunk_size=4)
    prepared = prepare_for_inference(model, raw, page_size=3)

    expected, unchanged_raw = forward_with_memory(model, query, raw)
    actual, unchanged_prepared = forward_with_memory(model, query, prepared)

    assert unchanged_raw is raw
    assert unchanged_prepared is prepared
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_prepared_incremental_decode_matches_legacy_and_dense_sequence():
    model = _model()
    ids = _ids(25)
    context, tail = ids[:, :13], ids[:, 13:]
    raw = ingest(model, context, chunk_size=4)
    prepared = prepare_for_inference(model, raw, page_size=4)

    legacy_steps = []
    prepared_steps = []
    for step in range(tail.shape[1]):
        legacy, raw = forward_with_memory(model, tail[:, step : step + 1], raw, extend=True)
        current, prepared = forward_with_memory(
            model, tail[:, step : step + 1], prepared, extend=True
        )
        legacy_steps.append(legacy)
        prepared_steps.append(current)

    with torch.no_grad():
        dense = model.forward_sequence(ids)[:, 13:]
    legacy_logits = torch.cat(legacy_steps, dim=1)
    prepared_logits = torch.cat(prepared_steps, dim=1)
    assert torch.allclose(prepared_logits, legacy_logits, atol=2e-6, rtol=2e-5)
    assert torch.allclose(prepared_logits, dense, atol=1e-5, rtol=1e-4)
    assert prepared.position == ids.shape[1]


def test_induction_successor_is_next_chronological_write_even_when_not_admitted():
    model = _model()
    block = model.backbone.blocks[0]
    with torch.no_grad():
        block.read.admit_window.fill_(-8.0)
        block.read.short_read.adm_p_gain.fill_(1.0)
        block.read.short_read.adm_novelty_gain.zero_()
        block.read.short_read.adm_saturation_gain.zero_()
        block.read.short_read.adm_energy_gain.zero_()
        block.read.short_read.adm_bias.zero_()
    writes = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    zeros = torch.zeros(1, 4)
    layer = LayerMemory(
        bank={
            "write": writes.clone(),
            "novelty_mean": zeros.clone(),
            "plasticity_mean": torch.tensor([[100.0, 0.0, 200.0, 0.0]]),
            "energy_mean": zeros.clone(),
            "saturation_mean": zeros.clone(),
        },
        core=tuple(torch.zeros(1, 8) for _ in range(3)),
        long=tuple(torch.zeros(1, 8) for _ in range(3)),
    )
    raw = DABSNMemory(
        layers=[layer], position=4, batch_size=1, fingerprint=fingerprint_of(model)
    )

    prepared = prepare_for_inference(model, raw, page_size=2)
    rows = _valid_rows(prepared)[0]
    assert [row[0] for row in rows] == [0, 2]
    assert torch.equal(rows[0][2], writes[0, 1])
    assert torch.equal(rows[1][2], writes[0, 3])


def test_prepared_memory_rejects_changed_read_parameters_without_mutation():
    model = _model()
    prepared = prepare_for_inference(model, ingest(model, _ids(8)), page_size=4)
    entries = prepared.layers[0].bank.entries.clone()
    with torch.no_grad():
        model.backbone.blocks[0].read.short_read.adm_bias.add_(1.0)

    with pytest.raises(ValueError, match="read parameters"):
        forward_with_memory(model, _ids(2, seed=9), prepared)
    assert torch.equal(prepared.layers[0].bank.entries, entries)


def test_prepared_v2_save_load_is_bitwise_stable(tmp_path):
    model = _model()
    prepared = prepare_for_inference(model, ingest(model, _ids(11)), page_size=3)
    path = save_prepared(prepared, tmp_path / "memory-v2.dmem")
    restored = load_prepared(path, model=model)

    assert restored.position == prepared.position
    assert restored.format == "dabsn_prepared_memory_v2"
    for expected_layer, actual_layer in zip(prepared.layers, restored.layers):
        assert torch.equal(expected_layer.bank.entries, actual_layer.bank.entries)
        for expected_page, actual_page in zip(expected_layer.bank.pages, actual_layer.bank.pages):
            for name in (
                "write",
                "successor_write",
                "position",
                "scalar_features",
                "inverse_write_norm",
                "admission",
                "key_bias",
                "valid",
            ):
                assert torch.equal(getattr(expected_page, name), getattr(actual_page, name))


def test_prepared_page_address_math_is_int64_above_two_gigaelements():
    model = _model()
    prepared = prepare_for_inference(model, ingest(model, _ids(3)), page_size=2)
    bank = prepared.layers[0].bank
    linear = bank.linear_offset(page=524_288, slot=1, hidden_index=4_095, hidden_dim=4_096)
    assert linear.dtype == torch.int64
    assert int(linear) > 2**31


@pytest.mark.parametrize("active_component", ["short", "permanent", "induction"])
def test_prepared_components_match_dense_equations_independently(active_component):
    model = _model()
    read = model.backbone.blocks[0].read
    prepared = prepare_for_inference(model, ingest(model, _ids(9)), page_size=32)
    bank = prepared.layers[0].bank
    page = bank.pages[0]
    torch.manual_seed(31)
    expression = torch.randn(2, 3, 8)
    cocktail = F.normalize(torch.randn(2, 3, 4), dim=-1)
    query = F.normalize(expression, dim=-1)
    starts = torch.tensor([9, 9], dtype=torch.int64)
    scale = (F.softplus(read.short_read.log_temp) + 1.0) * (8**0.5)
    gains = {
        "short_gain": torch.tensor(float(active_component == "short")),
        "pad_gain": torch.tensor(float(active_component == "permanent")),
        "induct_gain": torch.tensor(float(active_component == "induction")),
    }
    actual = prepared_three_way_read(
        query,
        cocktail,
        bank,
        starts,
        scale=scale,
        cocktail_gain=read.cocktail_gain,
        **gains,
    )

    scores = torch.bmm(query, F.normalize(page.write, dim=-1).transpose(1, 2)) * scale
    short_scores = scores + (page.key_bias + page.admission).unsqueeze(1)
    permanent_scores = short_scores + read.cocktail_gain * torch.bmm(
        cocktail, page.scalar_features.transpose(1, 2)
    )
    query_positions = starts[:, None] + torch.arange(3)[None, :]
    allow = page.valid.unsqueeze(1) & (page.position.unsqueeze(1) <= query_positions.unsqueeze(2))
    induct_allow = page.valid.unsqueeze(1) & (
        page.position.unsqueeze(1) < query_positions.unsqueeze(2)
    )
    if active_component == "short":
        weights = torch.softmax(short_scores.masked_fill(~allow, -torch.inf), dim=-1)
        expected = torch.bmm(weights, page.write)
    elif active_component == "permanent":
        weights = torch.softmax(permanent_scores.masked_fill(~allow, -torch.inf), dim=-1)
        expected = torch.bmm(weights, page.write)
    else:
        weights = torch.softmax(short_scores.masked_fill(~induct_allow, -torch.inf), dim=-1)
        expected = torch.bmm(weights, page.successor_write)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_query_to_bank_content_score_runs_once_per_page(monkeypatch):
    model = _model()
    read = model.backbone.blocks[0].read
    prepared = prepare_for_inference(model, ingest(model, _ids(10)), page_size=3)
    bank = prepared.layers[0].bank
    query = F.normalize(torch.randn(2, 2, 8), dim=-1)
    cocktail = F.normalize(torch.randn(2, 2, 4), dim=-1)
    calls = []
    original = torch.bmm

    def counted(left, right, *args, **kwargs):
        calls.append((tuple(left.shape), tuple(right.shape)))
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, "bmm", counted)
    prepared_three_way_read(
        query,
        cocktail,
        bank,
        torch.tensor([10, 10]),
        scale=torch.tensor(1.0),
        short_gain=read.short_gain,
        pad_gain=read.pad_gain,
        induct_gain=read.induct_gain,
        cocktail_gain=read.cocktail_gain,
    )
    hidden_qk = [shape for shape in calls if shape[0][-1] == 8 and shape[1][1] == 8]
    assert len(hidden_qk) == len(bank.pages)


def test_prepared_workspace_includes_simultaneously_live_accumulators():
    estimate = prepared_memory_bytes(batch=1, committed_entries=1, hidden=1024,
                                     page_size=1, dtype=torch.float32, query_tile=256)
    # Three old plus three new numerators coexist at the page operator boundary.
    assert estimate["workspace_bytes_upper_bound"] >= 6 * 256 * 1024 * 4


def test_prepared_read_does_not_score_unused_tail_capacity():
    from dabsn.prepared_memory import prepare_admitted_bank

    read = _model().backbone.blocks[0].read
    raw = {"write": torch.randn(2, 17, 8)}
    for name in ("novelty_mean", "plasticity_mean", "energy_mean", "saturation_mean"):
        raw[name] = torch.rand(2, 17)
    bank = prepare_admitted_bank(read, raw, page_size=1024, start_position=71)
    query = F.normalize(torch.randn(2, 3, 8), dim=-1)
    cocktail = F.normalize(torch.randn(2, 3, 4), dim=-1)
    kwargs = dict(scale=torch.tensor(1.), short_gain=read.short_gain,
                  pad_gain=read.pad_gain, induct_gain=read.induct_gain,
                  cocktail_gain=read.cocktail_gain)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True) as profile:
        actual = prepared_three_way_read(query, cocktail, bank, torch.tensor([88, 88]), **kwargs)
    scored_capacities = [event.input_shapes[1][-1] for event in profile.events()
                         if event.name == "aten::bmm" and event.input_shapes[0] == [2, 3, 8]]
    assert scored_capacities and max(scored_capacities) <= 17
    assert bank.pages[0].write.shape[1] == 1024  # Persistent capacity remains appendable.
    compact = prepare_admitted_bank(read, raw, page_size=17, start_position=71)
    expected = prepared_three_way_read(query, cocktail, compact, torch.tensor([88, 88]), **kwargs)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    scalar_reads = sum(event.count for event in profile.key_averages()
                       if event.key == "aten::_local_scalar_dense")
    assert scalar_reads == 0


def test_prepared_append_does_not_extract_host_scalars_per_token_or_lane():
    from dabsn.prepared_memory import append_prepared_bank, prepare_admitted_bank

    read = _model().backbone.blocks[0].read
    raw = {"write": torch.randn(2, 13, 8)}
    for name in ("novelty_mean", "plasticity_mean", "energy_mean", "saturation_mean"):
        raw[name] = torch.rand(2, 13)
    bank = prepare_admitted_bank(read, {key: value[:, :1] for key, value in raw.items()}, page_size=3)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        appended = append_prepared_bank(bank, read, **{key: value[:, 1:] for key, value in raw.items()})
    scalar_reads = sum(event.count for event in profile.key_averages() if event.key == "aten::_local_scalar_dense")
    assert scalar_reads <= 1
    assert appended.next_position == 13


def test_prepared_capacity_estimate_matches_materialized_bank():
    model = _model()
    prepared = prepare_for_inference(model, ingest(model, _ids(10)), page_size=3)
    bank = prepared.layers[0].bank
    estimate = prepared_memory_bytes(
        batch=2,
        committed_entries=int(bank.entries.max()),
        hidden=8,
        page_size=3,
        dtype=torch.float32,
    )
    assert estimate["persistent_bytes"] == bank.nbytes()
    assert estimate["padded_entries"] == len(bank.pages) * 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_preparation_detaches_state_and_handles_zero_writes(dtype):
    from dabsn.prepared_memory import prepare_admitted_bank
    from dabsn.read import DABSNRead

    read = DABSNRead(8).to(dtype)
    raw = {"write": torch.zeros(2, 5, 8, dtype=dtype, requires_grad=True)}
    for name in ("novelty_mean", "plasticity_mean", "energy_mean", "saturation_mean"):
        raw[name] = torch.zeros(2, 5, dtype=dtype)
    bank = prepare_admitted_bank(read, raw, page_size=3)
    for page in bank.pages:
        for value in vars(page).values():
            assert value.grad_fn is None and not value.requires_grad
            assert bool(torch.isfinite(value).all())


def test_prepared_save_refuses_overwrite_unless_explicit(tmp_path):
    model = _model()
    prepared = prepare_for_inference(model, ingest(model, _ids(5)), page_size=3)
    path = save_prepared(prepared, tmp_path / "memory.dmem")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_prepared(prepared, path)
    assert path.read_bytes() == before
    save_prepared(prepared, path, overwrite=True)
    assert load_prepared(path).position == prepared.position


def test_prepared_payload_rejects_bad_page_shapes_and_index_dtypes():
    from dabsn.prepared_memory import from_payload, to_payload

    model = _model()
    memory = prepare_for_inference(model, ingest(model, _ids(5)), page_size=3)
    payload = to_payload(memory)
    payload["layers"][0]["bank"]["pages"][0]["position"] = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="int64"):
        from_payload(payload)
    payload = to_payload(memory)
    payload["layers"][0]["bank"]["pages"][0]["successor_write"] = torch.zeros(2, 2, 8)
    with pytest.raises(ValueError, match="shape"):
        from_payload(payload)


@pytest.mark.parametrize("page_size", [1, 3, 16])
def test_sparse_append_and_tiled_causal_queries_match_one_shot(page_size):
    from dabsn.prepared_memory import prepare_admitted_bank, append_prepared_bank

    read = _model().backbone.blocks[0].read
    with torch.no_grad():
        read.admit_window.fill_(-8)
    torch.manual_seed(719)
    raw = {"write": torch.randn(2, 13, 8)}
    for name in ("novelty_mean", "plasticity_mean", "energy_mean", "saturation_mean"):
        raw[name] = torch.rand(2, 13)
    complete = prepare_admitted_bank(read, raw, page_size=page_size)
    incremental = prepare_admitted_bank(read, {k: v[:, :1] for k, v in raw.items()}, page_size=page_size)
    for index in range(1, 13):
        incremental = append_prepared_bank(incremental, read, **{k: v[:, index:index+1] for k, v in raw.items()})
    assert bool((complete.entries < 13).all())
    torch.testing.assert_close(incremental.entries, complete.entries, rtol=0, atol=0)
    for actual, expected in zip(incremental.pages, complete.pages):
        for name in vars(expected):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)

    query = F.normalize(torch.randn(2, 19, 8), dim=-1)
    cocktail = F.normalize(torch.randn(2, 19, 4), dim=-1)
    kwargs = dict(scale=torch.tensor(1.), short_gain=read.short_gain, pad_gain=read.pad_gain,
                  induct_gain=read.induct_gain, cocktail_gain=read.cocktail_gain)
    # Includes all-masked queries, bank-internal queries, and queries after the bank.
    starts = torch.tensor([-2, -1])
    untiled = prepared_three_way_read(query, cocktail, complete, starts, **kwargs)
    tiled = prepared_three_way_read(query, cocktail, incremental, starts, query_tile_size=3, **kwargs)
    torch.testing.assert_close(tiled, untiled, atol=2e-6, rtol=2e-5)
    assert bool(torch.isfinite(tiled).all())
    assert torch.equal(tiled[0, :2], torch.zeros_like(tiled[0, :2]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA prepared-read parity gate")
def test_cuda_prepared_read_matches_legacy_cuda():
    model = _model().cuda()
    context = _ids(37)[:, :31].cuda()
    query = _ids(6, seed=17).cuda()
    raw = ingest(model, context, chunk_size=8)
    prepared = prepare_for_inference(model, raw, page_size=7)

    expected, _ = forward_with_memory(model, query, raw)
    actual, _ = forward_with_memory(model, query, prepared)
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-4)
    assert (
        model.backbone.blocks[0].read.last_read_contract["kernel_backend"]
        == "cuda_inductor_pagewise_bmm_score_once"
    )
