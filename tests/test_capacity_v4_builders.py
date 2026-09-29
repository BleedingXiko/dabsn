import torch

from dabsn import (
    CapacityConfig,
    DABSNConfig,
    DABSNSequenceLM,
    build_dabsn,
    build_dabsn_from_config,
    finalize_online_temporal_state,
    initial_online_temporal_state,
    load_online_temporal_state,
    online_temporal_state_dict,
    save_dabsn,
    load_dabsn,
)
from dabsn.capacity import DABSNCapacitySubsystem


def test_existing_build_dabsn_modes():
    off = build_dabsn(8, 5, 8, read_geometry="seq", state_dim=8)
    assert off.backbone.blocks[0].capacity_module is None

    dense = build_dabsn(
        8, 5, 8, read_geometry="seq", state_dim=8,
        capacity=CapacityConfig(mode="dense", size=64),
    )
    assert dense.backbone.blocks[0].capacity_module.mode == "dense"

    sparse = build_dabsn(
        8, 5, 8, read_geometry="seq", state_dim=8,
        capacity=CapacityConfig(mode="sparse", size=64, active_units=8),
    )
    cap = sparse.backbone.blocks[0].capacity_module
    assert cap.mode == "sparse"
    assert cap.size == 64
    assert cap.active_units == 8


def test_existing_build_from_mapping():
    model = build_dabsn_from_config({
        "input_dim": 8,
        "out_dim": 5,
        "hidden_dim": 8,
        "geometry": "seq",
        "state_dim": 8,
        "capacity": {"mode": "sparse", "size": 128, "active_units": 8},
    })
    cap = model.backbone.blocks[0].capacity_module
    assert cap is not None and cap.mode == "sparse" and cap.active_units == 8


def test_per_layer_capacity_config():
    cfg = DABSNConfig(
        input_dim=8,
        out_dim=5,
        hidden_dim=8,
        layers=[
            {"hidden_dim": 8, "state_dim": 8, "read_geometry": "seq",
             "capacity": {"mode": "sparse", "size": 64, "active_units": 4}},
            {"hidden_dim": 8, "state_dim": 8, "read_geometry": "seq",
             "capacity": {"mode": "dense", "size": 32}},
            {"hidden_dim": 8, "state_dim": 8, "read_geometry": "seq"},
        ],
    )
    model = build_dabsn_from_config(cfg)
    assert model.backbone.blocks[0].capacity_module.mode == "sparse"
    assert model.backbone.blocks[1].capacity_module.mode == "dense"
    assert model.backbone.blocks[2].capacity_module is None


def _capacity_online_parity(mode: str):
    torch.manual_seed(7)
    cfg = CapacityConfig(mode=mode, size=32, active_units=4, address_seed=2)
    full = DABSNCapacitySubsystem(6, 5, cfg).double()
    online = DABSNCapacitySubsystem(6, 5, cfg).double()
    online.load_state_dict(full.state_dict())
    x = torch.randn(2, 9, 6, dtype=torch.float64)
    cot = torch.randn(2, 9, 5, dtype=torch.float64)
    y = full.forward_sequence(x)
    (y * cot).sum().backward()

    recurrent, eligibility = online.initial_online_state(
        2, device=x.device, recurrent_dtype=x.dtype, eligibility_dtype=x.dtype
    )
    outputs = []
    start = 0
    for width in (2, 3, 1, 3):
        xx = x[:, start:start + width]
        cc = cot[:, start:start + width]
        yy, recurrent, eligibility = online.forward_online(xx, recurrent, eligibility)
        outputs.append(yy)
        (yy * cc).sum().backward()
        start += width
    assert torch.equal(y, torch.cat(outputs, dim=1))
    for name in ("key", "bias", "retention_logit", "value", "gain"):
        expected = getattr(full, name).grad
        actual = getattr(online, name).grad
        if actual.is_sparse:
            actual = actual.to_dense()
        assert torch.allclose(expected, actual, atol=2e-15, rtol=0)


def test_dense_online_exact_credit():
    _capacity_online_parity("dense")


def test_sparse_online_exact_credit():
    _capacity_online_parity("sparse")


def test_sequence_lm_online_uses_canonical_state():
    model = DABSNSequenceLM(
        vocab=64,
        hidden_dim=16,
        depth=1,
        layer_geometries="seq",
        state_dim=16,
        temporal_credit="online",
        capacity=CapacityConfig(mode="sparse", size=128, active_units=8),
    ).double()
    state = initial_online_temporal_state(
        model, 2, stream_policy="carry_lanes", stream_ids=("a", "b"),
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ids = torch.randint(0, 64, (2, 4))
    target = torch.randint(0, 64, (2, 4))
    logits, pending = model.forward_sequence_online(ids, state)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), target.reshape(-1))
    loss.backward()
    state = finalize_online_temporal_state(pending)
    cap = model.backbone.blocks[0].capacity_module
    assert cap.key.grad is not None and cap.key.grad.is_sparse
    assert tuple(state.layers[0].capacity_recurrent["features"].shape) == (2, 128)

    payload = online_temporal_state_dict(state)
    restored = load_online_temporal_state(model, payload, device="cpu")
    assert torch.equal(
        restored.layers[0].capacity_recurrent["features"],
        state.layers[0].capacity_recurrent["features"],
    )


def test_model_checkpoint_roundtrip(tmp_path):
    model = build_dabsn(
        8, 5, 8, read_geometry="seq", state_dim=8,
        capacity=CapacityConfig(mode="sparse", size=64, active_units=8),
    )
    path = tmp_path / "capacity.safetensors"
    save_dabsn(model, path)
    restored = load_dabsn(path)
    cap = restored.backbone.blocks[0].capacity_module
    assert cap.mode == "sparse" and cap.size == 64 and cap.active_units == 8
    torch.manual_seed(19)
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        assert torch.equal(model.forward_sequence(x), restored.forward_sequence(x))


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
