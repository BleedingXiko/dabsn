"""Contracts for native optional Capacity, starting at its canonical initialization."""
import pytest
import torch
import dabsn


def test_capacity_config_is_public():
    assert hasattr(dabsn, "CapacityConfig")


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("geometry", ["seq", "field", "hybrid"])
def test_optional_capacity_build_and_learn(mode, geometry):
    cfg = dabsn.CapacityConfig(mode=mode, size=16, active_units=4)
    model = dabsn.build_dabsn(3, 2, 5, state_dim=4, read_geometry=geometry, capacity=cfg)
    cap = model.backbone.blocks[0].capacity_module
    assert torch.count_nonzero(cap.value) == 0
    x = torch.randn(2, 4, 3)
    initial = model.forward_sequence(x)
    initial.square().mean().backward()
    assert cap.value.grad is not None
    assert torch.isfinite(cap.value.grad).all()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    assert torch.count_nonzero(cap.value) > 0


def test_disabled_capacity_preserves_rng_parameters_and_outputs():
    torch.manual_seed(932)
    baseline = dabsn.build_dabsn(3, 2, 5, state_dim=4, read_geometry="seq")
    torch.manual_seed(932)
    disabled = dabsn.build_dabsn(3, 2, 5, state_dim=4, read_geometry="seq",
                                 capacity=dabsn.CapacityConfig(enabled=False))
    assert baseline.state_dict().keys() == disabled.state_dict().keys()
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[name])
    x = torch.randn(2, 4, 3)
    assert torch.equal(baseline(x), disabled(x))


def test_provider_roundtrip_and_layer_override(tmp_path):
    from dabsn import component_registry, save_graph, load_graph, DABSNGraph
    cfg = {"input_dim": 3, "hidden_dim": 5, "state_dim": 4,
           "read_geometry": "seq", "capacity": {"mode": "sparse", "size": 16, "active_units": 4}}
    binding = component_registry.build(dabsn.ComponentSpec("world", "dabsn:block", cfg))
    graph = DABSNGraph([binding])
    x = torch.randn(2, 4, 3)
    path = tmp_path / "graph.safetensors"
    save_graph(graph, path)
    restored = load_graph(path)
    assert restored.bindings[0].module.capacity_module.size == 16
    torch.testing.assert_close(graph(x), restored(x))
    model = dabsn.build_dabsn_from_config(dabsn.DABSNConfig(
        input_dim=3, out_dim=2, hidden_dim=5,
        capacity={"mode": "dense", "size": 16},
        layers=[{"hidden_dim": 5, "read_geometry": "seq"},
                {"hidden_dim": 5, "read_geometry": "seq", "capacity": {"enabled": False}}]))
    assert model.backbone.blocks[0].capacity_module is not None
    assert model.backbone.blocks[1].capacity_module is None


def test_pretrain_capacity_mapping_and_no_capacity_metadata():
    cfg = dabsn.DABSNPretrainConfig(corpus_text="text.txt", capacity={"mode": "sparse", "size": 16, "active_units": 4})
    assert cfg.capacity.size == 16
    assert "capacity" not in dabsn.DABSNLayerSpec(5, 4, "seq").to_metadata()


def test_online_restore_rejects_changed_capacity_addressing():
    from dabsn.temporal_credit import initial_online_temporal_state, online_temporal_state_dict, load_online_temporal_state
    original = dabsn.DABSNSequenceLM(vocab=17, hidden_dim=4, depth=1,
        capacity={"mode": "sparse", "size": 16, "active_units": 4, "address_seed": 11})
    changed = dabsn.DABSNSequenceLM(vocab=17, hidden_dim=4, depth=1,
        capacity={"mode": "sparse", "size": 16, "active_units": 4, "address_seed": 12})
    state = initial_online_temporal_state(original, 2, device="cpu", dtype=torch.float32,
        stream_policy="carry_lanes", stream_ids=("a", "b"))
    payload = online_temporal_state_dict(state)
    with pytest.raises(ValueError, match="Capacity contract"):
        load_online_temporal_state(changed, payload)
