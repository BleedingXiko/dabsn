import torch
from safetensors.torch import save_file

from dabsn import import_prototype_moe_checkpoint, inspect_dabsn, load_dabsn
from dabsn.model import DABSNSequenceLM


def _prototype_fixture(path):
    torch.manual_seed(92)
    base = DABSNSequenceLM(
        vocab=17,
        hidden_dim=4,
        depth=1,
        layers=[{"hidden_dim": 4, "state_dim": 4, "read_geometry": "seq"}],
        tie_embeddings=False,
    )
    state = {
        "embed.weight": base.embed.weight.detach().clone(),
        "readout.weight": base.readout.weight.detach().clone(),
        "readout.bias": base.readout.bias.detach().clone(),
    }
    for name, value in base.backbone.blocks[0].state_dict().items():
        state[f"backbone.blocks.0.dabsn.{name}"] = value.detach().clone()
    state.update(
        {
            "backbone.blocks.0.moe.norm.weight": torch.randn(4),
            "backbone.blocks.0.moe.router.weight": torch.randn(3, 4),
            "backbone.blocks.0.moe.experts.0.fc1.weight": torch.randn(8, 4),
            "backbone.blocks.0.moe.experts.0.fc2.weight": torch.randn(4, 8),
            "backbone.blocks.0.moe.experts.1.fc1.weight": torch.randn(8, 4),
            "backbone.blocks.0.moe.experts.1.fc2.weight": torch.randn(4, 8),
            "backbone.blocks.0.moe.experts.2.fc1.weight": torch.randn(8, 4),
            "backbone.blocks.0.moe.experts.2.fc2.weight": torch.randn(4, 8),
        }
    )
    config = {
        "model_kind": "sequence_lm",
        "vocab": 17,
        "hidden_dim": 4,
        "depth": 1,
        "layers": [{"hidden_dim": 4, "state_dim": 4, "read_geometry": "seq"}],
        "tie_embeddings": False,
        "residual": False,
        "mlp_ratio": None,
        "mlp_middle_depth": 0,
        "mlp_depth_index": 0,
        "moe_experts": 3,
        "moe_top_k": 2,
        "moe_ratio": 2.0,
        "moe_aux_coeff": 0.01,
        "moe_stack_residual": True,
    }
    import json

    save_file(
        state,
        str(path),
        metadata={
            "format": "dabsn-model",
            "version": "1",
            "config": json.dumps(config),
            "extra": "{}",
            "shared": "{}",
        },
    )
    return {name: value.clone() for name, value in state.items()}


def test_prototype_moe_import_is_one_way_native_graph_conversion(tmp_path):
    source = tmp_path / "prototype.safetensors"
    destination = tmp_path / "native-v2.safetensors"
    original = _prototype_fixture(source)
    source_bytes = source.read_bytes()

    digest = import_prototype_moe_checkpoint(source, destination)

    assert source.read_bytes() == source_bytes
    assert len(digest) == 64
    metadata = inspect_dabsn(destination)
    assert metadata["version"] == 2
    assert [item["provider_key"] for item in metadata["manifest"]["graph"]] == [
        "dabsn:block",
        "dabsn:sparse_moe",
    ]
    model = load_dabsn(destination)
    state = model.state_dict()
    torch.testing.assert_close(
        state["backbone.graph.components.1.router.proj.weight"],
        original["backbone.blocks.0.moe.router.weight"],
    )
    expected_w1 = torch.stack(
        [original[f"backbone.blocks.0.moe.experts.{index}.fc1.weight"].T for index in range(3)]
    )
    torch.testing.assert_close(state["backbone.graph.components.1.expert_group.w1"], expected_w1)
    assert model.graph.dabsn_memory_count == 1


def test_prototype_import_refuses_in_place(tmp_path):
    source = tmp_path / "prototype.safetensors"
    _prototype_fixture(source)
    import pytest

    with pytest.raises(ValueError, match="never rewrites"):
        import_prototype_moe_checkpoint(source, source)
