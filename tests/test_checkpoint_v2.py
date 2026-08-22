import json

import pytest
import torch
from safetensors.torch import save_file

from dabsn import (
    ComponentSpec,
    DABSNGraph,
    DABSNSequenceLM,
    artifact_digest,
    build_graph_from_config,
    component_registry,
    inspect_dabsn,
    load_dabsn,
    load_graph,
    migrate_dabsn_checkpoint,
    save_dabsn,
    save_graph,
)
from dabsn.checkpoint import CHECKPOINT_FORMAT, CHECKPOINT_VERSION


def _legacy_model():
    torch.manual_seed(300)
    return DABSNSequenceLM(
        vocab=31,
        hidden_dim=6,
        depth=2,
        layer_geometries="seq",
        residual=True,
        mlp_ratio=2.0,
        mlp_middle_depth=2,
        mlp_depth_index=0,
    ).eval()


def _portable_graph_model():
    first = component_registry.build(
        ComponentSpec(
            "dabsn.0",
            "dabsn:block",
            {
                "input_dim": 6,
                "hidden_dim": 6,
                "state_dim": 8,
                "read_geometry": "seq",
                "residual": True,
            },
        )
    )
    mlp = component_registry.build(
        ComponentSpec("mlp.0", "dabsn:residual_mlp", {"dim": 6, "ratio": 2.0})
    )
    return DABSNSequenceLM.from_graph(
        DABSNGraph([first, mlp], require_world_builder=True), vocab=31
    ).eval()


def _write_v1(model, path):
    config = {
        "model_kind": "sequence_lm",
        "vocab": model.vocab,
        "hidden_dim": model.hidden_dim,
        "depth": model.depth,
        "state_dim": model.state_dim,
        "layers": [spec.to_metadata() for spec in model.layers],
        "tie_embeddings": model.tie_embeddings,
        "grad_checkpoint": False,
        "residual": model.residual,
        "mlp_ratio": model.mlp_ratio,
        "mlp_middle_depth": model.mlp_middle_depth,
        "mlp_depth_index": model.mlp_depth_index,
    }
    save_file(
        {name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()},
        str(path),
        metadata={
            "format": CHECKPOINT_FORMAT,
            "version": "1",
            "config": json.dumps(config),
            "extra": "{}",
            "shared": "{}",
        },
    )


def test_v2_legacy_wrapper_manifest_is_complete_and_roundtrips(tmp_path):
    model = _legacy_model()
    ids = torch.randint(0, model.vocab, (2, 7))
    expected = model.forward_sequence(ids)
    path = tmp_path / "legacy-v2.safetensors"
    save_dabsn(model, path, extra={"experiment": "fixture"})
    metadata = inspect_dabsn(path)
    manifest = metadata["manifest"]
    assert metadata["version"] == CHECKPOINT_VERSION == 2
    assert manifest["construction"] == "legacy-wrapper"
    assert [item["component_id"] for item in manifest["graph"]] == [
        "dabsn.0",
        "legacy-inline-mlp.0",
        "legacy-middle-mlp.0",
        "legacy-middle-mlp.1",
        "dabsn.1",
        "legacy-inline-mlp.1",
    ]
    assert manifest["dabsn_memory_owners"] == ["dabsn.0", "dabsn.1"]
    assert manifest["contract_fingerprint"]
    assert set(manifest["parameter_namespace_map"]) == set(model.state_dict())
    restored = load_dabsn(path).eval()
    torch.testing.assert_close(restored.forward_sequence(ids), expected, atol=0, rtol=0)


def test_first_class_portable_graph_reconstructs_from_registered_providers(tmp_path):
    model = _portable_graph_model()
    model.backbone.grad_checkpoint = True
    model.backbone.graph.set_activation_checkpointing(True)
    ids = torch.randint(0, 31, (2, 7))
    expected = model.forward_sequence(ids)
    path = tmp_path / "graph-v2.safetensors"
    save_dabsn(model, path)
    metadata = inspect_dabsn(path)
    assert metadata["config"]["model_kind"] == "sequence_graph"
    assert metadata["config"]["grad_checkpoint"] is True
    assert metadata["manifest"]["construction"] == "graph"
    assert [x["provider_key"] for x in metadata["manifest"]["graph"]] == [
        "dabsn:block",
        "dabsn:residual_mlp",
    ]
    restored = load_dabsn(path).eval()
    assert restored.backbone.grad_checkpoint is True
    torch.testing.assert_close(restored.forward_sequence(ids), expected, atol=0, rtol=0)


def test_raw_domain_graph_roundtrips_without_language_wrapper(tmp_path):
    first = component_registry.build(
        ComponentSpec(
            "dabsn.image_world",
            "dabsn:block",
            {
                "input_dim": 6,
                "hidden_dim": 6,
                "state_dim": 8,
                "read_geometry": "field",
                "residual": True,
            },
        )
    )
    transform = component_registry.build(
        ComponentSpec("mlp.image", "dabsn:residual_mlp", {"dim": 6, "ratio": 2.0})
    )
    graph = DABSNGraph([first, transform])
    value = torch.randn(2, 5, 6)
    expected = graph(value)
    path = tmp_path / "raw-domain-graph.safetensors"
    save_graph(graph, path, extra={"domain": "image-fixture"})
    metadata = inspect_dabsn(path)
    assert metadata["manifest"]["construction"] == "raw-graph"
    restored = load_graph(path)
    torch.testing.assert_close(restored(value), expected, atol=0, rtol=0)
    with pytest.raises(TypeError, match="load_graph"):
        load_dabsn(path)


def test_raw_graph_builds_directly_from_minimal_provider_json():
    graph = build_graph_from_config(
        {
            "model_kind": "graph",
            "components": [
                {
                    "component_id": "dabsn.0",
                    "provider_key": "dabsn:block",
                    "config": {
                        "input_dim": 6,
                        "hidden_dim": 6,
                        "state_dim": 8,
                        "read_geometry": "seq",
                        "residual": True,
                    },
                },
                {
                    "component_id": "mlp.0",
                    "provider_key": "dabsn:residual_mlp",
                    "config": {"dim": 6, "ratio": 2.0},
                },
            ],
            "require_world_builder": True,
        }
    )
    output = graph(torch.randn(2, 5, 6))
    assert output.shape == (2, 5, 6)
    assert graph.require_world_builder is True


def test_v1_loads_and_migration_writes_a_distinct_v2_artifact(tmp_path):
    model = _legacy_model()
    ids = torch.randint(0, model.vocab, (2, 5))
    expected = model.forward_sequence(ids)
    source = tmp_path / "source-v1.safetensors"
    destination = tmp_path / "migrated-v2.safetensors"
    _write_v1(model, source)
    source_digest = artifact_digest(source)
    assert inspect_dabsn(source)["manifest"]["construction"] == "v1-migration-view"
    torch.testing.assert_close(load_dabsn(source).forward_sequence(ids), expected, atol=0, rtol=0)
    digest = migrate_dabsn_checkpoint(source, destination)
    assert artifact_digest(source) == source_digest
    assert digest == artifact_digest(destination)
    assert inspect_dabsn(destination)["version"] == 2
    with pytest.raises(ValueError, match="never rewrites"):
        migrate_dabsn_checkpoint(source, source)


def test_inspection_only_parses_provider_names_and_never_resolves_them(tmp_path, monkeypatch):
    model = _portable_graph_model()
    path = tmp_path / "inspect.safetensors"
    save_dabsn(model, path)
    monkeypatch.setattr(
        component_registry,
        "resolve",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("executed")),
    )
    assert inspect_dabsn(path)["manifest"]["graph"][0]["provider_key"] == "dabsn:block"


def test_metadata_versions_must_agree(tmp_path):
    path = tmp_path / "bad-version.safetensors"
    save_file(
        {"x": torch.ones(1)},
        str(path),
        metadata={
            "format": CHECKPOINT_FORMAT,
            "version": "2",
            "config": "{}",
            "extra": "{}",
            "shared": "{}",
            "manifest": json.dumps({"schema_version": 1}),
        },
    )
    with pytest.raises(ValueError, match="versions disagree"):
        inspect_dabsn(path)


def test_adversarial_deep_metadata_is_rejected(tmp_path):
    value = None
    for _ in range(40):
        value = [value]
    path = tmp_path / "deep.safetensors"
    save_file(
        {"x": torch.ones(1)},
        str(path),
        metadata={
            "format": CHECKPOINT_FORMAT,
            "version": "1",
            "config": json.dumps({"layers": value}),
            "extra": "{}",
            "shared": "{}",
        },
    )
    with pytest.raises(ValueError, match="invalid DABSN metadata"):
        inspect_dabsn(path)


def test_graph_reconstruction_discovers_before_authorizing_plugins(monkeypatch):
    from dabsn.checkpoint import build_dabsn_from_checkpoint_config

    calls = []
    original_discover = component_registry.discover
    original_authorize = component_registry.authorize

    def discover():
        calls.append("discover")
        return original_discover()

    def authorize(key):
        calls.append(f"authorize:{key}")
        return original_authorize(key)

    monkeypatch.setattr(component_registry, "discover", discover)
    monkeypatch.setattr(component_registry, "authorize", authorize)
    build_dabsn_from_checkpoint_config(
        {"model_kind": "sequence_graph", "vocab": 31, "hidden_dim": 6},
        graph_components=[
            {
                "component_id": "dabsn.0",
                "provider_key": "dabsn:block",
                "provider_distribution": "dabsn",
                "provider_version": "2.0.0",
                "component_abi_version": 2,
                "config_schema_version": 1,
                "config": {
                    "input_dim": 6,
                    "hidden_dim": 6,
                    "state_dim": 6,
                    "read_geometry": "seq",
                    "residual": False,
                },
            }
        ],
        trusted_providers=["dabsn:block"],
    )
    assert calls[:2] == ["discover", "authorize:dabsn:block"]
