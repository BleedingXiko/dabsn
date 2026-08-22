import copy

import torch

from dabsn import DABSNGraph, DABSNLayerSpec, DABSNSequenceLM


def _model():
    torch.manual_seed(2020)
    return DABSNSequenceLM(
        vocab=37,
        hidden_dim=6,
        depth=3,
        layers=[
            DABSNLayerSpec(8, 7, "seq"),
            DABSNLayerSpec(8, 6, "seq"),
            DABSNLayerSpec(5, 7, "seq"),
        ],
        residual=True,
        mlp_ratio=2.0,
        mlp_middle_depth=2,
        mlp_depth_index=1,
    )


def test_legacy_constructor_has_one_ordered_graph_as_its_executor():
    model = _model()
    assert isinstance(model.backbone.graph, DABSNGraph)
    assert model.backbone.graph.component_ids == (
        "dabsn.0",
        "legacy-inline-mlp.0",
        "dabsn.1",
        "legacy-inline-mlp.1",
        "legacy-middle-mlp.0",
        "legacy-middle-mlp.1",
        "dabsn.2",
        "legacy-inline-mlp.2",
    )


def test_graph_cutover_preserves_legacy_parameter_names_and_count():
    model = _model()
    names = tuple(model.state_dict())
    assert any(name.startswith("backbone.blocks.0.core.") for name in names)
    assert "backbone.blocks.0.mlp_fc1.weight" in names
    assert "backbone.middle_mlps.0.fc1.weight" in names
    assert not any("backbone.graph" in name for name in names)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3227


def test_graph_execution_matches_explicit_historical_order_exactly():
    model = _model().eval()
    reference = copy.deepcopy(model).eval()
    ids = torch.randint(0, model.vocab, (2, 7))

    hidden = reference.embed(ids)
    for index, block in enumerate(reference.backbone.blocks):
        hidden = block(hidden)
        if index == reference.backbone.mlp_depth_index:
            for mlp in reference.backbone.middle_mlps:
                hidden = mlp(hidden)
    expected = reference.readout(hidden)
    actual = model.forward_sequence(ids)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_legacy_model_compiles_fullgraph_with_forward_and_backward_parity():
    eager = _model()
    compiled_model = copy.deepcopy(eager)
    ids = torch.randint(0, eager.vocab, (2, 5))

    expected = eager.forward_sequence(ids)
    expected.square().mean().backward()
    compiled = torch.compile(
        compiled_model.forward_sequence,
        backend="aot_eager",
        fullgraph=True,
    )
    actual = compiled(ids)
    actual.square().mean().backward()

    torch.testing.assert_close(actual, expected)
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        eager.named_parameters(),
        compiled_model.named_parameters(),
    ):
        assert actual_name == expected_name
        if expected_parameter.grad is None:
            assert actual_parameter.grad is None
        else:
            assert actual_parameter.grad is not None
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


def test_graph_memory_count_tracks_only_dabsn_components():
    model = _model()
    assert model.backbone.graph.dabsn_memory_count == 3
    assert model.backbone.graph.dabsn_memory_count == len(model.backbone.blocks)


def test_backbone_components_are_registered_once_under_legacy_names():
    model = _model()
    parameters = list(model.named_parameters())
    assert len(parameters) == len({id(parameter) for _, parameter in parameters})
    assert all("graph.components" not in name for name, _ in parameters)


def test_canonical_5x768_20mlp_compatibility_identity():
    model = DABSNSequenceLM(
        vocab=50_257,
        hidden_dim=768,
        depth=5,
        layers=[DABSNLayerSpec(768, 768, "seq") for _ in range(5)],
        tie_embeddings=False,
        residual=True,
        mlp_ratio=4.0,
        mlp_middle_depth=20,
        mlp_depth_index=1,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 215_938_249
    assert model.graph.dabsn_memory_count == 5
    assert model.readout.weight is not model.embed.weight
    names = tuple(model.state_dict())
    assert "backbone.blocks.4.core.W.weight" in names
    assert "backbone.middle_mlps.19.fc2.weight" in names
    assert not any("backbone.graph" in name for name in names)
