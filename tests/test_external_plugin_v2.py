import importlib
import sys
from pathlib import Path

import torch

from dabsn import (
    ComponentRegistry,
    ComponentSpec,
    DABSNGraph,
    DABSNSequenceLM,
    component_registry,
    load_dabsn,
    load_graph,
    save_dabsn,
    save_graph,
)
from dabsn.conformance import check_component

FIXTURE = Path(__file__).parent / "fixtures/dabsn_external_fixture/src"


def _providers():
    sys.path.insert(0, str(FIXTURE))
    try:
        package = importlib.import_module("dabsn_external_fixture")
    finally:
        sys.path.remove(str(FIXTURE))
    return package


def test_external_package_registers_custom_op_and_composes_without_core_edit():
    package = _providers()
    registry = ComponentRegistry()
    registry.register(package.ScaleProvider(), distribution="fixture", version="1.0.0")
    binding = registry.build(
        ComponentSpec("external.0", "fixture.scale", {"width": 8, "scale": 1.5})
    )
    graph = DABSNGraph([binding])
    value = torch.randn(2, 4, 8, requires_grad=True)
    output = graph(value)
    torch.testing.assert_close(output, value * 1.5)
    output.sum().backward()
    torch.library.opcheck(torch.ops.dabsn_fixture.scale.default, (value.detach(), 1.5))


def test_stateless_external_kernel_graph_is_a_portable_artifact(tmp_path):
    package = _providers()
    key = "fixture.scale"
    if key not in component_registry.keys():
        component_registry.register(
            package.ScaleProvider(),
            distribution="dabsn-external-fixture",
            version="1.0.0",
        )
    graph = DABSNGraph(
        [
            component_registry.build(
                ComponentSpec("external.scale", key, {"width": 8, "scale": 1.5})
            )
        ]
    )
    value = torch.randn(2, 4, 8)
    path = tmp_path / "stateless-external.safetensors"
    save_graph(graph, path)
    restored = load_graph(path, trusted_providers=[key])
    assert restored.state_dict() == {}
    torch.testing.assert_close(restored(value), value * 1.5)


def test_h_native_attention_cnn_and_transformer_are_only_provider_code():
    package = _providers()
    registry = ComponentRegistry()
    providers = [
        (package.HAttentionProvider(), {"width": 8, "latent": 4, "heads": 2}),
        (package.CNNProvider(), {"width": 8, "kernel": 3}),
        (package.TransformerProvider(), {"width": 8, "heads": 2, "inner": 16}),
    ]
    bindings = []
    for index, (provider, config) in enumerate(providers):
        registry.register(provider, distribution="fixture", version="1.0.0")
        bindings.append(
            registry.build(ComponentSpec(f"external.{index}", provider.provider_key, config))
        )
    graph = DABSNGraph(bindings)
    value = torch.randn(2, 5, 8, requires_grad=True)
    output = graph(value)
    assert output.shape == value.shape
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_external_architecture_providers_pass_public_conformance():
    package = _providers()
    providers = [
        (package.ScaleProvider(), {"width": 8, "scale": 1.5}),
        (package.HAttentionProvider(), {"width": 8, "latent": 4, "heads": 2}),
        (package.CNNProvider(), {"width": 8, "kernel": 3}),
        (package.TransformerProvider(), {"width": 8, "heads": 2, "inner": 16}),
    ]
    for provider, config in providers:
        if provider.provider_key not in component_registry.keys():
            component_registry.register(
                provider,
                distribution="dabsn-external-fixture",
                version="1.0.0",
            )
        report = check_component(provider.provider_key, config)
        failures = [check for check in report.checks if check.status == "fail"]
        assert not failures, failures


def test_entry_point_manifest_exposes_stable_provider_names():
    text = (FIXTURE.parent / "pyproject.toml").read_text(encoding="utf-8")
    for key in (
        "fixture.scale",
        "fixture.h_attention",
        "fixture.cnn",
        "fixture.transformer",
        "fixture.world_transformer_expert",
    ):
        assert key in text


def test_whole_transformer_experts_are_portable_nested_provider_specs():
    package = _providers()
    key = "fixture.world_transformer_expert"
    if key not in component_registry.keys():
        component_registry.register(
            package.WorldTransformerExpertProvider(),
            distribution="dabsn-external-fixture",
            version="1.0.0",
        )
    expert_specs = [
        {
            "provider_key": key,
            "provider_distribution": "dabsn-external-fixture",
            "provider_version": "1.0.0",
            "component_abi_version": 2,
            "config_schema_version": 1,
            "config": {"width": 8, "latent": 4, "heads": 2, "inner": 16},
        }
        for _ in range(3)
    ]
    binding = component_registry.build(
        ComponentSpec(
            "moe.transformer",
            "dabsn:sparse_moe",
            {
                "hidden_dim": 8,
                "experts": 3,
                "top_k": 2,
                "router": "switch",
                "balance_coefficient": 0.01,
                "normalization": "none",
                "residual": True,
                "routing_granularity": "individual_h",
                "expert_specs": expert_specs,
            },
        )
    )
    value = torch.randn(2, 5, 8, requires_grad=True)
    result = binding.module.forward_with_terms(value)
    assert result.value.shape == value.shape
    result.value.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert binding.to_spec().config["expert_specs"] == expert_specs


def test_nested_transformer_expert_checkpoint_reconstructs_exactly(tmp_path):
    package = _providers()
    key = "fixture.world_transformer_expert"
    if key not in component_registry.keys():
        component_registry.register(
            package.WorldTransformerExpertProvider(),
            distribution="dabsn-external-fixture",
            version="1.0.0",
        )
    expert_specs = [
        {
            "provider_key": key,
            "provider_distribution": "dabsn-external-fixture",
            "provider_version": "1.0.0",
            "component_abi_version": 2,
            "config_schema_version": 1,
            "config": {"width": 8, "latent": 4, "heads": 2, "inner": 16},
        }
        for _ in range(2)
    ]
    first = component_registry.build(
        ComponentSpec(
            "dabsn.0",
            "dabsn:block",
            {
                "input_dim": 8,
                "hidden_dim": 8,
                "state_dim": 8,
                "read_geometry": "seq",
                "residual": True,
            },
        )
    )
    moe = component_registry.build(
        ComponentSpec(
            "moe.0",
            "dabsn:sparse_moe",
            {
                "hidden_dim": 8,
                "experts": 2,
                "top_k": 1,
                "router": "switch",
                "balance_coefficient": 0.01,
                "normalization": "none",
                "residual": True,
                "routing_granularity": "individual_h",
                "expert_specs": expert_specs,
            },
        )
    )
    model = DABSNSequenceLM.from_graph(
        DABSNGraph([first, moe], require_world_builder=True), vocab=23
    ).eval()
    ids = torch.randint(0, 23, (2, 4))
    expected = model.forward_sequence(ids)
    path = tmp_path / "external-experts.safetensors"
    save_dabsn(model, path)
    restored = load_dabsn(path, trusted_providers=[key]).eval()
    torch.testing.assert_close(restored.forward_sequence(ids), expected, atol=0, rtol=0)


def test_external_nonlanguage_graph_checkpoint_needs_no_core_edit(tmp_path):
    package = _providers()
    providers = [
        (package.CNNProvider(), {"width": 8, "kernel": 3}),
        (package.HAttentionProvider(), {"width": 8, "latent": 4, "heads": 2}),
    ]
    bindings = []
    for index, (provider, config) in enumerate(providers):
        if provider.provider_key not in component_registry.keys():
            component_registry.register(
                provider,
                distribution="dabsn-external-fixture",
                version="1.0.0",
            )
        bindings.append(
            component_registry.build(
                ComponentSpec(f"external.domain.{index}", provider.provider_key, config)
            )
        )
    graph = DABSNGraph(bindings)
    value = torch.randn(2, 6, 8)
    expected = graph(value)
    path = tmp_path / "external-domain-graph.safetensors"
    save_graph(graph, path)
    restored = load_graph(
        path,
        trusted_providers=[provider.provider_key for provider, _ in providers],
    )
    torch.testing.assert_close(restored(value), expected, atol=0, rtol=0)
