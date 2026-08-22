import pytest
import torch.nn as nn

from dabsn.components import (
    AnonymousComponentError,
    AxisContract,
    BuildContext,
    ComponentCapabilities,
    ComponentContract,
    ComponentProvider,
    ComponentRegistry,
    ComponentSpec,
    IncompatibleProviderError,
    MissingProviderError,
    UntrustedProviderError,
    ValueContract,
    bind_module,
)


def _world(width=8):
    return ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("experience", "T", dynamic=True),
        AxisContract("world", width),
    )


def test_value_contract_is_immutable_and_fingerprinted_canonically():
    first = _world(8)
    second = _world(8)
    assert first == second
    assert first.fingerprint() == second.fingerprint()
    with pytest.raises(Exception):
        first.leaves = ()


def test_value_contract_reports_the_exact_incompatible_axis():
    expected = _world(8)
    received = _world(16)
    errors = expected.incompatibilities(received)
    assert errors == ("leaf 0 axis 'world' expected size 8, received 16",)


def test_direct_module_binding_is_zero_value_wrapper_and_nonportable():
    module = nn.Linear(8, 8, bias=False)
    contract = ComponentContract(_world(8), _world(8))
    binding = bind_module("research.linear", module, contract)
    assert binding.module is module
    with pytest.raises(AnonymousComponentError, match="anonymous"):
        binding.to_spec()


class _Provider:
    provider_key = "tests:linear"
    component_abi_version = 2
    config_schema_version = 2
    capabilities = ComponentCapabilities(compile_fullgraph=True)

    def validate_config(self, config):
        if int(config["width"]) <= 0:
            raise ValueError("width")

    def contract(self, config):
        world = _world(int(config["width"]))
        return ComponentContract(world, world)

    def build(self, config, context):
        return nn.Linear(int(config["width"]), int(config["width"]), bias=False)

    def migrate_config(self, old_version, config):
        assert old_version == 1
        return {"width": int(config["hidden"])}


def test_provider_protocol_and_schema_migration_build_portable_binding():
    registry = ComponentRegistry()
    provider = _Provider()
    assert isinstance(provider, ComponentProvider)
    registry.register(provider, distribution="fixture", version="1.2.3")
    binding = registry.build(
        ComponentSpec(
            "linear.0",
            "tests:linear",
            {"hidden": 8},
            config_schema_version=1,
        ),
        BuildContext(),
    )
    assert binding.portable
    assert binding.config == {"width": 8}
    assert binding.provider_distribution == "fixture"
    assert binding.provider_version == "1.2.3"


def test_provider_failures_are_typed_and_distinct():
    registry = ComponentRegistry()
    with pytest.raises(MissingProviderError, match="missing"):
        registry.resolve("tests:missing")
    registry.register(_Provider(), trusted=False)
    with pytest.raises(UntrustedProviderError, match="untrusted"):
        registry.resolve("tests:linear")

    class Old(_Provider):
        provider_key = "tests:old"
        component_abi_version = 1

    with pytest.raises(IncompatibleProviderError, match="required 2"):
        registry.register(Old())


def test_provider_distribution_and_version_must_match_checkpoint_identity():
    registry = ComponentRegistry()
    registry.register(_Provider(), distribution="fixture", version="1.2.3")
    with pytest.raises(IncompatibleProviderError, match="distribution"):
        registry.build(
            ComponentSpec(
                "wrong-dist",
                "tests:linear",
                {"width": 8},
                config_schema_version=2,
                provider_distribution="different",
                provider_version="1.2.3",
            )
        )
    with pytest.raises(IncompatibleProviderError, match="requires version"):
        registry.build(
            ComponentSpec(
                "wrong-version",
                "tests:linear",
                {"width": 8},
                config_schema_version=2,
                provider_distribution="fixture",
                provider_version="9.0.0",
            )
        )


def test_contract_supports_structure_native_and_h_native_regimes():
    temporal = _world(8)
    h_native = ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("world", 8),
        AxisContract("latent", 4),
    )
    image = ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("spatial:y", "Y", dynamic=True),
        AxisContract("spatial:x", "X", dynamic=True),
        AxisContract("world", 8),
    )
    assert temporal != h_native != image
    assert temporal.leaves[0].axes[1].name == "experience"
    assert h_native.leaves[0].axes[1].name == "world"
    assert image.leaves[0].axes[1].name == "spatial:y"


def test_provider_config_is_bounded_before_provider_build():
    nested = {}
    cursor = nested
    for _ in range(34):
        child = {}
        cursor["child"] = child
        cursor = child
    registry = ComponentRegistry()
    registry.register(_Provider())
    with pytest.raises(ValueError, match="nesting limit"):
        registry.build(
            ComponentSpec(
                "too-deep",
                "tests:linear",
                {"width": 4, "nested": nested},
                config_schema_version=2,
            )
        )


def test_provider_config_rejects_noncanonical_values():
    registry = ComponentRegistry()
    registry.register(_Provider())
    with pytest.raises(ValueError, match="unsupported type"):
        registry.build(
            ComponentSpec(
                "tuple-config",
                "tests:linear",
                {"width": 4, "bad": (1, 2)},
                config_schema_version=2,
            )
        )


def test_builtin_provider_rejects_misspelled_architecture_fields():
    from dabsn import component_registry

    with pytest.raises(ValueError, match="unknown dabsn:residual_mlp"):
        component_registry.build(
            ComponentSpec(
                "typo",
                "dabsn:residual_mlp",
                {"dim": 8, "ratio": 2.0, "resdiual": True},
            )
        )
