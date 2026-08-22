"""Canonical stack residual and optional post-DABSN MLP contract."""

import copy

import pytest
import torch
import torch.nn.functional as F

from dabsn import (
    DABSNBlock,
    DABSNConfig,
    DABSNLayerSpec,
    DABSNModel,
    DABSNSequenceLM,
    DABSNTaskModel,
    dabsn_adamw_param_groups,
    inspect_dabsn,
    load_dabsn,
    save_dabsn,
)
from dabsn.checkpoint import build_dabsn_from_checkpoint_config
from dabsn.config import DABSNPretrainConfig
from dabsn.model import build_dabsn_from_config
from dabsn.pretrain import _validate_model_config


def _plain_copy(block: DABSNBlock) -> DABSNBlock:
    plain = DABSNBlock(
        block.input_dim,
        block.hidden_dim,
        block.state_dim,
        block.read_geometry,
    )
    state = {
        name: value
        for name, value in block.state_dict().items()
        if not name.startswith(("residual_skip.", "mlp_norm.", "mlp_fc1.", "mlp_fc2."))
    }
    missing, unexpected = plain.load_state_dict(state, strict=False)
    assert not missing and not unexpected
    return plain


def test_residual_without_mlp_is_pure_dabsn_plus_identity_skip():
    torch.manual_seed(11)
    block = DABSNBlock(6, 6, 5, "seq", residual=True, mlp_ratio=None)
    plain = _plain_copy(block)
    inputs = torch.randn(2, 7, 6)

    expected = inputs + plain(inputs)
    actual = block(inputs)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert block.residual_skip is None
    assert block.mlp_norm is None
    assert block.mlp_fc1 is None
    assert block.mlp_fc2 is None


def test_width_changing_residual_uses_the_prototype_projection_initialization():
    torch.manual_seed(12)
    block = DABSNBlock(5, 8, 7, "seq", residual=True)
    plain = _plain_copy(block)
    inputs = torch.randn(2, 6, 5)

    expected = block.residual_skip(inputs) + plain(inputs)
    torch.testing.assert_close(block(inputs), expected, atol=0, rtol=0)
    assert block.residual_skip.bias is None
    observed_std = float(block.residual_skip.weight.detach().std(unbiased=False))
    assert observed_std == pytest.approx(5**-0.5, rel=0.35)


def test_mlp_is_post_dabsn_residual_and_fc2_zero_makes_it_initially_identity():
    torch.manual_seed(13)
    block = DABSNBlock(6, 6, 5, "seq", residual=True, mlp_ratio=2.0)
    plain = _plain_copy(block)
    inputs = torch.randn(2, 7, 6)

    dabsn_residual = inputs + plain(inputs)
    torch.testing.assert_close(block(inputs), dabsn_residual, atol=0, rtol=0)
    assert torch.count_nonzero(block.mlp_fc2.weight) == 0

    with torch.no_grad():
        block.mlp_fc2.weight.normal_(mean=0.0, std=0.02)
    branch = block.mlp_fc2(F.relu(block.mlp_fc1(block.mlp_norm(dabsn_residual))).square())
    torch.testing.assert_close(
        block(inputs),
        dabsn_residual + branch,
        atol=0,
        rtol=0,
    )


def test_mlp_norm_never_touches_the_dabsn_path():
    torch.manual_seed(14)
    block = DABSNBlock(6, 6, 5, "seq", residual=True, mlp_ratio=2.0)
    inputs = torch.randn(2, 7, 6)
    seen_core_inputs = []
    seen_mlp_inputs = []
    core_hook = block.core.register_forward_pre_hook(
        lambda _module, args: seen_core_inputs.append(args[0].detach().clone())
    )
    mlp_hook = block.mlp_norm.register_forward_pre_hook(
        lambda _module, args: seen_mlp_inputs.append(args[0].detach().clone())
    )
    try:
        block(inputs)
    finally:
        core_hook.remove()
        mlp_hook.remove()

    assert len(seen_core_inputs) == len(seen_mlp_inputs) == 1
    torch.testing.assert_close(seen_core_inputs[0], inputs, atol=0, rtol=0)
    assert not torch.equal(seen_mlp_inputs[0], inputs)


def test_mlp_parameters_receive_gradients_once_zero_initialized_fc2_moves():
    torch.manual_seed(15)
    block = DABSNBlock(6, 6, 5, "seq", residual=True, mlp_ratio=2.0)
    inputs = torch.randn(2, 7, 6)

    first_loss = block(inputs).square().mean()
    first_loss.backward()
    assert block.mlp_fc2.weight.grad is not None
    assert float(block.mlp_fc2.weight.grad.norm()) > 0
    assert torch.count_nonzero(block.mlp_fc1.weight.grad) == 0

    with torch.no_grad():
        block.mlp_fc2.weight.add_(-0.1 * block.mlp_fc2.weight.grad)
    block.zero_grad(set_to_none=True)
    block(inputs).square().mean().backward()
    for parameter in (
        block.mlp_norm.weight,
        block.mlp_fc1.weight,
        block.mlp_fc2.weight,
        block.residual_skip,
    ):
        if parameter is None:
            continue
        tensor = parameter if isinstance(parameter, torch.Tensor) else parameter.weight
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert float(tensor.grad.norm()) > 0


def test_public_constructors_thread_only_the_two_canonical_settings():
    config = DABSNConfig(
        input_dim=4,
        out_dim=3,
        layers=[DABSNLayerSpec(7, 6, "seq")],
        output_adapter="token",
        residual=True,
        mlp_ratio=2.0,
    )
    model = build_dabsn_from_config(config)
    block = model.backbone.blocks[0]
    assert block.residual is True
    assert block.mlp_ratio == 2.0

    lm = DABSNSequenceLM(
        vocab=31,
        hidden_dim=6,
        depth=1,
        layers=[DABSNLayerSpec(7, 6, "seq")],
        residual=True,
        mlp_ratio=3.0,
    )
    assert lm.backbone.blocks[0].residual is True
    assert lm.backbone.blocks[0].mlp_ratio == 3.0

    task = DABSNTaskModel(
        raw_input_dim=4,
        model_input_dim=6,
        out_dim=3,
        layers=[DABSNLayerSpec(7, 6, "seq")],
        input_adapter="linear",
        output_adapter="token",
        residual=True,
        mlp_ratio=2.0,
    )
    assert task.backbone.blocks[0].residual is True
    assert task.backbone.blocks[0].mlp_ratio == 2.0


def test_legacy_checkpoint_config_defaults_to_plain_blocks():
    model = build_dabsn_from_checkpoint_config(
        {
            "model_kind": "sequence_lm",
            "vocab": 31,
            "hidden_dim": 6,
            "depth": 1,
            "layers": [{"hidden_dim": 6, "state_dim": 5, "read_geometry": "seq"}],
        }
    )
    block = model.backbone.blocks[0]
    assert model.residual is False
    assert model.mlp_ratio is None
    assert block.residual_skip is None
    assert block.mlp_norm is None


def test_enabled_checkpoint_roundtrip_records_and_rebuilds_both_settings(tmp_path):
    torch.manual_seed(18)
    model = DABSNSequenceLM(
        vocab=31,
        hidden_dim=6,
        depth=2,
        layers=[
            DABSNLayerSpec(8, 7, "seq"),
            DABSNLayerSpec(5, 6, "seq"),
        ],
        residual=True,
        mlp_ratio=2.0,
    ).eval()
    for block in model.backbone.blocks:
        with torch.no_grad():
            block.mlp_fc2.weight.normal_(mean=0.0, std=0.02)
    tokens = torch.randint(0, 31, (2, 7))
    expected = model.forward_sequence(tokens)
    path = tmp_path / "residual-mlp.safetensors"

    save_dabsn(model, path)
    config = inspect_dabsn(path)["config"]
    assert config["residual"] is True
    assert config["mlp_ratio"] == 2.0
    restored = load_dabsn(path)
    assert restored.residual is True
    assert restored.mlp_ratio == 2.0
    torch.testing.assert_close(restored.forward_sequence(tokens), expected, atol=0, rtol=0)


def test_configuration_rejects_nonpositive_mlp_ratios():
    with pytest.raises(ValueError, match="mlp_ratio"):
        DABSNConfig(input_dim=4, out_dim=3, mlp_ratio=0)
    with pytest.raises(ValueError, match="mlp_ratio"):
        DABSNBlock(4, 4, 4, "seq", mlp_ratio=-1)


def test_optimizer_decay_policy_covers_new_parameter_kinds():
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq")],
        output_adapter="token",
        residual=True,
        mlp_ratio=2.0,
    )
    groups = dabsn_adamw_param_groups(model, 0.1)
    decayed = {id(parameter) for parameter in groups[0]["params"]}
    no_decay = {id(parameter) for parameter in groups[1]["params"]}
    block = model.backbone.blocks[0]
    assert id(block.residual_skip.weight) in decayed
    assert id(block.mlp_fc1.weight) in decayed
    assert id(block.mlp_fc2.weight) in decayed
    assert id(block.mlp_norm.weight) in no_decay


def test_resume_validation_includes_residual_and_mlp_architecture(tmp_path):
    corpus = tmp_path / "tokens.bin"
    corpus.write_bytes(bytes(range(32)))
    common = dict(
        corpus_bin=str(corpus),
        corpus_dtype="uint8",
        vocab=31,
        hidden_dim=6,
        depth=1,
        layer_geometries="seq",
        state_dim=5,
        train_context=4,
        steps=1,
        batch_size=1,
        eval_batch_size=1,
        distributed="none",
        grad_accum_steps=1,
    )
    model = DABSNSequenceLM(
        vocab=31,
        hidden_dim=6,
        depth=1,
        state_dim=5,
        residual=True,
        mlp_ratio=2.0,
        tie_embeddings=True,
    )
    _validate_model_config(
        model,
        DABSNPretrainConfig(**common, residual=True, mlp_ratio=2.0),
    )
    with pytest.raises(ValueError, match="residual"):
        _validate_model_config(model, DABSNPretrainConfig(**common))
    with pytest.raises(ValueError, match="mlp_ratio"):
        _validate_model_config(
            model,
            DABSNPretrainConfig(**common, residual=True, mlp_ratio=4.0),
        )


def test_compiled_enabled_stack_preserves_forward_and_gradients():
    torch.manual_seed(16)
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq"), DABSNLayerSpec(5, 6, "field")],
        output_adapter="token",
        residual=True,
        mlp_ratio=2.0,
    )
    for block in model.backbone.blocks:
        with torch.no_grad():
            block.mlp_fc2.weight.normal_(mean=0.0, std=0.02)
    reference = copy.deepcopy(model)
    eager_inputs = torch.randn(2, 6, 4, requires_grad=True)
    compiled_inputs = eager_inputs.detach().clone().requires_grad_(True)

    eager = reference.forward_sequence(eager_inputs)
    compiled = torch.compile(
        model.forward_sequence,
        backend="aot_eager",
        dynamic=False,
    )(compiled_inputs)
    eager.square().mean().backward()
    compiled.square().mean().backward()

    torch.testing.assert_close(compiled, eager)
    torch.testing.assert_close(compiled_inputs.grad, eager_inputs.grad)
    for (_, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_gradient_checkpointing_trains_enabled_width_changing_stack():
    torch.manual_seed(19)
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq"), DABSNLayerSpec(5, 6, "field")],
        output_adapter="token",
        grad_checkpoint=True,
        residual=True,
        mlp_ratio=2.0,
    ).train()
    for block in model.backbone.blocks:
        with torch.no_grad():
            block.mlp_fc2.weight.normal_(mean=0.0, std=0.02)
    inputs = torch.randn(2, 6, 4, requires_grad=True)
    model.forward_sequence(inputs).square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for block in model.backbone.blocks:
        for parameter in (
            block.residual_skip.weight,
            block.mlp_norm.weight,
            block.mlp_fc1.weight,
            block.mlp_fc2.weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph feature gate")
def test_cuda_graph_trains_enabled_residual_mlp_stack():
    from dabsn.read import DABSNRead
    from dabsn.runtime import make_graphed_train_callable

    torch.manual_seed(20)
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq"), DABSNLayerSpec(5, 6, "seq")],
        output_adapter="token",
        residual=True,
        mlp_ratio=2.0,
    ).cuda()
    for block in model.backbone.blocks:
        with torch.no_grad():
            block.mlp_fc2.weight.normal_(mean=0.0, std=0.02)
    for module in model.modules():
        if isinstance(module, DABSNRead):
            module._capture_safe_bank = True
    inputs = torch.randn(2, 16, 4, device="cuda")
    graphed = make_graphed_train_callable(model, (inputs,), verify=True)
    loss = graphed(inputs).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for block in model.backbone.blocks:
        assert block.mlp_fc2.weight.grad is not None
        assert torch.isfinite(block.mlp_fc2.weight.grad).all()
