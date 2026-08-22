"""Native standalone residual MLP tower between configured DABSN blocks."""

import copy

import pytest
import torch

from dabsn import (
    DABSNConfig,
    DABSNLayerSpec,
    DABSNModel,
    DABSNSequenceLM,
    dabsn_adamw_param_groups,
    inspect_dabsn,
    load_dabsn,
    save_dabsn,
)
from dabsn.checkpoint import build_dabsn_from_checkpoint_config
from dabsn.config import DABSNPretrainConfig
from dabsn.memory import forward_with_memory, ingest, memory_cost
from dabsn.pretrain import _validate_model_config


def _model(*, middle_depth: int = 3, depth_index: int = 0):
    torch.manual_seed(41)
    return DABSNSequenceLM(
        vocab=37,
        hidden_dim=6,
        depth=3,
        layers=[
            DABSNLayerSpec(8, 7, "seq"),
            DABSNLayerSpec(8, 6, "seq"),
            DABSNLayerSpec(5, 7, "seq"),
        ],
        tie_embeddings=False,
        residual=True,
        mlp_ratio=2.0,
        mlp_middle_depth=middle_depth,
        mlp_depth_index=depth_index,
    )


def _move_middle_tower(model):
    with torch.no_grad():
        for mlp in model.backbone.middle_mlps:
            mlp.fc2.weight.normal_(mean=0.0, std=0.02)


def test_middle_tower_runs_once_after_the_selected_dabsn_block():
    model = _model(middle_depth=2, depth_index=1).eval()
    order = []
    hooks = []
    for index, block in enumerate(model.backbone.blocks):
        hooks.append(
            block.register_forward_pre_hook(
                lambda _module, _args, index=index: order.append(f"dabsn-{index}")
            )
        )
    for index, mlp in enumerate(model.backbone.middle_mlps):
        hooks.append(
            mlp.register_forward_pre_hook(
                lambda _module, _args, index=index: order.append(f"mlp-{index}")
            )
        )
    try:
        model.forward_sequence(torch.randint(0, model.vocab, (2, 5)))
    finally:
        for hook in hooks:
            hook.remove()

    assert order == ["dabsn-0", "dabsn-1", "mlp-0", "mlp-1", "dabsn-2"]


def test_middle_tower_uses_boundary_width_and_canonical_branch_shape():
    model = _model(middle_depth=2, depth_index=0)
    assert len(model.backbone.blocks) == 3
    assert len(model.backbone.middle_mlps) == 2
    for mlp in model.backbone.middle_mlps:
        assert mlp.dim == 8
        assert mlp.fc1.weight.shape == (16, 8)
        assert mlp.fc1.bias is None
        assert mlp.fc2.weight.shape == (8, 16)
        assert mlp.fc2.bias is None
        assert torch.count_nonzero(mlp.fc2.weight) == 0


def test_zero_initialized_middle_tower_is_initially_exact_identity():
    model = _model().eval()
    without_tower = copy.deepcopy(model)
    without_tower.backbone.middle_mlps = torch.nn.ModuleList()
    ids = torch.randint(0, model.vocab, (2, 7))
    torch.testing.assert_close(
        model.forward_sequence(ids),
        without_tower.forward_sequence(ids),
        atol=0,
        rtol=0,
    )


def test_disabled_defaults_leave_the_old_model_state_and_output_unchanged():
    torch.manual_seed(42)
    old = DABSNSequenceLM(vocab=37, hidden_dim=8, depth=2, mlp_ratio=2.0)
    torch.manual_seed(42)
    explicit = DABSNSequenceLM(
        vocab=37,
        hidden_dim=8,
        depth=2,
        mlp_ratio=2.0,
        mlp_middle_depth=0,
        mlp_depth_index=99,
    )
    assert not explicit.backbone.middle_mlps
    assert old.state_dict().keys() == explicit.state_dict().keys()
    ids = torch.randint(0, 37, (2, 6))
    torch.testing.assert_close(old(ids), explicit(ids), atol=0, rtol=0)


def test_middle_tower_parameters_train_after_zero_initialized_output_moves():
    model = _model(middle_depth=2).train()
    ids = torch.randint(0, model.vocab, (2, 7))
    model(ids).square().mean().backward()
    for mlp in model.backbone.middle_mlps:
        assert mlp.fc2.weight.grad is not None
        assert float(mlp.fc2.weight.grad.norm()) > 0
        assert torch.count_nonzero(mlp.fc1.weight.grad) == 0

    _move_middle_tower(model)
    model.zero_grad(set_to_none=True)
    model(ids).square().mean().backward()
    for mlp in model.backbone.middle_mlps:
        for parameter in (mlp.norm.weight, mlp.fc1.weight, mlp.fc2.weight):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert float(parameter.grad.norm()) > 0


def test_compiled_and_gradient_checkpointed_middle_tower_preserve_gradients():
    model = _model(middle_depth=2).train()
    model.backbone.grad_checkpoint = True
    _move_middle_tower(model)
    ids = torch.randint(0, model.vocab, (2, 7))
    logits = torch.compile(model.forward_sequence, backend="aot_eager", dynamic=False)(ids)
    logits.square().mean().backward()
    for mlp in model.backbone.middle_mlps:
        for parameter in (mlp.norm.weight, mlp.fc1.weight, mlp.fc2.weight):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_optimizer_groups_classify_middle_matrices_and_norm_correctly():
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq"), DABSNLayerSpec(7, 6, "seq")],
        output_adapter="token",
        mlp_ratio=2.0,
        mlp_middle_depth=2,
        mlp_depth_index=0,
    )
    groups = dabsn_adamw_param_groups(model, 0.1)
    decay = {id(parameter) for parameter in groups[0]["params"]}
    no_decay = {id(parameter) for parameter in groups[1]["params"]}
    for mlp in model.backbone.middle_mlps:
        assert id(mlp.fc1.weight) in decay
        assert id(mlp.fc2.weight) in decay
        assert id(mlp.norm.weight) in no_decay


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mlp_middle_depth": -1}, "mlp_middle_depth"),
        ({"mlp_middle_depth": 1}, "mlp_ratio"),
        (
            {"depth": 1, "mlp_ratio": 2.0, "mlp_middle_depth": 1},
            "at least two",
        ),
        (
            {
                "depth": 2,
                "mlp_ratio": 2.0,
                "mlp_middle_depth": 1,
                "mlp_depth_index": 1,
            },
            "before the final",
        ),
    ],
)
def test_configuration_rejects_invalid_middle_towers(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DABSNConfig(input_dim=4, out_dim=3, **kwargs)


def test_checkpoint_roundtrip_preserves_middle_tower(tmp_path):
    model = _model(middle_depth=3, depth_index=1).eval()
    _move_middle_tower(model)
    ids = torch.randint(0, model.vocab, (2, 7))
    expected = model.forward_sequence(ids)
    path = tmp_path / "middle-mlp.safetensors"

    save_dabsn(model, path)
    config = inspect_dabsn(path)["config"]
    assert config["mlp_middle_depth"] == 3
    assert config["mlp_depth_index"] == 1
    restored = load_dabsn(path)
    assert restored.mlp_middle_depth == 3
    assert restored.mlp_depth_index == 1
    torch.testing.assert_close(restored.forward_sequence(ids), expected, atol=0, rtol=0)


def test_legacy_checkpoint_config_disables_middle_tower():
    model = build_dabsn_from_checkpoint_config(
        {
            "model_kind": "sequence_lm",
            "vocab": 31,
            "hidden_dim": 6,
            "depth": 2,
            "layers": [
                {"hidden_dim": 6, "state_dim": 5, "read_geometry": "seq"},
                {"hidden_dim": 6, "state_dim": 5, "read_geometry": "seq"},
            ],
            "mlp_ratio": 2.0,
        }
    )
    assert model.mlp_middle_depth == 0
    assert model.mlp_depth_index == 0
    assert not model.backbone.middle_mlps


@pytest.mark.parametrize("chunk_size", [1, 3, 9])
def test_carried_memory_replays_tower_and_keeps_one_state_per_dabsn(chunk_size):
    model = _model(middle_depth=3, depth_index=0).eval()
    _move_middle_tower(model)
    ids = torch.randint(0, model.vocab, (2, 9))
    reference = model.forward_sequence(ids)
    logits, memory = forward_with_memory(model, ids, chunk_size=chunk_size, extend=True)
    torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-4)
    assert len(memory.layers) == len(model.backbone.blocks) == 3
    assert memory.fingerprint["mlp_middle_depth"] == 3
    assert memory.fingerprint["mlp_depth_index"] == 0


@pytest.mark.parametrize(("depth", "depth_index"), [(2, 0), (5, 2)])
def test_memory_has_one_layer_per_dabsn_never_one_per_middle_mlp(depth, depth_index):
    model = DABSNSequenceLM(
        vocab=31,
        hidden_dim=6,
        depth=depth,
        layer_geometries="seq",
        residual=True,
        mlp_ratio=2.0,
        mlp_middle_depth=4,
        mlp_depth_index=depth_index,
    ).eval()
    memory = ingest(model, torch.randint(0, 31, (1, 5)), chunk_size=2)
    assert len(memory.layers) == depth
    assert len(model.backbone.middle_mlps) == 4


@pytest.mark.parametrize(
    "change",
    [
        {"mlp_middle_depth": 2, "mlp_depth_index": 0},
        {"mlp_middle_depth": 3, "mlp_depth_index": 1},
    ],
)
def test_memory_fingerprint_rejects_a_different_middle_layout(change):
    model = _model(middle_depth=3, depth_index=0).eval()
    memory = ingest(model, torch.randint(0, model.vocab, (2, 5)))
    other = _model(
        **{
            "middle_depth": change["mlp_middle_depth"],
            "depth_index": change["mlp_depth_index"],
        }
    ).eval()
    with pytest.raises(ValueError, match="different architecture"):
        forward_with_memory(other, torch.randint(0, model.vocab, (2, 2)), memory)


def test_legacy_memory_fingerprint_defaults_both_middle_fields_to_zero():
    model = DABSNSequenceLM(vocab=31, hidden_dim=6, depth=2).eval()
    memory = ingest(model, torch.randint(0, 31, (1, 5)))
    memory.fingerprint.pop("mlp_middle_depth")
    memory.fingerprint.pop("mlp_depth_index")
    forward_with_memory(model, torch.randint(0, 31, (1, 2)), memory)


def test_middle_mlp_count_does_not_change_dmem_cost_or_actual_bank_bytes():
    common = dict(
        vocab=31,
        hidden_dim=6,
        depth=2,
        layer_geometries="seq",
        residual=True,
        mlp_ratio=2.0,
    )
    torch.manual_seed(47)
    plain = DABSNSequenceLM(**common).eval()
    torch.manual_seed(47)
    tower = DABSNSequenceLM(**common, mlp_middle_depth=7, mlp_depth_index=0).eval()
    ids = torch.randint(0, 31, (1, 6))
    plain_memory = ingest(plain, ids)
    tower_memory = ingest(tower, ids)
    assert memory_cost(plain, 6, batch_size=1) == memory_cost(tower, 6, batch_size=1)
    assert plain_memory.nbytes() == tower_memory.nbytes()


def test_resume_validation_includes_both_middle_tower_fields(tmp_path):
    corpus = tmp_path / "tokens.bin"
    corpus.write_bytes(bytes(range(32)))
    common = dict(
        corpus_bin=str(corpus),
        corpus_dtype="uint8",
        vocab=37,
        hidden_dim=6,
        depth=3,
        layers="seq:8:7,seq:8:6,seq:5:7",
        tie_embeddings=False,
        train_context=4,
        steps=1,
        batch_size=1,
        eval_batch_size=1,
        distributed="none",
        grad_accum_steps=1,
        residual=True,
        mlp_ratio=2.0,
    )
    model = _model(middle_depth=3, depth_index=1)
    _validate_model_config(
        model,
        DABSNPretrainConfig(**common, mlp_middle_depth=3, mlp_depth_index=1),
    )
    with pytest.raises(ValueError, match="mlp_middle_depth"):
        _validate_model_config(
            model,
            DABSNPretrainConfig(**common, mlp_middle_depth=2, mlp_depth_index=1),
        )
    with pytest.raises(ValueError, match="mlp_depth_index"):
        _validate_model_config(
            model,
            DABSNPretrainConfig(**common, mlp_middle_depth=3, mlp_depth_index=0),
        )
