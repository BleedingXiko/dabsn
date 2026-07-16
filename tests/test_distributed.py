import json
from pathlib import Path

import pytest
import torch

from dabsn import (
    DABSNLayerSpec,
    DABSNModel,
    DABSNSequenceLM,
    dabsn_adamw_param_groups,
    load_dabsn,
    load_sharded_training_checkpoint,
    prepare_distributed_model,
    save_dabsn,
    save_distributed_dabsn,
    save_sharded_training_checkpoint,
    setup_distributed,
)
from dabsn.cli import main as cli_main
from dabsn.distributed import (
    DistributedState,
    DABSNSequenceModule,
    _portable_state_dict,
    load_distributed_optimizer,
    optimizer_checkpoint_path,
    wrap_distributed,
)


def test_sequence_lm_checkpoint_roundtrip(tmp_path):
    model = DABSNSequenceLM(
        vocab=31,
        hidden_dim=12,
        depth=2,
        layers="seq:12:10,hybrid:12:10",
        tie_embeddings=True,
        grad_checkpoint=True,
    )
    tokens = torch.randint(0, 31, (2, 7))
    expected = model.forward_sequence(tokens)
    path = tmp_path / "lm.safetensors"
    from dabsn import save_dabsn

    save_dabsn(model, path)
    restored = load_dabsn(path)
    assert isinstance(restored, DABSNSequenceLM)
    assert restored.tie_embeddings
    assert restored.backbone.grad_checkpoint
    assert restored.readout.weight is restored.embed.weight
    assert torch.equal(restored.forward_sequence(tokens), expected)


def test_non_distributed_portable_training_checkpoint(tmp_path):
    state = setup_distributed("none", "cpu")
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq"), DABSNLayerSpec(8, 7, "field")],
        output_adapter="token",
        grad_checkpoint=True,
    )
    groups = dabsn_adamw_param_groups(model, 0.1)
    wrapped = prepare_distributed_model(model, state)
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    inputs = torch.randn(2, 6, 4)
    targets = torch.randint(0, 3, (2, 6))
    loss = torch.nn.functional.cross_entropy(
        wrapped(inputs).reshape(-1, 3),
        targets.reshape(-1),
    )
    loss.backward()
    optimizer.step()

    path = tmp_path / "distributed.safetensors"
    save_distributed_dabsn(wrapped, path, state, optimizer=optimizer, step=17)
    assert path.is_file()
    assert optimizer_checkpoint_path(path).is_file()
    restored = load_dabsn(path)
    assert restored.backbone.grad_checkpoint
    torch.testing.assert_close(
        restored.forward_sequence(inputs),
        model.forward_sequence(inputs),
        atol=0,
        rtol=0,
    )

    restored_groups = dabsn_adamw_param_groups(restored, 0.1)
    restored_wrapped = prepare_distributed_model(restored, state)
    restored_optimizer = torch.optim.AdamW(restored_groups, lr=1e-3)
    assert load_distributed_optimizer(
        restored_optimizer,
        restored_wrapped,
        path,
        state,
    ) == 17
    assert restored_optimizer.state_dict()["state"]

    sidecar = optimizer_checkpoint_path(path)
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    payload["extra"]["training_transaction"] = "wrong-transaction"
    torch.save(payload, sidecar)
    with pytest.raises(ValueError, match="different checkpoint transactions"):
        load_distributed_optimizer(
            restored_optimizer,
            restored_wrapped,
            path,
            state,
        )


def test_sharded_checkpoint_recovers_last_committed_directory(tmp_path):
    state = setup_distributed("none", "cpu")
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq")],
        output_adapter="token",
    )
    wrapped = prepare_distributed_model(model, state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "distributed-checkpoint"
    save_sharded_training_checkpoint(wrapped, optimizer, path, state, step=5)
    backup = path.with_name(path.name + ".previous")
    path.rename(backup)

    from dabsn.distributed import inspect_sharded_training_checkpoint

    assert inspect_sharded_training_checkpoint(path)["step"] == 5
    assert path.is_dir()
    assert not backup.exists()

def test_non_distributed_sharded_training_checkpoint(tmp_path):
    state = setup_distributed("none", "cpu")
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq")],
        output_adapter="token",
    )
    groups = dabsn_adamw_param_groups(model, 0.1)
    wrapped = prepare_distributed_model(model, state)
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    inputs = torch.randn(2, 5, 4)
    targets = torch.randint(0, 3, (2, 5))
    torch.nn.functional.cross_entropy(
        wrapped(inputs).reshape(-1, 3), targets.reshape(-1)
    ).backward()
    optimizer.step()

    path = tmp_path / "distributed-checkpoint"
    save_sharded_training_checkpoint(
        wrapped,
        optimizer,
        path,
        state,
        step=23,
        extra={"kind": "test"},
    )
    restored = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq")],
        output_adapter="token",
    )
    restored_groups = dabsn_adamw_param_groups(restored, 0.1)
    restored_wrapped = prepare_distributed_model(restored, state)
    restored_optimizer = torch.optim.AdamW(restored_groups, lr=1e-3)
    manifest = load_sharded_training_checkpoint(
        restored_wrapped,
        restored_optimizer,
        path,
        state,
    )
    assert manifest["step"] == 23
    assert manifest["extra"] == {"kind": "test"}
    assert not path.with_name(path.name + ".tmp").exists()
    assert not path.with_name(path.name + ".previous").exists()
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert restored_optimizer.state_dict()["state"]


def test_nested_fsdp_state_names_normalize_to_portable_model_keys():
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq"), DABSNLayerSpec(8, 7, "field")],
        output_adapter="token",
    )
    wrapped = DABSNSequenceModule(model)
    nested = {}
    for name, value in wrapped.state_dict().items():
        name = "_fsdp_wrapped_module." + name
        name = name.replace(
            ".backbone.blocks.0.",
            ".backbone.blocks.0._fsdp_wrapped_module.",
        )
        nested[name] = value
    portable = _portable_state_dict(wrapped, nested)
    assert set(portable) == set(model.state_dict())
    assert all(value.device.type == "cpu" for value in portable.values())


def test_fsdp_requires_torchrun(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.raises(RuntimeError, match="torchrun"):
        setup_distributed("fsdp", "cuda")


def test_cli_resume_rejects_missing_model(tmp_path):
    config = tmp_path / "model.json"
    config.write_text(
        json.dumps(
            {
                "input_dim": 4,
                "out_dim": 3,
                "layers": [{"hidden_dim": 7, "state_dim": 6, "read_geometry": "seq"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="existing checkpoint"):
        cli_main(
            [
                "train",
                "--config",
                str(config),
                "--data",
                str(tmp_path / "unused.pt"),
                "--output",
                str(tmp_path / "missing.pt"),
                "--backend",
                "reference",
                "--resume",
            ]
        )


def test_cli_resume_rejects_missing_optimizer_sidecar(tmp_path):
    config = tmp_path / "model.json"
    config.write_text(
        json.dumps(
            {
                "input_dim": 4,
                "out_dim": 3,
                "layers": [{"hidden_dim": 7, "state_dim": 6, "read_geometry": "seq"}],
            }
        ),
        encoding="utf-8",
    )
    data = tmp_path / "data.pt"
    torch.save(
        {
            "inputs": torch.randn(2, 5, 4),
            "targets": torch.randint(0, 3, (2, 5)),
        },
        data,
    )
    checkpoint = tmp_path / "model.safetensors"
    save_dabsn(
        DABSNModel(4, 3, [DABSNLayerSpec(7, 6, "seq")], output_adapter="token"),
        checkpoint,
    )
    with pytest.raises(FileNotFoundError, match="optimizer sidecar"):
        cli_main(
            [
                "train",
                "--config",
                str(config),
                "--data",
                str(data),
                "--output",
                str(checkpoint),
                "--backend",
                "reference",
                "--resume",
            ]
        )


def test_fsdp_wrapper_contract(monkeypatch):
    import torch.distributed.fsdp as fsdp

    captured = {}

    def fake_fsdp(module, **kwargs):
        captured.update(kwargs)
        return module

    monkeypatch.setattr(fsdp, "FullyShardedDataParallel", fake_fsdp)
    module = torch.nn.Linear(3, 4)
    state = DistributedState(
        kind="fsdp",
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cuda", 0),
        backend="nccl",
    )
    assert wrap_distributed(module, state) is module
    assert captured["sharding_strategy"] is fsdp.ShardingStrategy.FULL_SHARD
    assert captured["use_orig_params"] is True
    assert captured["sync_module_states"] is True
    assert captured["limit_all_gathers"] is True
    assert captured["forward_prefetch"] is True
    assert captured["auto_wrap_policy"] is not None


def test_distributed_state_reports_sharding_truth():
    state = setup_distributed("none", "cpu")
    assert state.report() == {
        "kind": "none",
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "device": "cpu",
        "backend": None,
        "parameter_sharded": False,
        "gradient_sharded": False,
        "optimizer_sharded": False,
        "batch_parallel": False,
        "context_parallel": False,
    }


def test_two_gpu_fsdp_gate_is_packaged():
    root = Path(__file__).resolve().parents[1]
    shell = root / "tools" / "fsdp_check.sh"
    gate = root / "tools" / "fsdp_check.py"
    assert shell.is_file() and shell.stat().st_mode & 0o111
    assert gate.is_file() and gate.stat().st_mode & 0o111
    text = gate.read_text(encoding="utf-8")
    for required in (
        'setup_distributed("fsdp", "cuda")',
        "FullyShardedDataParallel.fsdp_modules",
        "save_distributed_dabsn",
        "load_distributed_optimizer",
        "save_sharded_training_checkpoint",
        "load_sharded_training_checkpoint",
        "compiled_gradient_preflight",
        "fp16_grad_accumulation",
        "fp16_scaler_resume_exact",
        "sharded_inference_exact",
        "cuda_triton",
    ):
        assert required in text
