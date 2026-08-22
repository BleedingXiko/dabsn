import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from dabsn import (
    DABSNLayerSpec,
    DABSNModel,
    inspect_dabsn,
    load_dabsn,
    load_graph,
    save_dabsn,
)
from dabsn.cli import main as cli_main
from dabsn.config import DABSNPretrainConfig
from dabsn.distributed import inspect_sharded_training_checkpoint
from dabsn.pretrain import pretrain_next_token


def _model_config(path):
    path.write_text(
        json.dumps(
            {
                "input_dim": 4,
                "out_dim": 3,
                "layers": [{"hidden_dim": 7, "state_dim": 6, "read_geometry": "seq"}],
                "output_adapter": "token",
            }
        ),
        encoding="utf-8",
    )


def _graph_config(path):
    path.write_text(
        json.dumps(
            {
                "model_kind": "graph",
                "components": [
                    {
                        "component_id": "dabsn.0",
                        "provider_key": "dabsn:block",
                        "config": {
                            "input_dim": 4,
                            "hidden_dim": 4,
                            "state_dim": 6,
                            "read_geometry": "seq",
                            "residual": True,
                        },
                    },
                    {
                        "component_id": "mlp.0",
                        "provider_key": "dabsn:residual_mlp",
                        "config": {"dim": 4, "ratio": 2.0},
                    },
                ],
                "require_world_builder": True,
            }
        ),
        encoding="utf-8",
    )


def test_cli_trains_and_reloads_raw_graph_from_json(tmp_path):
    config = tmp_path / "graph.json"
    data = tmp_path / "graph-data.pt"
    output = tmp_path / "graph.safetensors"
    _graph_config(config)
    torch.save(
        {
            "inputs": torch.randn(2, 5, 4),
            "targets": torch.randn(2, 5, 4),
        },
        data,
    )
    assert (
        cli_main(
            [
                "train",
                "--config",
                str(config),
                "--data",
                str(data),
                "--output",
                str(output),
                "--backend",
                "reference",
                "--steps",
                "1",
            ]
        )
        == 0
    )
    assert inspect_dabsn(output)["config"]["model_kind"] == "graph"
    restored = load_graph(output)
    assert restored(torch.randn(2, 3, 4)).shape == (2, 3, 4)
    assert (tmp_path / "graph.safetensors.optimizer.pt").is_file()


def test_cli_sharded_graph_resume_and_portable_final_export(tmp_path):
    config = tmp_path / "graph.json"
    data = tmp_path / "graph-data.pt"
    checkpoint = tmp_path / "graph-training"
    final = tmp_path / "graph-final.safetensors"
    _graph_config(config)
    torch.save(
        {
            "inputs": torch.randn(2, 5, 4),
            "targets": torch.randn(2, 5, 4),
        },
        data,
    )
    base = [
        "train",
        "--config",
        str(config),
        "--data",
        str(data),
        "--output",
        str(checkpoint),
        "--backend",
        "reference",
        "--checkpoint-mode",
        "sharded",
    ]
    assert cli_main([*base, "--steps", "1"]) == 0
    manifest = inspect_sharded_training_checkpoint(checkpoint)
    assert manifest["config"]["model_kind"] == "graph"
    assert manifest["providers"] == ["dabsn:block", "dabsn:residual_mlp"]
    assert (
        cli_main(
            [
                *base,
                "--steps",
                "2",
                "--resume",
                "--final-export",
                str(final),
            ]
        )
        == 0
    )
    assert inspect_sharded_training_checkpoint(checkpoint)["step"] == 2
    restored = load_graph(final)
    assert restored(torch.randn(2, 3, 4)).shape == (2, 3, 4)


def test_checkpoint_is_atomic_self_describing_safetensors(tmp_path):
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq")],
        output_adapter="token",
    )
    path = tmp_path / "model.safetensors"
    save_dabsn(model, path, extra={"step": 9})
    metadata = inspect_dabsn(path)
    assert metadata["format"] == "dabsn-model"
    assert metadata["config"]["model_kind"] == "model"
    assert metadata["extra"] == {"step": 9}
    assert not (tmp_path / "model.safetensors.tmp").exists()

    pickle_path = tmp_path / "legacy.pt"
    torch.save({"state_dict": model.state_dict()}, pickle_path)
    with pytest.raises(ValueError, match="SafeTensors"):
        load_dabsn(pickle_path)


def test_finetune_means_weights_in_new_optimizer_run(tmp_path):
    data = tmp_path / "data.pt"
    source = tmp_path / "source.safetensors"
    output = tmp_path / "finetuned.safetensors"
    torch.save(
        {
            "inputs": torch.randn(2, 5, 4),
            "targets": torch.randint(0, 3, (2, 5)),
        },
        data,
    )
    model = DABSNModel(
        4,
        3,
        [DABSNLayerSpec(7, 6, "seq")],
        output_adapter="token",
    )
    save_dabsn(model, source)
    source_before = source.read_bytes()
    assert (
        cli_main(
            [
                "finetune",
                "--checkpoint",
                str(source),
                "--data",
                str(data),
                "--output",
                str(output),
                "--backend",
                "reference",
                "--steps",
                "1",
            ]
        )
        == 0
    )
    assert source.read_bytes() == source_before
    metadata = inspect_dabsn(output)["extra"]
    assert metadata["step"] == 1
    assert metadata["training_kind"] == "finetune"
    assert metadata["training_transaction"]
    assert (tmp_path / "finetuned.safetensors.optimizer.pt").is_file()


def test_pretrain_cuda_graph_flag_falls_back_cleanly_on_cpu(tmp_path):
    # cuda_graph is a CUDA-only capture; on CPU it must transparently fall back
    # to the eager path and produce bit-identical weights, so the flag is safe
    # to leave on in configs that also run CPU smoke tests.
    corpus = tmp_path / "tokens.bin"
    np.asarray(list(range(16)) * 32, dtype=np.uint8).tofile(corpus)
    common = dict(
        corpus_bin=str(corpus),
        corpus_dtype="uint8",
        vocab=16,
        hidden_dim=6,
        depth=1,
        layer_geometries="seq",
        state_dim=5,
        tie_embeddings=True,
        train_context=4,
        steps=4,
        batch_size=2,
        eval_batch_size=2,
        val_batches=0,
        val_fraction=0.0,
        learning_rate=1e-3,
        warmup_steps=10,
        seed=321,
        precision="fp32",
        distributed="none",
        grad_checkpoint=False,
        grad_accum_steps=2,
        checkpoint_every=0,
        log_every=0,
        val_every=0,
    )
    eager = tmp_path / "eager.safetensors"
    graph = tmp_path / "graph.safetensors"
    report_eager = pretrain_next_token(
        DABSNPretrainConfig(**{**common, "cuda_graph": False}),
        eager,
        device="cpu",
        backend="cpu",
    )
    report_graph = pretrain_next_token(
        DABSNPretrainConfig(**{**common, "cuda_graph": True}),
        graph,
        device="cpu",
        backend="cpu",
    )
    assert report_eager["cuda_graph"] is None
    assert report_graph["cuda_graph"] == "skipped: cuda_graph requires a CUDA device"
    expected = load_file(str(eager))
    actual = load_file(str(graph))
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


def test_pretrain_portable_resume_matches_uninterrupted(tmp_path):
    corpus = tmp_path / "tokens.bin"
    np.asarray(list(range(16)) * 32, dtype=np.uint8).tofile(corpus)
    common = dict(
        corpus_bin=str(corpus),
        corpus_dtype="uint8",
        vocab=16,
        hidden_dim=6,
        depth=1,
        layer_geometries="seq",
        state_dim=5,
        tie_embeddings=True,
        train_context=4,
        steps=2,
        batch_size=2,
        eval_batch_size=2,
        val_batches=0,
        val_fraction=0.0,
        learning_rate=1e-3,
        warmup_steps=10,
        seed=321,
        precision="fp32",
        distributed="none",
        grad_checkpoint=False,
        grad_accum_steps=1,
        checkpoint_every=1,
        log_every=0,
        val_every=0,
    )
    uninterrupted = tmp_path / "uninterrupted.safetensors"
    resumed = tmp_path / "resumed.safetensors"
    pretrain_next_token(
        DABSNPretrainConfig(**common),
        uninterrupted,
        device="cpu",
        backend="cpu",
    )
    pretrain_next_token(
        DABSNPretrainConfig(**{**common, "steps": 1}),
        resumed,
        device="cpu",
        backend="cpu",
    )
    pretrain_next_token(
        DABSNPretrainConfig(**common),
        resumed,
        device="cpu",
        backend="cpu",
        resume=True,
    )
    expected = load_file(str(uninterrupted))
    actual = load_file(str(resumed))
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    assert inspect_dabsn(resumed)["extra"]["step"] == 2
