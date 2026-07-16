import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from dabsn import DABSNLayerSpec, DABSNModel, inspect_dabsn, load_dabsn, save_dabsn
from dabsn.cli import main as cli_main
from dabsn.config import DABSNPretrainConfig
from dabsn.pretrain import pretrain_next_token


def _model_config(path):
    path.write_text(
        json.dumps(
            {
                "input_dim": 4,
                "out_dim": 3,
                "layers": [
                    {"hidden_dim": 7, "state_dim": 6, "read_geometry": "seq"}
                ],
                "output_adapter": "token",
            }
        ),
        encoding="utf-8",
    )


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
    assert cli_main(
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
    ) == 0
    assert source.read_bytes() == source_before
    metadata = inspect_dabsn(output)["extra"]
    assert metadata["step"] == 1
    assert metadata["training_kind"] == "finetune"
    assert metadata["training_transaction"]
    assert (tmp_path / "finetuned.safetensors.optimizer.pt").is_file()


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
