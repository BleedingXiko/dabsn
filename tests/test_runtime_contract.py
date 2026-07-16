import json
import subprocess
import sys

import torch
import torch.nn as nn
from safetensors import safe_open

from dabsn import DABSNLayerSpec, DABSNModel, DABSNTaskModel
from dabsn.adapters import register_input_adapter, register_output_head
from dabsn.checkpoint import dabsn_config_dict, load_dabsn, save_dabsn
from dabsn.runtime import verify_gradients


def test_stack_gradients_and_checkpoint(tmp_path):
    layers = [DABSNLayerSpec(8, read_geometry=g) for g in ("seq", "field", "hybrid")]
    model = DABSNModel(5, 3, layers, output_adapter="token")
    x = torch.randn(2, 6, 5, requires_grad=True)
    output = model.forward_sequence(x); output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(block.core.W.weight.grad is not None for block in model.backbone.blocks)
    path = tmp_path / "model.safetensors"
    save_dabsn(model, path)
    clone = load_dabsn(path)
    assert torch.equal(model.forward_sequence(x.detach()), clone.forward_sequence(x.detach()))


def test_outer_compiled_mixed_geometry_stack_gradients():
    layers = [DABSNLayerSpec(7, read_geometry=g) for g in ("seq", "field", "hybrid")]
    model = DABSNModel(4, 3, layers, output_adapter="token")
    inputs = torch.randn(2, 6, 4, requires_grad=True)
    compiled_forward = torch.compile(
        model.forward_sequence,
        backend="aot_eager",
        dynamic=False,
    )
    compiled_forward(inputs).square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    for block in model.backbone.blocks:
        for gradient in (block.core.W.weight.grad, block.read_gain.grad):
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            assert float(gradient.detach().norm()) > 0.0


def test_public_gradient_preflight():
    model = DABSNModel(4, 2, [DABSNLayerSpec(6, read_geometry="seq")], output_adapter="token")
    report = verify_gradients(model, torch.randn(1, 5, 4))
    assert report and all(row["ok"] for row in report)


def test_custom_adapter_checkpoint_roundtrip(tmp_path):
    class StructuredInput(nn.Module):
        def __init__(self, raw_dim, model_dim):
            super().__init__()
            self.output_dim = model_dim
            self.fusion = nn.Sequential(
                nn.LayerNorm(raw_dim),
                nn.Linear(raw_dim, model_dim),
            )

        def forward(self, inputs):
            return self.fusion(inputs.float())

    class DistributionHead(nn.Module):
        def __init__(self, hidden_dim, out_dim):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_dim)
            self.projection = nn.Linear(hidden_dim, out_dim)

        def forward(self, hidden):
            return self.projection(self.norm(hidden))

    register_input_adapter("test_structured", StructuredInput)
    register_output_head("test_distribution", DistributionHead)
    model = DABSNTaskModel(
        raw_input_dim=7,
        model_input_dim=11,
        out_dim=5,
        layers=[DABSNLayerSpec(13, 9, "seq")],
        input_adapter="test_structured",
        output_adapter="test_distribution",
    )
    metadata = dabsn_config_dict(model)
    assert metadata["input_dim"] == 7
    assert metadata["hidden_dim"] == 11
    assert metadata["out_dim"] == 5

    inputs = torch.randn(2, 6, 7)
    path = tmp_path / "custom.safetensors"
    save_dabsn(model, path)
    restored = load_dabsn(path)
    assert torch.equal(model.forward_sequence(inputs), restored.forward_sequence(inputs))


def test_cli_help():
    result = subprocess.run([sys.executable, "-m", "dabsn.cli", "--help"], text=True, capture_output=True)
    assert result.returncode == 0 and "doctor" in result.stdout


def test_cli_train_evaluate_infer_export_lifecycle(tmp_path):
    config = {
        "input_dim": 4,
        "out_dim": 3,
        "hidden_dim": 6,
        "depth": 1,
        "geometry": "seq",
        "output_adapter": "token",
    }
    config_path = tmp_path / "config.json"
    data_path = tmp_path / "data.pt"
    checkpoint_path = tmp_path / "model.safetensors"
    inference_path = tmp_path / "inference.pt"
    export_path = tmp_path / "export.safetensors"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    torch.save(
        {
            "inputs": torch.randn(2, 5, 4),
            "targets": torch.randint(0, 3, (2, 5)),
        },
        data_path,
    )

    commands = [
        (
            "train",
            "--config", str(config_path),
            "--data", str(data_path),
            "--output", str(checkpoint_path),
            "--steps", "1",
        ),
        (
            "evaluate",
            "--checkpoint", str(checkpoint_path),
            "--data", str(data_path),
        ),
        (
            "infer",
            "--checkpoint", str(checkpoint_path),
            "--data", str(data_path),
            "--output", str(inference_path),
        ),
        (
            "export",
            "--checkpoint", str(checkpoint_path),
            "--output", str(export_path),
        ),
    ]
    for command in commands:
        result = subprocess.run(
            [sys.executable, "-m", "dabsn.cli", *command],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert checkpoint_path.is_file()
    assert inference_path.is_file()
    with safe_open(str(export_path), framework="pt") as exported:
        assert exported.metadata()["format"] == "dabsn-model"
