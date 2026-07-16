"""Task-neutral DABSN framework execution APIs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..checkpoint import save_dabsn

LossFunction = Callable[[Tensor, Tensor], Tensor]


def _default_loss(output: Tensor, target: Tensor) -> Tensor:
    if target.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
        return F.cross_entropy(output.reshape(-1, output.shape[-1]), target.reshape(-1))
    return F.mse_loss(output, target)


def _sequence_forward(model: nn.Module, inputs: Tensor) -> Tensor:
    forward_sequence = getattr(model, "forward_sequence", None)
    if callable(forward_sequence):
        return forward_sequence(inputs)
    return model(inputs)


def train_step(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    loss_fn: LossFunction | None = None,
    clip_grad_norm: float | None = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = _sequence_forward(model, inputs)
    loss = (loss_fn or _default_loss)(output, targets)
    loss.backward()
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm))
    optimizer.step()
    return float(loss.detach())


def train(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int | None = None,
    loss_fn: LossFunction | None = None,
    clip_grad_norm: float | None = None,
) -> list[float]:
    losses: list[float] = []
    for index, (inputs, targets) in enumerate(batches):
        if steps is not None and index >= int(steps):
            break
        losses.append(
            train_step(
                model,
                inputs,
                targets,
                optimizer,
                loss_fn=loss_fn,
                clip_grad_norm=clip_grad_norm,
            )
        )
    return losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    loss_fn: LossFunction | None = None,
) -> dict[str, float | int]:
    model.eval()
    losses: list[float] = []
    for inputs, targets in batches:
        output = _sequence_forward(model, inputs)
        losses.append(float((loss_fn or _default_loss)(output, targets)))
    if not losses:
        raise ValueError("evaluate requires at least one batch")
    return {
        "loss": sum(losses) / len(losses),
        "batches": len(losses),
    }


@torch.no_grad()
def infer(
    model: nn.Module,
    inputs: Tensor,
    *,
    positions: Tensor | None = None,
) -> Tensor:
    model.eval()
    forward_hidden = getattr(model, "forward_hidden", None)
    if callable(forward_hidden):
        hidden = forward_hidden(inputs)
        if positions is None:
            return model.project_sequence(hidden)
        return model.project_positions(hidden, positions)
    if positions is None:
        return model(inputs)
    return model(inputs, positions)


def export_dabsn(
    model: nn.Module,
    path: str | Path,
    *,
    sample_input: Tensor | None = None,
    format: str = "safetensors",
) -> None:
    """Export clean construction metadata and weights, or a Torch program."""

    destination = Path(path)
    normalized = format.lower()
    if normalized in {"weights", "safetensors"}:
        save_dabsn(model, destination)
        return
    if normalized == "torch-export":
        if sample_input is None:
            raise ValueError("torch-export requires sample_input")
        exported = torch.export.export(model, (sample_input,))
        torch.export.save(exported, destination)
        return
    raise ValueError("format must be safetensors or torch-export")


def verify_gradients(
    model: nn.Module,
    sample_input: Tensor,
    *,
    loss_builder: Callable[[Any], Tensor] | None = None,
    compile_forward: bool = True,
    backend: str = "aot_eager",
) -> list[dict[str, float | int | bool]]:
    """Fail loudly if any block in the user's real stack has dead gradients."""

    model.train()
    model.zero_grad(set_to_none=True)
    forward = getattr(model, "forward_sequence", None)
    if not callable(forward):
        raise TypeError(
            "verify_gradients must run on the unwrapped DABSN model before DDP/FSDP wrapping"
        )
    if compile_forward:
        forward = torch.compile(forward, backend=backend, dynamic=False)
    output = forward(sample_input)
    if loss_builder is None:
        if isinstance(output, dict):
            tensors = [value for value in output.values() if torch.is_tensor(value)]
            if not tensors:
                raise ValueError("gradient verification output dict has no tensors")
            loss = sum(tensor.float().square().mean() for tensor in tensors)
        else:
            loss = output.float().square().mean()
    else:
        loss = loss_builder(output)
    loss.backward()

    backbone = getattr(model, "backbone", None)
    if backbone is None and hasattr(model, "body"):
        backbone = model.body.backbone
    if backbone is None or not hasattr(backbone, "blocks"):
        raise TypeError("verify_gradients requires a model with a DABSN backbone")
    rows: list[dict[str, float | int | bool]] = []
    failures: list[int] = []
    for index, block in enumerate(backbone.blocks):
        core_gradient = block.core.W.weight.grad
        gain_gradient = block.read_gain.grad
        core_norm = 0.0 if core_gradient is None else float(core_gradient.detach().norm())
        gain_norm = 0.0 if gain_gradient is None else float(gain_gradient.detach().abs().max())
        finite = bool(
            torch.isfinite(torch.tensor([core_norm, gain_norm])).all().item()
        )
        ok = finite and core_norm > 0.0 and gain_norm > 0.0
        rows.append(
            {
                "block": index,
                "core_weight_grad_norm": core_norm,
                "read_gain_grad_norm": gain_norm,
                "finite": finite,
                "ok": ok,
            }
        )
        if not ok:
            failures.append(index)
    if failures:
        raise RuntimeError(
            f"DABSN compiled-stack gradient preflight failed for blocks {failures}: {rows}"
        )
    return rows
