"""Task-neutral DABSN framework execution APIs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..checkpoint import save_dabsn, save_graph
from ..components import ComponentOutput, ResultDeclaration
from ..distributed import clip_grad_norm as distributed_clip_grad_norm
from ..distributed import reduce_declared_results

LossFunction = Callable[[object, object], Tensor]


def _default_loss(output: object, target: object) -> Tensor:
    if not isinstance(output, Tensor) or not isinstance(target, Tensor):
        raise TypeError("structured outputs or targets require an explicit loss_fn")
    if target.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
        # Chunk the FP32 upcast so a large [B*T, V] logits tensor never spawns a
        # second full-size FP32 copy; auto-engages only above the element budget
        # and is numerically equivalent to F.cross_entropy otherwise.
        from .loss import chunked_cross_entropy_from_logits

        return chunked_cross_entropy_from_logits(output, target)
    return F.mse_loss(output, target)


def _sequence_forward(model: nn.Module, inputs: object) -> object:
    forward_sequence = getattr(model, "forward_sequence", None)
    if callable(forward_sequence):
        return forward_sequence(inputs)
    return model(inputs)


def _training_forward(model: nn.Module, inputs: object) -> ComponentOutput:
    forward = getattr(model, "forward_with_terms", None)
    if callable(forward):
        result = forward(inputs)
        if not isinstance(result, ComponentOutput):
            raise TypeError("forward_with_terms must return ComponentOutput")
        return result
    return ComponentOutput(_sequence_forward(model, inputs))


def _post_optimizer_step(model: nn.Module, *, step_applied: bool) -> None:
    action = getattr(model, "post_optimizer_step", None)
    if callable(action):
        action(step_applied=step_applied)


def _result_declarations(model: nn.Module):
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "loss_declarations") and hasattr(current, "report_declarations"):
            return current.loss_declarations, current.report_declarations
        graph = getattr(current, "graph", None)
        if graph is not None and hasattr(graph, "loss_declarations"):
            return graph.loss_declarations, graph.report_declarations
        backbone = getattr(current, "backbone", None)
        graph = None if backbone is None else getattr(backbone, "graph", None)
        if graph is not None and hasattr(graph, "loss_declarations"):
            return graph.loss_declarations, graph.report_declarations
        child = getattr(current, "module", None)
        if isinstance(child, nn.Module):
            current = child
            continue
        child = getattr(current, "body", None)
        if isinstance(child, nn.Module):
            current = child
            continue
        break
    return (), ()


def _reduced_step_metrics(
    model: nn.Module,
    primary_loss: Tensor,
    result: ComponentOutput,
    distributed_state,
) -> tuple[float, dict[str, Tensor]]:
    loss_declarations, report_declarations = _result_declarations(model)
    if len(loss_declarations) != len(result.loss_terms):
        loss_declarations = tuple(
            ResultDeclaration(f"unnamed_{index}", "mean", "framework")
            for index in range(len(result.loss_terms))
        )
    if len(report_declarations) != len(result.reports):
        report_declarations = tuple(
            ResultDeclaration(f"unnamed_{index}", "none", "none")
            for index in range(len(result.reports))
        )
    enabled = bool(distributed_state is not None and distributed_state.batch_parallel)
    group = distributed_state.data_group if enabled else None
    if enabled:
        reduced_primary = reduce_declared_results(
            (primary_loss,),
            (ResultDeclaration("primary", "mean", "framework"),),
            group=group,
        )[0]
        reduced_terms = reduce_declared_results(
            result.loss_terms,
            loss_declarations,
            group=group,
        )
        reduced_reports = reduce_declared_results(
            result.reports,
            report_declarations,
            group=group,
        )
    else:
        reduced_primary = primary_loss.detach()
        reduced_terms = tuple(term.detach() for term in result.loss_terms)
        reduced_reports = tuple(report.detach() for report in result.reports)
    reduced_total = reduced_primary + sum(
        reduced_terms,
        reduced_primary.new_zeros(()),
    )
    metrics = {"loss": reduced_total.detach()}
    metrics.update(
        {
            f"loss_term/{index}:{declaration.name}": value.detach()
            for index, (declaration, value) in enumerate(
                zip(loss_declarations, reduced_terms)
            )
        }
    )
    metrics.update(
        {
            f"report/{index}:{declaration.name}": value.detach()
            for index, (declaration, value) in enumerate(
                zip(report_declarations, reduced_reports)
            )
        }
    )
    return float(reduced_total.detach().cpu()), metrics


def apply_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: Any | None = None,
) -> bool:
    """Apply one optimizer update and run lifecycle actions only if it happened.

    AMP scalers lower their scale when non-finite gradients cause ``step`` to be
    skipped. Comparing the scale before and after ``update`` is the supported
    device-agnostic signal used by DABSN's CLI and pretrainer. Accumulation
    callers invoke this function only at their update boundary.
    """

    if scaler is None:
        optimizer.step()
        step_applied = True
    else:
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        step_applied = float(scaler.get_scale()) >= scale_before
    _post_optimizer_step(model, step_applied=step_applied)
    return step_applied


def train_step(
    model: nn.Module,
    inputs: object,
    targets: object,
    optimizer: torch.optim.Optimizer,
    *,
    loss_fn: LossFunction | None = None,
    clip_grad_norm: float | None = None,
    distributed_state: Any | None = None,
    metrics_callback: Callable[[dict[str, Tensor]], None] | None = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = _training_forward(model, inputs)
    primary_loss = (loss_fn or _default_loss)(result.value, targets)
    loss = primary_loss
    if result.loss_terms:
        for term in result.loss_terms:
            loss = loss + term
    loss.backward()
    if clip_grad_norm is not None:
        distributed_clip_grad_norm(model, float(clip_grad_norm))
    apply_optimizer_step(model, optimizer)
    reduced_loss, metrics = _reduced_step_metrics(
        model,
        primary_loss.detach(),
        result,
        distributed_state,
    )
    if metrics_callback is not None:
        metrics_callback(metrics)
    return reduced_loss


def train(
    model: nn.Module,
    batches: Iterable[tuple[object, object]],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int | None = None,
    loss_fn: LossFunction | None = None,
    clip_grad_norm: float | None = None,
    distributed_state: Any | None = None,
    metrics_callback: Callable[[dict[str, Tensor]], None] | None = None,
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
                distributed_state=distributed_state,
                metrics_callback=metrics_callback,
            )
        )
    return losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: Iterable[tuple[object, object]],
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
    inputs: object,
    *,
    positions: Tensor | None = None,
) -> object:
    model.eval()
    forward_hidden = getattr(model, "forward_hidden", None)
    if callable(forward_hidden):
        hidden = forward_hidden(inputs)
        typed_model = cast(Any, model)
        if positions is None:
            return typed_model.project_sequence(hidden)
        return typed_model.project_positions(hidden, positions)
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
        from ..graph import DABSNGraph

        if isinstance(model, DABSNGraph):
            save_graph(model, destination)
        else:
            save_dabsn(cast(Any, model), destination)
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
    sample_input: object,
    *,
    loss_builder: Callable[[Any], Tensor] | None = None,
    compile_forward: bool = True,
    backend: str = "aot_eager",
) -> list[dict[str, object]]:
    """Fail loudly when a trainable graph stage is disconnected or non-finite."""

    model.train()
    model.zero_grad(set_to_none=True)
    forward = getattr(model, "forward_sequence", None)
    if not callable(forward):
        forward = model.forward
    if compile_forward:
        forward = torch.compile(forward, backend=backend, dynamic=False)
    output = forward(sample_input)
    if loss_builder is None:
        from torch.utils._pytree import tree_flatten

        leaves, _ = tree_flatten(output)
        tensors = [value for value in leaves if torch.is_tensor(value)]
        if not tensors:
            raise ValueError("gradient verification output PyTree has no tensors")
        loss = sum(
            (tensor.float().square().mean() for tensor in tensors),
            tensors[0].new_zeros(()),
        )
    else:
        loss = loss_builder(output)
    if loss.requires_grad:
        loss.backward()

    backbone = getattr(model, "backbone", None)
    if backbone is None and hasattr(model, "body"):
        backbone = cast(Any, model).body.backbone
    if backbone is not None and hasattr(backbone, "blocks"):
        rows: list[dict[str, object]] = []
        failures: list[int] = []
        for index, block in enumerate(cast(Any, backbone).blocks):
            core_gradient = block.core.W.weight.grad
            gain_gradient = block.read_gain.grad
            core_norm = 0.0 if core_gradient is None else float(core_gradient.detach().norm())
            gain_norm = 0.0 if gain_gradient is None else float(gain_gradient.detach().abs().max())
            finite = bool(torch.isfinite(torch.tensor([core_norm, gain_norm])).all().item())
            ok = finite and core_norm > 0.0 and gain_norm > 0.0
            rows.append(
                {
                    "kind": "dabsn_block",
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

    from ..graph import DABSNGraph

    graph = model if isinstance(model, DABSNGraph) else getattr(model, "graph", None)
    if graph is None and backbone is not None:
        graph = getattr(backbone, "graph", None)
    if isinstance(graph, DABSNGraph):
        stages = tuple(
            (binding.component_id, binding.module) for binding in cast(Any, graph)._bindings
        )
    else:
        stages = (("model", model),)

    rows = []
    component_failures: list[str] = []
    for stage_name, stage in stages:
        parameters = tuple(parameter for parameter in stage.parameters() if parameter.requires_grad)
        gradients = tuple(parameter.grad for parameter in parameters if parameter.grad is not None)
        finite = all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
        if gradients:
            squared_norm = sum(
                (gradient.detach().double().square().sum() for gradient in gradients),
                gradients[0].new_zeros((), dtype=torch.float64),
            )
            gradient_norm = float(squared_norm.sqrt())
        else:
            gradient_norm = 0.0
        trainable = bool(parameters)
        ok = (not trainable) or (bool(gradients) and finite and gradient_norm > 0.0)
        rows.append(
            {
                "kind": "component",
                "component": stage_name,
                "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                "parameters_with_gradients": len(gradients),
                "gradient_norm": gradient_norm,
                "finite": finite,
                "ok": ok,
            }
        )
        if not ok:
            component_failures.append(stage_name)
    if component_failures:
        raise RuntimeError(
            "compiled graph gradient preflight failed for components "
            f"{component_failures}: {rows}"
        )
    return rows
