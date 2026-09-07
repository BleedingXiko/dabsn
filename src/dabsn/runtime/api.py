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
    temporal_state: Any | None = None,
    temporal_chunk_size: int | None = None,
    stream_ids: tuple[str | int, ...] | None = None,
    stream_reset_mask: Tensor | None = None,
    eligibility_batch_mode: str = "vectorized",
    eligibility_lane_group_size: int | None = None,
):
    if temporal_state is not None:
        return _online_train_step(
            model,
            inputs,
            targets,
            optimizer,
            loss_fn=loss_fn,
            clip_grad_norm=clip_grad_norm,
            distributed_state=distributed_state,
            metrics_callback=metrics_callback,
            temporal_state=temporal_state,
            temporal_chunk_size=temporal_chunk_size,
            stream_ids=stream_ids,
            stream_reset_mask=stream_reset_mask,
            eligibility_batch_mode=eligibility_batch_mode,
            eligibility_lane_group_size=eligibility_lane_group_size,
        )
    if (
        temporal_chunk_size is not None
        or stream_ids is not None
        or stream_reset_mask is not None
        or eligibility_batch_mode != "vectorized"
        or eligibility_lane_group_size is not None
    ):
        raise ValueError("online chunk/stream arguments require temporal_state")
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


def _online_train_step(
    model: nn.Module,
    inputs: object,
    targets: object,
    optimizer: torch.optim.Optimizer,
    *,
    loss_fn: LossFunction | None,
    clip_grad_norm: float | None,
    distributed_state: Any | None,
    metrics_callback: Callable[[dict[str, Tensor]], None] | None,
    temporal_state,
    temporal_chunk_size: int | None,
    stream_ids: tuple[str | int, ...] | None,
    stream_reset_mask: Tensor | None,
    eligibility_batch_mode: str,
    eligibility_lane_group_size: int | None,
):
    """Execute one bounded online-credit update using the normal optimizer."""

    from ..temporal_credit import OnlineTemporalState

    if not isinstance(temporal_state, OnlineTemporalState):
        raise TypeError("temporal_state must be OnlineTemporalState")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    primary_loss, next_state, output = _online_backward_step(
        model,
        inputs,
        targets,
        loss_fn=loss_fn,
        temporal_state=temporal_state,
        temporal_chunk_size=temporal_chunk_size,
        stream_ids=stream_ids,
        stream_reset_mask=stream_reset_mask,
        eligibility_batch_mode=eligibility_batch_mode,
        eligibility_lane_group_size=eligibility_lane_group_size,
    )
    if clip_grad_norm is not None:
        distributed_clip_grad_norm(model, float(clip_grad_norm))
    apply_optimizer_step(model, optimizer)
    result = output
    reduced_loss, metrics = _reduced_step_metrics(
        model,
        primary_loss.detach(),
        result,
        distributed_state,
    )
    if metrics_callback is not None:
        metrics_callback(metrics)
    return reduced_loss, next_state


def _online_backward_step(
    model: nn.Module,
    inputs: object,
    targets: object,
    *,
    temporal_state,
    loss_fn: LossFunction | None = None,
    temporal_chunk_size: int | None = None,
    stream_ids: tuple[str | int, ...] | None = None,
    stream_reset_mask: Tensor | None = None,
    eligibility_batch_mode: str = "vectorized",
    eligibility_lane_group_size: int | None = None,
    loss_scale: float = 1.0,
    accumulation_position: int | None = None,
):
    """Backpropagate one online microbatch without touching optimizer state."""

    from ..temporal_credit import (
        OnlineTemporalState,
        _merge_online_temporal_states,
        _slice_online_temporal_state,
        bind_online_temporal_contract,
        finalize_online_temporal_state,
        prepare_online_temporal_state,
    )

    if not isinstance(temporal_state, OnlineTemporalState):
        raise TypeError("temporal_state must be OnlineTemporalState")
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("online training currently requires tensor inputs and targets")
    if inputs.dim() < 2 or targets.shape[:2] != inputs.shape[:2]:
        raise ValueError("online inputs and targets must share [B,T]")
    batch, total_steps = int(inputs.shape[0]), int(inputs.shape[1])
    if batch != temporal_state.batch_size or total_steps <= 0:
        raise ValueError("online input shape does not match temporal state")
    chunk_size = total_steps if temporal_chunk_size is None else int(temporal_chunk_size)
    if chunk_size <= 0:
        raise ValueError("temporal_chunk_size must be positive")
    from ..temporal_credit import _online_blocks

    blocks = _online_blocks(model)
    if chunk_size < total_steps:
        bad = [index for index, block in enumerate(blocks) if block.read_geometry != "seq"]
        if bad:
            raise ValueError(
                "bounded multi-chunk online training requires seq read geometry; "
                f"noncausal field/hybrid blocks: {bad}"
            )
    if stream_ids is None:
        if temporal_state.stream_policy == "carry_lanes":
            raise ValueError("carry_lanes requires explicit stream_ids on every update")
        resolved_stream_ids = temporal_state.stream_ids
    else:
        resolved_stream_ids = tuple(stream_ids)
    prepared = prepare_online_temporal_state(
        model,
        temporal_state,
        stream_ids=resolved_stream_ids,
        stream_reset_mask=stream_reset_mask,
    )
    if eligibility_batch_mode not in {"vectorized", "lane_grouped"}:
        raise ValueError("eligibility_batch_mode must be vectorized or lane_grouped")
    if eligibility_batch_mode == "vectorized":
        if eligibility_lane_group_size is not None:
            raise ValueError("eligibility_lane_group_size requires lane_grouped mode")
        group_size = batch
    else:
        if eligibility_lane_group_size is None or int(eligibility_lane_group_size) <= 0:
            raise ValueError("lane_grouped mode requires a positive eligibility_lane_group_size")
        group_size = min(batch, int(eligibility_lane_group_size))
    prepared = bind_online_temporal_contract(
        prepared,
        temporal_chunk_size=chunk_size,
        eligibility_batch_mode=eligibility_batch_mode,
        eligibility_lane_group_size=eligibility_lane_group_size,
    )
    if not torch.isfinite(torch.tensor(float(loss_scale))) or float(loss_scale) < 0:
        raise ValueError("loss_scale must be finite and non-negative")

    forward_online = getattr(model, "forward_sequence_online", None)
    loss_declarations, _ = _result_declarations(model)
    group_states = []
    group_outputs = []
    weighted_losses: list[Tensor] = []
    weighted_terms: list[list[Tensor]] = [[] for _ in loss_declarations]
    for lane_start in range(0, batch, group_size):
        lane_stop = min(batch, lane_start + group_size)
        next_state = _slice_online_temporal_state(prepared, lane_start, lane_stop)
        output = None
        for start in range(0, total_steps, chunk_size):
            stop = min(total_steps, start + chunk_size)
            chunk_inputs = inputs[lane_start:lane_stop, start:stop]
            if callable(forward_online):
                if loss_declarations:
                    result, pending_state = forward_online(
                        chunk_inputs, next_state, with_terms=True
                    )
                else:
                    result, pending_state = forward_online(chunk_inputs, next_state)
            else:
                result = model(
                    chunk_inputs,
                    temporal_state=next_state,
                    with_terms=bool(loss_declarations),
                )
                if not isinstance(result, tuple) or len(result) != 2:
                    raise TypeError("prepared model did not return online output and state")
                result, pending_state = result
            if isinstance(result, ComponentOutput):
                output = cast(Tensor, result.value)
                terms = result.loss_terms
            else:
                output = result
                terms = ()
            if len(terms) != len(loss_declarations):
                raise RuntimeError("online component loss-term count changed")
            chunk_targets = targets[lane_start:lane_stop, start:stop]
            chunk_loss = (loss_fn or _default_loss)(output, chunk_targets)
            weight = (
                float(lane_stop - lane_start)
                / float(batch)
                * float(stop - start)
                / float(total_steps)
            )
            weighted_loss = chunk_loss * weight
            weighted_aux = sum(
                (term * weight for term in terms),
                weighted_loss.new_zeros(()),
            )
            ((weighted_loss + weighted_aux) * float(loss_scale)).backward()
            weighted_losses.append(weighted_loss.detach())
            for values, term in zip(weighted_terms, terms):
                values.append((term * weight).detach())
            next_state = finalize_online_temporal_state(pending_state)
        assert output is not None
        group_outputs.append(output.detach())
        group_states.append(next_state)
    completed = _merge_online_temporal_states(tuple(group_states))
    if accumulation_position is not None:
        completed = OnlineTemporalState(
            layers=completed.layers,
            stream_policy=completed.stream_policy,
            stream_ids=completed.stream_ids,
            pending_reset_mask=completed.pending_reset_mask,
            chunk_position=completed.chunk_position,
            accumulation_position=int(accumulation_position),
            temporal_chunk_size=completed.temporal_chunk_size,
            eligibility_batch_mode=completed.eligibility_batch_mode,
            eligibility_lane_group_size=completed.eligibility_lane_group_size,
            schema_version=completed.schema_version,
        )
    aggregate_terms = tuple(torch.stack(values).sum() for values in weighted_terms)
    result = ComponentOutput(torch.cat(group_outputs, dim=0), aggregate_terms)
    return torch.stack(weighted_losses).sum(), completed, result


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
