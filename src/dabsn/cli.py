"""Command-line interface for the canonical DABSN framework."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, cast

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F

from . import __version__
from .checkpoint import (
    build_graph_from_config,
    inspect_dabsn,
    load_dabsn,
    load_graph,
)
from .config import DABSNConfig, DABSNPretrainConfig
from .conformance import check_component
from .distributed import inspect_sharded_training_checkpoint, optimizer_checkpoint_path
from .graph import DABSNGraph
from .kernels import enable as enable_kernels
from .kernels import status as kernel_status
from .model import build_dabsn_from_config, dabsn_adamw_param_groups
from .pretrain import pretrain_next_token
from .runtime import (
    apply_optimizer_step,
    autocast_context,
    cleanup_distributed,
    clip_grad_norm,
    export_dabsn,
    load_distributed_optimizer,
    load_sharded_model_checkpoint,
    load_sharded_training_checkpoint,
    make_grad_scaler,
    no_sync_context,
    parse_parallel_topology,
    prepare_distributed_model,
    prepare_distributed_module,
    resolve_precision,
    save_distributed_dabsn,
    save_sharded_training_checkpoint,
    setup_distributed,
    shard_batch,
    unwrap_dabsn_artifact,
    verify_gradients,
)


def _json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _read_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return cast(dict[str, object], payload)


def _checkpoint_config(manifest: Mapping[str, object]) -> Mapping[str, object]:
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("distributed checkpoint manifest has no object-valued config")
    if not all(isinstance(key, str) for key in config):
        raise ValueError("distributed checkpoint config keys must be strings")
    return cast(Mapping[str, object], config)


def _checkpoint_step(manifest: Mapping[str, object]) -> int:
    step = manifest.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("distributed checkpoint manifest has an invalid step")
    return step


def _checkpoint_extra(manifest: Mapping[str, object]) -> Mapping[str, object]:
    extra = manifest.get("extra", {})
    if not isinstance(extra, Mapping):
        raise ValueError("distributed checkpoint manifest has a non-object extra field")
    return cast(Mapping[str, object], extra)


def _load_config(path: str | Path) -> DABSNConfig:
    return DABSNConfig(**cast(Any, _read_json(path)))


def _build_training_artifact(
    config: Mapping[str, object],
    *,
    device: torch.device,
    grad_checkpoint: bool,
    trusted_providers: tuple[str, ...],
):
    if config.get("model_kind") == "graph" or "components" in config:
        return build_graph_from_config(
            config,
            trusted_providers=trusted_providers,
        ).to(device)
    return build_dabsn_from_config(
        DABSNConfig(**cast(Any, dict(config))),
        grad_checkpoint=grad_checkpoint,
    ).to(device)


def _load_portable_artifact(
    path: str | Path,
    *,
    device: torch.device,
    trusted_providers: tuple[str, ...],
):
    metadata = inspect_dabsn(path)
    if metadata["config"].get("model_kind") == "graph":
        return load_graph(
            path,
            map_location=device,
            trusted_providers=trusted_providers,
        )
    return load_dabsn(
        path,
        map_location=device,
        trusted_providers=trusted_providers,
    )


def _set_grad_checkpoint(artifact, enabled: bool | None) -> None:
    if enabled is None:
        return
    if isinstance(artifact, DABSNGraph):
        artifact.set_activation_checkpointing(bool(enabled))
        return
    backbone = getattr(artifact, "backbone", None)
    if backbone is not None and hasattr(backbone, "grad_checkpoint"):
        backbone.grad_checkpoint = bool(enabled)
        graph = getattr(backbone, "graph", None)
        if isinstance(graph, DABSNGraph):
            graph.set_activation_checkpointing(bool(enabled))


def _prepare_training_artifact(artifact, state, precision: str):
    if isinstance(artifact, DABSNGraph):
        return prepare_distributed_module(artifact, state, precision=precision)
    return prepare_distributed_model(artifact, state, precision=precision)


def _load_data(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "inputs" not in payload:
        raise ValueError("data file must be a dict containing inputs and optional targets")
    return payload


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
        return F.cross_entropy(output.reshape(-1, output.shape[-1]), target.reshape(-1))
    return F.mse_loss(output, target)


def _activate_backend(requested: str, device: torch.device) -> dict[str, object]:
    backend = device.type if requested == "auto" else requested
    if backend not in {"reference", "cpu", "cuda"}:
        raise ValueError(f"backend {backend!r} is not supported")
    return enable_kernels(backend, required=True)


def _kernels(args) -> int:
    if args.enable is not None:
        _json(enable_kernels(args.enable, required=args.required))
    else:
        _json(kernel_status())
    return 0


def _doctor(args) -> int:
    config = (
        _load_config(args.config)
        if args.config
        else DABSNConfig(
            input_dim=5,
            out_dim=3,
            layers=[
                {"hidden_dim": 7, "state_dim": 6, "read_geometry": "seq"},
                {"hidden_dim": 9, "state_dim": 7, "read_geometry": "field"},
                {"hidden_dim": 11, "state_dim": 8, "read_geometry": "hybrid"},
            ],
        )
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    backend = _activate_backend(args.backend, device)
    model = build_dabsn_from_config(config, grad_checkpoint=args.grad_checkpoint).to(device)
    sample = torch.randn(args.batch, args.steps, config.input_dim, device=device)
    rows = verify_gradients(model, sample, compile_forward=not args.no_compile)
    _json({"ok": True, "backend": backend["active_backend"], "rows": rows})
    return 0


def _save_training_state(
    args,
    model,
    optimizer,
    state,
    *,
    step: int,
    extra: dict[str, object],
) -> None:
    extra = dict(extra)
    scaler = getattr(args, "_scaler", None)
    if scaler is not None:
        extra["amp_scaler"] = scaler.state_dict()
    if args.checkpoint_mode == "portable":
        save_distributed_dabsn(
            model,
            args.output,
            state,
            optimizer=optimizer,
            step=step,
            extra=extra,
            scaler=scaler,
        )
    else:
        save_sharded_training_checkpoint(
            model,
            optimizer,
            args.output,
            state,
            step=step,
            extra=extra,
            scaler=scaler,
        )


def _supervised_train(args, *, initial_checkpoint: str | None = None) -> int:
    """Run framework-level supervised training or weight-only fine-tuning."""

    state = setup_distributed(
        args.distributed,
        args.device,
        topology=parse_parallel_topology(args.parallel_axis),
    )
    try:
        config = _read_json(args.config) if args.config else None
        trusted_providers = tuple(args.trust_provider)
        output = Path(args.output)
        if (
            initial_checkpoint is not None
            and Path(initial_checkpoint).resolve() == output.resolve()
        ):
            raise ValueError("fine-tune output must differ from its input checkpoint")

        if args.resume:
            exists = output.is_file() if args.checkpoint_mode == "portable" else output.is_dir()
            if not exists:
                raise FileNotFoundError(f"--resume requires an existing checkpoint: {output}")
            if (
                args.checkpoint_mode == "portable"
                and not optimizer_checkpoint_path(output).is_file()
            ):
                raise FileNotFoundError(
                    "--resume requires the optimizer sidecar written beside the "
                    f"SafeTensors model: {optimizer_checkpoint_path(output)}"
                )

        precision = resolve_precision(args.precision, state.device)
        backend = _activate_backend(args.backend, state.device)
        if state.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        torch.manual_seed(args.seed)

        if initial_checkpoint is not None:
            model = _load_portable_artifact(
                initial_checkpoint,
                device=state.device,
                trusted_providers=trusted_providers,
            )
            _set_grad_checkpoint(model, args.grad_checkpoint)
        elif args.resume and args.checkpoint_mode == "portable":
            model = _load_portable_artifact(
                output,
                device=state.device,
                trusted_providers=trusted_providers,
            )
            _set_grad_checkpoint(model, args.grad_checkpoint)
        elif args.resume:
            manifest = inspect_sharded_training_checkpoint(output)
            checkpoint_config = _checkpoint_config(manifest)
            model = _build_training_artifact(
                checkpoint_config,
                device=state.device,
                grad_checkpoint=bool(args.grad_checkpoint),
                trusted_providers=trusted_providers,
            )
            _set_grad_checkpoint(model, args.grad_checkpoint)
        else:
            if config is None:
                raise ValueError("new training requires --config")
            model = _build_training_artifact(
                config,
                device=state.device,
                grad_checkpoint=bool(args.grad_checkpoint),
                trusted_providers=trusted_providers,
            )

        payload = _load_data(args.data)
        targets = payload.get("targets")
        if targets is None:
            raise ValueError("training data requires targets")
        inputs = shard_batch(payload["inputs"], state)
        targets = shard_batch(targets, state)
        if args.verify_gradients:
            verify_gradients(
                model,
                inputs[: min(2, inputs.shape[0])],
                compile_forward=not args.no_compile,
            )
            model.zero_grad(set_to_none=True)

        train_model = _prepare_training_artifact(model, state, precision)
        parameter_groups = dabsn_adamw_param_groups(train_model, args.weight_decay)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.learning_rate)
        scaler = make_grad_scaler(state, precision)
        args._scaler = scaler
        start_step = 0
        if args.resume:
            if args.checkpoint_mode == "portable":
                restored_step = load_distributed_optimizer(
                    optimizer,
                    train_model,
                    output,
                    state,
                    scaler=scaler,
                )
                if restored_step is None:
                    raise FileNotFoundError(
                        "--resume requires the optimizer sidecar written beside the "
                        f"SafeTensors model: {output}.optimizer.pt"
                    )
                start_step = restored_step
            else:
                manifest = load_sharded_training_checkpoint(
                    train_model,
                    optimizer,
                    output,
                    state,
                    scaler=scaler,
                )
                start_step = _checkpoint_step(manifest)
                scaler_state = _checkpoint_extra(manifest).get("amp_scaler")
                if scaler is not None and scaler_state is not None:
                    if not isinstance(scaler_state, dict):
                        raise ValueError("checkpoint amp_scaler state must be an object")
                    scaler.load_state_dict(cast(dict[str, Any], scaler_state))
            if args.steps < start_step:
                raise ValueError(
                    f"--steps ({args.steps}) cannot be less than resumed step ({start_step})"
                )

        accumulation = max(1, int(args.grad_accum_steps))
        if inputs.shape[0] < accumulation:
            raise ValueError(
                f"per-rank batch {inputs.shape[0]} is smaller than grad accumulation {accumulation}"
            )
        input_chunks = inputs.chunk(accumulation, dim=0)
        target_chunks = targets.chunk(accumulation, dim=0)
        losses: list[float] = []
        train_model.train()
        for step in range(start_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device=state.device)
            for index, (micro_inputs, micro_targets) in enumerate(zip(input_chunks, target_chunks)):
                synchronize = index == len(input_chunks) - 1
                with no_sync_context(train_model, synchronize=synchronize):
                    with autocast_context(state.device, precision):
                        raw_model = unwrap_dabsn_artifact(train_model)
                        graph = raw_model if isinstance(raw_model, DABSNGraph) else getattr(
                            raw_model, "graph", None
                        )
                        if graph is not None and graph.loss_declarations:
                            component_result = train_model(micro_inputs, with_terms=True)
                            loss = _loss(component_result.value, micro_targets)
                            loss = loss + sum(component_result.loss_terms)
                        else:
                            loss = _loss(train_model(micro_inputs), micro_targets)
                    local_loss += loss.detach() / len(input_chunks)
                    scaled_loss = loss / len(input_chunks)
                    if scaler is None:
                        scaled_loss.backward()
                    else:
                        scaler.scale(scaled_loss).backward()
            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.clip_grad_norm is not None:
                clip_grad_norm(train_model, args.clip_grad_norm)
            apply_optimizer_step(
                unwrap_dabsn_artifact(train_model),
                optimizer,
                scaler=scaler,
            )
            if state.batch_parallel:
                torch_dist.all_reduce(
                    local_loss,
                    op=torch_dist.ReduceOp.SUM,
                    group=state.data_group,
                )
                local_loss /= state.data_world_size
            losses.append(float(local_loss.cpu()))
            if args.checkpoint_every and step % args.checkpoint_every == 0 and step < args.steps:
                _save_training_state(
                    args,
                    train_model,
                    optimizer,
                    state,
                    step=step,
                    extra={"training_kind": args.training_kind, "step": step},
                )

        final_step = max(start_step, args.steps)
        _save_training_state(
            args,
            train_model,
            optimizer,
            state,
            step=final_step,
            extra={"training_kind": args.training_kind, "step": final_step},
        )
        if args.checkpoint_mode == "sharded" and args.final_export:
            save_distributed_dabsn(
                train_model,
                args.final_export,
                state,
                extra={"training_kind": args.training_kind, "step": final_step},
            )
        if state.is_main:
            _json(
                {
                    "backend": backend["active_backend"],
                    "checkpoint": str(output),
                    "checkpoint_mode": args.checkpoint_mode,
                    "distributed": state.report(),
                    "final_export": args.final_export,
                    "grad_checkpoint": bool(
                        getattr(getattr(model, "backbone", None), "grad_checkpoint", False)
                    ),
                    "losses": losses,
                    "precision": precision,
                    "steps": final_step,
                    "training_kind": args.training_kind,
                }
            )
        return 0
    finally:
        cleanup_distributed(state)


def _train(args) -> int:
    return _supervised_train(args)


def _finetune(args) -> int:
    return _supervised_train(args, initial_checkpoint=args.checkpoint)


def _pretrain(args) -> int:
    config = DABSNPretrainConfig(**cast(Any, _read_json(args.config)))
    if args.distributed is not None:
        config = replace(config, distributed=args.distributed)
    if args.precision is not None:
        config = replace(config, precision=args.precision)
    if args.parallel_axis:
        config = replace(config, parallel_axes=tuple(args.parallel_axis))
    result = pretrain_next_token(
        config,
        args.output,
        device=args.device,
        backend=args.backend,
        resume=args.resume,
        verify=args.verify_gradients,
        compile_verification=not args.no_compile,
        checkpoint_mode=args.checkpoint_mode,
        final_export=args.final_export,
    )
    if result is not None:
        _json(result)
    return 0


def _load_execution_model(args, state, precision: str):
    if args.checkpoint_mode == "portable":
        raw = _load_portable_artifact(
            args.checkpoint,
            device=state.device,
            trusted_providers=tuple(args.trust_provider),
        )
        wrapped = _prepare_training_artifact(raw, state, precision)
        return raw, wrapped
    manifest = inspect_sharded_training_checkpoint(args.checkpoint)
    raw = _build_training_artifact(
        _checkpoint_config(manifest),
        device=state.device,
        grad_checkpoint=False,
        trusted_providers=tuple(args.trust_provider),
    )
    wrapped = _prepare_training_artifact(raw, state, precision)
    load_sharded_model_checkpoint(wrapped, args.checkpoint, state)
    return raw, wrapped


def _evaluate(args) -> int:
    state = setup_distributed(
        args.distributed,
        args.device,
        topology=parse_parallel_topology(args.parallel_axis),
    )
    try:
        precision = resolve_precision(args.precision, state.device)
        backend = _activate_backend(args.backend, state.device)
        _, model = _load_execution_model(args, state, precision)
        payload = _load_data(args.data)
        if "targets" not in payload:
            raise ValueError("evaluation data requires targets")
        inputs = shard_batch(payload["inputs"], state)
        targets = shard_batch(payload["targets"], state)
        model.eval()
        with torch.no_grad(), autocast_context(state.device, precision):
            loss = _loss(model(inputs), targets).detach().float()
        if state.batch_parallel:
            torch_dist.all_reduce(loss, op=torch_dist.ReduceOp.SUM, group=state.data_group)
            loss /= state.data_world_size
        if state.is_main:
            _json(
                {
                    "backend": backend["active_backend"],
                    "batches": state.data_world_size if state.batch_parallel else 1,
                    "distributed": state.report(),
                    "loss": float(loss.cpu()),
                    "precision": precision,
                }
            )
        return 0
    finally:
        cleanup_distributed(state)


def _infer(args) -> int:
    state = setup_distributed(
        args.distributed,
        args.device,
        topology=parse_parallel_topology(args.parallel_axis),
    )
    try:
        precision = resolve_precision(args.precision, state.device)
        backend = _activate_backend(args.backend, state.device)
        _, model = _load_execution_model(args, state, precision)
        payload = _load_data(args.data)
        inputs = shard_batch(payload["inputs"], state)
        positions = payload.get("positions")
        if positions is not None:
            if positions.dim() == 2:
                positions = shard_batch(positions, state)
            else:
                positions = positions.to(state.device)
        model.eval()
        with torch.no_grad(), autocast_context(state.device, precision):
            local_output = model(inputs, positions).detach()
        if state.batch_parallel:
            gathered = [torch.empty_like(local_output) for _ in range(state.data_world_size)]
            torch_dist.all_gather(gathered, local_output, group=state.data_group)
            output = torch.cat(gathered, dim=0).cpu()
        else:
            output = local_output.cpu()
        if state.is_main:
            torch.save(output, args.output)
            _json(
                {
                    "backend": backend["active_backend"],
                    "distributed": state.report(),
                    "output": args.output,
                    "precision": precision,
                    "shape": list(output.shape),
                }
            )
        return 0
    finally:
        cleanup_distributed(state)


def _export(args) -> int:
    if args.checkpoint_mode == "portable":
        model = _load_portable_artifact(
            args.checkpoint,
            device=torch.device("cpu"),
            trusted_providers=tuple(args.trust_provider),
        )
        sample = _load_data(args.data)["inputs"] if args.data else None
        export_dabsn(model, args.output, sample_input=sample, format=args.format)
        _json({"format": args.format, "output": args.output})
        return 0
    if args.format == "torch-export":
        raise ValueError("torch-export requires a portable checkpoint")
    state = setup_distributed(
        args.distributed,
        args.device,
        topology=parse_parallel_topology(args.parallel_axis),
    )
    try:
        precision = resolve_precision(args.precision, state.device)
        _, model = _load_execution_model(args, state, precision)
        save_distributed_dabsn(model, args.output, state)
        if state.is_main:
            _json({"format": "safetensors", "output": args.output})
        return 0
    finally:
        cleanup_distributed(state)


def _component_check(args) -> int:
    state = setup_distributed(
        args.distributed,
        args.device,
        topology=parse_parallel_topology(getattr(args, "parallel_axis", ())),
    )
    config = _read_json(args.config)
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    try:
        report = check_component(
            args.provider,
            config,
            device=state.device,
            dtype=dtype,
            trust_provider=args.trust_provider,
            distributed_state=state,
        )
        if state.is_main:
            _json(report.to_dict())
        return 0 if report.passed else 1
    finally:
        cleanup_distributed(state)


def _add_execution_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--device", default="auto")
    command.add_argument("--backend", choices=("auto", "reference", "cpu", "cuda"), default="auto")
    command.add_argument("--distributed", choices=("none", "ddp", "fsdp", "fsdp2"), default="none")
    command.add_argument(
        "--parallel-axis",
        action="append",
        default=[],
        metavar="NAME=SIZE",
        help="repeatable named worker-mesh axis; products must equal the launched workers",
    )
    command.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    command.add_argument(
        "--trust-provider",
        action="append",
        default=[],
        help="authorize an installed provider key for this command; repeatable",
    )


def _add_training_arguments(
    command: argparse.ArgumentParser,
    *,
    config_required: bool,
) -> None:
    command.add_argument("--config", required=config_required)
    command.add_argument("--data", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--steps", type=int, default=1)
    command.add_argument("--learning-rate", type=float, default=1e-3)
    command.add_argument("--weight-decay", type=float, default=0.01)
    _add_execution_arguments(command)
    command.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    command.add_argument("--grad-accum-steps", type=int, default=1)
    command.add_argument("--clip-grad-norm", type=float, default=1.0)
    command.add_argument("--verify-gradients", action="store_true")
    command.add_argument("--no-compile", action="store_true")
    command.add_argument("--checkpoint-every", type=int, default=0)
    command.add_argument("--checkpoint-mode", choices=("portable", "sharded"), default="portable")
    command.add_argument("--final-export")
    command.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dabsn", description="DABSN recurrent modeling framework")
    parser.add_argument("--version", action="version", version=f"dabsn {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    kernels = commands.add_parser("kernels", help="show or enable native backends")
    kernels.add_argument("--enable", choices=("auto", "reference", "cpu", "cuda"))
    kernels.add_argument("--required", action="store_true")
    kernels.set_defaults(handler=_kernels)

    doctor = commands.add_parser("doctor", help="run compiled-stack gradient preflight")
    doctor.add_argument("--config")
    doctor.add_argument("--batch", type=int, default=2)
    doctor.add_argument("--steps", type=int, default=8)
    doctor.add_argument("--device", default="auto")
    doctor.add_argument("--backend", choices=("auto", "reference", "cpu", "cuda"), default="auto")
    doctor.add_argument("--grad-checkpoint", action="store_true")
    doctor.add_argument("--no-compile", action="store_true")
    doctor.set_defaults(handler=_doctor)

    component = commands.add_parser("component", help="component provider tooling")
    component_commands = component.add_subparsers(dest="component_command", required=True)
    component_check = component_commands.add_parser("check", help="run provider conformance")
    component_check.add_argument("provider")
    component_check.add_argument("--config", required=True)
    component_check.add_argument("--device", default="cpu")
    component_check.add_argument(
        "--distributed",
        choices=("none", "ddp", "fsdp", "fsdp2"),
        default="none",
    )
    component_check.add_argument(
        "--parallel-axis",
        action="append",
        default=[],
        metavar="NAME=SIZE",
        help="repeatable named worker-mesh axis used by provider parallel plans",
    )
    component_check.add_argument(
        "--dtype",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
    )
    component_check.add_argument(
        "--trust-provider",
        action="store_true",
        help="explicitly authorize importing and executing an installed third-party provider",
    )
    component_check.set_defaults(handler=_component_check)

    train_command = commands.add_parser(
        "train", help="train a new model from config and prepared data"
    )
    _add_training_arguments(train_command, config_required=False)
    train_command.add_argument(
        "--resume", action="store_true", help="continue the same model, optimizer, and step"
    )
    train_command.set_defaults(handler=_train, training_kind="train")

    finetune_command = commands.add_parser(
        "finetune", help="start a new optimizer from existing model weights"
    )
    _add_training_arguments(finetune_command, config_required=False)
    finetune_command.add_argument(
        "--checkpoint", required=True, help="portable SafeTensors model to initialize from"
    )
    finetune_command.set_defaults(handler=_finetune, resume=False, training_kind="finetune")

    pretrain_command = commands.add_parser(
        "pretrain", help="next-token pretraining from an mmap token corpus"
    )
    pretrain_command.add_argument("--config", required=True)
    pretrain_command.add_argument("--output", required=True)
    pretrain_command.add_argument("--device", default="auto")
    pretrain_command.add_argument(
        "--backend", choices=("auto", "reference", "cpu", "cuda"), default="auto"
    )
    pretrain_command.add_argument("--distributed", choices=("none", "ddp", "fsdp", "fsdp2"))
    pretrain_command.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"))
    pretrain_command.add_argument(
        "--checkpoint-mode", choices=("portable", "sharded"), default="portable"
    )
    pretrain_command.add_argument("--final-export")
    pretrain_command.add_argument("--resume", action="store_true")
    pretrain_command.add_argument("--verify-gradients", action="store_true")
    pretrain_command.add_argument("--no-compile", action="store_true")
    pretrain_command.set_defaults(handler=_pretrain)

    evaluate_command = commands.add_parser(
        "evaluate", help="evaluate a portable or sharded checkpoint"
    )
    evaluate_command.add_argument("--checkpoint", required=True)
    evaluate_command.add_argument(
        "--checkpoint-mode", choices=("portable", "sharded"), default="portable"
    )
    evaluate_command.add_argument("--data", required=True)
    _add_execution_arguments(evaluate_command)
    evaluate_command.set_defaults(handler=_evaluate)

    infer_command = commands.add_parser(
        "infer", help="run inference from a portable or sharded checkpoint"
    )
    infer_command.add_argument("--checkpoint", required=True)
    infer_command.add_argument(
        "--checkpoint-mode", choices=("portable", "sharded"), default="portable"
    )
    infer_command.add_argument("--data", required=True)
    infer_command.add_argument("--output", required=True)
    _add_execution_arguments(infer_command)
    infer_command.set_defaults(handler=_infer)

    export_command = commands.add_parser(
        "export", help="export SafeTensors weights or a Torch program"
    )
    export_command.add_argument("--checkpoint", required=True)
    export_command.add_argument(
        "--checkpoint-mode", choices=("portable", "sharded"), default="portable"
    )
    export_command.add_argument("--output", required=True)
    export_command.add_argument(
        "--format", choices=("safetensors", "torch-export"), default="safetensors"
    )
    export_command.add_argument("--data")
    _add_execution_arguments(export_command)
    export_command.set_defaults(handler=_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
