"""Mmap-backed next-token pretraining for large DABSN runs."""

from __future__ import annotations

import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
from torch import Tensor

from .checkpoint import inspect_dabsn, load_dabsn
from .config import DABSNPretrainConfig
from .distributed import (
    autocast_context,
    cleanup_distributed,
    clip_grad_norm,
    load_distributed_optimizer,
    load_sharded_training_checkpoint,
    make_grad_scaler,
    no_sync_context,
    optimizer_checkpoint_path,
    prepare_distributed_model,
    resolve_precision,
    save_distributed_dabsn,
    save_sharded_training_checkpoint,
    setup_distributed,
)
from .kernels import enable as enable_kernels
from .model import DABSNSequenceLM
from .runtime.api import verify_gradients
from .runtime.grad_accum import ManualGradientAccumulator
from .runtime.graph import make_graphed_train_callable
from .runtime.loss import chunked_cross_entropy_from_logits


def _load_corpus(config: DABSNPretrainConfig) -> np.ndarray:
    if config.corpus_bin:
        path = Path(config.corpus_bin)
        if not path.is_file():
            raise FileNotFoundError(f"corpus_bin does not exist: {path}")
        return np.memmap(path, dtype=np.dtype(config.corpus_dtype), mode="r")
    path = Path(str(config.corpus_text))
    if not path.is_file():
        raise FileNotFoundError(f"corpus_text does not exist: {path}")
    return np.frombuffer(path.read_bytes(), dtype=np.uint8)


def _split_corpus(
    corpus: np.ndarray,
    config: DABSNPretrainConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    max_context = max((config.train_context, *config.eval_contexts), default=config.train_context)
    if config.val_batches <= 0 or config.val_fraction == 0.0:
        if len(corpus) <= max_context + 1:
            raise ValueError(f"corpus needs more than {max_context + 1} tokens")
        return corpus, None
    train_tokens = int(len(corpus) * (1.0 - config.val_fraction))
    if train_tokens <= config.train_context + 1:
        raise ValueError("training corpus split is too short")
    if len(corpus) - train_tokens <= max_context + 1:
        raise ValueError("validation corpus split is too short for the largest eval context")
    return corpus[:train_tokens], corpus[train_tokens:]


def _next_token_batch(
    data: np.ndarray,
    *,
    batch_size: int,
    context: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    max_start = len(data) - int(context) - 1
    if max_start <= 0:
        raise ValueError(f"corpus split is too short for context={context}")
    starts = rng.integers(0, max_start, size=int(batch_size), endpoint=False)
    windows = np.stack(
        [data[int(start):int(start) + int(context) + 1] for start in starts]
    ).astype(np.int64, copy=False)
    cpu = torch.from_numpy(windows)
    if device.type == "cuda":
        cpu = cpu.pin_memory()
    batch = cpu.to(device, non_blocking=device.type == "cuda")
    return batch[:, :-1], batch[:, 1:]


def _mean_across_ranks(value: Tensor, state) -> float:
    reduced = value.detach().float()
    if reduced.numel() != 1:
        reduced = reduced.mean()
    if state.enabled:
        torch_dist.all_reduce(reduced, op=torch_dist.ReduceOp.SUM)
        reduced /= state.world_size
    return float(reduced.cpu())


def _max_across_ranks(value: float, state) -> float:
    reduced = torch.tensor(float(value), device=state.device)
    if state.enabled:
        torch_dist.all_reduce(reduced, op=torch_dist.ReduceOp.MAX)
    return float(reduced.cpu())


def _cosine_learning_rate(
    step: int,
    total_steps: int,
    base: float,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    if warmup_steps and step <= warmup_steps:
        return base * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    scale = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * scale


def _memory_report(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_gb": torch.cuda.memory_allocated(index) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(index) / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated(index) / 1e9,
    }


def _configure_cuda_allocator() -> None:
    """Enable expandable segments so the caching allocator can grow into
    fragmented free space instead of OOMing on one large contiguous request.

    Appends to any existing ``PYTORCH_CUDA_ALLOC_CONF`` without clobbering an
    operator's setting, is a no-op on platforms that ignore the flag, and can be
    turned off with ``DABSN_DISABLE_EXPANDABLE_SEGMENTS=1``. Must run before the
    first CUDA allocation to take effect.
    """
    if os.environ.get("DABSN_DISABLE_EXPANDABLE_SEGMENTS", "0") == "1":
        return
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in existing:
        return
    parts = [piece for piece in existing.split(",") if piece]
    parts.append("expandable_segments:True")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)


def _oom_actionable_message(
    config: DABSNPretrainConfig, state, exc: BaseException, phase: str
) -> str:
    """Turn a bare CUDA OOM into an operator-actionable report.

    Names the exact live shape (B/T/H/depth/precision/grad-accum), the allocator
    stats at the failure, and a ranked list of levers -- so an OOM is never a
    dead end.
    """
    hidden = getattr(config, "hidden_dim", "?")
    depth = getattr(config, "depth", "?")
    mem = _memory_report(state.device) if getattr(state, "device", None) is not None else {}
    mem_str = (
        f"allocated={mem.get('allocated_gb', 0):.2f}GB "
        f"reserved={mem.get('reserved_gb', 0):.2f}GB "
        f"peak={mem.get('peak_gb', 0):.2f}GB"
        if mem
        else "n/a"
    )
    fixes = [
        f"lower batch_size (now {config.batch_size}) or raise grad_accum_steps "
        f"(now {config.grad_accum_steps}) to keep the token budget",
        "set grad_checkpoint=True on the model to trade compute for activation memory",
        "shrink train_context (now "
        f"{getattr(config, 'train_context', '?')}) or enable block-time chunking "
        "(DABSN_BLOCK_CHUNK_T)",
        "enable chunked loss (DABSN_LOSS_CHUNK_SCORES) to avoid the [B*T,V] logits peak",
        "reduce read scores per tile (DABSN_TRAIN_DENSE_MAX_SCORES)",
    ]
    ranked = "\n".join(f"  {i + 1}. {fix}" for i, fix in enumerate(fixes))
    return (
        f"CUDA out of memory during {phase}. "
        f"Live shape: batch={config.batch_size} context={getattr(config, 'train_context', '?')} "
        f"hidden={hidden} depth={depth} precision={config.precision} "
        f"grad_accum={config.grad_accum_steps}. Allocator: {mem_str}.\n"
        f"Actionable fixes (most effective first):\n{ranked}\n"
        f"Original error: {type(exc).__name__}: {str(exc)[:300]}"
    )


def _validate_model_config(model: DABSNSequenceLM, config: DABSNPretrainConfig) -> None:
    expected_layers = [spec.to_metadata() for spec in config.layer_specs()]
    actual_layers = [spec.to_metadata() for spec in model.layers]
    mismatches = []
    if model.vocab != config.vocab:
        mismatches.append(f"vocab checkpoint={model.vocab} requested={config.vocab}")
    if model.hidden_dim != config.hidden_dim:
        mismatches.append(
            f"hidden_dim checkpoint={model.hidden_dim} requested={config.hidden_dim}"
        )
    if actual_layers != expected_layers:
        mismatches.append("layers differ")
    if model.tie_embeddings != config.tie_embeddings:
        mismatches.append(
            f"tie_embeddings checkpoint={model.tie_embeddings} requested={config.tie_embeddings}"
        )
    if mismatches:
        raise ValueError("resume configuration does not match checkpoint: " + "; ".join(mismatches))


def _evaluate(
    model,
    validation: np.ndarray,
    config: DABSNPretrainConfig,
    state,
    precision: str,
    rng: np.random.Generator,
) -> dict[int, float]:
    contexts = config.eval_contexts or (config.train_context,)
    losses: dict[int, float] = {}
    model.eval()
    with torch.no_grad():
        for context in contexts:
            eval_batch_size = max(
                1,
                min(
                    int(config.eval_batch_size),
                    (int(config.eval_batch_size) * 1024) // max(int(context), 1),
                ),
            )
            local = torch.zeros((), device=state.device)
            for _ in range(config.val_batches):
                inputs, targets = _next_token_batch(
                    validation,
                    batch_size=eval_batch_size,
                    context=int(context),
                    device=state.device,
                    rng=rng,
                )
                with autocast_context(state.device, precision):
                    loss = chunked_cross_entropy_from_logits(model(inputs), targets)
                local += loss.detach() / config.val_batches
            losses[int(context)] = _mean_across_ranks(local, state)
    model.train()
    return losses


def _gather_rng_states(rng: np.random.Generator, state) -> list[dict[str, object]]:
    local = rng.bit_generator.state
    if not state.enabled:
        return [local]
    gathered = [None for _ in range(state.world_size)]
    torch_dist.all_gather_object(gathered, local)
    return gathered


def _restore_rng_state(
    rng: np.random.Generator,
    saved: object,
    state,
) -> None:
    if not isinstance(saved, list) or len(saved) != state.world_size:
        raise ValueError(
            "exact pretrain resume requires one saved corpus RNG state per active worker; "
            f"checkpoint has {0 if not isinstance(saved, list) else len(saved)}, "
            f"current world size is {state.world_size}"
        )
    rng.bit_generator.state = saved[state.rank]


def pretrain_next_token(
    config: DABSNPretrainConfig,
    output: str | Path,
    *,
    device: str | torch.device | None = "auto",
    backend: str = "auto",
    resume: bool = False,
    verify: bool = False,
    compile_verification: bool = True,
    checkpoint_mode: str = "portable",
    final_export: str | Path | None = None,
) -> dict[str, object] | None:
    """Run deterministic mmap-backed next-token pretraining.

    ``portable`` writes a SafeTensors model plus a rank-zero optimizer
    sidecar. ``sharded`` writes a reshardable PyTorch Distributed Checkpoint;
    pass ``final_export`` only when a consolidated SafeTensors model can fit in
    rank zero host memory.
    """

    # Expandable segments must be set before the first CUDA allocation, which
    # setup_distributed may trigger -- configure the allocator first.
    _configure_cuda_allocator()
    state = setup_distributed(config.distributed, device)
    destination = Path(output)
    try:
        if checkpoint_mode not in {"portable", "sharded"}:
            raise ValueError("checkpoint_mode must be portable or sharded")
        selected_backend = state.device.type if backend == "auto" else backend
        kernel_report = enable_kernels(selected_backend, required=True)
        precision = resolve_precision(config.precision, state.device)
        os.environ["DABSN_LONG_SCAN_CHUNK"] = str(int(config.long_scan_chunk))
        if int(getattr(config, "loss_chunk_scores", 0)) > 0:
            os.environ["DABSN_LOSS_CHUNK_SCORES"] = str(int(config.loss_chunk_scores))
        if int(getattr(config, "block_chunk_t", 0)) != 0:
            os.environ["DABSN_BLOCK_CHUNK_T"] = str(int(config.block_chunk_t))
        if state.device.type == "cuda":
            torch.cuda.set_device(state.local_rank)
            torch.cuda.reset_peak_memory_stats(state.device)
            torch.set_float32_matmul_precision("high")
            torch.cuda.manual_seed_all(config.seed)
        torch.manual_seed(config.seed)

        corpus = _load_corpus(config)
        training, validation = _split_corpus(corpus, config)
        rng = np.random.default_rng(config.seed + state.rank * 100_003)
        exists = destination.is_file() if checkpoint_mode == "portable" else destination.is_dir()
        if resume and not exists:
            raise FileNotFoundError(f"--resume requires an existing checkpoint: {destination}")
        resume_extra: dict[str, object] = {}
        if resume and checkpoint_mode == "portable":
            resume_extra = inspect_dabsn(destination)["extra"]
            loaded = load_dabsn(destination, map_location=state.device)
            if not isinstance(loaded, DABSNSequenceLM):
                raise TypeError("pretrain resume requires a DABSNSequenceLM checkpoint")
            model = loaded
            _validate_model_config(model, config)
        else:
            model = DABSNSequenceLM(
                vocab=config.vocab,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                layers=config.layers,
                layer_geometries=config.layer_geometries,
                state_dim=config.state_dim,
                tie_embeddings=config.tie_embeddings,
                grad_checkpoint=config.grad_checkpoint,
            ).to(state.device)

        global_tokens_per_step = (
            config.batch_size
            * config.train_context
            * state.world_size
        )
        total_steps = config.steps
        if config.token_budget:
            total_steps = max(1, math.ceil(config.token_budget / global_tokens_per_step))

        if verify:
            sample, _ = _next_token_batch(
                training,
                batch_size=min(2, config.batch_size),
                context=min(config.train_context, 32),
                device=state.device,
                rng=np.random.default_rng(config.seed + state.rank * 100_003 + 1),
            )
            verify_gradients(
                model,
                sample,
                compile_forward=compile_verification,
            )
            model.zero_grad(set_to_none=True)

        train_model = prepare_distributed_model(model, state, precision=precision)
        optimizer = torch.optim.AdamW(
            train_model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        scaler = make_grad_scaler(state, precision)
        start_step = 0
        if resume:
            if checkpoint_mode == "portable":
                restored_step = load_distributed_optimizer(
                    optimizer,
                    train_model,
                    destination,
                    state,
                    scaler=scaler,
                )
                if restored_step is None:
                    raise FileNotFoundError(
                        "--resume requires the optimizer sidecar: "
                        f"{optimizer_checkpoint_path(destination)}"
                    )
                start_step = restored_step
            else:
                manifest = load_sharded_training_checkpoint(
                    train_model,
                    optimizer,
                    destination,
                    state,
                )
                start_step = int(manifest["step"])
                resume_extra = manifest.get("extra", {})
                scaler_state = manifest.get("extra", {}).get("amp_scaler")
                if scaler is not None and scaler_state is not None:
                    scaler.load_state_dict(scaler_state)
            if total_steps < start_step:
                raise ValueError(
                    f"requested final step {total_steps} is before resumed step {start_step}"
                )
            _restore_rng_state(rng, resume_extra.get("corpus_rng_states"), state)

        # Optional CUDA-graph capture of the fwd+bwd. Graphs remove the tens of
        # thousands of eager kernel launches the recurrent T-loop issues per
        # microbatch, but capture is a single-process CUDA optimization: sharded
        # collectives cannot be recorded here, so distributed runs stay eager and
        # rely on comm-overlap instead. Each replay is an independent single
        # backward accumulated by ManualGradientAccumulator, so correctness never
        # depends on autograd summing across graph replays.
        graph_call = train_model
        accumulator: ManualGradientAccumulator | None = None
        graph_note = None
        if config.cuda_graph:
            if state.device.type != "cuda":
                graph_note = "skipped: cuda_graph requires a CUDA device"
            elif state.enabled:
                graph_note = (
                    "skipped: cuda_graph capture is single-process only; "
                    f"distributed={state.kind} runs eager with comm-overlap"
                )
            else:
                sample_inputs, _ = _next_token_batch(
                    training,
                    batch_size=config.batch_size,
                    context=config.train_context,
                    device=state.device,
                    rng=np.random.default_rng(config.seed + 777),
                )
                # On CUDA a capture failure is a real defect, not a reason to
                # silently fall back to the eager Python T-loop (which is the slow
                # path this whole feature exists to remove). Let it raise.
                graph_call = make_graphed_train_callable(
                    train_model, (sample_inputs,), verify=True
                )
                if graph_call is train_model:
                    raise RuntimeError(
                        "cuda_graph was requested but make_graphed_train_callable "
                        "returned the module uncaptured; refusing to run the eager "
                        "fallback. Investigate capture support on this device."
                    )
                accumulator = ManualGradientAccumulator(train_model)
                graph_note = "captured"
                model.zero_grad(set_to_none=True)
            if state.is_main and graph_note:
                print(f"[pretrain] cuda_graph {graph_note}", flush=True)

        last_loss = math.nan
        last_validation: dict[int, float] = {}
        interval_started = time.perf_counter()
        interval_tokens = 0
        train_model.train()
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step + 1, total_steps + 1):
            learning_rate = _cosine_learning_rate(
                step,
                total_steps,
                config.learning_rate,
                config.warmup_steps,
                config.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            local_loss = torch.zeros((), device=state.device)
            inputs, targets = _next_token_batch(
                training,
                batch_size=config.batch_size,
                context=config.train_context,
                device=state.device,
                rng=rng,
            )
            update_boundary = step % config.grad_accum_steps == 0 or step == total_steps
            if accumulator is not None:
                accumulator.begin_microbatch()
            with no_sync_context(train_model, synchronize=update_boundary):
                with autocast_context(state.device, precision):
                    # Chunk the FP32 CE upcast so the [B*T, V] logits never spawn
                    # a second full-size FP32 copy at the loss (auto-engaged by
                    # the element budget; identical loss otherwise). The graphed
                    # forward is captured in `graph_call`; the loss stays outside
                    # capture, so this does not change the graph.
                    loss = chunked_cross_entropy_from_logits(graph_call(inputs), targets)
                local_loss = loss.detach()
                scaled = loss / config.grad_accum_steps
                if scaler is None:
                    scaled.backward()
                else:
                    scaler.scale(scaled).backward()
            if accumulator is not None:
                # Each graph replay is an independent single backward; move its
                # gradients into persistent buffers before the next replay clears
                # them. The /grad_accum_steps scale already lives in the loss.
                accumulator.add_microbatch(scale=1.0)
            if update_boundary:
                if accumulator is not None:
                    accumulator.install()
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if config.clip_grad_norm > 0:
                    clip_grad_norm(train_model, config.clip_grad_norm)
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if accumulator is not None:
                    accumulator.reset()

            interval_tokens += global_tokens_per_step
            should_log = bool(
                config.log_every and (step == start_step + 1 or step % config.log_every == 0)
            )
            should_validate = bool(
                validation is not None
                and config.val_batches > 0
                and (config.val_every or config.log_every)
                and (
                    step % (config.val_every or config.log_every) == 0
                    or step == total_steps
                )
            )
            if should_log or should_validate or step == total_steps:
                last_loss = _mean_across_ranks(local_loss, state)
            if should_validate:
                last_validation = _evaluate(
                    train_model,
                    validation,
                    config,
                    state,
                    precision,
                    rng,
                )
            if should_log and state.is_main:
                elapsed = _max_across_ranks(time.perf_counter() - interval_started, state)
                tokens_per_second = interval_tokens / max(elapsed, 1e-9)
                print(
                    f"[pretrain] step={step}/{total_steps} loss={last_loss:.6f} "
                    f"ppl={math.exp(min(last_loss, 20.0)):.3f} lr={learning_rate:.6g} "
                    f"tokens={step * global_tokens_per_step:,} tok/s={tokens_per_second:,.0f} "
                    f"memory={_memory_report(state.device)}",
                    flush=True,
                )
                interval_started = time.perf_counter()
                interval_tokens = 0
            elif should_log:
                _max_across_ranks(time.perf_counter() - interval_started, state)
                interval_started = time.perf_counter()
                interval_tokens = 0
            if should_validate and state.is_main:
                print(f"[pretrain] validation={last_validation}", flush=True)

            should_checkpoint = bool(
                config.checkpoint_every
                and step % config.checkpoint_every == 0
                and step < total_steps
            )
            if should_checkpoint:
                checkpoint_extra = {
                    "objective": "next-token",
                    "step": step,
                    "tokens_seen": step * global_tokens_per_step,
                    "pretrain_config": config.metadata(),
                    "corpus_rng_states": _gather_rng_states(rng, state),
                }
                if scaler is not None:
                    checkpoint_extra["amp_scaler"] = scaler.state_dict()
                if checkpoint_mode == "portable":
                    save_distributed_dabsn(
                        train_model,
                        destination,
                        state,
                        optimizer=optimizer,
                        step=step,
                        extra=checkpoint_extra,
                        scaler=scaler,
                    )
                else:
                    save_sharded_training_checkpoint(
                        train_model,
                        optimizer,
                        destination,
                        state,
                        step=step,
                        extra=checkpoint_extra,
                    )

        final_step = max(start_step, total_steps)
        final_extra = {
            "objective": "next-token",
            "step": final_step,
            "tokens_seen": final_step * global_tokens_per_step,
            "pretrain_config": config.metadata(),
            "last_train_loss": last_loss,
            "last_validation": last_validation,
            "corpus_rng_states": _gather_rng_states(rng, state),
        }
        if scaler is not None:
            final_extra["amp_scaler"] = scaler.state_dict()
        if checkpoint_mode == "portable":
            save_distributed_dabsn(
                train_model,
                destination,
                state,
                optimizer=optimizer,
                step=final_step,
                extra=final_extra,
                scaler=scaler,
            )
        else:
            save_sharded_training_checkpoint(
                train_model,
                optimizer,
                destination,
                state,
                step=final_step,
                extra=final_extra,
            )
            if final_export is not None:
                save_distributed_dabsn(
                    train_model,
                    final_export,
                    state,
                    extra=final_extra,
                )
        if not state.is_main:
            return None
        return {
            "backend": kernel_report["active_backend"],
            "checkpoint": str(destination),
            "checkpoint_mode": checkpoint_mode,
            "distributed": state.report(),
            "global_tokens_per_step": global_tokens_per_step,
            "last_train_loss": last_loss,
            "last_validation": last_validation,
            "precision": precision,
            "step": final_step,
            "tokens_seen": final_step * global_tokens_per_step,
            "final_export": None if final_export is None else str(final_export),
            "cuda_graph": graph_note if config.cuda_graph else None,
        }
    except torch.cuda.OutOfMemoryError as exc:
        # Never surface a bare allocator OOM: attach the live shape, allocator
        # stats, and a ranked list of levers so the operator knows exactly what
        # to change.
        raise torch.cuda.OutOfMemoryError(
            _oom_actionable_message(config, state, exc, "pretraining")
        ) from exc
    finally:
        cleanup_distributed(state)


__all__ = ["DABSNPretrainConfig", "pretrain_next_token"]
