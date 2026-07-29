#!/usr/bin/env python3
"""Two-GPU DABSN FSDP forward/backward, sharding, and resume gate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F

from dabsn import DABSNLayerSpec, DABSNModel, dabsn_adamw_param_groups, load_dabsn
from dabsn.kernels import enable, status
from dabsn.runtime import (
    autocast_context,
    cleanup_distributed,
    clip_grad_norm,
    load_distributed_optimizer,
    load_sharded_training_checkpoint,
    make_grad_scaler,
    no_sync_context,
    prepare_distributed_model,
    save_distributed_dabsn,
    save_sharded_training_checkpoint,
    setup_distributed,
    verify_gradients,
)


def _state_max_abs(left, right) -> float:
    maximum = 0.0
    right_state = right.state_dict()
    for name, value in left.state_dict().items():
        maximum = max(
            maximum,
            float((value.detach().cpu().float() - right_state[name].detach().cpu().float()).abs().max()),
        )
    return maximum


def _update_max_abs(model, initial_state) -> float:
    return max(
        float((value.detach().cpu().float() - initial_state[name].float()).abs().max())
        for name, value in model.state_dict().items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = setup_distributed("fsdp", "cuda")
    try:
        if state.world_size < 2:
            raise RuntimeError("FSDP gate requires at least two CUDA workers")
        kernel_report = enable("cuda", required=True)
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(20260715)
        layers = [
            DABSNLayerSpec(16, 14, "seq"),
            DABSNLayerSpec(18, 16, "field"),
            DABSNLayerSpec(20, 18, "hybrid"),
        ]
        raw_model = DABSNModel(
            8,
            11,
            layers,
            output_adapter="token",
            grad_checkpoint=True,
            residual=True,
            mlp_ratio=2.0,
        ).to(state.device)
        compiled_gradient_rows = verify_gradients(
            raw_model,
            torch.randn(2, 8, 8, device=state.device),
            compile_forward=True,
        )
        raw_model.zero_grad(set_to_none=True)
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in raw_model.state_dict().items()
        }
        reference = copy.deepcopy(raw_model)

        generator = torch.Generator(device="cpu").manual_seed(20260716)
        global_inputs = torch.randn(state.world_size * 2, 12, 8, generator=generator).to(state.device)
        global_targets = torch.randint(
            0,
            11,
            (state.world_size * 2, 12),
            generator=generator,
        ).to(state.device)

        reference_optimizer = torch.optim.AdamW(
            dabsn_adamw_param_groups(reference, 0.1),
            lr=1e-3,
        )
        reference_loss = F.cross_entropy(
            reference.forward_sequence(global_inputs).reshape(-1, 11),
            global_targets.reshape(-1),
        )
        reference_loss.backward()
        reference_optimizer.step()

        parameter_groups = dabsn_adamw_param_groups(raw_model, 0.1)
        train_model = prepare_distributed_model(raw_model, state, precision="fp32")
        optimizer = torch.optim.AdamW(parameter_groups, lr=1e-3)
        local_inputs = global_inputs.chunk(state.world_size, dim=0)[state.rank]
        local_targets = global_targets.chunk(state.world_size, dim=0)[state.rank]
        loss = F.cross_entropy(
            train_model(local_inputs).reshape(-1, 11),
            local_targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        training_traces = raw_model.read_traces()
        training_read_backends = [
            str(trace["read_contract"]["kernel_backend"])
            for trace in training_traces
        ]
        training_core_backends = [
            str(getattr(block.core, "_last_core_backend", None))
            for block in raw_model.backbone.blocks
        ]
        training_long_backends = [
            str(getattr(block.read, "_last_long_backend", None))
            for block in raw_model.backbone.blocks
        ]
        for index in range(len(layers)):
            if training_core_backends[index] != "cuda_triton":
                raise AssertionError(
                    f"block {index} training core used {training_core_backends[index]}"
                )
            if training_long_backends[index] != "cuda_triton":
                raise AssertionError(
                    f"block {index} training long read used {training_long_backends[index]}"
                )
            if training_read_backends[index] != "dense_bmm_trainable":
                raise AssertionError(
                    f"block {index} training admitted read used {training_read_backends[index]}"
                )

        from torch.distributed.fsdp import FullyShardedDataParallel

        fsdp_modules = FullyShardedDataParallel.fsdp_modules(train_model)
        if len(fsdp_modules) < len(layers) + 1:
            raise AssertionError(
                f"expected root plus {len(layers)} block FSDP wrappers, found {len(fsdp_modules)}"
            )
        local_parameter_numel = sum(parameter.numel() for parameter in train_model.parameters())
        global_parameter_numel = torch.tensor(local_parameter_numel, device=state.device)
        torch_dist.all_reduce(global_parameter_numel, op=torch_dist.ReduceOp.SUM)
        full_parameter_numel = sum(parameter.numel() for parameter in reference.parameters())
        if local_parameter_numel >= full_parameter_numel:
            raise AssertionError(
                f"rank {state.rank} holds {local_parameter_numel} parameters; "
                f"FULL_SHARD should hold less than the full {full_parameter_numel}"
            )
        if int(global_parameter_numel.item()) != full_parameter_numel:
            raise AssertionError(
                f"parameter shards sum to {int(global_parameter_numel.item())}, "
                f"expected {full_parameter_numel}"
            )

        args.output.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output / "fsdp-model.safetensors"
        save_distributed_dabsn(
            train_model,
            checkpoint,
            state,
            optimizer=optimizer,
            step=1,
        )
        restored = load_dabsn(checkpoint, map_location=state.device)
        restored_groups = dabsn_adamw_param_groups(restored, 0.1)
        restored_train_model = prepare_distributed_model(restored, state, precision="fp32")
        restored_optimizer = torch.optim.AdamW(restored_groups, lr=1e-3)
        restored_step = load_distributed_optimizer(
            restored_optimizer,
            restored_train_model,
            checkpoint,
            state,
        )
        if restored_step != 1:
            raise AssertionError(f"optimizer resume step was {restored_step}, expected 1")

        sharded_checkpoint = args.output / "fsdp-sharded-checkpoint"
        save_sharded_training_checkpoint(
            train_model,
            optimizer,
            sharded_checkpoint,
            state,
            step=1,
            extra={"gate": "two-gpu-fsdp"},
        )
        sharded_raw = DABSNModel(
            8,
            11,
            layers,
            output_adapter="token",
            grad_checkpoint=True,
            residual=True,
            mlp_ratio=2.0,
        ).to(state.device)
        sharded_groups = dabsn_adamw_param_groups(sharded_raw, 0.1)
        sharded_model = prepare_distributed_model(sharded_raw, state, precision="fp32")
        sharded_optimizer = torch.optim.AdamW(sharded_groups, lr=1e-3)
        sharded_manifest = load_sharded_training_checkpoint(
            sharded_model,
            sharded_optimizer,
            sharded_checkpoint,
            state,
        )
        if int(sharded_manifest["step"]) != 1:
            raise AssertionError(f"sharded resume step was {sharded_manifest['step']}, expected 1")
        train_model.eval()
        sharded_model.eval()
        with torch.no_grad():
            expected_local = train_model(local_inputs)
            sharded_local = sharded_model(local_inputs)
        torch.testing.assert_close(sharded_local, expected_local, atol=0, rtol=0)
        inference_read_backends = [
            str(trace["read_contract"]["kernel_backend"])
            for trace in raw_model.read_traces()
        ]
        allowed_inference_backends = {"compact_flash_infer", "dense_bmm_cuda"}
        for index, backend in enumerate(inference_read_backends):
            if backend not in allowed_inference_backends:
                raise AssertionError(
                    f"block {index} inference admitted read used {backend}"
                )

        consolidated = args.output / "fsdp-consolidated.safetensors"
        save_distributed_dabsn(sharded_model, consolidated, state)

        torch.manual_seed(20260717)
        amp_raw = DABSNModel(
            8,
            11,
            layers,
            output_adapter="token",
            grad_checkpoint=True,
            residual=True,
            mlp_ratio=2.0,
        ).to(state.device)
        amp_groups = dabsn_adamw_param_groups(amp_raw, 0.1)
        amp_model = prepare_distributed_model(amp_raw, state, precision="fp16")
        amp_optimizer = torch.optim.AdamW(amp_groups, lr=1e-3)
        amp_scaler = make_grad_scaler(state, "fp16")
        amp_optimizer.zero_grad(set_to_none=True)
        for microbatch, (micro_inputs, micro_targets) in enumerate(
            zip(local_inputs.chunk(2), local_targets.chunk(2))
        ):
            with no_sync_context(amp_model, synchronize=microbatch == 1):
                with autocast_context(state.device, "fp16"):
                    amp_loss = F.cross_entropy(
                        amp_model(micro_inputs).reshape(-1, 11),
                        micro_targets.reshape(-1),
                    ) / 2
                amp_scaler.scale(amp_loss).backward()
        amp_scaler.unscale_(amp_optimizer)
        amp_grad_norm = clip_grad_norm(amp_model, 1.0)
        amp_scaler.step(amp_optimizer)
        amp_scaler.update()
        if not torch.isfinite(amp_grad_norm):
            raise AssertionError(f"fp16 accumulated gradient norm is not finite: {amp_grad_norm}")

        amp_checkpoint = args.output / "fsdp-amp-model.safetensors"
        expected_scaler_state = amp_scaler.state_dict()
        save_distributed_dabsn(
            amp_model,
            amp_checkpoint,
            state,
            optimizer=amp_optimizer,
            step=2,
            scaler=amp_scaler,
        )
        amp_restored = load_dabsn(amp_checkpoint, map_location=state.device)
        amp_restored_groups = dabsn_adamw_param_groups(amp_restored, 0.1)
        amp_restored_model = prepare_distributed_model(
            amp_restored,
            state,
            precision="fp16",
        )
        amp_restored_optimizer = torch.optim.AdamW(amp_restored_groups, lr=1e-3)
        amp_restored_scaler = make_grad_scaler(state, "fp16")
        amp_restored_step = load_distributed_optimizer(
            amp_restored_optimizer,
            amp_restored_model,
            amp_checkpoint,
            state,
            scaler=amp_restored_scaler,
        )
        if amp_restored_step != 2:
            raise AssertionError(f"fp16 optimizer resume step was {amp_restored_step}, expected 2")
        if amp_restored_scaler.state_dict() != expected_scaler_state:
            raise AssertionError("fp16 FSDP gradient scaler state did not resume exactly")

        torch_dist.barrier()
        final_kernel_status = status()
        if final_kernel_status["active_backend"] != "cuda":
            raise AssertionError(final_kernel_status)
        for index, block in enumerate(raw_model.backbone.blocks):
            if getattr(block.core, "_last_core_backend", None) != "cuda_triton":
                raise AssertionError(f"block {index} core did not use Triton")
            if getattr(block.read, "_last_long_backend", None) != "cuda_triton":
                raise AssertionError(f"block {index} long read did not use Triton")

        if state.is_main:
            portable = load_dabsn(checkpoint, map_location="cpu")
            update_max_abs = _state_max_abs(portable, reference)
            reference_update_max_abs = _update_max_abs(reference, initial_state)
            portable_update_max_abs = _update_max_abs(portable, initial_state)
            if min(reference_update_max_abs, portable_update_max_abs) <= 1e-7:
                raise AssertionError(
                    "the global-batch update check did not change model parameters"
                )
            if update_max_abs > 5e-5:
                raise AssertionError(
                    f"FSDP global-batch update differs from reference by {update_max_abs}"
                )
            report = {
                "passed": True,
                "distributed": state.report(),
                "fsdp_modules": len(fsdp_modules),
                "full_parameter_numel": full_parameter_numel,
                "global_parameter_shard_numel": int(global_parameter_numel.item()),
                "rank0_parameter_shard_numel": local_parameter_numel,
                "kernel_backend": kernel_report["active_backend"],
                "training_core_backends": training_core_backends,
                "training_long_backends": training_long_backends,
                "training_read_backends": training_read_backends,
                "inference_read_backends": inference_read_backends,
                "optimizer_resume_step": restored_step,
                "sharded_optimizer_resume_step": int(sharded_manifest["step"]),
                "sharded_inference_exact": True,
                "safetensors_final_export": str(consolidated),
                "fp16_grad_accumulation": True,
                "fp16_grad_norm": float(amp_grad_norm.detach().cpu()),
                "fp16_optimizer_resume_step": amp_restored_step,
                "fp16_scaler_resume_exact": True,
                "compiled_gradient_preflight": compiled_gradient_rows,
                "reference_update_max_abs": update_max_abs,
                "reference_parameter_change_max_abs": reference_update_max_abs,
                "fsdp_parameter_change_max_abs": portable_update_max_abs,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "devices": [torch.cuda.get_device_name(index) for index in range(state.world_size)],
            }
            report_path = args.output / "fsdp-report.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
            print("DABSN TWO-GPU FSDP CHECK: PASS", flush=True)
        torch_dist.barrier()
        return 0
    finally:
        cleanup_distributed(state)


if __name__ == "__main__":
    raise SystemExit(main())
