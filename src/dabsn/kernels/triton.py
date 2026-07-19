"""Lazy activation hooks for the DABSN Triton/CUDA runtime."""

from __future__ import annotations

import importlib.util
import os

import torch
from torch import Tensor

_EAGER_CORE_FORWARD_FROM_STATE = None
_EAGER_THREE_WAY_READ = None
_IMPORT_ERROR: Exception | None = None


def triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def _runtime():
    global _IMPORT_ERROR
    try:
        from . import triton_runtime

        return triton_runtime
    except Exception as exc:
        _IMPORT_ERROR = exc
        raise RuntimeError(
            f"DABSN Triton runtime unavailable: {type(exc).__name__}: {exc}"
        ) from exc


def cuda_core_forward(
    core,
    inputs: Tensor,
    return_writes: bool = False,
    return_cocktail: bool = False,
    *,
    initial_state=None,
    return_final_state: bool = False,
):
    if not inputs.is_cuda:
        if bool(getattr(type(core), "_cuda_native_required", False)):
            raise RuntimeError("Triton DABSN core was required, but received a CPU tensor")
        return _EAGER_CORE_FORWARD_FROM_STATE(
            core,
            inputs,
            initial_state=initial_state,
            return_writes=return_writes,
            return_cocktail=return_cocktail,
            return_final_state=return_final_state,
        )
    runtime = _runtime()
    hidden = int(core.hidden_dim)
    logit_saturation_decay = core.logit_saturation_decay.expand(hidden).contiguous()
    projected = core.W(inputs).contiguous()
    gate_projected = core.Wg(inputs).contiguous()
    requested_backend = os.environ.get("DABSN_CORE_BACKEND", "auto").lower()
    if requested_backend not in {"auto", "batched", "persistent"}:
        raise ValueError(
            "DABSN_CORE_BACKEND must be auto, batched, or persistent; "
            f"got {requested_backend!r}"
        )
    # The persistent scan minimizes launch overhead for latency/small batches.
    # Large-batch training must instead share each recurrent matrix load across
    # the batch through GEMM. Width/depth are not used as architecture guesses;
    # this dispatch is based only on the actual execution shape.
    batched = torch.is_grad_enabled() and (
        requested_backend == "batched"
        or (
            requested_backend == "auto"
            and inputs.shape[0] >= int(os.environ.get("DABSN_BATCHED_CORE_MIN_BATCH", "64"))
        )
    )
    if batched:
        from .batched_runtime import dabsn_core_scan_batched

        outputs = dabsn_core_scan_batched(
            projected,
            gate_projected,
            core.Ug.weight.contiguous(),
            core.A.weight.contiguous(),
            core.beta.contiguous(),
            core.log_kappa.contiguous(),
            core.logit_recover.contiguous(),
            core.k_s.contiguous(),
            core.k_y.contiguous(),
            core.k_b.contiguous(),
            core.k_n.contiguous(),
            core.k_bias.contiguous(),
            core.r_s.contiguous(),
            core.r_y.contiguous(),
            core.r_b.contiguous(),
            core.r_n.contiguous(),
            core.r_bias.contiguous(),
            logit_saturation_decay,
            core.k_saturation.contiguous(),
            core.r_saturation.contiguous(),
            core.logit_alpha.reshape(()),
            core.log_lambda.reshape(()),
            core.logit_saturation_suppress.reshape(()),
            return_tape=return_cocktail,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )
        core._last_core_backend = "cuda_batched_gemm"
    else:
        outputs = runtime.dabsn_core_scan_trainable_fused(
            projected,
            gate_projected,
            core.Ug.weight.contiguous(),
            core.A.weight.contiguous(),
            core.beta.contiguous(),
            core.log_kappa.contiguous(),
            core.logit_recover.contiguous(),
            core.k_s.contiguous(),
            core.k_y.contiguous(),
            core.k_b.contiguous(),
            core.k_n.contiguous(),
            core.k_bias.contiguous(),
            core.r_s.contiguous(),
            core.r_y.contiguous(),
            core.r_b.contiguous(),
            core.r_n.contiguous(),
            core.r_bias.contiguous(),
            logit_saturation_decay,
            core.k_saturation.contiguous(),
            core.r_saturation.contiguous(),
            core.logit_alpha.reshape(()),
            core.log_lambda.reshape(()),
            core.logit_saturation_suppress.reshape(()),
            return_tape=return_cocktail,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )
        core._last_core_backend = "cuda_triton"
    trajectory, novelty, plasticity, expression, write = outputs[:5]
    if return_cocktail:
        energy, saturation = outputs[5], outputs[6]
        result = trajectory, novelty, plasticity, expression, write, energy, saturation
        final_offset = 7
    elif return_writes:
        result = trajectory, novelty, plasticity, expression, write
        final_offset = 5
    else:
        result = trajectory, novelty, plasticity
        final_offset = 5
    if return_final_state:
        return result, tuple(outputs[final_offset:final_offset + 3])
    return result


def cuda_three_way_read(
    read,
    query: Tensor,
    bank_keys: Tensor,
    bank_writes: Tensor,
    next_writes: Tensor,
    cocktail: Tensor,
    bank_cocktail: Tensor,
    bank_key_bias: Tensor,
    bank_admission: Tensor,
    scale: Tensor,
    allow: Tensor | None,
    induct_allow: Tensor | None,
    eligible: Tensor | None,
    induct_eligible: Tensor | None,
) -> Tensor:
    if not query.is_cuda:
        if bool(getattr(type(read), "_cuda_native_required", False)):
            raise RuntimeError("Triton DABSN read was required, but received a CPU tensor")
        return _EAGER_THREE_WAY_READ(
            read,
            query,
            bank_keys,
            bank_writes,
            next_writes,
            cocktail,
            bank_cocktail,
            bank_key_bias,
            bank_admission,
            scale,
            allow,
            induct_allow,
            eligible,
            induct_eligible,
        )
    runtime = _runtime()
    gains = {
        "short_gain": read.short_gain,
        "pad_gain": read.pad_gain,
        "induct_gain": read.induct_gain,
        "cocktail_gain": read.cocktail_gain,
    }
    if allow is None:
        bank_idx = getattr(read, "_compact_bank_idx", None)
        bank_valid = getattr(read, "_compact_bank_valid", None)
        mode = getattr(read, "_compact_read_mode", None)
        if bank_idx is None or bank_valid is None or mode not in {"seq", "field"}:
            raise RuntimeError("compact DABSN read is missing bank index/valid metadata")
        if torch.is_grad_enabled():
            # At language-model training sizes the admitted bank is normally
            # dense and [B,T,N] is modest. Native BMM gives both forward and
            # backward tensor-core contractions; the compact flash forward's
            # legacy backward is one serial program per query. Large score
            # tensors retain the memory-bounded compact path.
            score_entries = query.shape[0] * query.shape[1] * bank_keys.shape[1]
            dense_limit = int(os.environ.get("DABSN_TRAIN_DENSE_MAX_SCORES", "8388608"))
            if score_entries <= dense_limit:
                output = runtime.dense_bmm_three_way_read(
                    query,
                    bank_keys,
                    bank_writes,
                    next_writes,
                    cocktail,
                    bank_cocktail,
                    bank_key_bias,
                    bank_admission,
                    scale,
                    bank_idx,
                    bank_valid,
                    mode=mode,
                    **gains,
                )
                read._last_three_way_backend = "dense_bmm_trainable"
            else:
                output = runtime.admitted_three_way_read_compact_flash_trainable(
                    query,
                    bank_keys,
                    bank_writes,
                    next_writes,
                    cocktail,
                    bank_cocktail,
                    bank_key_bias,
                    bank_admission,
                    scale,
                    bank_idx,
                    bank_valid,
                    mode=mode,
                    **gains,
                )
                read._last_three_way_backend = "compact_flash_trainable"
            return output
        output, backend = runtime.admitted_three_way_read_dispatch(
            query,
            bank_keys,
            bank_writes,
            next_writes,
            cocktail,
            bank_cocktail,
            bank_key_bias,
            bank_admission,
            scale,
            bank_idx,
            bank_valid,
            mode=mode,
            return_backend=True,
            **gains,
        )
        read._last_three_way_backend = backend
        return output
    output = runtime.admitted_three_way_read_trainable(
        query,
        bank_keys,
        bank_writes,
        next_writes,
        cocktail,
        bank_cocktail,
        bank_key_bias,
        bank_admission,
        scale,
        allow,
        induct_allow,
        eligible,
        induct_eligible,
        **gains,
    )
    read._last_three_way_backend = "dense_mask_triton_trainable"
    return output


def enable_triton_kernels(*, required: bool = False) -> dict[str, bool]:
    global _EAGER_CORE_FORWARD_FROM_STATE, _EAGER_THREE_WAY_READ
    from dabsn.core import DABSNCore
    from dabsn.read import DABSNRead

    if required and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required for DABSN Triton kernels, but is unavailable")
    if required and not triton_available():
        raise RuntimeError("Triton was required for DABSN CUDA kernels, but is unavailable")
    if not triton_available():
        return {"core_scan": False, "admitted_three_way_read": False}
    _runtime()
    if _EAGER_CORE_FORWARD_FROM_STATE is None:
        _EAGER_CORE_FORWARD_FROM_STATE = DABSNCore.forward_from_state
    if _EAGER_THREE_WAY_READ is None:
        _EAGER_THREE_WAY_READ = DABSNRead._three_way_read
    DABSNCore.forward = cuda_core_forward
    DABSNCore.forward_from_state = cuda_core_forward
    DABSNCore._cuda_native_enabled = True
    DABSNCore._cuda_native_required = bool(required)
    DABSNRead._three_way_read = cuda_three_way_read
    DABSNRead._cuda_native_enabled = True
    DABSNRead._cuda_native_required = bool(required)
    return {"core_scan": True, "admitted_three_way_read": True}


def triton_status() -> dict[str, object]:
    from dabsn.core import DABSNCore
    from dabsn.read import DABSNRead

    return {
        "cuda_available": torch.cuda.is_available(),
        "triton_available": triton_available(),
        "core_backend_policy": os.environ.get("DABSN_CORE_BACKEND", "auto"),
        "batched_core_min_batch": int(os.environ.get("DABSN_BATCHED_CORE_MIN_BATCH", "64")),
        "batched_step_compile": os.environ.get("DABSN_BATCHED_STEP_COMPILE", "1") == "1",
        "train_dense_max_scores": int(os.environ.get("DABSN_TRAIN_DENSE_MAX_SCORES", "8388608")),
        "core_scan_enabled": bool(getattr(DABSNCore, "_cuda_native_enabled", False)),
        "admitted_three_way_read_enabled": bool(
            getattr(DABSNRead, "_cuda_native_enabled", False)
        ),
        "import_error": None if _IMPORT_ERROR is None else repr(_IMPORT_ERROR),
    }
