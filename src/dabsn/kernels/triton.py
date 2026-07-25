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


_VALID_CORE_BACKENDS = {"auto", "batched", "persistent", "batched_fused"}


def select_core_backend(
    batch: int,
    hidden: int,
    *,
    requested: str = "auto",
    grad_enabled: bool = True,
    min_batch: int | None = None,
    min_work: int | None = None,
    fused_max_h: int | None = None,
) -> str:
    """Choose the core-scan kernel from the *execution shape*, not model guesses.

    Returns one of ``"persistent"``, ``"batched"``, or ``"batched_fused"``.

    The persistent scan is one program per sequence: it minimizes launch
    overhead for latency/small-batch work but turns the two recurrent matrices
    into GEMV traffic (no tensor cores). The batched scan shares each recurrent
    matrix load across the whole device batch through a real GEMM, so it wins
    whenever there is enough tensor-core work to amortize the launches.

    ``B >= 64`` alone routed large-model/small-microbatch training (a 1B model
    uses a small per-GPU microbatch with grad-accum) to the GEMV path precisely
    when ``H`` is largest -- backwards for scale. The decision here also fires
    when ``B*H`` clears a work floor, so a wide model uses tensor cores even at
    microbatch 4-16. Both thresholds are env-overridable and used only as an
    execution-shape signal, never as an architecture assumption.

    Wide cores (``H > fused_max_h``; a 1B model is H~2048) are served by the
    ``batched`` per-step GEMM scan, which is cuBLAS-optimal at that width and,
    under CUDA graphs, the transformer-competitive scale path. An experimental
    single-CTA/H-tile fused-wide kernel was evaluated but only routed if it beats
    cuBLAS by >=1.15x on the target GPU (Phase 7); at large H it does not, so the
    batched GEMM remains the wide-H path -- a recorded decision, not a gap.

    ``batched_fused`` (the single-launch tiled tensor-core scan) carries the full
    per-tile recurrent state in registers across T, so it only supports one-tile
    widths (``H <= fused_max_h``, default 256, env ``DABSN_FUSED_CORE_MAX_H``).
    ``auto`` selects it whenever it is width-safe and the work floor is cleared:
    its parity against the reference scan is a standing gate, asserted on CPU by
    the test suite and on device by ``tools/train_scale_gate.py`` (A100 bf16:
    worst grad |Δ| 1.2e-04 against a 2e-2 tolerance). An
    *explicit* ``batched_fused`` request at ``H > fused_max_h`` is routed to
    ``batched`` (with a one-time warning naming the reason) rather than sent to a
    kernel that would hard-fail -- never OOM/crash without an actionable path.
    """

    normalized = (requested or "auto").lower()
    if normalized not in _VALID_CORE_BACKENDS:
        raise ValueError(
            "DABSN_CORE_BACKEND must be one of "
            f"{sorted(_VALID_CORE_BACKENDS)}; got {requested!r}"
        )
    # The fused scan's Function stores a backward context it will never use under
    # no_grad, so prefer the plain batched GEMM there; same math, no dead tape.
    if not grad_enabled and normalized == "batched_fused":
        normalized = "batched"
    # No-grad is NOT automatically a latency workload. Routing every no-grad call
    # to the persistent GEMV scan meant batch inference, evaluation, and the
    # forward half of a benchmark ran off the tensor cores entirely -- measurably
    # slower at batch 256 than the same shapes WITH autograd, because the grad
    # path got the batched GEMM and the no-grad path did not. The work floor
    # below already distinguishes latency from throughput shapes, and it does so
    # from B and H rather than from whether gradients happen to be enabled, so
    # let it decide here too: a single sequence still gets the persistent scan.
    if fused_max_h is None:
        from .batched_runtime import _fused_max_h as _derive

        fused_max_h = _derive(None, int(batch))
    fused_ok = int(hidden) <= int(fused_max_h)
    if normalized == "batched_fused":
        if fused_ok:
            return "batched_fused"
        # Explicit request the fused kernel cannot serve: keep training alive on
        # the batched scan and say exactly why, once.
        from dabsn.runtime.dispatch import warn_routing_once

        warn_routing_once(
            "core_scan",
            "batched_fused requested but H exceeds the single-launch width; "
            "using batched (set DABSN_CORE_BACKEND=batched to silence)",
            hidden=int(hidden),
            fused_max_h=int(fused_max_h),
        )
        return "batched"
    if normalized == "batched":
        return "batched"
    if normalized == "persistent":
        return "persistent"
    # auto
    if min_batch is None:
        min_batch = int(os.environ.get("DABSN_BATCHED_CORE_MIN_BATCH", "64"))
    if min_work is None:
        min_work = int(os.environ.get("DABSN_BATCHED_CORE_MIN_WORK", "4096"))
    if int(batch) >= min_batch or int(batch) * int(hidden) >= min_work:
        # Enough tensor-core work to amortize the launches. Prefer the
        # single-launch fused scan whenever it is width-safe; otherwise the
        # batched GEMM scan (no width bound).
        if fused_ok:
            return "batched_fused"
        return "batched"
    return "persistent"


def _fused_max_h_status() -> int:
    """The fused-scan width ceiling for status reporting (device-derived)."""
    try:
        from .batched_runtime import _fused_max_h

        return int(_fused_max_h(None, 16))
    except Exception:
        return int(os.environ.get("DABSN_FUSED_CORE_MAX_H", "256"))


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
    selected = select_core_backend(
        int(inputs.shape[0]),
        hidden,
        requested=requested_backend,
        grad_enabled=bool(torch.is_grad_enabled()),
    )
    from dabsn.runtime.dispatch import log_routing_once

    log_routing_once(
        "core_scan",
        selected,
        batch=int(inputs.shape[0]),
        hidden=hidden,
        dtype=str(projected.dtype).replace("torch.", ""),
        requested=requested_backend,
        grad=bool(torch.is_grad_enabled()),
    )
    if selected in {"batched", "batched_fused"}:
        if selected == "batched_fused":
            from .batched_runtime import dabsn_core_scan_batched_fused as _scan
        else:
            from .batched_runtime import dabsn_core_scan_batched as _scan

        outputs = _scan(
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
        core._last_core_backend = (
            "cuda_batched_fused" if selected == "batched_fused" else "cuda_batched_gemm"
        )
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
        B, T, N = query.shape[0], query.shape[1], bank_keys.shape[1]
        dense_limit = int(os.environ.get("DABSN_TRAIN_DENSE_MAX_SCORES", "8388608"))

        def _query_chunk_t(requested_backend: str) -> int:
            """Query rows per tile so one tile's [B,chunk,N] scores stay in budget.

            Derived from the live shape, never pinned by hand. There used to be a
            DABSN_READ_QUERY_CHUNK override here; a flag whose correct value a
            human has to know is a decision the framework failed to make, and it
            silently outranked the budget it was meant to help -- a stale pin
            would hold a width that no longer fit the shape in front of it.
            B and N are known here, so the width is computed.
            """
            if requested_backend == "dense":
                return T
            return max(1, dense_limit // max(1, B * N))

        def _chunked_dense(chunk_t: int) -> Tensor:
            pieces = []
            for start in range(0, T, chunk_t):
                stop = min(T, start + chunk_t)
                pieces.append(runtime.dense_bmm_three_way_read(
                    query[:, start:stop], bank_keys, bank_writes, next_writes,
                    cocktail[:, start:stop], bank_cocktail, bank_key_bias,
                    bank_admission, scale, bank_idx, bank_valid, mode=mode,
                    query_offset=start, total_T=T, **gains,
                ))
            return torch.cat(pieces, dim=1)

        if torch.is_grad_enabled():
            # At language-model training sizes the admitted bank is front-packed
            # and modest, so dense BMM does O(T*admitted) work on tensor cores in
            # BOTH forward and backward -- the compact flash forward's legacy
            # backward is one serial program per query. The old dispatch kept BMM
            # only while B*T*admitted <= 8M and fell to that serial path beyond it,
            # so long context (or larger batch) dropped OFF tensor cores exactly
            # when it hurt most. Instead, TILE the query-time axis so every tile
            # stays under the score cap: the read never leaves tensor cores at any
            # context length, and per-tile activations shrink (lower peak memory,
            # so batch can stay larger). Query rows are independent given the bank,
            # so this is bit-identical to the untiled read.
            requested = os.environ.get("DABSN_TRAIN_READ_BACKEND", "dense_chunked").lower()
            if requested not in {"dense_chunked", "dense", "flash"}:
                raise ValueError(
                    "DABSN_TRAIN_READ_BACKEND must be dense_chunked, dense, or flash; "
                    f"got {requested!r}"
                )
            if requested == "flash":
                output = runtime.admitted_three_way_read_compact_flash_trainable(
                    query, bank_keys, bank_writes, next_writes, cocktail,
                    bank_cocktail, bank_key_bias, bank_admission, scale,
                    bank_idx, bank_valid, mode=mode, **gains,
                )
                read._last_three_way_backend = "compact_flash_trainable"
                return output
            chunk_t = _query_chunk_t(requested)
            if chunk_t >= T:
                output = runtime.dense_bmm_three_way_read(
                    query, bank_keys, bank_writes, next_writes, cocktail,
                    bank_cocktail, bank_key_bias, bank_admission, scale,
                    bank_idx, bank_valid, mode=mode, query_offset=0, total_T=T, **gains,
                )
                read._last_three_way_backend = "dense_bmm_trainable"
            else:
                output = _chunked_dense(chunk_t)
                read._last_three_way_backend = "dense_bmm_trainable_chunked"
            return output

        # Forward-only. The density dispatcher below picks dense vs flash purely
        # on how full the bank is -- it has no notion of how BIG the resulting
        # score tensor would be. At a large batch or long context the dense
        # branch materializes [B,T,N] scores plus per-read temporaries, which is
        # how a forward pass OOMs on 4 GiB allocations while the training path
        # (which tiles) sails through the same shapes. Query rows are independent
        # given the bank, so apply the same score budget here: bit-identical
        # output, bounded peak, and inference inherits the long-context headroom
        # training already had.
        infer_chunk_t = _query_chunk_t(
            os.environ.get("DABSN_TRAIN_READ_BACKEND", "dense_chunked").lower())
        if infer_chunk_t < T:
            output = _chunked_dense(infer_chunk_t)
            read._last_three_way_backend = "dense_bmm_infer_chunked"
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
        "batched_core_min_work": int(os.environ.get("DABSN_BATCHED_CORE_MIN_WORK", "4096")),
        "fused_core_max_h": _fused_max_h_status(),
        "batched_step_compile": os.environ.get("DABSN_BATCHED_STEP_COMPILE", "1") == "1",
        "train_dense_max_scores": int(os.environ.get("DABSN_TRAIN_DENSE_MAX_SCORES", "8388608")),
        "train_read_backend": os.environ.get("DABSN_TRAIN_READ_BACKEND", "dense_chunked"),
        "core_scan_enabled": bool(getattr(DABSNCore, "_cuda_native_enabled", False)),
        "admitted_three_way_read_enabled": bool(
            getattr(DABSNRead, "_cuda_native_enabled", False)
        ),
        "import_error": None if _IMPORT_ERROR is None else repr(_IMPORT_ERROR),
    }
