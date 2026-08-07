"""Canonical admitted, permanent, induction, and long-memory read."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _parameter(value: float) -> nn.Parameter:
    return nn.Parameter(torch.full((1,), float(value)))


def _stream_is_capturing() -> bool:
    """True only while a CUDA graph is actively being captured on the current
    stream. Unlike ``_cuda_native_enabled`` (set for the whole GPU session),
    this is the narrow window in which a host sync such as ``.item()`` is
    illegal -- so it is the correct gate for forcing a static, sync-free bank
    width. Returns False on CPU or when CUDA graphs are unavailable.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # pragma: no cover - very old / exotic CUDA builds
        return False


class _AdmittedReadParameters(nn.Module):
    """Parameters shared by the canonical three-way admitted read."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.log_temp = _parameter(0.0)
        self.p_gain = _parameter(1.0)
        self.novelty_gain = _parameter(0.5)
        self.null_bias = _parameter(-2.0)
        self.out = nn.Linear(hidden_dim * 3, hidden_dim)
        self.adm_p_gain = _parameter(1.0)
        self.adm_novelty_gain = _parameter(0.5)
        self.adm_saturation_gain = _parameter(0.5)
        self.adm_energy_gain = _parameter(0.25)
        self.adm_bias = _parameter(-2.0)

    def admission(
        self,
        novelty: Tensor,
        plasticity: Tensor,
        energy: Tensor,
        saturation: Tensor,
    ) -> Tensor:
        hidden = plasticity.shape[-1]
        return (
            self.adm_p_gain * torch.log1p(plasticity.mean(dim=-1) * hidden)
            + self.adm_novelty_gain * novelty.mean(dim=-1)
            + self.adm_saturation_gain * saturation.mean(dim=-1)
            + self.adm_energy_gain * energy.mean(dim=-1)
            + self.adm_bias
        )


class DABSNRead(nn.Module):
    """The one canonical DABSN memory read.

    Admission, permanent memory, induction, predictive expectation, and
    retention are unconditional. Geometry selects only the eligibility mask:
    causal prefix for ``seq``, whole object for ``field``, or a learned mixture
    for ``hybrid``.
    """

    def __init__(self, hidden_dim: int, read_geometry: str = "seq") -> None:
        super().__init__()
        if read_geometry not in {"seq", "field", "hybrid"}:
            raise ValueError(
                f"read_geometry must be seq, field, or hybrid; got {read_geometry}"
            )
        self.hidden_dim = hidden_dim
        self.read_geometry = read_geometry
        self.last_n_max: int | None = None
        self.last_hybrid_gate: float | None = None
        self.last_seq_norm: float | None = None
        self.last_field_norm: float | None = None
        self.last_field_neighbors: float | None = None
        self.last_admit_gate_mean: float | None = None
        self.last_long_scan_norm: float | None = None
        self.last_read_contract: dict[str, object] = {}
        self.collect_traces = os.environ.get("DABSN_COLLECT_TRACES", "0") == "1"
        self.hybrid_gate_override: float | None = None

        self.pad_gain = _parameter(1.0)
        self.short_gain = _parameter(1.0)
        self.induct_gain = _parameter(0.1)
        self.cocktail_gain = _parameter(1.0)
        if read_geometry == "hybrid":
            self.hybrid_gate = _parameter(0.0)
        self.admit_window = _parameter(4.0)
        self.short_read = _AdmittedReadParameters(hidden_dim)
        self.logit_retain = _parameter(2.2)
        self.long_gain = _parameter(0.1)
        self.logit_expect_retain = _parameter(2.2)
        self.logit_retention_decay = _parameter(2.2)
        self.logit_retention_strength = _parameter(1.4)

    def forward(
        self,
        y: Tensor,
        budget: Tensor,
        expression: Tensor,
        write: Tensor,
        novelty: Tensor,
        plasticity: Tensor,
        energy: Tensor,
        saturation: Tensor,
        field_shape: tuple[int, int] | None = None,
    ) -> Tensor:
        return self._unified_forward(
            y,
            budget,
            expression,
            write,
            novelty,
            plasticity,
            energy,
            saturation,
            field_shape,
        )

    def _unified_forward(
        self,
        y: Tensor,
        budget: Tensor,
        expression: Tensor,
        write: Tensor,
        novelty: Tensor,
        plasticity: Tensor,
        energy: Tensor,
        saturation: Tensor,
        field_shape: tuple[int, int] | None = None,
    ) -> Tensor:
        del field_shape
        read_parameters = self.short_read
        batch, seq_len, hidden = expression.shape
        query = F.normalize(expression, dim=-1)
        # Keys are normalized AFTER the admitted-bank gather (see gather_bank):
        # F.normalize is row-wise and the gather selects whole rows, so
        # normalize(gather(write)) == gather(normalize(write)) exactly, but the
        # backward now only pins the [B, n_max, H] gathered keys instead of a
        # full [B, T, H] normalized-write tape.
        scale = (F.softplus(read_parameters.log_temp) + 1.0) * (hidden ** 0.5)
        key_bias = (
            read_parameters.p_gain * torch.log1p(plasticity.mean(dim=-1) * hidden)
            + read_parameters.novelty_gain * novelty.mean(dim=-1)
            + read_parameters.null_bias
        )
        admission = read_parameters.admission(
            novelty,
            plasticity,
            energy,
            saturation,
        )
        cocktail = F.normalize(
            torch.stack(
                [
                    energy.mean(dim=-1),
                    saturation.mean(dim=-1),
                    novelty.mean(dim=-1),
                    plasticity.mean(dim=-1),
                ],
                dim=-1,
            ),
            dim=-1,
        )

        window = F.softplus(self.admit_window)
        if self.read_geometry == "seq":
            admit_floor = torch.cummax(admission, dim=1).values - window
            admission_policy = "causal_write_prefix_max"
        else:
            admit_floor = admission.max(dim=1, keepdim=True).values - window
            admission_policy = "whole_sequence_max"
        member = admission >= admit_floor
        self.last_admit_gate_mean = None

        counts = member.sum(dim=1)
        order = torch.argsort(
            member.to(query.dtype),
            dim=1,
            descending=True,
            stable=True,
        )
        # Bank-width policy. The read scores every query position against every
        # admitted bank entry, so the dense tensor-core path costs O(T * n_max).
        # Sizing the bank to the data-dependent admitted count
        # (``counts.max().item()``) keeps that sub-quadratic -- O(T * admitted)
        # -- which is the entire point of the sparse admitted read. The only
        # catch is that ``.item()`` is a host sync, which is illegal *while a
        # CUDA graph is being captured*.
        #
        # This used to be gated on ``_cuda_native_enabled``, which is set for
        # the whole GPU session: that forced the full-width (n_max == seq_len)
        # bank on EVERY CUDA step, silently turning the sparse read into a full
        # O(T^2) attention on all GPU training -- not just under capture. That
        # is the context-length blow-up. Gate strictly on *actual* stream
        # capture instead: ordinary GPU training (and the eager warmup steps of
        # a graphed run) sizes the bank to the admitted count and stays sparse;
        # only the narrow capture window takes a static width, and even there an
        # explicit measured cap (``_capture_bank_width``) keeps it sub-quadratic
        # when the harness provides one. ``_capture_safe_bank`` remains an
        # explicit manual override for callers that need the static path.
        capturing = expression.is_cuda and _stream_is_capturing()
        capture_safe_bank = bool(getattr(self, "_capture_safe_bank", False)) or capturing
        if capture_safe_bank:
            cap = getattr(self, "_capture_bank_width", None)
            n_max = int(cap) if cap else seq_len
            n_max = max(1, min(n_max, seq_len))
        else:
            n_max = max(int(counts.max().item()), 1)
        self.last_n_max = n_max
        seq_idx = order[:, :n_max]
        seq_valid = (
            torch.arange(n_max, device=expression.device).unsqueeze(0)
            < counts.clamp_min(1).unsqueeze(1)
        )

        def gather_bank(index: Tensor):
            gather_hidden = index.unsqueeze(-1).expand(-1, -1, hidden)
            bank_writes = torch.gather(write, 1, gather_hidden)
            # Normalize the gathered writes (bit-identical to gathering a full
            # normalized-key tape, but O(n_max) not O(T) in the saved graph).
            bank_keys = F.normalize(bank_writes, dim=-1)
            bank_cocktail = torch.gather(
                cocktail,
                1,
                index.unsqueeze(-1).expand(-1, -1, cocktail.shape[-1]),
            )
            bank_key_bias = torch.gather(key_bias, 1, index)
            bank_admission = torch.gather(admission, 1, index)
            return bank_keys, bank_writes, bank_cocktail, bank_key_bias, bank_admission

        seq_keys, seq_writes, seq_cocktail, seq_bias, seq_admission = gather_bank(seq_idx)
        field_idx = seq_idx
        field_valid = seq_valid
        field_keys, field_writes, field_cocktail = seq_keys, seq_writes, seq_cocktail
        field_bias, field_admission = seq_bias, seq_admission
        read_paths = 2 if self.read_geometry == "hybrid" else 1
        self.last_read_contract = {
            "read_strategy": "eager_admitted_bank",
            "exact": True,
            "approximation_reason": None,
            "bank_growth": "grows_with_T" if n_max >= seq_len else "data_dependent",
            "stored_writes_preserved": True,
            "max_admitted_bank": int(n_max),
            "estimated_read_work": int(batch) * int(seq_len) * int(n_max) * 3 * read_paths,
            "dense_masks_materialized": True,
            "context_parallel": False,
            "admission_policy": admission_policy,
            "kernel_backend": "eager_reference",
        }

        compact_cuda = expression.is_cuda and bool(
            getattr(type(self), "_cuda_native_enabled", False)
        )
        if compact_cuda:
            query_pos = None
            seq_allow = seq_eligible = None
            field_allow = field_eligible = None
        else:
            query_pos = torch.arange(seq_len, device=expression.device).view(1, seq_len, 1)
            seq_allow = (seq_idx.unsqueeze(1) <= query_pos) & seq_valid.unsqueeze(1)
            seq_eligible = seq_allow.any(dim=-1)
            field_allow = field_valid.unsqueeze(1).expand(-1, seq_len, -1)
            field_eligible = field_allow.any(dim=-1)
        trace_enabled = self.collect_traces or not expression.is_cuda
        if self.read_geometry != "seq" and trace_enabled:
            self.last_field_neighbors = float(
                field_valid.detach().sum(dim=1).float().mean().cpu()
            )
        elif self.read_geometry != "seq":
            self.last_field_neighbors = None

        next_idx = (seq_idx + 1).clamp(max=seq_len - 1)
        next_writes = torch.gather(
            write,
            1,
            next_idx.unsqueeze(-1).expand(-1, -1, hidden),
        )
        if compact_cuda:
            seq_induct_allow = seq_induct_eligible = None
            field_induct_allow = field_induct_eligible = None
        else:
            seq_induct_allow = (seq_idx.unsqueeze(1) < query_pos) & seq_valid.unsqueeze(1)
            field_induct_allow = (
                (field_idx.unsqueeze(1) < (seq_len - 1)) & field_valid.unsqueeze(1)
            ).expand(-1, seq_len, -1)
            seq_induct_eligible = seq_induct_allow.any(dim=-1)
            field_induct_eligible = field_induct_allow.any(dim=-1)
        next_field_writes = None
        if self.read_geometry != "seq":
            next_field_writes = torch.gather(
                write,
                1,
                (field_idx + 1)
                .clamp(max=seq_len - 1)
                .unsqueeze(-1)
                .expand(-1, -1, hidden),
            )

        if self.read_geometry == "seq":
            self._compact_bank_idx = seq_idx
            self._compact_bank_valid = seq_valid
            self._compact_read_mode = "seq"
            retrieved = self._three_way_read(
                query,
                seq_keys,
                seq_writes,
                next_writes,
                cocktail,
                seq_cocktail,
                seq_bias,
                seq_admission,
                scale,
                seq_allow,
                seq_induct_allow,
                seq_eligible,
                seq_induct_eligible,
            )
            self.last_hybrid_gate = None
            self.last_seq_norm = (
                float(retrieved.detach().norm().cpu()) if trace_enabled else None
            )
            self.last_field_norm = None
            self.last_field_neighbors = None
        elif self.read_geometry == "field":
            self._compact_bank_idx = field_idx
            self._compact_bank_valid = field_valid
            self._compact_read_mode = "field"
            retrieved = self._three_way_read(
                query,
                field_keys,
                field_writes,
                next_field_writes,
                cocktail,
                field_cocktail,
                field_bias,
                field_admission,
                scale,
                field_allow,
                field_induct_allow,
                field_eligible,
                field_induct_eligible,
            )
            self.last_hybrid_gate = None
            self.last_seq_norm = None
            self.last_field_norm = (
                float(retrieved.detach().norm().cpu()) if trace_enabled else None
            )
        else:
            self._compact_bank_idx = seq_idx
            self._compact_bank_valid = seq_valid
            self._compact_read_mode = "seq"
            seq_retrieved = self._three_way_read(
                query,
                seq_keys,
                seq_writes,
                next_writes,
                cocktail,
                seq_cocktail,
                seq_bias,
                seq_admission,
                scale,
                seq_allow,
                seq_induct_allow,
                seq_eligible,
                seq_induct_eligible,
            )
            self._compact_bank_idx = field_idx
            self._compact_bank_valid = field_valid
            self._compact_read_mode = "field"
            field_retrieved = self._three_way_read(
                query,
                field_keys,
                field_writes,
                next_field_writes,
                cocktail,
                field_cocktail,
                field_bias,
                field_admission,
                scale,
                field_allow,
                field_induct_allow,
                field_eligible,
                field_induct_eligible,
            )
            if self.hybrid_gate_override is None:
                gate = torch.sigmoid(self.hybrid_gate)
            else:
                gate = torch.as_tensor(
                    float(self.hybrid_gate_override),
                    device=query.device,
                    dtype=query.dtype,
                )
            self.last_hybrid_gate = float(gate.detach().cpu()) if trace_enabled else None
            seq_contribution = (1.0 - gate) * seq_retrieved
            field_contribution = gate * field_retrieved
            self.last_seq_norm = (
                float(seq_contribution.detach().norm().cpu()) if trace_enabled else None
            )
            self.last_field_norm = (
                float(field_contribution.detach().norm().cpu()) if trace_enabled else None
            )
            retrieved = seq_contribution + field_contribution

        self.last_read_contract.update(
            {
                "read_strategy": (
                    "compact_flash_admitted_bank" if compact_cuda else "eager_admitted_bank"
                ),
                "dense_masks_materialized": not compact_cuda,
                "kernel_backend": getattr(
                    self, "_last_three_way_backend", "eager_reference"
                ),
            }
        )

        output = read_parameters.out(torch.cat([y, budget, retrieved], dim=-1))
        long = self._long_scan(write, plasticity, novelty)
        self.last_long_scan_norm = (
            float(long.detach().norm().cpu()) if trace_enabled else None
        )
        return output + self.long_gain * long

    def _three_way_read(
        self,
        query: Tensor,
        bank_keys: Tensor,
        bank_writes: Tensor,
        next_writes: Tensor,
        cocktail: Tensor,
        bank_cocktail: Tensor,
        bank_key_bias: Tensor,
        bank_admission: Tensor,
        scale: Tensor,
        allow: Tensor,
        induct_allow: Tensor,
        eligible: Tensor,
        induct_eligible: Tensor,
    ) -> Tensor:
        compatibility = torch.bmm(query, bank_keys.transpose(1, 2)) * scale
        cocktail_compatibility = (
            torch.bmm(cocktail, bank_cocktail.transpose(1, 2)) * self.cocktail_gain
        )
        content = compatibility + bank_key_bias.unsqueeze(1)
        short_scores = content + bank_admission.unsqueeze(1)
        permanent_scores = content + cocktail_compatibility + bank_admission.unsqueeze(1)

        def read(
            scores: Tensor,
            values: Tensor,
            allow_mask: Tensor,
            has_eligible: Tensor,
        ) -> Tensor:
            scores = scores.masked_fill(~allow_mask, float("-inf"))
            scores = scores.masked_fill(~has_eligible.unsqueeze(-1), 0.0)
            weights = F.softmax(scores, dim=-1)
            weights = torch.where(
                has_eligible.unsqueeze(-1),
                weights,
                torch.zeros_like(weights),
            )
            return torch.bmm(weights, values)

        return (
            self.short_gain * read(short_scores, bank_writes, allow, eligible)
            + self.pad_gain * read(permanent_scores, bank_writes, allow, eligible)
            + self.induct_gain
            * read(short_scores, next_writes, induct_allow, induct_eligible)
        )

    def initial_long_scan_state(
        self,
        batch_size: int,
        hidden_dim: int | None = None,
        *,
        device=None,
        dtype=None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = int(hidden_dim or self.short_read.out.out_features)
        reference = next(self.parameters())
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype
        long = torch.zeros(batch_size, hidden, device=device, dtype=dtype)
        expected = torch.zeros_like(long)
        retention = torch.ones_like(long)
        return long, expected, retention

    def _long_scan_from_state(
        self,
        write: Tensor,
        plasticity: Tensor,
        novelty: Tensor,
        *,
        initial_state: tuple[Tensor, Tensor, Tensor] | None = None,
        return_final_state: bool = False,
    ):
        batch, seq_len, hidden = write.shape
        retain = torch.sigmoid(self.logit_retain)
        if initial_state is None:
            long, expected, retention = self.initial_long_scan_state(
                batch,
                hidden,
                device=write.device,
                dtype=write.dtype,
            )
        else:
            long, expected, retention = initial_state
            long = long.to(device=write.device, dtype=write.dtype)
            expected = expected.to(device=write.device, dtype=write.dtype)
            retention = retention.to(device=write.device, dtype=write.dtype)
            if (
                long.shape != (batch, hidden)
                or expected.shape != (batch, hidden)
                or retention.shape != (batch, hidden)
            ):
                raise ValueError(
                    "initial long-scan state must have shape "
                    f"{(batch, hidden)}, got {tuple(long.shape)}, "
                    f"{tuple(expected.shape)}, {tuple(retention.shape)}"
                )
        expect_retain = torch.sigmoid(self.logit_expect_retain)
        retention_decay = torch.sigmoid(self.logit_retention_decay)
        retention_strength = torch.sigmoid(self.logit_retention_strength)
        outputs: list[Tensor] = []
        for step in range(seq_len):
            prediction_error = torch.tanh((write[:, step, :] - expected).abs())
            plastic_salience = plasticity[:, step, :] * prediction_error
            expected = (
                expect_retain * expected
                + (1.0 - expect_retain) * write[:, step, :]
            )
            retention = (
                retention_decay * retention
                + (1.0 - retention_decay) * (1.0 - prediction_error)
            )
            effective_retain = (
                retain + (1.0 - retain) * retention_strength * retention
            )
            long = (
                effective_retain * long
                + (1.0 - effective_retain)
                * (plastic_salience * write[:, step, :])
            )
            outputs.append(long)
        output = torch.stack(outputs, dim=1)
        if return_final_state:
            return output, (long, expected, retention)
        return output

    def _long_scan(
        self,
        write: Tensor,
        plasticity: Tensor,
        novelty: Tensor,
    ) -> Tensor:
        return self._long_scan_from_state(write, plasticity, novelty)

    def forward_query_bank(
        self,
        y: Tensor,
        budget: Tensor,
        expression: Tensor,
        write: Tensor,
        novelty: Tensor,
        plasticity: Tensor,
        energy: Tensor,
        saturation: Tensor,
        *,
        bank_y: Tensor,
        bank_b: Tensor,
        bank_ay: Tensor,
        bank_write: Tensor,
        bank_novelty: Tensor,
        bank_p: Tensor,
        bank_energy: Tensor,
        bank_saturation: Tensor,
        query_start: int,
        initial_long_state: tuple[Tensor, Tensor, Tensor] | None = None,
        return_long_state: bool = False,
    ):
        del bank_y, bank_b, bank_ay
        if self.read_geometry != "seq":
            raise RuntimeError("query-bank read is exact only for seq geometry")
        read_parameters = self.short_read
        batch, query_len, hidden = expression.shape
        bank_len = bank_write.shape[1]
        if bank_len <= 0:
            raise RuntimeError("query-bank read requires a non-empty bank")
        query_start = int(query_start)

        query = F.normalize(expression, dim=-1)
        # Keys normalized after the gather (Phase 2c) -- see the short read: the
        # row-wise normalize commutes with the row gather bit-for-bit.
        scale = (F.softplus(read_parameters.log_temp) + 1.0) * (hidden ** 0.5)
        key_bias = (
            read_parameters.p_gain * torch.log1p(bank_p.mean(dim=-1) * hidden)
            + read_parameters.novelty_gain * bank_novelty.mean(dim=-1)
            + read_parameters.null_bias
        )
        admission = read_parameters.admission(
            bank_novelty,
            bank_p,
            bank_energy,
            bank_saturation,
        )
        query_cocktail = F.normalize(
            torch.stack(
                [
                    energy.mean(dim=-1),
                    saturation.mean(dim=-1),
                    novelty.mean(dim=-1),
                    plasticity.mean(dim=-1),
                ],
                dim=-1,
            ),
            dim=-1,
        )
        bank_cocktail = F.normalize(
            torch.stack(
                [
                    bank_energy.mean(dim=-1),
                    bank_saturation.mean(dim=-1),
                    bank_novelty.mean(dim=-1),
                    bank_p.mean(dim=-1),
                ],
                dim=-1,
            ),
            dim=-1,
        )

        window = F.softplus(self.admit_window)
        admit_floor = torch.cummax(admission, dim=1).values - window
        member = admission >= admit_floor
        counts = member.sum(dim=1)
        n_max = max(int(counts.max().item()), 1)
        order = torch.argsort(
            member.to(query.dtype),
            dim=1,
            descending=True,
            stable=True,
        )
        bank_idx = order[:, :n_max]
        bank_valid = (
            torch.arange(n_max, device=expression.device).unsqueeze(0)
            < counts.clamp_min(1).unsqueeze(1)
        )

        gather_hidden = bank_idx.unsqueeze(-1).expand(-1, -1, hidden)
        bank_writes = torch.gather(bank_write, 1, gather_hidden)
        bank_keys = F.normalize(bank_writes, dim=-1)
        selected_cocktail = torch.gather(
            bank_cocktail,
            1,
            bank_idx.unsqueeze(-1).expand(-1, -1, bank_cocktail.shape[-1]),
        )
        selected_bias = torch.gather(key_bias, 1, bank_idx)
        selected_admission = torch.gather(admission, 1, bank_idx)
        next_idx = (bank_idx + 1).clamp(max=bank_len - 1)
        next_writes = torch.gather(
            bank_write,
            1,
            next_idx.unsqueeze(-1).expand(-1, -1, hidden),
        )

        compact_cuda = expression.is_cuda and bool(
            getattr(type(self), "_cuda_native_enabled", False)
        )
        if compact_cuda:
            allow = induct_allow = eligible = induct_eligible = None
        else:
            query_pos = (
                query_start + torch.arange(query_len, device=expression.device)
            ).view(1, query_len, 1)
            allow = (bank_idx.unsqueeze(1) <= query_pos) & bank_valid.unsqueeze(1)
            induct_allow = (bank_idx.unsqueeze(1) < query_pos) & bank_valid.unsqueeze(1)
            eligible = allow.any(dim=-1)
            induct_eligible = induct_allow.any(dim=-1)
        self._compact_bank_idx = bank_idx - query_start
        self._compact_bank_valid = bank_valid
        self._compact_read_mode = "seq"
        retrieved = self._three_way_read(
            query,
            bank_keys,
            bank_writes,
            next_writes,
            query_cocktail,
            selected_cocktail,
            selected_bias,
            selected_admission,
            scale,
            allow,
            induct_allow,
            eligible,
            induct_eligible,
        )
        kernel_backend = getattr(self, "_last_three_way_backend", "eager_reference")

        trace_enabled = self.collect_traces or not expression.is_cuda
        self.last_n_max = n_max
        self.last_hybrid_gate = None
        self.last_seq_norm = (
            float(retrieved.detach().norm().cpu()) if trace_enabled else None
        )
        self.last_field_norm = None
        self.last_field_neighbors = None
        self.last_read_contract = {
            "read_strategy": (
                "query_only_compact_flash_admitted_bank"
                if compact_cuda
                else "query_only_eager_admitted_bank"
            ),
            "exact": True,
            "approximation_reason": None,
            "bank_growth": "grows_with_T" if n_max >= bank_len else "data_dependent",
            "stored_writes_preserved": True,
            "max_admitted_bank": int(n_max),
            "estimated_read_work": int(batch) * int(query_len) * int(n_max) * 3,
            "dense_masks_materialized": not compact_cuda,
            "context_parallel": True,
            "query_only": True,
            "query_start": query_start,
            "bank_entries": int(bank_len),
            "admission_policy": "causal_write_prefix_max",
            "kernel_backend": kernel_backend,
        }

        short_output = read_parameters.out(torch.cat([y, budget, retrieved], dim=-1))
        long_result = self._long_scan_from_state(
            write,
            plasticity,
            novelty,
            initial_state=initial_long_state,
            return_final_state=return_long_state,
        )
        if return_long_state:
            long, final_long_state = long_result
        else:
            long = long_result
            final_long_state = None
        self.last_long_scan_norm = (
            float(long.detach().norm().cpu()) if trace_enabled else None
        )
        output = short_output + self.long_gain * long
        if return_long_state:
            return output, final_long_state
        return output
