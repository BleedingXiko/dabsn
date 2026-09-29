"""Native dense/sparse parametric Capacity subsystem for canonical DABSN blocks.

Capacity is not a model-builder wrapper. It is an optional peer subsystem owned by
``DABSNBlock`` and configured through the canonical DABSN config/model builders.

Dense mode executes every Capacity row. Sparse mode uses a frozen geometric address
projection and executes exactly ``active_units`` rows per token. Sparse online credit
returns row-sparse gradients for the Capacity row parameters and updates recurrent /
eligibility carry only at routed rows.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import CapacityConfig


def _row_sparse(rows: Tensor, values: Tensor, size: tuple[int, ...]) -> Tensor:
    return torch.sparse_coo_tensor(
        rows.unsqueeze(0), values, size=size, device=values.device, dtype=values.dtype
    ).coalesce()


class _DenseOnlineCapacity(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        key: Tensor,
        bias: Tensor,
        retention_logit: Tensor,
        value: Tensor,
        gain: Tensor,
        prev: Tensor,
        ekey: Tensor,
        ebias: Tensor,
        erho: Tensor,
    ):
        rho = torch.sigmoid(retention_logit)
        z = prev
        batch, steps, _ = x.shape
        state_dim = int(value.shape[1])
        out = x.new_empty(batch, steps, state_dim)
        acts = []
        previous = []
        states = []
        for t in range(steps):
            old = z
            a = torch.tanh(F.linear(x[:, t], key, bias))
            z = rho * z + (1.0 - rho) * a
            out[:, t] = gain * (z @ value)
            acts.append(a)
            previous.append(old)
            states.append(z)
        ctx.save_for_backward(
            x, key, rho, value, gain, ekey, ebias, erho,
            torch.stack(acts, dim=1), torch.stack(previous, dim=1),
            torch.stack(states, dim=1),
        )
        ctx.set_materialize_grads(False)
        final = z.detach()
        ctx.mark_non_differentiable(final)
        return out, final

    @staticmethod
    def backward(ctx, gout, gfinal):
        (x, key, rho, value, gain, ekey, ebias, erho,
         acts, previous, states) = ctx.saved_tensors
        if gout is None:
            gout = torch.zeros(
                (*x.shape[:2], value.shape[1]), device=x.device, dtype=x.dtype
            )
        calc_dtype = ekey.dtype
        ek = ekey
        eb = ebias
        er = erho
        grad_key = torch.zeros_like(key, dtype=calc_dtype)
        grad_bias = torch.zeros_like(rho, dtype=calc_dtype)
        grad_rho = torch.zeros_like(rho, dtype=calc_dtype)
        grad_value = torch.zeros_like(value, dtype=calc_dtype)
        grad_gain = torch.zeros((), device=gain.device, dtype=calc_dtype)
        grad_x = torch.zeros_like(x)
        rho_e = rho.to(calc_dtype)
        slope_rho = rho_e * (1.0 - rho_e)
        key_e = key.to(calc_dtype)
        value_e = value.to(calc_dtype)
        gain_e = gain.to(calc_dtype)
        for t in range(x.shape[1]):
            xt = x[:, t].to(calc_dtype)
            a = acts[:, t].to(calc_dtype)
            old = previous[:, t].to(calc_dtype)
            z = states[:, t].to(calc_dtype)
            s = (1.0 - rho_e) * (1.0 - a.square())
            ek = rho_e[None, :, None] * ek + s[:, :, None] * xt[:, None, :]
            eb = rho_e[None, :] * eb + s
            er = rho_e[None, :] * er + slope_rho[None, :] * (old - a)
            cot = gout[:, t].to(calc_dtype)
            gz = gain_e * (cot @ value_e.transpose(0, 1))
            grad_key += torch.einsum("bc,bcd->cd", gz, ek)
            grad_bias += (gz * eb).sum(0)
            grad_rho += (gz * er).sum(0)
            grad_value += torch.einsum("bc,bs->cs", gain_e * z, cot)
            grad_gain += (cot * (z @ value_e)).sum()
            grad_x[:, t] = ((gz * s) @ key_e).to(x.dtype)
        # Persistent eligibility carry is detached and intentionally updated by replay.
        ekey.copy_(ek.detach())
        ebias.copy_(eb.detach())
        erho.copy_(er.detach())
        return (
            grad_x,
            grad_key.to(key.dtype),
            grad_bias.to(rho.dtype),
            grad_rho.to(rho.dtype),
            grad_value.to(value.dtype),
            grad_gain.to(gain.dtype),
            None, None, None, None,
        )


class _SparseOnlineCapacity(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        key: Tensor,
        bias: Tensor,
        retention_logit: Tensor,
        value: Tensor,
        gain: Tensor,
        prev: Tensor,
        ekey: Tensor,
        ebias: Tensor,
        erho: Tensor,
        indices: Tensor,
    ):
        if indices.dtype != torch.long:
            raise TypeError("indices must be int64")
        batch, steps, _ = x.shape
        active = int(indices.shape[-1])
        state_dim = int(value.shape[1])
        rho = torch.sigmoid(retention_logit)
        # Online state is consumed carry. Update only routed rows, avoiding O(C) copy.
        z = prev
        out = x.new_empty(batch, steps, state_dim)
        acts = x.new_empty(batch, steps, active)
        previous = x.new_empty(batch, steps, active)
        states = x.new_empty(batch, steps, active)
        batch_ix = torch.arange(batch, device=x.device)[:, None]
        for t in range(steps):
            idx = indices[:, t]
            old = z[batch_ix, idx]
            krow = key[idx]
            brow = bias[idx]
            r = rho[idx]
            a = torch.tanh(torch.einsum("bd,bkd->bk", x[:, t], krow) + brow)
            new = r * old + (1.0 - r) * a
            z[batch_ix, idx] = new
            out[:, t] = gain * torch.einsum("bk,bks->bs", new, value[idx])
            acts[:, t] = a
            previous[:, t] = old
            states[:, t] = new
        ctx.save_for_backward(
            x, key, rho, value, gain, ekey, ebias, erho,
            indices, acts, previous, states,
        )
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(prev)
        return out, prev

    @staticmethod
    def backward(ctx, gout, gfinal):
        (x, key, rho, value, gain, ekey, ebias, erho,
         indices, acts, previous, states) = ctx.saved_tensors
        if gout is None:
            gout = torch.zeros(
                (*x.shape[:2], value.shape[1]), device=x.device, dtype=x.dtype
            )
        calc_dtype = ekey.dtype
        unique_rows = torch.unique(indices.reshape(-1), sorted=True)
        nrows = int(unique_rows.numel())
        dim = int(key.shape[1])
        state_dim = int(value.shape[1])
        gkey_rows = torch.zeros(nrows, dim, device=x.device, dtype=calc_dtype)
        gbias_rows = torch.zeros(nrows, device=x.device, dtype=calc_dtype)
        grho_rows = torch.zeros(nrows, device=x.device, dtype=calc_dtype)
        gvalue_rows = torch.zeros(nrows, state_dim, device=x.device, dtype=calc_dtype)
        ggain = torch.zeros((), device=x.device, dtype=calc_dtype)
        gx = torch.zeros_like(x)
        ek, eb, er = ekey, ebias, erho
        batch_ix = torch.arange(x.shape[0], device=x.device)[:, None]
        rho_e = rho.to(calc_dtype)
        slope_rho = rho_e * (1.0 - rho_e)
        gain_e = gain.to(calc_dtype)
        for t in range(x.shape[1]):
            idx = indices[:, t]
            local = torch.searchsorted(unique_rows, idx.contiguous())
            r = rho_e[idx]
            a = acts[:, t].to(calc_dtype)
            old = previous[:, t].to(calc_dtype)
            z = states[:, t].to(calc_dtype)
            xt = x[:, t].to(calc_dtype)
            krow = key[idx].to(calc_dtype)
            vrow = value[idx].to(calc_dtype)
            ek_old = ek[batch_ix, idx]
            eb_old = eb[batch_ix, idx]
            er_old = er[batch_ix, idx]
            slope = (1.0 - r) * (1.0 - a.square())
            ek_new = r[..., None] * ek_old + slope[..., None] * xt[:, None, :]
            eb_new = r * eb_old + slope
            er_new = r * er_old + slope_rho[idx] * (old - a)
            cot = gout[:, t].to(calc_dtype)
            gz = gain_e * torch.einsum("bs,bks->bk", cot, vrow)
            flat = local.reshape(-1)
            gkey_rows.index_add_(0, flat, (gz[..., None] * ek_new).reshape(-1, dim))
            gbias_rows.index_add_(0, flat, (gz * eb_new).reshape(-1))
            grho_rows.index_add_(0, flat, (gz * er_new).reshape(-1))
            gvalue_rows.index_add_(
                0, flat, (gain_e * z[..., None] * cot[:, None, :]).reshape(-1, state_dim)
            )
            ggain += (cot * torch.einsum("bk,bks->bs", z, vrow)).sum()
            gx[:, t] = torch.einsum("bk,bkd->bd", gz * slope, krow).to(x.dtype)
            ek[batch_ix, idx] = ek_new.detach()
            eb[batch_ix, idx] = eb_new.detach()
            er[batch_ix, idx] = er_new.detach()
        return (
            gx,
            _row_sparse(unique_rows, gkey_rows.to(key.dtype), tuple(key.shape)),
            _row_sparse(unique_rows, gbias_rows.to(rho.dtype), tuple(rho.shape)),
            _row_sparse(unique_rows, grho_rows.to(rho.dtype), tuple(rho.shape)),
            _row_sparse(unique_rows, gvalue_rows.to(value.dtype), tuple(value.shape)),
            ggain.to(gain.dtype),
            None, None, None, None, None,
        )


class DABSNCapacitySubsystem(nn.Module):
    """Capacity state transform owned directly by one canonical DABSN block."""

    def __init__(self, input_dim: int, state_dim: int, config: CapacityConfig) -> None:
        super().__init__()
        cfg = config.validate()
        if not cfg.enabled:
            raise ValueError("disabled Capacity must not construct a subsystem")
        self.config = cfg
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.size = int(cfg.size)
        self.mode = cfg.mode
        self.active_units = self.size if self.mode == "dense" else int(cfg.active_units)
        self.key = nn.Parameter(torch.empty(self.size, self.input_dim))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.value = nn.Parameter(torch.zeros(self.size, self.state_dim))
        nn.init.normal_(self.key, std=0.10)
        self.retention_logit = nn.Parameter(torch.full(
            (self.size,), math.log(cfg.initial_retention / (1.0 - cfg.initial_retention))
        ))
        self.gain = nn.Parameter(torch.tensor(float(cfg.injection_gain)))

        self.num_buckets = 1
        self.bucket_size = self.size
        self.address_branching: int | None = None
        self.address_levels = 1
        if self.mode == "sparse":
            self.num_buckets = self.size // self.active_units
            self.bucket_size = self.active_units
            self.address_branching = cfg.resolve_product_branching(self.num_buckets)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(cfg.address_seed))
            if self.address_branching is None:
                proj = torch.randn(self.num_buckets, self.input_dim, generator=gen)
                proj = F.normalize(proj, dim=-1)
            else:
                n = self.num_buckets
                levels = 0
                while n > 1:
                    n //= self.address_branching
                    levels += 1
                self.address_levels = levels
                proj = torch.randn(
                    self.address_levels, self.address_branching, self.input_dim, generator=gen
                )
                proj = F.normalize(proj, dim=-1)
            self.register_buffer("address_projection", proj)
            self.register_buffer(
                "bucket_offsets", torch.arange(self.bucket_size, dtype=torch.long)
            )

    @property
    def active_fraction(self) -> float:
        return self.active_units / self.size

    def route(self, x: Tensor) -> Tensor:
        if self.mode != "sparse":
            raise RuntimeError("dense Capacity has no sparse route")
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"capacity route expects [B,T,{self.input_dim}]")
        proj = self.address_projection.to(dtype=x.dtype)
        if self.address_branching is None:
            bucket = F.linear(x, proj).argmax(dim=-1)
        else:
            bucket = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
            for level in range(self.address_levels):
                digit = F.linear(x, proj[level]).argmax(dim=-1)
                bucket = bucket * self.address_branching + digit
        units = bucket[..., None] * self.bucket_size + self.bucket_offsets
        return units

    def initial_online_state(
        self,
        batch: int,
        *,
        device,
        recurrent_dtype: torch.dtype,
        eligibility_dtype: torch.dtype,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        recurrent = {
            "features": torch.zeros(batch, self.size, device=device, dtype=recurrent_dtype)
        }
        eligibility = {
            "key": torch.zeros(
                batch, self.size, self.input_dim, device=device, dtype=eligibility_dtype
            ),
            "bias": torch.zeros(batch, self.size, device=device, dtype=eligibility_dtype),
            "retention": torch.zeros(batch, self.size, device=device, dtype=eligibility_dtype),
        }
        return recurrent, eligibility

    def forward_sequence(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"capacity input must be [B,T,{self.input_dim}]")
        features = x.new_zeros(x.shape[0], self.size)
        outputs = []
        rho = torch.sigmoid(self.retention_logit)
        if self.mode == "dense":
            for t in range(x.shape[1]):
                a = torch.tanh(F.linear(x[:, t], self.key, self.bias))
                features = rho * features + (1.0 - rho) * a
                outputs.append(self.gain * (features @ self.value))
        else:
            routes = self.route(x)
            batch_ix = torch.arange(x.shape[0], device=x.device)[:, None]
            for t in range(x.shape[1]):
                idx = routes[:, t]
                old = features[batch_ix, idx]
                r = rho[idx]
                a = torch.tanh(
                    torch.einsum("bd,bkd->bk", x[:, t], self.key[idx]) + self.bias[idx]
                )
                new = r * old + (1.0 - r) * a
                # Functional scatter keeps ordinary BPTT semantics.
                features = features.scatter(1, idx, new)
                outputs.append(self.gain * torch.einsum("bk,bks->bs", new, self.value[idx]))
        if outputs:
            return torch.stack(outputs, dim=1)
        return x.new_empty(x.shape[0], 0, self.state_dim)

    def forward_online(
        self,
        x: Tensor,
        recurrent: Mapping[str, Tensor],
        eligibility: Mapping[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        features = recurrent["features"]
        key_e = eligibility["key"]
        bias_e = eligibility["bias"]
        retention_e = eligibility["retention"]
        if self.mode == "dense":
            out, final = _DenseOnlineCapacity.apply(
                x, self.key, self.bias, self.retention_logit,
                self.value, self.gain, features, key_e, bias_e, retention_e,
            )
        else:
            out, final = _SparseOnlineCapacity.apply(
                x, self.key, self.bias, self.retention_logit,
                self.value, self.gain, features, key_e, bias_e, retention_e,
                self.route(x),
            )
        return (
            out,
            {"features": final.detach()},
            {"key": key_e, "bias": bias_e, "retention": retention_e},
        )

    @torch.no_grad()
    def routing_health(self, x: Tensor) -> dict[str, object]:
        if self.mode != "sparse":
            return {
                "mode": "dense",
                "active_units": self.size,
                "active_fraction": 1.0,
                "normalized_entropy": 1.0,
                "dead_buckets": 0,
                "max_to_mean_load": 1.0,
            }
        indices = self.route(x)
        buckets = torch.div(indices[..., 0], self.bucket_size, rounding_mode="floor").reshape(-1)
        counts = torch.bincount(buckets, minlength=self.num_buckets)
        total = int(counts.sum().item())
        if total == 0:
            raise ValueError("routing_health requires at least one routed token")
        p = counts.to(torch.float64) / total
        nz = p[p > 0]
        entropy = 1.0 if self.num_buckets == 1 else float(
            (-(nz * nz.log()).sum() / math.log(self.num_buckets)).item()
        )
        mean = counts.to(torch.float64).mean()
        return {
            "mode": "sparse",
            "tokens": int(x.shape[0] * x.shape[1]),
            "num_buckets": self.num_buckets,
            "active_units": self.active_units,
            "active_fraction": self.active_fraction,
            "normalized_entropy": entropy,
            "dead_buckets": int((counts == 0).sum().item()),
            "dead_fraction": float((counts == 0).to(torch.float64).mean().item()),
            "max_to_mean_load": float((counts.max().to(torch.float64) / mean).item()),
            "counts": counts.cpu(),
        }

    def sparse_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.mode != "sparse":
            return ()
        return (self.key, self.bias, self.retention_logit, self.value)


def capacity_config(value: CapacityConfig | Mapping[str, Any] | None) -> CapacityConfig | None:
    if value is None:
        return None
    if isinstance(value, CapacityConfig):
        cfg = value
    elif isinstance(value, Mapping):
        cfg = CapacityConfig(**dict(value))
    else:
        raise TypeError("capacity must be CapacityConfig, mapping, or None")
    cfg.validate()
    return cfg if cfg.enabled else None


__all__ = ["DABSNCapacitySubsystem", "capacity_config"]
