"""Paged, exactly admitted carried-memory state for large-bank inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence, cast
from weakref import WeakKeyDictionary

import torch
import torch.nn.functional as F
from torch import Tensor


FORMAT = "dabsn_prepared_memory_v2"
_IDENTITIES: WeakKeyDictionary = WeakKeyDictionary()


def prepared_memory_bytes(
    *,
    batch: int,
    committed_entries: int,
    hidden: int,
    page_size: int,
    dtype: torch.dtype,
    query_tile: int | None = None,
) -> dict[str, int]:
    """Exact persistent-bank bytes and a conservative page-query workspace."""
    if batch <= 0 or committed_entries < 0 or hidden <= 0 or page_size <= 0:
        raise ValueError("batch/hidden/page_size must be positive and entries non-negative")
    pages = max((committed_entries + page_size - 1) // page_size, 1)
    padded = pages * page_size
    value_bytes = torch.empty((), dtype=dtype).element_size()
    # Inverse norms remain fp32 even when stored values are low precision.
    norm_bytes = max(value_bytes, 4)
    per_entry = (2 * hidden + 6) * value_bytes + norm_bytes + 8 + 1
    bank_state = batch * (8 + value_bytes + 8 + 8)
    persistent = batch * padded * per_entry + bank_state
    q = int(query_tile if query_tile is not None else 1)
    if q <= 0:
        raise ValueError("query_tile must be positive")
    # Bound the explicit tensor workspace conservatively: old and new three-way
    # accumulators coexist across the opaque operator boundary, as do temporary
    # score/mask/exp/reduction results and promoted page values. Compiler/library
    # caches and allocator fragmentation are outside this tensor-only estimate.
    compute_bytes = max(value_bytes, 4)
    workspace = batch * (
        64 * q * page_size + 32 * q * hidden + 4 * page_size * hidden
    ) * compute_bytes
    return {
        "page_count": pages,
        "padded_entries": padded,
        "persistent_bytes": persistent,
        "workspace_bytes_upper_bound": workspace,
    }


@dataclass
class PreparedBankPage:
    """One fixed-capacity page of admitted events."""

    write: Tensor
    successor_write: Tensor
    position: Tensor
    scalar_features: Tensor
    inverse_write_norm: Tensor
    admission: Tensor
    key_bias: Tensor
    valid: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.write, Tensor) or self.write.ndim != 3 or not self.write.is_floating_point():
            raise ValueError("prepared write must be a floating tensor of shape [B,P,H]")
        batch, capacity, hidden = self.write.shape
        if min(batch, capacity, hidden) <= 0:
            raise ValueError("prepared page dimensions must be positive")
        shapes = {"successor_write": (batch, capacity, hidden), "position": (batch, capacity),
                  "scalar_features": (batch, capacity, 4), "inverse_write_norm": (batch, capacity),
                  "admission": (batch, capacity), "key_bias": (batch, capacity), "valid": (batch, capacity)}
        for name, shape in shapes.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"prepared {name} must have shape {shape}")
            if value.device != self.write.device:
                raise ValueError(f"prepared {name} device must match write")
            expected_dtype = (torch.int64 if name == "position" else torch.bool if name == "valid"
                              else None if name == "inverse_write_norm" else self.write.dtype)
            if expected_dtype is not None and value.dtype != expected_dtype:
                raise ValueError(f"prepared {name} must have dtype {expected_dtype}")
        if self.inverse_write_norm.dtype not in (torch.float32, torch.float64):
            raise ValueError("prepared inverse norms must be float32 or float64")

    def to(self, *, device=None, dtype=None) -> "PreparedBankPage":
        def value(tensor: Tensor) -> Tensor:
            return tensor.to(device=device, dtype=dtype)

        return PreparedBankPage(
            write=value(self.write),
            successor_write=value(self.successor_write),
            position=self.position.to(device=device),
            scalar_features=value(self.scalar_features),
            inverse_write_norm=self.inverse_write_norm.to(device=device),
            admission=value(self.admission),
            key_bias=value(self.key_bias),
            valid=self.valid.to(device=device),
        )

    def clone(self) -> "PreparedBankPage":
        return PreparedBankPage(
            write=self.write.clone(),
            successor_write=self.successor_write.clone(),
            position=self.position.clone(),
            scalar_features=self.scalar_features.clone(),
            inverse_write_norm=self.inverse_write_norm.clone(),
            admission=self.admission.clone(),
            key_bias=self.key_bias.clone(),
            valid=self.valid.clone(),
        )

    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.write,
                self.successor_write,
                self.position,
                self.scalar_features,
                self.inverse_write_norm,
                self.admission,
                self.key_bias,
                self.valid,
            )
        )


@dataclass
class PreparedAdmittedBank:
    """Paged admitted bank plus the state needed to extend it exactly."""

    pages: tuple[PreparedBankPage, ...]
    entries: Tensor
    prefix_max: Tensor
    pending_page: Tensor
    pending_slot: Tensor
    next_position: int
    page_size: int
    identity: Mapping[str, object] = field(default_factory=dict)
    # Host metadata obtained at preparation/append/load, never by synchronizing
    # a device count on a query. None keeps manually constructed banks safe.
    read_entry_bound: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.page_size <= 0 or self.next_position < 0:
            raise ValueError("prepared page size must be positive and position nonnegative")
        if self.read_entry_bound is not None and not 0 <= self.read_entry_bound <= len(self.pages) * self.page_size:
            raise ValueError("prepared read entry bound is outside page capacity")
        if self.entries.ndim != 1 or self.entries.numel() == 0 or self.entries.dtype != torch.int64:
            raise ValueError("prepared entries must be nonempty int64 [B]")
        for name in ("prefix_max", "pending_page", "pending_slot"):
            value = getattr(self, name)
            if value.shape != self.entries.shape or value.device != self.entries.device:
                raise ValueError(f"prepared {name} shape/device must match entries")
            if name != "prefix_max" and value.dtype != torch.int64:
                raise ValueError(f"prepared {name} must be int64")
        if self.pages:
            first = self.pages[0].write
            for page in self.pages:
                if (page.write.shape != first.shape or page.write.shape[:2] != (self.entries.numel(), self.page_size)
                        or page.write.device != self.entries.device or page.write.dtype != first.dtype):
                    raise ValueError("prepared pages must share batch, capacity, width, dtype and device")

    def locate(self, lane: int, entry: int) -> tuple[int, int]:
        count = int(self.entries[lane].item())
        if entry < 0 or entry >= count:
            raise IndexError(f"entry {entry} is outside lane {lane}'s {count} committed events")
        return divmod(int(entry), int(self.page_size))

    def linear_offset(
        self, *, page: int, slot: int, hidden_index: int, hidden_dim: int
    ) -> Tensor:
        """Return an explicitly int64 element offset for kernel-address tests."""
        values = torch.tensor(
            [page, self.page_size, slot, hidden_dim, hidden_index], dtype=torch.int64
        )
        return (values[0] * values[1] + values[2]) * values[3] + values[4]

    def to(self, *, device=None, dtype=None) -> "PreparedAdmittedBank":
        return PreparedAdmittedBank(
            pages=tuple(page.to(device=device, dtype=dtype) for page in self.pages),
            entries=self.entries.to(device=device),
            prefix_max=self.prefix_max.to(device=device, dtype=dtype),
            pending_page=self.pending_page.to(device=device),
            pending_slot=self.pending_slot.to(device=device),
            next_position=self.next_position,
            page_size=self.page_size,
            identity=dict(self.identity),
            read_entry_bound=self.read_entry_bound,
        )

    def nbytes(self) -> int:
        state = (self.entries, self.prefix_max, self.pending_page, self.pending_slot)
        return sum(page.nbytes() for page in self.pages) + sum(
            tensor.numel() * tensor.element_size() for tensor in state
        )


@dataclass
class PreparedLayerMemory:
    bank: PreparedAdmittedBank
    core: tuple[Tensor, Tensor, Tensor]
    long: tuple[Tensor, Tensor, Tensor]

    def to(self, *, device=None, dtype=None) -> "PreparedLayerMemory":
        def value(tensor: Tensor) -> Tensor:
            return tensor.to(device=device, dtype=dtype)

        return PreparedLayerMemory(
            bank=self.bank.to(device=device, dtype=dtype),
            core=tuple(value(tensor) for tensor in self.core),
            long=tuple(value(tensor) for tensor in self.long),
        )

    def nbytes(self) -> int:
        return self.bank.nbytes() + sum(
            tensor.numel() * tensor.element_size() for tensor in (*self.core, *self.long)
        )


@dataclass
class PreparedDABSNMemory:
    layers: tuple[PreparedLayerMemory, ...] = ()
    position: int = 0
    batch_size: int = 0
    fingerprint: Mapping[str, object] = field(default_factory=dict)
    source_format: str = "dabsn_memory_v1"
    format: str = FORMAT

    @property
    def entries(self) -> int:
        if not self.layers:
            return 0
        return int(self.layers[0].bank.entries.max().item())

    def to(self, device=None, dtype=None) -> "PreparedDABSNMemory":
        return PreparedDABSNMemory(
            layers=tuple(layer.to(device=device, dtype=dtype) for layer in self.layers),
            position=self.position,
            batch_size=self.batch_size,
            fingerprint=dict(self.fingerprint),
            source_format=self.source_format,
        )

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers)


def read_parameter_identity(read: Any) -> str:
    """Hash only scalar parameters baked into prepared admission metadata."""
    digest = hashlib.sha256()
    baked_names = {
        "admit_window",
        "short_read.p_gain",
        "short_read.novelty_gain",
        "short_read.null_bias",
        "short_read.adm_p_gain",
        "short_read.adm_novelty_gain",
        "short_read.adm_saturation_gain",
        "short_read.adm_energy_gain",
        "short_read.adm_bias",
    }
    selected = [(name, parameter) for name, parameter in read.named_parameters() if name in baked_names]
    if {name for name, _ in selected} != baked_names:
        missing = sorted(baked_names - {name for name, _ in selected})
        raise ValueError(f"read is missing prepared-memory parameters: {missing}")
    # Normal in-place parameter updates increment _version; cache only metadata,
    # never bank state. Inference tensors without version counters are rehashed.
    try:
        versions = tuple((name, id(p), p._version, p.dtype, p.device) for name, p in selected)
    except RuntimeError:
        versions = None
    cached = _IDENTITIES.get(read)
    if versions is not None and cached is not None and cached[0] == versions:
        return cached[1]
    for name, parameter in sorted(selected):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    result = digest.hexdigest()
    if versions is not None:
        _IDENTITIES[read] = (versions, result)
    return result


def _page(
    batch: int,
    page_size: int,
    hidden: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedBankPage:
    return PreparedBankPage(
        write=torch.zeros(batch, page_size, hidden, device=device, dtype=dtype),
        successor_write=torch.zeros(batch, page_size, hidden, device=device, dtype=dtype),
        position=torch.zeros(batch, page_size, device=device, dtype=torch.int64),
        scalar_features=torch.zeros(batch, page_size, 4, device=device, dtype=dtype),
        inverse_write_norm=torch.zeros(
            batch, page_size, device=device,
            dtype=torch.float64 if dtype == torch.float64 else torch.float32,
        ),
        admission=torch.zeros(batch, page_size, device=device, dtype=dtype),
        key_bias=torch.zeros(batch, page_size, device=device, dtype=dtype),
        valid=torch.zeros(batch, page_size, device=device, dtype=torch.bool),
    )


def _bank_values(
    read: Any,
    write: Tensor,
    novelty_mean: Tensor,
    plasticity_mean: Tensor,
    energy_mean: Tensor,
    saturation_mean: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    hidden = int(write.shape[-1])
    parameters = read.short_read
    admission = (
        parameters.adm_p_gain * torch.log1p(plasticity_mean * hidden)
        + parameters.adm_novelty_gain * novelty_mean
        + parameters.adm_saturation_gain * saturation_mean
        + parameters.adm_energy_gain * energy_mean
        + parameters.adm_bias
    )
    key_bias = (
        parameters.p_gain * torch.log1p(plasticity_mean * hidden)
        + parameters.novelty_gain * novelty_mean
        + parameters.null_bias
    )
    norm_input = write if write.dtype == torch.float64 else write.float()
    inverse_norm = norm_input.norm(dim=-1).clamp_min(1.0e-12).reciprocal()
    # Scalar zero normalization must also remain finite in fp16.
    scalar_features = F.normalize(
        torch.stack([energy_mean, saturation_mean, novelty_mean, plasticity_mean], dim=-1)
        .to(norm_input.dtype), dim=-1,
    ).to(write.dtype)
    return admission, key_bias, scalar_features, inverse_norm


@torch.no_grad()
def prepare_admitted_bank(
    read: Any,
    raw_bank: Mapping[str, Tensor],
    *,
    page_size: int,
    start_position: int = 0,
) -> PreparedAdmittedBank:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    write = raw_bank["write"]
    if write.ndim != 3 or write.shape[1] <= 0:
        raise ValueError("raw write bank must be non-empty [B,N,H]")
    batch, length, hidden = write.shape
    novelty = raw_bank["novelty_mean"]
    plasticity = raw_bank["plasticity_mean"]
    energy = raw_bank["energy_mean"]
    saturation = raw_bank["saturation_mean"]
    expected_shape = (batch, length)
    if any(tuple(value.shape) != expected_shape for value in (novelty, plasticity, energy, saturation)):
        raise ValueError("raw scalar bank fields must have shape [B,N]")

    admission, key_bias, features, inverse_norm = _bank_values(
        read, write, novelty, plasticity, energy, saturation
    )
    prefix = torch.cummax(admission, dim=1).values
    member = admission >= prefix - F.softplus(read.admit_window)
    entries = member.sum(dim=1, dtype=torch.int64)
    committed_max = int(entries.max().item())
    page_count = max((committed_max + page_size - 1) // page_size, 1)
    pages = [
        _page(batch, page_size, hidden, device=write.device, dtype=write.dtype)
        for _ in range(page_count)
    ]
    pending_page = torch.full((batch,), -1, device=write.device, dtype=torch.int64)
    pending_slot = torch.full((batch,), -1, device=write.device, dtype=torch.int64)
    positions = torch.arange(length, device=write.device, dtype=torch.int64) + int(start_position)

    order = torch.argsort(member.to(torch.int8), dim=1, descending=True, stable=True)
    packed_width = page_count * page_size
    if packed_width > length:
        order = F.pad(order, (0, packed_width - length))
    packed_rank = torch.arange(packed_width, device=write.device).unsqueeze(0)
    packed_valid = packed_rank < entries.unsqueeze(1)
    for page_index, page in enumerate(pages):
        start = page_index * page_size
        stop = start + page_size
        source = order[:, start:stop]
        valid = packed_valid[:, start:stop]
        gather_hidden = source.unsqueeze(-1).expand(-1, -1, hidden)
        successor = (source + 1).clamp(max=length - 1)
        page.write.copy_(torch.gather(write, 1, gather_hidden).masked_fill(~valid.unsqueeze(-1), 0))
        page.successor_write.copy_(
            torch.gather(
                write, 1, successor.unsqueeze(-1).expand(-1, -1, hidden)
            ).masked_fill(~valid.unsqueeze(-1), 0)
        )
        page.position.copy_(
            torch.gather(positions.unsqueeze(0).expand(batch, -1), 1, source).masked_fill(
                ~valid, 0
            )
        )
        page.scalar_features.copy_(
            torch.gather(features, 1, source.unsqueeze(-1).expand(-1, -1, 4)).masked_fill(
                ~valid.unsqueeze(-1), 0
            )
        )
        page.inverse_write_norm.copy_(torch.gather(inverse_norm, 1, source).masked_fill(~valid, 0))
        page.admission.copy_(torch.gather(admission, 1, source).masked_fill(~valid, 0))
        page.key_bias.copy_(torch.gather(key_bias, 1, source).masked_fill(~valid, 0))
        page.valid.copy_(valid)

    last_is_pending = member[:, -1] & (entries > 0)
    last_packed = (entries - 1).clamp_min(0)
    pending_page.copy_(
        torch.where(last_is_pending, torch.div(last_packed, page_size, rounding_mode="floor"), -1)
    )
    pending_slot.copy_(torch.where(last_is_pending, last_packed.remainder(page_size), -1))

    return PreparedAdmittedBank(
        pages=tuple(pages),
        entries=entries,
        prefix_max=prefix[:, -1].clone(),
        pending_page=pending_page,
        pending_slot=pending_slot,
        next_position=int(start_position) + int(length),
        page_size=int(page_size),
        identity={"read_parameters": read_parameter_identity(read)},
        read_entry_bound=committed_max,
    )


@torch.no_grad()
def append_prepared_bank(
    bank: PreparedAdmittedBank,
    read: Any,
    write: Tensor,
    novelty_mean: Tensor,
    plasticity_mean: Tensor,
    energy_mean: Tensor,
    saturation_mean: Tensor,
) -> PreparedAdmittedBank:
    """Functionally extend a bank, copying only pages that receive writes."""
    if write.ndim != 3:
        raise ValueError("write must have shape [B,T,H]")
    batch, steps, hidden = write.shape
    if batch != int(bank.entries.shape[0]):
        raise ValueError("append batch does not match prepared bank")
    expected = (batch, steps)
    if any(
        tuple(value.shape) != expected
        for value in (novelty_mean, plasticity_mean, energy_mean, saturation_mean)
    ):
        raise ValueError("append scalar fields must have shape [B,T]")
    if bank.identity.get("read_parameters") != read_parameter_identity(read):
        raise ValueError("prepared bank read parameters do not match the current read")

    if steps == 0:
        return bank

    admission, key_bias, features, inverse_norm = _bank_values(
        read, write, novelty_mean, plasticity_mean, energy_mean, saturation_mean
    )
    prefix = torch.maximum(torch.cummax(admission, dim=1).values, bank.prefix_max[:, None])
    member = admission >= prefix - F.softplus(read.admit_window)
    added = member.sum(dim=1, dtype=torch.int64)
    entries = bank.entries + added
    order = torch.argsort(member.to(torch.int8), dim=1, descending=True, stable=True)
    # One small metadata transfer per append, not per event/lane. All membership,
    # rank, gather and page updates stay on device; Python owns page allocation.
    old_counts, new_counts, previous_pages = torch.stack(
        (bank.entries, entries, bank.pending_page)
    ).cpu().tolist()
    touched = {index for index in previous_pages if index >= 0}
    for old, new in zip(old_counts, new_counts):
        if new > old:
            touched.update(range(old // bank.page_size, (new - 1) // bank.page_size + 1))
    pages = list(bank.pages)
    needed = max((max(new_counts) + bank.page_size - 1) // bank.page_size, 1)
    while len(pages) < needed:
        pages.append(_page(batch, bank.page_size, hidden, device=write.device, dtype=write.dtype))
    slots = torch.arange(bank.page_size, device=write.device, dtype=torch.int64)[None, :]
    for index in sorted(touched):
        if index < len(bank.pages):
            pages[index] = pages[index].clone()
        page = pages[index]
        ranks = slots + index * bank.page_size - bank.entries[:, None]
        append_mask = (ranks >= 0) & (ranks < added[:, None])
        source = torch.gather(order, 1, ranks.clamp(min=0, max=steps - 1))
        previous_mask = (bank.pending_page[:, None] == index) & (slots == bank.pending_slot[:, None])

        def fill(destination, values, selected=source):
            gather_index = selected
            mask = append_mask
            if values.ndim == 3:
                gather_index = selected.unsqueeze(-1).expand(-1, -1, values.shape[-1])
                mask = mask.unsqueeze(-1)
            destination.copy_(torch.where(mask, torch.gather(values, 1, gather_index), destination))

        fill(page.write, write)
        page.successor_write.copy_(torch.where(previous_mask.unsqueeze(-1), write[:, :1], page.successor_write))
        fill(page.successor_write, write, (source + 1).clamp(max=steps - 1))
        fill(page.scalar_features, features)
        fill(page.inverse_write_norm, inverse_norm)
        fill(page.admission, admission)
        fill(page.key_bias, key_bias)
        page.position.copy_(torch.where(append_mask, source + bank.next_position, page.position))
        page.valid.logical_or_(append_mask)
    pending_page = torch.where(member[:, -1], torch.div(entries - 1, bank.page_size, rounding_mode="floor"), -1)
    pending_slot = torch.where(member[:, -1], (entries - 1).remainder(bank.page_size), -1)

    return PreparedAdmittedBank(
        pages=tuple(pages),
        entries=entries,
        prefix_max=prefix[:, -1].clone(),
        pending_page=pending_page,
        pending_slot=pending_slot,
        next_position=bank.next_position + steps,
        page_size=bank.page_size,
        identity=dict(bank.identity),
        read_entry_bound=max(new_counts),
    )


def _prepared_page_merge(
    query: Tensor,
    cocktail: Tensor,
    query_positions: Tensor,
    write: Tensor,
    successor_write: Tensor,
    position: Tensor,
    scalar_features: Tensor,
    inverse_write_norm: Tensor,
    admission: Tensor,
    key_bias: Tensor,
    valid: Tensor,
    scale: Tensor,
    cocktail_gain: Tensor,
    short_max: Tensor,
    short_denominator: Tensor,
    short_numerator: Tensor,
    permanent_max: Tensor,
    permanent_denominator: Tensor,
    permanent_numerator: Tensor,
    induction_max: Tensor,
    induction_denominator: Tensor,
    induction_numerator: Tensor,
) -> tuple[Tensor, ...]:
    """Score and merge one fixed page; suitable for full-graph CUDA compile."""
    scores = torch.bmm(query, write.transpose(1, 2))
    scores = scores * inverse_write_norm.unsqueeze(1) * scale
    short_scores = scores + (key_bias + admission).unsqueeze(1)
    permanent_scores = short_scores + cocktail_gain * torch.bmm(
        cocktail, scalar_features.transpose(1, 2)
    )
    allow = valid.unsqueeze(1) & (position.unsqueeze(1) <= query_positions.unsqueeze(2))
    induct_allow = valid.unsqueeze(1) & (
        position.unsqueeze(1) < query_positions.unsqueeze(2)
    )

    def reduce(
        page_scores: Tensor,
        page_allow: Tensor,
        values: Tensor,
        old_max: Tensor,
        old_denominator: Tensor,
        old_numerator: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        masked = page_scores.masked_fill(~page_allow, -torch.inf)
        page_max = masked.max(dim=-1).values
        has = page_allow.any(dim=-1)
        safe_max = torch.where(has, page_max, torch.zeros_like(page_max))
        weights = torch.where(
            page_allow,
            torch.exp(masked - safe_max.unsqueeze(-1)),
            torch.zeros_like(masked),
        )
        page_denominator = weights.sum(dim=-1)
        page_numerator = torch.bmm(weights, values)
        new_max = torch.maximum(old_max, page_max)
        old_factor = torch.where(
            torch.isfinite(old_max),
            torch.exp(old_max - new_max),
            torch.zeros_like(new_max),
        )
        page_factor = torch.where(
            has, torch.exp(page_max - new_max), torch.zeros_like(new_max)
        )
        numerator = (
            old_numerator * old_factor.unsqueeze(-1)
            + page_numerator * page_factor.unsqueeze(-1)
        )
        denominator = old_denominator * old_factor + page_denominator * page_factor
        return new_max, denominator, numerator

    short = reduce(
        short_scores, allow, write, short_max, short_denominator, short_numerator
    )
    permanent = reduce(
        permanent_scores,
        allow,
        write,
        permanent_max,
        permanent_denominator,
        permanent_numerator,
    )
    induction = reduce(
        short_scores,
        induct_allow,
        successor_write,
        induction_max,
        induction_denominator,
        induction_numerator,
    )
    return (*short, *permanent, *induction)


def _prepared_single_page_read(
    query: Tensor, cocktail: Tensor, query_starts: Tensor,
    write: Tensor, successor_write: Tensor, position: Tensor,
    scalar_features: Tensor, inverse_write_norm: Tensor, admission: Tensor,
    key_bias: Tensor, valid: Tensor, scale: Tensor, cocktail_gain: Tensor,
    short_gain: Tensor, pad_gain: Tensor, induct_gain: Tensor,
) -> Tensor:
    """One-page equation with promotion and final reduction inside compilation."""
    dtype = torch.float32 if query.dtype in {torch.float16, torch.bfloat16} else query.dtype
    q, values = query.to(dtype), write.to(dtype)
    successors = successor_write.to(dtype)
    positions = query_starts[:, None] + torch.arange(query.shape[1], device=query.device)[None, :]
    scores = torch.bmm(q, values.transpose(1, 2))
    scores = scores * inverse_write_norm.to(dtype).unsqueeze(1) * scale.to(dtype)
    short_scores = scores + (key_bias.to(dtype) + admission.to(dtype)).unsqueeze(1)
    permanent_scores = short_scores + cocktail_gain.to(dtype) * torch.bmm(
        cocktail.to(dtype), scalar_features.to(dtype).transpose(1, 2))
    allow = valid.unsqueeze(1) & (position.unsqueeze(1) <= positions.unsqueeze(2))
    induct_allow = valid.unsqueeze(1) & (position.unsqueeze(1) < positions.unsqueeze(2))

    def reduce(scores, mask, values):
        masked = scores.masked_fill(~mask, -torch.inf)
        maximum = masked.max(dim=-1).values
        safe_maximum = torch.where(mask.any(dim=-1), maximum, torch.zeros_like(maximum))
        weights = torch.where(mask, torch.exp(masked - safe_maximum.unsqueeze(-1)), 0.)
        return torch.bmm(weights, values) / weights.sum(dim=-1).clamp_min(1e-12).unsqueeze(-1)

    return (short_gain.to(dtype) * reduce(short_scores, allow, values)
            + pad_gain.to(dtype) * reduce(permanent_scores, allow, values)
            + induct_gain.to(dtype) * reduce(short_scores, induct_allow, successors)).to(query.dtype)


@torch.no_grad()
def prepared_three_way_read(
    query: Tensor,
    query_cocktail: Tensor,
    bank: PreparedAdmittedBank,
    query_starts: Tensor,
    *,
    scale: Tensor,
    short_gain: Tensor,
    pad_gain: Tensor,
    induct_gain: Tensor,
    cocktail_gain: Tensor,
    query_tile_size: int = 256,
) -> Tensor:
    """Exact bounded-workspace pagewise short/permanent/induction read."""
    output_dtype = query.dtype
    if query_tile_size <= 0:
        raise ValueError("query_tile_size must be positive")
    if query.shape[1] > query_tile_size:
        return torch.cat([
            prepared_three_way_read(
                query[:, start:start + query_tile_size],
                query_cocktail[:, start:start + query_tile_size],
                bank, query_starts + start, scale=scale, short_gain=short_gain,
                pad_gain=pad_gain, induct_gain=induct_gain, cocktail_gain=cocktail_gain,
                query_tile_size=query_tile_size,
            ) for start in range(0, query.shape[1], query_tile_size)
        ], dim=1)
    if len(bank.pages) == 1:
        from .kernels.prepared_admitted import prepared_admitted_single_page

        page = bank.pages[0]
        capacity = bank.page_size if bank.read_entry_bound is None else max(1, bank.read_entry_bound)
        return prepared_admitted_single_page(
            query, query_cocktail, query_starts, page.write[:, :capacity],
            page.successor_write[:, :capacity], page.position[:, :capacity],
            page.scalar_features[:, :capacity], page.inverse_write_norm[:, :capacity],
            page.admission[:, :capacity], page.key_bias[:, :capacity], page.valid[:, :capacity],
            scale, cocktail_gain, short_gain, pad_gain, induct_gain,
        )
    compute_dtype = torch.float32 if query.dtype in {torch.float16, torch.bfloat16} else query.dtype
    q = query.to(compute_dtype)
    cocktail = query_cocktail.to(compute_dtype)
    batch, query_len, hidden = query.shape
    query_positions = query_starts[:, None] + torch.arange(
        query_len, device=query.device, dtype=torch.int64
    )[None, :]

    zero_denominator = torch.zeros((batch, query_len), device=query.device, dtype=compute_dtype)
    zero_numerator = torch.zeros(
        (batch, query_len, hidden), device=query.device, dtype=compute_dtype
    )
    accumulator: tuple[Tensor, ...] = tuple(
        value
        for _ in range(3)
        for value in (
            torch.full((batch, query_len), -torch.inf, device=query.device, dtype=compute_dtype),
            zero_denominator.clone(),
            zero_numerator.clone(),
        )
    )
    from .kernels.prepared_admitted import prepared_admitted_page
    merge_page = prepared_admitted_page

    for page_index, page in enumerate(bank.pages):
        # Admission compacts each lane into a dense prefix. A host-side upper
        # bound shared by all lanes removes allocation padding without omitting
        # any event, changing capacity, or extracting a query-time GPU scalar.
        capacity = (bank.page_size if bank.read_entry_bound is None else
                    min(bank.page_size, max(1, bank.read_entry_bound - page_index * bank.page_size)))
        writes = page.write[:, :capacity].to(compute_dtype)
        accumulator = merge_page(
            q,
            cocktail,
            query_positions,
            writes,
            page.successor_write[:, :capacity].to(compute_dtype),
            page.position[:, :capacity],
            page.scalar_features[:, :capacity].to(compute_dtype),
            page.inverse_write_norm[:, :capacity].to(compute_dtype),
            page.admission[:, :capacity].to(compute_dtype),
            page.key_bias[:, :capacity].to(compute_dtype),
            page.valid[:, :capacity],
            scale.to(compute_dtype),
            cocktail_gain.to(compute_dtype),
            *accumulator,
        )

    outputs = []
    for index in range(3):
        denominator = accumulator[index * 3 + 1]
        numerator = accumulator[index * 3 + 2]
        outputs.append(
            torch.where(
                (denominator > 0).unsqueeze(-1),
                numerator / denominator.clamp_min(1.0e-12).unsqueeze(-1),
                torch.zeros_like(numerator),
            )
        )
    result = (
        short_gain.to(compute_dtype) * outputs[0]
        + pad_gain.to(compute_dtype) * outputs[1]
        + induct_gain.to(compute_dtype) * outputs[2]
    )
    return result.to(output_dtype)


def _page_payload(page: PreparedBankPage) -> dict[str, Tensor]:
    return {
        name: getattr(page, name).detach().cpu()
        for name in (
            "write",
            "successor_write",
            "position",
            "scalar_features",
            "inverse_write_norm",
            "admission",
            "key_bias",
            "valid",
        )
    }


def to_payload(memory: PreparedDABSNMemory) -> dict[str, object]:
    return {
        "format": FORMAT,
        "source_format": memory.source_format,
        "position": memory.position,
        "batch_size": memory.batch_size,
        "fingerprint": dict(memory.fingerprint),
        "layers": [
            {
                "bank": {
                    "pages": [_page_payload(page) for page in layer.bank.pages],
                    "entries": layer.bank.entries.detach().cpu(),
                    "prefix_max": layer.bank.prefix_max.detach().cpu(),
                    "pending_page": layer.bank.pending_page.detach().cpu(),
                    "pending_slot": layer.bank.pending_slot.detach().cpu(),
                    "next_position": layer.bank.next_position,
                    "page_size": layer.bank.page_size,
                    "identity": dict(layer.bank.identity),
                },
                "core": [tensor.detach().cpu() for tensor in layer.core],
                "long": [tensor.detach().cpu() for tensor in layer.long],
            }
            for layer in memory.layers
        ],
    }


def from_payload(payload: Mapping[str, object]) -> PreparedDABSNMemory:
    if payload.get("format") != FORMAT:
        raise ValueError(
            f"not a prepared DABSN memory payload: format={payload.get('format')!r}, expected {FORMAT!r}"
        )
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, Sequence):
        raise ValueError("prepared memory layers must be a sequence")
    layers = []
    for raw_layer in raw_layers:
        entry = cast(Mapping[str, Any], raw_layer)
        raw_bank = cast(Mapping[str, Any], entry["bank"])
        pages = tuple(
            PreparedBankPage(**cast(dict[str, Tensor], raw_page))
            for raw_page in cast(Sequence[Mapping[str, Tensor]], raw_bank["pages"])
        )
        bank = PreparedAdmittedBank(
            pages=pages,
            entries=cast(Tensor, raw_bank["entries"]),
            prefix_max=cast(Tensor, raw_bank["prefix_max"]),
            pending_page=cast(Tensor, raw_bank["pending_page"]),
            pending_slot=cast(Tensor, raw_bank["pending_slot"]),
            next_position=int(raw_bank["next_position"]),
            page_size=int(raw_bank["page_size"]),
            identity=dict(cast(Mapping[str, object], raw_bank.get("identity", {}))),
            read_entry_bound=int(cast(Tensor, raw_bank["entries"]).max().item()),
        )
        layers.append(
            PreparedLayerMemory(
                bank=bank,
                core=tuple(cast(Sequence[Tensor], entry["core"])),
                long=tuple(cast(Sequence[Tensor], entry["long"])),
            )
        )
    return PreparedDABSNMemory(
        layers=tuple(layers),
        position=int(cast(Any, payload["position"])),
        batch_size=int(cast(Any, payload["batch_size"])),
        fingerprint=dict(cast(Mapping[str, object], payload.get("fingerprint", {}))),
        source_format=str(payload.get("source_format", "dabsn_memory_v1")),
    )


def save_prepared(
    memory: PreparedDABSNMemory, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Atomically save v2 memory; existing files require explicit overwrite."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".dabsn-memory-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(to_payload(memory), stream)
            stream.flush()
            os.fsync(stream.fileno())
        load_prepared_payload(temporary)
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Atomic no-clobber publication, including races with another saver.
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_prepared_payload(path: str | Path, *, map_location="cpu") -> PreparedDABSNMemory:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("prepared memory file must contain a mapping")
    return from_payload(payload)


__all__ = [
    "FORMAT",
    "PreparedAdmittedBank",
    "PreparedBankPage",
    "PreparedDABSNMemory",
    "PreparedLayerMemory",
    "append_prepared_bank",
    "load_prepared_payload",
    "prepared_memory_bytes",
    "prepare_admitted_bank",
    "prepared_three_way_read",
    "read_parameter_identity",
    "save_prepared",
]
