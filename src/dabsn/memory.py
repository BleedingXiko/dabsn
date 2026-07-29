"""Carried DABSN memory: ingest a corpus once, save it, reload it, query it.

A DABSN block's read scores every query position against a *bank* of past
writes. :func:`ingest` runs a model over context in time-chunks and keeps that
bank instead of discarding it, so the result is a standalone object that can be
written to disk and reloaded into a fresh process.

The entry point that makes this exact is :meth:`DABSNRead.forward_query_bank`,
which takes an externally supplied bank plus a ``query_start`` causal offset.
Chunked ingest against a growing bank reproduces one dense pass to float32
rounding -- see ``tests/test_memory_carry.py``.

Three states carry, and all three are required for exactness:

``bank``
    Per layer, the five write-tape fields the read actually scores against:
    ``write``, ``novelty``, ``plasticity``, ``energy``, ``saturation``. Grows
    with ingested tokens.
``core``
    The core's ``(budget, energy, saturation)`` triple -- the homeostatic state.
    It persists like everything else; energy and saturation are part of the
    retrieval address via the cocktail term, so dropping them would score
    carried writes against a state they were never written under.
``long``
    The long-scan ``(long, expected, retention)`` EMA.

Nothing here touches :meth:`DABSNBlock.forward`, which stays the hot path for
``checkpoint()``, the dispatch hooks, and CUDA-graph capture.

Usage::

    memory = ingest(model, context_ids)
    save(memory, "corpus.dmem")

    memory = load("corpus.dmem", model=model)
    logits, memory = forward_with_memory(model, query_ids, memory)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import Tensor

from .model import DABSNBlock, DABSNSequenceLM

__all__ = [
    "DABSNMemory",
    "LayerMemory",
    "ingest",
    "forward_with_memory",
    "save",
    "load",
    "embed_in_checkpoint",
    "extract_from_checkpoint",
]

FORMAT = "dabsn_memory_v1"

#: What the bank actually has to hold.
#:
#: ``forward_query_bank`` takes eight bank tapes and uses four of them *only*
#: through ``.mean(dim=-1)``:
#:
#: * ``admission()`` (``read.py:58``) reduces novelty/plasticity/energy/saturation
#: * ``key_bias`` (``read.py:609``) reduces plasticity and novelty
#: * ``bank_cocktail`` (``read.py:631``) reduces all four
#:
#: So they cost one scalar per position, not one vector. Only ``write`` is ever
#: gathered at full width (for ``bank_writes`` and the induction head's
#: ``next_writes``). Storing those four as full ``[B, T, H]`` tapes -- which the
#: first version of this module did -- costs 5x more than the read can use, and
#: turns a sub-quadratic bank into something heavier than a KV cache.
#:
#: ``bank_y``/``bank_b``/``bank_ay`` are deleted on entry to the read and are
#: never stored at all.
VECTOR_FIELDS = ("write",)
MEAN_FIELDS = ("novelty", "plasticity", "energy", "saturation")
BANK_FIELDS = VECTOR_FIELDS + tuple(f"{name}_mean" for name in MEAN_FIELDS)

_CORE_FIELDS = ("budget", "energy", "saturation")
_LONG_FIELDS = ("long", "expected", "retention")


@dataclass
class LayerMemory:
    """One block's carried state."""

    bank: dict[str, Tensor]
    core: tuple[Tensor, Tensor, Tensor]
    long: tuple[Tensor, Tensor, Tensor]

    @property
    def entries(self) -> int:
        return int(self.bank["write"].shape[1]) if self.bank else 0

    def to(self, *, device=None, dtype=None) -> "LayerMemory":
        def cast(t: Tensor) -> Tensor:
            return t.to(device=device, dtype=dtype)

        return LayerMemory(
            bank={k: cast(v) for k, v in self.bank.items()},
            core=tuple(cast(t) for t in self.core),  # type: ignore[arg-type]
            long=tuple(cast(t) for t in self.long),  # type: ignore[arg-type]
        )

    def nbytes(self) -> int:
        tensors = list(self.bank.values()) + list(self.core) + list(self.long)
        return sum(t.numel() * t.element_size() for t in tensors)


@dataclass
class DABSNMemory:
    """Carried memory for a whole model: one :class:`LayerMemory` per block."""

    layers: list[LayerMemory] = field(default_factory=list)
    position: int = 0
    batch_size: int = 0
    fingerprint: dict[str, object] = field(default_factory=dict)

    @property
    def entries(self) -> int:
        """Bank entries in layer 0, i.e. tokens ingested."""
        return self.layers[0].entries if self.layers else 0

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers)

    def to(self, device=None, dtype=None) -> "DABSNMemory":
        return DABSNMemory(
            layers=[layer.to(device=device, dtype=dtype) for layer in self.layers],
            position=self.position,
            batch_size=self.batch_size,
            fingerprint=dict(self.fingerprint),
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        mb = self.nbytes() / 1024 / 1024
        return (
            f"DABSNMemory(tokens={self.position}, batch={self.batch_size}, "
            f"layers={len(self.layers)}, {mb:.1f} MiB)"
        )


def fingerprint_of(model: DABSNSequenceLM) -> dict[str, object]:
    """Architecture identity a memory file must match to be reloadable."""
    return {
        "vocab": int(model.vocab),
        "hidden_dim": int(model.hidden_dim),
        "depth": int(model.depth),
        "layers": [spec.to_metadata() for spec in model.layers],
        "tie_embeddings": bool(model.tie_embeddings),
        "residual": bool(model.residual),
        "mlp_ratio": model.mlp_ratio,
    }


def _check_fingerprint(memory: DABSNMemory, model: DABSNSequenceLM) -> None:
    expected = fingerprint_of(model)
    if not memory.fingerprint:
        return
    observed = dict(memory.fingerprint)
    # Memory files created before residual/MLP support are exactly the disabled
    # architecture, so missing keys normalize to the backward-compatible defaults.
    observed.setdefault("residual", False)
    observed.setdefault("mlp_ratio", None)
    if observed != expected:
        mismatched = [
            f"{key}: memory={observed.get(key)!r} model={value!r}"
            for key, value in expected.items()
            if observed.get(key) != value
        ]
        raise ValueError(
            "memory was built by a different architecture: " + "; ".join(mismatched)
        )


def _require_seq_geometry(model: DABSNSequenceLM) -> None:
    bad = [
        index
        for index, block in enumerate(model.backbone.blocks)
        if block.read_geometry != "seq"
    ]
    if bad:
        raise NotImplementedError(
            "carried memory is exact only for seq read geometry; "
            f"blocks {bad} use field/hybrid. forward_query_bank refuses those."
        )


def _chunks(total: int, size: int | None) -> Iterator[tuple[int, int]]:
    if size is None or size <= 0 or size >= total:
        yield 0, total
        return
    for start in range(0, total, size):
        yield start, min(total, start + size)


def _ingest_block(
    block: DABSNBlock,
    inputs: Tensor,
    layer: LayerMemory | None,
    position: int,
) -> tuple[Tensor, LayerMemory]:
    """Run one block over ``inputs``, extending ``layer``'s bank.

    Mirrors :meth:`DABSNBlock.forward` -- core, read, ``state_to_hidden`` -- but
    from carried state and against the accumulated bank. ``block.forward`` is
    deliberately not reused: it owns no carry and must stay capture-clean.
    """
    batch = inputs.shape[0]
    core_state = (
        layer.core
        if layer is not None
        else block.core.initial_state(batch, device=inputs.device, dtype=inputs.dtype)
    )
    result, final_core = block.core.forward_from_state(
        inputs,
        initial_state=core_state,
        return_writes=True,
        return_cocktail=True,
        return_final_state=True,
    )
    trajectory, novelty, plasticity, expression, write, energy, saturation = result
    y, budget = trajectory.split(block.state_dim, dim=-1)

    chunk = {
        "write": write,
        "novelty_mean": novelty.mean(dim=-1),
        "plasticity_mean": plasticity.mean(dim=-1),
        "energy_mean": energy.mean(dim=-1),
        "saturation_mean": saturation.mean(dim=-1),
    }
    bank = (
        chunk
        if layer is None
        else {key: torch.cat([layer.bank[key], chunk[key]], dim=1) for key in BANK_FIELDS}
    )

    # Hand the read back full-width tapes without owning full-width memory: a
    # stride-0 expand of the stored mean is a view, and ``.mean(-1)`` over H
    # copies of m returns m. Costs ~2e-7, an order below the float32 rounding
    # the carry already carries.
    hidden_dim = int(write.shape[-1])

    def widen(name: str) -> Tensor:
        return bank[f"{name}_mean"].unsqueeze(-1).expand(-1, -1, hidden_dim)

    retrieved, final_long = block.read.forward_query_bank(
        y,
        budget,
        expression,
        write,
        novelty,
        plasticity,
        energy,
        saturation,
        # Deleted on entry to forward_query_bank; never scored, never stored.
        bank_y=None,
        bank_b=None,
        bank_ay=None,
        bank_write=bank["write"],
        bank_novelty=widen("novelty"),
        bank_p=widen("plasticity"),
        bank_energy=widen("energy"),
        bank_saturation=widen("saturation"),
        query_start=position,
        initial_long_state=None if layer is None else layer.long,
        return_long_state=True,
    )
    dabsn_output = block.state_to_hidden(y + block.read_gain * retrieved)
    hidden = block._finish_block(inputs, dabsn_output)
    return hidden, LayerMemory(bank=bank, core=final_core, long=final_long)


def _run(
    model: DABSNSequenceLM,
    ids: Tensor,
    memory: DABSNMemory | None,
    *,
    chunk_size: int | None,
    extend: bool,
) -> tuple[Tensor, DABSNMemory]:
    _require_seq_geometry(model)
    if ids.dim() != 2:
        raise ValueError(f"ids must be [B, T]; got {tuple(ids.shape)}")
    blocks = list(model.backbone.blocks)

    if memory is None:
        memory = DABSNMemory(
            layers=[],
            position=0,
            batch_size=int(ids.shape[0]),
            fingerprint=fingerprint_of(model),
        )
        state: list[LayerMemory | None] = [None] * len(blocks)
    else:
        _check_fingerprint(memory, model)
        if memory.batch_size != int(ids.shape[0]):
            raise ValueError(
                f"memory holds batch {memory.batch_size} but ids are batch "
                f"{int(ids.shape[0])}; a bank is per-lane and cannot be rebatched"
            )
        state = list(memory.layers)

    position = memory.position
    outputs: list[Tensor] = []
    with torch.no_grad():
        for start, stop in _chunks(int(ids.shape[1]), chunk_size):
            hidden = model.embed(ids[:, start:stop])
            for index, block in enumerate(blocks):
                hidden, state[index] = _ingest_block(
                    block, hidden, state[index], position
                )
            outputs.append(hidden)
            position += stop - start

    result = DABSNMemory(
        layers=[s for s in state if s is not None],
        position=position,
        batch_size=int(ids.shape[0]),
        fingerprint=fingerprint_of(model),
    )
    if not extend and memory is not None:
        # Caller wanted a read-only query: hand back the memory they came in
        # with, not one grown by the query tokens.
        result = memory
    return torch.cat(outputs, dim=1), result


def ingest(
    model: DABSNSequenceLM,
    ids: Tensor,
    memory: DABSNMemory | None = None,
    *,
    chunk_size: int | None = 512,
) -> DABSNMemory:
    """Read ``ids`` into memory, extending ``memory`` when one is supplied.

    ``chunk_size`` bounds the activation working set: ingest never materializes
    the whole sequence's graph, so a corpus larger than one pass still fits.
    Exactness does not depend on the width -- any chunking reproduces one dense
    pass.
    """
    _, result = _run(model, ids, memory, chunk_size=chunk_size, extend=True)
    return result


def forward_with_memory(
    model: DABSNSequenceLM,
    ids: Tensor,
    memory: DABSNMemory | None = None,
    *,
    chunk_size: int | None = None,
    extend: bool = False,
    return_hidden: bool = False,
) -> tuple[Tensor, DABSNMemory]:
    """Score ``ids`` against carried ``memory``.

    Returns ``(logits, memory)`` with logits shaped ``[B, T, vocab]``. With
    ``extend=False`` (the default) the returned memory is the input memory
    unchanged, so repeated independent queries all see the same corpus.
    """
    hidden, result = _run(model, ids, memory, chunk_size=chunk_size, extend=extend)
    if return_hidden:
        return hidden, result
    return model.project_sequence(hidden), result


def _to_payload(memory: DABSNMemory) -> dict[str, object]:
    return {
        "format": FORMAT,
        "position": int(memory.position),
        "batch_size": int(memory.batch_size),
        "fingerprint": memory.fingerprint,
        "layers": [
            {
                "bank": {k: v.detach().cpu() for k, v in layer.bank.items()},
                "core": [t.detach().cpu() for t in layer.core],
                "long": [t.detach().cpu() for t in layer.long],
            }
            for layer in memory.layers
        ],
    }


def _from_payload(payload: dict[str, object]) -> DABSNMemory:
    if payload.get("format") != FORMAT:
        raise ValueError(
            f"not a DABSN memory payload: format={payload.get('format')!r}, "
            f"expected {FORMAT!r}"
        )
    layers = []
    for entry in payload["layers"]:  # type: ignore[index]
        missing = [k for k in BANK_FIELDS if k not in entry["bank"]]
        if missing:
            raise ValueError(f"memory payload is missing bank fields: {missing}")
        layers.append(
            LayerMemory(
                bank={k: entry["bank"][k] for k in BANK_FIELDS},
                core=tuple(entry["core"]),  # type: ignore[arg-type]
                long=tuple(entry["long"]),  # type: ignore[arg-type]
            )
        )
    return DABSNMemory(
        layers=layers,
        position=int(payload["position"]),  # type: ignore[arg-type]
        batch_size=int(payload["batch_size"]),  # type: ignore[arg-type]
        fingerprint=dict(payload.get("fingerprint") or {}),  # type: ignore[arg-type]
    )


def save(memory: DABSNMemory, path: str | Path) -> Path:
    """Write ``memory`` to its own file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_to_payload(memory), destination)
    return destination


def load(
    path: str | Path,
    *,
    model: DABSNSequenceLM | None = None,
    map_location="cpu",
    device=None,
    dtype=None,
) -> DABSNMemory:
    """Read a memory file. Pass ``model`` to check it matches before use."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "dabsn_memory" in payload:
        payload = payload["dabsn_memory"]
    memory = _from_payload(payload)
    if model is not None:
        _check_fingerprint(memory, model)
        reference = next(model.parameters())
        memory = memory.to(
            device=device if device is not None else reference.device,
            dtype=dtype if dtype is not None else reference.dtype,
        )
    elif device is not None or dtype is not None:
        memory = memory.to(device=device, dtype=dtype)
    return memory


def embed_in_checkpoint(
    memory: DABSNMemory,
    checkpoint: str | Path,
    *,
    destination: str | Path | None = None,
) -> Path:
    """Attach ``memory`` to an existing checkpoint under ``dabsn_memory``.

    The checkpoint stays loadable by anything that ignores unknown keys, so a
    model and the corpus it has already read can ship as one file.
    """
    source = Path(checkpoint)
    target = Path(destination) if destination is not None else source
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"can only embed memory into a dict checkpoint; {source} holds "
            f"{type(payload).__name__}"
        )
    payload["dabsn_memory"] = _to_payload(memory)
    torch.save(payload, target)
    return target


def extract_from_checkpoint(
    checkpoint: str | Path,
    *,
    model: DABSNSequenceLM | None = None,
    map_location="cpu",
) -> DABSNMemory | None:
    """Pull an embedded memory back out, or ``None`` if the file has none."""
    payload = torch.load(Path(checkpoint), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "dabsn_memory" not in payload:
        return None
    memory = _from_payload(payload["dabsn_memory"])
    if model is not None:
        _check_fingerprint(memory, model)
        reference = next(model.parameters())
        memory = memory.to(device=reference.device, dtype=reference.dtype)
    return memory


def memory_cost(model: DABSNSequenceLM, tokens: int, *, batch_size: int = 1,
                bytes_per_element: int = 4) -> dict[str, object]:
    """Bank bytes for ``tokens`` ingested tokens, before running anything.

    One vector of ``state_dim`` per layer for the write tape, plus one scalar
    per layer for each of the four reduced fields.
    """
    per_token = sum(
        int(spec.resolved_state_dim) + len(MEAN_FIELDS) for spec in model.layers
    )
    total = per_token * tokens * batch_size * bytes_per_element
    return {
        "bytes_per_token": per_token * bytes_per_element * batch_size,
        "total_bytes": total,
        "total_mib": total / 1024 / 1024,
    }
