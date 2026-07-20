"""Fixed-shape CUDA-graph execution for DABSN training steps.

CUDA Graphs remove Python and kernel-launch overhead by recording the forward
and backward device work once and replaying it.  A graph is specialised to the
*shape* of the tensors the caller supplies, never to a hard-coded context
length or batch size: a training run picks one shape, records it, and replays
it; a different run (or a caller-managed shape cache) records any other shape
that fits device memory.

This is deliberately task-neutral.  It graphs an :class:`~torch.nn.Module`
using its live activation dtype and shape and returns a callable with an
identical signature.  It does not know about tokens, targets, a loss function,
or a maximum sequence length, and it never wraps the module in
``torch.compile``/Dynamo -- ``make_graphed_callables`` records the eager
forward/backward directly, so DABSN's custom-autograd boundaries are preserved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

Reduce = Callable[[object], Tensor]


def _default_reduce(output: object) -> Tensor:
    """Collapse arbitrary module output to a scalar for gradient checking."""

    if torch.is_tensor(output):
        return output.float().square().mean()
    if isinstance(output, (list, tuple)):
        tensors = [value for value in output if torch.is_tensor(value)]
    elif isinstance(output, dict):
        tensors = [value for value in output.values() if torch.is_tensor(value)]
    else:  # pragma: no cover - defensive
        raise TypeError(f"cannot reduce module output of type {type(output)!r}")
    if not tensors:
        raise ValueError("module output contained no tensors to reduce")
    return sum(tensor.float().square().mean() for tensor in tensors)


def _read_submodules(module: nn.Module) -> list[nn.Module]:
    """DABSN admitted-read submodules (duck-typed to keep this file neutral)."""

    return [m for m in module.modules() if hasattr(m, "read_geometry")]


def _pin_capture_safe_reads(module: nn.Module, sample_args: Sequence[Tensor]):
    """Make the admitted read shape-stable for graph capture, and sub-quadratic.

    The read sizes its bank to the data-dependent admitted count by default --
    not capturable, and it *differs between warmup and capture*, which corrupts
    the recorded graph (misaligned addresses). CUDA-graph capture requires one
    static width. Forcing the full ``seq_len`` width is capturable but turns the
    sparse read into an O(T^2) attention, which at long context dominates the
    core cost. Instead: run a short eager pre-warmup to observe the natural
    admitted width, then pin a padded static cap (``_capture_bank_width``) so the
    captured read is fixed-shape AND O(T * admitted). Returns a restore closure.
    """

    reads = _read_submodules(module)
    saved = [
        (m, getattr(m, "_capture_safe_bank", None), getattr(m, "_capture_bank_width", None))
        for m in reads
    ]

    observed = 0
    if reads:
        try:
            with torch.no_grad():
                for _ in range(2):
                    module(*sample_args)
            for m in reads:
                width = getattr(m, "last_n_max", None)
                if isinstance(width, int):
                    observed = max(observed, width)
        except Exception:  # pragma: no cover - measurement is best-effort
            observed = 0

    # Generous pad so a captured batch that admits a bit more than the warmup
    # batches is still covered exactly; the read clamps the cap to seq_len, so
    # observed==0 (measurement failed) safely falls back to the full width.
    cap = observed * 2 + 64 if observed > 0 else None
    for m in reads:
        m._capture_safe_bank = True
        if cap is not None:
            m._capture_bank_width = cap

    def restore() -> None:
        for m, prev_flag, prev_width in saved:
            if prev_flag is None:
                if hasattr(m, "_capture_safe_bank"):
                    delattr(m, "_capture_safe_bank")
            else:
                m._capture_safe_bank = prev_flag
            if prev_width is None:
                if hasattr(m, "_capture_bank_width"):
                    delattr(m, "_capture_bank_width")
            else:
                m._capture_bank_width = prev_width

    return restore


def _find_backbone(module: nn.Module):
    """Locate a DABSN backbone if the graphed module exposes one."""

    backbone = getattr(module, "backbone", None)
    if backbone is None:
        body = getattr(module, "body", None)
        backbone = getattr(body, "backbone", None)
    if backbone is not None and hasattr(backbone, "blocks"):
        return backbone
    return None


def _block_grad_signature(module: nn.Module) -> list[dict[str, float]] | None:
    """Per-block core/read gradient norms, or ``None`` for a non-DABSN module."""

    backbone = _find_backbone(module)
    if backbone is None:
        return None
    rows: list[dict[str, float]] = []
    for index, block in enumerate(backbone.blocks):
        core_grad = block.core.W.weight.grad
        gain_grad = block.read_gain.grad
        rows.append(
            {
                "block": float(index),
                "core": 0.0 if core_grad is None else float(core_grad.detach().float().norm()),
                "read": 0.0 if gain_grad is None else float(gain_grad.detach().float().norm()),
            }
        )
    return rows


def _assert_close(name: str, eager: float, graphed: float, *, rtol: float, atol: float) -> None:
    if not torch.isfinite(torch.tensor(graphed)).item():
        raise RuntimeError(f"CUDA-graph {name} is not finite: {graphed!r}")
    if abs(eager - graphed) > atol + rtol * abs(eager):
        raise RuntimeError(
            f"CUDA-graph {name} parity failed: eager={eager:.9g}, graphed={graphed:.9g}"
        )


def _assert_signatures_close(
    eager: Sequence[dict[str, float]],
    graphed: Sequence[dict[str, float]],
    *,
    rtol: float,
    atol: float,
) -> None:
    if len(eager) != len(graphed):
        raise RuntimeError("CUDA-graph capture changed the DABSN depth")
    failures: list[dict[str, float | str]] = []
    for expected, observed in zip(eager, graphed):
        if int(expected["block"]) != int(observed["block"]):
            raise RuntimeError("CUDA-graph capture changed block ordering")
        for key in ("core", "read"):
            a, b = expected[key], observed[key]
            if not torch.isfinite(torch.tensor(b)).item() or abs(a - b) > atol + rtol * abs(a):
                failures.append(
                    {"block": expected["block"], "gradient": key, "eager": a, "graphed": b}
                )
    if failures:
        raise RuntimeError(f"CUDA-graph gradient parity failed: {failures}")


def make_graphed_train_callable(
    module: nn.Module,
    sample_args: Sequence[Tensor],
    *,
    warmup_iters: int = 3,
    verify: bool = True,
    loss_reduce: Reduce | None = None,
    loss_rtol: float = 2.0e-2,
    loss_atol: float = 1.0e-3,
    grad_rtol: float = 3.0e-2,
    grad_atol: float = 1.0e-4,
) -> nn.Module:
    """Return a CUDA-graphed training callable for ``module`` at a live shape.

    Parameters
    ----------
    module:
        The parameter-owning module to record.  Its natural output is returned
        unchanged by the graphed callable, so the caller keeps ownership of the
        loss and optimiser step. Treat each replay as an independent single
        backward and accumulate with :class:`ManualGradientAccumulator`;
        correctness then never depends on autograd accumulating across graph
        replay streams.
    sample_args:
        Example inputs whose shapes/dtypes/device fix the recorded graph.  Any
        positive shape that fits device memory is allowed; no context length is
        assumed or capped.
    warmup_iters:
        Eager warmup iterations before capture (``make_graphed_callables``
        requires at least one to prime autograd and allocators).
    verify:
        When ``True`` (and on CUDA), compare eager vs. graphed loss and, for a
        DABSN backbone, every block's core/read gradient norms, raising on drift.
        The default tolerances guard against *gross* drift -- a dead block, an
        unwired gradient, a NaN -- not bit-parity: a captured graph and a fresh
        eager pass select different GEMM kernels, so fp16/bf16 training legitimately
        differs by ~1e-2. A real defect shows as a ~100% norm gap, far outside these.

    Notes
    -----
    On CPU or any non-CUDA input the module is returned unchanged -- CUDA Graphs
    are a CUDA-only optimisation and every reference/CPU code path stays eager.
    """

    sample_args = tuple(sample_args)
    if not sample_args or any(not torch.is_tensor(t) or not t.is_cuda for t in sample_args):
        return module
    if warmup_iters < 1:
        raise ValueError("warmup_iters must be at least one")

    reduce = loss_reduce or _default_reduce

    module.train()
    module.zero_grad(set_to_none=True)

    # Pin the admitted read to a fixed, sub-quadratic width for the whole build
    # (warmup + capture must agree, or the recorded graph misaligns). Restored
    # afterward so eager/non-graph steps keep their data-dependent sparse width.
    restore_reads = _pin_capture_safe_reads(module, sample_args)
    try:
        eager_loss = None
        eager_signature = None
        if verify:
            eager_loss = float(reduce(module(*sample_args)).detach())
            reduce(module(*sample_args)).backward()
            eager_signature = _block_grad_signature(module)
            module.zero_grad(set_to_none=True)

        # make_graphed_callables records the eager forward/backward as CUDA work.
        # It does NOT invoke Dynamo/AOTAutograd, so DABSN custom-autograd Functions
        # are captured intact rather than traced or rewritten.
        graphed = torch.cuda.make_graphed_callables(
            module,
            sample_args=sample_args,
            num_warmup_iters=int(warmup_iters),
            allow_unused_input=True,
        )

        # Flush capture's transient so the first independent single backward
        # begins from the same state as every later replay. Cross-replay
        # accumulation is intentionally delegated to ManualGradientAccumulator.
        reduce(graphed(*sample_args)).backward()
        torch.cuda.synchronize()
        module.zero_grad(set_to_none=True)

        if verify:
            graph_loss = float(reduce(graphed(*sample_args)).detach())
            reduce(graphed(*sample_args)).backward()
            graph_signature = _block_grad_signature(module)
            module.zero_grad(set_to_none=True)
            _assert_close("loss", eager_loss, graph_loss, rtol=loss_rtol, atol=loss_atol)
            if eager_signature is not None and graph_signature is not None:
                _assert_signatures_close(
                    eager_signature, graph_signature, rtol=grad_rtol, atol=grad_atol
                )
    finally:
        restore_reads()

    return graphed


__all__ = ["make_graphed_train_callable"]
