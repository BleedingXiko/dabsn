"""Sparse Capacity must stay practical at long context.

Regression guard for the event-sized gradient scratch: sparse BPTT and
captured online credit once allocated one parameter-gradient slot per selected
(sequence, token, active row) event, O(B*T*A*D), which grows with sequence
length and reached 128 GiB per buffer on a real L4 batch. Gradient scratch must
be the smaller of that and one slot per row per sequence, O(B*C*D), and both
layouts must compute the same learning signal.

These tests run on CPU. ``test_capacity_kernels_cuda.py`` holds the CUDA parity
and peak-memory evidence; CPU passing is not CUDA verification.
"""
import pytest
import torch

from dabsn.capacity import DABSNCapacitySubsystem
from dabsn.config import CapacityConfig
from dabsn.kernels import capacity as native


def dense(tensor):
    return tensor.to_dense() if tensor.is_sparse else tensor


def spy_backward(monkeypatch):
    """Record the gradient-table row count each native backward requests."""
    calls = []
    real = native._backward

    def wrapped(*args):
        x, indices, rows = args[0], args[5], args[7]
        calls.append({"rows": int(rows), "events": int(indices.numel()),
                      "batch": int(x.shape[0])})
        return real(*args)

    monkeypatch.setattr(native, "_backward", wrapped)
    return calls


def force_layout(monkeypatch, layout):
    monkeypatch.setattr(native, "capacity_gradient_layout", lambda *shape: layout)


# --------------------------------------------------------------------------
# The decision itself: shape-only, never larger than either table.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("batch", [1, 2, 16])
@pytest.mark.parametrize("steps", [1, 7, 512, 8192])
@pytest.mark.parametrize("active,rows", [(2, 8), (4, 64), (64, 4096), (16, 65536)])
def test_layout_choice_is_the_smaller_table(batch, steps, active, rows):
    slots = native.capacity_gradient_slots(batch, steps, active, rows)
    layout = native.capacity_gradient_layout(batch, steps, active, rows)
    assert slots[layout] == min(slots.values())
    if layout == "row":
        # Strictly smaller row tables imply rows < events, which keeps CUDA in
        # its private per-sequence row mode (no cross-lane write races).
        assert slots["row"] < slots["event"] and rows < batch * steps * active


@pytest.mark.parametrize("batch,active,rows", [(1, 8, 512), (8, 64, 8192), (32, 16, 65536)])
def test_gradient_scratch_does_not_grow_with_sequence_length(batch, active, rows):
    used = []
    for steps in (128, 1024, 8192, 65536, 1 << 20):
        slots = native.capacity_gradient_slots(batch, steps, active, rows)
        used.append(slots[native.capacity_gradient_layout(batch, steps, active, rows)])
    assert max(used) <= batch * rows
    # Once rows win, longer context never costs more gradient scratch.
    assert used[-1] == used[-2] == batch * rows


# --------------------------------------------------------------------------
# End to end: the kernel is actually asked for the bounded table.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("batch,steps,size,active", [(2, 512, 16, 2), (1, 300, 64, 4), (3, 128, 32, 8)])
def test_long_context_backward_requests_row_sized_scratch(monkeypatch, online, batch, steps, size, active):
    native.enable_capacity("cpu", required=True)
    calls = spy_backward(monkeypatch)
    model = DABSNCapacitySubsystem(5, 3, CapacityConfig(mode="sparse", size=size, active_units=active))
    x = torch.randn(batch, steps, 5, requires_grad=True)
    if online:
        rec, elig = model.initial_online_state(batch, device="cpu", recurrent_dtype=x.dtype,
                                               eligibility_dtype=x.dtype)
        with native.capture_capacity_rows():
            y, _, _ = model.forward_online(x, rec, elig)
            y.square().sum().backward()
        native.finalize_capacity_gradients(model)
    else:
        model.forward_sequence(x).square().sum().backward()
    assert len(calls) == 1
    call = calls[0]
    assert call["events"] == batch * steps * active
    assert call["rows"] == size, "sparse gradient scratch must be row-sized, not event-sized"
    assert call["rows"] * call["batch"] < call["events"]
    for parameter in model.sparse_parameters():
        assert dense(parameter.grad).shape == parameter.shape
        assert torch.isfinite(dense(parameter.grad)).all()


def test_short_context_keeps_event_scratch_when_smaller(monkeypatch):
    native.enable_capacity("cpu", required=True)
    calls = spy_backward(monkeypatch)
    model = DABSNCapacitySubsystem(3, 2, CapacityConfig(mode="sparse", size=64, active_units=2))
    x = torch.randn(2, 3, 3, requires_grad=True)
    model.forward_sequence(x).sum().backward()
    assert calls[0]["rows"] == calls[0]["events"] == 12


# --------------------------------------------------------------------------
# Both layouts compute the same thing.
# --------------------------------------------------------------------------

CASES = [
    # batch, steps, input, state, size, active, address
    (2, 9, 3, 5, 8, 2, "flat"),
    (1, 6, 4, 2, 16, 4, "product"),
    (3, 5, 6, 3, 32, 4, "flat"),
]
TOLERANCE = {torch.float64: dict(rtol=1e-12, atol=1e-13),
             torch.float32: dict(rtol=2e-5, atol=2e-6),
             torch.bfloat16: dict(rtol=2e-2, atol=2e-2)}


def _inputs(case, dtype, repeated, seed):
    batch, steps, d, _, _, _, _ = case
    generator = torch.Generator().manual_seed(seed)
    if repeated:
        # Zero input ties every token to the first bucket: one row set repeated.
        return torch.zeros(batch, steps, d, dtype=dtype)
    return torch.randn(batch, steps, d, generator=generator).to(dtype)


def _model(case, dtype, seed):
    _, _, d, s, size, active, address = case
    torch.manual_seed(seed)
    model = DABSNCapacitySubsystem(d, s, CapacityConfig(
        mode="sparse", size=size, active_units=active, address_mode=address, address_branching=2))
    with torch.no_grad():
        # Learned, nonzero output rows so every gradient path carries signal.
        model.value.normal_(std=0.3)
        model.retention_logit.normal_()
    return model.to(dtype)


def _bptt(case, dtype, repeated, seed):
    batch, steps = case[:2]
    model = _model(case, dtype, seed)
    x = _inputs(case, dtype, repeated, seed + 1).requires_grad_()
    carry = torch.randn(batch, case[4], generator=torch.Generator().manual_seed(seed + 2),
                        dtype=torch.float32).to(torch.promote_types(dtype, torch.float32))
    carry.requires_grad_()
    target = torch.randn(batch, steps, case[3], generator=torch.Generator().manual_seed(seed + 3)).to(dtype)
    y, final = model.forward_sequence(x, initial_features=carry, return_state=True)
    ((y.float() * target.float()).sum() + final.float().square().sum()).backward()
    return [y.detach(), final.detach(), x.grad, carry.grad,
            *[dense(p.grad) for p in model.parameters()]]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("repeated", [False, True])
def test_bptt_row_and_event_layouts_agree(monkeypatch, case, dtype, repeated):
    native.enable_capacity("cpu", required=True)
    results = {}
    for layout in ("event", "row"):
        force_layout(monkeypatch, layout)
        first = _bptt(case, dtype, repeated, seed=31)
        again = _bptt(case, dtype, repeated, seed=31)
        for a, b in zip(first, again):  # deterministic within each layout
            assert torch.equal(a, b)
        results[layout] = first
    for a, b in zip(results["event"], results["row"]):
        torch.testing.assert_close(a, b, **TOLERANCE[dtype])


class _TwoCalls(torch.nn.Module):
    """Two Capacity calls feeding the same parameters, fresh streams each call."""

    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def forward(self, x):
        outputs = []
        for view in (x, x.flip(1)):  # the second call routes different rows
            rec, elig = self.capacity.initial_online_state(
                x.shape[0], device=x.device, recurrent_dtype=x.dtype, eligibility_dtype=torch.float32)
            outputs.append(self.capacity.forward_online(view, rec, elig)[0])
        return outputs[0] + outputs[1]


def _online_run(monkeypatch, case, dtype, layout, deferred, scale):
    batch, steps = case[:2]
    model = _TwoCalls(_model(case, dtype, seed=57))
    optimizer = torch.optim.SparseAdam(list(model.capacity.sparse_parameters()), lr=1e-2)
    if layout is not None:
        force_layout(monkeypatch, layout)
    records = []
    with (native.capture_capacity_rows(model) if deferred else _null()):
        for replay in range(3):
            model.zero_grad(set_to_none=True)
            # Selections change between replays; one replay has zero loss so
            # every selected row must still be reported with a zero gradient.
            x = _inputs(case, dtype, False, seed=100 + replay).requires_grad_()
            y = model(x)
            loss_scale = 0.0 if replay == 1 else scale
            (y.float().square().sum() * loss_scale).backward()
            native.finalize_capacity_gradients(model)
            grads = [p.grad.coalesce() for p in model.capacity.sparse_parameters()]
            native.finalize_capacity_gradients(model)  # second call consumes nothing
            for p, g in zip(model.capacity.sparse_parameters(), grads):
                assert torch.equal(p.grad.coalesce().indices(), g.indices())
                assert torch.equal(p.grad.coalesce().values(), g.values())
            selected = torch.unique(torch.cat((model.capacity.route(x).reshape(-1),
                                               model.capacity.route(x.flip(1)).reshape(-1))))
            for g in grads:
                assert torch.equal(g.indices()[0], selected)
            records.append([x.grad.clone(), y.detach(), *[g.indices() for g in grads],
                            *[g.values() for g in grads],
                            model.capacity.gain.grad.clone()])
            optimizer.step()
            records[-1].extend(p.detach().clone() for p in model.capacity.sparse_parameters())
    return records


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_captured_online_row_totals_match_events_and_eager(monkeypatch, case, dtype):
    native.enable_capacity("cpu", required=True)
    eager = _online_run(monkeypatch, case, dtype, None, deferred=False, scale=1.0)
    monkeypatch.undo()
    native.enable_capacity("cpu", required=True)
    event = _online_run(monkeypatch, case, dtype, "event", deferred=True, scale=1.0)
    row = _online_run(monkeypatch, case, dtype, "row", deferred=True, scale=1.0)
    for replay in range(3):
        for a, b, c in zip(eager[replay], event[replay], row[replay]):
            if a.dtype == torch.long:
                assert torch.equal(a, b) and torch.equal(a, c)
            else:
                torch.testing.assert_close(b, a, **TOLERANCE[dtype])
                torch.testing.assert_close(c, a, **TOLERANCE[dtype])


def test_captured_online_carries_identical_eligibility_in_both_layouts(monkeypatch):
    native.enable_capacity("cpu", required=True)
    case = CASES[0]
    states = {}
    for layout in ("event", "row"):
        force_layout(monkeypatch, layout)
        model = _model(case, torch.float64, seed=5)
        rec, elig = model.initial_online_state(case[0], device="cpu", recurrent_dtype=torch.float64,
                                               eligibility_dtype=torch.float64)
        with native.capture_capacity_rows():
            for chunk in range(3):  # carried across chunks, like a stream
                x = _inputs(case, torch.float64, False, seed=chunk)
                y, rec, elig = model.forward_online(x, rec, elig)
                y.square().sum().backward()
                native.finalize_capacity_gradients(model)
        states[layout] = [rec["features"].clone(), *[v.clone() for v in elig.values()]]
    for a, b in zip(states["event"], states["row"]):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-13)
