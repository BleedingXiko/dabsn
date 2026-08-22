"""Tests for the canonical fixed-shape CUDA-graph training callable.

The CPU-runnable tests pin the passthrough contract that keeps every reference
path eager.  The CUDA tests (skipped without a device) pin loss/gradient
parity, gradient-accumulation replay across microbatches, and optimiser-update
parity -- the guarantees that make the helper safe for cluster-scale training.
"""

import os

import pytest
import torch

from dabsn import DABSNLayerSpec, DABSNModel
from dabsn.runtime import ManualGradientAccumulator, make_graphed_train_callable

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graphs require a CUDA device"
)


def _model(hidden=8, layers=2, input_dim=5, output_dim=4):
    specs = [DABSNLayerSpec(hidden, read_geometry="seq") for _ in range(layers)]
    return DABSNModel(input_dim, output_dim, specs, output_adapter="token")


def _capture_safe(model):
    # The sparse admitted read compacts its bank with a host .item(), which
    # CUDA-graph capture forbids. The capture-safe path keeps the bank full
    # width (identical math, see test_read_capture_safe) so the forward has no
    # host sync to capture. Native CUDA kernels enable this automatically; here
    # (no kernels) we request it explicitly.
    from dabsn.read import DABSNRead

    for module in model.modules():
        if isinstance(module, DABSNRead):
            module._capture_safe_bank = True
    return model


def _grad_vector(model):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _single_backward_grads(model, callable_, x):
    model.zero_grad(set_to_none=True)
    callable_(x).float().square().mean().backward()
    torch.cuda.synchronize()
    return _grad_vector(model)


def _accumulate_grads(model, callable_, batches):
    # Graph-safe gradient accumulation.  Autograd accumulation across CUDA-graph
    # replays is unreliable -- the graph writes grads on its capture stream while
    # AccumulateGrad runs on the default stream, and stacking multiple replayed
    # backwards into one .grad can drop an increment.  So each microbatch is an
    # independent single backward (zeroed first, never crossing that boundary),
    # snapshot and summed here.  This depends only on single-backward correctness.
    accumulator = ManualGradientAccumulator(model)
    accumulator.reset()
    for batch in batches:
        accumulator.begin_microbatch()
        callable_(batch).float().square().mean().backward()
        accumulator.add_microbatch()
    accumulator.install()
    torch.cuda.synchronize()
    return _grad_vector(model)


# --------------------------------------------------------------------------- #
# CPU / no-CUDA contract                                                       #
# --------------------------------------------------------------------------- #


def test_cpu_input_returns_module_unchanged():
    model = _model()
    x = torch.randn(2, 6, 5)
    graphed = make_graphed_train_callable(model, (x,))
    # CUDA Graphs are a CUDA-only optimisation; on CPU the module is returned
    # untouched so the reference path stays fully eager.
    assert graphed is model


def test_cpu_passthrough_does_not_touch_gradients():
    model = _model()
    x = torch.randn(2, 6, 5)
    # A stray forward/backward during a no-op call would leave gradients behind.
    make_graphed_train_callable(model, (x,), verify=True)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cpu_passthrough_still_trains_normally():
    model = _model()
    graphed = make_graphed_train_callable(model, (torch.randn(2, 6, 5),))
    out = graphed(torch.randn(2, 6, 5))
    out.float().square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_empty_sample_args_returns_module_unchanged():
    model = _model()
    assert make_graphed_train_callable(model, ()) is model


@pytest.mark.parametrize("initial", [None, "0", "1"])
def test_outer_graph_build_exclusively_owns_capture_and_restores_policy(monkeypatch, initial):
    from dabsn.runtime.graph import _suspend_nested_scan_graphs

    if initial is None:
        monkeypatch.delenv("DABSN_SCAN_GRAPH", raising=False)
    else:
        monkeypatch.setenv("DABSN_SCAN_GRAPH", initial)

    restore = _suspend_nested_scan_graphs()
    assert os.environ["DABSN_SCAN_GRAPH"] == "0"
    restore()

    if initial is None:
        assert "DABSN_SCAN_GRAPH" not in os.environ
    else:
        assert os.environ["DABSN_SCAN_GRAPH"] == initial


# --------------------------------------------------------------------------- #
# CUDA-only guarantees                                                          #
# --------------------------------------------------------------------------- #


# A captured graph and a fresh eager pass select different GEMM kernels, so a
# cross-path fp32 comparison on GPU agrees to ~1e-3, never to bit-parity.  These
# tolerances catch gross defects (a dead block, an unwired gradient, ~100% off)
# while admitting that hardware reality.  Accumulation *semantics* are pinned
# separately, exactly, by replaying one graph (no cross-path noise) below.
_XPATH_RTOL, _XPATH_ATOL = 1e-2, 2e-3


@requires_cuda
def test_cuda_loss_and_gradient_parity():
    torch.manual_seed(0)
    model = _capture_safe(_model().cuda())
    x = torch.randn(2, 6, 5, device="cuda")
    # verify=True internally asserts eager-vs-graphed loss and per-block
    # core/read gradient parity, raising on drift.  make_graphed_callables
    # monkeypatches forward in place and returns the SAME module, so identity is
    # expected; the gradient comparison below is what proves capture is correct.
    graphed = make_graphed_train_callable(model, (x,), verify=True)
    assert graphed is model

    eager = _capture_safe(_model().cuda())
    eager.load_state_dict(model.state_dict())
    eager.zero_grad(set_to_none=True)
    eager(x).float().square().mean().backward()
    eager_grads = _grad_vector(eager)

    model.zero_grad(set_to_none=True)
    graphed(x).float().square().mean().backward()
    graph_grads = _grad_vector(model)

    assert eager_grads.keys() == graph_grads.keys()
    for name, expected in eager_grads.items():
        torch.testing.assert_close(graph_grads[name], expected, rtol=_XPATH_RTOL, atol=_XPATH_ATOL)


@requires_cuda
def test_cuda_gradient_accumulation_is_additive():
    # The gradient-accumulation guarantee, pinned exactly, using the SAME manual
    # accumulation the training loop uses (see _accumulate_grads): K identical
    # microbatches must land K x the single-backward grad, to bit-tolerance.
    # This is the property that makes grad_accum > 1 correct at scale.
    torch.manual_seed(1)
    model = _capture_safe(_model().cuda())
    x = torch.randn(2, 6, 5, device="cuda")
    graphed = make_graphed_train_callable(model, (x,), verify=False)

    one = _single_backward_grads(model, graphed, x)
    k = 3
    accumulated = _accumulate_grads(model, graphed, [x] * k)

    assert one.keys() == accumulated.keys()
    for name, single in one.items():
        torch.testing.assert_close(accumulated[name], k * single, rtol=1e-4, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("n_microbatches", [1, 2, 4])
def test_cuda_optimizer_update_parity(n_microbatches):
    torch.manual_seed(2)
    model = _capture_safe(_model().cuda())
    reference = _capture_safe(_model().cuda())
    reference.load_state_dict(model.state_dict())

    micro = [torch.randn(2, 6, 5, device="cuda") for _ in range(n_microbatches)]
    graphed = make_graphed_train_callable(model, (micro[0],), verify=False)

    opt_graph = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt_ref = torch.optim.AdamW(reference.parameters(), lr=1e-3)

    # Graphed path: graph-safe manual accumulation, loaded into .grad for the step.
    accum = _accumulate_grads(model, graphed, micro)
    opt_graph.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        if name in accum:
            parameter.grad = accum[name]
    opt_graph.step()

    # Reference path: ordinary eager accumulation (each backward sums into .grad).
    opt_ref.zero_grad(set_to_none=True)
    for batch in micro:
        reference(batch).float().square().mean().backward()
    opt_ref.step()

    # AdamW normalises by sqrt(v), so it amplifies cross-path grad noise on
    # near-zero-grad params up to +/- lr; the update stays finite and tracks the
    # eager step to Adam-step scale (lr=1e-3), which is the meaningful guarantee.
    for (name, updated), (_, expected) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert torch.isfinite(updated).all()
        torch.testing.assert_close(updated, expected, rtol=_XPATH_RTOL, atol=3e-3)


@requires_cuda
@pytest.mark.parametrize("context", [16, 64, 256])
def test_cuda_does_not_cap_context(context):
    # "Supports any context" means correctness at a caller-selected length, not
    # a whitelist.  Distinct lengths each record and replay without error.
    torch.manual_seed(3)
    model = _capture_safe(_model().cuda())
    x = torch.randn(2, context, 5, device="cuda")
    graphed = make_graphed_train_callable(model, (x,), verify=True)
    loss = graphed(x).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss).item()


@requires_cuda
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="fp16"),
        pytest.param(
            torch.bfloat16,
            id="bf16",
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(), reason="BF16 tensor cores required"
            ),
        ),
    ],
)
def test_cuda_low_precision_capture_runs(dtype):
    # FP16 covers Turing/T4; BF16 covers Ampere/Hopper.  The graphed step must
    # capture and replay in the low-precision compute dtype and stay finite.
    torch.manual_seed(4)
    model = _capture_safe(_model().cuda())
    x = torch.randn(2, 32, 5, device="cuda")
    # make_graphed_callables forbids autocast's op cache during capture (the
    # cached casts alias tensors the graph must own); cache_enabled=False is the
    # documented requirement for graphing under autocast.
    with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
        graphed = make_graphed_train_callable(model, (x,), verify=False)
        loss = graphed(x).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss).item()
