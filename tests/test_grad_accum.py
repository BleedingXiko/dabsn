import copy

import pytest
import torch


def test_low_precision_accumulation_does_not_materialize_fp32_gradient_copy():
    from torch.utils._python_dispatch import TorchDispatchMode
    from dabsn.runtime.grad_accum import ManualGradientAccumulator

    class RecordCopies(TorchDispatchMode):
        def __init__(self):
            self.copies = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._to_copy.default:
                self.copies.append(args[0].numel())
            return func(*args, **(kwargs or {}))

    model = torch.nn.Linear(17, 11, bias=False).bfloat16()
    accumulator = ManualGradientAccumulator(model)
    expected = torch.zeros_like(model.weight, dtype=torch.float32)
    recorder = RecordCopies()
    for _ in range(3):
        gradient = torch.randn_like(model.weight)
        expected.add_(gradient.float())
        model.weight.grad = gradient
        with recorder:
            accumulator.add_microbatch()
    torch.testing.assert_close(accumulator.buffers[model.weight], expected, rtol=0, atol=0)
    assert model.weight.numel() not in recorder.copies

from dabsn.runtime import ManualGradientAccumulator


@pytest.mark.parametrize("microbatches", [1, 2, 4])
def test_manual_accumulator_matches_eager(microbatches):
    torch.manual_seed(17)
    model = torch.nn.Linear(5, 3)
    reference = copy.deepcopy(model)
    batches = [torch.randn(7, 5) for _ in range(microbatches)]

    accumulator = ManualGradientAccumulator(model)
    accumulator.reset()
    for batch in batches:
        accumulator.begin_microbatch()
        model(batch).float().square().mean().div(microbatches).backward()
        accumulator.add_microbatch()
    accumulator.install()

    reference.zero_grad(set_to_none=True)
    for batch in batches:
        reference(batch).float().square().mean().div(microbatches).backward()

    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)


def test_manual_accumulator_reset_clears_installed_and_buffered_gradients():
    model = torch.nn.Linear(3, 2)
    accumulator = ManualGradientAccumulator(model)
    accumulator.begin_microbatch()
    model(torch.randn(4, 3)).sum().backward()
    accumulator.add_microbatch()
    accumulator.install()
    accumulator.reset()
    assert accumulator.microbatches == 0
    assert not accumulator.active
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.count_nonzero(buffer) == 0 for buffer in accumulator.buffers.values())
