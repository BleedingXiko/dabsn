import copy

import pytest
import torch

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
