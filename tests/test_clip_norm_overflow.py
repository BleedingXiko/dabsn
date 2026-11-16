import pytest
import torch

from dabsn.distributed import clip_grad_norm


def test_finite_gradient_norm_overflow_recovers_without_zeroing():
    model = torch.nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 1e30)
    before = [p.grad.clone() for p in model.parameters()]
    expected_norm = torch.stack([g.double().norm() for g in before]).norm()
    returned = clip_grad_norm(model, 1.0)
    assert torch.isfinite(returned)
    torch.testing.assert_close(returned.double(), expected_norm)
    for parameter, gradient in zip(model.parameters(), before):
        torch.testing.assert_close(parameter.grad,
                                  (gradient.double() / (expected_norm + 1e-6)).float())
        assert torch.count_nonzero(parameter.grad) == parameter.numel()
    assert torch.stack([p.grad.double().norm() for p in model.parameters()]).norm() <= 1.000001


def test_finite_clipping_matches_existing_path_exactly():
    model = torch.nn.Linear(2, 2)
    other = torch.nn.Linear(2, 2)
    for p, q in zip(model.parameters(), other.parameters()):
        p.grad = torch.arange(p.numel()).reshape_as(p).float() + .3
        q.grad = p.grad.clone()
    expected = torch.nn.utils.clip_grad_norm_(other.parameters(), 1.)
    actual = clip_grad_norm(model, 1.)
    assert torch.equal(actual, expected)
    for p, q in zip(model.parameters(), other.parameters()):
        assert torch.equal(p.grad, q.grad)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_gradient_rejected_before_mutation(bad):
    model = torch.nn.Linear(2, 2)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    model.weight.grad[0, 0] = bad
    bias_before = model.bias.grad.clone()
    with pytest.raises(RuntimeError, match='non-finite'):
        clip_grad_norm(model, 1.)
    assert torch.equal(model.bias.grad, bias_before)
