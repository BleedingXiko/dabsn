"""Strict CUDA repeatability regressions from the first returned T4 gate."""
import copy
import io

import pytest
import torch

from dabsn.capacity import DABSNCapacitySubsystem
from dabsn.config import CapacityConfig
from dabsn.kernels.capacity import enable_capacity

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA device")


def dense(t):
    return t.to_dense() if t.is_sparse else t


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
def test_learned_cuda_repeated_forward_and_all_gradients_are_bit_exact(mode, online, batch):
    enable_capacity("cuda", required=True)
    torch.manual_seed(218)
    model = DABSNCapacitySubsystem(17, 11, CapacityConfig(mode=mode, size=256, active_units=16)).cuda()
    x = torch.randn(batch, 23, 17, device="cuda")
    target = torch.randn(batch, 23, 11, device="cuda")
    # Open the zero-initialized value path through a real first update.
    (model.forward_sequence(x) * target).sum().backward()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(dense(p.grad), alpha=-0.01)
    expected = None
    for _ in range(6):
        model.zero_grad(set_to_none=True)
        xx = x.clone().requires_grad_()
        if online:
            rec, elig = model.initial_online_state(batch, device="cuda", recurrent_dtype=x.dtype,
                                                   eligibility_dtype=x.dtype)
            y, rec, elig = model.forward_online(xx, rec, elig)
            state = [*rec.values(), *elig.values()]
        else:
            y, final = model.forward_sequence(xx, return_state=True)
            state = [final]
        (y * target).sum().backward()
        actual = [y.detach().clone(), xx.grad.clone(),
                  *[dense(p.grad).clone() for p in model.parameters()],
                  *[v.detach().clone() for v in state]]
        if expected is not None:
            names = ["output", "input gradient", *[name + " gradient" for name, _ in model.named_parameters()],
                     *[f"state {i}" for i in range(len(state))]]
            failures = []
            for name, a, b in zip(names, actual, expected):
                try:
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                except AssertionError as error:
                    failures.append(f"{name}: {error}")
            assert not failures, "\n".join(failures)
        expected = actual


@pytest.mark.parametrize("input_dim", [7, 17, 33])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dense_reverse_carry_matches_reference_across_chunks(input_dim, dtype):
    """Exercise row layouts on both sides of a warp, including a terminal VJP."""
    torch.manual_seed(311)
    model = DABSNCapacitySubsystem(input_dim, 11, CapacityConfig(mode="dense", size=256)).cuda().to(dtype)
    x = torch.randn(3, 23, input_dim, device="cuda", dtype=dtype)
    target = torch.randn(3, 23, 11, device="cuda", dtype=dtype)
    enable_capacity("cuda", required=True)
    (model.forward_sequence(x) * target).sum().backward()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(p.grad, alpha=-0.01)
    model.zero_grad(set_to_none=True)
    reference = copy.deepcopy(model)
    initial = torch.randn(3, 256, device="cuda", dtype=dtype)
    terminal = torch.randn_like(initial)
    expected = None
    for backend, module, split in (("reference", reference, False), ("cuda", model, False),
                                   ("cuda", model, True)):
        enable_capacity(backend, required=True)
        module.zero_grad(set_to_none=True)
        xx = x.clone().requires_grad_()
        carry = initial.clone().requires_grad_()
        if split:
            left, middle = module.forward_sequence(xx[:, :9], initial_features=carry, return_state=True)
            right, final = module.forward_sequence(xx[:, 9:], initial_features=middle, return_state=True)
            y = torch.cat((left, right), dim=1)
        else:
            y, final = module.forward_sequence(xx, initial_features=carry, return_state=True)
        ((y * target).sum() + (final * terminal).sum()).backward()
        actual = [y.detach().clone(), final.detach().clone(), xx.grad.clone(), carry.grad.clone(),
                  *[p.grad.clone() for p in module.parameters()]]
        if expected is None:
            expected = actual
        else:
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=3e-4 if dtype == torch.float32 else 2e-8,
                                           atol=3e-5 if dtype == torch.float32 else 2e-9)


@pytest.mark.parametrize("trailing", [(), (3,)])
def test_cuda_segmented_row_reduction_keeps_original_duplicate_order(trailing):
    from dabsn.kernels.capacity_triton import row_reduce
    rows = torch.tensor([2, 0, 2, 1, 0, 2, 1, 0], device="cuda")
    base = torch.tensor([1e20, 1e20, 1., 5., -1e20, -1e20, -2., 3.], device="cuda")
    values = base if not trailing else base[:, None].expand(-1, 3).contiguous()
    expected = torch.zeros((4, *trailing))
    for row, value in zip(rows.cpu(), values.cpu()):
        expected[row] += value
    for _ in range(4):
        actual = row_reduce(rows, values, 4)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("precision", [torch.float32, torch.float16])
def test_cuda_serialized_optimizer_continuation_is_bit_exact(mode, precision):
    from dabsn.runtime.optimizer import build_training_optimizer
    enable_capacity("cuda", required=True)
    torch.manual_seed(142)
    config = CapacityConfig(mode=mode, size=256, active_units=16)
    model = DABSNCapacitySubsystem(17, 11, config).cuda()
    optimizer = build_training_optimizer(model, lr=0.003)
    x = torch.randn(3, 23, 17, device="cuda")
    target = torch.randn(3, 23, 11, device="cuda")

    def step(module, opt):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=precision, enabled=precision != torch.float32):
            y = module.forward_sequence(x)
        (y.float() * target).mean().backward()
        opt.step()

    for _ in range(3):
        step(model, optimizer)
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    restored = DABSNCapacitySubsystem(17, 11, config).cuda()
    restored.load_state_dict(saved["model"])
    restored_optimizer = build_training_optimizer(restored, lr=0.003)
    restored_optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    for _ in range(2):
        step(model, optimizer)
        step(restored, restored_optimizer)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("mostly_unique", [False, True])
def test_arbitrary_sparse_slots_and_online_row_reductions_are_bit_exact(mostly_unique):
    from dabsn.kernels.capacity import capacity_scan
    enable_capacity("cuda", required=True)
    batch, steps, active, count = 3, 11, 4, 512
    model = DABSNCapacitySubsystem(7, 5, CapacityConfig(mode="sparse", size=count,
                                                     active_units=active)).cuda()
    x = torch.randn(batch, steps, 7, device="cuda")
    if mostly_unique:
        indices = torch.randperm(count,device="cuda")[:batch*steps*active].reshape(batch,steps,active)
        indices[1,0] = indices[0,0]
    else:
        indices = torch.stack([torch.arange(active,device="cuda").roll(t%active)
                               for t in range(steps)]).expand(batch,-1,-1).contiguous()
    def forward(online):
        rec, elig = model.initial_online_state(batch,device="cuda",recurrent_dtype=x.dtype,
                                               eligibility_dtype=x.dtype)
        return capacity_scan(x,model.key,model.bias,model.retention_logit,model.value,
                             model.gain,rec["features"],indices,sparse=True,
                             eligibility=tuple(elig.values()) if online else None)[0]
    forward(False).sum().backward()
    with torch.no_grad():
        for p in model.parameters(): p.add_(dense(p.grad),alpha=-0.03)
    expected = None
    for _ in range(4):
        model.zero_grad(set_to_none=True)
        y = forward(True)
        y.square().sum().backward()
        actual = [y.detach().clone(),*[dense(p.grad).clone() for p in model.parameters()]]
        if expected is not None:
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        expected = actual
