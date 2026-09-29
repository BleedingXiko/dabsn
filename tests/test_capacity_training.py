import copy

import pytest
import torch
from torch import nn

from dabsn.capacity import DABSNCapacitySubsystem
from dabsn.config import CapacityConfig
from dabsn.runtime import ManualGradientAccumulator, apply_optimizer_step
import dabsn.runtime as runtime


def model():
    return DABSNCapacitySubsystem(3, 5, CapacityConfig(mode="sparse", size=8, active_units=2))


def row_grad(parameter, rows=(1, 1, 4), value=1.0):
    return torch.sparse_coo_tensor(torch.tensor([rows]),
        torch.full((len(rows), *parameter.shape[1:]), value), parameter.shape)


def optimizer(module, **kwargs):
    assert hasattr(runtime, "build_training_optimizer"), "Capacity optimizer factory is missing"
    return runtime.build_training_optimizer(module, lr=.01, **kwargs)


def test_off_mode_retains_adamw_and_caller_groups():
    module = nn.Linear(3, 5)
    actual = optimizer(module, weight_decay=.12)
    assert type(actual) is torch.optim.AdamW
    assert actual.param_groups[0]["weight_decay"] == .12


def test_mixed_updates_only_selected_rows_and_restores_exactly():
    module = model()
    opt = optimizer(module, weight_decay=.2, betas=(.8, .95), eps=1e-7)
    before = module.key.detach().clone()
    for p in module.sparse_parameters():
        p.grad = row_grad(p)
    module.gain.grad = torch.ones_like(module.gain)
    opt.step()
    torch.testing.assert_close(module.key[[0, 2, 3, 5, 6, 7]], before[[0, 2, 3, 5, 6, 7]], rtol=0, atol=0)
    assert not torch.equal(module.key[[1, 4]], before[[1, 4]])
    restored = model()
    restored.load_state_dict(module.state_dict())
    other = optimizer(restored, weight_decay=.2, betas=(.8, .95), eps=1e-7)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    for mod, optim in ((module, opt), (restored, other)):
        optim.zero_grad()
        for p in mod.sparse_parameters():
            p.grad = row_grad(p, (2, 4), .3)
        mod.gain.grad = torch.full_like(mod.gain, .4)
        optim.step()
    for left, right in zip(module.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_dense_bptt_gradients_are_converted_at_update_boundary():
    module = model()
    opt = optimizer(module)
    module.forward_sequence(torch.randn(1, 2, 3)).sum().backward()
    assert module.value.grad is not None
    opt.prepare_gradients()
    assert all(p.grad is None or p.grad.is_sparse for p in module.sparse_parameters())
    opt.step()


def test_manual_accumulation_keeps_sparse_rows_and_coalesces_before_norm():
    module = model()
    accumulator = ManualGradientAccumulator(module)
    for _ in range(2):
        module.key.grad = row_grad(module.key)
        accumulator.add_microbatch(scale=.5)
    accumulator.install()
    assert module.key.grad.is_sparse, "Capacity accumulation must not allocate a dense row table"
    expected = torch.zeros_like(module.key)
    expected[1] = 2
    expected[4] = 1
    torch.testing.assert_close(module.key.grad.to_dense(), expected)
    norm = runtime.clip_grad_norm(module, 1.0)
    torch.testing.assert_close(norm, torch.tensor(15.).sqrt())
    torch.testing.assert_close(module.key.grad.coalesce().values().norm(), torch.tensor(1.), atol=1e-6, rtol=1e-6)
    accumulator.reset()
    assert module.key.grad is None


@pytest.mark.parametrize("bad_sparse", [False, True])
def test_amp_overflow_skips_all_parameter_groups(bad_sparse):
    module = model()
    opt = optimizer(module)
    scaler = torch.amp.GradScaler("cpu", init_scale=4)
    scaler.scale(module.gain.square()).backward()
    for p in module.sparse_parameters():
        p.grad = row_grad(p, value=float("inf") if bad_sparse else 4.)
    if not bad_sparse:
        module.gain.grad.fill_(float("inf"))
    before = {name: p.detach().clone() for name, p in module.named_parameters()}
    assert apply_optimizer_step(module, opt, scaler=scaler) is False
    for name, p in module.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    assert not opt.state


def test_zero_current_bptt_gradient_keeps_addressed_rows_in_sparseadam():
    module = model()
    actual = optimizer(module)
    reference_parameter = nn.Parameter(module.value.detach().clone())
    expected = torch.optim.SparseAdam([reference_parameter], lr=.01)
    x = torch.randn(2, 3, 3)
    rows = module.route(x).unique()
    for scale in (1., 0.):
        actual.zero_grad()
        expected.zero_grad()
        (module.forward_sequence(x).sum() * scale).backward()
        reference_parameter.grad = torch.sparse_coo_tensor(rows[None], module.value.grad[rows], module.value.shape, check_invariants=False)
        actual.step()
        expected.step()
        torch.testing.assert_close(module.value, reference_parameter, atol=0, rtol=0)


def test_verify_gradients_accepts_sparse_coo_without_densifying():
    class Online(nn.Module):
        def __init__(self):
            super().__init__()
            self.capacity = model()
        def forward(self, x):
            recurrent, eligibility = self.capacity.initial_online_state(x.shape[0], device=x.device,
                recurrent_dtype=x.dtype, eligibility_dtype=x.dtype)
            return self.capacity.forward_online(x, recurrent, eligibility)[0]
    rows = runtime.verify_gradients(Online(), torch.randn(2, 3, 3), compile_forward=False,
                                   loss_builder=lambda output: (output - 1).square().sum())
    assert rows[0]["ok"]


def test_verify_gradients_checks_capacity_finiteness_in_legacy_blocks():
    from dabsn import build_dabsn
    module = build_dabsn(3, 2, 5, state_dim=4, capacity=CapacityConfig(mode="dense", size=8))
    module.backbone.blocks[0].capacity_module.value.register_hook(lambda grad: grad * float("nan"))
    with pytest.raises(RuntimeError, match="gradient preflight"):
        runtime.verify_gradients(module, torch.randn(2, 3, 3), compile_forward=False)


def test_checkpoint_restore_does_not_initialize_previously_unused_optimizer_slots(tmp_path):
    from dabsn import build_dabsn
    from dabsn.distributed import DistributedState, save_sharded_training_checkpoint, load_sharded_training_checkpoint
    def make():
        module = build_dabsn(3, 2, 5, state_dim=4,
            capacity=CapacityConfig(mode="sparse", size=8, active_units=2))
        return module, optimizer(module)
    module, original = make()
    module.forward_sequence(torch.randn(2, 3, 3)).square().mean().backward()
    original.step()
    expected = {name for name, p in module.named_parameters() if p in original.state}
    assert len(expected) < len(list(module.parameters()))
    save_sharded_training_checkpoint(module, original, tmp_path / "checkpoint", DistributedState(), step=1)
    restored, other = make()
    load_sharded_training_checkpoint(restored, other, tmp_path / "checkpoint", DistributedState())
    actual = {name for name, p in restored.named_parameters() if p in other.state}
    assert actual == expected


def test_train_step_accepts_one_scaler_for_sparse_and_dense_groups():
    from dabsn import build_dabsn
    module = build_dabsn(3, 2, 5, state_dim=4,
        capacity=CapacityConfig(mode="sparse", size=8, active_units=2))
    optim = optimizer(module)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    before = module.backbone.blocks[0].capacity_module.value.detach().clone()
    loss = runtime.train_step(module, torch.randn(2, 3, 3), torch.ones(2, 3, 2), optim, scaler=scaler)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(module.backbone.blocks[0].capacity_module.value, before)
