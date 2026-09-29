import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from dabsn.capacity import DABSNCapacitySubsystem
from dabsn.config import CapacityConfig
from dabsn.distributed import DistributedState, wrap_distributed
from dabsn.runtime import build_training_optimizer, clip_grad_norm


class SparseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.capacity = DABSNCapacitySubsystem(3, 5, CapacityConfig(mode="sparse", size=8, active_units=2))

    def forward(self, x):
        recurrent, eligibility = self.capacity.initial_online_state(
            x.shape[0], device=x.device, recurrent_dtype=x.dtype, eligibility_dtype=x.dtype)
        return self.capacity.forward_online(x, recurrent, eligibility)[0]


def _mixed_precision_rows_worker(rank, path):
    import copy
    from dabsn.capacity_distributed import shard_capacity_rows
    from dabsn.kernels.capacity import enable_capacity
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=2)
    try:
        for backend in ("reference", "cpu"):
            enable_capacity(backend, required=True)
            for dtype in (torch.float16, torch.bfloat16):
                torch.manual_seed(632)
                reference = DABSNCapacitySubsystem(3, 5,
                    CapacityConfig(mode="sparse", size=8, active_units=2))
                warmup = torch.randn(2, 3, 3)
                reference.forward_sequence(warmup).sum().backward()
                with torch.no_grad():
                    reference.value.add_(reference.value.grad, alpha=-.05)
                reference.zero_grad(set_to_none=True)
                model = copy.deepcopy(reference)
                shard_capacity_rows(model, dist.group.WORLD)
                x = (torch.randn(2, 3, 3) + rank).to(dtype)
                expected, expected_state = reference.forward_sequence(x, return_state=True)
                actual, state = model.forward_sequence(x, return_state=True)
                assert actual.dtype == dtype and state.dtype == torch.float32
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
                torch.testing.assert_close(state, expected_state, atol=2e-3, rtol=2e-2)
                actual.float().sum().backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                model.zero_grad(set_to_none=True)
                rec, elig = model.initial_online_state(2, device="cpu",
                    recurrent_dtype=dtype, eligibility_dtype=torch.float32)
                result, rec, elig = model.forward_online(x, rec, elig)
                result.float().sum().backward()
                assert result.dtype == dtype and rec["features"].dtype == torch.float32
                assert all(t.dtype == torch.float32 for t in elig.values())
                torch.testing.assert_close(result, expected, atol=2e-3, rtol=2e-2)
    finally:
        dist.destroy_process_group()


def test_row_partitions_preserve_mixed_precision_state_and_execution(tmp_path):
    mp.spawn(_mixed_precision_rows_worker, args=(str(tmp_path / "mixed"),), nprocs=2, join=True)


def _worker(rank, path, kind):
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=2)
    try:
        torch.manual_seed(45)
        model = SparseModel()
        if kind == "ddp":
            wrapped = wrap_distributed(model, DistributedState(kind="ddp", rank=rank, world_size=2))
            opt = build_training_optimizer(wrapped, lr=.01)
            x = torch.randn(2, 3, 3) + rank
            (wrapped(x) - 1).square().sum().backward()
            # Zero initialization gives a value-table gradient and is deliberately preserved.
            local = model.capacity.value.grad.to_dense()
            expected = local.clone()
            dist.all_reduce(expected)
            expected.div_(2)
            opt.prepare_gradients()
            torch.testing.assert_close(model.capacity.value.grad.to_dense(), expected)
            clip_grad_norm(wrapped, .1)
            clipped = model.capacity.value.grad.to_dense().clone()
            # Changing one worker after prepare must not trigger a second reduction.
            # AMP unscale and clipping similarly mutate the synchronized COO values.
            if rank == 0:
                model.capacity.value.grad._values().mul_(2)
            local_prepared = model.capacity.value.grad.to_dense().clone()
            opt.prepare_gradients()
            torch.testing.assert_close(model.capacity.value.grad.to_dense(), local_prepared)
            if rank == 0:
                model.capacity.value.grad._values().mul_(.5)
            opt.step()
            torch.testing.assert_close(model.capacity.value.grad.to_dense(), clipped)
            gathered = [torch.empty_like(model.capacity.value) for _ in range(2)]
            dist.all_gather(gathered, model.capacity.value)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        else:
            import dabsn.capacity_distributed as cd
            assert hasattr(cd, "shard_capacity_rows"), "whole-row partitioning is missing"
            reference = SparseModel()
            reference.load_state_dict(model.state_dict())
            # Learned nonzero values exercise key/bias/retention derivatives too.
            with torch.no_grad():
                model.capacity.value.normal_()
                reference.load_state_dict(model.state_dict())
            cd.shard_capacity_rows(model.capacity, dist.group.WORLD, axis="data")
            assert model.capacity.key.shape == (4, 3)
            x = torch.randn(2, 3, 3) + rank
            expected = reference(x)
            actual = model(x)
            torch.testing.assert_close(actual, expected)
            assert model.capacity.last_precision["accumulation_dtype"] == "torch.float32"
            actual.square().sum().backward()
            expected.square().sum().backward()
            for p, ref in zip(model.capacity.sparse_parameters(), reference.capacity.sparse_parameters()):
                expected_gradient = ref.grad.to_dense()
                dist.all_reduce(expected_gradient)
                expected_gradient.div_(2)
                assert p.grad.is_sparse
                torch.testing.assert_close(p.grad.to_dense(), expected_gradient[rank*4:(rank+1)*4], atol=1e-6, rtol=1e-5)
            before = model.capacity.value.detach().clone()
            build_training_optimizer(model, lr=.01).step()
            assert not torch.equal(before, model.capacity.value)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["ddp", "rows"])
def test_two_rank_sparse_communication(kind, tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), kind), nprocs=2, join=True)


def _checkpoint_worker(rank, rendezvous, checkpoint):
    from dabsn import build_dabsn, load_dabsn
    from dabsn.capacity_distributed import shard_capacity_rows
    from dabsn.distributed import (save_sharded_training_checkpoint, load_sharded_training_checkpoint,
                                    full_training_state_dict, save_distributed_dabsn)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        state = DistributedState(kind="ddp", rank=rank, world_size=2)
        def make():
            torch.manual_seed(123)
            module = build_dabsn(3, 2, 5, state_dim=4,
                capacity=CapacityConfig(mode="sparse", size=10, active_units=2))
            cap = module.backbone.blocks[0].capacity_module
            shard_capacity_rows(cap, dist.group.WORLD, axis="data")
            return module, build_training_optimizer(module, lr=.01)
        def update(module, optim):
            optim.zero_grad()
            x = torch.randn(2, 3, 3, generator=torch.Generator().manual_seed(17 + rank))
            (module.forward_sequence(x) - 1).square().mean().backward()
            cap = module.backbone.blocks[0].capacity_module
            row_ids = {id(p) for p in cap.sparse_parameters()}
            for p in module.parameters():
                if p.grad is not None and id(p) not in row_ids:
                    dist.all_reduce(p.grad)
                    p.grad.div_(2)
            optim.step()
        module, optim = make()
        update(module, optim)
        save_sharded_training_checkpoint(module, optim, checkpoint, state, step=1)
        save_distributed_dabsn(module, checkpoint + ".safetensors", state)
        full_saved = full_training_state_dict(module, optim)
        if rank == 0:
            torch.save(full_saved, checkpoint + ".expected.pt")
        exported = load_dabsn(checkpoint + ".safetensors")
        assert exported.backbone.blocks[0].capacity_module.key.shape == (10, 3)
        update(module, optim)
        restored, restored_optimizer = make()
        load_sharded_training_checkpoint(restored, restored_optimizer, checkpoint, state)
        update(restored, restored_optimizer)
        for p, q in zip(module.parameters(), restored.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        consolidated = full_training_state_dict(module, optim)
        if rank == 0:
            assert consolidated["model"]["backbone.blocks.0.capacity_module.key"].shape == (10, 3)
    finally:
        dist.destroy_process_group()


def test_two_rank_checkpoint_and_portable_export(tmp_path):
    mp.spawn(_checkpoint_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "checkpoint")), nprocs=2, join=True)


def test_two_rank_row_checkpoint_reshards_to_single_process(tmp_path):
    from dabsn import build_dabsn
    from dabsn.distributed import load_sharded_training_checkpoint, full_training_state_dict
    checkpoint = str(tmp_path / "checkpoint")
    mp.spawn(_checkpoint_worker, args=(str(tmp_path / "rendezvous"), checkpoint), nprocs=2, join=True)
    module = build_dabsn(3, 2, 5, state_dim=4,
        capacity=CapacityConfig(mode="sparse", size=10, active_units=2))
    optimizer = build_training_optimizer(module, lr=.01)
    load_sharded_training_checkpoint(module, optimizer, checkpoint, DistributedState(),
                                     allow_topology_change=True, restore_rng=False)
    expected = torch.load(checkpoint + ".expected.pt", weights_only=False)
    actual = full_training_state_dict(module, optimizer)
    torch.testing.assert_close(actual["model"], expected["model"], atol=0, rtol=0)
    torch.testing.assert_close(actual["optimizer"]["state"], expected["optimizer"]["state"], atol=0, rtol=0)
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]


def _overflow_worker(rank, rendezvous):
    from dabsn.capacity_distributed import shard_capacity_rows
    from dabsn.runtime import apply_optimizer_step
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        module = SparseModel()
        shard_capacity_rows(module.capacity, dist.group.WORLD)
        optimizer = build_training_optimizer(module, lr=.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=4)
        lifecycle = []
        module.post_optimizer_step = lambda *, step_applied: lifecycle.append(step_applied)
        for overflow in (True, False):
            optimizer.zero_grad()
            scaler.scale(module.capacity.gain.square()).backward()
            for p in module.capacity.sparse_parameters():
                values = p.new_full((1, *p.shape[1:]), float("inf") if overflow and rank == 0 else 4.)
                p.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), values, p.shape, check_invariants=False)
            before = [p.detach().clone() for p in module.parameters()]
            # Exercise the pretrainer's already-unscaled boundary as well.
            scaler.unscale_(optimizer)
            applied = apply_optimizer_step(module, optimizer, scaler=scaler)
            assert applied is not overflow
            if overflow:
                for p, previous in zip(module.parameters(), before):
                    torch.testing.assert_close(p, previous, rtol=0, atol=0)
                assert not optimizer.state
        assert lifecycle == [False, True]
    finally:
        dist.destroy_process_group()


def test_owner_only_amp_overflow_skips_every_rank(tmp_path):
    mp.spawn(_overflow_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def _uneven_worker(rank, rendezvous):
    from dabsn.capacity_distributed import shard_capacity_rows
    from dabsn.distributed import _build_parallel_execution_context
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(31)
        reference = DABSNCapacitySubsystem(3, 5, CapacityConfig(mode="sparse", size=5, active_units=1))
        with torch.no_grad():
            reference.address_projection.zero_()
            reference.value.normal_()
        capacity = DABSNCapacitySubsystem(3, 5, reference.config)
        capacity.load_state_dict(reference.state_dict())
        partition = shard_capacity_rows(capacity, dist.group.WORLD)
        assert capacity.key.shape[0] == (3 if rank == 0 else 2)
        state = DistributedState(kind="ddp", rank=rank, world_size=2)
        capacity._dabsn_parallel_context = _build_parallel_execution_context(state.topology, rank=rank, device=torch.device("cpu"))
        x = torch.randn(2, 3, 3) + rank
        recurrent, eligibility = capacity.initial_online_state(2, device=x.device,
            recurrent_dtype=x.dtype, eligibility_dtype=x.dtype)
        rr, re = reference.initial_online_state(2, device=x.device,
            recurrent_dtype=x.dtype, eligibility_dtype=x.dtype)
        capacity.forward_online(x, recurrent, eligibility)[0].sum().backward()
        reference.forward_online(x, rr, re)[0].sum().backward()
        squared = torch.zeros(())
        for name, p in capacity.named_parameters():
            q = dict(reference.named_parameters())[name]
            expected = q.grad.to_dense() if q.grad.is_sparse else q.grad.clone()
            dist.all_reduce(expected)
            expected.div_(2)
            if name != "gain":
                torch.testing.assert_close(p.grad.to_dense(), expected[partition.start:partition.stop])
                if rank == 1:
                    assert p.grad._nnz() == 0
            else:
                p.grad.copy_(expected)
            squared += expected.square().sum()
        actual_norm = clip_grad_norm(capacity, float("inf"))
        torch.testing.assert_close(actual_norm, squared.sqrt())
        build_training_optimizer(capacity, lr=.01).step()
    finally:
        dist.destroy_process_group()


def test_uneven_partition_with_empty_owner_gradient_and_global_norm(tmp_path):
    mp.spawn(_uneven_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def _tp_graph():
    from dabsn import ComponentSpec, DABSNGraph, component_registry
    return DABSNGraph([component_registry.build(ComponentSpec("capacity", "dabsn:block", {
        "input_dim": 4, "hidden_dim": 6, "state_dim": 4, "read_geometry": "seq",
        "tensor_parallel_axis": "tensor", "capacity": {"mode": "sparse", "size": 5, "active_units": 1},
    }))])


def _tp_checkpoint_worker(rank, rendezvous, checkpoint):
    from dabsn import ParallelTopology, ParallelAxis
    from dabsn.distributed import prepare_distributed_module, full_training_state_dict, save_sharded_training_checkpoint
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(19)
        state = DistributedState(kind="ddp", rank=rank, world_size=2,
            topology=ParallelTopology((ParallelAxis("data", 1), ParallelAxis("tensor", 2))))
        module = prepare_distributed_module(_tp_graph(), state)
        optimizer = build_training_optimizer(module, lr=.01)
        x = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(63))
        for index in range(2):
            optimizer.zero_grad()
            (module(x) - 1).square().mean().backward()
            optimizer.step()
            full = full_training_state_dict(module, optimizer)
            if index == 0:
                save_sharded_training_checkpoint(module, optimizer, checkpoint, state, step=1)
            if rank == 0:
                torch.save(full, checkpoint + f".step{index + 1}.pt")
    finally:
        dist.destroy_process_group()


def test_tp_capacity_checkpoint_reshards_and_continues_in_single_process(tmp_path):
    from dabsn.distributed import load_sharded_training_checkpoint, full_training_state_dict, prepare_distributed_module
    checkpoint = str(tmp_path / "checkpoint")
    mp.spawn(_tp_checkpoint_worker, args=(str(tmp_path / "rendezvous"), checkpoint), nprocs=2, join=True)
    module = prepare_distributed_module(_tp_graph(), DistributedState())
    optimizer = build_training_optimizer(module, lr=.01)
    with pytest.raises(ValueError, match="restore_rng=False"):
        load_sharded_training_checkpoint(module, optimizer, checkpoint, DistributedState(), allow_topology_change=True)
    load_sharded_training_checkpoint(module, optimizer, checkpoint, DistributedState(),
                                     allow_topology_change=True, restore_rng=False)
    expected = torch.load(checkpoint + ".step1.pt", weights_only=False)
    actual = full_training_state_dict(module, optimizer)
    torch.testing.assert_close(actual["model"], expected["model"], atol=0, rtol=0)
    torch.testing.assert_close(actual["optimizer"]["state"], expected["optimizer"]["state"], atol=0, rtol=0)
    optimizer.zero_grad()
    x = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(63))
    (module(x) - 1).square().mean().backward()
    optimizer.step()
    expected = torch.load(checkpoint + ".step2.pt", weights_only=False)
    actual = full_training_state_dict(module, optimizer)
    torch.testing.assert_close(actual["model"], expected["model"], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual["optimizer"]["state"], expected["optimizer"]["state"], atol=2e-6, rtol=2e-5)


def test_topology_change_still_rejects_worker_state_and_unsupported_providers():
    from dabsn import ParallelTopology, ParallelAxis
    from dabsn.distributed import _validate_checkpoint_topology
    topology = ParallelTopology((ParallelAxis("tensor", 2),))
    base = {"rng_world_size": 2, "topology": {"topology_fingerprint": topology.fingerprint()},
            "providers": ["dabsn:block"], "reshardable_providers": ["dabsn:block"]}
    for extra, match in (({"has_worker_training_state": True}, "worker"),
                         ({"providers": ["unsupported:provider"]}, "provider")):
        with pytest.raises(ValueError, match=match):
            _validate_checkpoint_topology({**base, **extra}, DistributedState(),
                                          allow_topology_change=True, restore_rng=False)
