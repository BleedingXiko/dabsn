"""Regression for FSDP1 replacing parameter owners under legacy graph views."""

import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from dabsn import CapacityConfig, build_dabsn


class AddAfterForward(nn.Module):
    """A real module replacement whose output makes bypasses observable."""
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, inputs, **kwargs):
        return self.module(inputs, **kwargs) + .125


@pytest.mark.parametrize("owner", ["blocks", "middle_mlps"])
def test_legacy_graph_uses_current_registered_owner(owner):
    model = build_dabsn(3, 2, 5, depth=2, state_dim=4,
                        mlp_ratio=2., mlp_middle_depth=1, mlp_depth_index=0)
    x = torch.randn(2, 3, 3)
    before = model.forward_sequence(x)
    modules = getattr(model.backbone, owner)
    modules[0] = AddAfterForward(modules[0])
    after = model.forward_sequence(x)
    assert not torch.equal(before, after), "graph bypassed a replacement of its registered parameter owner"


def _fsdp1_worker(rank, rendezvous, device_type, kind="fsdp"):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from dabsn.capacity_distributed import capacity_modules, shard_capacity_rows
    from dabsn.distributed import (_component_fsdp_boundaries, DistributedState, wrap_distributed,
        DABSNSequenceModule, save_sharded_training_checkpoint, load_sharded_training_checkpoint)
    from dabsn.runtime import build_training_optimizer, train_step, apply_optimizer_step
    from dabsn.kernels.capacity import enable_capacity

    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    enable_capacity(device.type, required=True)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo",
                            init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        for mode in ("off", "dense", "sparse"):
            for geometry in ("seq", "field", "hybrid"):
                torch.manual_seed(92)
                cfg = None if mode == "off" else CapacityConfig(mode=mode, size=10, active_units=2)
                model = build_dabsn(3, 2, 6, depth=2, state_dim=4,
                    read_geometry=geometry, mlp_ratio=2., mlp_middle_depth=1,
                    mlp_depth_index=0, capacity=cfg).to(device)
                reference = DABSNSequenceModule(copy.deepcopy(model))
                model = DABSNSequenceModule(model)
                state = DistributedState(kind=kind, rank=rank, local_rank=rank, world_size=2, device=device)
                if device_type == "cuda" or kind != "fsdp":
                    wrapped = wrap_distributed(model, state)
                else:
                    # CPU permits actual FSDP1 sharding, but its initial broadcast
                    # requires CUDA. Identical seeded weights replace only that
                    # device-specific initialization step in this diagnostic.
                    ignored = []
                    for cap in capacity_modules(model, sparse_only=False):
                        if cap.mode == "sparse":
                            shard_capacity_rows(cap, dist.group.WORLD)
                            ignored.extend(cap.sparse_parameters())
                        ignored.append(cap.gain)
                        cap._capacity_gain_replicated = True
                        cap._capacity_gain_replica_group = dist.group.WORLD
                    boundary_ids = {id(module) for module in _component_fsdp_boundaries(model)}
                    def policy(module, recurse, nonwrapped_numel):
                        return recurse or id(module) in boundary_ids
                    wrapped = FSDP(model, device_id=device, use_orig_params=True,
                        auto_wrap_policy=policy, ignored_states=ignored or None,
                        sync_module_states=False)
                optimizer = build_training_optimizer(wrapped, lr=.001)
                reference_optimizer = build_training_optimizer(reference, lr=.001)
                scaler = torch.amp.GradScaler(device.type, init_scale=8)
                reference_scaler = torch.amp.GradScaler(device.type, init_scale=8)
                x = torch.randn(4, 3, 3, generator=torch.Generator().manual_seed(71)).to(device)
                targets = torch.ones(4, 3, 2, device=device)
                for _ in range(2):
                    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        train_step(wrapped, x.chunk(2)[rank], targets.chunk(2)[rank], optimizer, scaler=scaler)
                        train_step(reference, x, targets, reference_optimizer, scaler=reference_scaler)
                with torch.no_grad():
                    torch.testing.assert_close(wrapped(x), reference(x), rtol=3e-3, atol=5e-4)
                expected_backend = "cuda_triton" if device.type == "cuda" else "cpu_native_cpp"
                assert all(cap.last_backend == expected_backend
                           for cap in capacity_modules(wrapped, sparse_only=False))
                if kind == "fsdp2" and mode == "off" and geometry == "seq":
                    from dabsn.runtime.optimizer import unscale_training_gradients
                    optimizer.zero_grad()
                    parameter = next(wrapped.parameters())
                    parameter.grad = torch.full_like(parameter, float("inf") if rank == 0 else 1.)
                    unscale_training_gradients(optimizer, scaler)
                    assert not apply_optimizer_step(wrapped, optimizer, scaler=scaler)
                if mode == "sparse" and geometry == "seq":
                    checkpoint = rendezvous + ".checkpoint"
                    save_sharded_training_checkpoint(wrapped, optimizer, checkpoint, state, step=2, scaler=scaler)
                    train_step(wrapped, x.chunk(2)[rank], targets.chunk(2)[rank], optimizer, scaler=scaler)
                    trained_output = wrapped(x).detach().clone()
                    load_sharded_training_checkpoint(wrapped, optimizer, checkpoint, state, scaler=scaler)
                    train_step(wrapped, x.chunk(2)[rank], targets.chunk(2)[rank], optimizer, scaler=scaler)
                    torch.testing.assert_close(wrapped(x), trained_output, atol=1e-6, rtol=1e-5)
                    if kind != "ddp":
                        optimizer.zero_grad()
                        cap = capacity_modules(wrapped)[0]
                        scaler.scale(cap.gain.square()).backward()
                        p = cap.value
                        p.grad = torch.sparse_coo_tensor(torch.tensor([[0]], device=device),
                            p.new_full((1, p.shape[1]), float("inf") if rank == 0 else 1.), p.shape)
                        before = [p.detach().clone() for p in wrapped.parameters()]
                        assert not apply_optimizer_step(wrapped, optimizer, scaler=scaler)
                        for p, previous in zip(wrapped.parameters(), before):
                            torch.testing.assert_close(p, previous, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_fsdp1_legacy_world_inline_and_middle_views(tmp_path):
    mp.spawn(_fsdp1_worker, args=(str(tmp_path / "rendezvous"), "cpu"), nprocs=2, join=True)


@pytest.mark.parametrize("kind", ["fsdp2", "ddp"])
def test_two_rank_prepared_runtime_preserves_wrapper_boundaries(tmp_path, kind):
    mp.spawn(_fsdp1_worker, args=(str(tmp_path / "rendezvous"), "cpu", kind), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.parametrize("kind", ["fsdp", "fsdp2", "ddp"])
def test_two_gpu_fsdp1_legacy_world_inline_and_middle_views(tmp_path, kind):
    mp.spawn(_fsdp1_worker, args=(str(tmp_path / "rendezvous"), "cuda", kind), nprocs=2, join=True)
