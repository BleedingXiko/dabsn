import json

import pytest
import torch

from dabsn import DABSNLayerSpec, DABSNModel, ParallelAxis, ParallelTopology
from dabsn.distributed import (
    DABSNSequenceModule,
    DistributedState,
    inspect_sharded_training_checkpoint,
    load_sharded_training_checkpoint,
    save_sharded_training_checkpoint,
)


def _objects():
    model = DABSNSequenceModule(
        DABSNModel(4, 3, [DABSNLayerSpec(5, 6, "seq")], output_adapter="token")
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_dcp_v2_has_validated_transaction_manifest_and_commit(tmp_path):
    model, optimizer = _objects()
    destination = tmp_path / "checkpoint"
    state = DistributedState(device=torch.device("cpu"))
    save_sharded_training_checkpoint(
        model,
        optimizer,
        destination,
        state,
        step=7,
        extra={"loader_cursor": 99, "accumulation": 4},
    )
    manifest = inspect_sharded_training_checkpoint(destination)
    assert manifest["version"] == 2
    assert manifest["step"] == 7
    assert manifest["transaction"]
    assert manifest["topology"]["world_size"] == 1
    assert manifest["loader_cursor"] == 99
    assert manifest["accumulation"] == 4
    assert manifest["digests"]
    commit = json.loads((destination / "COMMITTED.json").read_text())
    assert commit["transaction"] == manifest["transaction"]


def test_incomplete_v2_checkpoint_never_inspects_as_committed(tmp_path):
    destination = tmp_path / "incomplete"
    destination.mkdir()
    (destination / "dabsn-training.json").write_text(
        json.dumps(
            {
                "format": "dabsn-distributed-checkpoint",
                "version": 2,
                "transaction": "abc",
                "digests": {},
            }
        )
    )
    with pytest.raises(ValueError, match="no commit marker"):
        inspect_sharded_training_checkpoint(destination)


def test_corrupt_committed_shard_is_rejected(tmp_path):
    model, optimizer = _objects()
    destination = tmp_path / "checkpoint"
    state = DistributedState(device=torch.device("cpu"))
    save_sharded_training_checkpoint(model, optimizer, destination, state, step=1)
    shard = next(
        path
        for path in destination.rglob("*")
        if path.is_file() and path.name not in {"dabsn-training.json", "COMMITTED.json"}
    )
    with shard.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="digest validation"):
        inspect_sharded_training_checkpoint(destination)


def test_dcp_exact_resume_restores_rng_and_scaler(tmp_path):
    class Scaler:
        def __init__(self, scale):
            self.scale = float(scale)

        def state_dict(self):
            return {"scale": self.scale}

        def load_state_dict(self, state):
            self.scale = float(state["scale"])

    model, optimizer = _objects()
    destination = tmp_path / "checkpoint"
    state = DistributedState(device=torch.device("cpu"))
    torch.manual_seed(1234)
    saved_scaler = Scaler(64.0)
    saved_scheduler = Scaler(0.25)
    save_sharded_training_checkpoint(
        model,
        optimizer,
        destination,
        state,
        step=3,
        scaler=saved_scaler,
        scheduler=saved_scheduler,
    )
    expected_random = torch.rand(4)
    torch.manual_seed(9999)
    restored_model, restored_optimizer = _objects()
    restored_scaler = Scaler(1.0)
    restored_scheduler = Scaler(1.0)
    load_sharded_training_checkpoint(
        restored_model,
        restored_optimizer,
        destination,
        state,
        scaler=restored_scaler,
        scheduler=restored_scheduler,
    )
    torch.testing.assert_close(torch.rand(4), expected_random, atol=0, rtol=0)
    assert restored_scaler.scale == 64.0
    assert restored_scheduler.scale == 0.25


def test_dcp_scaler_state_requires_matching_resume_object(tmp_path):
    class Scaler:
        def state_dict(self):
            return {"scale": 2.0}

    model, optimizer = _objects()
    destination = tmp_path / "checkpoint"
    state = DistributedState(device=torch.device("cpu"))
    save_sharded_training_checkpoint(
        model,
        optimizer,
        destination,
        state,
        step=1,
        scaler=Scaler(),
    )
    with pytest.raises(ValueError, match="provide the matching scaler"):
        load_sharded_training_checkpoint(model, optimizer, destination, state)


def test_rng_topology_mismatch_fails_before_loading_parameters(tmp_path):
    model, optimizer = _objects()
    destination = tmp_path / "checkpoint"
    source_state = DistributedState(device=torch.device("cpu"))
    save_sharded_training_checkpoint(
        model,
        optimizer,
        destination,
        source_state,
        step=1,
    )
    restored_model, restored_optimizer = _objects()
    before = {name: value.detach().clone() for name, value in restored_model.state_dict().items()}
    mismatched = DistributedState(world_size=2, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="checkpoint worker count 1, received 2"):
        load_sharded_training_checkpoint(
            restored_model,
            restored_optimizer,
            destination,
            mismatched,
        )
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_same_worker_count_with_different_axis_ownership_fails_before_load(tmp_path):
    model, optimizer = _objects()
    destination = tmp_path / "checkpoint"
    source_state = DistributedState(
        device=torch.device("cpu"),
        topology=ParallelTopology((ParallelAxis("data", 1),)),
    )
    save_sharded_training_checkpoint(model, optimizer, destination, source_state, step=1)
    restored_model, restored_optimizer = _objects()
    before = {name: value.detach().clone() for name, value in restored_model.state_dict().items()}
    different_axes = DistributedState(
        device=torch.device("cpu"),
        topology=ParallelTopology((ParallelAxis("tensor", 1),)),
    )
    with pytest.raises(ValueError, match="parallel topology differs"):
        load_sharded_training_checkpoint(
            restored_model,
            restored_optimizer,
            destination,
            different_axes,
        )
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_intentional_topology_change_requires_rng_opt_out(tmp_path):
    model, optimizer = _objects()
    destination = tmp_path / "checkpoint"
    source_state = DistributedState(device=torch.device("cpu"))
    save_sharded_training_checkpoint(model, optimizer, destination, source_state, step=1)
    restored_model, restored_optimizer = _objects()
    different_axes = DistributedState(
        device=torch.device("cpu"),
        topology=ParallelTopology((ParallelAxis("replica", 1),)),
    )
    with pytest.raises(ValueError, match="restore_rng=False"):
        load_sharded_training_checkpoint(
            restored_model,
            restored_optimizer,
            destination,
            different_axes,
            allow_topology_change=True,
        )
    load_sharded_training_checkpoint(
        restored_model,
        restored_optimizer,
        destination,
        different_axes,
        allow_topology_change=True,
        restore_rng=False,
    )
