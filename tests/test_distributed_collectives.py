"""Real multi-process distributed training, proven without a second GPU.

Every other distributed test in this repo is single-process: checkpoint
roundtrips, state-dict key normalization, and a *fake* FSDP class injected by
monkeypatch. Those check plumbing, and they all pass whether or not a gradient
ever crosses a process boundary. Nothing asserted that two ranks training on
two shards of a batch arrive at the same gradients as one rank training on the
whole batch -- which is the entire point of data parallelism.

Gloo makes that testable anywhere. It is a CPU collective backend, so a real
process group, a real all-reduce and a real DDP backward all run on a laptop.
The arithmetic being checked (are the reduced gradients right?) is identical to
what NCCL does on 8 GPUs; only the transport differs. So this closes the gap
between "the wrapper is constructed correctly" and "distributed training
computes the correct thing", without renting hardware to find out.

The ranks are launched as ordinary subprocesses of this same file rather than
via multiprocessing.spawn: that is what torchrun does, it avoids re-importing a
pytest module inside a child, and a rank that deadlocks shows up as a timeout
here instead of hanging the test session.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
import torch

from dabsn import DABSNSequenceLM

# Two ranks over four sequences: each rank owns two, so the reduced gradient
# must equal the single-process gradient over all four.
_WORLD = 2
_BATCH, _SEQ, _VOCAB, _HIDDEN = 4, 6, 32, 16
_RANK_TIMEOUT_S = 180


def _build_model() -> DABSNSequenceLM:
    torch.manual_seed(1234)
    model = DABSNSequenceLM(
        vocab=_VOCAB,
        hidden_dim=_HIDDEN,
        depth=2,
        layers=f"seq:{_HIDDEN}:{_HIDDEN},field:{_HIDDEN}:{_HIDDEN}",
        tie_embeddings=False,
        residual=True,
        mlp_ratio=2.0,
    )
    with torch.no_grad():
        for block in model.backbone.blocks:
            block.mlp_fc2.weight.normal_(mean=0.0, std=0.02)
    return model


def _make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator=gen)
    targets = torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator=gen)
    return ids, targets


def _loss(module, ids, targets) -> torch.Tensor:
    logits = module(ids)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)
    )


def _strip(name: str) -> str:
    for prefix in ("module.", "_fsdp_wrapped_module.", "body."):
        name = name.replace(prefix, "")
    return name


def _single_process_grads() -> dict[str, torch.Tensor]:
    from dabsn.distributed import DABSNSequenceModule

    module = DABSNSequenceModule(_build_model())  # same forward the ranks use
    ids, targets = _make_batch()
    _loss(module, ids, targets).backward()
    return {
        _strip(n): p.grad.detach().clone()
        for n, p in module.named_parameters()
        if p.grad is not None
    }


def _rank_main(rank: int, world: int, kind: str, out_path: str) -> int:
    """One rank, run as its own process. Writes its gradients to `out_path`."""
    import torch.distributed as dist

    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from dabsn.distributed import DABSNSequenceModule, DistributedState, wrap_distributed

        module = DABSNSequenceModule(_build_model())  # same seed on every rank
        state = DistributedState(
            kind=kind,
            rank=rank,
            local_rank=rank,
            world_size=world,
            device=torch.device("cpu"),
        )
        wrapped = wrap_distributed(module, state)

        ids, targets = _make_batch()
        per_rank = _BATCH // world
        lo, hi = rank * per_rank, (rank + 1) * per_rank
        # DDP averages gradients across ranks, so each rank takes the mean over
        # its own shard: mean-of-means equals the whole-batch mean at equal
        # shard sizes, which is what makes the comparison exact.
        loss = _loss(wrapped, ids[lo:hi], targets[lo:hi])
        loss.backward()

        grads = {
            _strip(n): p.grad.detach().clone()
            for n, p in wrapped.named_parameters()
            if p.grad is not None
        }
        torch.save(
            {"grads": grads, "loss": float(loss.detach()), "report": state.report()}, out_path
        )
        return 0
    finally:
        dist.destroy_process_group()


def _run_world(kind: str) -> dict[int, dict]:
    """Launch `_WORLD` ranks as subprocesses and collect what they produced."""
    import socket

    with socket.socket() as sock:  # a port nothing else holds
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    tmp = tempfile.mkdtemp(prefix="dabsn-dist-")
    procs, paths = [], {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rank in range(_WORLD):
        paths[rank] = os.path.join(tmp, f"rank{rank}.pt")
        env = dict(
            os.environ,
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(_WORLD),
            LOCAL_RANK=str(rank),
            PYTHONPATH=os.path.join(root, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
            OMP_NUM_THREADS="1",
        )
        procs.append(
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), str(rank), kind, paths[rank]],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    results, failures = {}, []
    for rank, proc in enumerate(procs):
        try:
            output = proc.communicate(timeout=_RANK_TIMEOUT_S)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            failures.append(
                f"rank {rank} deadlocked (no exit in {_RANK_TIMEOUT_S}s) -- "
                "a collective that one rank entered and another did not"
            )
            continue
        if proc.returncode != 0:
            failures.append(f"rank {rank} exited {proc.returncode}:\n{output[-2000:]}")
            continue
        results[rank] = torch.load(paths[rank], weights_only=False)
    if failures:
        raise AssertionError("distributed ranks failed:\n" + "\n".join(failures))
    return results


def test_ddp_two_ranks_reproduce_single_process_gradients():
    """The load-bearing claim of data parallelism, actually checked.

    Two ranks each take half the batch. DDP all-reduces (averages) the
    gradients. That must land on the same numbers as one process running the
    whole batch -- if it does not, every multi-GPU run silently trains a
    different model than the single-GPU run it was validated against.
    """
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    reference = _single_process_grads()
    results = _run_world("ddp")
    assert set(results) == set(range(_WORLD))

    rank0 = results[0]["grads"]
    assert rank0, "rank 0 produced no gradients"
    checked = 0
    for name, expected in reference.items():
        got = rank0.get(name)
        if got is None:  # unused-parameter path; DDP marks these
            continue
        torch.testing.assert_close(
            got,
            expected,
            rtol=1e-4,
            atol=1e-6,
            msg=lambda m, n=name: f"gradient mismatch on {n}\n{m}",
        )
        checked += 1
    assert checked >= 10, f"only {checked} parameters compared; the check is too weak"

    # Every rank must hold identical gradients after the reduction, or the
    # ranks diverge on the very next optimizer step.
    for name, tensor in results[1]["grads"].items():
        if name in rank0:
            torch.testing.assert_close(tensor, rank0[name], rtol=1e-5, atol=1e-7)


def test_distributed_state_reports_real_group_membership():
    """The sharding report must describe the group that actually exists."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    results = _run_world("ddp")
    for rank in range(_WORLD):
        report = results[rank]["report"]
        assert report["world_size"] == _WORLD
        assert report["kind"] == "ddp"
        assert report["batch_parallel"] is True


def test_capture_refuses_a_ddp_wrapped_module():
    """Recording a step whose gradients a collective owns must fail loudly.

    DDP's reducer hooks keep AccumulateGrad references alive across iterations,
    so a captured region replays against another iteration's autograd nodes and
    the gradients quietly stop being correct. PyTorch only warns about this. A
    warning arrives too late in a multi-hour run, so the framework refuses, and
    the message carries the supported arrangement: capture the inner module,
    then wrap the graphed callable.
    """
    from dabsn.runtime.graph import _reject_capture_of_a_replicated_module

    plain = torch.nn.Linear(4, 4)
    _reject_capture_of_a_replicated_module(plain)  # unwrapped: no objection

    class _FakeDDP(torch.nn.Module):
        pass

    # Match by qualified name so the guard needs no live process group to fire.
    _FakeDDP.__module__ = "torch.nn.parallel.distributed"
    _FakeDDP.__qualname__ = "DistributedDataParallel"

    with pytest.raises(RuntimeError) as excinfo:
        _reject_capture_of_a_replicated_module(_FakeDDP())
    message = str(excinfo.value)
    assert "DDP" in message
    assert "make_graphed_train_callable" in message  # names the way out
    assert "DABSN_SCAN_GRAPH=0" in message


# ---------------------------------------------------------------------------
# Tensor parallelism: the model itself split across ranks
#
# Data parallelism replicates the model, so the biggest trainable model is the
# biggest that fits on one device. Above ~10B parameters that stops being true
# and the model has to be split. These tests prove the split is exact -- the
# shards partition the hidden units, they do not approximate them.

_TP_IN, _TP_HIDDEN, _TP_BATCH, _TP_STEPS = 5, 12, 3, 4


def _build_core():
    from dabsn import DABSNCore

    torch.manual_seed(99)
    return DABSNCore(_TP_IN, _TP_HIDDEN)


def _tp_inputs() -> torch.Tensor:
    gen = torch.Generator().manual_seed(5)
    return torch.randn(_TP_BATCH, _TP_STEPS, _TP_IN, generator=gen)


def test_hidden_shards_partition_every_unit_exactly_once():
    """No unit may be dropped or computed twice, at any H and any world size."""
    from dabsn.distributed import hidden_shard

    for hidden in range(1, 40):
        for world in range(1, 9):
            covered = []
            for rank in range(world):
                cut = hidden_shard(hidden, rank, world)
                covered.extend(range(cut.start, cut.stop))
            assert covered == list(range(hidden)), f"H={hidden} world={world} covered {covered}"
            widths = [hidden_shard(hidden, r, world) for r in range(world)]
            sizes = [c.stop - c.start for c in widths]
            # The remainder spreads one unit at a time instead of piling onto
            # one rank, so no rank carries a disproportionate tail.
            assert max(sizes) - min(sizes) <= 1, f"unbalanced shards {sizes}"


def test_single_rank_tensor_parallel_scan_matches_the_core():
    """With world size 1 the sharded path must be the unsharded path.

    If these disagree, every multi-rank number afterwards is measured against
    the wrong baseline, so this is checked before any collective is involved.
    """
    from dabsn.distributed import shard_core_tensor_parallel, tensor_parallel_core_scan

    core = _build_core()
    inputs = _tp_inputs()
    expected = core(inputs)[0]  # trajectory: cat([y, budget])
    shard = shard_core_tensor_parallel(core, 0, 1)
    actual = tensor_parallel_core_scan(shard, inputs)
    assert actual.shape == (_TP_BATCH, _TP_STEPS, 2 * _TP_HIDDEN)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _tp_rank_main(rank: int, world: int, out_path: str) -> int:
    import torch.distributed as dist

    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from dabsn.distributed import shard_core_tensor_parallel, tensor_parallel_core_scan

        core = _build_core()  # identical on every rank
        shard = shard_core_tensor_parallel(core, rank, world)
        for key, value in shard.items():
            if torch.is_tensor(value):
                shard[key] = value.requires_grad_(True)
        out = tensor_parallel_core_scan(shard, _tp_inputs())
        out.sum().backward()
        torch.save(
            {
                "out": out.detach().clone(),
                "slice": (shard["slice"].start, shard["slice"].stop),
                "grad_A": shard["A"].grad.detach().clone(),
            },
            out_path,
        )
        return 0
    finally:
        dist.destroy_process_group()


def test_two_rank_tensor_parallel_reproduces_the_whole_model():
    """Two ranks, each holding half the hidden units, rebuild the exact output.

    This is the claim that makes a model larger than one device possible: the
    concatenated shard outputs equal the unsharded output, and each rank's
    gradient equals the corresponding rows of the unsharded gradient. Gloo runs
    it here, so it is verified without a second GPU.
    """
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    from dabsn.distributed import reassemble_tensor_parallel_trajectory

    core = _build_core()
    inputs = _tp_inputs()
    reference = core(inputs)[0].detach().clone()
    core.zero_grad(set_to_none=True)
    core(inputs)[0].sum().backward()
    ref_grad_A = core.A.weight.grad.detach().clone()

    results = _run_world("tp")
    assert set(results) == set(range(_WORLD))

    pieces = [results[rank]["out"] for rank in range(_WORLD)]
    rebuilt = reassemble_tensor_parallel_trajectory(pieces)
    torch.testing.assert_close(rebuilt, reference, rtol=1e-5, atol=1e-6)

    # Each rank's recurrent-matrix gradient must be the matching row block of
    # the unsharded gradient: sharding places the work, it does not change it.
    for rank in range(_WORLD):
        lo, hi = results[rank]["slice"]
        torch.testing.assert_close(results[rank]["grad_A"], ref_grad_A[lo:hi], rtol=1e-4, atol=1e-6)


def _tp_core_rank_main(rank: int, world: int, out_path: str) -> int:
    import torch.distributed as dist

    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from dabsn.core import TensorParallelDABSNCore

        source = _build_core()  # identical on every rank
        core = TensorParallelDABSNCore(
            source, group=dist.group.WORLD, rank=rank, world_size=world
        )
        trajectory = core.forward_from_state(_tp_inputs())[0]
        trajectory.sum().backward()
        torch.save(
            {
                "out": trajectory.detach().clone(),
                "slice": (core.tensor_slice.start, core.tensor_slice.stop),
                "grad_A": core.A.weight.grad.detach().clone(),
                "grad_Ug": core.Ug.weight.grad.detach().clone(),
                "grad_alpha": core.logit_alpha.grad.detach().clone(),
                "backend": core._last_core_backend,
            },
            out_path,
        )
        return 0
    finally:
        dist.destroy_process_group()


def test_two_rank_tensor_parallel_core_matches_the_unsharded_core():
    """The live sharded core -- not the standalone helper -- must be exact.

    ``tensor_parallel_core_scan`` in ``distributed.py`` has a two-rank test
    above, but nothing outside this file calls it. The class that actually runs
    during training is ``TensorParallelDABSNCore``, and until now no test put
    two ranks through it, so its gradient path was unproven -- the one place
    where being wrong is silent rather than loud.

    Both gradient directions are checked, because they fail differently. The
    recurrent-matrix rows are local: this rank owns the rows that produce its
    own units, and its gradient must equal the matching block of the unsharded
    gradient. The replicated scalars are the opposite: every rank computes a
    partial for the same parameter, so they are only correct once summed across
    ranks. A missing reduction leaves the first check passing and the second
    failing, which is precisely the asymmetry worth pinning.
    """
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    core = _build_core()
    inputs = _tp_inputs()
    reference = core(inputs)[0].detach().clone()
    core.zero_grad(set_to_none=True)
    core(inputs)[0].sum().backward()
    ref_grad_A = core.A.weight.grad.detach().clone()
    ref_grad_Ug = core.Ug.weight.grad.detach().clone()
    ref_grad_alpha = core.logit_alpha.grad.detach().clone()

    results = _run_world("tpcore")
    assert set(results) == set(range(_WORLD))

    for rank in range(_WORLD):
        got = results[rank]
        assert got["backend"] == "tensor_parallel_registered_step"
        # Every rank reassembles the complete public trajectory, so each one
        # must reproduce the unsharded output on its own.
        torch.testing.assert_close(got["out"], reference, rtol=1e-5, atol=1e-6)

        lo, hi = got["slice"]
        torch.testing.assert_close(got["grad_A"], ref_grad_A[lo:hi], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(got["grad_Ug"], ref_grad_Ug[lo:hi], rtol=1e-4, atol=1e-6)
        # Replicated scalar: the per-rank hook all-reduces, so each rank should
        # already hold the whole-model gradient, not its own share of it.
        torch.testing.assert_close(got["grad_alpha"], ref_grad_alpha, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    # Rank entry point. `kind` selects which parallelism this rank exercises:
    # "tp" and "tpcore" split the model across ranks, everything else
    # replicates it.
    _RANK, _KIND, _OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    if _KIND == "tp":
        raise SystemExit(_tp_rank_main(_RANK, _WORLD, _OUT))
    if _KIND == "tpcore":
        raise SystemExit(_tp_core_rank_main(_RANK, _WORLD, _OUT))
    raise SystemExit(_rank_main(_RANK, _WORLD, _KIND, _OUT))
