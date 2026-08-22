import os
import subprocess
import sys
import tempfile

import pytest
import torch

from dabsn import ExpertParallelExpertGroup, ReLU2MLPExpertGroup

_EP_WORLD = 2
_EP_LOCAL_EXPERTS = 2
_EP_HIDDEN = 4
_EP_INNER = 6


def _expert_weights(global_expert: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(900 + global_expert)
    return (
        torch.randn(_EP_HIDDEN, _EP_INNER, generator=generator) * 0.07,
        torch.randn(_EP_INNER, _EP_HIDDEN, generator=generator) * 0.07,
    )


def _configured_group(experts: int, *, first_global_expert: int = 0):
    group = ReLU2MLPExpertGroup(
        experts,
        _EP_HIDDEN,
        _EP_INNER,
        backend="reference",
    )
    with torch.no_grad():
        for local_expert in range(experts):
            w1, w2 = _expert_weights(first_global_expert + local_expert)
            group.w1[local_expert].copy_(w1)
            group.w2[local_expert].copy_(w2)
    return group


def _rank_inputs(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1200 + rank)
    inputs = torch.randn(7, _EP_HIDDEN, generator=generator)
    if rank == 0:
        expert_ids = torch.tensor([0, 2, 1, 3, 2, 0, 3])
    else:
        expert_ids = torch.tensor([3, 1, 2, 0, 1, 3, 0])
    return inputs, expert_ids


def _centralized_reference():
    group = _configured_group(_EP_WORLD * _EP_LOCAL_EXPERTS)
    outputs = []
    input_grads = []
    loss = torch.zeros(())
    tracked_inputs = []
    for rank in range(_EP_WORLD):
        inputs, expert_ids = _rank_inputs(rank)
        inputs = inputs.requires_grad_(True)
        tracked_inputs.append(inputs)
        output = group.forward_dispatched(inputs, expert_ids)
        outputs.append(output.detach().clone())
        loss = loss + output.square().sum()
    loss.backward()
    for inputs in tracked_inputs:
        assert inputs.grad is not None
        input_grads.append(inputs.grad.detach().clone())
    assert group.w1.grad is not None and group.w2.grad is not None
    return outputs, input_grads, group.w1.grad.detach(), group.w2.grad.detach()


def test_single_rank_expert_parallel_wrapper_is_exact_and_differentiable():
    torch.manual_seed(93)
    local = ReLU2MLPExpertGroup(3, 4, 7, backend="reference")
    wrapped = ExpertParallelExpertGroup(local, world_size=1, rank=0)
    inputs = torch.randn(11, 4, requires_grad=True)
    expert_ids = torch.tensor([0, 2, 1, 0, 2, 2, 1, 0, 1, 1, 2])
    counts = torch.bincount(expert_ids, minlength=3)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    expected = local.forward_dispatched(inputs, expert_ids, offsets)
    actual = wrapped.forward_dispatched(inputs, expert_ids, offsets)

    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert len(wrapped) == 3


def test_expert_parallel_wrapper_refuses_uninitialized_multirank_use():
    local = ReLU2MLPExpertGroup(2, 4, 8, backend="reference")
    try:
        ExpertParallelExpertGroup(local)
    except RuntimeError as exc:
        assert "initialized process group" in str(exc)
    else:
        raise AssertionError("uninitialized expert parallelism did not fail")


def test_single_rank_expert_parallel_sorts_for_grouped_backend():
    torch.manual_seed(94)
    local = ReLU2MLPExpertGroup(2, 16, 8, backend="grouped")
    wrapped = ExpertParallelExpertGroup(local, world_size=1, rank=0)
    inputs = torch.randn(9, 16)
    expert_ids = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1])
    actual = wrapped.forward_dispatched(inputs, expert_ids)
    expected = torch.empty_like(inputs)
    for expert in range(2):
        positions = torch.nonzero(expert_ids == expert).flatten()
        expected.index_copy_(
            0,
            positions,
            local.forward_expert(expert, inputs.index_select(0, positions)),
        )
    torch.testing.assert_close(actual, expected)


def _expert_parallel_rank_main(rank: int, world: int, output_path: str) -> int:
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        local = _configured_group(
            _EP_LOCAL_EXPERTS,
            first_global_expert=rank * _EP_LOCAL_EXPERTS,
        )
        wrapped = ExpertParallelExpertGroup(local)
        inputs, expert_ids = _rank_inputs(rank)
        inputs = inputs.requires_grad_(True)
        output = wrapped.forward_dispatched(inputs, expert_ids)
        output.square().sum().backward()
        assert inputs.grad is not None
        assert local.w1.grad is not None and local.w2.grad is not None
        torch.save(
            {
                "output": output.detach(),
                "input_grad": inputs.grad.detach(),
                "w1_grad": local.w1.grad.detach(),
                "w2_grad": local.w2.grad.detach(),
            },
            output_path,
        )
        return 0
    finally:
        dist.destroy_process_group()


def _run_expert_parallel_world() -> dict[int, dict[str, torch.Tensor]]:
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    temp_root = tempfile.mkdtemp(prefix="dabsn-ep-")
    processes = []
    paths = []
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rank in range(_EP_WORLD):
        output_path = os.path.join(temp_root, f"rank-{rank}.pt")
        paths.append(output_path)
        environment = dict(
            os.environ,
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(_EP_WORLD),
            LOCAL_RANK=str(rank),
            PYTHONPATH=os.path.join(project_root, "src")
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
            OMP_NUM_THREADS="1",
        )
        processes.append(
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), str(rank), output_path],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    failures = []
    for rank, process in enumerate(processes):
        try:
            output = process.communicate(timeout=180)[0]
        except subprocess.TimeoutExpired:
            process.kill()
            failures.append(f"rank {rank} deadlocked")
            continue
        if process.returncode != 0:
            failures.append(f"rank {rank} exited {process.returncode}:\n{output[-3000:]}")
    if failures:
        raise AssertionError("expert-parallel ranks failed:\n" + "\n".join(failures))
    return {rank: torch.load(path, weights_only=True) for rank, path in enumerate(paths)}


def test_two_rank_expert_parallel_matches_centralized_outputs_and_gradients():
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    expected_outputs, expected_input_grads, expected_w1, expected_w2 = _centralized_reference()
    results = _run_expert_parallel_world()
    for rank in range(_EP_WORLD):
        torch.testing.assert_close(results[rank]["output"], expected_outputs[rank])
        torch.testing.assert_close(results[rank]["input_grad"], expected_input_grads[rank])
        start = rank * _EP_LOCAL_EXPERTS
        stop = start + _EP_LOCAL_EXPERTS
        torch.testing.assert_close(results[rank]["w1_grad"], expected_w1[start:stop])
        torch.testing.assert_close(results[rank]["w2_grad"], expected_w2[start:stop])


if __name__ == "__main__":
    raise SystemExit(_expert_parallel_rank_main(int(sys.argv[1]), _EP_WORLD, sys.argv[2]))
