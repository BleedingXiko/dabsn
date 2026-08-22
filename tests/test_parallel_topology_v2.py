import os
import socket
import subprocess
import sys
import tempfile

import pytest
import torch

from dabsn import (
    BuildContext,
    ComponentSpec,
    DABSNGraph,
    ExpertParallelExpertGroup,
    ParallelAxis,
    ParallelTopology,
    ReLU2MLPExpertGroup,
    component_registry,
    parse_parallel_topology,
)
from dabsn.distributed import (
    cleanup_distributed,
    clip_grad_norm,
    prepare_distributed_module,
    setup_distributed,
)

_WORLD = 2
_EXPERTS = 4
_HIDDEN = 4
_INNER = 6
_DABSN_INPUT = 5
_DABSN_HIDDEN = 7


def _config(*, distributed: bool) -> dict[str, object]:
    config: dict[str, object] = {
        "hidden_dim": _HIDDEN,
        "experts": _EXPERTS,
        "top_k": 2,
        "inner_dim": _INNER,
        "router": "switch",
        "balance_coefficient": 0.01,
        "normalization": "none",
        "routing_granularity": "individual_h",
        "backend": "reference",
        "zero_output": False,
        "residual": False,
    }
    if distributed:
        config["expert_parallel_axis"] = "expert"
    return config


def _binding(*, distributed: bool):
    return component_registry.build(
        ComponentSpec("moe.0", "dabsn:sparse_moe", _config(distributed=distributed)),
        BuildContext(device=torch.device("cpu"), dtype=torch.float32),
    )


def _dabsn_binding(*, distributed: bool):
    config: dict[str, object] = {
        "input_dim": _DABSN_INPUT,
        "hidden_dim": _DABSN_HIDDEN,
        "state_dim": _DABSN_HIDDEN,
        "read_geometry": "seq",
        "residual": False,
    }
    if distributed:
        config["tensor_parallel_axis"] = "tensor"
    return component_registry.build(
        ComponentSpec("dabsn.0", "dabsn:block", config),
        BuildContext(device=torch.device("cpu"), dtype=torch.float32),
    )


def _input() -> torch.Tensor:
    generator = torch.Generator().manual_seed(4102)
    return torch.randn(2, 5, _HIDDEN, generator=generator)


def _dabsn_input() -> torch.Tensor:
    generator = torch.Generator().manual_seed(812)
    return torch.randn(2, 4, _DABSN_INPUT, generator=generator)


def test_parallel_topology_parser_is_explicit_and_architecture_neutral():
    topology = parse_parallel_topology(("data=2", "tensor=3", "my_provider_axis=5"))
    assert topology is not None
    assert topology.world_size == 30
    assert topology.to_dict() == {
        "axes": [
            {"name": "data", "size": 2},
            {"name": "tensor", "size": 3},
            {"name": "my_provider_axis", "size": 5},
        ]
    }
    with pytest.raises(ValueError, match="name=size"):
        parse_parallel_topology(("data:2",))


def _centralized_reference():
    torch.manual_seed(731)
    binding = _binding(distributed=False)
    graph = DABSNGraph([binding])
    value = _input().requires_grad_(True)
    output = graph(value)
    output.square().mean().backward()
    group = binding.module.expert_group
    assert isinstance(group, ReLU2MLPExpertGroup)
    assert value.grad is not None
    assert binding.module.router.proj.weight.grad is not None
    assert group.w1.grad is not None and group.w2.grad is not None
    return {
        "output": output.detach(),
        "input_grad": value.grad.detach(),
        "router_grad": binding.module.router.proj.weight.grad.detach(),
        "w1_grad": group.w1.grad.detach(),
        "w2_grad": group.w2.grad.detach(),
    }


def _rank_main(rank: int, output_path: str) -> int:
    topology = ParallelTopology((ParallelAxis("data", 1), ParallelAxis("expert", _WORLD)))
    state = setup_distributed("ddp", "cpu", backend="gloo", topology=topology)
    try:
        torch.manual_seed(731)
        binding = _binding(distributed=True)
        graph = DABSNGraph([binding])
        model = prepare_distributed_module(graph, state)
        value = _input().requires_grad_(True)
        output = model(value)
        output.square().mean().backward()
        group = binding.module.expert_group
        assert isinstance(group, ExpertParallelExpertGroup)
        local = group.local_group
        assert isinstance(local, ReLU2MLPExpertGroup)
        context = state.parallel_context
        assert context is not None
        consolidator = binding.parallel_state_consolidator
        assert consolidator is not None
        consolidated = consolidator(
            binding.module,
            binding.config,
            binding.module.state_dict(),
            context,
        )
        assert value.grad is not None
        assert binding.module.router.proj.weight.grad is not None
        assert local.w1.grad is not None and local.w2.grad is not None
        torch.save(
            {
                "output": output.detach(),
                "input_grad": value.grad.detach(),
                "router_grad": binding.module.router.proj.weight.grad.detach(),
                "w1_grad": local.w1.grad.detach(),
                "w2_grad": local.w2.grad.detach(),
                "report": state.report(),
                "consolidated_shapes": {
                    name: tuple(tensor.shape) for name, tensor in consolidated.items()
                },
            },
            output_path,
        )
        return 0
    finally:
        cleanup_distributed(state)


def _run_world() -> dict[int, dict[str, object]]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    root = tempfile.mkdtemp(prefix="dabsn-provider-parallel-")
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processes = []
    paths = []
    for rank in range(_WORLD):
        path = os.path.join(root, f"rank-{rank}.pt")
        paths.append(path)
        environment = dict(
            os.environ,
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(_WORLD),
            LOCAL_RANK=str(rank),
            PYTHONPATH=os.path.join(project, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
            OMP_NUM_THREADS="1",
        )
        processes.append(
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), str(rank), path],
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
        if process.returncode:
            failures.append(f"rank {rank} exited {process.returncode}:\n{output[-4000:]}")
    if failures:
        raise AssertionError("provider-parallel workers failed:\n" + "\n".join(failures))
    return {rank: torch.load(path, weights_only=True) for rank, path in enumerate(paths)}


def _dabsn_reference():
    torch.manual_seed(993)
    binding = _dabsn_binding(distributed=False)
    value = _dabsn_input().requires_grad_(True)
    output = binding.module(value)
    output.square().mean().backward()
    unclipped_norm = torch.nn.utils.clip_grad_norm_(binding.module.parameters(), 0.05)
    core = binding.module.core
    return {
        "output": output.detach(),
        "input_grad": value.grad.detach(),
        "A_grad": core.A.weight.grad.detach(),
        "W_grad": core.W.weight.grad.detach(),
        "alpha_grad": core.logit_alpha.grad.detach(),
        "unclipped_norm": unclipped_norm.detach(),
    }


def _dabsn_rank_main(rank: int, output_path: str) -> int:
    from dabsn.core import TensorParallelDABSNCore

    topology = ParallelTopology((ParallelAxis("data", 1), ParallelAxis("tensor", _WORLD)))
    state = setup_distributed("ddp", "cpu", backend="gloo", topology=topology)
    try:
        torch.manual_seed(993)
        binding = _dabsn_binding(distributed=True)
        graph = DABSNGraph([binding])
        model = prepare_distributed_module(graph, state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        live_parameter_ids = {id(parameter) for parameter in model.parameters()}
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        assert optimizer_parameter_ids == live_parameter_ids
        value = _dabsn_input().requires_grad_(True)
        output = model(value)
        output.square().mean().backward()
        unclipped_norm = clip_grad_norm(model, 0.05)
        core = binding.module.core
        assert isinstance(core, TensorParallelDABSNCore)
        context = state.parallel_context
        assert context is not None
        consolidated = binding.parallel_state_consolidator(
            binding.module,
            binding.config,
            binding.module.state_dict(),
            context,
        )
        initial = core.initial_state(value.shape[0], device=value.device, dtype=value.dtype)
        carried, final = core.forward_from_state(
            value,
            initial_state=initial,
            return_writes=True,
            return_cocktail=True,
            return_final_state=True,
        )
        torch.save(
            {
                "output": output.detach(),
                "input_grad": value.grad.detach(),
                "A_grad": core.A.weight.grad.detach(),
                "W_grad": core.W.weight.grad.detach(),
                "alpha_grad": core.logit_alpha.grad.detach(),
                "unclipped_norm": unclipped_norm.detach(),
                "slice": (core.tensor_slice.start, core.tensor_slice.stop),
                "carried_shapes": [tuple(tensor.shape) for tensor in carried],
                "final_shapes": [tuple(tensor.shape) for tensor in final],
                "consolidated_shapes": {
                    name: tuple(tensor.shape) for name, tensor in consolidated.items()
                },
            },
            output_path,
        )
        return 0
    finally:
        cleanup_distributed(state)


def _run_dabsn_world() -> dict[int, dict[str, object]]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    root = tempfile.mkdtemp(prefix="dabsn-core-provider-parallel-")
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processes = []
    paths = []
    for rank in range(_WORLD):
        path = os.path.join(root, f"rank-{rank}.pt")
        paths.append(path)
        environment = dict(
            os.environ,
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(_WORLD),
            LOCAL_RANK=str(rank),
            PYTHONPATH=os.path.join(project, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
            OMP_NUM_THREADS="1",
        )
        processes.append(
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "dabsn", str(rank), path],
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
        if process.returncode:
            failures.append(f"rank {rank} exited {process.returncode}:\n{output[-4000:]}")
    if failures:
        raise AssertionError("DABSN provider-parallel workers failed:\n" + "\n".join(failures))
    return {rank: torch.load(path, weights_only=True) for rank, path in enumerate(paths)}


def test_provider_owned_expert_distribution_matches_one_unsplit_model():
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    expected = _centralized_reference()
    results = _run_world()
    local_count = _EXPERTS // _WORLD
    for rank in range(_WORLD):
        result = results[rank]
        torch.testing.assert_close(result["output"], expected["output"])
        torch.testing.assert_close(result["input_grad"], expected["input_grad"])
        torch.testing.assert_close(result["router_grad"], expected["router_grad"])
        start = rank * local_count
        stop = start + local_count
        torch.testing.assert_close(result["w1_grad"], expected["w1_grad"][start:stop])
        torch.testing.assert_close(result["w2_grad"], expected["w2_grad"][start:stop])
        assert result["report"]["coordinate"] == {"data": 0, "expert": rank}
        assert "expert_group.local_group.w1" not in result["consolidated_shapes"]
        # w1 projects hidden->inner and w2 projects inner->hidden, so the
        # consolidated stacks are [E, hidden, inner] and [E, inner, hidden]
        # respectively -- the orientation the expert group actually stores
        # (src/dabsn/moe.py) and matmuls against, not the transpose of it.
        assert result["consolidated_shapes"]["expert_group.w1"] == (
            _EXPERTS,
            _HIDDEN,
            _INNER,
        )
        assert result["consolidated_shapes"]["expert_group.w2"] == (
            _EXPERTS,
            _INNER,
            _HIDDEN,
        )


def test_provider_owned_dabsn_distribution_matches_one_unsplit_block():
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    expected = _dabsn_reference()
    results = _run_dabsn_world()
    for rank in range(_WORLD):
        result = results[rank]
        start, stop = result["slice"]
        torch.testing.assert_close(result["output"], expected["output"], rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(
            result["input_grad"], expected["input_grad"], rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(
            result["unclipped_norm"], expected["unclipped_norm"], rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(
            result["W_grad"], expected["W_grad"][start:stop], rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(
            result["alpha_grad"], expected["alpha_grad"], rtol=1e-4, atol=1e-6
        )
        assert result["carried_shapes"] == [
            (2, 4, 2 * _DABSN_HIDDEN),
            (2, 4, _DABSN_HIDDEN),
            (2, 4, _DABSN_HIDDEN),
            (2, 4, _DABSN_HIDDEN),
            (2, 4, _DABSN_HIDDEN),
            (2, 4, _DABSN_HIDDEN),
            (2, 4, _DABSN_HIDDEN),
        ]
        assert result["final_shapes"] == [(2, _DABSN_HIDDEN)] * 3
        assert result["consolidated_shapes"]["core.A.weight"] == (
            _DABSN_HIDDEN,
            _DABSN_HIDDEN,
        )
        assert result["consolidated_shapes"]["core.W.weight"] == (
            _DABSN_HIDDEN,
            _DABSN_INPUT,
        )


if __name__ == "__main__":
    if sys.argv[1] == "dabsn":
        raise SystemExit(_dabsn_rank_main(int(sys.argv[2]), sys.argv[3]))
    raise SystemExit(_rank_main(int(sys.argv[1]), sys.argv[2]))
