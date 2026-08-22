import pytest
import torch
import torch.nn as nn

from dabsn import (
    AxisContract,
    ComponentCapabilities,
    ComponentContract,
    DABSNGraph,
    DABSNLayerSpec,
    DABSNSequenceLM,
    ParallelAxis,
    ParallelDirective,
    ParallelExecutionContext,
    ParallelExecutionReceipt,
    ParallelPlan,
    ParallelTopology,
    ValueContract,
    bind_module,
)
from dabsn.config import DABSNPretrainConfig
from dabsn.distributed import (
    DABSNSequenceModule,
    DistributedState,
    _execute_component_parallel_plans,
    no_sync_context,
    state_dict_digest,
    wrap_distributed,
)


def test_distributed_state_reports_fsdp2_sharding_truthfully():
    state = DistributedState(kind="fsdp2", world_size=2)
    assert state.parameter_sharded
    assert state.gradient_sharded
    assert state.optimizer_sharded
    assert state.batch_parallel
    assert state.report()["kind"] == "fsdp2"


def test_pretrain_config_accepts_explicit_fsdp2_backend(tmp_path):
    corpus = tmp_path / "tokens.bin"
    corpus.write_bytes(bytes(range(16)))
    config = DABSNPretrainConfig(
        corpus_bin=str(corpus),
        corpus_dtype="uint8",
        vocab=16,
        hidden_dim=4,
        depth=1,
        train_context=4,
        steps=1,
        batch_size=1,
        eval_batch_size=1,
        distributed="fsdp2",
    )
    assert config.distributed == "fsdp2"


def test_fsdp2_wrap_is_bottom_up_then_root(monkeypatch):
    model = DABSNSequenceLM(
        vocab=19,
        hidden_dim=6,
        depth=2,
        layers=[DABSNLayerSpec(6, 7, "seq"), DABSNLayerSpec(6, 7, "seq")],
        mlp_ratio=2.0,
        mlp_middle_depth=1,
        mlp_depth_index=0,
    )
    sequence = DABSNSequenceModule(model)
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        return module

    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", fake_fully_shard)
    state = DistributedState(kind="fsdp2", world_size=2, device=torch.device("cpu"))
    assert wrap_distributed(sequence, state, precision="fp32") is sequence
    assert calls[-1] is sequence
    assert calls[:-1] == [*model.backbone.blocks, *model.backbone.middle_mlps]


def test_fsdp2_honors_declared_component_boundaries(monkeypatch):
    contract = ComponentContract(
        ValueContract.tensor(
            AxisContract("batch", "B", dynamic=True),
            AxisContract("experience", "T", dynamic=True),
            AxisContract("world", 6),
        ),
        ValueContract.tensor(
            AxisContract("batch", "B", dynamic=True),
            AxisContract("experience", "T", dynamic=True),
            AxisContract("world", 6),
        ),
    )
    first = bind_module(
        "dabsn.0",
        nn.Linear(6, 6),
        contract,
        capabilities=ComponentCapabilities(
            distributed=True,
            world_builder=True,
            dabsn_memory_owner=True,
            parallel_plan=ParallelPlan(fsdp_boundary=False),
        ),
    )
    second = bind_module(
        "user.0",
        nn.Linear(6, 6),
        contract,
        capabilities=ComponentCapabilities(
            distributed=True,
            parallel_plan=ParallelPlan(fsdp_boundary=True),
        ),
    )
    model = DABSNSequenceLM.from_graph(
        DABSNGraph([first, second], require_world_builder=True), vocab=19
    )
    sequence = DABSNSequenceModule(model)
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        return module

    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", fake_fully_shard)
    state = DistributedState(kind="fsdp2", world_size=2, device=torch.device("cpu"))
    wrap_distributed(sequence, state, precision="fp32")
    assert calls == [second.module, sequence]


def test_parallel_plan_rejects_undeclared_distributed_or_checkpoint_support():
    with pytest.raises(ValueError, match="requires distributed=True"):
        ComponentCapabilities(parallel_plan=ParallelPlan())
    with pytest.raises(ValueError, match="requires activation_checkpoint=True"):
        ComponentCapabilities(
            distributed=True,
            parallel_plan=ParallelPlan(activation_checkpoint_boundary=True),
        )


@pytest.mark.parametrize("kind", ["tensor", "expert", "replicate", "communication"])
def test_specialized_parallel_plan_directives_fail_before_distributed_launch(kind):
    from dabsn.graph import UnsupportedExecutionModeError

    contract = ComponentContract(
        ValueContract.tensor(
            AxisContract("batch", "B", dynamic=True),
            AxisContract("experience", "T", dynamic=True),
            AxisContract("world", 6),
        ),
        ValueContract.tensor(
            AxisContract("batch", "B", dynamic=True),
            AxisContract("experience", "T", dynamic=True),
            AxisContract("world", 6),
        ),
    )
    capabilities = ComponentCapabilities(
        distributed=True,
        world_builder=True,
        dabsn_memory_owner=True,
        parallel_plan=ParallelPlan(directives=(ParallelDirective(f"{kind}.0", kind, ("data",)),)),
    )
    binding = bind_module("external.0", nn.Linear(6, 6), contract, capabilities=capabilities)
    model = DABSNSequenceLM.from_graph(DABSNGraph([binding], require_world_builder=True), vocab=19)
    sequence = DABSNSequenceModule(model)
    state = DistributedState(kind="ddp", world_size=2, device=torch.device("cpu"))
    with pytest.raises(UnsupportedExecutionModeError, match="installed no parallel executor"):
        wrap_distributed(sequence, state, precision="fp32")


def test_named_parallel_topology_has_deterministic_coordinates():
    topology = ParallelTopology(
        (ParallelAxis("data", 2), ParallelAxis("tensor", 2), ParallelAxis("expert", 2))
    )
    assert topology.world_size == 8
    assert dict(topology.coordinate(0)) == {"data": 0, "tensor": 0, "expert": 0}
    assert dict(topology.coordinate(5)) == {"data": 1, "tensor": 0, "expert": 1}
    assert topology.axis_size("expert") == 2
    assert topology.fingerprint() == topology.fingerprint()


def test_provider_parallel_executor_must_consume_every_directive_exactly_once():
    calls = []

    def executor(module, config, plan, context):
        calls.append((module, dict(config), context.rank))
        return ParallelExecutionReceipt(
            (plan.directives[0].directive_id,),
            parameter_sharding={name: () for name, _parameter in module.named_parameters()},
        )

    directive = ParallelDirective("tensor.world", "tensor", ("tensor",), {"dim": -1})
    plan = ParallelPlan(directives=(directive,))
    contract = ComponentContract(
        ValueContract.tensor(AxisContract("batch", "B"), AxisContract("world", 6)),
        ValueContract.tensor(AxisContract("batch", "B"), AxisContract("world", 6)),
    )
    binding = bind_module(
        "external.0",
        nn.Linear(6, 6),
        contract,
        capabilities=ComponentCapabilities(
            distributed=True,
            world_builder=True,
            dabsn_memory_owner=True,
            parallel_plan=plan,
        ),
        parallel_executor=executor,
    )
    model = DABSNSequenceLM.from_graph(DABSNGraph([binding], require_world_builder=True), vocab=19)
    context = ParallelExecutionContext(
        topology=ParallelTopology((ParallelAxis("tensor", 1),)),
        rank=0,
        device=torch.device("cpu"),
        coordinate={"tensor": 0},
        groups={"tensor": None},
        group_ranks={"tensor": (0,)},
    )
    _execute_component_parallel_plans(DABSNSequenceModule(model), context)
    _execute_component_parallel_plans(DABSNSequenceModule(model), context)
    assert len(calls) == 1


def test_single_worker_data_axis_does_not_install_a_useless_data_wrapper(monkeypatch):
    directive = ParallelDirective("tensor.world", "tensor", ("tensor",))
    plan = ParallelPlan(directives=(directive,))
    contract = ComponentContract(
        ValueContract.tensor(AxisContract("batch", "B"), AxisContract("world", 6)),
        ValueContract.tensor(AxisContract("batch", "B"), AxisContract("world", 6)),
    )
    binding = bind_module(
        "external.0",
        nn.Linear(6, 6),
        contract,
        capabilities=ComponentCapabilities(
            distributed=True,
            parallel_plan=plan,
        ),
        parallel_executor=lambda module, config, requested, context: ParallelExecutionReceipt(
            (directive.directive_id,),
            parameter_sharding={name: () for name, _ in module.named_parameters()},
        ),
    )
    graph = DABSNGraph([binding])
    sequence = DABSNSequenceModule(graph)
    topology = ParallelTopology((ParallelAxis("data", 1), ParallelAxis("tensor", 2)))
    context = ParallelExecutionContext(
        topology=topology,
        rank=0,
        device=torch.device("cpu"),
        coordinate={"data": 0, "tensor": 0},
        groups={"data": None, "tensor": None},
        group_ranks={"data": (0,), "tensor": (0, 1)},
    )
    state = DistributedState(
        kind="ddp",
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        topology=topology,
        parallel_context=context,
    )

    def forbidden_wrapper(*args, **kwargs):
        raise AssertionError("one-worker data axis must not install DDP")

    monkeypatch.setattr("torch.nn.parallel.DistributedDataParallel", forbidden_wrapper)
    assert wrap_distributed(sequence, state) is sequence
    assert binding.parallelized_topology == topology.fingerprint()


def test_provider_owned_topology_does_not_require_a_dummy_data_axis(monkeypatch):
    directive = ParallelDirective("tensor.world", "tensor", ("tensor",))
    plan = ParallelPlan(directives=(directive,))
    contract = ComponentContract(
        ValueContract.tensor(AxisContract("batch", "B"), AxisContract("world", 6)),
        ValueContract.tensor(AxisContract("batch", "B"), AxisContract("world", 6)),
    )
    binding = bind_module(
        "external.0",
        nn.Linear(6, 6),
        contract,
        capabilities=ComponentCapabilities(distributed=True, parallel_plan=plan),
        parallel_executor=lambda module, config, requested, context: ParallelExecutionReceipt(
            (directive.directive_id,),
            parameter_sharding={name: () for name, _ in module.named_parameters()},
        ),
    )
    sequence = DABSNSequenceModule(DABSNGraph([binding]))
    topology = ParallelTopology((ParallelAxis("tensor", 2),))
    context = ParallelExecutionContext(
        topology=topology,
        rank=0,
        device=torch.device("cpu"),
        coordinate={"tensor": 0},
        groups={"tensor": None},
        group_ranks={"tensor": (0, 1)},
    )
    state = DistributedState(
        kind="ddp",
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        topology=topology,
        parallel_context=context,
    )
    monkeypatch.setattr(
        "torch.nn.parallel.DistributedDataParallel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DDP must not install")),
    )
    assert wrap_distributed(sequence, state) is sequence


def test_topology_without_data_or_provider_ownership_is_rejected():
    contract = ComponentContract(
        ValueContract.tensor(AxisContract("batch", "B"), AxisContract("world", 6)),
        ValueContract.tensor(AxisContract("batch", "B"), AxisContract("world", 6)),
    )
    sequence = DABSNSequenceModule(
        DABSNGraph([bind_module("ordinary", nn.Linear(6, 6), contract)])
    )
    topology = ParallelTopology((ParallelAxis("unowned", 2),))
    context = ParallelExecutionContext(
        topology=topology,
        rank=0,
        device=torch.device("cpu"),
        coordinate={"unowned": 0},
        groups={"unowned": None},
        group_ranks={"unowned": (0, 1)},
    )
    state = DistributedState(
        kind="ddp",
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        topology=topology,
        parallel_context=context,
    )
    with pytest.raises(ValueError, match="provider parallel plan"):
        wrap_distributed(sequence, state)


def test_provider_parallel_executor_partial_receipt_is_rejected():
    from dabsn.graph import UnsupportedExecutionModeError

    directive = ParallelDirective("expert.bank", "expert", ("expert",))
    plan = ParallelPlan(directives=(directive,))
    contract = ComponentContract(
        ValueContract.tensor(AxisContract("batch", "B"), AxisContract("world", 6)),
        ValueContract.tensor(AxisContract("batch", "B"), AxisContract("world", 6)),
    )
    binding = bind_module(
        "external.0",
        nn.Linear(6, 6),
        contract,
        capabilities=ComponentCapabilities(
            distributed=True,
            world_builder=True,
            dabsn_memory_owner=True,
            parallel_plan=plan,
        ),
        parallel_executor=lambda module, config, requested, context: ParallelExecutionReceipt(()),
    )
    model = DABSNSequenceLM.from_graph(DABSNGraph([binding], require_world_builder=True), vocab=19)
    context = ParallelExecutionContext(
        topology=ParallelTopology((ParallelAxis("expert", 1),)),
        rank=0,
        device=torch.device("cpu"),
        coordinate={"expert": 0},
        groups={"expert": None},
        group_ranks={"expert": (0,)},
    )
    with pytest.raises(UnsupportedExecutionModeError, match="unconsumed"):
        _execute_component_parallel_plans(DABSNSequenceModule(model), context)


def test_fsdp2_accumulation_uses_explicit_gradient_sync_control():
    class Fake:
        def __init__(self):
            self.calls = []

        def set_requires_gradient_sync(self, value):
            self.calls.append(value)

    model = Fake()
    with no_sync_context(model, synchronize=False):
        assert model.calls == [False]
    assert model.calls == [False, True]


def test_fsdp2_portable_export_uses_dcp_full_cpu_state(monkeypatch):
    from dabsn.distributed import full_model_state_dict

    calls = []

    def fake_get_model_state_dict(model, *, options):
        calls.append(options)
        return {"weight": torch.ones(2, 2), "metadata": "ignored"}

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        fake_get_model_state_dict,
    )
    result = full_model_state_dict(
        torch.nn.Linear(2, 2), DistributedState(kind="fsdp2", world_size=2)
    )
    assert list(result) == ["weight"]
    assert calls[0].full_state_dict is True
    assert calls[0].cpu_offload is True
    assert calls[0].strict is True


def test_training_state_digest_is_order_independent_and_tensor_sensitive():
    first = {"b": [1, torch.tensor([2.0])], "a": {"x": True}}
    reordered = {"a": {"x": True}, "b": [1, torch.tensor([2.0])]}
    changed = {"a": {"x": True}, "b": [1, torch.tensor([3.0])]}
    assert state_dict_digest(first) == state_dict_digest(reordered)
    assert state_dict_digest(first) != state_dict_digest(changed)
