import pytest
import torch
import torch.nn as nn

from dabsn import (
    AxisContract,
    ComponentCapabilities,
    ComponentContract,
    ComponentOutput,
    DABSNGraph,
    DABSNSequenceLM,
    DistributedState,
    ParallelAxis,
    ParallelExecutionContext,
    ParallelExecutionReceipt,
    ParallelTopology,
    ResultDeclaration,
    ValueContract,
    bind_module,
)
from dabsn.distributed import DABSNSequenceModule, clip_grad_norm, reduce_declared_results
from dabsn.runtime import ManualGradientAccumulator
from dabsn.runtime.api import apply_optimizer_step, train_step, verify_gradients


def _world(width):
    return ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("experience", "T", dynamic=True),
        AxisContract("world", width),
    )


class TermsComponent(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.lifecycle_calls = 0

    def forward(self, value):
        return self.proj(value)

    def forward_with_terms(self, value):
        output = self.proj(value)
        return ComponentOutput(output, (self.proj.weight.square().mean(),))

    def post_optimizer_step(self):
        self.lifecycle_calls += 1


def _model():
    width = 6
    contract = ComponentContract(_world(width), _world(width))
    first_module = TermsComponent(width)
    first = bind_module(
        "dabsn.0",
        first_module,
        contract,
        capabilities=ComponentCapabilities(world_builder=True, dabsn_memory_owner=True),
        loss_terms=(ResultDeclaration("fixture_aux", "mean"),),
    )
    graph = DABSNGraph([first], require_world_builder=True)
    return DABSNSequenceLM.from_graph(graph, vocab=19), first_module


def test_all_public_model_wrappers_return_fixed_component_output():
    model, _ = _model()
    result = model.forward_with_terms(torch.randint(0, 19, (2, 4)))
    assert isinstance(result, ComponentOutput)
    assert result.value.shape == (2, 4, 19)
    assert len(result.loss_terms) == 1
    assert result.reports == ()
    assert result.next_state == ()


def test_generic_trainer_adds_auxiliary_loss_and_runs_lifecycle_after_step():
    torch.manual_seed(9)
    model, component = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.randint(0, 19, (2, 4))
    targets = torch.randn(2, 4, 19)
    before = component.proj.weight.detach().clone()
    observed = train_step(
        model,
        inputs,
        targets,
        optimizer,
        loss_fn=lambda output, target: (output - target).square().mean(),
        clip_grad_norm=1.0,
    )
    assert observed > 0
    assert not torch.equal(component.proj.weight, before)
    assert component.lifecycle_calls == 1


def test_gradient_preflight_is_component_native_for_raw_graphs():
    model, _ = _model()
    graph = model.backbone.graph
    rows = verify_gradients(graph, torch.randn(2, 4, 6))
    assert rows == [
        {
            "kind": "component",
            "component": "dabsn.0",
            "trainable_parameters": 42,
            "parameters_with_gradients": 2,
            "gradient_norm": rows[0]["gradient_norm"],
            "finite": True,
            "ok": True,
        }
    ]
    assert rows[0]["gradient_norm"] > 0.0


def test_gradient_preflight_rejects_a_disconnected_trainable_component():
    class DetachedComponent(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(6))

        def forward(self, value):
            return value + self.weight.detach()

    contract = ComponentContract(_world(6), _world(6))
    graph = DABSNGraph([bind_module("detached", DetachedComponent(), contract)])
    with pytest.raises(RuntimeError, match="detached"):
        verify_gradients(graph, torch.randn(2, 4, 6), compile_forward=False)


def test_task_neutral_runtime_preserves_structured_values_for_user_loss():
    class StructuredComponent(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 2)

        def forward(self, value):
            return {"prediction": self.projection(value["observations"])}

    structured = ValueContract(
        (_world(3).leaves[0],),
        tree={"observations": "tensor"},
    )
    predicted = ValueContract(
        (
            ValueContract.tensor(
                AxisContract("batch", "B", dynamic=True),
                AxisContract("agent", "A", dynamic=True),
                AxisContract("action", 2),
            ).leaves[0],
        ),
        tree={"prediction": "tensor"},
    )
    graph = DABSNGraph(
        [
            bind_module(
                "structured.policy",
                StructuredComponent(),
                ComponentContract(structured, predicted),
            )
        ]
    )
    optimizer = torch.optim.SGD(graph.parameters(), lr=0.05)
    inputs = {"observations": torch.randn(2, 4, 3)}
    targets = {"actions": torch.randn(2, 4, 2)}

    loss = train_step(
        graph,
        inputs,
        targets,
        optimizer,
        loss_fn=lambda output, target: (
            output["prediction"] - target["actions"]
        ).square().mean(),
    )
    assert loss > 0.0
    rows = verify_gradients(graph, inputs, compile_forward=False)
    assert rows[0]["component"] == "structured.policy"
    assert rows[0]["ok"] is True


def test_graph_activation_checkpoint_preserves_terms_and_parameter_gradients():
    torch.manual_seed(712)
    width = 6
    contract = ComponentContract(_world(width), _world(width))
    first = TermsComponent(width)
    second = TermsComponent(width)
    graph = DABSNGraph(
        [
            bind_module(
                "checkpointed.first",
                first,
                contract,
                capabilities=ComponentCapabilities(activation_checkpoint=True),
                loss_terms=(ResultDeclaration("fixture_aux_first", "mean"),),
            ),
            bind_module(
                "checkpointed.second",
                second,
                contract,
                capabilities=ComponentCapabilities(activation_checkpoint=True),
                loss_terms=(ResultDeclaration("fixture_aux_second", "mean"),),
            ),
        ]
    )
    value = torch.randn(2, 4, width)

    plain = graph.forward_with_terms(value)
    plain_loss = plain.value.square().mean() + sum(plain.loss_terms)
    plain_loss.backward()
    expected = {
        name: parameter.grad.detach().clone()
        for name, parameter in graph.named_parameters()
    }

    graph.zero_grad(set_to_none=True)
    graph.set_activation_checkpointing(True)
    recomputed = graph.forward_with_terms(value)
    recomputed_loss = recomputed.value.square().mean() + sum(recomputed.loss_terms)
    recomputed_loss.backward()
    torch.testing.assert_close(recomputed.value, plain.value, atol=0, rtol=0)
    assert len(recomputed.loss_terms) == len(plain.loss_terms) == 2
    for actual, expected_term in zip(recomputed.loss_terms, plain.loss_terms):
        torch.testing.assert_close(actual, expected_term, atol=0, rtol=0)
    for name, parameter in graph.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], atol=0, rtol=0)


def test_graph_activation_checkpoint_refuses_an_unsupported_component():
    contract = ComponentContract(_world(6), _world(6))
    graph = DABSNGraph([bind_module("unsupported", nn.Linear(6, 6), contract)])
    with pytest.raises(RuntimeError, match="unsupported"):
        graph.set_activation_checkpointing(True)


def test_legacy_model_training_output_has_empty_static_terms():
    model = DABSNSequenceLM(vocab=19, hidden_dim=6, depth=1)
    result = model.forward_with_terms(torch.randint(0, 19, (2, 4)))
    assert result.value.shape == (2, 4, 19)
    assert result.loss_terms == ()


def test_distributed_sequence_wrapper_returns_terms_through_normal_forward():
    model, component = _model()
    wrapped = DABSNSequenceModule(model)
    result = wrapped(torch.randint(0, 19, (2, 4)), with_terms=True)
    assert isinstance(result, ComponentOutput)
    assert len(result.loss_terms) == 1
    wrapped.post_optimizer_step(step_applied=False)
    assert component.lifecycle_calls == 0
    wrapped.post_optimizer_step(step_applied=True)
    assert component.lifecycle_calls == 1


def test_declared_distributed_report_reductions_are_owned_and_exact(monkeypatch):
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    calls = []

    def all_reduce(value, *, op, group=None):
        calls.append(op)
        value.mul_(2)

    monkeypatch.setattr("torch.distributed.all_reduce", all_reduce)
    values = tuple(torch.tensor(float(index + 1)) for index in range(4))
    declarations = (
        ResultDeclaration("mean", "mean", "framework"),
        ResultDeclaration("sum", "sum", "framework"),
        ResultDeclaration("none", "none", "framework"),
        ResultDeclaration("component", "sum", "component"),
    )
    reduced = reduce_declared_results(values, declarations)
    torch.testing.assert_close(reduced[0], values[0])
    torch.testing.assert_close(reduced[1], values[1] * 2)
    assert reduced[2] is values[2]
    assert reduced[3] is values[3]
    assert calls == [torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp.SUM]


def test_provider_shard_clipping_counts_unique_parameters_once(monkeypatch):
    contract = ComponentContract(_world(1), _world(1))
    module = nn.Linear(1, 1)
    binding = bind_module("fixture.parallel", module, contract)
    binding.parallel_execution_receipt = ParallelExecutionReceipt(
        ("fixture.distribute",),
        parameter_sharding={"weight": ("tensor",), "bias": ()},
    )
    graph = DABSNGraph([binding])
    context = ParallelExecutionContext(
        topology=ParallelTopology((ParallelAxis("tensor", 2),)),
        rank=0,
        device=torch.device("cpu"),
        coordinate={"tensor": 0},
        groups={"tensor": None},
        group_ranks={"tensor": (0, 1)},
    )
    object.__setattr__(graph, "_dabsn_parallel_context", context)
    module.weight.grad = torch.tensor([[3.0]])
    module.bias.grad = torch.tensor([4.0])
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)

    def add_remote_contribution(value, *, op):
        assert op is torch.distributed.ReduceOp.SUM
        # Remote unique weight shard: 12^2. Replicated bias contributes 4^2/2.
        value.add_(144.0 + 8.0)

    monkeypatch.setattr("torch.distributed.all_reduce", add_remote_contribution)
    norm = clip_grad_norm(graph, 6.5)
    # FP32, matching the gradients and matching what the unsharded branch of the
    # same function returns. The cross-rank sum of squares is still accumulated
    # in FP64 internally -- that precision is the reason the value below is
    # exact -- but the dtype must not depend on whether the model is sharded, or
    # a sharded run cannot be compared against a single-process one.
    torch.testing.assert_close(norm, torch.tensor(13.0, dtype=torch.float32))
    torch.testing.assert_close(module.weight.grad, torch.tensor([[1.5]]), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(module.bias.grad, torch.tensor([2.0]), rtol=1e-5, atol=1e-6)


def test_generic_trainer_reduces_declared_metrics_only_over_data_axis(monkeypatch):
    class MetricsComponent(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

        def forward(self, value):
            return self.proj(value)

        def forward_with_terms(self, value):
            output = self.proj(value)
            return ComponentOutput(
                output,
                (self.proj.weight.square().mean(),),
                (output.detach().new_tensor(3.0),),
            )

    contract = ComponentContract(_world(2), _world(2))
    binding = bind_module(
        "fixture.metrics",
        MetricsComponent(),
        contract,
        loss_terms=(ResultDeclaration("regularizer", "mean", "framework"),),
        reports=(ResultDeclaration("items", "sum", "framework"),),
    )
    graph = DABSNGraph([binding])
    optimizer = torch.optim.SGD(graph.parameters(), lr=0.01)
    state = DistributedState(
        kind="ddp",
        world_size=2,
        topology=ParallelTopology((ParallelAxis("data", 2),)),
    )
    calls = []

    def all_reduce(value, *, op, group=None):
        calls.append(group)
        value.mul_(2)

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.all_reduce", all_reduce)
    captured = []
    observed = train_step(
        graph,
        torch.randn(2, 3, 2),
        torch.randn(2, 3, 2),
        optimizer,
        distributed_state=state,
        metrics_callback=captured.append,
    )
    assert observed == float(captured[0]["loss"])
    torch.testing.assert_close(captured[0]["report/0:items"], torch.tensor(6.0))
    assert len(calls) == 3


class _FakeScaler:
    def __init__(self, *, skip: bool):
        self.skip = skip
        self.scale = 8.0

    def get_scale(self):
        return self.scale

    def step(self, optimizer):
        if not self.skip:
            optimizer.step()

    def update(self):
        if self.skip:
            self.scale /= 2.0


def test_amp_skipped_step_runs_no_component_lifecycle_action():
    model, component = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.forward_with_terms(torch.randint(0, 19, (2, 4))).value.square().mean().backward()
    before = component.proj.weight.detach().clone()
    applied = apply_optimizer_step(model, optimizer, scaler=_FakeScaler(skip=True))
    assert applied is False
    assert torch.equal(component.proj.weight, before)
    assert component.lifecycle_calls == 0


def test_successful_scaled_step_runs_lifecycle_exactly_once():
    model, component = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.forward_with_terms(torch.randint(0, 19, (2, 4))).value.square().mean().backward()
    before = component.proj.weight.detach().clone()
    applied = apply_optimizer_step(model, optimizer, scaler=_FakeScaler(skip=False))
    assert applied is True
    assert not torch.equal(component.proj.weight, before)
    assert component.lifecycle_calls == 1


def test_accumulation_microbatches_run_no_lifecycle_until_update_boundary():
    model, component = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accumulator = ManualGradientAccumulator(model)
    for _ in range(2):
        accumulator.begin_microbatch()
        model.forward_with_terms(torch.randint(0, 19, (2, 4))).value.square().mean().div(
            2
        ).backward()
        accumulator.add_microbatch()
        assert component.lifecycle_calls == 0
    accumulator.install()
    assert apply_optimizer_step(model, optimizer) is True
    assert component.lifecycle_calls == 1
    accumulator.reset()
