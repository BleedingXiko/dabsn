"""Native kernels must preserve v4 learning, routing and temporal credit."""
import importlib.util
from pathlib import Path

import pytest
import torch

from dabsn.capacity import DABSNCapacitySubsystem
from dabsn.config import CapacityConfig


def oracle(config):
    spec = importlib.util.spec_from_file_location(
        "dabsn._capacity_v4_oracle", Path(__file__).parent / "fixtures/capacity_v4_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DABSNCapacitySubsystem(3, 2, config).double()


def dense(tensor):
    return tensor.to_dense() if tensor.is_sparse else tensor


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("online", [False, True])
def test_native_zero_init_learning_then_v4_gradient_parity(mode, online):
    from dabsn.kernels.capacity import enable_capacity, capacity_runtime_status
    enable_capacity("cpu", required=True)
    torch.manual_seed(713)
    cfg = CapacityConfig(mode=mode, size=8, active_units=2, address_mode="flat")
    ref = oracle(cfg)
    native = DABSNCapacitySubsystem(3, 2, cfg).double()
    native.load_state_dict(ref.state_dict())
    x = torch.randn(2, 5, 3, dtype=torch.float64)
    target = torch.randn(2, 5, 2, dtype=torch.float64)
    assert torch.count_nonzero(native.value) == 0
    for iteration in range(3):
        results = []
        for model in (ref, native):
            model.zero_grad(set_to_none=True)
            xx = x.clone().requires_grad_()
            if online:
                recurrent, eligibility = model.initial_online_state(
                    2, device="cpu", recurrent_dtype=x.dtype, eligibility_dtype=x.dtype
                )
                yy, rr, ee = model.forward_online(xx, recurrent, eligibility)
            else:
                yy = model.forward_sequence(xx)
            (yy * target).sum().backward()
            results.append((yy.detach(), xx.grad.clone(), [dense(p.grad).clone() for p in model.parameters()]))
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(dense(p.grad), alpha=-0.03)
        for a, b in zip(results[0][:2], results[1][:2]):
            torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-9)
        for a, b in zip(results[0][2], results[1][2]):
            torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-9)
        assert torch.count_nonzero(native.value) > 0
        assert native.key.grad.is_sparse == (online and mode == "sparse")
    assert native.last_backend == "cpu_native_cpp"
    assert capacity_runtime_status()["cpu_calls"] > 0


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_native_carried_state_and_input_gradients(mode):
    from dabsn.kernels.capacity import enable_capacity
    enable_capacity("cpu", required=True)
    cfg = CapacityConfig(mode=mode, size=8, active_units=2)
    model = DABSNCapacitySubsystem(3, 2, cfg).double()
    x = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    # An actual first update opens the zero initialized value path.
    model.forward_sequence(x).sum().backward()
    with torch.no_grad():
        model.value.add_(model.value.grad, alpha=-0.1)
    model.zero_grad(set_to_none=True)
    full, final = model.forward_sequence(x, return_state=True)
    left, carry = model.forward_sequence(x[:, :2], return_state=True)
    right, end = model.forward_sequence(x[:, 2:], initial_features=carry, return_state=True)
    torch.testing.assert_close(torch.cat((left, right), 1), full)
    torch.testing.assert_close(end, final)
    gfull = torch.autograd.grad(full.sum() + final.sum(), x, retain_graph=True)[0]
    gsplit = torch.autograd.grad(left.sum() + right.sum() + end.sum(), x)[0]
    torch.testing.assert_close(gfull, gsplit)


@pytest.mark.parametrize("address", ["flat", "product"])
def test_native_address_ties_choose_first_bucket(address):
    from dabsn.kernels.capacity import enable_capacity
    enable_capacity("cpu", required=True)
    model = DABSNCapacitySubsystem(3, 2, CapacityConfig(
        mode="sparse", size=8, active_units=2, address_mode=address, address_branching=2
    )).double()
    torch.testing.assert_close(model.route(torch.zeros(2, 3, 3, dtype=torch.float64)),
                               torch.tensor([0, 1]).expand(2, 3, 2))


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_native_compiled_bptt_preserves_parameter_learning(mode):
    from dabsn.kernels.capacity import enable_capacity
    enable_capacity("cpu", required=True)
    cfg = CapacityConfig(mode=mode, size=8, active_units=2)
    model = DABSNCapacitySubsystem(3, 2, cfg)
    compiled = torch.compile(model.forward_sequence, backend="aot_eager", fullgraph=True)
    x = torch.randn(2, 3, 3, requires_grad=True)
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        compiled(x).sum().backward()
        assert model.value.grad.abs().sum() > 0
        with torch.no_grad():
            for p in model.parameters():
                p.add_(p.grad,alpha=-0.05)
    assert model.key.grad.abs().sum() > 0


def test_native_online_selected_rows_hold_and_replay_carried_eligibility():
    from dabsn.kernels.capacity import enable_capacity, capacity_scan
    enable_capacity("cpu", required=True)
    model = DABSNCapacitySubsystem(3, 2, CapacityConfig(mode="sparse", size=8, active_units=2)).double()
    x = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    model.forward_sequence(x).sum().backward()
    with torch.no_grad():
        model.value.add_(model.value.grad, alpha=-0.1)
    model.zero_grad(set_to_none=True)
    recurrent, elig = model.initial_online_state(2,device="cpu",recurrent_dtype=x.dtype,eligibility_dtype=x.dtype)
    recurrent["features"].fill_(0.4)
    for e in elig.values():
        e.fill_(0.7)
    # Deliberately change slot order: each physical row owns its recurrence.
    indices = torch.tensor([[[0,1],[1,0],[0,1]],[[2,3],[3,2],[2,3]]])
    y, final = capacity_scan(x,model.key,model.bias,model.retention_logit,model.value,
                             model.gain,recurrent["features"],indices,
                             eligibility=tuple(elig.values()),sparse=True)
    y.sum().backward()
    assert torch.all(final[:,4:] == 0.4)
    for e in elig.values():
        assert torch.all(e[:,4:] == 0.7)
    assert model.key.grad.is_sparse
    assert torch.equal(model.key.grad.coalesce().indices()[0],torch.tensor([0,1,2,3]))


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_native_empty_sequence_keeps_state_and_final_state_gradient(mode):
    from dabsn.kernels.capacity import enable_capacity
    enable_capacity("cpu", required=True)
    model = DABSNCapacitySubsystem(3, 2, CapacityConfig(mode=mode,size=8,active_units=2)).double()
    x = torch.empty(2,0,3,dtype=torch.float64,requires_grad=True)
    state = torch.randn(2,8,dtype=torch.float64,requires_grad=True)
    y, final = model.forward_sequence(x,initial_features=state,return_state=True)
    assert y.shape == (2,0,2)
    torch.testing.assert_close(final,state)
    final.sum().backward()
    torch.testing.assert_close(state.grad,torch.ones_like(state))


def test_native_cpu_autocast_keeps_fp32_eligibility_finite():
    from dabsn.kernels.capacity import enable_capacity
    enable_capacity("cpu", required=True)
    model = DABSNCapacitySubsystem(3,2,CapacityConfig(mode="sparse",size=8,active_units=2))
    x = torch.randn(2,3,3,dtype=torch.bfloat16,requires_grad=True)
    rec, elig = model.initial_online_state(2,device="cpu",recurrent_dtype=torch.bfloat16,
                                          eligibility_dtype=torch.float32)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        y, _, _ = model.forward_online(x,rec,elig)
        y.float().sum().backward()
    assert y.dtype == torch.bfloat16
    for p in model.parameters():
        assert torch.isfinite(dense(p.grad)).all()
    for e in elig.values():
        assert e.dtype == torch.float32 and torch.isfinite(e).all()


def test_native_rejects_out_of_range_routes_before_pointer_access():
    from dabsn.kernels.capacity import capacity_scan
    model = DABSNCapacitySubsystem(3,2,CapacityConfig(mode="sparse",size=8,active_units=2))
    with pytest.raises((ValueError,RuntimeError)):
        capacity_scan(torch.randn(1,1,3),model.key,model.bias,model.retention_logit,model.value,
                      model.gain,torch.zeros(1,8),torch.tensor([[[0,8]]]),sparse=True)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_native_activation_checkpoint_learned_gradients(mode):
    from torch.utils.checkpoint import checkpoint
    from dabsn.kernels.capacity import enable_capacity
    enable_capacity("cpu", required=True)
    model = DABSNCapacitySubsystem(3,2,CapacityConfig(mode=mode,size=8,active_units=2)).double()
    x = torch.randn(2,5,3,dtype=torch.float64,requires_grad=True)
    model.forward_sequence(x).sum().backward()
    with torch.no_grad():
        model.value.add_(model.value.grad,alpha=-0.05)
    result = []
    for use_checkpoint in (False,True):
        model.zero_grad(set_to_none=True)
        xx = x.detach().clone().requires_grad_()
        yy = (checkpoint(model.forward_sequence,xx,use_reentrant=False) if use_checkpoint
              else model.forward_sequence(xx))
        yy.square().sum().backward()
        result.append((yy,xx.grad,[p.grad.clone() for p in model.parameters()]))
    for a,b in zip(result[0][:2],result[1][:2]):
        torch.testing.assert_close(a,b)
    for a,b in zip(result[0][2],result[1][2]):
        torch.testing.assert_close(a,b)


def test_deferred_online_rows_finalize_once_and_match_normal_gradients():
    from dabsn.kernels.capacity import enable_capacity, capture_capacity_rows, finalize_capacity_gradients
    enable_capacity("cpu",required=True)
    cfg = CapacityConfig(mode="sparse",size=8,active_units=2)
    model = DABSNCapacitySubsystem(3,2,cfg).double()
    x = torch.randn(2,4,3,dtype=torch.float64,requires_grad=True)
    for step in range(3):
        gradients = []
        for deferred in (False,True):
            model.zero_grad(set_to_none=True)
            rec,elig=model.initial_online_state(2,device="cpu",recurrent_dtype=x.dtype,eligibility_dtype=x.dtype)
            if deferred:
                with capture_capacity_rows():
                    y,_,_=model.forward_online(x,rec,elig)
                    y.sum().backward()
                assert model.key.grad is None
                finalize_capacity_gradients(model)
            else:
                y,_,_=model.forward_online(x,rec,elig)
                y.sum().backward()
            gradients.append([dense(p.grad).clone() for p in model.parameters()])
            if deferred:
                finalize_capacity_gradients(model)
                for p,g in zip(model.parameters(),gradients[-1]):
                    torch.testing.assert_close(dense(p.grad),g)
        for a,b in zip(*gradients):
            torch.testing.assert_close(a,b)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(dense(p.grad),alpha=-0.05)


def test_capture_capacity_scope_preserves_hook_free_module_contract():
    from dabsn.kernels import capacity as native

    class ReusedCapacity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.capacity = DABSNCapacitySubsystem(3, 2,
                CapacityConfig(mode="sparse", size=8, active_units=2)).double()

        def forward(self, x):
            outputs = []
            for _ in range(2):
                recurrent, eligibility = self.capacity.initial_online_state(
                    x.shape[0], device=x.device, recurrent_dtype=x.dtype,
                    eligibility_dtype=x.dtype)
                outputs.append(self.capacity.forward_online(x, recurrent, eligibility)[0])
            return sum(outputs)

    native.enable_capacity("cpu", required=True)
    model = ReusedCapacity()
    original_forward = model.forward
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    with native.capture_capacity_rows(model):
        assert not model._forward_pre_hooks
        assert not model._forward_hooks and not model._backward_hooks
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            model(x).sum().backward()
            native.finalize_capacity_gradients(model)
            for parameter in model.capacity.sparse_parameters():
                assert set(parameter._capacity_pending_rows) == {0, 1}
                assert parameter.grad.is_sparse
    assert model.forward == original_forward
    assert "forward" not in model.__dict__


def test_capture_capacity_scope_retains_installed_graph_forward():
    from dabsn.kernels.capacity import capture_capacity_rows
    module = torch.nn.Sequential(DABSNCapacitySubsystem(3, 2,
        CapacityConfig(mode="sparse", size=8, active_units=2)))
    replacement = lambda x: x
    with capture_capacity_rows(module):
        assert not module._forward_pre_hooks
        module.forward = replacement
    assert module.forward is replacement


def test_capture_scope_without_sparse_capacity_leaves_forward_untouched():
    from dabsn.kernels.capacity import capture_capacity_rows
    module = torch.nn.Linear(3, 2)
    original = module.forward
    with capture_capacity_rows(module):
        assert module.forward == original
        assert "forward" not in module.__dict__
        assert not module._forward_pre_hooks


@pytest.mark.parametrize("backend", ["reference", "cpu"])
@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
def test_bfloat16_capacity_keeps_fp32_carry_and_accepts_fp32_carried_state(backend,mode,dtype):
    from dabsn.kernels.capacity import enable_capacity
    enable_capacity(backend,required=True)
    model=DABSNCapacitySubsystem(3,2,CapacityConfig(mode=mode,size=8,active_units=2)).to(dtype=dtype)
    x=torch.randn(2,4,3,dtype=dtype,requires_grad=True)
    y,state=model.forward_sequence(x,initial_features=torch.zeros(2,8),return_state=True)
    y.float().sum().backward()
    assert state.dtype==torch.float32
    rec,elig=model.initial_online_state(2,device="cpu",recurrent_dtype=x.dtype,eligibility_dtype=x.dtype)
    assert rec["features"].dtype==torch.float32
    assert all(e.dtype==torch.float32 for e in elig.values())
    model.zero_grad(set_to_none=True)
    y,rec,elig=model.forward_online(x,rec,elig)
    y.float().sum().backward()
    assert rec["features"].dtype==torch.float32
    assert torch.isfinite(dense(model.value.grad)).all()


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_online_nonzero_initial_eligibility_matches_v4(mode):
    from dabsn.kernels.capacity import enable_capacity
    enable_capacity("cpu",required=True)
    cfg=CapacityConfig(mode=mode,size=8,active_units=2)
    ref=oracle(cfg)
    native=DABSNCapacitySubsystem(3,2,cfg).double()
    native.load_state_dict(ref.state_dict())
    x=torch.randn(2,4,3,dtype=torch.float64)
    rec,elig=native.initial_online_state(2,device="cpu",recurrent_dtype=x.dtype,eligibility_dtype=x.dtype)
    for t in [*rec.values(),*elig.values()]:
        t.normal_()
    results=[]
    for model in (ref,native):
        # Learn the output rows through a real update first.
        model.forward_sequence(x).sum().backward()
        with torch.no_grad(): model.value.add_(model.value.grad,alpha=-0.1)
        model.zero_grad(set_to_none=True)
        rr={k:v.clone() for k,v in rec.items()}
        ee={k:v.clone() for k,v in elig.items()}
        xx=x.clone().requires_grad_()
        yy,rr,ee=model.forward_online(xx,rr,ee)
        yy.sum().backward()
        results.append([yy,xx.grad,*[dense(p.grad) for p in model.parameters()],*rr.values(),*ee.values()])
    for a,b in zip(*results):
        torch.testing.assert_close(a,b,rtol=2e-9,atol=2e-10)


@pytest.mark.parametrize("backend",["reference","cpu"])
def test_sparse_bptt_records_addressed_zero_gradient_rows_only_on_backward(backend):
    from dabsn.kernels.capacity import enable_capacity
    enable_capacity(backend,required=True)
    model=DABSNCapacitySubsystem(3,2,CapacityConfig(mode="sparse",size=8,active_units=2)).double()
    x=torch.randn(2,4,3,dtype=torch.float64,requires_grad=True)
    y=model.forward_sequence(x)
    assert not model.key._capacity_touched_rows.any()
    (y.sum()*0).backward()
    rows=model.route(x).unique()
    assert torch.equal(model.key._capacity_touched_rows.nonzero().flatten(),rows)
    assert all(torch.equal(p._capacity_touched_rows,model.key._capacity_touched_rows)
               for p in model.sparse_parameters())


@pytest.mark.parametrize("mode",["dense","sparse"])
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
def test_native_autocast_quantizes_operands_with_fp32_state(mode,dtype):
    from dabsn.kernels.capacity import enable_capacity
    enable_capacity("cpu",required=True)
    model=DABSNCapacitySubsystem(3,2,CapacityConfig(mode=mode,size=8,active_units=2))
    x=torch.randn(2,4,3,requires_grad=True)
    with torch.autocast("cpu",dtype=dtype):
        y,state=model.forward_sequence(x,return_state=True)
    assert y.dtype==dtype
    assert state.dtype==torch.float32
    y.float().sum().backward()
    assert model.value.grad.dtype==torch.float32
    assert model.last_precision["operand_dtype"]==str(dtype)
    assert model.last_precision["accumulation_dtype"]==str(torch.float32)
