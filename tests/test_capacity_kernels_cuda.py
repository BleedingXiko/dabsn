"""Target-GPU evidence: skipped on CPU, never reported as CUDA verification."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from dabsn.capacity import DABSNCapacitySubsystem
from dabsn.config import CapacityConfig
from dabsn.kernels.capacity import enable_capacity, finalize_capacity_gradients

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="requires real CUDA device")


def dense(t):
    return t.to_dense() if t.is_sparse else t


@pytest.mark.parametrize("mode",["dense","sparse"])
@pytest.mark.parametrize("online",[False,True])
def test_cuda_v4_zero_init_learning_gradients_and_state(mode,online):
    enable_capacity("cuda",required=True)
    torch.manual_seed(14)
    config=CapacityConfig(mode=mode,size=32,active_units=4)
    spec=importlib.util.spec_from_file_location("dabsn._capacity_cuda_oracle",
        Path(__file__).parent/"fixtures/capacity_v4_reference.py")
    reference=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    ref=reference.DABSNCapacitySubsystem(7,5,config).cuda().double()
    native=DABSNCapacitySubsystem(7,5,config).cuda().double()
    native.load_state_dict(ref.state_dict())
    x=torch.randn(2,9,7,device="cuda",dtype=torch.float64)
    target=torch.randn(2,9,5,device="cuda",dtype=torch.float64)
    for _ in range(3):
        results=[]
        for model in (ref,native):
            model.zero_grad(set_to_none=True)
            xx=x.clone().requires_grad_()
            if online:
                rec,elig=model.initial_online_state(2,device="cuda",recurrent_dtype=x.dtype,eligibility_dtype=x.dtype)
                yy,rec,elig=model.forward_online(xx,rec,elig)
            else:
                yy=model.forward_sequence(xx)
            (yy*target).sum().backward()
            results.append([yy.detach(),xx.grad,*[dense(p.grad).clone() for p in model.parameters()]])
            with torch.no_grad():
                for p in model.parameters(): p.add_(dense(p.grad),alpha=-0.03)
        for actual,expected in zip(*results):
            torch.testing.assert_close(actual,expected,rtol=2e-8,atol=2e-9)
    assert native.last_backend=="cuda_triton"


def test_cuda_online_graph_replays_independent_state_and_finalizes_rows_once():
    from dabsn.runtime.graph import make_graphed_train_callable
    enable_capacity("cuda",required=True)

    class OnlineModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.capacity=DABSNCapacitySubsystem(7,5,CapacityConfig(mode="sparse",size=32,active_units=4))

        def forward(self,x):
            # Fresh independent streams on every invocation, including replay.
            rec,elig=self.capacity.initial_online_state(x.shape[0],device=x.device,
                recurrent_dtype=torch.float32,eligibility_dtype=torch.float32)
            return self.capacity.forward_online(x,rec,elig)[0]

    model=OnlineModule().cuda()
    x=torch.randn(2,8,7,device="cuda")
    model(x).sum().backward()
    with torch.no_grad():
        for p in model.parameters(): p.add_(dense(p.grad),alpha=-0.05)
    model.zero_grad(set_to_none=True)
    eager=copy.deepcopy(model)
    graph=make_graphed_train_callable(model,(x,),warmup_iters=2,verify=False)
    for sign in (1.0,-1.0,0.5):
        xx=x*sign
        target=torch.randn(2,8,5,device="cuda")
        values=[]
        for module in (eager,graph):
            module.zero_grad(set_to_none=True)
            yy=module(xx)
            (yy*target).sum().backward()
            finalize_capacity_gradients(module)
            values.append([yy.detach(),*[dense(p.grad).clone() for p in module.parameters()]])
            finalize_capacity_gradients(module)
            for p,g in zip(module.parameters(),values[-1][1:]):
                torch.testing.assert_close(dense(p.grad),g)
        for a,b in zip(*values):
            torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-5)
        with torch.no_grad():
            for module in (eager,graph):
                for p in module.parameters(): p.add_(dense(p.grad),alpha=-0.01)


def test_cuda_graph_build_preserves_registered_online_state():
    from dabsn.runtime.graph import make_graphed_train_callable
    enable_capacity("cuda", required=True)

    class CarriedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.capacity = DABSNCapacitySubsystem(
                7, 5, CapacityConfig(mode="sparse", size=32, active_units=4))
            rec, elig = self.capacity.initial_online_state(2, device="cpu",
                recurrent_dtype=torch.float32, eligibility_dtype=torch.float32)
            self.register_buffer("features", rec["features"])
            for name, value in elig.items():
                self.register_buffer("elig_" + name, value)
            self.elig_names = tuple(elig)

        def forward(self, x):
            return self.capacity.forward_online(x, {"features": self.features},
                {name: getattr(self, "elig_" + name) for name in self.elig_names})[0]

    model = CarriedModule().cuda()
    x = torch.randn(2, 8, 7, device="cuda")
    # Reach learned output weights and nonzero carried state through a real update.
    model(x).sum().backward()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(dense(p.grad), alpha=-0.03)
    model.zero_grad(set_to_none=True)
    eager = copy.deepcopy(model)
    before = {name: value.clone() for name, value in model.named_buffers()}
    graph = make_graphed_train_callable(model, (x,), warmup_iters=2, verify=True)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for scale in (1.0, -0.5):
        results = []
        for module in (eager, graph):
            module.zero_grad(set_to_none=True)
            y = module(x * scale)
            y.square().mean().backward()
            finalize_capacity_gradients(module)
            results.append([y.detach().clone(),
                *[dense(p.grad).clone() for p in module.parameters()],
                *[value.clone() for value in module.buffers()]])
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5)
