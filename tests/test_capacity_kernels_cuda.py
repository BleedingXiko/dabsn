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


def _force_layout(monkeypatch, layout):
    from dabsn.kernels import capacity as native
    monkeypatch.setattr(native, "capacity_gradient_layout", lambda *shape: layout)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize("batch,steps,size,active", [(2, 40, 32, 4), (1, 64, 64, 8), (3, 33, 16, 2)])
def test_cuda_sparse_bptt_row_totals_match_event_slots(monkeypatch, dtype, stable, batch, steps, size, active):
    """Row-total and event-slot gradient tables agree, incl. carried-state gradients.

    ``stable=False`` exercises arbitrary caller routes (the row-sharded path),
    which run the per-sequence fallback instead of row-tile parallelism.
    """
    from dabsn.kernels.capacity import capacity_scan
    enable_capacity("cuda", required=True)
    torch.manual_seed(91)
    model = DABSNCapacitySubsystem(7, 5, CapacityConfig(mode="sparse", size=size, active_units=active)).cuda()
    with torch.no_grad():
        model.value.normal_(std=0.3)
        model.retention_logit.normal_()
    model = model.to(dtype)
    x0 = torch.randn(batch, steps, 7, device="cuda").to(dtype)
    if stable:
        routes = model.route(x0)
    else:
        # Distinct rows per token, deliberately not slot-stable; repeats across time.
        routes = torch.stack([torch.randperm(size, device="cuda")[:active]
                              for _ in range(batch * steps)]).reshape(batch, steps, active)
    carry0 = torch.randn(batch, size, device="cuda")
    target = torch.randn(batch, steps, 5, device="cuda")
    results = {}
    for layout in ("event", "row"):
        _force_layout(monkeypatch, layout)
        runs = []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            x = x0.clone().requires_grad_()
            carry = carry0.clone().requires_grad_()
            y, final = capacity_scan(x, model.key, model.bias, model.retention_logit, model.value,
                                     model.gain, carry, routes, sparse=True, stable_slots=stable)
            ((y.float() * target).sum() + final.float().square().sum()).backward()
            runs.append([y.detach(), final.detach(), x.grad, carry.grad,
                         *[p.grad.clone() for p in model.parameters()]])
        for a, b in zip(*runs):  # deterministic repeats within one layout
            assert torch.equal(a, b)
        results[layout] = runs[0]
    tolerance = {torch.float64: dict(rtol=1e-11, atol=1e-12), torch.float32: dict(rtol=2e-5, atol=2e-6),
                 torch.bfloat16: dict(rtol=2e-2, atol=2e-2)}[dtype]
    for a, b in zip(results["event"], results["row"]):
        torch.testing.assert_close(a, b, **tolerance)


def test_cuda_online_graph_row_totals_replay_and_finalize_once():
    """Long enough that captured online credit uses row totals (B*C < B*T*A)."""
    from dabsn.kernels import capacity as native
    from dabsn.runtime.graph import make_graphed_train_callable
    enable_capacity("cuda", required=True)
    batch, steps, size, active = 2, 64, 32, 4
    assert native.capacity_gradient_layout(batch, steps, active, size) == "row"

    class TwoCalls(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.capacity = DABSNCapacitySubsystem(7, 5, CapacityConfig(mode="sparse", size=size,
                                                                        active_units=active))

        def forward(self, x):
            out = 0
            for view in (x, x.flip(1)):  # two calls, different rows, same parameters
                rec, elig = self.capacity.initial_online_state(x.shape[0], device=x.device,
                    recurrent_dtype=torch.float32, eligibility_dtype=torch.float32)
                out = out + self.capacity.forward_online(view, rec, elig)[0]
            return out

    model = TwoCalls().cuda()
    x = torch.randn(batch, steps, 7, device="cuda")
    model(x).sum().backward()
    with torch.no_grad():
        for p in model.parameters(): p.add_(dense(p.grad), alpha=-0.05)
    model.zero_grad(set_to_none=True)
    eager = copy.deepcopy(model)
    graph = make_graphed_train_callable(model, (x,), warmup_iters=2, verify=False)
    for replay, sign in enumerate((1.0, -1.0, 0.5, 2.0)):
        xx = torch.randn_like(x) * sign  # selections change on every replay
        target = torch.randn(batch, steps, 5, device="cuda")
        values = []
        for module in (eager, graph):
            module.zero_grad(set_to_none=True)
            yy = module(xx)
            loss = (yy * target).sum() * (0.0 if replay == 2 else 1.0)
            loss.backward()
            finalize_capacity_gradients(module)
            grads = [p.grad.coalesce() if p.grad.is_sparse else p.grad for p in module.parameters()]
            finalize_capacity_gradients(module)  # must consume nothing new
            for p, g in zip(module.parameters(), grads):
                torch.testing.assert_close(dense(p.grad), dense(g), rtol=0, atol=0)
            values.append((yy.detach(), grads))
        # Routing uses a frozen projection: identical in both copies.
        selected = torch.unique(torch.cat((eager.capacity.route(xx).reshape(-1),
                                           eager.capacity.route(xx.flip(1)).reshape(-1))))
        for module_grads in (values[0][1], values[1][1]):
            sparse_grads = [g for g in module_grads if g.is_sparse]
            assert len(sparse_grads) == 4
            for g in sparse_grads:  # zero-gradient selected rows still count as selected
                assert torch.equal(g.indices()[0], selected)
        torch.testing.assert_close(values[0][0], values[1][0], rtol=3e-4, atol=3e-5)
        for a, b in zip(values[0][1], values[1][1]):
            torch.testing.assert_close(dense(a), dense(b), rtol=3e-4, atol=3e-5)
        with torch.no_grad():
            for module in (eager, graph):
                for p in module.parameters(): p.add_(dense(p.grad), alpha=-0.01)


@pytest.mark.parametrize("online", [False, True])
def test_cuda_sparse_gradient_peak_memory_is_row_sized(monkeypatch, online):
    """Measured peak memory: event slots grow with T, row totals do not."""
    from dabsn.kernels import capacity as native
    enable_capacity("cuda", required=True)
    batch, steps, d, s, size, active = 4, 4096, 256, 256, 1024, 16
    event_table = batch * steps * active * (d + s + 2) * 4  # fp32 key/value/bias/retention slots
    row_table = batch * size * (d + s + 2) * 4
    assert native.capacity_gradient_layout(batch, steps, active, size) == "row"
    model = DABSNCapacitySubsystem(d, s, CapacityConfig(mode="sparse", size=size, active_units=active)).cuda()
    x = torch.randn(batch, steps, d, device="cuda")

    def peak(layout):
        _force_layout(monkeypatch, layout)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if online:
            rec, elig = model.initial_online_state(batch, device="cuda", recurrent_dtype=torch.float32,
                                                   eligibility_dtype=torch.float32)
            with native.capture_capacity_rows():
                y, _, _ = model.forward_online(x, rec, elig)
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                y.square().sum().backward()
            finalize_capacity_gradients(model)
        else:
            y = model.forward_sequence(x)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            y.square().sum().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    # Canonical-bucket routes run as grouped GEMMs (BPTT and online): parameter
    # gradients reduce straight into C rows, with no per-event scratch.
    grouped_peak = peak("row")
    print(f"online={online} grouped_peak={grouped_peak/2**20:.1f}MiB "
          f"event_table={event_table/2**20:.1f}MiB row_table={row_table/2**20:.1f}MiB")
    assert grouped_peak < event_table / 4


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


def _grouped_reference_case(mode, batch, steps, d, s, size, active, autocast=False):
    """Grouped CUDA BPTT against the pure-PyTorch reference on identical weights."""
    from dabsn.kernels import capacity as native
    torch.manual_seed(31)
    model = DABSNCapacitySubsystem(d, s, CapacityConfig(mode=mode, size=size, active_units=active)).cuda()
    with torch.no_grad():
        model.value.normal_(std=0.05)
        model.bias.normal_(std=0.1)
        model.retention_logit.normal_()
    x = torch.randn(batch, steps, d, device="cuda")
    cot = torch.randn(batch, steps, s, device="cuda")
    initial = torch.randn(batch, size, device="cuda")
    gfinal = torch.randn(batch, size, device="cuda")
    results = []
    for backend in ("reference", "cuda"):
        native._BACKEND = backend
        model.zero_grad(set_to_none=True)
        xx = x.clone().requires_grad_()
        start = initial.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            y, final = model.forward_sequence(xx, start, return_state=True)
        torch.autograd.backward([y.float(), final], [cot, gfinal])
        results.append([y.detach().float(), final.detach(), xx.grad, start.grad,
                        *[dense(p.grad) for p in model.parameters()]])
    enable_capacity("cuda", required=True)
    return results


@pytest.mark.parametrize("mode,shape", [
    ("sparse", (2, 37, 96, 80, 512, 32)),
    ("sparse", (3, 64, 7, 5, 32, 4)),
    ("sparse", (2, 48, 130, 70, 256, 128)),
    ("dense", (2, 41, 96, 80, 200, 200)),
    ("dense", (3, 16, 7, 5, 32, 32)),
])
def test_cuda_grouped_bptt_matches_reference(mode, shape):
    reference, native = _grouped_reference_case(mode, *shape)
    names = ["y", "final", "dx", "dinitial", "dkey", "dbias", "dvalue", "dretention", "dgain"]
    for name, expected, actual in zip(names, reference, native):
        scale = max(expected.abs().max().item(), 1e-6)
        assert (actual - expected).abs().max().item() <= 2e-5 * scale, name


def test_cuda_grouped_bptt_is_bitwise_deterministic():
    _, first = _grouped_reference_case("sparse", 2, 64, 96, 80, 512, 32)
    _, second = _grouped_reference_case("sparse", 2, 64, 96, 80, 512, 32)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_cuda_grouped_bptt_graph_capture_matches_eager():
    from dabsn.runtime.graph import make_graphed_train_callable
    enable_capacity("cuda", required=True)
    torch.manual_seed(5)

    class Sequence(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.capacity = DABSNCapacitySubsystem(48, 40, CapacityConfig(mode="sparse", size=256, active_units=16))

        def forward(self, x):
            return self.capacity.forward_sequence(x)

    model = Sequence().cuda()
    with torch.no_grad():
        model.capacity.value.normal_(std=0.05)
    eager = copy.deepcopy(model)
    x = torch.randn(2, 24, 48, device="cuda")
    graph = make_graphed_train_callable(model, (x,), warmup_iters=2, verify=False)
    for scale in (1.0, -0.7, 2.0):
        values = []
        for module in (eager, graph):
            module.zero_grad(set_to_none=True)
            y = module(x * scale)
            y.square().mean().backward()
            values.append([y.detach().clone(), *[dense(p.grad).clone() for p in module.parameters()]])
        for a, b in zip(*values):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 64, 96, 80, 512, 32), (1, 64, 16, 64, 512, 32), (2, 6, 96, 80, 512, 32)])
def test_cuda_per_row_sparse_bptt_key_gradient_matches_reference(monkeypatch, shape):
    """Non-bucketed sparse BPTT (per-row kernels) must agree with the reference dK."""
    import dabsn.kernels.capacity_triton as tri
    monkeypatch.setattr(tri, "grouped_supported", lambda *args: False)
    reference, native = _grouped_reference_case("sparse", *shape)
    names = ["y", "final", "dx", "dinitial", "dkey", "dbias", "dvalue", "dretention", "dgain"]
    for name, expected, actual in zip(names, reference, native):
        scale = max(expected.abs().max().item(), 1e-6)
        assert (actual - expected).abs().max().item() <= 2e-5 * scale, name


@pytest.mark.parametrize("mode,shape", [("sparse", (2, 40, 96, 80, 512, 32)), ("dense", (2, 33, 64, 48, 96, 96))])
def test_cuda_grouped_bptt_operand_mode_matches_per_row_kernels(monkeypatch, mode, shape):
    """BF16 operand mode: grouped tensor-core products agree with the per-row kernels
    up to one BF16 rounding step of an activation (summation order differs)."""
    import dabsn.kernels.capacity_triton as tri
    from dabsn.kernels import capacity as native
    batch, steps, d, s, size, active = shape
    torch.manual_seed(7)
    model = DABSNCapacitySubsystem(d, s, CapacityConfig(mode=mode, size=size, active_units=active)).cuda()
    with torch.no_grad():
        model.value.normal_(std=0.05)
        model.bias.normal_(std=0.1)
    x = torch.randn(batch, steps, d, device="cuda")
    cot = torch.randn(batch, steps, s, device="cuda")
    enable_capacity("cuda", required=True)
    results = []
    for grouped in (False, True):
        monkeypatch.setattr(tri, "grouped_supported",
                            (lambda *args: args[0].is_cuda and not args[2]) if grouped else (lambda *args: False))
        model.zero_grad(set_to_none=True)
        xx = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model.forward_sequence(xx)
        y.float().backward(cot)
        results.append([y.detach().float(), xx.grad, *[dense(p.grad) for p in model.parameters()]])
    names = ["y", "dx", *[name for name, _ in model.named_parameters()]]
    for name, expected, actual in zip(names, *results):
        scale = max(expected.abs().max().item(), 1e-6)
        # The scalar gain gradient sums ~1e5 terms with ~50x cancellation, so the
        # ~6e-5 accumulation error of 16-bit tensor-core products (autocast
        # policy; per-row kernels use FP32 FMA) is magnified on this one scalar.
        tolerance = 2**-4 if name == "gain" else 2**-6
        assert (actual - expected).abs().max().item() <= tolerance * scale, name


def _online_reference_case(mode, batch, steps, d, s, size, active, calls=3, per_row=False):
    """Online CUDA vs reference over carried calls: outputs, gradients, state and traces."""
    import dabsn.kernels.capacity_triton as tri
    from dabsn.kernels import capacity as native
    torch.manual_seed(23)
    model = DABSNCapacitySubsystem(d, s, CapacityConfig(mode=mode, size=size, active_units=active)).cuda()
    with torch.no_grad():
        model.value.normal_(std=0.05)
        model.bias.normal_(std=0.1)
        model.retention_logit.normal_()
    xs = [torch.randn(batch, steps, d, device="cuda") for _ in range(calls)]
    cots = [torch.randn(batch, steps, s, device="cuda") for _ in range(calls)]
    real = tri.grouped_supported
    results = []
    try:
        for backend in ("reference", "cuda"):
            native._BACKEND = backend
            tri.grouped_supported = (lambda *args: False) if per_row else real
            rec, elig = model.initial_online_state(batch, device="cuda", recurrent_dtype=torch.float32,
                                                   eligibility_dtype=torch.float32)
            trace = []
            for x, cot in zip(xs, cots):
                model.zero_grad(set_to_none=True)
                xx = x.clone().requires_grad_()
                y, rec, elig = model.forward_online(xx, rec, elig)
                y.backward(cot)
                trace.append([y.detach(), xx.grad, *[dense(p.grad).clone() for p in model.parameters()],
                              rec["features"].clone(), *[v.clone() for v in elig.values()]])
            results.append(trace)
    finally:
        tri.grouped_supported = real
        enable_capacity("cuda", required=True)
    return results


@pytest.mark.parametrize("mode,shape", [
    ("sparse", (2, 37, 96, 80, 512, 32)),
    ("sparse", (3, 64, 7, 5, 32, 4)),
    ("dense", (2, 29, 64, 48, 96, 96)),
])
@pytest.mark.parametrize("per_row", [False, True])
def test_cuda_online_matches_reference_across_carried_calls(mode, shape, per_row):
    reference, native = _online_reference_case(mode, *shape, per_row=per_row)
    for expected_call, actual_call in zip(reference, native):
        for index, (expected, actual) in enumerate(zip(expected_call, actual_call)):
            scale = max(expected.abs().max().item(), 1e-6)
            assert (actual - expected).abs().max().item() <= 3e-5 * scale, index


def test_cuda_grouped_online_is_bitwise_deterministic():
    _, first = _online_reference_case("sparse", 2, 40, 96, 80, 512, 32, calls=2)
    _, second = _online_reference_case("sparse", 2, 40, 96, 80, 512, 32, calls=2)
    for call_a, call_b in zip(first, second):
        for a, b in zip(call_a, call_b):
            assert torch.equal(a, b)


@pytest.mark.parametrize("online", [False, True])
def test_cuda_grouped_honours_float32_matmul_precision(online):
    """``highest`` keeps IEEE FP32 products; ``high`` allows TF32 like cuBLAS."""
    previous = torch.get_float32_matmul_precision()
    try:
        for policy, tolerance in (("highest", 2e-5), ("high", 2e-2)):
            torch.set_float32_matmul_precision(policy)
            if online:
                reference, native = _online_reference_case("sparse", 2, 40, 96, 80, 512, 32, calls=2)
                pairs = [pair for ref_call, nat_call in zip(reference, native) for pair in zip(ref_call, nat_call)]
            else:
                pairs = list(zip(*_grouped_reference_case("sparse", 2, 40, 96, 80, 512, 32)))
            for expected, actual in pairs:
                scale = max(expected.abs().max().item(), 1e-6)
                assert (actual - expected).abs().max().item() <= tolerance * scale, policy
    finally:
        torch.set_float32_matmul_precision(previous)
