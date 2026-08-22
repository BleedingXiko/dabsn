import pytest
import torch

from dabsn.kernels.admitted import (
    _admitted_three_way_read_native_op,
    _admitted_three_way_read_op,
)
from dabsn.kernels.batched_runtime import _batched_forward_tapes, _core_scan_batched_op
from dabsn.kernels.compact_admitted import (
    _admitted_three_way_read_compact_dense_op,
    _compact_dense_bmm_op,
    _compact_masks,
    dense_bmm_three_way_read_exact,
)
from dabsn.kernels.compact_flash import _compact_flash_op
from dabsn.kernels.local_field import local_field_gather
from dabsn.kernels.long import _linrec_reference, linear_recurrence
from dabsn.kernels.permanent import _permanent_reference, permanent_delta_scan


@pytest.fixture(autouse=True)
def _isolate_torch_compile_cache():
    """Prevent unrelated registered operators exhausting Dynamo's variant cache."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_native_scan_operator_names_are_stable():
    assert torch.ops.dabsn.admitted_three_way_read.default
    assert torch.ops.dabsn.admitted_three_way_read_native.default
    assert torch.ops.dabsn.admitted_three_way_read_compact_dense.default
    assert torch.ops.dabsn.admitted_three_way_read_compact_dense_bmm.default
    assert torch.ops.dabsn.admitted_three_way_read_compact_flash.default
    assert torch.ops.dabsn.core_scan_batched.default
    assert torch.ops.dabsn.permanent_delta_scan.default
    assert torch.ops.dabsn.linear_recurrence.default
    assert torch.ops.dabsn.local_field_gather.default
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "dabsn::admitted_three_way_read_compact_dense", "CPU"
    )
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "dabsn::admitted_three_way_read_compact_dense", "CUDA"
    )
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "dabsn::admitted_three_way_read_compact_flash", "CPU"
    )
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "dabsn::_admitted_three_way_read_compact_flash_backward", "CPU"
    )


def test_long_runtime_enable_does_not_replace_read_methods():
    import inspect

    from dabsn.kernels.long import enable_fused_long_read

    source = inspect.getsource(enable_fused_long_read)
    assert 'setattr(DABSNRead, "_long_scan"' not in source
    assert 'setattr(DABSNRead, "_long_scan_from_state"' not in source


def _admitted_inputs(*, dtype=torch.double, requires_grad=True):
    batch, steps, bank, hidden = 1, 3, 2, 2
    query = torch.randn(batch, steps, hidden, dtype=dtype, requires_grad=requires_grad)
    keys = torch.randn(batch, bank, hidden, dtype=dtype, requires_grad=requires_grad)
    writes = torch.randn(batch, bank, hidden, dtype=dtype, requires_grad=requires_grad)
    next_writes = torch.randn(batch, bank, hidden, dtype=dtype, requires_grad=requires_grad)
    cocktail = torch.randn(batch, steps, 4, dtype=dtype, requires_grad=requires_grad)
    bank_cocktail = torch.randn(batch, bank, 4, dtype=dtype, requires_grad=requires_grad)
    bias = torch.randn(batch, bank, dtype=dtype, requires_grad=requires_grad)
    admission = torch.randn(batch, bank, dtype=dtype, requires_grad=requires_grad)

    def scalar():
        return torch.ones(1, dtype=dtype, requires_grad=requires_grad)

    allow = torch.tensor([[[1, 0], [1, 1], [1, 1]]], dtype=torch.bool)
    induct_allow = torch.tensor([[[0, 0], [1, 0], [1, 1]]], dtype=torch.bool)
    eligible = allow.any(dim=-1)
    induct_eligible = induct_allow.any(dim=-1)
    return (
        query,
        keys,
        writes,
        next_writes,
        cocktail,
        bank_cocktail,
        bias,
        admission,
        scalar(),
        allow,
        induct_allow,
        eligible,
        induct_eligible,
        scalar(),
        scalar(),
        scalar(),
        scalar(),
    )


def test_admitted_read_registered_op_opcheck_gradcheck_and_compile():
    torch.manual_seed(96)
    inputs = _admitted_inputs()
    result = torch.library.opcheck(
        torch.ops.dabsn.admitted_three_way_read.default,
        inputs,
    )
    assert all(value == "SUCCESS" for value in result.values())
    assert torch.autograd.gradcheck(
        lambda *values: _admitted_three_way_read_op(*values)[0],
        inputs,
        fast_mode=True,
    )
    assert torch.autograd.gradgradcheck(
        lambda *values: _admitted_three_way_read_op(*values)[0],
        inputs,
        fast_mode=True,
    )
    compiled = torch.compile(_admitted_three_way_read_op, backend="aot_eager", fullgraph=True)
    output = compiled(*inputs)[0]
    output.square().mean().backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in inputs
        if value.is_floating_point()
    )

    native_inputs = _admitted_inputs()
    native_result = torch.library.opcheck(
        torch.ops.dabsn.admitted_three_way_read_native.default,
        native_inputs,
    )
    assert all(value == "SUCCESS" for value in native_result.values())
    assert torch.autograd.gradcheck(
        _admitted_three_way_read_native_op,
        native_inputs,
        fast_mode=True,
    )
    native_compiled = torch.compile(
        _admitted_three_way_read_native_op,
        backend="aot_eager",
        fullgraph=True,
    )
    native_output = native_compiled(*native_inputs)
    native_output.square().mean().backward()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in native_inputs
        if value.is_floating_point()
    )

    nested_inputs = tuple(
        value.detach().unsqueeze(0).repeat(2, *([1] * value.dim())) for value in native_inputs
    )
    vmapped = torch.vmap(_admitted_three_way_read_native_op)(*nested_inputs)
    expected_vmap = torch.stack(
        [
            _admitted_three_way_read_native_op(*(value[index] for value in nested_inputs))
            for index in range(2)
        ]
    )
    torch.testing.assert_close(vmapped, expected_vmap)


def test_admitted_read_cpu_bf16_autocast_has_explicit_dtype_and_backward():
    reference_inputs = _admitted_inputs(dtype=torch.float32)
    native_inputs = tuple(
        value.detach().clone().requires_grad_(value.requires_grad) for value in reference_inputs
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reference_output = _admitted_three_way_read_op(*reference_inputs)[0]
        native_output = _admitted_three_way_read_native_op(*native_inputs)
    assert reference_output.dtype == torch.bfloat16
    assert native_output.dtype == torch.bfloat16
    torch.testing.assert_close(native_output, reference_output)
    reference_output.float().square().mean().backward()
    native_output.float().square().mean().backward()
    for reference, native in zip(reference_inputs, native_inputs):
        if not reference.is_floating_point():
            continue
        assert reference.grad is not None and native.grad is not None
        assert reference.grad.dtype == reference.dtype
        assert native.grad.dtype == native.dtype
        torch.testing.assert_close(native.grad, reference.grad)

    detached = tuple(value.detach() for value in native_inputs)
    compiled = torch.compile(
        _admitted_three_way_read_native_op,
        backend="aot_eager",
        fullgraph=True,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        compiled_output = compiled(*detached)
    assert compiled_output.dtype == torch.bfloat16


def _compact_admitted_inputs(*, mode: int, dtype=torch.double, requires_grad=True):
    dense = _admitted_inputs(dtype=dtype, requires_grad=requires_grad)
    bank_idx = torch.tensor([[0, 2]], dtype=torch.long)
    bank_valid = torch.tensor([[True, True]])
    return (
        *dense[:9],
        bank_idx,
        bank_valid,
        mode,
        *dense[13:],
        2,
    )


@torch.no_grad()
def _compact_reference(inputs):
    query = inputs[0]
    bank_idx, bank_valid, mode = inputs[9:12]
    allow, induct_allow, eligible, induct_eligible = _compact_masks(
        query,
        bank_idx,
        bank_valid,
        mode=mode,
        query_offset=0,
        total_steps=query.shape[1],
    )
    return _admitted_three_way_read_op(
        *inputs[:9],
        allow,
        induct_allow,
        eligible,
        induct_eligible,
        *inputs[12:16],
    )[0]


@pytest.mark.parametrize("mode", [0, 1])
def test_compact_dense_admitted_read_chunked_forward_backward_and_operator_abi(mode):
    torch.manual_seed(109 + mode)
    reference_inputs = _compact_admitted_inputs(mode=mode)
    compact_inputs = tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in reference_inputs
    )

    query = reference_inputs[0]
    bank_idx, bank_valid = reference_inputs[9:11]
    allow, induct_allow, eligible, induct_eligible = _compact_masks(
        query,
        bank_idx,
        bank_valid,
        mode=mode,
        query_offset=0,
        total_steps=query.shape[1],
    )
    expected = _admitted_three_way_read_op(
        *reference_inputs[:9],
        allow,
        induct_allow,
        eligible,
        induct_eligible,
        *reference_inputs[12:16],
    )[0]
    actual = _admitted_three_way_read_compact_dense_op(*compact_inputs)[0]
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    for reference, compact in zip(reference_inputs, compact_inputs):
        if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
            continue
        assert reference.grad is not None and compact.grad is not None
        torch.testing.assert_close(compact.grad, reference.grad)

    detached = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value for value in compact_inputs
    )
    result = torch.library.opcheck(
        torch.ops.dabsn.admitted_three_way_read_compact_dense.default,
        detached,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert all(value == "SUCCESS" for value in result.values())
    compiled = torch.compile(
        _admitted_three_way_read_compact_dense_op,
        backend="aot_eager",
        fullgraph=True,
    )
    compiled(*detached)


def test_compact_dense_admitted_read_gradcheck():
    inputs = _compact_admitted_inputs(mode=0)
    assert torch.autograd.gradcheck(
        lambda *values: _admitted_three_way_read_compact_dense_op(*values)[0],
        inputs,
        fast_mode=True,
    )


@pytest.mark.parametrize("mode", [0, 1])
def test_compact_dense_bmm_composite_is_exact_when_chunked(mode):
    source = _compact_admitted_inputs(mode=mode, dtype=torch.float32)
    baseline = tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in source
    )
    composite = tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in source
    )
    mode_name = "seq" if mode == 0 else "field"
    expected = dense_bmm_three_way_read_exact(
        *baseline[:9],
        baseline[9],
        baseline[10],
        mode=mode_name,
        short_gain=baseline[12],
        pad_gain=baseline[13],
        induct_gain=baseline[14],
        cocktail_gain=baseline[15],
    )
    actual = _compact_dense_bmm_op(*composite)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    for expected_input, actual_input in zip(baseline, composite):
        if not isinstance(expected_input, torch.Tensor) or not expected_input.is_floating_point():
            continue
        assert expected_input.grad is not None and actual_input.grad is not None
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-6, atol=1e-6)

    detached = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value for value in composite
    )
    result = torch.library.opcheck(
        torch.ops.dabsn.admitted_three_way_read_compact_dense_bmm.default,
        detached,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert all(value == "SUCCESS" for value in result.values())
    compiled = torch.compile(_compact_dense_bmm_op, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(*detached), actual.detach(), rtol=1e-6, atol=1e-6)


@pytest.mark.filterwarnings(
    "ignore:Input #.* requires gradient and is not a double precision floating point"
)
def test_compact_dense_bmm_composite_fp32_gradcheck():
    inputs = _compact_admitted_inputs(mode=0, dtype=torch.float32)
    assert torch.autograd.gradcheck(
        _compact_dense_bmm_op,
        inputs,
        eps=1e-3,
        atol=3e-2,
        rtol=3e-2,
        fast_mode=True,
    )


def test_compact_dense_bmm_supports_functionalization_and_second_derivatives():
    inputs = _compact_admitted_inputs(mode=0, dtype=torch.float32)
    eager = _compact_dense_bmm_op(*inputs)
    functional = torch.func.functionalize(_compact_dense_bmm_op)(*inputs)
    torch.testing.assert_close(functional, eager)

    differentiable = tuple(
        value
        for value in inputs
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.requires_grad
    )
    first = torch.autograd.grad(eager.square().sum(), differentiable, create_graph=True)
    second_objective = sum(value.square().sum() for value in first if value.requires_grad)
    second = torch.autograd.grad(second_objective, differentiable, allow_unused=True)
    assert all(value is None or torch.isfinite(value).all() for value in second)


@pytest.mark.parametrize("mode", [0, 1])
def test_compact_flash_operator_cpu_reference_forward_backward_and_abi(mode):
    reference = _compact_admitted_inputs(mode=mode)
    flash = tuple(
        value.detach().clone().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in reference[:-1]
    )
    expected = _admitted_three_way_read_compact_dense_op(*reference)[0]
    actual = _compact_flash_op(*flash)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    for expected_input, actual_input in zip(reference, flash):
        if not isinstance(expected_input, torch.Tensor) or not expected_input.is_floating_point():
            continue
        assert expected_input.grad is not None and actual_input.grad is not None
        torch.testing.assert_close(actual_input.grad, expected_input.grad)

    detached = tuple(
        value.detach() if isinstance(value, torch.Tensor) else value for value in flash
    )
    result = torch.library.opcheck(
        torch.ops.dabsn.admitted_three_way_read_compact_flash.default,
        detached,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert all(value == "SUCCESS" for value in result.values())
    compiled = torch.compile(_compact_flash_op, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(*detached), actual.detach())


def test_compact_flash_operator_gradcheck():
    inputs = _compact_admitted_inputs(mode=0)[:-1]
    assert torch.autograd.gradcheck(_compact_flash_op, inputs, fast_mode=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compact-flash parity gate")
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_registered_compact_flash_cuda_matches_deployed_triton(mode, dtype):
    from dabsn.kernels.triton import enable_triton_kernels
    from dabsn.kernels.triton_runtime import admitted_three_way_read_compact_flash_trainable

    enable_triton_kernels(required=True)
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "dabsn::admitted_three_way_read_compact_flash", "CUDA"
    )
    source = _compact_admitted_inputs(mode=mode, dtype=dtype)[:-1]
    registered = tuple(
        value.detach().clone().cuda().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in source
    )
    deployed = tuple(
        value.detach().clone().cuda().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in source
    )
    mode_name = "seq" if mode == 0 else "field"
    expected = admitted_three_way_read_compact_flash_trainable(
        *deployed[:9],
        deployed[9],
        deployed[10],
        mode=mode_name,
        short_gain=deployed[12],
        pad_gain=deployed[13],
        induct_gain=deployed[14],
        cocktail_gain=deployed[15],
    )
    actual = _compact_flash_op(*registered)
    tolerance = 2e-2 if dtype in {torch.bfloat16, torch.float16} else 2e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    for expected_input, actual_input in zip(deployed, registered):
        if not isinstance(expected_input, torch.Tensor) or not expected_input.is_floating_point():
            continue
        assert expected_input.grad is not None and actual_input.grad is not None
        torch.testing.assert_close(
            actual_input.grad,
            expected_input.grad,
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity gate")
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_compact_dense_bmm_registered_cuda_matches_deployed_dense_bmm(mode, dtype):
    """Certify the production registered ABI against the deployed dense BMM math."""
    from dabsn.kernels.triton_runtime import dense_bmm_three_way_read

    torch.manual_seed(211 + mode)
    source = _compact_admitted_inputs(mode=mode, dtype=dtype)
    registered = tuple(
        value.detach().clone().cuda().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in source
    )
    baseline = tuple(
        value.detach().clone().cuda().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in source
    )
    mode_name = "seq" if mode == 0 else "field"
    expected = dense_bmm_three_way_read(
        *baseline[:9],
        baseline[9],
        baseline[10],
        mode=mode_name,
        short_gain=baseline[12],
        pad_gain=baseline[13],
        induct_gain=baseline[14],
        cocktail_gain=baseline[15],
    )
    actual = _compact_dense_bmm_op(*registered)
    tolerance = 2e-2 if dtype in {torch.bfloat16, torch.float16} else 1e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    for expected_input, actual_input in zip(baseline, registered):
        if not isinstance(expected_input, torch.Tensor) or not expected_input.is_floating_point():
            continue
        assert expected_input.grad is not None and actual_input.grad is not None
        torch.testing.assert_close(
            actual_input.grad,
            expected_input.grad,
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fullgraph gate")
def test_compact_dense_bmm_registered_cuda_fullgraph_forward_backward():
    inputs = tuple(
        value.detach().cuda().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for value in _compact_admitted_inputs(mode=0, dtype=torch.float32)
    )

    def forward_backward(*values):
        output = _compact_dense_bmm_op(*values)
        return torch.autograd.grad(output.square().mean(), values[:9] + values[12:16])

    compiled = torch.compile(forward_backward, fullgraph=True)
    eager = forward_backward(*inputs)
    observed = compiled(*inputs)
    for actual, expected in zip(observed, eager):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _core_scan_inputs(*, requires_grad=True):
    batch, steps, hidden = 1, 2, 2

    def vector(value=0.0):
        return torch.full((hidden,), value, requires_grad=requires_grad)

    return (
        torch.randn(batch, steps, hidden, requires_grad=requires_grad),
        torch.randn(batch, steps, hidden, requires_grad=requires_grad),
        torch.randn(hidden, hidden, requires_grad=requires_grad),
        torch.randn(hidden, hidden, requires_grad=requires_grad),
        vector(),
        vector(-2.25),
        vector(-4.6),
        vector(),
        vector(),
        vector(),
        vector(),
        vector(),
        vector(),
        vector(),
        vector(),
        vector(),
        vector(),
        vector(2.2),
        vector(),
        vector(),
        torch.full((1,), -2.0, requires_grad=requires_grad),
        torch.full((1,), -1.0, requires_grad=requires_grad),
        torch.full((1,), -6.0, requires_grad=requires_grad),
        torch.zeros(batch, hidden, requires_grad=requires_grad),
        torch.ones(batch, hidden, requires_grad=requires_grad),
        torch.zeros(batch, hidden, requires_grad=requires_grad),
    )


def test_core_scan_registered_op_matches_reference_backward_and_compiles():
    torch.manual_seed(95)
    inputs = _core_scan_inputs()
    expected = _batched_forward_tapes(*inputs)
    actual = _core_scan_batched_op(*inputs)
    assert len(actual) == 11
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference)
    sum(output.float().square().mean() for output in actual).backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in inputs)

    detached = tuple(value.detach() for value in inputs)
    result = torch.library.opcheck(
        torch.ops.dabsn.core_scan_batched.default,
        detached,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert all(value == "SUCCESS" for value in result.values())
    compiled = torch.compile(_core_scan_batched_op, backend="aot_eager", fullgraph=True)
    compiled(*detached)


def test_linear_recurrence_registered_op_matches_reference_and_gradcheck():
    torch.manual_seed(90)
    a = torch.sigmoid(torch.randn(2, 3, 4, dtype=torch.double)).requires_grad_()
    g = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    initial = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    torch.testing.assert_close(linear_recurrence(a, g, initial), _linrec_reference(a, g, initial))
    result = torch.library.opcheck(
        torch.ops.dabsn.linear_recurrence.default,
        (a, g, initial),
    )
    assert all(value == "SUCCESS" for value in result.values())
    assert torch.autograd.gradcheck(linear_recurrence, (a, g, initial), fast_mode=True)


def test_permanent_scan_registered_op_matches_reference_and_gradcheck():
    torch.manual_seed(91)
    key = torch.randn(1, 3, 2, dtype=torch.double, requires_grad=True)
    value = torch.randn(1, 3, 2, dtype=torch.double, requires_grad=True)
    beta = torch.sigmoid(torch.randn(1, 3, dtype=torch.double)).requires_grad_()
    torch.testing.assert_close(
        permanent_delta_scan(key, value, beta),
        _permanent_reference(key, value, beta),
    )
    result = torch.library.opcheck(
        torch.ops.dabsn.permanent_delta_scan.default,
        (key, value, beta),
    )
    assert all(value == "SUCCESS" for value in result.values())
    assert torch.autograd.gradcheck(
        permanent_delta_scan,
        (key, value, beta),
        fast_mode=True,
    )


def test_registered_scans_compile_fullgraph():
    def function(a, g, initial, key, value, beta):
        return linear_recurrence(a, g, initial) + permanent_delta_scan(key, value, beta)

    compiled = torch.compile(function, backend="aot_eager", fullgraph=True)
    inputs = (
        torch.sigmoid(torch.randn(1, 3, 2)).requires_grad_(),
        torch.randn(1, 3, 2, requires_grad=True),
        torch.randn(1, 2, requires_grad=True),
        torch.randn(1, 3, 2, requires_grad=True),
        torch.randn(1, 3, 2, requires_grad=True),
        torch.sigmoid(torch.randn(1, 3)).requires_grad_(),
    )
    compiled(*inputs).square().mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in inputs)


def test_registered_scans_vmap_match_loop():
    a = torch.sigmoid(torch.randn(3, 1, 4, 2))
    g = torch.randn(3, 1, 4, 2)
    initial = torch.randn(3, 1, 2)
    actual = torch.vmap(linear_recurrence)(a, g, initial)
    expected = torch.stack([linear_recurrence(a[i], g[i], initial[i]) for i in range(3)])
    torch.testing.assert_close(actual, expected)


def test_local_field_registered_op_opcheck_gradcheck_compile_and_vmap():
    inputs = torch.randn(2, 4, 3, dtype=torch.double, requires_grad=True)
    patch = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    actual, route = local_field_gather(inputs, patch)
    torch.testing.assert_close(actual, inputs[:, patch])
    assert route == "torch_indexing_then_shared_core"
    result = torch.library.opcheck(
        torch.ops.dabsn.local_field_gather.default,
        (inputs, patch),
    )
    assert all(value == "SUCCESS" for value in result.values())
    assert torch.autograd.gradcheck(
        torch.ops.dabsn.local_field_gather.default,
        (inputs, patch),
        fast_mode=True,
    )
    compiled = torch.compile(
        torch.ops.dabsn.local_field_gather.default,
        backend="aot_eager",
        fullgraph=True,
    )
    compiled(inputs, patch).sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()

    batched = torch.randn(3, 2, 4, 3)
    vmapped = torch.vmap(
        torch.ops.dabsn.local_field_gather.default,
        in_dims=(0, None),
    )(batched, patch)
    expected = torch.stack([batched[index][:, patch] for index in range(3)])
    torch.testing.assert_close(vmapped, expected)


def test_every_registered_native_operator_supports_functionalization():
    cases = (
        (_admitted_three_way_read_op, _admitted_inputs(requires_grad=False)),
        (_admitted_three_way_read_native_op, _admitted_inputs(requires_grad=False)),
        (
            _admitted_three_way_read_compact_dense_op,
            _compact_admitted_inputs(mode=0, requires_grad=False),
        ),
        (
            _compact_dense_bmm_op,
            _compact_admitted_inputs(mode=0, dtype=torch.float32, requires_grad=False),
        ),
        (_compact_flash_op, _compact_admitted_inputs(mode=0, requires_grad=False)[:-1]),
        (_core_scan_batched_op, _core_scan_inputs(requires_grad=False)),
        (
            torch.ops.dabsn.linear_recurrence.default,
            (
                torch.sigmoid(torch.randn(1, 3, 2)),
                torch.randn(1, 3, 2),
                torch.randn(1, 2),
            ),
        ),
        (
            torch.ops.dabsn.permanent_delta_scan.default,
            (
                torch.randn(1, 3, 2),
                torch.randn(1, 3, 2),
                torch.sigmoid(torch.randn(1, 3)),
            ),
        ),
        (
            torch.ops.dabsn.local_field_gather.default,
            (
                torch.randn(2, 4, 3),
                torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
            ),
        ),
    )
    for operator, inputs in cases:
        expected = operator(*inputs)
        actual = torch.func.functionalize(operator)(*inputs)
        expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
        actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
        assert actual_spec == expected_spec
        for observed, reference in zip(actual_leaves, expected_leaves):
            torch.testing.assert_close(observed, reference)
