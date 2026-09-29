# Optional native Capacity

Capacity is an optional part of each DABSN block. It computes recurrent features
over the block's input experience axis and adds its output beside native Read,
before the state-to-hidden projection:

```text
state_to_hidden(core_state + read_gain * read + capacity(input))
```

It works with sequence, field, and hybrid Read geometries. Read retains its own
geometry; enabling Capacity does not make a noncausal field streamable.

## Construction

```python
from dabsn import CapacityConfig, DABSNConfig, build_dabsn_from_config

model = build_dabsn_from_config(DABSNConfig(
    input_dim=8, out_dim=3, hidden_dim=12, state_dim=10,
    capacity=CapacityConfig(mode="sparse", size=64, active_units=8),
    layers=[
        {"hidden_dim": 12, "state_dim": 10, "read_geometry": "seq"},
        {"hidden_dim": 16, "state_dim": 10, "read_geometry": "seq",
         "capacity": {"enabled": False}},
    ],
))
```

The same mapping is accepted by direct builders, `DABSNSequenceLM`,
`DABSNPretrainConfig`, JSON configuration loading, and the `dabsn:block` graph
provider. A layer's explicit configuration overrides the model setting;
`enabled=False` disables that layer. Omitting Capacity creates no Capacity
parameters and preserves the existing initialization and execution path.

Output weights start at zero. The input keys retain the supplied initialization.
Dense mode updates every feature. Sparse mode uses fixed geometric addressing,
with reproducible `address_seed`, flat or product addressing, and exactly
`active_units` selected rows per experience. Unselected recurrent features and
eligibility rows retain their state. There is no learned router.

## Training and state

Use `dabsn.runtime.build_training_optimizer(model, ...)` with normal AdamW
options. With sparse Capacity, it owns SparseAdam for Capacity's key, bias,
retention, and output rows, and AdamW for other parameters, including the gain.
Sparse rows have zero weight decay. Without sparse Capacity it returns ordinary
AdamW. The combined optimizer exposes parameter groups, scheduler access,
zeroing, stepping, and state serialization.

Ordinary backpropagation keeps ordinary gradient behavior. The optimizer's
`prepare_gradients()` converts Capacity rows to coalesced COO at the explicit
optimizer boundary. Call it after accumulation and before AMP unscaling or
optional clipping in custom training loops. Framework training handles this
boundary. Online temporal credit returns sparse row contributions directly and
carries recurrence and eligibility through the existing online state API.
The ordinary backward and online current-step input credit remain distinct.

`dabsn.runtime.train_step(..., scaler=scaler)` handles AMP scaling, gradient
preparation, unscaling, and the coordinated optimizer update. Custom distributed
AMP loops use `dabsn.runtime.optimizer.unscale_training_gradients(optimizer,
scaler)` after preparation and before optional clipping; it supports FSDP2's
mixture of distributed tensors and provider-owned Capacity rows.

DDP synchronizes coalesced sparse row contributions at the optimizer boundary.
FSDP, FSDP2, and tensor-parallel providers assign complete Capacity rows to their
owners and keep SparseAdam state with those rows. Recurrent features and
eligibility remain local to each stream. Use the framework's distributed
gradient verification and optional clipping helpers with this optimizer; they
operate on sparse values without converting the gradient to a dense tensor.

Model and graph checkpoints include configuration, addressing buffers, and
parameters. Online checkpoints also record the Capacity contract and carried
state; a different addressing configuration is rejected. Ordinary and prepared
memory carry Capacity features through prompt ingestion and generation.
Read-only queries leave the supplied memory unchanged. Memory payloads containing
Capacity recurrent state use a new version so an older reader cannot silently discard these features;
existing memory files without Capacity remain supported.

## Execution and verification

Reference, native C++ CPU, and Triton CUDA implementations share the same
addressing and recurrence contracts. Native sparse execution selects rows before
key and output projection. Backend selection and observed execution are exposed
through the kernel controls; unsupported dtype/backend combinations raise rather
than silently changing precision.

Native mixed precision retains FP32 accumulation and eligibility, with FP16 or
BF16 projection operands where supported. FP64 remains available for numerical
reference comparisons. Each Capacity subsystem exposes `last_backend` and
`last_precision` after execution so callers can inspect the selected path.

For CUDA graph training, use `make_graphed_train_callable` and the framework
gradient accumulator or optimizer. Sparse replay writes fixed-size row
contributions; gradient finalization coalesces them outside capture. Carried
state must be supplied as tensor arguments or registered module buffers so graph
preparation can restore it after warmup. Actual replay advances that state once.

CUDA correctness, supported mixed precision, multi-GPU behavior, capture, and
performance require execution on the corresponding hardware. A reduced active
row count describes computation, not a measured speedup. Compare warmed
execution separately from compilation, and report optimizer cost and memory
alongside kernel timing.
