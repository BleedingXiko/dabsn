# DABSN architecture contract

## Tensor shapes

- Sequence input: `[batch, steps, input_dim]`.
- Flat field input: `[batch, cells, input_dim]`.
- Spatial field input: `[batch, height, width, input_dim]`.
- Spatial-temporal field input:
  `[batch, height, width, steps, input_dim]`.
- A block returns one hidden vector per input position. Spatial inputs are
  restored to their spatial layout after the backbone.

Task-specific input adapters transform raw tensors into one of these layouts.
Output heads transform the final hidden width into task predictions.

## Block composition

Each `DABSNBlock` contains:

1. `DABSNCore`, a recurrent novelty-budget-plasticity system.
2. `DABSNRead`, an admitted read over committed writes and carried memory.
3. A learned read gain and an optional state-width projection.
4. An optional stack residual and optional post-DABSN residual MLP.

The core emits the recurrent trajectory `cat[y, budget]` together with novelty,
plasticity, expression, committed write, energy, and saturation signals. The
read combines admitted short memory, successor induction, permanent associative
memory, predictive expectation, and recurrent long memory. A block returns:

Let `d = state_to_hidden(y + read_gain * read)`. With both optional additions
enabled, a block returns:

```text
h   = skip(x) + d
out = h + fc2(relu(fc1(mlp_rmsnorm(h))) ** 2)
```

`skip` is identity at equal widths and a bias-free learned projection when a
block changes width. `mlp_rmsnorm` belongs only to the MLP branch: the DABSN
input, recurrence, read, output, and residual trunk are never normalized.
`fc2` is zero-initialized, so the MLP branch is exactly an identity update at
initialization. With `mlp_ratio=None`, no MLP or normalization parameters exist;
with `residual=True` that leaves the pure `skip(x) + dabsn(x)` block.

`DABSNBackbone` stacks blocks with independently selected hidden widths,
recurrent-state widths, and read geometries. `residual` and `mlp_ratio` are
model-level settings shared by every block; there are no per-block MLP modes.

## Read geometry

Geometry changes memory eligibility, not the core recurrence:

- `seq`: each position can read its causal prefix.
- `field`: each position can read the complete object or field.
- `hybrid`: a learned gate mixes sequence and field reads.

Admission, induction, permanent memory, predictive expectation, retention, and
long memory are shared across all three geometries.

## Carried state

`DABSNCore.forward_from_state` accepts and returns the core recurrence's budget,
energy, and saturation tensors. This is a core-only carry boundary used by the
reference, native CPU, and native CUDA implementations; it is not a complete
streamed-block API. The public backbone does not currently carry the read's admitted writes,
permanent memory, predictive expectation, retention, and long-memory state
between separate calls. Those read states persist only within one full forward
pass.

The C++/OpenMP and Triton scans accept an external core state, return the final
core state, and propagate gradients through that boundary. The release gate
compares one full scan with two state-connected scans for outputs, final state,
input gradients, initial-state gradients, and parameter gradients.

## Runtime contract

The public model API is identical across three runtime families:

- `reference`: eager PyTorch.
- `cpu`: C++/OpenMP core, admitted-read, long-memory, permanent-memory, and
  local-field kernels with backward support.
- `cuda`: Triton core, admitted-read, long-memory, permanent-memory, and local
  field kernels with backward support.

Call `dabsn.kernels.enable(backend, required=True)` before model execution to
require a native runtime. Backend selection is process-wide because activation
installs dispatch hooks on the core and read classes.

## Package map

- `dabsn.config`: layer specifications and model configuration.
- `dabsn.adapters`: built-in adapters and registration APIs.
- `dabsn.core`: recurrence and carried-state execution.
- `dabsn.read`: admitted, permanent, long, unified, and local-field reads.
- `dabsn.model`: blocks, backbones, task models, and language models.
- `dabsn.checkpoint`: self-describing SafeTensors model save and load.
- `dabsn.kernels`: runtime selection, native primitives, and status reporting.
- `dabsn.runtime`: training, evaluation, inference, export, and gradient checks.
- `dabsn.distributed`: DDP/FSDP setup, DABSN-block sharding, mixed precision,
  gradient accumulation, portable SafeTensors export, and reshardable
  distributed training checkpoints.
- `dabsn.pretrain`: mmap-backed next-token pretraining and exact corpus-stream
  resume.

## Distributed execution

FSDP runs under `torchrun` with `FULL_SHARD`, `use_orig_params=True`, and an
auto-wrap policy over `DABSNBlock`. It shards parameters, gradients, and
optimizer state while distributing independent batch examples across workers.
It is not context, tensor, or pipeline parallelism: one sequence and each
wrapped block remain executable on one worker. Portable save collectively
materializes the full model and optimizer on rank zero. Sharded save uses
PyTorch Distributed Checkpoint so model and optimizer state remain distributed
and may be resharded at load. Consolidating that state to one SafeTensors file
still requires enough rank-zero host memory for the complete model.
