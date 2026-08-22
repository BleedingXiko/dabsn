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

The inline MLP is a convenience, not the composition mechanism. Anything richer
than "the same dense MLP after every DABSN" — a sparse mixture of experts, an
attention block, a transformer, a CNN, an SSM, a mixed stack, or an architecture
that does not exist yet — is expressed as further components in the ordered
graph rather than as a mode inside `DABSNBlock`. The block gains no branch for
each new architecture, which is the property that keeps the core stable while
the search space stays open.

## Optional interposed MLP tower

`mlp_middle_depth` inserts that many ordinary residual MLP blocks after the
DABSN block selected by zero-based `mlp_depth_index`. The configured DABSN list
remains DABSN-only: with two blocks and an index of zero, execution is
`DABSN[0] -> middle MLPs -> DABSN[1]`. The tower uses the selected block's output
width, which is already the following DABSN block's input width, and uses the
same `mlp_ratio` contract above. At depth zero it is absent without changing
existing parameters or execution.

The tower is stateless. `dabsn.memory` replays it at the same boundary during
carried inference, but `.dmem` continues to store exactly one carried state and
read bank per configured DABSN block.

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

## Composition contract

A model is an ordered graph of components. `DABSNBlock` is one component among
several, not a container the rest of the architecture is bolted onto. This is
what makes an architecture describable rather than hand-assembled, and it is the
reason adding attention, a CNN, an SSM, a transformer, or a nested mixture of
experts requires no core edit, no DABSN kernel, no `if attention` branch in the
framework, and no special model loader.

A component declares, and is held to:

- an input and output `ValueContract` — named axes with fixed or dynamic
  extents. The graph validates every producer/consumer edge at construction, so
  an incompatible pair fails when the model is built rather than mid-training.
- `ComponentCapabilities` — `eager`, `compile_fullgraph`, `dynamic_shapes`,
  `export`, `cuda_graph`, `activation_checkpoint`, `amp_fp32`, `amp_bf16`,
  `amp_fp16`, `distributed`, `streaming_state`, `world_builder`,
  `dabsn_memory_owner`, `deterministic`, `parallel_plan`. A capability is a
  claim the conformance kit tests, not documentation.
- explicit training results — loss terms, reports, and carried state are
  returned through `ComponentOutput` and declared by arity. A component that
  declares a loss term and returns none fails loudly; this is what keeps an
  auxiliary loss from being silently dropped on its way to `backward()`.

Components are produced by registered providers. A provider owns config
validation, the contract it implies, construction, and config-schema migration.
`ComponentSpec` (`component_id`, `provider_key`, `config`, ABI version, schema
version, distribution, version) is the complete portable description of one
component. Component ABI version is 2.

Only the DABSN component consumes the language-input representation and builds
the recurrent world representation. Downstream components receive that world
output — never embeddings, token IDs, DABSN memory banks, or a hidden bypass.
`.dmem` cardinality follows DABSN components only; ordinary downstream
components never create DABSN memories.

## Conformance

A provider's declared capabilities are verified by a fixed matrix rather than
trusted: `config-schema-and-build`, `dynamic-axis-contract`, `compile-fullgraph`,
`torch-export`, `amp-bf16`, `amp-fp16`, `streaming-state-carry`,
`determinism-declaration`, `fsdp-wrapping`, and the fake-tensor and CUDA-graph
checks. A capability a provider does not declare is reported as a skip, never as
a pass. Data-dependent dispatch — sparse MoE routing, where per-expert counts are
known only at runtime — legitimately fails whole-graph tracing; that is a
declared limitation of the component, not a framework defect, and the honest
declaration is what keeps the matrix meaningful.

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

## Persistence contract

A checkpoint is a self-describing, non-pickle SafeTensors artifact carrying the
format name and schema version, the complete ordered graph specification, stable
component IDs, provider keys with their distributions and versions and
configuration, a representation-contract fingerprint, the parameter namespace
map, the tied/shared tensor map, DABSN memory ownership, and construction and
framework versions. Metadata is canonical JSON under explicit size, nesting, and
value limits.

Saving validates the schema, checks tensor and metadata consistency, writes to a
temporary file, flushes and synchronizes it, and atomically replaces the target.

Loading reconstructs the graph from provider keys and configuration through the
registry and verifies the contract fingerprint before any tensor is applied.
**There is no architecture-specific branch in the loader.** That absence is the
contract: if reconstructing some architecture required the loader to know what it
is, that architecture would not be portable. Third-party providers must be named
as trusted at load time, because reconstruction executes their construction code.

Migration is explicit rather than loose defaulting — v1 artifacts migrate to the
v2 graph form, and a legacy inline `mlp_ratio` migrates to a residual MLP after
each corresponding DABSN. Every existing `0.1.x` artifact remains loadable.

## Package map

- `dabsn.config`: layer specifications and model configuration.
- `dabsn.adapters`: built-in adapters and registration APIs.
- `dabsn.core`: recurrence, carried-state execution, and the tensor-parallel core.
- `dabsn.read`: admitted, permanent, long, unified, and local-field reads.
- `dabsn.memory`: carried `.dmem` state and bank replay across calls.
- `dabsn.model`: blocks, backbones, task models, and language models.
- `dabsn.components`: the component ABI — value contracts, capabilities,
  declared results, specs, bindings, the provider registry, and parallel plans.
- `dabsn.providers`: the built-in providers (`dabsn:block`,
  `dabsn:residual_mlp`, `dabsn:sparse_moe`).
- `dabsn.graph`: the ordered component graph and its edge validation.
- `dabsn.moe`: routers, expert groups, dropless dispatch, and the sparse MoE
  component.
- `dabsn.ops`: registered custom operators used by dispatch.
- `dabsn.conformance`: the public capability conformance matrix.
- `dabsn.events`: structured observability and strict fallback semantics.
- `dabsn.benchmarking`: hardware fingerprints, timed reports with confidence
  intervals, and operator traces.
- `dabsn.checkpoint`: self-describing SafeTensors save, load, and migration.
- `dabsn.kernels`: runtime selection, native primitives, and status reporting.
- `dabsn.runtime`: training, evaluation, inference, export, CUDA-graph capture,
  and gradient checks.
- `dabsn.distributed`: DDP/FSDP setup, DABSN-block sharding, tensor and expert
  parallelism, mixed precision, gradient accumulation, portable SafeTensors
  export, and reshardable distributed training checkpoints.
- `dabsn.pretrain`: mmap-backed next-token pretraining and exact corpus-stream
  resume.

## Distributed execution

Four parallelism kinds ship, and they answer different questions.

**FSDP** runs under `torchrun` with `FULL_SHARD`, `use_orig_params=True`, and an
auto-wrap policy over `DABSNBlock`. It shards parameters, gradients, and
optimizer state while distributing independent batch examples across workers.
One sequence and each wrapped block remain executable on one worker. **DDP**
replicates the model and reduces gradients across batch shards.

**Tensor parallelism** splits the recurrent hidden dimension itself:
`TensorParallelDABSNCore` gives each worker a contiguous set of state units and
the matching rows of the recurrent matrix. Because a unit's next state depends on
*every* unit's current state, the recurrence gathers the full activation at every
step; that per-step collective is required by the mathematics, not an
implementation choice. The gather and the recurrent product are one autograd
node, dispatching to a fused symmetric-memory collective where the interconnect
supports it and to an explicit gather-then-matmul everywhere else — the same
arithmetic either way. The backward is a reduce-scatter: every worker holds a
partial gradient for every unit, so omitting it yields a forward that matches
exactly and gradients that are quietly incomplete.

**Expert parallelism** distributes MoE experts across workers and exchanges
assignments by rank, returning every assignment in its original order. Variable
all-to-all split sizes make this path incompatible with CUDA-graph capture
unless a separately proven static communication plan is supplied.

Pipeline and context parallelism do not ship.

Portable save collectively materializes the full model and optimizer on rank
zero. Sharded save uses PyTorch Distributed Checkpoint so model and optimizer
state remain distributed and may be resharded at load. Consolidating that state
to one SafeTensors file still requires enough rank-zero host memory for the
complete model.
