# DABSN

**A recurrent modeling framework for persistent state and admitted memory.**

DABSN is a PyTorch architecture for causal sequences, whole fields, and
structured data that mixes both. Its core maintains a nonlinear recurrent
state while its read system combines admitted short memory, successor
induction, permanent associative memory, and a recurrent long-memory channel.
The same block design is used across all three geometries.

The library includes native C++/OpenMP CPU kernels and Triton/CUDA kernels for
forward and backward execution, task-owned input and output adapters, structured
checkpoints, distributed execution, inference and gradient-verification
utilities, and the complete source and result tables for the accompanying paper.

DABSN is also a **composition framework**. A model is an ordered graph of
components with declared contracts, and DABSN is one component in it — so a
sparse mixture of experts, attention, a transformer, a CNN, or an architecture
nobody has written yet composes with DABSN through a registered provider,
without a core edit or a special loader, and saves and reloads through the
ordinary checkpoint path. See [Composing architectures](#composing-architectures).

Multi-GPU execution offers DDP and block-wrapped FSDP with full parameter,
gradient, and optimizer sharding, plus tensor parallelism over the recurrent
hidden dimension and expert parallelism over MoE experts.

### What this is, and what it is not

DABSN is a **model and runtime library, not a training framework.** Every model
is an ordinary `torch.nn.Module`: put it in your own loop, or in torchtitan,
Lightning, or whatever you already use. The library owns the architecture, the
kernels, the component ABI, checkpoints, and distributed execution — the parts
that are specific to DABSN and that you cannot reasonably write yourself.

It deliberately does not own your training loop. Data format, schedule,
optimizer choice, logging cadence, and checkpoint policy are yours. The `train`,
`pretrain`, and `finetune` entry points below are **convenience examples for the
plain dense path**, kept because they are useful and tested — not a supported
training product, and not the way to train a composed architecture. Reach past
them the moment your run needs something they do not do.

## Paper

### One Layer, Both Gaps

**A Persistent-Modulation Recurrence that Generalizes Copy and Tracks
Non-Solvable Group State**

The paper tests one architecture, trained separately per task, against two
regimes commonly treated as opposing requirements:

| Task | Train length | Evaluation length | DABSN result |
| --- | ---: | ---: | ---: |
| Copy, vocabulary 64 | 64 | 3,200 (50x) | 0.961 +/- 0.035, three seeds |
| A5/60 word problem | 256 | 16,384 (64x) | 1.000, two seeds |

These are separately trained models, not one checkpoint reused across tasks.
The paper includes causal ablations of the nonlinear state and read pathways;
the machine-readable tables used for every reported result are included with
the source.

- [Read the paper](https://github.com/BleedingXiko/dabsn/blob/main/paper1/main.pdf)
- [Inspect the paper source and result tables](https://github.com/BleedingXiko/dabsn/tree/main/paper1)
- [Read the architecture contract](https://github.com/BleedingXiko/dabsn/blob/main/ARCHITECTURE.md)

## Installation

Install DABSN into an environment containing the PyTorch build appropriate for
your machine:

```bash
pip install dabsn
```

Turing GPUs such as the GTX 1660 Ti use the final compatible Torch/Triton
combination:

```bash
pip install 'dabsn[cuda-turing]'
```

Python 3.10 or newer and PyTorch 2.6 or newer are required. Linux CUDA builds
of PyTorch provide the matching Triton runtime. Native backend
selection is explicit: `required=True` raises instead of silently switching to
another runtime family.

## Quick start

```python
import torch

from dabsn import DABSNLayerSpec, DABSNModel, dabsn_adamw_param_groups
from dabsn.kernels import enable, status
from dabsn.runtime import train_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enable(device.type, required=True)

model = DABSNModel(
    input_dim=24,
    out_dim=10,
    layers=[
        DABSNLayerSpec(128, 96, "seq"),
        DABSNLayerSpec(192, 128, "seq"),
    ],
    output_adapter="token",
    residual=True,
    mlp_ratio=4.0,
).to(device)

inputs = torch.randn(8, 256, 24, device=device)
targets = torch.randint(0, 10, (8, 256), device=device)
optimizer = torch.optim.AdamW(
    dabsn_adamw_param_groups(model, weight_decay=0.1),
    lr=1e-3,
)

loss = train_step(
    model,
    inputs,
    targets,
    optimizer,
    clip_grad_norm=1.0,
)

print({"loss": loss, "backend": status()["active_backend"]})
```

Every model is a normal `torch.nn.Module`. Use the supplied runtime helpers or
an ordinary PyTorch training loop:
[`examples/minimal_train.py`](https://github.com/BleedingXiko/dabsn/blob/main/examples/minimal_train.py)
is the same step written with no DABSN helpers and no native backend.

`residual=True` makes each block return `skip(x) + dabsn(x)`, using a learned
bias-free projection only when the block changes width. `mlp_ratio=4.0` adds the
fixed post-DABSN update `h + mlp(mlp_rmsnorm(h))`. The normalization exists only
inside that MLP branch and never touches DABSN or the residual trunk. Set
`mlp_ratio=None` (the default) for pure DABSN with no MLP parameters; both new
settings default off so existing checkpoints retain their original architecture.

For a DABSN front/rear pair with ordinary nonlinear processing in between, set
`mlp_middle_depth` to the number of standalone residual MLP blocks and
`mlp_depth_index` to the zero-based DABSN block after which they run. The middle
blocks use the same `mlp_ratio`, RMSNorm, ReLU-squared, bias-free projections,
and zero-initialized output projection as the post-DABSN branch:

```python
from dabsn import DABSNSequenceLM

# DABSN[0] -> 20 MLP blocks -> DABSN[1]
model = DABSNSequenceLM(
    vocab=50_257,
    hidden_dim=768,
    depth=2,
    layers="seq:768:768,seq:768:768",
    residual=True,
    mlp_ratio=4.0,
    mlp_middle_depth=20,
    mlp_depth_index=0,
    tie_embeddings=False,
)
```

`layers` continues to list DABSN blocks only. With a middle depth of zero
(the default), the insertion index has no effect and existing behavior is
unchanged. Carried `.dmem` memory likewise remains one bank per DABSN block;
the stateless middle MLPs add no per-token memory.

## Model geometry

Each layer owns an output width, recurrent-state width, and read geometry.
Widths may change across a stack.

| Geometry | Memory eligibility | Typical structure |
| --- | --- | --- |
| `seq` | causal prefix | language, events, control streams |
| `field` | whole object | images, boards, sets, spatial state |
| `hybrid` | learned sequence/field mixture | structured streams with both relations |

Layer stacks can be written directly or parsed from compact specifications:

```python
from dabsn import parse_dabsn_layer_specs

layers = parse_dabsn_layer_specs(
    "seq:128:96,hybrid:192:128,field:128:96"
)
```

The outer model API is the same for every geometry. Geometry changes memory
eligibility, not the recurrent block or checkpoint format.

## Task adapters

DABSN owns the recurrent body; applications own the meaning of their data.
Input adapters transform raw task records into model-width features, and
output heads transform hidden states into task predictions. Registered
adapters become construction and checkpoint metadata rather than notebook-only
glue.

This example handles industrial telemetry with continuous measurements,
elapsed time, sensor identity, and missingness. Its output jointly predicts an
event class and a log-normal time-to-event distribution.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from dabsn import DABSNLayerSpec, DABSNTaskModel
from dabsn.adapters import register_input_adapter, register_output_head


class TelemetryInput(nn.Module):
    def __init__(self, raw_dim: int, model_dim: int, sensors: int = 32):
        super().__init__()
        if raw_dim != 8:
            raise ValueError("expected five values, elapsed time, sensor ID, and mask")
        self.output_dim = model_dim
        self.sensors = sensors
        self.value_norm = nn.LayerNorm(5)
        self.sensor = nn.Embedding(sensors, 12)
        self.missing = nn.Embedding(2, 4)
        self.register_buffer("frequencies", torch.tensor([1., 2., 4., 8.]))
        self.fuse = nn.Sequential(
            nn.Linear(5 + 8 + 12 + 4, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, x):
        values = self.value_norm(torch.nan_to_num(x[..., :5].float()))
        elapsed = x[..., 5].float().clamp_min(0)
        sensor = x[..., 6].long().clamp(0, self.sensors - 1)
        missing = x[..., 7].long().clamp(0, 1)
        phase = torch.log1p(elapsed).unsqueeze(-1) * self.frequencies
        time = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return self.fuse(torch.cat([
            values, time, self.sensor(sensor), self.missing(missing)
        ], dim=-1))


class EventForecast(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.classes = out_dim - 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.event_logits = nn.Linear(hidden_dim, self.classes)
        self.time_parameters = nn.Linear(hidden_dim, 2)

    def forward(self, hidden):
        hidden = self.norm(hidden)
        return torch.cat([
            self.event_logits(hidden), self.time_parameters(hidden)
        ], dim=-1)

    def unpack(self, output):
        logits = output[..., :self.classes]
        log_time_mean = output[..., -2]
        log_time_scale = F.softplus(output[..., -1]) + 1e-4
        return logits, log_time_mean, log_time_scale


register_input_adapter(
    "telemetry",
    lambda raw_dim, model_dim: TelemetryInput(raw_dim, model_dim or raw_dim),
)
register_output_head("event_forecast", EventForecast)

model = DABSNTaskModel(
    raw_input_dim=8,
    model_input_dim=96,
    out_dim=6,  # four event classes plus two distribution parameters
    layers=[
        DABSNLayerSpec(96, 64, "seq"),
        DABSNLayerSpec(128, 96, "seq"),
    ],
    input_adapter="telemetry",
    output_adapter="event_forecast",
)
```

The [complete telemetry example](https://github.com/BleedingXiko/dabsn/blob/main/examples/custom_adapter.py)
includes synthetic data, the joint classification/distribution loss, and an
optimizer step. A separate [local 2D field adapter](https://github.com/BleedingXiko/dabsn/blob/main/examples/field2d_local_adapter.py)
demonstrates native neighborhood gather/scatter for spatial models.

## Composing architectures

The layer stack above builds "DABSN blocks, optionally with the same dense MLP
after each one." That is the convenience path. Anything else — a sparse mixture
of experts, an attention block, a transformer, a CNN, an SSM, several of them
mixed — is an **ordered graph of components**, and DABSN is one component in it.

Adding an architecture requires no core edit, no DABSN kernel, no `if attention`
branch in the framework, and no special model loader. Components declare what
they accept and produce, the graph validates every edge when the model is built,
and the checkpoint carries enough to rebuild the whole thing.

```python
import torch

from dabsn.checkpoint import save_graph, load_graph
from dabsn.components import BuildContext, ComponentSpec, component_registry
from dabsn.graph import DABSNGraph

component_registry.discover()
context = BuildContext(device=torch.device("cuda"), dtype=torch.bfloat16)

dabsn = component_registry.build(
    ComponentSpec("dabsn.0", "dabsn:block", {
        "input_dim": 768, "hidden_dim": 768, "state_dim": 768,
        "read_geometry": "seq", "residual": True,
    }),
    context,
)
experts = component_registry.build(
    ComponentSpec("moe.0", "dabsn:sparse_moe", {
        "hidden_dim": 768, "experts": 24, "top_k": 4, "inner_dim": 3072,
        "router": "switch", "balance_coefficient": 0.01,
    }),
    context,
)

graph = DABSNGraph([dabsn, experts], require_world_builder=True)
save_graph(graph, "arch.safetensors")
restored = load_graph("arch.safetensors")
```

### Mixture of experts

`dabsn:sparse_moe` routes each item to `top_k` of `experts` and is **dropless**:
exactly `N x K` assignments are produced for `N` items, so the assignment buffer
is a static shape even though per-expert counts vary. No capacity factor and no
token dropping on the default path.

Two routers are selectable, and neither is imposed: `"switch"` carries an
explicit load-balance loss, `"aux_loss_free"` updates a per-expert selection
bias after real optimizer steps. Routing returns expert counts and shares,
balance entropy, cold-expert count, busiest and quietest share, selected
confidence, and output-norm differentiation.

The built-in `inner_dim` experts are grouped ReLU-squared MLPs. **An expert does
not have to be an MLP.** Replace `inner_dim` with `expert_specs` — one provider
spec per expert — and an expert becomes any registered component: an attention
block, a whole transformer, a CNN, another MoE, or your own module. Mixing kinds
within one group is just mixing entries in the list.

```python
attention = {
    "provider_key": "yourpkg.attention", "provider_distribution": "yourpkg",
    "provider_version": "1.0.0", "component_abi_version": 2,
    "config_schema_version": 1,
    "config": {"width": 768, "heads": 12},
}
mlp = {
    "provider_key": "dabsn:residual_mlp", "provider_distribution": "dabsn",
    "provider_version": dabsn_version, "component_abi_version": 2,
    "config_schema_version": 1,
    "config": {"dim": 768, "ratio": 4.0},
}

ComponentSpec("moe.0", "dabsn:sparse_moe", {
    "hidden_dim": 768, "experts": 4, "top_k": 2,
    "router": "switch", "balance_coefficient": 0.01,
    "expert_specs": [attention, mlp, attention, mlp],   # heterogeneous
})
```

Those specs are the checkpoint. A model whose experts are half attention and
half MLP saves and reloads through the ordinary path, because the loader
rebuilds each expert from its provider key rather than from a hardcoded list of
architectures DABSN knows about.

Routing granularity is explicit component configuration. The built-in component
routes individual hidden vectors; structure-native routing belongs to a provider
that declares that contract rather than being silently assumed.

### Writing a provider

A provider owns config validation, the contract its config implies,
construction, and config-schema migration. Registering one makes an architecture
addressable by name, portable in checkpoints, and subject to the same
conformance matrix as the built-ins:

```python
from dabsn import check_component
from dabsn.components import component_registry

component_registry.register(MyProvider(), distribution="mypkg", version="1.0.0")
report = check_component("mypkg.my_component", config, device="cuda")
print(report.to_dict())
```

`check_component` runs the declared capability matrix — build and schema,
dynamic axes, whole-graph compile, export, AMP, streaming state, determinism,
FSDP wrapping. A capability the provider does not claim is reported as a skip,
never as a pass. Loading a checkpoint that names a third-party provider requires
passing it in `trusted_providers`, because reconstruction runs its code.

Note that data-dependent dispatch — sparse routing, where per-expert counts are
known only at runtime — cannot be traced whole-graph or exported. That is a
declared property of such components, not a defect, and the conformance report
says so rather than claiming otherwise.

## Native runtimes

```python
from dabsn.kernels import enable, status

enable("cuda", required=True)   # CUDA/Triton + batched-GEMM forward/backward
# enable("cpu", required=True)  # C++/OpenMP forward and backward
# enable("reference")           # explicit PyTorch reference runtime

print(status())
```

Backend activation is process-wide because it installs model dispatch hooks.
Requested native execution never silently falls back. The status report names
the active implementation for the core scan, admitted read, permanent memory,
long-memory recurrence, and local-field gather.

CUDA training dispatch is execution-shape aware. Small batches use the
persistent Triton scan. Batches of 64 or more use the batched recurrent runtime,
which shares each recurrent-matrix read across the device batch through GEMM;
this changes neither the DABSN equations nor model depth. For modest training
score tensors (up to 8,388,608 `[B,T,N]` entries by default), admitted-read
forward and backward use native BMM instead of pairing a tiled forward with the
older serial-query backward. The controls are explicit when a benchmark needs
to pin them:

```bash
export DABSN_CORE_BACKEND=batched       # auto | batched | persistent | batched_fused
export DABSN_BATCHED_STEP_COMPILE=1     # compile only the pure pointwise step
export DABSN_TRAIN_DENSE_MAX_SCORES=8388608
```

The complete DABSN model or backbone is never compiled by this dispatch. The
batched custom-autograd recurrence has a separately tested explicit backward,
including every parameter and carried-state gradient.

The release gates cover:

- `seq`, `field`, and `hybrid` model forward/backward parity;
- single-block and stacked execution;
- recurrent execution with and without an explicit initial core state;
- gradients through inputs, parameters, and carried state;
- admitted, permanent, long-memory, and local-field primitives;
- configuration-aware checkpoint reload.

The repository does not claim that its fused kernels outperform every existing
sequence runtime. Their contract is native DABSN execution with explicit
forward/backward parity and no hidden backend switch.

## Performance and scaling

Most of the throughput-relevant behavior is automatic once a native runtime is
enabled; nothing below changes the DABSN equations. All of it is geometry
agnostic (`seq`, `field`, `hybrid`) because it lives at the core-scan and
admitted-read level, not in any task head.

**Automatic (no configuration):**

- **Sub-quadratic admitted read.** The read scores each query position against
  the *admitted* bank, whose width is the data-dependent admitted count, so the
  cost is `O(T * admitted)` — not `O(T^2)`. The width is sized dynamically from
  the learned admission; it only approaches `seq_len` (quadratic) if the model
  genuinely learns to admit almost every position, which is the correct cost for
  a task that needs it. Inference and ordinary GPU training both use this
  dynamic width. A static width is used only while a CUDA graph is actively being
  captured (where a host sync is illegal), and even then the capture path pins a
  measured, padded, still-sub-quadratic cap.
- **Work-aware core dispatch.** `select_core_backend` picks the persistent Triton
  scan for small work and the batched tensor-core GEMM scan once the batch is
  large enough to fill the device (`B >= 64` or `B*H >=
  DABSN_BATCHED_CORE_MIN_WORK`, default 4096), so wide/large-batch training uses
  tensor cores automatically.
- **Tensor-core compute dtype.** With a BF16/FP16 model the recurrent GEMMs run
  on tensor cores; pointwise state stays FP32.

**Opt-in:**

- **Fused single-launch core scan** (`DABSN_CORE_BACKEND=batched_fused`, hidden
  width `<= 256`). Runs the whole `T`-step scan in one Triton launch with state
  carried in registers, removing the per-step launch overhead. Wider cores use
  the batched per-step GEMM, which has no such width bound. `auto` never selects
  the fused backend on its own — request it explicitly.
- **CUDA-graph training** (`make_graphed_train_callable`, or `cuda_graph=True` in
  `DABSNPretrainConfig`). Captures the forward+backward once and replays it,
  removing kernel-launch overhead — the dominant cost of the sequential scan at
  small microbatches. Single-process CUDA only; pair each replay with
  `ManualGradientAccumulator` for exact microbatch accumulation. Capture failure
  raises rather than silently degrading.

```python
from dabsn.runtime import make_graphed_train_callable, ManualGradientAccumulator
```

**Batch vs. context.** The core is a *sequential* recurrence: it advances one
position at a time and cannot parallelize across context the way attention does.
Its device parallelism therefore comes from the **batch**, not the sequence
length — a tiny microbatch leaves the GPU idle on every step regardless of
context. Raise the microbatch as high as memory allows and use gradient
accumulation for the effective batch. Because the read is sub-quadratic in `T`,
context length scales close to linearly, so long-context training is bounded by
the (linear) number of scan steps rather than a quadratic read.

## Gradient preflight

Before a long training run, verify the complete model stack:

```python
from dabsn.runtime import verify_gradients

rows = verify_gradients(model, sample_input, compile_forward=True)
print(rows)
```

This compiles the outer forward boundary, runs one backward pass, and raises if
any block has missing, zero, or non-finite representative gradients.

## Distributed training

Launch two or more CUDA workers with `torchrun` and select FSDP explicitly:

```bash
torchrun --standalone --nproc-per-node=2 -m dabsn.cli train \
  --config model.json \
  --data batch.pt \
  --output run/model.safetensors \
  --device cuda \
  --backend cuda \
  --distributed fsdp \
  --precision bf16 \
  --grad-checkpoint \
  --grad-accum-steps 4 \
  --verify-gradients
```

Use `--precision fp16` on Turing GPUs such as the T4 or GTX 1660 Ti. The input
file contains one global batch; its first dimension must be divisible by the
number of workers. Each rank receives a distinct batch shard. FSDP uses
`FULL_SHARD`, wraps each `DABSNBlock`, retains original parameters for the
optimizer, and uses the FSDP-aware gradient scaler and global gradient clip.

Portable mode writes a self-describing SafeTensors model to
`run/model.safetensors`. Optimizer, AMP scaler, and completed-step state are
stored in the trusted local sidecar
`run/model.safetensors.optimizer.pt`. Add `--resume` to continue the same run.
Resume rejects a missing model or sidecar instead of silently starting over.

For a checkpoint too large to gather on rank zero, use distributed checkpoint
mode:

```bash
torchrun --nnodes=2 --nproc-per-node=8 \
  --rdzv-id=dabsn-pretrain-01 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=trainer-0.example:29400 \
  -m dabsn.cli train \
  --config model.json \
  --data batch.pt \
  --output run/checkpoint \
  --device cuda --backend cuda --distributed fsdp --precision bf16 \
  --checkpoint-mode sharded --steps 10000 --resume
```

The sharded directory contains reshardable model and optimizer files plus
`dabsn-training.json`. It avoids a full rank-zero state gather. If a complete
model can fit in rank-zero host memory, add
`--final-export run/model.safetensors` to consolidate a shareable inference
file.

FSDP is parameter, gradient, optimizer-state, and batch parallelism. It does not
split one sequence or one oversized matrix across GPUs.

### Tensor and expert parallelism

Two further kinds ship, for the cases FSDP does not answer.

**Tensor parallelism** splits the recurrent hidden dimension itself, so a core
wider than one device still runs. Each worker owns a contiguous set of state
units and the matching rows of the recurrent matrix:

```python
from dabsn.core import TensorParallelDABSNCore

sharded = TensorParallelDABSNCore(core, group=group, rank=rank, world_size=world)
```

Because a unit's next state depends on every unit's current state, the
recurrence exchanges the full activation at every step. That collective is
required by the recurrence, not an implementation shortcut. It is fused with the
recurrent matmul into a single autograd node, using a symmetric-memory
collective where the interconnect supports it and an explicit gather elsewhere;
both compute the same values. Reassemble per-rank trajectories with
`reassemble_tensor_parallel_trajectory`.

**Expert parallelism** places MoE experts on different workers and exchanges
routed assignments by rank:

```python
from dabsn import ExpertParallelExpertGroup, GenericExpertGroup

group = ExpertParallelExpertGroup(
    GenericExpertGroup(local_experts),
    process_group=process_group,
    world_size=world,
    rank=rank,
)
```

Every assignment returns in its original order, so a sharded expert group is
numerically indistinguishable from one unsharded model. Its variable all-to-all
split sizes are incompatible with CUDA-graph capture unless a separately proven
static communication plan is supplied.

Pipeline and context parallelism do not ship, so this repository does not claim
that it can train an arbitrary one-trillion-parameter configuration.

Programmatic users can access the same implementation through
`setup_distributed`, `prepare_distributed_model`, `save_distributed_dabsn`,
`save_sharded_training_checkpoint`, and their matching load functions from
`dabsn.runtime`. `clip_grad_norm` from `dabsn.runtime` clips correctly under
every arrangement above, accumulating the cross-worker sum of squares in FP64
and returning the norm in the gradients' dtype regardless of topology.

## Checkpoints and export

```python
from dabsn import load_dabsn, save_dabsn
from dabsn.runtime import export_dabsn

save_dabsn(model, "model.safetensors")
restored = load_dabsn("model.safetensors", map_location="cpu")

export_dabsn(model, "weights.safetensors", format="safetensors")
export_dabsn(
    model,
    "program.pt2",
    sample_input=sample_input,
    format="torch-export",
)
```

Model checkpoints are atomic, non-pickle SafeTensors files. Saving validates the
schema, checks tensor/metadata consistency, writes a temporary file, flushes and
synchronizes it, and atomically replaces the target. `artifact_digest` returns a
SHA-256 over the finished file.

An artifact carries the format and schema version, the complete ordered graph
specification, stable component IDs, provider keys with their distributions,
versions and configuration, a representation-contract fingerprint, the parameter
namespace map, the tied/shared tensor map, DABSN memory ownership, and
construction and framework versions. Metadata is canonical JSON under explicit
size, nesting, and value limits. `inspect_dabsn` reads all of it without
allocating a single model tensor.

Loading rebuilds the architecture from provider keys and configuration and
verifies the contract fingerprint before applying any tensor. **The loader
contains no architecture-specific branch** — that absence is what makes an
arbitrary composed model portable rather than dependent on the code that
happened to create it. Use `load_graph` for a raw component graph and
`load_dabsn` for a model; each rejects the other's artifact kind rather than
guessing.

Custom adapters and third-party providers remain application-owned and must be
registered — and named in `trusted_providers` — before loading a checkpoint that
references them, because reconstruction executes their construction code.
Migration is explicit: `migrate_dabsn_checkpoint` converts v1 artifacts to the v2
graph form, and every existing `0.1.x` checkpoint remains loadable. Optimizer
sidecars and distributed training directories are trusted run state, not files to
accept from an untrusted source.

## Train, pretrain, fine-tune, and resume (examples)

**Scope.** These are worked examples for the plain dense DABSN path, not a
training framework. They build a `DABSNSequenceLM` from flat configuration
fields, so they cannot construct a composed architecture — a mixture of experts,
attention experts, or anything else assembled as a component graph. For those,
build the model yourself (see [Composing architectures](#composing-architectures))
and train it in your own loop; everything the library actually owns —
checkpoints, distributed execution, kernels, gradient checks — works there
unchanged.

They are kept because they are tested and genuinely useful for the case they
cover, and because `--resume` will continue any `DABSNSequenceLM` checkpoint,
including one you built from a graph. They are not the intended path for a
serious run.

These commands are deliberately separate:

- `train` creates a new model from `model.json` and prepared input/target
  tensors. `train --resume` continues that exact run with its optimizer and
  completed step.
- `finetune` loads model weights but intentionally creates a new optimizer and
  starts at step zero. Its output must differ from its input checkpoint.
- `pretrain` builds a `DABSNSequenceLM` and learns next-token prediction from a
  token corpus. A binary corpus is memory-mapped: batches are sliced from disk
  without loading the entire corpus into RAM.

`AMP` means automatic mixed precision (`fp16` or `bf16`). It reduces tensor
memory and compute cost while the supplied scaler protects fp16 gradients.
Gradient accumulation divides each update across several smaller batches.

A minimal pretraining config is:

```json
{
  "corpus_bin": "/data/tokens.uint16",
  "corpus_dtype": "uint16",
  "vocab": 50257,
  "hidden_dim": 768,
  "depth": 12,
  "layer_geometries": ["seq"],
  "train_context": 2048,
  "steps": 16000,
  "batch_size": 4,
  "precision": "bf16",
  "distributed": "fsdp",
  "grad_checkpoint": true,
  "grad_accum_steps": 8,
  "checkpoint_every": 1000
}
```

Launch it with:

```bash
torchrun --standalone --nproc-per-node=8 -m dabsn.cli pretrain \
  --config pretrain.json \
  --output run/checkpoint \
  --device cuda --backend cuda \
  --checkpoint-mode sharded \
  --final-export run/model.safetensors \
  --verify-gradients
```

`steps` counts corpus microsteps, matching the canonical training loop. One
optimizer update occurs every `grad_accum_steps`; `checkpoint_every` must land
on an update boundary. The checkpoint records every rank's corpus RNG stream.
Bitwise data-stream continuation therefore requires the same worker count.
Changing the worker count may reshard model/optimizer state, but it is a new
global data trajectory and is not called exact continuation.

Fine-tuning uses a prepared tensor payload and a fresh output path:

```bash
dabsn finetune \
  --checkpoint base.safetensors \
  --data task-batch.pt \
  --output task-model.safetensors \
  --device cuda --backend cuda --precision bf16 --steps 2000
```

## Language modeling

```python
from dabsn import DABSNSequenceLM

model = DABSNSequenceLM(
    vocab=50_257,
    hidden_dim=512,
    depth=4,
    layers="seq:256:256,seq:768:512,seq:768:512,seq:256:256",
    tie_embeddings=False,
)

logits = model.forward_sequence(token_ids)
```

## CLI and reproductions

```bash
dabsn --help
dabsn kernels --enable cuda --required
dabsn doctor

dabsn-reproduce-copy --help
dabsn-reproduce-mqar --help
dabsn-reproduce-keyvalue --help
dabsn-reproduce-a5 --help
```

The full reproduction defaults correspond to the checked-in result tables.
Reduced settings are available for local execution checks and are not presented
as replacements for the reported experiments.

## Development

```bash
git clone https://github.com/BleedingXiko/dabsn.git
cd dabsn
pip install -e '.[test]'
pytest
```

Native release gates are available for a fresh wheel-installed checkout:

```bash
bash tools/cpu_check.sh
bash tools/gpu_check.sh
bash tools/fsdp_check.sh  # requires two NVIDIA GPUs
```

## Citation

If DABSN or its native runtimes contribute to your work, cite the paper:

```bibtex
@misc{rosdahl2026onelayer,
  title     = {One Layer, Both Gaps: A Persistent-Modulation Recurrence that
               Generalizes Copy and Tracks Non-Solvable Group State},
  author    = {Rosdahl, Nicholas},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21391204},
  url       = {https://github.com/BleedingXiko/dabsn}
}
```

[`CITATION.cff`](https://github.com/BleedingXiko/dabsn/blob/main/CITATION.cff)
carries the same metadata in machine-readable form, and GitHub's "Cite this
repository" control reads it directly.

## License

DABSN is released under the [Apache License 2.0](https://github.com/BleedingXiko/dabsn/blob/main/LICENSE).
