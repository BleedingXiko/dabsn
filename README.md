# DABSN

**Persistent state, admitted memory, parametric Capacity, and online learning.**

DABSN is a PyTorch framework built around a nonlinear recurrent core and
content-addressed memory. A block learns persistent state and selectively admits
writes to memory. Native **Capacity** adds dense or geometrically sparse
recurrent features inside that block, with separate controls for total stored
rows and rows active per input. Online temporal credit carries learning state
beyond the retained computation graph.

Use a DABSN block directly, compose it with other processing, or use
`DABSNHostMemory` to bring the admitted read into an existing architecture.

A transformer with one DABSN memory branch and no positional encoding has no
architectural context cap. Trained at 256 tokens, the paper-3 models retain
**96.9% MQAR accuracy at 64× the training length**, **100% at 8× the stored
pairs**, and **96–100% passkey digit accuracy at 1,048,576 tokens (4096×)**
across three seeds. [Results and limits](#papers).

The framework includes native CPU and CUDA forward/backward kernels, component
contracts, task adapters, SafeTensors checkpoints, and distributed execution.
Models are ordinary `torch.nn.Module` objects; applications own their training
loops, data, optimizers, and schedules.

## Install

Requires Python 3.10+ and PyTorch 2.6+. Install the PyTorch build appropriate for
your machine, then:

```bash
pip install dabsn
```

For Turing GPUs such as T4 and GTX 1660 Ti:

```bash
pip install 'dabsn[cuda-turing]'
```

This pins PyTorch 2.6.0 and Triton 3.2.0 on Linux. Use FP16 on Turing; the
persistent core is disabled on this pair and the batched core runs instead.
The exact software pins and sm_75 kernel compilation were tested; physical T4
execution was not.

## Quick start: one block with sparse Capacity

```python
import torch
from dabsn import CapacityConfig, build_dabsn
from dabsn.kernels import enable
from dabsn.runtime import build_training_optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enable(device.type, required=True)

model = build_dabsn(
    input_dim=8,
    out_dim=3,
    hidden_dim=32,
    state_dim=24,
    depth=1,
    read_geometry="seq",
    output_adapter="token",
    capacity=CapacityConfig(mode="sparse", size=256, active_units=16),
).to(device)

inputs = torch.randn(2, 16, 8, device=device)
targets = torch.randint(0, 3, (2, 16), device=device)
optimizer = build_training_optimizer(model, lr=1e-3, weight_decay=0.0)

optimizer.zero_grad(set_to_none=True)
logits = model.forward_sequence(inputs)
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
loss.backward()
optimizer.step()
```

This constructs one DABSN block with 256 Capacity rows, 16 active per input.
The combined optimizer prepares sparse row gradients during `step()`.
The dimensions and optimizer settings illustrate the API. See
[Capacity training](examples/capacity_train.py) and
[minimal training without Capacity](examples/minimal_train.py).

## Inside a DABSN block

A block has distinct input, output, and recurrent-state widths. The core
maintains baseline, energy, and saturation state. Read combines admitted content
memory, successor induction, permanent associative memory, and a recurrent
long-memory channel. Optional Capacity contributes alongside Read before the
state-to-output-width projection:

```text
input ──┬── Core ──┬── expressed state ────────────────┐
        │          └── Read (admitted + long memory) ──┼── sum → projection
        └── Capacity (dense or sparse) ────────────────┘
```

| Geometry | Read eligibility | Intended structure |
| --- | --- | --- |
| `seq` | Causal prefix | Language, events, control streams |
| `field` | Whole object | Images, boards, sets |
| `hybrid` | Learned sequence/field mixture | Structured streams |

`DABSNModel` handles continuous features, `DABSNSequenceLM` adds token
embeddings and a language-model head, and `DABSNTaskModel` uses registered
application-owned adapters. `residual=True` adds a block residual;
`mlp_ratio` adds a residual MLP branch. The
[architecture contract](ARCHITECTURE.md) describes the construction options.

## Capacity

Capacity expands a block's learned recurrent features while keeping the DABSN
core width and number of DABSN scans unchanged. Its keys, biases, retention
parameters, and output values are learned model parameters. Each feature has a
recurrent value; selected features update from the current input and contribute
to the block output.

| Mode | Configuration | Execution |
| --- | --- | --- |
| Dense | `CapacityConfig(mode="dense", size=4096)` | Updates and reads every row |
| Sparse | `CapacityConfig(mode="sparse", size=4096, active_units=64)` | Updates and reads exactly 64 geometrically selected rows per input |

`size` controls total Capacity rows; `active_units` controls active rows in
sparse mode. Fixed geometric addressing selects rows before the native key and
output projections. Addressing is reproducible through `address_seed`, with
flat and product addressing available. There is no learned router. Unselected
features and eligibility rows retain their state.

Total parameter and recurrent-state storage grows with `size`; sparse execution
uses the selected rows. Output weights start at zero. Capacity is optional and
works with `seq`, `field`, and `hybrid` Read geometries. Omitting its
configuration creates no Capacity parameters.

Both BPTT and online temporal credit are supported, with reference, C++ CPU,
and Triton CUDA paths. Use `build_training_optimizer`: sparse Capacity rows use
SparseAdam with zero weight decay, while other parameters use AdamW. In custom
AMP or clipping loops, call `prepare_gradients()` after accumulation and before
unscaling or clipping. Model checkpoints preserve addressing and parameters;
online checkpoints also preserve recurrent features and eligibility.

See the [Capacity API and state contract](docs/capacity.md),
[training example](examples/capacity_train.py), and
[measured native throughput](#native-execution).

## Composition

For mixed architectures, `DABSNGraph` composes registered components with
declared input/output contracts. DABSN, MLPs, sparse MoEs, and third-party
attention, CNN, or SSM components share the same graph and checkpoint path:

```python
from dabsn.components import BuildContext, ComponentSpec, component_registry
from dabsn.graph import DABSNGraph

context = BuildContext(device=device, dtype=torch.float32)
specs = [
    ComponentSpec("state", "dabsn:block", {
        "input_dim": 128, "hidden_dim": 128, "state_dim": 96,
        "read_geometry": "seq", "residual": True,
    }),
    ComponentSpec("experts", "dabsn:sparse_moe", {
        "hidden_dim": 128, "experts": 8, "top_k": 2, "inner_dim": 512,
        "router": "switch", "balance_coefficient": 0.01,
    }),
]
graph = DABSNGraph(
    [component_registry.build(spec, context) for spec in specs],
    require_world_builder=True,
)
```

The built-in MoE is dropless and routes each hidden vector independently:
`[B,T,H] → [B*T,H]`. Built-in experts are grouped ReLU-squared MLPs;
registered `expert_specs` may supply other architectures under the
two-axis routed-item contract. Routing is data-dependent, so whole-body
compilation and CUDA-graph capture require capabilities explicitly declared by
the component.

See the [architecture contract](ARCHITECTURE.md),
[component ABI](docs/dabsn-2.0/component-abi.md),
[MoE contract](docs/dabsn-2.0/native-moe.md),
[custom task adapter](examples/custom_adapter.py), and
[local 2D field adapter](examples/field2d_local_adapter.py).

## Memory for an existing host

`DABSNHostMemory` is the canonical DABSN read behind two projections and four
scalar tags derived from host latents. It instantiates no DABSN core.

```python
from dabsn import DABSNHostMemory

h = torch.randn(2, 256, 128)  # [batch, positions, host width]
norm = torch.nn.LayerNorm(128)
memory = DABSNHostMemory(dim=128)
h = h + memory(norm(h))
```

The module returns only the memory branch; the host chooses its residual,
normalization, and placement. Query/write projections start as identity maps,
and the learned admission window starts at 1.5. The default geometry is causal
`seq`; `field` and `hybrid` are also supported. The graph provider
`dabsn:host_memory` adds the residual at the composition boundary.

Admission depends on content. At fixed width, reading a sequence costs
`O(T * N)` for `T` positions and `N` admitted entries; querying an existing
bank costs `O(N)`. The bank can grow with history, and admitting a constant
fraction of positions makes sequence read work quadratic.

## Online learning and carried state

`temporal_credit="online"` carries architecture-specific eligibility across
bounded autograd chunks. `"bptt"` is the default. For batch size `B`, input
width `I`, and state width `S`, the core's carried learning object `E`
contains exactly `B * (2*S*I + 2*S² + 60*S)` scalars, independent of elapsed
history. MLPs, MoEs, projections, and heads use ordinary autograd within chunks.

This structured rule is not generic exact RTRL or full-stack BPTT gradient
equivalence. Stable stream identities and explicit lane resets are required
when carrying state between batches. Eligibility defaults to FP32 for
FP16/BF16 recurrent storage; the separate admitted read bank can still grow.

Prepared inference memory computes historical admission once, then supports
queries and extensions over exact admitted-event pages. It introduces no
fixed top-k, eviction, or lossy compression. This `.dmem` path currently
supports the sequence-geometry, legacy-constructor memory contract.

See [online training, exact resume, and prepared memory](docs/online-temporal-credit-and-prepared-memory.md)
for executable examples and state ownership.

## Native execution

```python
from dabsn.kernels import enable, status

enable("cuda", required=True)   # Triton/CUDA
# enable("cpu", required=True)  # C++/OpenMP
# enable("reference")          # PyTorch reference
print(status())
```

Backend activation is process-wide. Required native execution raises rather
than silently switching runtime families. CUDA paths include a persistent
core scan and BPTT reverse, fused online replay, a fused three-way admitted
read, coupled long-memory scans, and grouped Capacity GEMMs. Read backward
recomputes scores without materializing a full `[B,T,N]` score tensor.

On supported configurations, fused kernels use deterministic reductions
without data atomics and support CUDA-graph capture. BPTT core GEMMs, fused
reads, and grouped Capacity GEMMs use 16-bit operands under FP16/BF16 autocast
with FP32 accumulation. The online core retains FP32 operands and tapes.
FP32 GEMMs follow `torch.get_float32_matmul_precision()`.

```bash
export DABSN_TRAIN_READ_BACKEND=fused    # fused | dense_chunked | dense | flash
export DABSN_INFER_READ_BACKEND=fused    # fused | auto
export DABSN_LONG_BACKEND=fused          # fused | linrec
export DABSN_CORE_PERSISTENT=1           # 0 restores earlier core routes
```

Component measurements on an **NVIDIA A40**, forward + backward without an
optimizer update, under BF16 autocast (online core operands remain FP32):

| Component | Shape | Tokens/s | Reference at the same shape |
| --- | --- | ---: | ---: |
| Core, BPTT | H=768, B=16, T=2048 | 800,577 | cuDNN LSTM: 560,831 |
| Core, online credit | H=768, B=16, T=2048 | 576,030 | — |
| Core, BPTT | H=256, B=64, T=512 | 4,134,689 | cuDNN LSTM: 1,668,203 |
| Capacity, BPTT | B=8, T=2048, C=32768, A=1024, D=S=2048 | 919,418 | MLP: 2,811,733 |
| Capacity, online credit | Same | 476,457 | — |

Here `H` is core width, `B` batch, and `T` sequence length. Capacity has
`C` rows, `A` active rows, input/output width `D`, and state width `S`;
the reference MLP is 2048 → 1024 → 2048. The LSTM performs roughly twice the
recurrent work, so its comparison is per layer, not per FLOP.

At B=1, T=16,384, H=128 in BF16 on the same A40, admitted-read forward takes
about 4 ms against about 1 ms for PyTorch fused causal attention. The read
performs three reads with two value tensors. On an AMD EPYC 7713 allocation
with 8 vCPUs, PyTorch 2.8, and FP32, a full sequence-geometry block reaches
11,698 tokens/s for inference at H=256, B=1, T=1024, and 9,763 tokens/s for
forward + backward at H=256, B=8, T=128.

These are component and block measurements, not full-model training rates.
The recurrent core advances sequentially; batch supplies device parallelism,
while admission determines the separate read cost.

## Checkpoints and distributed execution

```python
from dabsn import load_dabsn, save_dabsn

save_dabsn(model, "model.safetensors")
restored = load_dabsn("model.safetensors", map_location="cpu")
```

Atomic SafeTensors checkpoints preserve model configuration, component/provider
identities, contracts, and weights. Raw graphs use `save_graph`/`load_graph`
from `dabsn.checkpoint`. Third-party providers must be registered and explicitly
trusted before loading. Optimizer state and carried training state are separate
from weights-only model files.

Distributed support includes DDP, block-wrapped FSDP, tensor parallelism over
recurrent state units, and expert parallelism. Tensor parallelism exchanges
activations at every recurrent step; expert parallelism exchanges routed
assignments. Pipeline and context parallelism do not ship.

The library supplies runtime helpers for distributed setup, sharded training
checkpoints, export, gradient verification, and distributed gradient clipping.
The `train`, `pretrain`, and `finetune` CLI commands are convenience examples
for dense DABSN models. Composed architectures use application-owned loops.
See the [runtime and persistence contracts](ARCHITECTURE.md#runtime-contract)
and `dabsn --help`.

## Papers

### 3. Attention is My Memory

**Positional Encoding-free Length Invariance in Transformers**

One memory branch gives a transformer content-addressed retrieval without
positional encoding. Main synthetic models use width 128 and four layers; the
branch adds 82,839 parameters, 2.9% of the base model.

| MQAR evaluation | NoPE + memory, front | RoPE + YaRN, fine-tuned |
| --- | ---: | ---: |
| Distance: 16,384 tokens, 64× training length | 96.9 ± 2.8% | 94.9 ± 5.7% |
| Load: 512 pairs, 8× largest training load | 100.0 ± 0.0% | 37.9 ± 10.9% |

Results are means ± standard deviations over three seeds. Memory models train
at 256 tokens; the YaRN baseline receives additional fine-tuning at 1024.
Extreme passkey evaluation reaches 100%, 96%, and 100% digit accuracy at
1,048,576 tokens over ten prompts per seed; plain NoPE scores 10% on its
evaluated seed.

The 459-run protocol includes learning-rate selection, larger models, and
enwik8. At the training length, position-free language modelling reaches
1.444 bits per character (RoPE: 1.450; ALiBi: 1.426); loss rises beyond it.
Retrieval generalization does not establish general length invariance on
copying or natural text, or constant compute and memory cost.

[Paper](https://doi.org/10.5281/zenodo.24268811) · [PDF](paper3/main.pdf) · [Source and results](paper3/) ·
[Reproduction](paper3/reproduce/README.md)

### 2. Experience as a Persistent Object

**Online Temporal Credit in DABSN**

The paper specifies `E`, proves partition composition at fixed parameters and
an exact identity for omitted influence, and constructs histories separating
`E` from the full terminal state–parameter Jacobian in both directions.
Experiments test delayed learning, causal edits with forward computation held
fixed, objectives selected after experience, and direct parameter action.

Online credit and full BPTT both reach 100% temporal-order accuracy at lengths
128, 512, and 2048; at 4096 their means are 84.8% and 85.9%. A core driven by
`E` for credit, direction, and pace, without core optimizer state, matches the
AdamW comparisons on copy, MQAR, and key–value recall. Online credit loses
BPTT's exact A5/60 length generalization, and 64-step truncated BPTT remains
much stronger on the uninterrupted-stream tasks.

[Paper](https://doi.org/10.5281/zenodo.24268687) · [PDF](paper2/main.pdf) · [Source, proofs, and results](paper2/) ·
[Reproduction](paper2/README.md)

### 1. One Layer, Both Gaps

**A Persistent-Modulation Recurrence that Generalizes Copy and Tracks Non-Solvable Group State**

One architecture, trained separately per task, tests copying and non-solvable
group-state tracking with causal ablations of its state and read pathways.

| Task | Train length | Evaluation length | Result |
| --- | ---: | ---: | ---: |
| Copy, vocabulary 64 | 64 | 3,200 (50×) | 0.961 ± 0.035, three seeds |
| A5/60 word problem | 256 | 16,384 (64×) | 1.000, two seeds |

[Paper](paper1/main.pdf) · [Source, results, and reproduction](paper1/)

## Citation

Cite the papers relevant to the architecture, online credit, or host memory:

```bibtex
@misc{rosdahl2026attentionmemory,
  title  = {Attention is My Memory: Positional Encoding-free Length Invariance
            in Transformers},
  author = {Rosdahl, Nicholas},
  year   = {2026},
  doi    = {10.5281/zenodo.24268811},
  url    = {https://doi.org/10.5281/zenodo.24268811}
}

@misc{rosdahl2026experience,
  title  = {Experience as a Persistent Object: Online Temporal Credit in DABSN},
  author = {Rosdahl, Nicholas},
  year   = {2026},
  doi    = {10.5281/zenodo.24268687},
  url    = {https://doi.org/10.5281/zenodo.24268687}
}

@misc{rosdahl2026onelayer,
  title  = {One Layer, Both Gaps: A Persistent-Modulation Recurrence that
            Generalizes Copy and Tracks Non-Solvable Group State},
  author = {Rosdahl, Nicholas},
  year   = {2026},
  doi    = {10.5281/zenodo.21391204},
  url    = {https://github.com/BleedingXiko/dabsn}
}
```

[CITATION.cff](CITATION.cff) contains software metadata, paper 1's preferred
citation, and references to papers 2 and 3. DABSN is released under the
[Apache License 2.0](LICENSE).
