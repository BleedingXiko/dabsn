# DABSN

**A recurrent modeling framework for persistent state and admitted memory.**

DABSN is a PyTorch architecture for causal sequences, whole fields, and
structured data that mixes both. Its core maintains a nonlinear recurrent
state while its read system combines admitted short memory, successor
induction, permanent associative memory, and a recurrent long-memory channel.
The same block design is used across all three geometries.

The framework includes native C++/OpenMP CPU kernels and Triton/CUDA kernels
for forward and backward execution, task-owned input and output adapters,
structured checkpoints, training and inference helpers, and the complete
source and result tables for the accompanying paper. Multi-GPU training uses
PyTorch DDP or block-wrapped FSDP with full parameter, gradient, and optimizer
sharding.

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

## Native runtimes

```python
from dabsn.kernels import enable, status

enable("cuda", required=True)   # Triton forward and backward
# enable("cpu", required=True)  # C++/OpenMP forward and backward
# enable("reference")           # explicit PyTorch reference runtime

print(status())
```

Backend activation is process-wide because it installs model dispatch hooks.
Requested native execution never silently falls back. The status report names
the active implementation for the core scan, admitted read, permanent memory,
long-memory recurrence, and local-field gather.

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

FSDP is parameter, gradient, optimizer-state, and batch parallelism. It does
not split one sequence, one oversized matrix, or the recurrent context across
GPUs. DABSN does not currently ship tensor, pipeline, or context parallelism;
therefore this repository does not claim that FSDP alone can train an
arbitrary one-trillion-parameter configuration.

Programmatic users can access the same implementation through
`setup_distributed`, `prepare_distributed_model`, `save_distributed_dabsn`,
`save_sharded_training_checkpoint`, and their matching load functions from
`dabsn.runtime`.

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

Model checkpoints are atomic, non-pickle SafeTensors files. They embed the full
clean DABSN construction config and preserve tied weights. Custom adapter
implementations remain application-owned and must be registered before loading
a checkpoint that names them. Optimizer sidecars and distributed training
directories are trusted run state, not files to accept from an untrusted
source.

## Train, pretrain, fine-tune, and resume

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
