# Online temporal credit and prepared inference memory

## Online training

`temporal_credit="online"` selects the supplied DABSN-specific hybrid eligibility
rule. `"bptt"` remains the default. This is not generic exact RTRL or a claim of
full-stack BPTT gradient equivalence. The core uses budget-only matrix traces and
channel-local three-state traces. Long-read recurrent-parameter credit is exact
for its four parameters; current-chunk input credit is retained, while historical
cross-core and historical upstream write credit is not claimed exact.

MLP, MoE, projection, and output-head computations use ordinary autograd within
each chunk. They are not `DABSNCore`. Custom recurrent providers must declare and
implement their temporal policy; unsupported providers fail explicitly.

```python
import torch
from dabsn import DABSNSequenceLM, initial_online_temporal_state
from dabsn.runtime import train_step

model = DABSNSequenceLM(vocab=257, hidden_dim=64, state_dim=48, depth=2,
                        temporal_credit="online")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
state = initial_online_temporal_state(
    model, 2, device="cpu", dtype=torch.float32,
    stream_policy="reset_each_batch", stream_ids=(0, 1),
    temporal_chunk_size=16,
)
ids = torch.randint(0, 257, (2, 64))
targets = torch.randint(0, 257, (2, 64))
loss, state = train_step(model, ids, targets, optimizer,
                         temporal_state=state, temporal_chunk_size=16)
state.assert_checkpointable()
```

The example optimizer is a synthetic API demonstration, not a research training
recommendation. No clipping is implicitly added by selecting online credit.

`reset_each_batch` is for independent shuffled windows. `carry_lanes` requires
stable stream IDs and explicit resets when a lane changes stream. Eligibility
memory scales with batch lanes and core parameters (`O(B*P)`), not elapsed
context. This does **not** mean total training memory is context-independent:
the read bank and read work still grow with retained history. No automatic
batch-capacity or throughput improvement is asserted.

Weights-only checkpoints can initialize either mode. Exact training resume also
needs optimizer, accumulated gradients, recurrent/eligibility state, stream IDs,
chunk settings, and accumulation position. Use `online_training_state_dict` and
`load_online_training_state` from `dabsn.temporal_credit` when writing a custom
trainer. Checkpointing pending replay is rejected.

Native CPU core replay is C++ with BLAS. CUDA uses precise native forward tapes,
direct Triton small-state scans, and chunk-level BLAS contractions for matrix
eligibility. The CUDA long-read recurrence also uses direct Triton scans. These
online paths do not use Dynamo/Inductor compilation. Performance depends on model
dimensions, chunk size, precision, and hardware; no universal speedup is claimed.
Native CUDA online credit requires native hardware support for its storage
dtype. BF16 requires compute capability 8.0 or newer; GPUs such as T4 must use
FP32 or FP16. BF16 emulation is not treated as a native compiled backend.

`initial_online_temporal_state(dtype=...)` specifies recurrent storage precision.
Eligibility defaults to `torch.promote_types(dtype, torch.float32)`; an independent
`eligibility_dtype` override is available for reference execution. Native replay
requires FP32 eligibility and rejects incompatible precision rather than silently
narrowing it. Resets and checkpoint restore preserve each layer's eligibility
dtype. Reassociating matrix-trace sums can change floating-point rounding, not
the structured credit rule.

## Prepared carried-memory inference

```python
from dabsn import (ingest, prepare_for_inference, forward_with_memory,
                   save_prepared, load_prepared)

model.eval()
raw_memory = ingest(model, ids)
prepared = prepare_for_inference(model, raw_memory, page_size=256)
logits, unchanged = forward_with_memory(model, ids[:, :1], prepared)
assert unchanged is prepared  # default extend=False
logits, extended = forward_with_memory(model, ids[:, :1], prepared, extend=True)
save_prepared(extended, "new-memory-v2.dmem")
restored = load_prepared("new-memory-v2.dmem", model=model)
```

This explicit v2 path currently follows the existing sequence-geometry carried
memory API for legacy-constructed models. It does not add `.dmem` graph carry or
field/hybrid carried-memory semantics. Online **training** composition supports
those geometries separately; do not confuse these two contracts.

Preparation computes the unchanged causal-prefix admission equation once.
Queries do not rescore admission across the historical bank. Pages retain exact
committed writes, chronological successor writes (even if that successor was
not admitted), original int64 positions, and scalar metadata. Append copies
only touched pages; non-extending queries preserve the input memory.

Each page computes the query/content score once and reuses it for the short,
permanent, and induction reads. Stable softmax merging and query tiling bound
score workspace. The registered inference operator is
`dabsn::prepared_admitted_page`; it has a FakeTensor implementation and no
training backward. The existing training read is unchanged and remains
quadratic. No ANN, fixed top-k, eviction, or lossy compression was added.

Prepared memory is detached inference state. Its baked admission/read-parameter
identity must match the model. Normal parameter updates invalidate that identity;
do not mutate parameters through `.data`, which bypasses PyTorch version tracking.
Legacy v1 files load as before. v2 files use `load_prepared`. Preparation never
overwrites v1. `save_prepared` atomically refuses an existing destination unless
`overwrite=True` is explicitly passed.

Capacity is real, not a kernel setting: at width 4096, 1,048,576 committed events
require over 16 GiB for two FP16 value vectors per event plus metadata, before
preparation/query workspace. That all-admitted case cannot fit a 16-GiB T4.
Sparse admission can fit a million-event history with fewer committed events;
that is a different capacity measurement, not evidence for the dense worst case.
