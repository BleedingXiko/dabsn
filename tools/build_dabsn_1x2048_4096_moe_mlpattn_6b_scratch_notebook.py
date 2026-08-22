#!/usr/bin/env python3
"""Build the ONE-block tied DABSN + 24-expert MoE run whose experts are MIXED.

This is the 24-expert top-4 run again -- same DABSN block, same H2048/S4096,
same corpus, same curriculum, same optimizer split -- with one change: half the
expert matrix is attention instead of MLP, and the whole body is now declared as
configuration against the dabsn 0.1.7 component API rather than assembled by a
notebook-side ``moe.py``.

What changed from DABSN_1X2048_4096_9MOE_6B_SCRATCH
---------------------------------------------------
1. The body is a component graph:

       dabsn:block        H=2048, S=4096, seq, stack residual
       dabsn:sparse_moe   top-4 of 24, rmsnorm, residual, expert_specs=[...]

   built by ``component_registry`` and wrapped with ``DABSNSequenceLM.from_graph``.
   Dropless routing, the load-balance term, and the ten router reports are the
   framework's, not the notebook's. ``save_dabsn`` writes the ordered provider
   graph into the checkpoint, so the architecture reconstructs from the file.

2. Twelve experts are attention. Both families share the identical
   2048 -> 8192 -> 2048 sandwich; the MLP squares its inner activation
   pointwise, the attention expert reads the same 8192 as 32 positions of 256
   channels and lets them attend. Cost of the difference: +262,400 parameters
   per attention expert, +0.39% on the model. DABSN is untouched -- same width,
   same state, same everything.

3. Both expert families are ordinary registered providers living in the
   notebook's own ``graph_moe.py``. No dabsn source is edited, and the same
   pattern is how any third-party expert family would arrive. The cost is that
   reloading a checkpoint requires importing that module and passing
   ``trusted_providers``; the trainer prints the exact keys when it saves.

Mixed families forfeit the fused grouped-MM expert path by construction -- one
grouped matmul cannot serve two different computations -- so dispatch loops over
the routed experts, which is exactly what the previous run's hand-written
dispatch did anyway.

Corpus reused read-only from the 202M Drive shards; never pruned or
re-downloaded. Isolated Drive root.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "tools" / "build_dabsn_202m_6b_scratch_notebook.py"
GRAPH_MOE_SOURCE_FILE = ROOT / "tools" / "dabsn_graph_moe_source.py"
OUTPUT = ROOT / "DABSN_1X2048_4096_MOE_MLPATTN_6B_SCRATCH.ipynb"


def _load_base_builder():
    spec = importlib.util.spec_from_file_location("dabsn_202m_builder", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base builder: {BASE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source anchor, found {count}")
    return source.replace(old, new, 1)


def _replace_exactly(source: str, old: str, new: str, label: str, expected: int) -> str:
    count = source.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} source anchors, found {count}")
    return source.replace(old, new)


base = _load_base_builder()
GRAPH_MOE_SOURCE = GRAPH_MOE_SOURCE_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Runtime + Drive: 0.1.7, shared read-only corpus, isolated run root.
# ---------------------------------------------------------------------------
SETUP = _replace_once(
    base.SETUP,
    '"--no-deps", "dabsn==0.1.4",',
    '"--no-deps", "dabsn==0.1.7",',
    "native DABSN package",
)
SETUP = _replace_once(
    SETUP,
    '''DRIVE_ROOT = Path("/content/drive/MyDrive/DABSN_202M_6B")
DRIVE_CORPUS = DRIVE_ROOT / "corpus_zstd"
DRIVE_RUNS = DRIVE_ROOT / "runs"
DRIVE_EXPORTS = DRIVE_ROOT / "exports"
for folder in (DRIVE_CORPUS, DRIVE_RUNS, DRIVE_EXPORTS):
    folder.mkdir(parents=True, exist_ok=True)

LOCAL_CACHE = Path("/content/dabsn_6b_corpus")''',
    '''SHARED_CORPUS_ROOT = Path("/content/drive/MyDrive/DABSN_202M_6B")
DRIVE_ROOT = Path("/content/drive/MyDrive/DABSN_1X2048_4096_MOE_MLPATTN_6B")
DRIVE_CORPUS = SHARED_CORPUS_ROOT / "corpus_zstd"
DRIVE_RUNS = DRIVE_ROOT / "runs"
DRIVE_EXPORTS = DRIVE_ROOT / "exports"
if not DRIVE_CORPUS.is_dir():
    raise RuntimeError(
        f"shared 202M corpus not found at {DRIVE_CORPUS}; this notebook reuses it "
        "read-only and never downloads a second copy"
    )
for folder in (DRIVE_RUNS, DRIVE_EXPORTS):
    folder.mkdir(parents=True, exist_ok=True)

LOCAL_CACHE = Path("/content/dabsn_moe_mlpattn_6b_corpus")''',
    "isolated Drive paths with shared read-only corpus",
)
SETUP = _replace_once(
    SETUP,
    'import torch, dabsn\nif not torch.cuda.is_available():',
    '''import inspect
import torch, dabsn
from dabsn import DABSNGraph, DABSNSequenceLM, component_registry
_lm_params = inspect.signature(DABSNSequenceLM).parameters
if "tie_embeddings" not in _lm_params:
    raise RuntimeError("Installed dabsn==0.1.7 lacks tied-embedding support.")
if not hasattr(DABSNSequenceLM, "from_graph"):
    raise RuntimeError(
        "Installed dabsn lacks DABSNSequenceLM.from_graph; this notebook declares its "
        "body as a component graph and requires dabsn>=0.1.7."
    )
if "dabsn:sparse_moe" not in component_registry.keys():
    from dabsn.providers import register_builtin_components
    register_builtin_components()
if "dabsn:sparse_moe" not in component_registry.keys():
    raise RuntimeError("Installed dabsn does not ship the dabsn:sparse_moe provider.")
if not torch.cuda.is_available():''',
    "native feature gate",
)
SETUP = _replace_once(
    SETUP,
    'print("persistent root:", DRIVE_ROOT)',
    'print("shared corpus (read-only):", DRIVE_CORPUS)\nprint("isolated run root:", DRIVE_ROOT)',
    "path report",
)


# ---------------------------------------------------------------------------
# 2. Architecture + curriculum. One tied block, S=2H, 24 experts top-4,
#    half MLP and half attention.
# ---------------------------------------------------------------------------
CONTROLS = r'''
# --- 2. Immutable architecture, optimizer groups, and curriculum ---
from dabsn import parse_dabsn_layer_specs

# ONE tied DABSN block, internal bank S=4096 = 2*H(2048). Unchanged from the
# 24-expert run: same width, same state, same geometry, same residual.
LAYERS = "seq:2048:4096"          # geometry:hidden(H):state(S) -- single block
TIE_EMBEDDINGS = True
TIED_EMBEDDING_LR = 0.004
STACK_RESIDUAL = True

# The body. DABSN scans the window and builds one causal world per position;
# the final-field component keeps the last of them, so everything after it sees
# exactly ONE world per sequence and the experience axis is gone. That world
# goes to a top-4 router over 24 experts, and the readout turns the result into
# the distribution over the token that follows the whole window.
#
#   tokens [B, T] -> DABSN scan -> [B, T, H] -> final field -> [B, 1, H]
#                 -> MoE top-4 of 24 -> [B, 1, H] -> readout -> [B, 1, vocab]
#
# Every expert takes that world as [N, H] and returns [N, H]:
#
#   MLP expert        H -> ratio*H -> H, relu()^2. Never sees D.
#   attention expert  lifts each of the H coordinates into D channels, giving
#                     [N, H, D], runs MOE_ATTN_DEPTH standard pre-norm blocks
#                     with attention over the H coordinates, projects back to H.
#
# D exists only inside the attention expert. The router hands the same field to
# all four selected experts and sums their weighted outputs, so the two families
# compose without the router knowing they differ.
#
# Attention is unmasked and carries a learned coordinate identity: the H units
# are a world, not an order, but they are also not interchangeable. Causality is
# the DABSN scan's job and is already finished before the router sees anything.
#
# Depth is the parity lever. MOE_ATTN_DEPTH=None solves for the number of blocks
# that makes an attention expert weigh what an MLP expert weighs, which is what
# holds the model's TOTAL parameter count against the all-MLP run.
MOE_EXPERTS = 24
MOE_TOP_K = 4
MOE_RATIO = 4.0                   # MLP expert inner = ratio * H
MOE_ATTENTION_EXPERTS = 12        # the last 12 of 24; 0 reproduces the all-MLP run
MOE_ATTN_DMODEL = 384             # D -- transformer width inside an attention expert
MOE_ATTN_HEADS = 6                # head_dim = 384/6 = 64
MOE_ATTN_FFN_RATIO = 4.0
MOE_ATTN_DEPTH = None             # None = solve depth for parity with an MLP expert
MOE_ROUTER = "switch"             # or "aux_loss_free" (no aux term, biased top-k)
MOE_AUX_COEFF = 0.01              # switch balance coefficient, carried inside the term
MOE_ZERO_OUTPUT = True            # every expert output zero-init -> pure DABSN at step 0

# The MoE dispatch is data-dependent, so it graph-breaks rather than fails; the
# DABSN scan keeps its own internal graphs either way.
COMPILE_FORWARD = True

SEQ_LEN = 1024
MICRO_BATCH = 32
EFFECTIVE_BATCH_TOKENS = 524_288
GRAD_ACCUM = EFFECTIVE_BATCH_TOKENS // (MICRO_BATCH * SEQ_LEN)
assert MICRO_BATCH * SEQ_LEN * GRAD_ACCUM == EFFECTIVE_BATCH_TOKENS

# Proven split-AdamW ratios. Router + expert matrices are 2-D, so they land in the
# body group automatically at BASE_LRS["body"] (the trainer asserts full coverage).
BASE_LRS = {
    "body": 1e-3,
    "embed": 0.7,     # unused while tied
    "head": 0.004,
    "one_d": 0.015,
}
# Same warmup that stabilized the single-block run; a lone recurrent core needs it.
SCHEDULE_KIND = "warmup_cosine"
WARMUP_RATIO = 0.03
TOKEN_WEIGHT_DECAY = 0.0
MAX_SESSION_HOURS = 23.25
# Whole-step CUDA-graph wrapper OFF: the sparse MoE branch is data-dependent and
# runs eager, and the DABSN scan keeps its own internal graphs regardless, so the
# recurrence stays fast without it.
CUDA_GRAPH = False

STAGES = {
    "01_broad_5p7b": {
        "tokens": 5_700_000_000, "lr_scale": 1.0,
        "weight_decay": 0.1, "cooldown": 0.70,
    },
    "02_quality_200m": {
        "tokens": 200_000_000, "lr_scale": 0.10,
        "weight_decay": 0.03, "cooldown": 0.70,
    },
}

RUN_CONTRACTS = {
    "01_broad_5p7b": "dabsn1x2048-4096-24moe-mlpattn-6b-v1:01_broad_5p7b",
    "02_quality_200m": "dabsn1x2048-4096-24moe-mlpattn-6b-v1:02_quality_200m",
}

_specs = parse_dabsn_layer_specs(LAYERS)
assert len(_specs) == 1, "this run is a single DABSN block by construction"
_H = _specs[0].hidden_dim
_inner = int(round(MOE_RATIO * _H))
assert MOE_ATTN_DMODEL % MOE_ATTN_HEADS == 0, "MOE_ATTN_DMODEL must divide by MOE_ATTN_HEADS"
_mlp_each = 2 * _H * _inner
_mlp_count = MOE_EXPERTS - MOE_ATTENTION_EXPERTS
print({
    "layers": LAYERS,
    "hidden_dim_H": _H,
    "state_dim_S": _specs[0].resolved_state_dim,
    "tie_embeddings": TIE_EMBEDDINGS,
    "moe_experts": MOE_EXPERTS,
    "moe_top_k": MOE_TOP_K,
    "moe_mlp_experts": _mlp_count,
    "moe_attention_experts": MOE_ATTENTION_EXPERTS,
    "mlp_expert_inner": _inner,
    "mlp_expert_params": _mlp_each,
    "attention_view": f"[N, {_H}, {MOE_ATTN_DMODEL}] "
                      f"({MOE_ATTN_HEADS} heads of {MOE_ATTN_DMODEL // MOE_ATTN_HEADS})",
    "attention_depth": MOE_ATTN_DEPTH or "solved for parameter parity",
    "moe_router": MOE_ROUTER,
    "moe_aux_coeff": MOE_AUX_COEFF,
    "compile_forward": COMPILE_FORWARD,
    "cuda_graph": CUDA_GRAPH,
    "context": SEQ_LEN,
    "micro_batch": MICRO_BATCH,
    "grad_accum": GRAD_ACCUM,
    "predictions_per_optimizer_step": MICRO_BATCH * GRAD_ACCUM,
    "total_pretraining_tokens": sum(s["tokens"] for s in STAGES.values()),
})
'''


# ---------------------------------------------------------------------------
# 3. Corpus: reuse the 202M shards read-only; never prune shared Drive data.
# ---------------------------------------------------------------------------
CORPUS_RUNTIME = _replace_once(
    base.CORPUS_RUNTIME,
    '''        if allowed_raw_names is not None and raw_name not in allowed_raw_names:
            # This can only be an uncommitted shard left by a disconnect between
            # compression and the atomic source-state update.
            packed.unlink()
            continue''',
    '''        if allowed_raw_names is not None and raw_name not in allowed_raw_names:
            # This notebook reuses the 202M corpus read-only and never prunes
            # shared Drive data. A stale or partially committed file is ignored
            # locally and left untouched on Drive for explicit human inspection.
            print(f"[corpus] ignoring uncommitted shared file without deleting it: {packed}")
            continue''',
    "shared Drive corpus no-prune policy",
)

# The base builder's anneal mix is already GRAD_ACCUM-relative (half/quarter/
# quarter), so MICRO_BATCH=32 -> GRAD_ACCUM=16 needs no rescaling here.


# ---------------------------------------------------------------------------
# 4. Trainer command: MoE args + tied embeddings + warmup, isolated paths.
# ---------------------------------------------------------------------------
TRAIN_RUNTIME = base.TRAIN_RUNTIME.replace(
    "DABSN_202M_6B", "DABSN_1X2048_4096_MOE_MLPATTN_6B"
).replace(
    "/content/dabsn_6b_state", "/content/dabsn_moe_mlpattn_6b_state"
)
TRAIN_RUNTIME = _replace_once(
    TRAIN_RUNTIME,
    '''        "--hybrid-mlp",
        "--hybrid-mlp-ratio", str(HYBRID_MLP_RATIO),
        "--hybrid-mlp-modes", HYBRID_MLP_MODES,
        "--stack-residual",''',
    '''        "--moe-experts", str(MOE_EXPERTS),
        "--moe-top-k", str(MOE_TOP_K),
        "--moe-ratio", str(MOE_RATIO),
        "--moe-attention-experts", str(MOE_ATTENTION_EXPERTS),
        "--moe-attn-d-model", str(MOE_ATTN_DMODEL),
        "--moe-attn-heads", str(MOE_ATTN_HEADS),
        "--moe-attn-ffn-ratio", str(MOE_ATTN_FFN_RATIO),
        *([] if MOE_ATTN_DEPTH is None else ["--moe-attn-depth", str(MOE_ATTN_DEPTH)]),
        "--moe-router", MOE_ROUTER,
        "--moe-aux-coeff", str(MOE_AUX_COEFF),
        *([] if MOE_ZERO_OUTPUT else ["--moe-no-zero-output"]),
        *([] if COMPILE_FORWARD else ["--no-compile-forward"]),
        *(["--stack-residual"] if STACK_RESIDUAL else []),
        *(
            ["--tie-embeddings", "--tied-embedding-lr", str(TIED_EMBEDDING_LR)]
            if TIE_EMBEDDINGS else []
        ),
        "--schedule-kind", SCHEDULE_KIND,
        *(["--warmup-ratio", str(WARMUP_RATIO)] if SCHEDULE_KIND == "warmup_cosine" else []),''',
    "MoE trainer arguments",
)
TRAIN_RUNTIME = _replace_once(
    TRAIN_RUNTIME,
    '"--eval-every", "25", "--eval-batches", "64",',
    '"--eval-every", "50", "--eval-batches", "64",',
    "validation cadence",
)
TRAIN_RUNTIME = _replace_once(
    TRAIN_RUNTIME,
    '"--save-every", "25", "--log-every", "5",',
    '"--save-every", "100", "--log-every", "5",',
    "checkpoint + upload cadence",
)


# ---------------------------------------------------------------------------
# 5. Plot: two pretraining stages; final model is the 200M anneal.
# ---------------------------------------------------------------------------
PLOT = base.PLOT.replace(
    "dabsn_202m_6b_training.png", "dabsn_moe_mlpattn_6b_training.png"
).replace('_stage_paths("03_masked_assistant_100m")', '_stage_paths("02_quality_200m")')


# ---------------------------------------------------------------------------
# Trainer source patches: warmup scheduler, then the graph-declared MoE body.
# ---------------------------------------------------------------------------
def _patch_posttraining_scheduler(source: str) -> str:
    """Add DABSN's published warmup+cosine schedule; leaves stable_linear default."""

    source = _replace_once(
        source,
        '''def _schedule_scale(step: int, total_steps: int, cooldown_frac: float) -> float:
    """Pinned track-3 stable phase followed by a linear cooldown to zero."""
    progress = step / max(1, total_steps)
    if progress < 1.0 - cooldown_frac:
        return 1.0
    return max(0.0, (1.0 - progress) / cooldown_frac)''',
        '''def _schedule_scale(
    step: int,
    total_steps: int,
    cooldown_frac: float,
    schedule_kind: str = "stable_linear",
    warmup_ratio: float = 0.0,
) -> float:
    if schedule_kind == "warmup_cosine":
        warmup_steps = max(1, round(total_steps * warmup_ratio))
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    if progress < 1.0 - cooldown_frac:
        return 1.0
    return max(0.0, (1.0 - progress) / cooldown_frac)''',
        "post-training scheduler function",
    )
    source = _replace_once(
        source,
        '''    p.add_argument("--cooldown-frac", type=float, default=0.7,
                   help="fraction of updates used for the pinned linear cooldown")''',
        '''    p.add_argument("--cooldown-frac", type=float, default=0.7,
                   help="fraction of updates used for the pinned linear cooldown")
    p.add_argument(
        "--schedule-kind", choices=("stable_linear", "warmup_cosine"),
        default="stable_linear",
    )
    p.add_argument("--warmup-ratio", type=float, default=0.0)''',
        "post-training scheduler arguments",
    )
    source = _replace_once(
        source,
        '''    if not 0 < args.cooldown_frac <= 1:
        raise SystemExit("cooldown-frac must be in (0, 1]")''',
        '''    if not 0 < args.cooldown_frac <= 1:
        raise SystemExit("cooldown-frac must be in (0, 1]")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise SystemExit("--warmup-ratio must be in [0, 1)")
    if args.schedule_kind == "warmup_cosine" and args.warmup_ratio <= 0.0:
        raise SystemExit("warmup_cosine requires --warmup-ratio > 0")''',
        "post-training scheduler validation",
    )
    source = _replace_once(
        source,
        '''        "cooldown_frac": args.cooldown_frac,''',
        '''        "cooldown_frac": args.cooldown_frac,
        "schedule_kind": args.schedule_kind,
        "warmup_ratio": args.warmup_ratio,''',
        "post-training scheduler contract",
    )
    source = _replace_once(
        source,
        '''        schedule_scale = _schedule_scale(state.step, total_steps, args.cooldown_frac)''',
        '''        schedule_scale = _schedule_scale(
            state.step, total_steps, args.cooldown_frac,
            args.schedule_kind, args.warmup_ratio,
        )''',
        "post-training scheduler call",
    )
    return source


def _patch_trainer_graph_moe(source: str) -> str:
    """Wire the config-declared MoE body into the trainer.

    Everything architecture-specific lives in ``graph_moe.py``; the trainer keeps
    its own concerns -- argument parsing, the optimizer split (already generic),
    the loss, the log line, and exact resume.
    """

    # 1. import
    source = _replace_once(
        source,
        "from .hybrid import HybridSequenceLM, hybrid_mlp_params, parse_mlp_modes",
        "from .hybrid import HybridSequenceLM, hybrid_mlp_params, parse_mlp_modes\n"
        "from .graph_moe import (\n"
        "    EXPERT_PROVIDER_KEYS,\n"
        "    MoEForward,\n"
        "    RouterTelemetry,\n"
        "    aligned_targets,\n"
        "    branch_summary,\n"
        "    build_latent_moe_lm,\n"
        "    dabsn_blocks,\n"
        "    moe_params,\n"
        ")",
        "graph MoE import",
    )

    # 2. CLI args (after --metrics-jsonl, a known unique line)
    source = _replace_once(
        source,
        '    p.add_argument("--metrics-jsonl", required=True, help="append-only train/validation records for plots")',
        '''    p.add_argument("--metrics-jsonl", required=True, help="append-only train/validation records for plots")
    p.add_argument("--moe-experts", type=int, default=0,
                   help="number of top-k MoE experts on the block branch; 0 keeps the single-MLP path")
    p.add_argument("--moe-top-k", type=int, default=4)
    p.add_argument("--moe-ratio", type=float, default=4.0)
    p.add_argument("--moe-attention-experts", type=int, default=0,
                   help="how many of the experts are attention experts (taken from the end)")
    p.add_argument("--moe-attn-d-model", type=int, default=384,
                   help="transformer width D inside an attention expert; H is its sequence")
    p.add_argument("--moe-attn-heads", type=int, default=6)
    p.add_argument("--moe-attn-ffn-ratio", type=float, default=4.0)
    p.add_argument("--moe-attn-depth", type=int, default=0,
                   help="blocks per attention expert; 0 solves for MLP-expert parameter parity")
    p.add_argument("--moe-router", choices=("switch", "aux_loss_free"), default="switch")
    p.add_argument("--moe-aux-coeff", type=float, default=0.01,
                   help="switch balance coefficient; carried inside the router's own term")
    p.add_argument("--moe-bias-update-rate", type=float, default=1.0e-3,
                   help="aux-loss-free router bias step applied after each real update")
    p.add_argument("--moe-no-zero-output", action="store_true",
                   help="do not zero-init expert output matrices (branch is live at step 0)")
    p.add_argument("--no-compile-forward", action="store_true",
                   help="run the MoE forward eager instead of under torch.compile")''',
        "graph MoE CLI arguments",
    )

    # 3. validation
    source = _replace_once(
        source,
        '''    if args.hybrid_mlp and (args.mlp_ratio is not None or args.mlp_middle_depth):
        raise SystemExit("legacy --hybrid-mlp and framework-native MLP options cannot be mixed")''',
        '''    if args.hybrid_mlp and (args.mlp_ratio is not None or args.mlp_middle_depth):
        raise SystemExit("legacy --hybrid-mlp and framework-native MLP options cannot be mixed")
    if args.moe_experts:
        if args.hybrid_mlp or args.mlp_ratio is not None or args.mlp_middle_depth:
            raise SystemExit("--moe-experts cannot be combined with --hybrid-mlp or native --mlp-ratio")
        if not 1 <= args.moe_top_k <= args.moe_experts:
            raise SystemExit(f"--moe-top-k must be in [1, {args.moe_experts}]")
        if not 0 <= args.moe_attention_experts <= args.moe_experts:
            raise SystemExit(f"--moe-attention-experts must be in [0, {args.moe_experts}]")
        if args.moe_ratio <= 0 or args.moe_aux_coeff < 0:
            raise SystemExit("--moe-ratio must be positive and --moe-aux-coeff non-negative")
        if args.cuda_graph:
            # Whole-step capture would record ONE routing decision and replay it
            # forever, and would bypass the declared loss terms entirely. The
            # DABSN scan keeps its own internal graphs regardless.
            raise SystemExit("--cuda-graph cannot capture data-dependent MoE routing")''',
        "graph MoE validation",
    )

    # 4. construction branch (before hybrid)
    source = _replace_once(
        source,
        "    if args.hybrid_mlp:\n        modes = parse_mlp_modes(args.hybrid_mlp_modes, len(layers))",
        '''    if args.moe_experts:
        if len(layers) != 1:
            raise SystemExit("the declared MoE body is one DABSN block followed by one MoE branch")
        raw_model = build_latent_moe_lm(
            vocab=GPT2_VOCAB,
            hidden_dim=_lm_width(layers),
            state_dim=int(layers[0].resolved_state_dim),
            read_geometry=str(layers[0].read_geometry),
            stack_residual=bool(args.stack_residual),
            tie_embeddings=bool(args.tie_embeddings),
            experts=int(args.moe_experts),
            top_k=int(args.moe_top_k),
            attention_experts=int(args.moe_attention_experts),
            mlp_ratio=float(args.moe_ratio),
            attention_d_model=int(args.moe_attn_d_model),
            attention_heads=int(args.moe_attn_heads),
            attention_ffn_ratio=float(args.moe_attn_ffn_ratio),
            attention_depth=int(args.moe_attn_depth) or None,
            router=str(args.moe_router),
            balance_coefficient=float(args.moe_aux_coeff),
            bias_update_rate=float(args.moe_bias_update_rate),
            zero_output=not args.moe_no_zero_output,
        ).to(device)
        summary = branch_summary(raw_model)
        print(
            f"[moe] declared body: "
            + " -> ".join(
                f"{b.provider_key}({b.component_id})" for b in raw_model.graph.bindings
            ),
            flush=True,
        )
        print(
            f"[moe] experts={summary['experts']} top_k={summary['top_k']} "
            f"router={args.moe_router} aux_coeff={args.moe_aux_coeff} "
            f"sparsity={summary['sparsity']:.1f}x moe_params={moe_params(raw_model):,}",
            flush=True,
        )
        for family in ("mlp", "attention"):
            entry = summary["families"].get(family)
            if entry is None:
                continue
            low, high = entry["indices"]
            print(
                f"[moe]   {family:<9} experts {low}-{high}: {entry['count']} x "
                f"{entry['params_each']:,} params = {entry['params']:,}",
                flush=True,
            )
        print(
            f"[moe] stored={summary['stored_expert_params']:,} "
            f"expected_active={summary['expected_active_expert_params']:,} "
            f"router={summary['router_params']:,} norm={summary['norm_params']:,}",
            flush=True,
        )
        if not args.moe_no_zero_output:
            print(
                "[moe] every expert output matrix is zero-init: the branch contributes "
                "nothing at step 0 and the model starts as pure DABSN. Expert fc1 "
                "gradients are therefore exactly zero on the first step by construction.",
                flush=True,
            )
    elif args.hybrid_mlp:
        modes = parse_mlp_modes(args.hybrid_mlp_modes, len(layers))''',
        "graph MoE construction branch",
    )

    # 5. the training-path forward: logits for the loop, declared terms on the side
    source = _replace_once(
        source,
        '''    forward_sequence = torch.compile(raw_model.forward_sequence, dynamic=False)
    print("[compile] global DABSN backbone boundary active; embedding/readout + Triton kernels compiled", flush=True)''',
        '''    moe_forward = None
    moe_telemetry = None
    if args.moe_experts:
        moe_telemetry = RouterTelemetry(int(args.moe_experts), int(args.moe_attention_experts))
        moe_forward = MoEForward(
            raw_model, moe_telemetry, compile_forward=not args.no_compile_forward
        )
        forward_sequence = moe_forward
        print(
            "[compile] MoE body runs the graph's authoritative forward_with_terms"
            + (" under torch.compile" if not args.no_compile_forward else " eager")
            + "; dropless routing is data-dependent and graph-breaks by design, "
            "and the DABSN scan keeps its own internal graphs",
            flush=True,
        )
    else:
        forward_sequence = torch.compile(raw_model.forward_sequence, dynamic=False)
        print("[compile] global DABSN backbone boundary active; embedding/readout + Triton kernels compiled", flush=True)''',
        "graph MoE forward selection",
    )

    # 6. the preflight also routes tokens; do not let it colour the first log line
    source = _replace_once(
        source,
        '''    _assert_compiled_stack_gradients(
        raw_model,
        forward_sequence,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )''',
        '''    _assert_compiled_stack_gradients(
        raw_model,
        forward_sequence,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )
    if moe_forward is not None:
        # The preflight is a real forward: it routed tokens and produced a
        # balance term. Neither belongs to step 1.
        moe_forward.pop_aux()
        moe_telemetry.pop()''',
        "graph MoE preflight reset",
    )

    # 7. parameter accounting: the branch is not DABSN body
    source = _replace_once(
        source,
        '''    if isinstance(model, HybridSequenceLM):
        mlp_side = hybrid_mlp_params(model)
    else:''',
        '''    if getattr(model, "block_name", None) == "dabsn-graph":
        mlp_side = moe_params(model)
    elif isinstance(model, HybridSequenceLM):
        mlp_side = hybrid_mlp_params(model)
    else:''',
        "graph MoE parameter accounting",
    )

    # 8. diagnostics reach DABSN blocks through the graph as well as the backbone
    source = _replace_exactly(
        source,
        "    for block in model.backbone.blocks:",
        "    for block in dabsn_blocks(model):",
        "graph-aware DABSN block diagnostics",
        3,
    )
    source = _replace_once(
        source,
        "    for index, block in enumerate(model.backbone.blocks):",
        "    for index, block in enumerate(dabsn_blocks(model)):",
        "graph-aware gradient preflight",
    )

    # 9. declared loss terms folded into the training loss before backward
    source = _replace_once(
        source,
        '''    per_update_tokens = args.batch_size * args.seq_len * args.grad_accum''',
        '''    per_update_tokens = args.batch_size * args.seq_len * args.grad_accum
    # Corpus tokens read per update stays the curriculum's unit. What the loss
    # divides by is how many PREDICTIONS were scored, and a one-world-per-
    # sequence model scores one per sequence, not one per position.
    per_update_predictions = (
        args.batch_size * args.grad_accum if args.moe_experts else per_update_tokens
    )''',
        "prediction count",
    )
    source = _replace_once(
        source,
        '''                logits = train_forward(x)
                if m is None:
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, GPT2_VOCAB), y.reshape(-1), reduction="sum"
                    )''',
        '''                logits = train_forward(x)
                y = aligned_targets(logits, y)
                if m is not None:
                    m = aligned_targets(logits, m)
                if m is None:
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, GPT2_VOCAB), y.reshape(-1), reduction="sum"
                    )''',
        "target alignment",
    )
    source = _replace_once(
        source,
        '''                loss = loss_sum / per_update_tokens
            loss.backward()''',
        '''                loss = loss_sum / per_update_predictions
                if moe_forward is not None:
                    # The router's balance coefficient is already inside this
                    # term; adding it as-is is the whole contract.
                    moe_aux = moe_forward.pop_aux()
                    if moe_aux is not None:
                        loss = loss + moe_aux
            loss.backward()''',
        "graph MoE loss term injection",
    )
    # The fail-closed guard greps main() for the exact backward scale. It has to
    # learn the new divisor, or it fires on a change that is precisely what it
    # exists to police -- and its job is unchanged: never backpropagate a raw sum.
    source = _replace_once(
        source,
        '''    src = inspect.getsource(main)
    required = "loss = loss_sum / per_update_tokens"
    forbidden = "loss = loss_sum\\n"
    if required not in src or forbidden in src:
        raise RuntimeError(
            "trainer invariant failed: LM backward must use loss_sum / per_update_tokens"
        )''',
        '''    src = inspect.getsource(main)
    required = "loss = loss_sum / per_update_predictions"
    forbidden = "loss = loss_sum\\n"
    if required not in src or forbidden in src:
        raise RuntimeError(
            "trainer invariant failed: LM backward must divide loss_sum by the "
            "number of predictions that produced it"
        )''',
        "mean-loss backward guard",
    )
    source = _replace_once(
        source,
        "            loss_value += float(loss_sum.detach()) / per_update_tokens",
        "            loss_value += float(loss_sum.detach()) / per_update_predictions",
        "logged loss scale",
    )
    source = _replace_once(
        source,
        '''            total += float(F.cross_entropy(model.forward_sequence(x).reshape(-1, GPT2_VOCAB), y.reshape(-1)))''',
        '''            eval_logits = model.forward_sequence(x)
            total += float(F.cross_entropy(
                eval_logits.reshape(-1, GPT2_VOCAB),
                aligned_targets(eval_logits, y).reshape(-1),
            ))''',
        "validation target alignment",
    )
    source = _replace_once(
        source,
        '''        probe_logits = forward_sequence(probe_ids)
        probe_loss = F.cross_entropy(
            probe_logits.reshape(-1, GPT2_VOCAB),
            probe_targets.reshape(-1),
        )''',
        '''        probe_logits = forward_sequence(probe_ids)
        probe_loss = F.cross_entropy(
            probe_logits.reshape(-1, GPT2_VOCAB),
            aligned_targets(probe_logits, probe_targets).reshape(-1),
        )''',
        "preflight target alignment",
    )

    # 10. lifecycle hook: the aux-loss-free router updates its bias after real steps
    source = _replace_once(
        source,
        '''        optimizer.step()
        if accumulator is not None:
            accumulator.reset()  # clear installed grads + buffers for the next update''',
        '''        optimizer.step()
        step_hook = getattr(raw_model, "post_optimizer_step", None)
        if step_hook is not None:
            step_hook(step_applied=True)
        if accumulator is not None:
            accumulator.reset()  # clear installed grads + buffers for the next update''',
        "graph MoE post-optimizer hook",
    )

    # 11. routing telemetry printed + logged at the normal log cadence
    source = _replace_once(
        source,
        '''                f"tok/s {rate:,.0f} eta {eta/3600:.2f}h epoch {train_loader.epoch}", flush=True
            )
            _append_metric(metrics_path, event="train", step=state.step, tokens_seen=state.tokens_seen,''',
        '''                f"tok/s {rate:,.0f} eta {eta/3600:.2f}h epoch {train_loader.epoch}", flush=True
            )
            moe_stats = moe_telemetry.pop() if moe_telemetry is not None else None
            if moe_stats is not None:
                print(
                    f"        moe experts={moe_stats['experts']} "
                    f"balance={moe_stats['balance']:.3f} (1=uniform) cold={moe_stats['cold']} "
                    f"busiest={moe_stats['max_frac']*100:.1f}% quietest={moe_stats['min_frac']*100:.2f}% "
                    f"fair={moe_stats['uniform']*100:.1f}% | "
                    f"conf={moe_stats['conf']:.3f} spread={moe_stats['spread']:.3f}", flush=True)
                if moe_stats["attention_experts"]:
                    print(
                        f"        moe family mlp({moe_stats['mlp_experts']}) "
                        f"{moe_stats['mlp_frac']*100:.1f}% vs "
                        f"attention({moe_stats['attention_experts']}) "
                        f"{moe_stats['attention_frac']*100:.1f}% of routed slots "
                        f"(equal share = "
                        f"{moe_stats['mlp_experts']/moe_stats['experts']*100:.1f}%/"
                        f"{moe_stats['attention_experts']/moe_stats['experts']*100:.1f}%)",
                        flush=True)
                _append_metric(metrics_path, event="moe", step=state.step, **moe_stats)
            _append_metric(metrics_path, event="train", step=state.step, tokens_seen=state.tokens_seen,''',
        "graph MoE routing telemetry",
    )

    # 12. model config metadata
    source = _replace_once(
        source,
        '''        "hybrid_stack_residual": bool(args.stack_residual) if args.hybrid_mlp else None,
    }''',
        '''        "hybrid_stack_residual": bool(args.stack_residual) if args.hybrid_mlp else None,
        "moe_experts": int(args.moe_experts) if args.moe_experts else 0,
        "moe_top_k": int(args.moe_top_k) if args.moe_experts else None,
        "moe_ratio": float(args.moe_ratio) if args.moe_experts else None,
        "moe_attention_experts": int(args.moe_attention_experts) if args.moe_experts else None,
        "moe_attn_d_model": int(args.moe_attn_d_model) if args.moe_experts else None,
        "moe_attn_heads": int(args.moe_attn_heads) if args.moe_experts else None,
        "moe_attn_ffn_ratio": float(args.moe_attn_ffn_ratio) if args.moe_experts else None,
        "moe_attn_depth": int(args.moe_attn_depth) if args.moe_experts else None,
        "moe_router": str(args.moe_router) if args.moe_experts else None,
        "moe_aux_coeff": float(args.moe_aux_coeff) if args.moe_experts else None,
        "moe_zero_output": (not args.moe_no_zero_output) if args.moe_experts else None,
        "moe_stack_residual": bool(args.stack_residual) if args.moe_experts else None,
    }''',
        "graph MoE config metadata",
    )

    # 13. resume-config backfill defaults
    source = _replace_once(
        source,
        '''        "mlp_ratio": None,
        "mlp_middle_depth": 0,
        "mlp_depth_index": 0,
    }''',
        '''        "mlp_ratio": None,
        "mlp_middle_depth": 0,
        "mlp_depth_index": 0,
        "moe_experts": 0,
        "moe_top_k": None,
        "moe_ratio": None,
        "moe_attention_experts": None,
        "moe_attn_d_model": None,
        "moe_attn_heads": None,
        "moe_attn_ffn_ratio": None,
        "moe_attn_depth": None,
        "moe_router": None,
        "moe_aux_coeff": None,
        "moe_zero_output": None,
        "moe_stack_residual": None,
    }''',
        "graph MoE resume defaults",
    )

    # 14. the portable file carries the provider graph; say what reloading needs
    source = _replace_once(
        source,
        '''    if not isinstance(model, HybridSequenceLM):
        save_dabsn(model, path, extra=extra)
        return''',
        '''    if not isinstance(model, HybridSequenceLM):
        save_dabsn(model, path, extra=extra)
        if getattr(model, "block_name", None) == "dabsn-graph":
            # The SafeTensors manifest carries the ordered provider graph,
            # nested expert specs included, so the architecture reconstructs
            # from the file -- but two of those providers ship with this
            # notebook rather than with dabsn.
            print(
                f"[export] {path.name}: reload with "
                f"load_dabsn(path, trusted_providers={list(EXPERT_PROVIDER_KEYS)}) "
                "after importing this notebook's graph_moe module",
                flush=True,
            )
        return''',
        "graph MoE export note",
    )
    return source


RUNTIME_SOURCES = base.runtime_sources()
RUNTIME_SOURCES["graph_moe.py"] = GRAPH_MOE_SOURCE
RUNTIME_SOURCES["trainer.py"] = _patch_posttraining_scheduler(RUNTIME_SOURCES["trainer.py"])
RUNTIME_SOURCES["trainer.py"] = _patch_trainer_graph_moe(RUNTIME_SOURCES["trainer.py"])
RUNTIME_CELLS = [
    cell
    for filename, source in RUNTIME_SOURCES.items()
    for cell in base.writefile_cells(filename, source)
]


# ---------------------------------------------------------------------------
# 6. Pre-flight gate: prove the declared body builds, routes, and reloads
#    before any paid hour is spent on it.
# ---------------------------------------------------------------------------
PREFLIGHT = r'''
# --- 2b. Pre-flight: build, route, and reload the declared body (CPU, seconds) ---
#
# Every claim below is checked by the framework, not asserted by this notebook.
# It runs a scaled-down copy of the exact same graph on CPU, so a wiring or
# capability mistake surfaces here rather than three hours into an A100 rental.
import json
import tempfile
from pathlib import Path

import torch

from dabsn import load_dabsn, save_dabsn
from dabsn.conformance import check_component
from dabsn_modded_nanogpt.graph_moe import (
    ATTENTION_EXPERT_KEY,
    EXPERT_PROVIDER_KEYS,
    FINAL_FIELD_KEY,
    MLP_EXPERT_KEY,
    MoEForward,
    RouterTelemetry,
    attention_depth_for_params,
    attention_expert_macs,
    attention_expert_params,
    branch_summary,
    build_latent_moe_lm,
    register_expert_providers,
)

register_expert_providers()

# 1. Conformance. Each provider declares capabilities; check_component builds it
#    and tests them. A declared capability that is not real fails here.
for key, config in (
    (FINAL_FIELD_KEY, {"width": 64}),
    (MLP_EXPERT_KEY, {"width": 64, "inner": 256, "zero_output": True}),
    (
        ATTENTION_EXPERT_KEY,
        {"width": 64, "d_model": 32, "heads": 4, "ffn_ratio": 4.0, "depth": 2,
         "zero_output": True},
    ),
):
    report = check_component(key, config)
    failures = [(c.name, c.detail) for c in report.checks if c.status == "fail"]
    print(f"[preflight] conformance {key}: {len(report.checks)} checks, {len(failures)} failed")
    if failures:
        raise RuntimeError(f"{key} declares a capability it does not have: {failures}")

# 2. The same graph at toy scale: build, route, backward.
_probe = build_latent_moe_lm(
    vocab=97,
    hidden_dim=256,
    state_dim=512,
    read_geometry="seq",
    stack_residual=STACK_RESIDUAL,
    tie_embeddings=TIE_EMBEDDINGS,
    experts=MOE_EXPERTS,
    top_k=MOE_TOP_K,
    attention_experts=MOE_ATTENTION_EXPERTS,
    mlp_ratio=MOE_RATIO,
    attention_d_model=MOE_ATTN_DMODEL // 4,
    attention_heads=MOE_ATTN_HEADS,
    attention_ffn_ratio=MOE_ATTN_FFN_RATIO,
    attention_depth=2,
    router=MOE_ROUTER,
    balance_coefficient=MOE_AUX_COEFF,
    zero_output=MOE_ZERO_OUTPUT,
)
print("[preflight] declared body:", " -> ".join(
    f"{b.provider_key}({b.component_id})" for b in _probe.graph.bindings))

_telemetry = RouterTelemetry(MOE_EXPERTS, MOE_ATTENTION_EXPERTS)
_forward = MoEForward(_probe, _telemetry, compile_forward=False)
_ids = torch.randint(0, 97, (8, 16))
_logits = _forward(_ids)
_aux = _forward.pop_aux()
assert _logits.shape == (8, 1, 97), (
    f"one world per sequence must yield one prediction, got {tuple(_logits.shape)}"
)
(_logits.float().square().mean() + (_aux if _aux is not None else 0)).backward()

_core_grad = None
for _binding in _probe.graph.bindings:
    if _binding.provider_key == "dabsn:block":
        _core_grad = float(_binding.module.core.W.weight.grad.detach().float().norm())
assert _core_grad and _core_grad > 0, f"DABSN core received no gradient: {_core_grad}"
_stats = _telemetry.pop()
assert _stats["forwards"] == 1 and _stats["experts"] == MOE_EXPERTS
print(f"[preflight] forward+backward OK: logits {tuple(_logits.shape)} "
      f"(one prediction per sequence) dabsn core grad={_core_grad:.4f} "
      f"balance_term={float(_aux.detach()) if _aux is not None else None}")
print(f"[preflight] routed {int(_stats['experts'])} experts: "
      f"mlp {_stats['mlp_frac']*100:.1f}% attention {_stats['attention_frac']*100:.1f}% "
      f"cold={_stats['cold']}")

# 3. The architecture survives a round trip through the portable format.
_probe.eval()
_expected = _probe.forward_sequence(_ids)
with tempfile.TemporaryDirectory() as _tmp:
    _path = Path(_tmp) / "preflight.safetensors"
    save_dabsn(_probe, _path)
    _restored = load_dabsn(_path, trusted_providers=list(EXPERT_PROVIDER_KEYS)).eval()
    torch.testing.assert_close(_restored.forward_sequence(_ids), _expected, atol=0, rtol=0)
print("[preflight] checkpoint round trip is bit-exact; the file carries all three components")

# 4. Full-scale accounting: parameters AND cost, against the all-MLP run.
_depth = MOE_ATTN_DEPTH or attention_depth_for_params(
    _mlp_each, _H, MOE_ATTN_DMODEL, MOE_ATTN_FFN_RATIO
)
_attn_each = attention_expert_params(_H, MOE_ATTN_DMODEL, MOE_ATTN_FFN_RATIO, _depth)
_attn_macs = attention_expert_macs(_H, MOE_ATTN_DMODEL, MOE_ATTN_FFN_RATIO, _depth)
_stored = _mlp_count * _mlp_each + MOE_ATTENTION_EXPERTS * _attn_each
_baseline = MOE_EXPERTS * _mlp_each
_fields = MICRO_BATCH                       # ONE world per sequence, not per position
_attn_calls = _fields * MOE_TOP_K * MOE_ATTENTION_EXPERTS // MOE_EXPERTS
print(json.dumps({
    "attention_depth": _depth,
    "params_per_mlp_expert": _mlp_each,
    "params_per_attention_expert": _attn_each,
    "delta_percent_per_expert": round(100.0 * (_attn_each - _mlp_each) / _mlp_each, 3),
    "stored_experts": _stored,
    "all_mlp_baseline": _baseline,
    "total_delta_percent": round(100.0 * (_stored - _baseline) / _baseline, 3),
    "macs_per_mlp_expert_call": _mlp_each,
    "macs_per_attention_expert_call": _attn_macs,
    "fields_per_microbatch": _fields,
    "expected_attention_calls_per_microbatch": _attn_calls,
    "expected_attention_macs_per_microbatch": _attn_calls * _attn_macs,
}, indent=1))

del _probe, _restored, _forward, _telemetry
print("[preflight] PASS -- the declared body is sound; training may proceed")
'''


CELLS = [
    base.markdown(
        """
        # DABSN — one world per sequence, 24-expert MoE, half MLP and half ATTENTION

        **Run All** on a Colab **A100 80GB**. Installs `dabsn==0.1.7`.

        DABSN scans the window and builds one causal world per position. The
        final-field component keeps the last of them and drops the rest, so from
        that point on there is **one world per sequence and no experience axis
        at all**. That world goes to a top-4 router over 24 experts, and the
        readout turns the result into the distribution over the token that
        follows the whole window.

        ```
        tokens [B, T] -> DABSN scan -> [B, T, H] -> final field -> [B, 1, H]
                      -> MoE top-4 of 24        -> [B, 1, H]
                      -> readout                -> [B, 1, vocab]
        ```

        ## The expert matrix

        Every expert takes the world as `[N, H]` and returns `[N, H]`. Twelve are
        the ordinary relu-squared MLP. Twelve are attention:

        | | what happens inside |
        |---|---|
        | MLP expert | `H → 4H → H`, `relu(·)²`. Never sees D. |
        | attention expert | lift each of the H coordinates into D channels → `[N, H, D]`, 19 standard pre-norm blocks attending over the H coordinates, project back to `[N, H]` |

        **D exists only inside the attention expert.** The router hands the same
        field to all four selected experts and sums their weighted outputs; it
        never knows the two families differ. That is the whole point — one
        router, multiple kinds of thing.

        Attention over the H coordinates is **unmasked** — they are a world, not
        an order — but carries a learned coordinate identity, because they are
        also not interchangeable. Causality belongs to the DABSN scan and is
        finished before the router sees anything.

        **Parameter parity by depth.** `MOE_ATTN_DEPTH=None` solves for the
        number of blocks that makes an attention expert weigh what an MLP expert
        weighs: at `D=384` that is **19 blocks, 34,422,528 parameters** against
        the MLP expert's 33,554,432. Stored experts total 815,723,520 against the
        all-MLP run's 805,306,368 — **+1.3%**. DABSN itself is untouched: same
        width, same state, same geometry.

        Compute is not matched and cannot be: an attention expert reuses its
        weights across 2,048 coordinates, so one call is ~130 G MACs against the
        MLP expert's 33.5 M. With **32 worlds per microbatch** — one per
        sequence, not one per position — that is 64 attention calls and ~8.3 T
        MACs, a few times the DABSN scan itself. The pre-flight prints all of it.

        ## How the architecture is declared

        Three ordinary components in one graph:

        ```
        dabsn:block                       H=2048, S=4096, seq, stack residual
        dabsn-world-experts:final_field   [B, T, H] -> [B, 1, H]
        dabsn:sparse_moe                  top-4 of 24, rmsnorm, residual, expert_specs=[24]
        ```

        `component_registry` resolves it, `DABSNGraph` validates every edge, and
        `DABSNSequenceLM.from_graph` wraps it. Routing, the balance term, and the
        ten router reports come from `dabsn`. `save_dabsn` writes the whole
        ordered graph — nested expert specs included — into the checkpoint, so
        the architecture reconstructs from the file.

        The three non-built-in components are ordinary registered providers in
        this notebook's `graph_moe.py`. No `dabsn` source is edited. Reloading a
        checkpoint needs `trusted_providers=[...]` plus an import of that module;
        the trainer prints the exact keys when it saves.

        ## Objective

        One world per sequence means **one prediction per sequence** — the token
        after the window. 32 labels per microbatch, 512 per optimizer step. The
        loss picks the matching target from its own output shape, so nothing in
        the loader changes.

        Corpus reused **read-only** from `MyDrive/DABSN_202M_6B/corpus_zstd`.
        Mutable artifacts live under `MyDrive/DABSN_1X2048_4096_MOE_MLPATTN_6B`.
        5.7B broad + 200M anneal, micro-batch 32 x 16 accum, warmup-cosine,
        23.25h wall cap. Each log line prints routing balance, cold experts, and
        **the share of routed slots each family won**.
        """
    ),
    base.code(SETUP),
    base.markdown(
        """
        ## Visible training runner

        Ordinary Python modules: the split-LR loop, masked loss, exact-resume
        format, and `graph_moe.py` — the two expert providers plus the small
        adapters that hand the loop its logits and keep the graph's declared
        loss terms and router reports. The DABSN implementation, the MoE
        component, the router, and the CUDA kernels are unchanged public
        `dabsn==0.1.7`.
        """
    ),
    *RUNTIME_CELLS,
    base.code(CONTROLS),
    base.markdown(
        """
        ## Pre-flight

        The declared body is built at toy scale on CPU and checked end to end —
        provider conformance, forward, backward, routing, and a bit-exact
        checkpoint round trip — before a paid hour is spent on it.
        """
    ),
    base.code(PREFLIGHT),
    base.code(CORPUS_RUNTIME),
    base.code(TRAIN_RUNTIME),
    base.markdown("## Stage 1 — 5.7B broad pretraining"),
    base.code(
        """
        broad_paths = _stage_paths("01_broad_5p7b")
        if _remote_final_to_local("01_broad_5p7b", broad_paths["final"]):
            broad_final = broad_paths["final"]
            print("Stage 1 already complete; skipped its corpus and training.")
        else:
            broad_data = assemble_stage("01_broad_5p7b", BROAD_SOURCES)
            broad_final = run_stage("01_broad_5p7b", broad_data)
        """
    ),
    base.markdown("## Stage 2 — 200M quality annealing"),
    base.code(
        """
        anneal_paths = _stage_paths("02_quality_200m")
        if _remote_final_to_local("02_quality_200m", anneal_paths["final"]):
            anneal_final = anneal_paths["final"]
            print("Stage 2 already complete; skipped its corpus and training.")
        else:
            anneal_data = build_anneal_stage()
            anneal_final = run_stage("02_quality_200m", anneal_data, broad_final)
        """
    ),
    base.code(PLOT),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"cells={len(CELLS)} bytes={OUTPUT.stat().st_size:,}")


if __name__ == "__main__":
    main()
