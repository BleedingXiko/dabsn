"""Framework train, evaluate, infer, export, and gradient-verification APIs."""

from ..distributed import (
    DABSNSequenceModule,
    DistributedState,
    autocast_context,
    cleanup_distributed,
    clip_grad_norm,
    load_distributed_optimizer,
    load_sharded_model_checkpoint,
    load_sharded_training_checkpoint,
    make_grad_scaler,
    no_sync_context,
    prepare_distributed_model,
    resolve_precision,
    save_distributed_dabsn,
    save_sharded_training_checkpoint,
    setup_distributed,
    shard_batch,
    unwrap_dabsn_model,
    wrap_distributed,
)

from .api import (
    evaluate,
    export_dabsn,
    infer,
    train,
    train_step,
    verify_gradients,
)
from .graph import make_graphed_train_callable
from .grad_accum import ManualGradientAccumulator
from .dispatch import log_routing_once, reset_routing_log, warn_routing_once
from .loss import (
    ChunkedLinearCrossEntropy,
    chunked_cross_entropy_from_logits,
    chunked_linear_cross_entropy,
)

__all__ = [
    "DABSNSequenceModule",
    "DistributedState",
    "autocast_context",
    "cleanup_distributed",
    "ChunkedLinearCrossEntropy",
    "chunked_cross_entropy_from_logits",
    "chunked_linear_cross_entropy",
    "log_routing_once",
    "reset_routing_log",
    "warn_routing_once",
    "clip_grad_norm",
    "evaluate",
    "export_dabsn",
    "infer",
    "load_distributed_optimizer",
    "load_sharded_model_checkpoint",
    "load_sharded_training_checkpoint",
    "make_grad_scaler",
    "make_graphed_train_callable",
    "ManualGradientAccumulator",
    "no_sync_context",
    "prepare_distributed_model",
    "resolve_precision",
    "save_distributed_dabsn",
    "save_sharded_training_checkpoint",
    "setup_distributed",
    "shard_batch",
    "train",
    "train_step",
    "unwrap_dabsn_model",
    "verify_gradients",
    "wrap_distributed",
]
