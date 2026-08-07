"""Canonical DABSN framework."""

from .config import (
    DABSN_ARCH,
    DABSNConfig,
    DABSNLayerSpec,
    DABSNPretrainConfig,
    parse_dabsn_layer_specs,
    resolve_dabsn_layers,
    resolve_layer_geometries,
)
from .core import DABSNCore
from .pretrain import pretrain_next_token
from .read import DABSNRead
from .checkpoint import dabsn_config_dict, inspect_dabsn, load_dabsn, save_dabsn
from .model import (
    DABSNBackbone,
    DABSNBlock,
    DABSNModel,
    DABSNSequenceLM,
    DABSNTaskModel,
    build_dabsn,
    build_dabsn_from_config,
    dabsn_adamw_param_groups,
)
from .runtime import (
    DABSNSequenceModule,
    DistributedState,
    cleanup_distributed,
    load_distributed_optimizer,
    load_sharded_model_checkpoint,
    load_sharded_training_checkpoint,
    prepare_distributed_model,
    save_distributed_dabsn,
    save_sharded_training_checkpoint,
    setup_distributed,
    train,
    train_step,
    evaluate,
    export_dabsn,
    infer,
    verify_gradients,
)

__all__ = [
    "DABSN_ARCH",
    "DABSNConfig",
    "DABSNCore",
    "DABSNBackbone",
    "DABSNBlock",
    "DABSNLayerSpec",
    "DABSNPretrainConfig",
    "DABSNRead",
    "DABSNModel",
    "DABSNSequenceLM",
    "DABSNTaskModel",
    "DABSNSequenceModule",
    "DistributedState",
    "build_dabsn",
    "build_dabsn_from_config",
    "dabsn_adamw_param_groups",
    "dabsn_config_dict",
    "cleanup_distributed",
    "evaluate",
    "export_dabsn",
    "infer",
    "load_dabsn",
    "load_distributed_optimizer",
    "load_sharded_model_checkpoint",
    "load_sharded_training_checkpoint",
    "inspect_dabsn",
    "parse_dabsn_layer_specs",
    "resolve_dabsn_layers",
    "resolve_layer_geometries",
    "prepare_distributed_model",
    "pretrain_next_token",
    "save_dabsn",
    "save_distributed_dabsn",
    "save_sharded_training_checkpoint",
    "setup_distributed",
    "train",
    "train_step",
    "verify_gradients",
]

__version__ = "0.1.5"
