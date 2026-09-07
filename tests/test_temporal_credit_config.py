from __future__ import annotations

import pytest

from dabsn import DABSNConfig, DABSNPretrainConfig, DABSNSequenceLM
from dabsn.model import build_dabsn_from_config
from dabsn.pretrain import _validate_model_config


def test_temporal_credit_defaults_to_bptt() -> None:
    config = DABSNConfig(input_dim=8, out_dim=8)

    assert config.temporal_credit == "bptt"
    assert config.metadata()["temporal_credit"] == "bptt"


@pytest.mark.parametrize("value", ["rtrl", "exact", "", None])
def test_temporal_credit_rejects_unknown_values(value: object) -> None:
    with pytest.raises(ValueError, match="temporal_credit"):
        DABSNConfig(input_dim=8, out_dim=8, temporal_credit=value)  # type: ignore[arg-type]


def test_constructed_model_exposes_resolved_temporal_credit_mode() -> None:
    model = build_dabsn_from_config(
        DABSNConfig(input_dim=8, out_dim=8, temporal_credit="online")
    )

    assert model.temporal_credit == "online"


def test_sequence_lm_exposes_and_validates_temporal_credit_mode() -> None:
    assert DABSNSequenceLM(vocab=17, hidden_dim=4, depth=1).temporal_credit == "bptt"
    assert (
        DABSNSequenceLM(
            vocab=17, hidden_dim=4, depth=1, temporal_credit="online"
        ).temporal_credit
        == "online"
    )
    with pytest.raises(ValueError, match="temporal_credit"):
        DABSNSequenceLM(  # type: ignore[arg-type]
            vocab=17, hidden_dim=4, depth=1, temporal_credit="rtrl"
        )


def test_pretrain_temporal_credit_contract_is_explicit() -> None:
    config = DABSNPretrainConfig(corpus_bin="unused.bin")

    assert config.temporal_credit == "bptt"
    assert config.temporal_stream_policy == "reset_each_batch"
    assert config.metadata()["temporal_credit"] == "bptt"
    assert config.metadata()["temporal_stream_policy"] == "reset_each_batch"


@pytest.mark.parametrize("value", ["rtrl", "exact", "", None])
def test_pretrain_temporal_credit_rejects_unknown_values(value: object) -> None:
    with pytest.raises(ValueError, match="temporal_credit"):
        DABSNPretrainConfig(  # type: ignore[arg-type]
            corpus_bin="unused.bin",
            temporal_credit=value,
        )


@pytest.mark.parametrize("value", ["carry", "reset", "", None])
def test_pretrain_stream_policy_rejects_unknown_values(value: object) -> None:
    with pytest.raises(ValueError, match="temporal_stream_policy"):
        DABSNPretrainConfig(  # type: ignore[arg-type]
            corpus_bin="unused.bin",
            temporal_stream_policy=value,
        )


def test_pretrain_online_chunk_and_lane_group_contract_is_validated() -> None:
    config = DABSNPretrainConfig(
        corpus_bin="unused.bin",
        temporal_credit="online",
        temporal_chunk_size=128,
        eligibility_batch_mode="lane_grouped",
        eligibility_lane_group_size=4,
    )
    assert config.temporal_chunk_size == 128
    assert config.eligibility_batch_mode == "lane_grouped"
    assert config.eligibility_lane_group_size == 4
    with pytest.raises(ValueError, match="temporal_chunk_size"):
        DABSNPretrainConfig(corpus_bin="unused.bin", temporal_chunk_size=-1)
    with pytest.raises(ValueError, match="eligibility_batch_mode"):
        DABSNPretrainConfig(  # type: ignore[arg-type]
            corpus_bin="unused.bin", eligibility_batch_mode="shared"
        )
    with pytest.raises(ValueError, match="eligibility_lane_group_size"):
        DABSNPretrainConfig(
            corpus_bin="unused.bin",
            eligibility_batch_mode="lane_grouped",
            eligibility_lane_group_size=None,
        )


def test_pretrain_resume_validation_refuses_temporal_credit_mode_change() -> None:
    model = DABSNSequenceLM(vocab=17, hidden_dim=4, depth=1, temporal_credit="bptt")
    config = DABSNPretrainConfig(
        corpus_bin="unused.bin",
        vocab=17,
        hidden_dim=4,
        depth=1,
        layers="seq:4:4",
        temporal_credit="online",
    )
    with pytest.raises(ValueError, match="temporal_credit"):
        _validate_model_config(model, config)
