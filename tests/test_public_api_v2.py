import json
from pathlib import Path

import torch

import dabsn
from dabsn.checkpoint import CHECKPOINT_VERSION
from dabsn.events import EventCode

SNAPSHOT = Path(__file__).parent / "fixtures/public-api-v2.json"


def test_public_api_matches_reviewed_v2_snapshot():
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert sorted(dabsn.__all__) == expected
    assert all(hasattr(dabsn, name) for name in expected)


def test_stable_abi_and_schema_versions_are_pinned():
    assert dabsn.COMPONENT_ABI_VERSION == 2
    assert CHECKPOINT_VERSION == 2


def test_registered_operator_and_event_names_are_pinned():
    assert {
        "stable_expert_permutation",
        "weighted_scatter_add",
        "permanent_delta_scan",
        "linear_recurrence",
        "local_field_gather",
    } <= set(dir(torch.ops.dabsn))
    assert {event.value for event in EventCode} == {
        "DABSN_PROVIDER_RESOLUTION",
        "DABSN_CONTRACT_VALIDATION",
        "DABSN_KERNEL_SELECTION",
        "DABSN_GRAPH_BREAK",
        "DABSN_COMPILE_RESULT",
        "DABSN_CAPTURE_RESULT",
        "DABSN_DTYPE_CHANGE",
        "DABSN_DISTRIBUTED_PLACEMENT",
        "DABSN_CHECKPOINT_TRANSACTION",
        "DABSN_ROUTING_BALANCE",
        "DABSN_PERFORMANCE_FALLBACK",
    }
