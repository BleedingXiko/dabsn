import pytest
import torch
import torch.nn as nn

from dabsn import (
    AxisContract,
    ComponentCapabilities,
    ComponentContract,
    DABSNGraph,
    EventCode,
    StrictFallbackError,
    ValueContract,
    add_event_listener,
    bind_module,
    emit_event,
    remove_event_listener,
    strict_events,
)
from dabsn.distributed import DistributedState, wrap_distributed
from dabsn.runtime.dispatch import (
    log_routing_once,
    reset_routing_log,
    warn_routing_once,
)


def _world(width):
    return ValueContract.tensor(
        AxisContract("batch", "B", dynamic=True),
        AxisContract("experience", "T", dynamic=True),
        AxisContract("world", width),
    )


def test_fallback_events_require_every_corrective_field():
    with pytest.raises(ValueError, match="corrective_action"):
        emit_event(
            EventCode.PERFORMANCE_FALLBACK,
            fallback=True,
            reason="fixture",
            requested_path="fast",
            selected_path="slow",
        )


def test_strict_mode_turns_fallback_into_exception():
    with strict_events():
        with pytest.raises(StrictFallbackError, match="corrective action"):
            emit_event(
                EventCode.PERFORMANCE_FALLBACK,
                fallback=True,
                reason="fixture",
                requested_path="fast",
                selected_path="slow",
                corrective_action="install the fast backend",
            )


def test_contract_and_capability_events_are_structured():
    events = []
    add_event_listener(events.append)
    try:
        contract = ComponentContract(_world(8), _world(8))
        graph = DABSNGraph(
            [
                bind_module(
                    "limited",
                    nn.Identity(),
                    contract,
                    capabilities=ComponentCapabilities(eager=True),
                )
            ],
            input_contract=_world(8),
        )
        with pytest.raises(Exception):
            graph.require_capabilities("compile_fullgraph")
    finally:
        remove_event_listener(events.append)
    assert any(event.code == EventCode.CONTRACT_VALIDATION for event in events)
    fallback = next(event for event in events if event.fallback)
    assert fallback.fields["requested_path"] == "compile_fullgraph"
    assert fallback.fields["selected_path"] == "none"


def test_legacy_dispatch_routes_emit_stable_structured_events():
    reset_routing_log()
    events = []
    add_event_listener(events.append)
    try:
        log_routing_once("core", "triton", shape=(2, 8, 16))
        warn_routing_once(
            "read",
            "requested compact kernel is unavailable",
            requested_path="compact-triton",
            selected_path="eager-reference",
            corrective_action="install a supported Triton build",
        )
    finally:
        remove_event_listener(events.append)
    assert [event.code for event in events] == [
        EventCode.KERNEL_SELECTION,
        EventCode.PERFORMANCE_FALLBACK,
    ]
    assert events[1].fallback is True
    assert events[1].fields["requested_path"] == "compact-triton"


def test_non_distributed_placement_is_still_observable():
    events = []
    add_event_listener(events.append)
    try:
        module = torch.nn.Linear(2, 2)
        assert wrap_distributed(module, DistributedState()) is module
    finally:
        remove_event_listener(events.append)
    placement = next(event for event in events if event.code == EventCode.DISTRIBUTED_PLACEMENT)
    assert placement.fields["kind"] == "none"
