"""Stable structured observability and strict fallback semantics."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Mapping


class EventCode(str, Enum):
    PROVIDER_RESOLUTION = "DABSN_PROVIDER_RESOLUTION"
    CONTRACT_VALIDATION = "DABSN_CONTRACT_VALIDATION"
    KERNEL_SELECTION = "DABSN_KERNEL_SELECTION"
    GRAPH_BREAK = "DABSN_GRAPH_BREAK"
    COMPILE_RESULT = "DABSN_COMPILE_RESULT"
    CAPTURE_RESULT = "DABSN_CAPTURE_RESULT"
    DTYPE_CHANGE = "DABSN_DTYPE_CHANGE"
    DISTRIBUTED_PLACEMENT = "DABSN_DISTRIBUTED_PLACEMENT"
    CHECKPOINT_TRANSACTION = "DABSN_CHECKPOINT_TRANSACTION"
    ROUTING_BALANCE = "DABSN_ROUTING_BALANCE"
    PERFORMANCE_FALLBACK = "DABSN_PERFORMANCE_FALLBACK"


@dataclass(frozen=True)
class StructuredEvent:
    code: EventCode
    component_id: str | None
    fields: Mapping[str, object]
    fallback: bool = False


class StrictFallbackError(RuntimeError):
    pass


_state = threading.local()
_listeners: list[Callable[[StructuredEvent], None]] = []


def _strict() -> bool:
    return bool(getattr(_state, "strict", False))


def add_event_listener(listener: Callable[[StructuredEvent], None]) -> None:
    if listener not in _listeners:
        _listeners.append(listener)


def remove_event_listener(listener: Callable[[StructuredEvent], None]) -> None:
    if listener in _listeners:
        _listeners.remove(listener)


@contextmanager
def strict_events(enabled: bool = True) -> Iterator[None]:
    previous = _strict()
    _state.strict = bool(enabled)
    try:
        yield
    finally:
        _state.strict = previous


def emit_event(
    code: EventCode,
    *,
    component_id: str | None = None,
    fallback: bool = False,
    **fields: object,
) -> StructuredEvent:
    if fallback:
        required = {"reason", "requested_path", "selected_path", "corrective_action"}
        missing = required - fields.keys()
        if missing:
            raise ValueError(f"fallback event is missing fields {sorted(missing)}")
    event = StructuredEvent(code, component_id, dict(fields), fallback)
    for listener in tuple(_listeners):
        listener(event)
    if fallback and _strict():
        raise StrictFallbackError(
            f"{code.value} strict fallback for {component_id or '<framework>'}: "
            f"{fields['reason']}; corrective action: {fields['corrective_action']}"
        )
    return event


__all__ = [
    "EventCode",
    "StrictFallbackError",
    "StructuredEvent",
    "add_event_listener",
    "emit_event",
    "remove_event_listener",
    "strict_events",
]
