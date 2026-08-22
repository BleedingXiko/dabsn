"""Log-once routing observability for the DABSN runtime.

Every kernel-selection decision (which core scan, which read backend) and every
silent fall-off from a fast path is announced exactly once per distinct
(component, decision, shape) so a long training run does not drown in per-step
logs, yet an operator can always see *why* a given backend was chosen and when a
CUDA call quietly dropped to a slower path.

Set ``DABSN_SILENCE_ROUTING=1`` to mute it entirely.
"""

from __future__ import annotations

import logging
import os
import threading

_LOG = logging.getLogger("dabsn.dispatch")
_SEEN: set = set()
_LOCK = threading.Lock()


def _silenced() -> bool:
    return os.environ.get("DABSN_SILENCE_ROUTING", "0") == "1"


def _fields_str(fields: dict) -> str:
    return "  ".join(f"{key}={value}" for key, value in fields.items())


def _once(key: tuple) -> bool:
    """Return True the first time ``key`` is seen (thread-safe)."""
    with _LOCK:
        if key in _SEEN:
            return False
        _SEEN.add(key)
        return True


def log_routing_once(component: str, decision: str, **fields) -> None:
    """Announce a backend selection once per (component, decision, shape)."""
    key = ("route", component, decision, tuple(sorted(fields.items())))
    if _once(key):
        from dabsn.events import EventCode, emit_event

        emit_event(
            EventCode.KERNEL_SELECTION,
            component_id=component,
            selected_path=decision,
            **fields,
        )
        if not _silenced():
            _LOG.info(
                "DABSN routing [%s] -> %s  %s",
                component,
                decision,
                _fields_str(fields),
            )


def warn_routing_once(component: str, message: str, **fields) -> None:
    """Warn once when a CUDA call silently falls off a fast path.

    Use this for the cases the user must be able to see -- a fused kernel
    dropping to the eager scan, an explicit backend request being overridden for
    safety -- with a concrete, actionable reason.
    """
    key = ("warn", component, message, tuple(sorted(fields.items())))
    if _once(key):
        from dabsn.events import EventCode, emit_event

        event_fields = dict(fields)
        requested_path = str(
            event_fields.pop("requested_path", event_fields.pop("requested_backend", "optimized"))
        )
        selected_path = str(
            event_fields.pop("selected_path", event_fields.pop("selected_backend", "reference"))
        )
        corrective_action = str(
            event_fields.pop(
                "corrective_action",
                "inspect the capability report and select a supported backend explicitly",
            )
        )
        emit_event(
            EventCode.PERFORMANCE_FALLBACK,
            component_id=component,
            fallback=True,
            reason=message,
            requested_path=requested_path,
            selected_path=selected_path,
            corrective_action=corrective_action,
            **event_fields,
        )
        if not _silenced():
            _LOG.warning(
                "DABSN routing WARNING [%s]: %s  %s",
                component,
                message,
                _fields_str(fields),
            )


def reset_routing_log() -> None:
    """Forget every logged decision (test hook; not used in production)."""
    with _LOCK:
        _SEEN.clear()
