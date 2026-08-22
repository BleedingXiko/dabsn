"""Ordered architecture graph for DABSN 2.x components."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .components import (
    BoundComponent,
    ComponentCapabilities,
    ComponentContractError,
    ComponentOutput,
    ComponentSpec,
    ValueContract,
)
from .events import EventCode, emit_event


class UnsupportedExecutionModeError(RuntimeError):
    pass


@dataclass(frozen=True)
class CapabilityFailure:
    component_id: str
    mode: str
    corrective_action: str


@dataclass(frozen=True)
class CapabilityReport:
    supported: Mapping[str, bool]
    failures: tuple[CapabilityFailure, ...]

    def require(self, *modes: str) -> None:
        missing = [failure for failure in self.failures if failure.mode in modes]
        if missing:
            text = "; ".join(
                f"{item.component_id}: {item.mode} unsupported ({item.corrective_action})"
                for item in missing
            )
            raise UnsupportedExecutionModeError(text)


class DABSNGraph(nn.Module):
    """A resolved ordered component graph with no registry lookup in forward."""

    def __init__(
        self,
        components: Sequence[BoundComponent],
        *,
        input_contract: ValueContract | None = None,
        require_world_builder: bool = False,
        register_modules: bool = True,
    ) -> None:
        super().__init__()
        if not components:
            raise ValueError("DABSNGraph requires at least one component")
        ids = [component.component_id for component in components]
        if len(ids) != len(set(ids)):
            raise ValueError(f"component IDs must be unique, received {ids}")
        if require_world_builder and not components[0].capabilities.world_builder:
            raise ValueError("language graphs must begin with the DABSN world-building component")

        if input_contract is not None:
            self._validate_edge(
                "<graph-input>",
                components[0].component_id,
                components[0].contract.input,
                input_contract,
            )
        for producer, consumer in zip(components, components[1:]):
            self._validate_edge(
                producer.component_id,
                consumer.component_id,
                consumer.contract.input,
                producer.contract.output,
            )

        # Modules are registered exactly once in execution order.  Provider
        # resolution and all architecture inspection have already finished.
        modules = tuple(component.module for component in components)
        if register_modules:
            self.components = nn.ModuleList(modules)
        else:
            # Compatibility wrappers retain ownership under their historical
            # parameter namespaces.  The graph is an execution view over those
            # exact modules, not a second registered copy of the architecture.
            object.__setattr__(self, "components", modules)
        self.component_ids = tuple(ids)
        self.require_world_builder = bool(require_world_builder)
        self.input_contract = input_contract or components[0].contract.input
        self.output_contract = components[-1].contract.output
        self._bindings = tuple(components)
        self._with_terms: tuple[Callable[..., ComponentOutput] | None, ...] = tuple(
            cast(
                Callable[..., ComponentOutput] | None,
                getattr(type(component.module), "forward_with_terms", None),
            )
            for component in components
        )
        self._with_state: tuple[Callable[..., ComponentOutput] | None, ...] = tuple(
            cast(
                Callable[..., ComponentOutput] | None,
                getattr(type(component.module), "forward_with_state", None),
            )
            for component in components
        )
        self._post_step = tuple(
            getattr(component.module, "post_optimizer_step", None) for component in components
        )
        self.loss_declarations = tuple(
            declaration for component in components for declaration in component.loss_terms
        )
        self.report_declarations = tuple(
            declaration for component in components for declaration in component.reports
        )
        self.state_declarations = tuple(
            declaration for component in components for declaration in component.states
        )
        offset = 0
        state_slices = []
        for component in components:
            next_offset = offset + len(component.states)
            state_slices.append(slice(offset, next_offset))
            offset = next_offset
        self._state_slices = tuple(state_slices)
        self._activation_checkpointing = False

    @staticmethod
    def _stream_is_capturing() -> bool:
        # See the matching short-circuit in `read.py`: this op is untraceable by
        # Dynamo (a torch.* op returning bool), so probing it during tracing
        # fails a whole-graph compile. Tracing is not capturing, so False is the
        # right answer there and runtime behaviour is unchanged.
        if torch.compiler.is_compiling():
            return False
        if not torch.cuda.is_available():
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError:
            return False

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Enable provider-declared recomputation boundaries for future forwards."""

        if enabled:
            self.require_capabilities("activation_checkpoint")
        self._activation_checkpointing = bool(enabled)

    def _checkpoint_component(self, index: int) -> bool:
        return bool(
            self._activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and self._bindings[index].capabilities.activation_checkpoint
            and not self._stream_is_capturing()
        )

    @staticmethod
    def _validate_edge(
        producer: str,
        consumer: str,
        expected: ValueContract,
        received: ValueContract,
    ) -> None:
        errors = expected.incompatibilities(received)
        if errors:
            emit_event(
                EventCode.CONTRACT_VALIDATION,
                component_id=consumer,
                producer=producer,
                compatible=False,
                errors=list(errors),
            )
            raise ComponentContractError(producer, consumer, expected, received, errors)
        emit_event(
            EventCode.CONTRACT_VALIDATION,
            component_id=consumer,
            producer=producer,
            compatible=True,
        )

    def forward_sequence(self, value):
        """Compatibility tensor/PyTree path with no training-result objects."""

        for index, module in enumerate(self.components):
            if self._checkpoint_component(index):
                value = checkpoint(module, value, use_reentrant=False)
            else:
                value = module(value)
        return value

    def forward(self, value):
        return self.forward_sequence(value)

    def forward_with_terms(self, value, state=None) -> ComponentOutput:
        """Authoritative training path with fixed flattened result tuples."""

        if state is not None and len(state) != len(self.state_declarations):
            raise ValueError(
                f"graph expected {len(self.state_declarations)} state tensors, "
                f"received {len(state)}"
            )
        losses: list[Tensor] = []
        reports: list[Tensor] = []
        states: list[Tensor] = []
        for index, module in enumerate(self.components):
            binding = self._bindings[index]
            state_method = self._with_state[index]
            terms_method = self._with_terms[index]
            prior_state = None if state is None else tuple(state[self._state_slices[index]])
            if state_method is not None:
                if self._checkpoint_component(index):
                    output = checkpoint(
                        lambda current, carried, _method=state_method, _module=module: _method(
                            _module, current, carried
                        ),
                        value,
                        prior_state,
                        use_reentrant=False,
                    )
                else:
                    output = state_method(module, value, prior_state)
            elif terms_method is not None:
                if self._checkpoint_component(index):
                    output = checkpoint(
                        lambda current, _method=terms_method, _module=module: _method(
                            _module, current
                        ),
                        value,
                        use_reentrant=False,
                    )
                else:
                    output = terms_method(module, value)
            else:
                if self._checkpoint_component(index):
                    value = checkpoint(module, value, use_reentrant=False)
                else:
                    value = module(value)
                continue
            if not isinstance(output, ComponentOutput):
                raise TypeError(
                    f"component {binding.component_id!r} result method must return ComponentOutput"
                )
            if len(output.loss_terms) != len(binding.loss_terms):
                raise RuntimeError(
                    f"component {binding.component_id!r} declared "
                    f"{len(binding.loss_terms)} loss terms but returned "
                    f"{len(output.loss_terms)}"
                )
            if len(output.reports) != len(binding.reports):
                raise RuntimeError(
                    f"component {binding.component_id!r} declared "
                    f"{len(binding.reports)} reports but returned {len(output.reports)}"
                )
            if len(output.next_state) != len(binding.states):
                raise RuntimeError(
                    f"component {binding.component_id!r} declared "
                    f"{len(binding.states)} state tensors but returned "
                    f"{len(output.next_state)}"
                )
            value = output.value
            losses.extend(output.loss_terms)
            reports.extend(output.reports)
            states.extend(output.next_state)
        return ComponentOutput(value, tuple(losses), tuple(reports), tuple(states))

    def forward_with_state(self, value, state=None) -> ComponentOutput:
        """Carry explicit streaming state through the authoritative result path."""

        return self.forward_with_terms(value, state)

    def post_optimizer_step(self, *, step_applied: bool) -> None:
        """Run lifecycle actions only after a real optimizer update."""

        if not step_applied:
            return
        for action in self._post_step:
            if action is not None:
                action()

    def capability_report(self) -> CapabilityReport:
        names = tuple(ComponentCapabilities().execution_modes())
        supported: dict[str, bool] = {name: True for name in names}
        failures: list[CapabilityFailure] = []
        for binding in self._bindings:
            modes = binding.capabilities.execution_modes()
            for name in names:
                if not modes[name]:
                    supported[name] = False
                    failures.append(
                        CapabilityFailure(
                            binding.component_id,
                            name,
                            "choose a provider/backend declaring this capability "
                            "or disable the requested execution mode",
                        )
                    )
        return CapabilityReport(supported, tuple(failures))

    def require_capabilities(self, *modes: str) -> None:
        report = self.capability_report()
        for failure in report.failures:
            if failure.mode in modes:
                emit_event(
                    EventCode.PERFORMANCE_FALLBACK,
                    component_id=failure.component_id,
                    fallback=True,
                    reason=f"component does not declare {failure.mode}",
                    requested_path=failure.mode,
                    selected_path="none",
                    corrective_action=failure.corrective_action,
                )
        report.require(*modes)

    def component_specs(self, *, portable: bool = True) -> tuple[ComponentSpec, ...]:
        if portable:
            return tuple(binding.to_spec() for binding in self._bindings)
        specs = []
        for binding in self._bindings:
            if binding.portable:
                specs.append(binding.to_spec())
        return tuple(specs)

    @property
    def dabsn_memory_count(self) -> int:
        return sum(
            binding.capabilities.dabsn_memory_owner or binding.capabilities.world_builder
            for binding in self._bindings
        )

    @property
    def bindings(self) -> tuple[BoundComponent, ...]:
        return self._bindings


__all__ = [
    "CapabilityFailure",
    "CapabilityReport",
    "DABSNGraph",
    "UnsupportedExecutionModeError",
]
