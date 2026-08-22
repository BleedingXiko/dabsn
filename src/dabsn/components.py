"""Public component, contract, provider, and plugin ABI for DABSN 2.x."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch.nn as nn
from torch import Tensor

from .events import EventCode, emit_event

COMPONENT_ABI_VERSION = 2
COMPONENT_ENTRY_POINT = "dabsn.components.v2"
MAX_COMPONENT_CONFIG_BYTES = 1_048_576
MAX_COMPONENT_CONFIG_DEPTH = 32
MAX_COMPONENT_CONFIG_NODES = 100_000
MAX_COMPONENT_CONFIG_STRING = 65_536


def _validate_component_config(
    value: object,
    *,
    depth: int = 0,
    nodes: list[int] | None = None,
) -> None:
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > MAX_COMPONENT_CONFIG_NODES:
        raise ValueError("component configuration exceeds the node limit")
    if depth > MAX_COMPONENT_CONFIG_DEPTH:
        raise ValueError("component configuration exceeds the nesting limit")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("component configuration contains a non-finite number")
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_COMPONENT_CONFIG_STRING:
            raise ValueError("component configuration string exceeds the size limit")
        return
    if isinstance(value, list):
        for item in value:
            _validate_component_config(item, depth=depth + 1, nodes=nodes)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("component configuration object keys must be strings")
            _validate_component_config(key, depth=depth + 1, nodes=nodes)
            _validate_component_config(item, depth=depth + 1, nodes=nodes)
        return
    raise ValueError(f"component configuration contains unsupported type {type(value).__name__}")


def validate_component_config(config: Mapping[str, object]) -> None:
    data = dict(config)
    _validate_component_config(data)
    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_COMPONENT_CONFIG_BYTES:
        raise ValueError("component configuration exceeds the encoded size limit")


class AxisEffect(str, Enum):
    PRESERVE = "preserve"
    COLLAPSE = "collapse"
    INTRODUCE = "introduce"
    REINTERPRET = "reinterpret"


@dataclass(frozen=True)
class AxisContract:
    """One semantic tensor axis in a :class:`TensorContract`."""

    name: str
    size: int | str | None = None
    dynamic: bool = False
    effect: AxisEffect = AxisEffect.PRESERVE

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("axis name must be a non-empty string")
        if isinstance(self.size, int) and self.size <= 0:
            raise ValueError(f"axis {self.name!r} has non-positive size {self.size}")
        if isinstance(self.size, str) and not self.size:
            raise ValueError("symbolic axis sizes must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "size": self.size,
            "dynamic": self.dynamic,
            "effect": self.effect.value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "AxisContract":
        allowed = {"name", "size", "dynamic", "effect"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"axis contract has unknown fields {sorted(unknown)}")
        size = raw.get("size")
        if size is not None and not isinstance(size, (int, str)):
            raise ValueError("axis-contract size must be an integer, string, or null")
        try:
            effect = AxisEffect(str(raw.get("effect", "preserve")))
        except ValueError as exc:
            raise ValueError(f"invalid axis effect {raw.get('effect')!r}") from exc
        return cls(
            name=str(raw["name"]),
            size=size,
            dynamic=bool(raw.get("dynamic", False)),
            effect=effect,
        )


@dataclass(frozen=True)
class TensorContract:
    """Construction-time contract for one tensor leaf.

    Runtime values remain ordinary tensors.  The contract is never wrapped
    around a value in the graph's forward path.
    """

    axes: tuple[AxisContract, ...]
    dtypes: tuple[str, ...] = ("floating",)
    autocast: str = "inherit"
    device: str = "inherit"
    layouts: tuple[str, ...] = ("strided",)
    strides: tuple[int | str | None, ...] | None = None
    mutation: str = "forbid"
    aliasing: str = "forbid"

    def __post_init__(self) -> None:
        names = [axis.name for axis in self.axes]
        if len(names) != len(set(names)):
            raise ValueError(f"tensor contract repeats semantic axes: {names}")
        if not self.dtypes:
            raise ValueError("tensor contract must accept at least one dtype policy")
        if not self.layouts:
            raise ValueError("tensor contract must accept at least one layout")
        if self.strides is not None and len(self.strides) != len(self.axes):
            raise ValueError("stride declaration must have one entry per axis")
        if self.mutation not in {"forbid", "allow"}:
            raise ValueError("mutation must be 'forbid' or 'allow'")
        if self.aliasing not in {"forbid", "allow", "preserve"}:
            raise ValueError("aliasing must be forbid, allow, or preserve")

    def to_dict(self) -> dict[str, object]:
        return {
            "axes": [axis.to_dict() for axis in self.axes],
            "dtypes": list(self.dtypes),
            "autocast": self.autocast,
            "device": self.device,
            "layouts": list(self.layouts),
            "strides": None if self.strides is None else list(self.strides),
            "mutation": self.mutation,
            "aliasing": self.aliasing,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "TensorContract":
        allowed = {
            "axes",
            "dtypes",
            "autocast",
            "device",
            "layouts",
            "strides",
            "mutation",
            "aliasing",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"tensor contract has unknown fields {sorted(unknown)}")
        raw_axes = raw.get("axes")
        if not isinstance(raw_axes, list) or not all(
            isinstance(item, Mapping) for item in raw_axes
        ):
            raise ValueError("tensor-contract axes must be a list of objects")
        raw_dtypes = raw.get("dtypes", ["floating"])
        raw_layouts = raw.get("layouts", ["strided"])
        raw_strides = raw.get("strides")
        if not isinstance(raw_dtypes, list) or not all(
            isinstance(item, str) for item in raw_dtypes
        ):
            raise ValueError("tensor-contract dtypes must be a list of strings")
        if not isinstance(raw_layouts, list) or not all(
            isinstance(item, str) for item in raw_layouts
        ):
            raise ValueError("tensor-contract layouts must be a list of strings")
        if raw_strides is not None and (
            not isinstance(raw_strides, list)
            or not all(item is None or isinstance(item, (int, str)) for item in raw_strides)
        ):
            raise ValueError("tensor-contract strides must be null or a list")
        return cls(
            axes=tuple(AxisContract.from_dict(item) for item in raw_axes),
            dtypes=tuple(raw_dtypes),
            autocast=str(raw.get("autocast", "inherit")),
            device=str(raw.get("device", "inherit")),
            layouts=tuple(raw_layouts),
            strides=None if raw_strides is None else tuple(raw_strides),
            mutation=str(raw.get("mutation", "forbid")),
            aliasing=str(raw.get("aliasing", "forbid")),
        )


@dataclass(frozen=True)
class ValueContract:
    """Immutable PyTree contract represented by ordered tensor leaves."""

    leaves: tuple[TensorContract, ...]
    tree: object = "tensor"
    streaming_state: tuple[TensorContract, ...] = ()

    def __post_init__(self) -> None:
        if not self.leaves:
            raise ValueError("value contract must contain at least one tensor leaf")

    @classmethod
    def tensor(
        cls,
        *axes: AxisContract,
        dtypes: tuple[str, ...] = ("floating",),
        autocast: str = "inherit",
        device: str = "inherit",
        layouts: tuple[str, ...] = ("strided",),
        strides: tuple[int | str | None, ...] | None = None,
        mutation: str = "forbid",
        aliasing: str = "forbid",
    ) -> "ValueContract":
        return cls(
            (
                TensorContract(
                    tuple(axes),
                    dtypes=dtypes,
                    autocast=autocast,
                    device=device,
                    layouts=layouts,
                    strides=strides,
                    mutation=mutation,
                    aliasing=aliasing,
                ),
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tree": self.tree,
            "leaves": [leaf.to_dict() for leaf in self.leaves],
            "streaming_state": [leaf.to_dict() for leaf in self.streaming_state],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ValueContract":
        allowed = {"tree", "leaves", "streaming_state"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"value contract has unknown fields {sorted(unknown)}")
        raw_leaves = raw.get("leaves")
        raw_state = raw.get("streaming_state", [])
        if not isinstance(raw_leaves, list) or not all(
            isinstance(item, Mapping) for item in raw_leaves
        ):
            raise ValueError("value-contract leaves must be a list of objects")
        if not isinstance(raw_state, list) or not all(
            isinstance(item, Mapping) for item in raw_state
        ):
            raise ValueError("value-contract streaming state must be a list of objects")
        return cls(
            leaves=tuple(TensorContract.from_dict(item) for item in raw_leaves),
            tree=raw.get("tree", "tensor"),
            streaming_state=tuple(TensorContract.from_dict(item) for item in raw_state),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def incompatibilities(self, received: "ValueContract") -> tuple[str, ...]:
        errors: list[str] = []
        if self.tree != received.tree:
            errors.append(f"PyTree expected {self.tree!r}, received {received.tree!r}")
        if len(self.leaves) != len(received.leaves):
            errors.append(
                f"leaf count expected {len(self.leaves)}, received {len(received.leaves)}"
            )
            return tuple(errors)
        for leaf_index, (expected, actual) in enumerate(zip(self.leaves, received.leaves)):
            if len(expected.axes) != len(actual.axes):
                errors.append(
                    f"leaf {leaf_index} rank expected {len(expected.axes)}, "
                    f"received {len(actual.axes)}"
                )
                continue
            for axis_index, (want, got) in enumerate(zip(expected.axes, actual.axes)):
                if want.name != got.name:
                    errors.append(
                        f"leaf {leaf_index} axis {axis_index} expected semantic axis "
                        f"{want.name!r}, received {got.name!r}"
                    )
                if want.size is not None and got.size is not None and want.size != got.size:
                    errors.append(
                        f"leaf {leaf_index} axis {want.name!r} expected size "
                        f"{want.size!r}, received {got.size!r}"
                    )
            if "any" not in expected.dtypes and not set(expected.dtypes).intersection(
                actual.dtypes
            ):
                errors.append(
                    f"leaf {leaf_index} dtype expected one of {expected.dtypes}, "
                    f"received {actual.dtypes}"
                )
            if expected.device != "any" and actual.device not in {
                expected.device,
                "inherit",
            }:
                errors.append(
                    f"leaf {leaf_index} device expected {expected.device!r}, "
                    f"received {actual.device!r}"
                )
            if not set(expected.layouts).intersection(actual.layouts):
                errors.append(
                    f"leaf {leaf_index} layout expected one of {expected.layouts}, "
                    f"received {actual.layouts}"
                )
        return tuple(errors)

    def accepts(self, received: "ValueContract") -> bool:
        return not self.incompatibilities(received)


@dataclass(frozen=True)
class ComponentContract:
    input: ValueContract
    output: ValueContract


@dataclass(frozen=True)
class ResultDeclaration:
    name: str
    reduction: str = "mean"
    distributed_owner: str = "framework"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("result declarations require a name")
        if self.reduction not in {"none", "sum", "mean", "max", "min"}:
            raise ValueError(f"unsupported reduction {self.reduction!r}")
        if self.distributed_owner not in {"framework", "component", "none"}:
            raise ValueError("distributed_owner must be framework, component, or none")


@dataclass(frozen=True)
class StateDeclaration:
    """One named, fixed-position streaming-state tensor."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("state declarations require a name")


@dataclass(frozen=True)
class ComponentOutput:
    """Fixed training result returned by a component or graph."""

    value: object
    loss_terms: tuple[Tensor, ...] = ()
    reports: tuple[Tensor, ...] = ()
    next_state: tuple[Tensor, ...] = ()


@dataclass(frozen=True)
class ParallelAxis:
    """One named dimension of a multi-worker execution topology."""

    name: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parallel-axis name must be a non-empty string")
        if int(self.size) <= 0:
            raise ValueError(f"parallel-axis {self.name!r} size must be positive")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "size": int(self.size)}


@dataclass(frozen=True)
class ParallelTopology:
    """Architecture-neutral named mesh shared by framework and providers."""

    axes: tuple[ParallelAxis, ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("parallel topology requires at least one axis")
        names = [axis.name for axis in self.axes]
        if len(names) != len(set(names)):
            raise ValueError(f"parallel topology repeats axis names: {names}")

    @property
    def world_size(self) -> int:
        size = 1
        for axis in self.axes:
            size *= int(axis.size)
        return size

    def axis_size(self, name: str) -> int:
        for axis in self.axes:
            if axis.name == name:
                return int(axis.size)
        raise KeyError(f"parallel topology has no axis {name!r}")

    def coordinate(self, rank: int) -> Mapping[str, int]:
        if not 0 <= int(rank) < self.world_size:
            raise ValueError(f"rank {rank} is outside topology world size {self.world_size}")
        remainder = int(rank)
        values: dict[str, int] = {}
        for axis in reversed(self.axes):
            values[axis.name] = remainder % int(axis.size)
            remainder //= int(axis.size)
        return MappingProxyType({axis.name: values[axis.name] for axis in self.axes})

    def to_dict(self) -> dict[str, object]:
        return {"axes": [axis.to_dict() for axis in self.axes]}

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParallelDirective:
    """One provider-owned distribution action over named topology axes."""

    directive_id: str
    kind: str
    mesh_axes: tuple[str, ...]
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.directive_id, str) or not self.directive_id:
            raise ValueError("parallel-directive ID must be a non-empty string")
        if self.kind not in {"tensor", "expert", "replicate", "communication"}:
            raise ValueError(
                "parallel-directive kind must be tensor, expert, replicate, or communication"
            )
        if not self.mesh_axes or any(
            not isinstance(axis, str) or not axis for axis in self.mesh_axes
        ):
            raise ValueError("parallel directives require non-empty mesh-axis names")
        if len(self.mesh_axes) != len(set(self.mesh_axes)):
            raise ValueError("parallel-directive mesh axes must be unique")
        validate_component_config(self.config)
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))

    def to_dict(self) -> dict[str, object]:
        return {
            "directive_id": self.directive_id,
            "kind": self.kind,
            "mesh_axes": list(self.mesh_axes),
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class ParallelPlan:
    fsdp_boundary: bool = True
    directives: tuple[ParallelDirective, ...] = ()
    activation_checkpoint_boundary: bool = False

    def __post_init__(self) -> None:
        ids = [directive.directive_id for directive in self.directives]
        if len(ids) != len(set(ids)):
            raise ValueError(f"parallel plan repeats directive IDs: {ids}")

    def to_dict(self) -> dict[str, object]:
        return {
            "fsdp_boundary": self.fsdp_boundary,
            "directives": [directive.to_dict() for directive in self.directives],
            "activation_checkpoint_boundary": self.activation_checkpoint_boundary,
        }


@dataclass(frozen=True)
class ParallelExecutionContext:
    """Resolved topology and this worker's provider-visible communication groups."""

    topology: ParallelTopology
    rank: int
    device: object
    coordinate: Mapping[str, int]
    groups: Mapping[str, object]
    group_ranks: Mapping[str, tuple[int, ...]]

    def group(self, axis: str) -> object:
        try:
            return self.groups[axis]
        except KeyError as exc:
            raise KeyError(f"no communication group was resolved for axis {axis!r}") from exc


@dataclass(frozen=True)
class ParallelExecutionReceipt:
    """Provider proof that it executed every requested distribution directive."""

    consumed_directives: tuple[str, ...]
    parameter_sharding: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not isinstance(name, str) or not name for name in self.consumed_directives):
            raise ValueError("consumed parallel-directive IDs must be non-empty strings")
        if len(self.consumed_directives) != len(set(self.consumed_directives)):
            raise ValueError("consumed parallel-directive IDs must be unique")
        normalized: dict[str, tuple[str, ...]] = {}
        for name, axes in self.parameter_sharding.items():
            if not isinstance(name, str) or not name:
                raise ValueError("parallel parameter names must be non-empty strings")
            axis_tuple = tuple(axes)
            if any(not isinstance(axis, str) or not axis for axis in axis_tuple):
                raise ValueError("parallel parameter axes must be non-empty strings")
            if len(axis_tuple) != len(set(axis_tuple)):
                raise ValueError(f"parallel parameter {name!r} repeats sharding axes")
            normalized[name] = axis_tuple
        object.__setattr__(self, "parameter_sharding", MappingProxyType(normalized))


@runtime_checkable
class ParallelPlanExecutor(Protocol):
    def __call__(
        self,
        module: nn.Module,
        config: Mapping[str, object],
        plan: ParallelPlan,
        context: ParallelExecutionContext,
    ) -> ParallelExecutionReceipt: ...


@runtime_checkable
class ParallelStateConsolidator(Protocol):
    def __call__(
        self,
        module: nn.Module,
        config: Mapping[str, object],
        local_state: Mapping[str, Tensor],
        context: ParallelExecutionContext,
    ) -> Mapping[str, Tensor]: ...


@dataclass(frozen=True)
class ComponentCapabilities:
    eager: bool = True
    compile_fullgraph: bool = False
    dynamic_shapes: bool = False
    export: bool = False
    cuda_graph: bool = False
    activation_checkpoint: bool = False
    amp_fp32: bool = True
    amp_bf16: bool = False
    amp_fp16: bool = False
    distributed: bool = False
    streaming_state: bool = False
    world_builder: bool = False
    dabsn_memory_owner: bool = False
    deterministic: bool | None = None
    parallel_plan: ParallelPlan | None = None

    def __post_init__(self) -> None:
        if self.parallel_plan is not None and not self.distributed:
            raise ValueError("a parallel plan requires distributed=True")
        if (
            self.parallel_plan is not None
            and self.parallel_plan.activation_checkpoint_boundary
            and not self.activation_checkpoint
        ):
            raise ValueError(
                "an activation-checkpoint boundary requires activation_checkpoint=True"
            )

    def execution_modes(self) -> Mapping[str, bool]:
        return MappingProxyType(
            {
                "eager": self.eager,
                "compile_fullgraph": self.compile_fullgraph,
                "dynamic_shapes": self.dynamic_shapes,
                "export": self.export,
                "cuda_graph": self.cuda_graph,
                "activation_checkpoint": self.activation_checkpoint,
                "amp_fp32": self.amp_fp32,
                "amp_bf16": self.amp_bf16,
                "amp_fp16": self.amp_fp16,
                "distributed": self.distributed,
                "streaming_state": self.streaming_state,
            }
        )


@dataclass(frozen=True)
class BuildContext:
    device: object | None = None
    dtype: object | None = None
    strict: bool = True


@runtime_checkable
class ComponentProvider(Protocol):
    provider_key: str
    component_abi_version: int
    config_schema_version: int

    def validate_config(self, config: Mapping[str, object]) -> None: ...

    def contract(self, config: Mapping[str, object]) -> ComponentContract: ...

    def build(self, config: Mapping[str, object], context: BuildContext) -> nn.Module: ...

    def migrate_config(
        self, old_version: int, config: Mapping[str, object]
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    provider_key: str
    config: Mapping[str, object]
    component_abi_version: int = COMPONENT_ABI_VERSION
    config_schema_version: int = 1
    provider_distribution: str | None = None
    provider_version: str | None = None


@dataclass
class BoundComponent:
    component_id: str
    module: nn.Module
    contract: ComponentContract
    capabilities: ComponentCapabilities = field(default_factory=ComponentCapabilities)
    loss_terms: tuple[ResultDeclaration, ...] = ()
    reports: tuple[ResultDeclaration, ...] = ()
    states: tuple[StateDeclaration, ...] = ()
    provider_key: str | None = None
    provider_distribution: str | None = None
    provider_version: str | None = None
    component_abi_version: int = COMPONENT_ABI_VERSION
    config_schema_version: int = 1
    config: Mapping[str, object] = field(default_factory=dict)
    portable: bool = False
    parallel_executor: ParallelPlanExecutor | None = None
    parallel_state_consolidator: ParallelStateConsolidator | None = None
    parallel_execution_receipt: ParallelExecutionReceipt | None = None
    parallelized_topology: str | None = None

    def to_spec(self) -> ComponentSpec:
        if not self.portable or self.provider_key is None:
            raise AnonymousComponentError(
                f"component {self.component_id!r} is an anonymous module and cannot "
                "be saved as a portable artifact; register a ComponentProvider"
            )
        return ComponentSpec(
            component_id=self.component_id,
            provider_key=self.provider_key,
            config=dict(self.config),
            component_abi_version=self.component_abi_version,
            config_schema_version=self.config_schema_version,
            provider_distribution=self.provider_distribution,
            provider_version=self.provider_version,
        )


class ComponentError(RuntimeError):
    pass


class ComponentContractError(ComponentError):
    def __init__(
        self,
        producer: str,
        consumer: str,
        expected: ValueContract,
        received: ValueContract,
        details: Sequence[str],
    ) -> None:
        self.producer = producer
        self.consumer = consumer
        self.expected = expected
        self.received = received
        self.details = tuple(details)
        joined = "; ".join(self.details)
        super().__init__(
            f"component contract mismatch between {producer!r} and {consumer!r}: "
            f"{joined}; expected={expected.to_dict()!r}; received={received.to_dict()!r}"
        )


class MissingProviderError(ComponentError):
    pass


class IncompatibleProviderError(ComponentError):
    pass


class UntrustedProviderError(ComponentError):
    pass


class AnonymousComponentError(ComponentError):
    pass


@dataclass
class _ProviderRecord:
    key: str
    provider: ComponentProvider | None
    trusted: bool
    distribution: str | None = None
    version: str | None = None
    entry_point: importlib.metadata.EntryPoint | None = None


class ComponentRegistry:
    """Construction-time provider registry; never consulted during forward."""

    def __init__(self) -> None:
        self._records: dict[str, _ProviderRecord] = {}

    def register(
        self,
        provider: ComponentProvider,
        *,
        trusted: bool = True,
        distribution: str | None = None,
        version: str | None = None,
        replace: bool = False,
    ) -> None:
        if not isinstance(provider, ComponentProvider):
            raise TypeError("provider does not implement the ComponentProvider ABI")
        key = provider.provider_key
        if not key or not any(separator in key for separator in (":", ".")):
            raise ValueError(
                "provider_key must be namespaced, for example 'acme:block' or 'acme.block'"
            )
        if key in self._records and not replace:
            raise ValueError(f"component provider {key!r} is already registered")
        if provider.component_abi_version != COMPONENT_ABI_VERSION:
            raise IncompatibleProviderError(
                f"provider {key!r} implements component ABI "
                f"{provider.component_abi_version}, required {COMPONENT_ABI_VERSION}"
            )
        self._records[key] = _ProviderRecord(key, provider, trusted, distribution, version)

    def discover(self) -> tuple[str, ...]:
        """Index entry points without importing or executing provider code."""

        points = importlib.metadata.entry_points()
        selected = points.select(group=COMPONENT_ENTRY_POINT)
        found: list[str] = []
        for point in selected:
            key = point.name
            found.append(key)
            if key in self._records:
                continue
            distribution = getattr(point, "dist", None)
            self._records[key] = _ProviderRecord(
                key=key,
                provider=None,
                trusted=False,
                distribution=getattr(distribution, "name", None),
                version=getattr(distribution, "version", None),
                entry_point=point,
            )
        return tuple(sorted(found))

    def authorize(self, provider_key: str) -> None:
        try:
            self._records[provider_key].trusted = True
        except KeyError as exc:
            raise MissingProviderError(
                f"component provider {provider_key!r} is not installed or registered"
            ) from exc

    def resolve(self, provider_key: str, *, required_abi: int = 2) -> _ProviderRecord:
        try:
            record = self._records[provider_key]
        except KeyError as exc:
            raise MissingProviderError(
                f"component provider {provider_key!r} is not installed or registered"
            ) from exc
        if not record.trusted:
            raise UntrustedProviderError(
                f"component provider {provider_key!r} is installed but untrusted; "
                "explicitly register or authorize it before construction"
            )
        if record.provider is None:
            if record.entry_point is None:
                raise MissingProviderError(f"provider {provider_key!r} has no loader")
            loaded = record.entry_point.load()
            provider = loaded() if isinstance(loaded, type) else loaded
            if not isinstance(provider, ComponentProvider):
                raise IncompatibleProviderError(
                    f"entry point for {provider_key!r} does not implement ComponentProvider"
                )
            record.provider = provider
        if record.provider.component_abi_version != required_abi:
            raise IncompatibleProviderError(
                f"provider {provider_key!r} has ABI "
                f"{record.provider.component_abi_version}, checkpoint requires {required_abi}"
            )
        emit_event(
            EventCode.PROVIDER_RESOLUTION,
            component_id=provider_key,
            provider_distribution=record.distribution,
            provider_version=record.version,
            component_abi_version=required_abi,
        )
        return record

    def build(self, spec: ComponentSpec, context: BuildContext | None = None) -> BoundComponent:
        validate_component_config(spec.config)
        record = self.resolve(spec.provider_key, required_abi=spec.component_abi_version)
        if (
            spec.provider_distribution is not None
            and record.distribution is not None
            and spec.provider_distribution != record.distribution
        ):
            raise IncompatibleProviderError(
                f"provider {spec.provider_key!r} requires distribution "
                f"{spec.provider_distribution!r}, installed {record.distribution!r}"
            )
        if (
            spec.provider_version is not None
            and record.version is not None
            and spec.provider_version != record.version
        ):
            raise IncompatibleProviderError(
                f"provider {spec.provider_key!r} requires version "
                f"{spec.provider_version!r}, installed {record.version!r}"
            )
        provider = record.provider
        assert provider is not None
        config: Mapping[str, object] = dict(spec.config)
        if spec.config_schema_version != provider.config_schema_version:
            config = provider.migrate_config(spec.config_schema_version, config)
            validate_component_config(config)
        provider.validate_config(config)
        contract = provider.contract(config)
        build_context = context or BuildContext()
        module = provider.build(config, build_context)
        capabilities_builder = getattr(provider, "capabilities_for_config", None)
        if callable(capabilities_builder):
            parameters = inspect.signature(capabilities_builder).parameters
            capabilities = (
                capabilities_builder(config, build_context)
                if len(parameters) >= 2
                else capabilities_builder(config)
            )
        else:
            capabilities = getattr(provider, "capabilities", ComponentCapabilities())
        loss_terms = tuple(getattr(provider, "loss_terms", ()))
        reports = tuple(getattr(provider, "reports", ()))
        states = tuple(getattr(provider, "states", ()))
        parallel_executor = getattr(provider, "parallelize", None)
        if parallel_executor is not None and not callable(parallel_executor):
            raise TypeError(f"provider {spec.provider_key!r} parallelize attribute is not callable")
        parallel_state_consolidator = getattr(provider, "consolidate_parallel_state", None)
        if parallel_state_consolidator is not None and not callable(parallel_state_consolidator):
            raise TypeError(
                f"provider {spec.provider_key!r} consolidate_parallel_state attribute "
                "is not callable"
            )
        state_names = [declaration.name for declaration in states]
        if len(state_names) != len(set(state_names)):
            raise ValueError(f"provider {spec.provider_key!r} declares duplicate state names")
        if bool(states) != capabilities.streaming_state:
            raise ValueError(
                f"provider {spec.provider_key!r} must declare state slots exactly "
                "when streaming_state capability is enabled"
            )
        return BoundComponent(
            component_id=spec.component_id,
            module=module,
            contract=contract,
            capabilities=capabilities,
            loss_terms=loss_terms,
            reports=reports,
            states=states,
            provider_key=spec.provider_key,
            provider_distribution=record.distribution or spec.provider_distribution,
            provider_version=record.version or spec.provider_version,
            component_abi_version=provider.component_abi_version,
            config_schema_version=provider.config_schema_version,
            config=MappingProxyType(dict(config)),
            portable=True,
            parallel_executor=parallel_executor,
            parallel_state_consolidator=parallel_state_consolidator,
        )

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))


component_registry = ComponentRegistry()


def register_component(
    provider: ComponentProvider,
    *,
    trusted: bool = True,
    distribution: str | None = None,
    version: str | None = None,
    replace: bool = False,
) -> None:
    component_registry.register(
        provider,
        trusted=trusted,
        distribution=distribution,
        version=version,
        replace=replace,
    )


def bind_module(
    component_id: str,
    module: nn.Module,
    contract: ComponentContract,
    *,
    capabilities: ComponentCapabilities | None = None,
    loss_terms: Sequence[ResultDeclaration] = (),
    reports: Sequence[ResultDeclaration] = (),
    states: Sequence[StateDeclaration] = (),
    parallel_executor: ParallelPlanExecutor | None = None,
) -> BoundComponent:
    """Bind an ordinary module without inserting a runtime tensor wrapper."""

    if not component_id:
        raise ValueError("component_id must be non-empty")
    state_slots = tuple(states)
    state_names = [declaration.name for declaration in state_slots]
    if len(state_names) != len(set(state_names)):
        raise ValueError("state declaration names must be unique per component")
    resolved_capabilities = capabilities or ComponentCapabilities()
    if bool(state_slots) != resolved_capabilities.streaming_state:
        raise ValueError(
            "state slots must be declared exactly when streaming_state capability is enabled"
        )
    return BoundComponent(
        component_id=component_id,
        module=module,
        contract=contract,
        capabilities=resolved_capabilities,
        loss_terms=tuple(loss_terms),
        reports=tuple(reports),
        states=state_slots,
        portable=False,
        parallel_executor=parallel_executor,
    )


def canonical_component_config(config: Mapping[str, object]) -> str:
    """Canonical non-executable JSON used by checkpoint metadata."""

    return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "COMPONENT_ABI_VERSION",
    "COMPONENT_ENTRY_POINT",
    "AnonymousComponentError",
    "AxisContract",
    "AxisEffect",
    "BoundComponent",
    "BuildContext",
    "ComponentCapabilities",
    "ComponentContract",
    "ComponentContractError",
    "ComponentError",
    "ComponentOutput",
    "ComponentProvider",
    "ComponentRegistry",
    "ComponentSpec",
    "IncompatibleProviderError",
    "MissingProviderError",
    "ParallelAxis",
    "ParallelDirective",
    "ParallelExecutionContext",
    "ParallelExecutionReceipt",
    "ParallelPlan",
    "ParallelPlanExecutor",
    "ParallelStateConsolidator",
    "ParallelTopology",
    "ResultDeclaration",
    "StateDeclaration",
    "TensorContract",
    "UntrustedProviderError",
    "ValueContract",
    "bind_module",
    "canonical_component_config",
    "component_registry",
    "register_component",
    "validate_component_config",
]
