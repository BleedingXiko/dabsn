"""Architecture-specific online temporal-credit state for DABSN cores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .core import DABSNCore

CORE_ELIGIBILITY_SCHEMA_VERSION = 1
LONG_READ_ELIGIBILITY_SCHEMA_VERSION = 1
ONLINE_TEMPORAL_STATE_SCHEMA_VERSION = 1
ONLINE_TEMPORAL_STATE_FORMAT = "dabsn-online-temporal-state"
ONLINE_TRAINING_STATE_FORMAT = "dabsn-online-training-state"

# These are the sixteen per-channel parameters represented by the three-state
# local trace in the proven full-hybrid rule. The order is part of the state ABI.
CORE_LOCAL_PARAMETER_NAMES = (
    "Wg.bias",
    "beta",
    "log_kappa",
    "logit_recover",
    "k_s",
    "k_y",
    "k_b",
    "k_n",
    "k_bias",
    "r_s",
    "r_y",
    "r_b",
    "r_n",
    "r_bias",
    "k_saturation",
    "r_saturation",
)

# These four shared controls use a three-state trace for every state channel.
CORE_GLOBAL_PARAMETER_NAMES = (
    "logit_alpha",
    "log_lambda",
    "logit_saturation_decay",
    "logit_saturation_suppress",
)


@dataclass(frozen=True)
class CoreEligibilityState:
    """One independent online eligibility state per batch lane.

    Large matrix parameters retain the discovered budget-only trace. Local and
    shared recurrence controls retain budget, energy, and saturation traces.
    No dimension depends on sequence length.
    """

    w_budget: Tensor
    a_budget: Tensor
    wg_budget: Tensor
    ug_budget: Tensor
    local: Tensor
    global_: Tensor
    schema_version: int = CORE_ELIGIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        tensors = self.tensors()
        if int(self.schema_version) != CORE_ELIGIBILITY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported core eligibility schema_version "
                f"{self.schema_version}; expected {CORE_ELIGIBILITY_SCHEMA_VERSION}"
            )
        if any(tensor.dim() != rank for tensor, rank in zip(tensors, (3, 3, 3, 3, 4, 4))):
            raise ValueError("core eligibility tensors have invalid ranks")

        batch = int(self.a_budget.shape[0])
        state = int(self.a_budget.shape[1])
        input_dim = int(self.wg_budget.shape[2])
        expected = {
            "w_budget": (batch, state, input_dim),
            "a_budget": (batch, state, state),
            "wg_budget": (batch, state, input_dim),
            "ug_budget": (batch, state, state),
            "local": (batch, state, 3, len(CORE_LOCAL_PARAMETER_NAMES)),
            "global_": (batch, state, 3, len(CORE_GLOBAL_PARAMETER_NAMES)),
        }
        for name, tensor in zip(
            ("w_budget", "a_budget", "wg_budget", "ug_budget", "local", "global_"),
            tensors,
        ):
            if tuple(tensor.shape) != expected[name]:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)}, expected {expected[name]}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        reference = tensors[0]
        if any(
            tensor.dtype != reference.dtype or tensor.device != reference.device
            for tensor in tensors[1:]
        ):
            raise ValueError("core eligibility tensors must have the same dtype and device")
        if not reference.is_floating_point():
            raise ValueError("core eligibility tensors must use a floating dtype")

    @property
    def batch_size(self) -> int:
        return int(self.w_budget.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.w_budget.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.w_budget.shape[2])

    @property
    def dtype(self) -> torch.dtype:
        return self.w_budget.dtype

    @property
    def device(self) -> torch.device:
        return self.w_budget.device

    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.w_budget,
            self.a_budget,
            self.wg_budget,
            self.ug_budget,
            self.local,
            self.global_,
        )

    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors())


@dataclass(frozen=True)
class LongReadEligibilityState:
    """Exact three-state eligibility for the four shared long-read controls."""

    traces: Tensor
    schema_version: int = LONG_READ_ELIGIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != LONG_READ_ELIGIBILITY_SCHEMA_VERSION:
            raise ValueError(
                "unsupported long-read eligibility schema_version "
                f"{self.schema_version}; expected {LONG_READ_ELIGIBILITY_SCHEMA_VERSION}"
            )
        if self.traces.dim() != 4 or self.traces.shape[2:] != (3, 4):
            raise ValueError("long-read eligibility traces must have shape [B,H,3,4]")
        if not self.traces.is_contiguous():
            raise ValueError("long-read eligibility traces must be contiguous")
        if not self.traces.is_floating_point():
            raise ValueError("long-read eligibility traces must use a floating dtype")

    @property
    def batch_size(self) -> int:
        return int(self.traces.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.traces.shape[1])

    def nbytes(self) -> int:
        return self.traces.numel() * self.traces.element_size()


@dataclass(frozen=True)
class LongReadOnlineTape:
    """Detached values for one bounded long-read recurrence chunk."""

    write: Tensor
    plasticity: Tensor
    novelty: Tensor
    long: Tensor
    expected: Tensor
    retention: Tensor

    def __post_init__(self) -> None:
        reference_shape = tuple(self.write.shape)
        if self.write.dim() != 3:
            raise ValueError("long-read tape tensors must have shape [B,T,H]")
        for name, value in (
            ("plasticity", self.plasticity),
            ("novelty", self.novelty),
            ("long", self.long),
            ("expected", self.expected),
            ("retention", self.retention),
        ):
            if tuple(value.shape) != reference_shape:
                raise ValueError(f"long-read tape {name} must have shape {reference_shape}")
            if value.grad_fn is not None:
                raise ValueError(f"long-read tape {name} must be detached")

    @property
    def batch_size(self) -> int:
        return int(self.write.shape[0])

    @property
    def steps(self) -> int:
        return int(self.write.shape[1])

    @property
    def hidden_dim(self) -> int:
        return int(self.write.shape[2])


def _validate_recurrent_tuple(
    name: str,
    values: tuple[Tensor, Tensor, Tensor],
    expected: tuple[int, int],
) -> None:
    if len(values) != 3 or any(tuple(value.shape) != expected for value in values):
        shapes = tuple(tuple(value.shape) for value in values)
        raise ValueError(f"{name} must contain three tensors shaped {expected}, got {shapes}")
    reference = values[0]
    if any(value.dtype != reference.dtype or value.device != reference.device for value in values):
        raise ValueError(f"{name} tensors must share dtype and device")
    if any(value.grad_fn is not None for value in values):
        raise ValueError(f"{name} tensors must be detached at an online chunk boundary")


@dataclass(frozen=True)
class OnlineLayerState:
    """Detached recurrent and eligibility state owned by one DABSN block."""

    core_recurrent: tuple[Tensor, Tensor, Tensor]
    core_eligibility: CoreEligibilityState
    long_recurrent: tuple[Tensor, Tensor, Tensor]
    long_eligibility: LongReadEligibilityState
    read_bank: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        batch = self.core_eligibility.batch_size
        state = self.core_eligibility.state_dim
        _validate_recurrent_tuple("core_recurrent", self.core_recurrent, (batch, state))
        if self.long_eligibility.batch_size != batch:
            raise ValueError("core and long-read eligibility batch sizes must match")
        _validate_recurrent_tuple(
            "long_recurrent",
            self.long_recurrent,
            (batch, self.long_eligibility.hidden_dim),
        )
        if self.read_bank:
            expected_names = {
                "write",
                "novelty_mean",
                "plasticity_mean",
                "energy_mean",
                "saturation_mean",
                "valid",
                "position",
            }
            if set(self.read_bank) != expected_names:
                raise ValueError("online read_bank has an invalid field schema")
            write = self.read_bank["write"]
            if write.dim() != 3 or write.shape[0] != batch:
                raise ValueError("online read_bank write must have shape [B,N,H]")
            entries = int(write.shape[1])
            for name in expected_names - {"write", "valid", "position"}:
                value = self.read_bank[name]
                if tuple(value.shape) != (batch, entries):
                    raise ValueError(f"online read_bank {name} must have shape [B,N]")
            if (
                self.read_bank["valid"].dtype != torch.bool
                or tuple(self.read_bank["valid"].shape) != (batch, entries)
            ):
                raise ValueError("online read_bank valid must be bool [B,N]")
            if (
                self.read_bank["position"].dtype != torch.int64
                or tuple(self.read_bank["position"].shape) != (batch, entries)
            ):
                raise ValueError("online read_bank position must be int64 [B,N]")
            if any(value.grad_fn is not None for value in self.read_bank.values()):
                raise ValueError("online read_bank values must be detached")

    @property
    def batch_size(self) -> int:
        return self.core_eligibility.batch_size

    def nbytes(self) -> int:
        recurrent = sum(
            value.numel() * value.element_size()
            for value in (*self.core_recurrent, *self.long_recurrent)
        )
        bank = sum(value.numel() * value.element_size() for value in self.read_bank.values())
        return recurrent + bank + self.core_eligibility.nbytes() + self.long_eligibility.nbytes()


@dataclass(frozen=True)
class OnlineTemporalState:
    """Versioned online state for stable independent stream lanes."""

    layers: tuple[OnlineLayerState, ...]
    stream_policy: str
    stream_ids: tuple[str | int, ...]
    pending_reset_mask: Tensor
    chunk_position: int = 0
    accumulation_position: int = 0
    temporal_chunk_size: int | None = None
    eligibility_batch_mode: str = "vectorized"
    eligibility_lane_group_size: int | None = None
    active_frames: tuple[OnlineCoreFrame, ...] = field(
        default_factory=tuple,
        compare=False,
        repr=False,
    )
    schema_version: int = ONLINE_TEMPORAL_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != ONLINE_TEMPORAL_STATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported online temporal state schema_version "
                f"{self.schema_version}; expected {ONLINE_TEMPORAL_STATE_SCHEMA_VERSION}"
            )
        if self.stream_policy not in {"reset_each_batch", "carry_lanes"}:
            raise ValueError("stream_policy must be reset_each_batch or carry_lanes")
        if not self.layers:
            raise ValueError("online temporal state requires at least one layer")
        batch = self.layers[0].batch_size
        if any(layer.batch_size != batch for layer in self.layers):
            raise ValueError("all online layer states must have the same batch size")
        if len(self.stream_ids) != batch:
            raise ValueError(f"stream_ids must contain exactly {batch} lane identifiers")
        if (
            self.pending_reset_mask.dtype != torch.bool
            or tuple(self.pending_reset_mask.shape) != (batch,)
            or not self.pending_reset_mask.is_contiguous()
        ):
            raise ValueError("pending_reset_mask must be a contiguous bool tensor shaped [B]")
        if int(self.chunk_position) < 0 or int(self.accumulation_position) < 0:
            raise ValueError("chunk_position and accumulation_position must be non-negative")
        if self.temporal_chunk_size is not None and int(self.temporal_chunk_size) <= 0:
            raise ValueError("temporal_chunk_size must be positive when bound")
        if self.eligibility_batch_mode not in {"vectorized", "lane_grouped"}:
            raise ValueError("eligibility_batch_mode must be vectorized or lane_grouped")
        if self.eligibility_batch_mode == "vectorized":
            if self.eligibility_lane_group_size is not None:
                raise ValueError("vectorized online state cannot have a lane group size")
        elif (
            self.eligibility_lane_group_size is None
            or int(self.eligibility_lane_group_size) <= 0
        ):
            raise ValueError("lane_grouped online state requires a positive group size")

    @property
    def batch_size(self) -> int:
        return self.layers[0].batch_size

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers) + (
            self.pending_reset_mask.numel() * self.pending_reset_mask.element_size()
        )

    def assert_checkpointable(self) -> None:
        if any(frame.active or not frame.replayed for frame in self.active_frames):
            raise RuntimeError("cannot checkpoint with a pending online replay")


@dataclass(frozen=True)
class CoreForwardResult:
    """Canonical core signals plus the recurrent state after the chunk."""

    trajectory: Tensor
    novelty: Tensor
    plasticity: Tensor
    expression: Tensor
    write: Tensor
    energy: Tensor
    saturation: Tensor
    final_state: tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class CoreOnlineTape:
    """Detached per-step values needed by eligibility-only replay."""

    inputs: Tensor
    budget: Tensor
    energy: Tensor
    saturation: Tensor
    y: Tensor
    expression: Tensor
    gate_preactivation: Tensor
    gate: Tensor
    novelty: Tensor
    next_saturation: Tensor
    novelty_effective: Tensor
    write_cost: Tensor
    recovery: Tensor
    plasticity: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def steps(self) -> int:
        return int(self.inputs.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.inputs.shape[2])

    @property
    def state_dim(self) -> int:
        return int(self.y.shape[2])

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.inputs,
            self.budget,
            self.energy,
            self.saturation,
            self.y,
            self.expression,
            self.gate_preactivation,
            self.gate,
            self.novelty,
            self.next_saturation,
            self.novelty_effective,
            self.write_cost,
            self.recovery,
            self.plasticity,
        )


@dataclass(frozen=True)
class CoreSignalGrads:
    """Spatial cotangents arriving at canonical core outputs.

    The first reference gate pins the supplied experiment's ``y`` path. The
    remaining fields reserve the native DABSN boundary and are implemented by
    the full local-VJP gate before online mode can be dispatched by a model.
    """

    y: Tensor | None = None
    next_budget: Tensor | None = None
    novelty: Tensor | None = None
    plasticity: Tensor | None = None
    expression: Tensor | None = None
    write: Tensor | None = None
    energy: Tensor | None = None
    saturation: Tensor | None = None


@dataclass
class OnlineCoreFrame:
    """Single-use owner for eligibility produced by one online backward."""

    initial_eligibility: CoreEligibilityState
    resulting_eligibility: CoreEligibilityState | None = None
    active: bool = False
    replayed: bool = False

    def begin(self, eligibility: CoreEligibilityState) -> None:
        if self.replayed:
            raise RuntimeError("online core frame was already replayed")
        if self.active:
            raise RuntimeError("online core frame is already active")
        if eligibility is not self.initial_eligibility:
            raise ValueError("online core frame received a different eligibility state")
        self.active = True

    def complete(self, eligibility: CoreEligibilityState) -> None:
        if not self.active or self.replayed:
            raise RuntimeError("online core frame cannot be completed in its current state")
        self.resulting_eligibility = eligibility
        self.active = False
        self.replayed = True


@dataclass
class OnlineLongReadFrame:
    """Single-use owner for one long-read eligibility replay."""

    initial_eligibility: LongReadEligibilityState
    resulting_eligibility: LongReadEligibilityState | None = None
    active: bool = False
    replayed: bool = False

    def begin(self, eligibility: LongReadEligibilityState) -> None:
        if self.replayed:
            raise RuntimeError("online long-read frame was already replayed")
        if self.active:
            raise RuntimeError("online long-read frame is already active")
        if eligibility is not self.initial_eligibility:
            raise ValueError("online long-read frame received a different eligibility state")
        self.active = True

    def complete(self, eligibility: LongReadEligibilityState) -> None:
        if not self.active or self.replayed:
            raise RuntimeError("online long-read frame cannot be completed in its current state")
        self.resulting_eligibility = eligibility
        self.active = False
        self.replayed = True


@dataclass(frozen=True)
class _Dual:
    value: Tensor
    derivative: Tensor

    def __add__(self, other: _Dual | Tensor | float) -> _Dual:
        rhs = _as_dual(other, self)
        return _Dual(self.value + rhs.value, self.derivative + rhs.derivative)

    def __radd__(self, other: Tensor | float) -> _Dual:
        return self + other

    def __sub__(self, other: _Dual | Tensor | float) -> _Dual:
        rhs = _as_dual(other, self)
        return _Dual(self.value - rhs.value, self.derivative - rhs.derivative)

    def __rsub__(self, other: Tensor | float) -> _Dual:
        lhs = _as_dual(other, self)
        return lhs - self

    def __mul__(self, other: _Dual | Tensor | float) -> _Dual:
        rhs = _as_dual(other, self)
        return _Dual(
            self.value * rhs.value,
            self.derivative * rhs.value.unsqueeze(-1)
            + self.value.unsqueeze(-1) * rhs.derivative,
        )

    def __rmul__(self, other: Tensor | float) -> _Dual:
        return self * other


def _as_dual(value: _Dual | Tensor | float, like: _Dual) -> _Dual:
    if isinstance(value, _Dual):
        return value
    tensor = torch.as_tensor(value, dtype=like.value.dtype, device=like.value.device)
    tensor = torch.broadcast_to(tensor, like.value.shape)
    return _Dual(tensor, torch.zeros_like(like.derivative))


def _dual_basis(value: Tensor, index: int, derivatives: int = 23) -> _Dual:
    tangent = torch.zeros(*value.shape, derivatives, dtype=value.dtype, device=value.device)
    tangent[..., index] = 1.0
    return _Dual(value, tangent)


def _dual_constant(value: Tensor, derivatives: int = 23) -> _Dual:
    return _Dual(
        value,
        torch.zeros(*value.shape, derivatives, dtype=value.dtype, device=value.device),
    )


def _dual_tanh(value: _Dual) -> _Dual:
    result = torch.tanh(value.value)
    return _Dual(result, value.derivative * (1.0 - result.square()).unsqueeze(-1))


def _dual_abs(value: _Dual) -> _Dual:
    return _Dual(value.value.abs(), value.derivative * torch.sign(value.value).unsqueeze(-1))


def _dual_exp(value: _Dual) -> _Dual:
    result = torch.exp(value.value)
    return _Dual(result, value.derivative * result.unsqueeze(-1))


def _dual_sigmoid(value: _Dual) -> _Dual:
    result = torch.sigmoid(value.value)
    return _Dual(
        result,
        value.derivative * (result * (1.0 - result)).unsqueeze(-1),
    )


def _dual_softplus(value: _Dual) -> _Dual:
    return _Dual(
        F.softplus(value.value),
        value.derivative * torch.sigmoid(value.value).unsqueeze(-1),
    )


def _dual_hardsigmoid(value: _Dual) -> _Dual:
    result = F.hardsigmoid(value.value)
    active = ((value.value > -3.0) & (value.value < 3.0)).to(value.value.dtype) / 6.0
    return _Dual(result, value.derivative * active.unsqueeze(-1))


def _dual_clamp(value: _Dual, minimum: float, maximum: float) -> _Dual:
    active = ((value.value > minimum) & (value.value < maximum)).to(value.value.dtype)
    return _Dual(
        torch.clamp(value.value, minimum, maximum),
        value.derivative * active.unsqueeze(-1),
    )


def _channel_value(value: Tensor, batch: int) -> Tensor:
    return value.reshape(1, -1).expand(batch, -1)


def _global_value(value: Tensor, batch: int, state: int) -> Tensor:
    return value.reshape(1, 1).expand(batch, state)


def _advance_core_eligibility(
    core: DABSNCore,
    tape: CoreOnlineTape,
    step: int,
    eligibility: CoreEligibilityState,
    *,
    _small_only: bool = False,
):
    """Faithful batched port of the supplied full-hybrid eligibility update."""

    batch = tape.batch_size
    state = tape.state_dim
    x = tape.inputs[:, step].to(dtype=eligibility.dtype)
    budget_value = tape.budget[:, step].to(dtype=eligibility.dtype)
    energy_value = tape.energy[:, step].to(dtype=eligibility.dtype)
    saturation_value = tape.saturation[:, step].to(dtype=eligibility.dtype)
    y_value = tape.y[:, step].to(dtype=eligibility.dtype)
    expression_value = tape.expression[:, step].to(dtype=eligibility.dtype)
    gate_preactivation_value = tape.gate_preactivation[:, step].to(dtype=eligibility.dtype)

    budget = _dual_basis(budget_value, 0)
    energy = _dual_basis(energy_value, 1)
    saturation = _dual_basis(saturation_value, 2)

    local_parameters = (
        core.Wg.bias,
        core.beta,
        core.log_kappa,
        core.logit_recover,
        core.k_s,
        core.k_y,
        core.k_b,
        core.k_n,
        core.k_bias,
        core.r_s,
        core.r_y,
        core.r_b,
        core.r_n,
        core.r_bias,
        core.k_saturation,
        core.r_saturation,
    )
    if local_parameters[0] is None:
        raise RuntimeError("DABSNCore.Wg must retain its bias for online credit")
    local = [
        _dual_basis(_channel_value(parameter.detach(), batch).to(eligibility.dtype), 3 + index)
        for index, parameter in enumerate(local_parameters)
    ]
    global_parameters = (
        core.logit_alpha,
        core.log_lambda,
        core.logit_saturation_decay,
        core.logit_saturation_suppress,
    )
    global_ = [
        _dual_basis(
            _global_value(parameter.detach(), batch, state).to(eligibility.dtype),
            19 + index,
        )
        for index, parameter in enumerate(global_parameters)
    ]

    # The hybrid rule keeps only same-channel state dependence in the local
    # three-state trace. Cross-channel matrix history is represented by the
    # separate budget-only traces below, exactly as in the supplied CPU kernel.
    y = _Dual(
        y_value,
        budget.derivative * (1.0 - y_value.square()).unsqueeze(-1),
    )
    a_diagonal = torch.diagonal(core.A.weight.detach(), 0).reshape(1, state).to(eligibility.dtype)
    ug_diagonal = torch.diagonal(core.Ug.weight.detach(), 0).reshape(1, state).to(
        eligibility.dtype
    )
    expression = _Dual(
        expression_value,
        y.derivative * a_diagonal.unsqueeze(-1),
    )
    gate_preactivation = _Dual(
        gate_preactivation_value,
        y.derivative * ug_diagonal.unsqueeze(-1) + local[0].derivative,
    )
    gate = _dual_hardsigmoid(gate_preactivation)
    novelty = _dual_tanh(_dual_abs(expression - budget))
    saturation_decay = _dual_sigmoid(global_[2])
    next_saturation = saturation_decay * saturation + (
        1.0 - saturation_decay
    ) * novelty * (1.0 - energy)
    saturation_suppress = _dual_sigmoid(global_[3])
    novelty_effective = novelty * (1.0 - saturation_suppress * next_saturation)

    k_signal = (
        local[4] * gate
        + local[5] * y
        + local[6] * _dual_tanh(budget)
        + local[7] * novelty_effective
        + local[8]
        + local[14] * next_saturation
    )
    r_signal = (
        local[9] * gate
        + local[10] * y
        + local[11] * _dual_tanh(budget)
        + local[12] * novelty_effective
        + local[13]
        + local[15] * next_saturation
    )
    write_cost = _dual_softplus(local[2]) * _dual_exp(0.5 * _dual_tanh(k_signal))
    recovery = _dual_clamp(
        _dual_sigmoid(local[3]) * _dual_exp(0.5 * _dual_tanh(r_signal)),
        0.0,
        1.0,
    )
    plasticity = gate * energy
    gate_alpha = _dual_sigmoid(global_[0])
    lam = _dual_softplus(global_[1])
    next_budget = (
        (1.0 - gate_alpha) * budget
        + local[1]
        + lam * plasticity * expression
    )
    next_energy = _dual_clamp(
        energy + recovery * (1.0 - energy) - write_cost * plasticity,
        0.0,
        1.0,
    )

    state_derivatives = torch.stack(
        [next_budget.derivative, next_energy.derivative, next_saturation.derivative],
        dim=2,
    )
    local_jacobian = state_derivatives[..., :3]
    local_direct = state_derivatives[..., 3:19]
    global_direct = state_derivatives[..., 19:23]
    next_local = torch.einsum("bsoi,bsip->bsop", local_jacobian, eligibility.local)
    next_local = next_local + local_direct
    next_global = torch.einsum("bsoi,bsip->bsop", local_jacobian, eligibility.global_)
    next_global = next_global + global_direct

    jbb = next_budget.derivative[..., 0]
    if _small_only:
        return next_local.contiguous(), next_global.contiguous(), jbb
    dy = 1.0 - y_value.square()
    hard_gate_gradient = (
        (gate_preactivation_value > -3.0) & (gate_preactivation_value < 3.0)
    ).to(eligibility.dtype) / 6.0
    lam_value = F.softplus(core.log_lambda.detach()).to(eligibility.dtype)
    energy_numeric = energy_value
    gate_numeric = tape.gate[:, step].to(eligibility.dtype)
    expression_numeric = expression_value
    a_diag = a_diagonal.expand(batch, -1)
    ug_diag = ug_diagonal.expand(batch, -1)
    common = lam_value * energy_numeric
    direct_w = (
        common.unsqueeze(-1)
        * dy.unsqueeze(-1)
        * x.unsqueeze(1)
        * (
            expression_numeric * hard_gate_gradient * ug_diag
            + gate_numeric * a_diag
        ).unsqueeze(-1)
    )
    direct_a = (
        (common * gate_numeric).unsqueeze(-1) * y_value.unsqueeze(1)
    )
    direct_wg = (
        (common * expression_numeric * hard_gate_gradient).unsqueeze(-1)
        * x.unsqueeze(1)
    )
    direct_ug = (
        (common * expression_numeric * hard_gate_gradient).unsqueeze(-1)
        * y_value.unsqueeze(1)
    )

    return CoreEligibilityState(
        w_budget=jbb.unsqueeze(-1) * eligibility.w_budget + direct_w,
        a_budget=jbb.unsqueeze(-1) * eligibility.a_budget + direct_a,
        wg_budget=jbb.unsqueeze(-1) * eligibility.wg_budget + direct_wg,
        ug_budget=jbb.unsqueeze(-1) * eligibility.ug_budget + direct_ug,
        local=next_local.contiguous(),
        global_=next_global.contiguous(),
    )


def core_online_replay_reference(
    core: DABSNCore,
    tape: CoreOnlineTape,
    signal_grads: CoreSignalGrads,
    eligibility_state: CoreEligibilityState,
    *,
    return_incoming_state_grad: bool = False,
    _local_vjp=None,
):
    """Replay full native-core signals through the hybrid eligibility rule.

    The local VJP reconstructs only channelwise arithmetic from saved values.
    The four recurrent/input matrix-vector products are never rerun.
    """

    expected_signal_shape = (tape.batch_size, tape.steps, tape.state_dim)
    for name in CoreSignalGrads.__dataclass_fields__:
        value = getattr(signal_grads, name)
        if value is not None and tuple(value.shape) != expected_signal_shape:
            raise ValueError(f"{name} cotangent must have shape {expected_signal_shape}")
    if (
        eligibility_state.batch_size != tape.batch_size
        or eligibility_state.input_dim != tape.input_dim
        or eligibility_state.state_dim != tape.state_dim
    ):
        raise ValueError("eligibility dimensions do not match tape")

    gradients = {
        parameter: torch.zeros_like(parameter, dtype=torch.promote_types(parameter.dtype, torch.float32))
        for parameter in core.parameters()
    }
    input_gradient = torch.zeros_like(tape.inputs)
    eligibility = eligibility_state
    local_parameters = (
        core.Wg.bias,
        core.beta,
        core.log_kappa,
        core.logit_recover,
        core.k_s,
        core.k_y,
        core.k_b,
        core.k_n,
        core.k_bias,
        core.r_s,
        core.r_y,
        core.r_b,
        core.r_n,
        core.r_bias,
        core.k_saturation,
        core.r_saturation,
    )
    if local_parameters[0] is None:
        raise RuntimeError("DABSNCore.Wg must retain its bias for online credit")
    global_parameters = (
        core.logit_alpha,
        core.log_lambda,
        core.logit_saturation_decay,
        core.logit_saturation_suppress,
    )
    first_incoming_state: tuple[Tensor, Tensor, Tensor] | None = None

    local_vjp = _core_step_local_vjp if _local_vjp is None else _local_vjp
    for step in range(tape.steps):
        incoming_state, direct_gradients, direct_input = local_vjp(
            core,
            tape,
            step,
            signal_grads,
        )
        if first_incoming_state is None:
            first_incoming_state = incoming_state
        incoming = torch.stack(incoming_state, dim=-1).to(dtype=eligibility.dtype)
        incoming_budget = incoming[..., 0]

        gradients[core.W.weight] += torch.einsum(
            "bs,bsi->si", incoming_budget, eligibility.w_budget
        ).to(gradients[core.W.weight].dtype)
        gradients[core.A.weight] += torch.einsum(
            "bs,bsi->si", incoming_budget, eligibility.a_budget
        ).to(gradients[core.A.weight].dtype)
        gradients[core.Wg.weight] += torch.einsum(
            "bs,bsi->si", incoming_budget, eligibility.wg_budget
        ).to(gradients[core.Wg.weight].dtype)
        gradients[core.Ug.weight] += torch.einsum(
            "bs,bsi->si", incoming_budget, eligibility.ug_budget
        ).to(gradients[core.Ug.weight].dtype)

        local_gradient = torch.einsum(
            "bso,bsop->sp", incoming, eligibility.local
        )
        for index, parameter in enumerate(local_parameters):
            gradients[parameter] += local_gradient[:, index].to(gradients[parameter].dtype)
        global_gradient = torch.einsum(
            "bso,bsop->p", incoming, eligibility.global_
        )
        for index, parameter in enumerate(global_parameters):
            gradients[parameter] += global_gradient[index].reshape_as(parameter).to(
                gradients[parameter].dtype
            )

        for parameter, gradient in direct_gradients.items():
            gradients[parameter] += gradient.to(gradients[parameter].dtype)
        input_gradient[:, step] = direct_input.to(input_gradient.dtype)

        eligibility = _advance_core_eligibility(core, tape, step, eligibility)

    gradients = {parameter: gradient.to(parameter.dtype) for parameter, gradient in gradients.items()}
    if return_incoming_state_grad:
        if tape.steps != 1 or first_incoming_state is None:
            raise ValueError("incoming-state gradient is exposed only for a single-step replay")
        return eligibility, gradients, input_gradient, first_incoming_state
    return eligibility, gradients, input_gradient


def _core_step_local_vjp(
    core: DABSNCore,
    tape: CoreOnlineTape,
    step: int,
    signal_grads: CoreSignalGrads,
) -> tuple[
    tuple[Tensor, Tensor, Tensor],
    dict[torch.nn.Parameter, Tensor],
    Tensor,
]:
    """Compute one exact spatial VJP without replaying recurrent matvecs."""

    dtype = tape.y.dtype
    device = tape.y.device

    def leaf(value: Tensor) -> Tensor:
        return value.detach().to(dtype=dtype, device=device).requires_grad_(True)

    with torch.enable_grad():
        y = leaf(tape.y[:, step])
        expression = leaf(tape.expression[:, step])
        gate_preactivation = leaf(tape.gate_preactivation[:, step])
        budget = leaf(tape.budget[:, step])
        energy = leaf(tape.energy[:, step])
        saturation = leaf(tape.saturation[:, step])

        gate = F.hardsigmoid(gate_preactivation)
        novelty = torch.tanh((expression - budget).abs())
        saturation_decay = torch.sigmoid(core.logit_saturation_decay)
        next_saturation = saturation_decay * saturation + (
            1.0 - saturation_decay
        ) * novelty * (1.0 - energy)
        novelty_effective = novelty * (
            1.0 - torch.sigmoid(core.logit_saturation_suppress) * next_saturation
        )
        k_signal = (
            core.k_s * gate
            + core.k_y * y
            + core.k_b * torch.tanh(budget)
            + core.k_n * novelty_effective
            + core.k_bias
            + core.k_saturation * next_saturation
        )
        r_signal = (
            core.r_s * gate
            + core.r_y * y
            + core.r_b * torch.tanh(budget)
            + core.r_n * novelty_effective
            + core.r_bias
            + core.r_saturation * next_saturation
        )
        write_cost = F.softplus(core.log_kappa) * torch.exp(0.5 * torch.tanh(k_signal))
        recovery = torch.clamp(
            torch.sigmoid(core.logit_recover) * torch.exp(0.5 * torch.tanh(r_signal)),
            0.0,
            1.0,
        )
        plasticity = gate * energy
        next_budget = (
            (1.0 - torch.sigmoid(core.logit_alpha)) * budget
            + core.beta
            + F.softplus(core.log_lambda) * plasticity * expression
        )
        # This value is part of recurrent carry rather than a public signal at
        # the current step. Building it here would add no term to the local VJP.
        _ = torch.clamp(
            energy + recovery * (1.0 - energy) - write_cost * plasticity,
            0.0,
            1.0,
        )
        write = plasticity * expression

        outputs = (
            y,
            next_budget,
            novelty,
            plasticity,
            expression,
            write,
            energy,
            next_saturation,
        )

        def cotangent(name: str, like: Tensor) -> Tensor:
            value = getattr(signal_grads, name)
            if value is None:
                return torch.zeros_like(like)
            return value[:, step].to(dtype=like.dtype, device=like.device)

        cotangents = tuple(
            cotangent(name, output)
            for name, output in zip(
                (
                    "y",
                    "next_budget",
                    "novelty",
                    "plasticity",
                    "expression",
                    "write",
                    "energy",
                    "saturation",
                ),
                outputs,
            )
        )
        local_parameters = (
            core.beta,
            core.log_kappa,
            core.logit_recover,
            core.k_s,
            core.k_y,
            core.k_b,
            core.k_n,
            core.k_bias,
            core.r_s,
            core.r_y,
            core.r_b,
            core.r_n,
            core.r_bias,
            core.k_saturation,
            core.r_saturation,
            core.logit_alpha,
            core.log_lambda,
            core.logit_saturation_decay,
            core.logit_saturation_suppress,
        )
        differentiated = (
            y,
            expression,
            gate_preactivation,
            budget,
            energy,
            saturation,
            *local_parameters,
        )
        derivatives = torch.autograd.grad(
            outputs,
            differentiated,
            grad_outputs=cotangents,
            allow_unused=True,
        )

    gradient_y, gradient_expression, gradient_gate, gradient_budget, gradient_energy, gradient_saturation = (
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(
            (y, expression, gate_preactivation, budget, energy, saturation),
            derivatives[:6],
        )
    )
    gradient_y = (
        gradient_y
        + F.linear(gradient_expression, core.A.weight.detach().T)
        + F.linear(gradient_gate, core.Ug.weight.detach().T)
    )
    gradient_y_preactivation = gradient_y * (1.0 - y.detach().square())
    incoming_state = (
        gradient_budget + gradient_y_preactivation,
        gradient_energy,
        gradient_saturation,
    )
    x = tape.inputs[:, step].to(dtype=dtype, device=device)
    direct_gradients: dict[torch.nn.Parameter, Tensor] = {
        core.W.weight: torch.einsum("bs,bi->si", gradient_y_preactivation, x),
        core.A.weight: torch.einsum("bs,bi->si", gradient_expression, y.detach()),
        core.Wg.weight: torch.einsum("bs,bi->si", gradient_gate, x),
        core.Ug.weight: torch.einsum("bs,bi->si", gradient_gate, y.detach()),
    }
    if core.Wg.bias is None:
        raise RuntimeError("DABSNCore.Wg must retain its bias for online credit")
    direct_gradients[core.Wg.bias] = gradient_gate.sum(dim=0)
    for parameter, gradient in zip(local_parameters, derivatives[6:]):
        if gradient is not None:
            direct_gradients[parameter] = gradient
    input_gradient = (
        F.linear(gradient_y_preactivation, core.W.weight.detach().T)
        + F.linear(gradient_gate, core.Wg.weight.detach().T)
    )
    return incoming_state, direct_gradients, input_gradient


def _core_step_local_vjp_analytic(
    core: DABSNCore,
    tape: CoreOnlineTape,
    step: int,
    signal_grads: CoreSignalGrads,
) -> tuple[
    tuple[Tensor, Tensor, Tensor],
    dict[torch.nn.Parameter, Tensor],
    Tensor,
]:
    """Closed-form current-step VJP matching the native CPU replay kernel."""
    dtype = tape.y.dtype
    device = tape.y.device
    shape = (tape.batch_size, tape.state_dim)

    def signal(name: str) -> Tensor:
        value = getattr(signal_grads, name)
        if value is None:
            return torch.zeros(shape, dtype=dtype, device=device)
        return value[:, step].to(dtype=dtype, device=device)

    budget = tape.budget[:, step].to(dtype=dtype)
    energy = tape.energy[:, step].to(dtype=dtype)
    saturation = tape.saturation[:, step].to(dtype=dtype)
    y = tape.y[:, step].to(dtype=dtype)
    expression = tape.expression[:, step].to(dtype=dtype)
    gate_pre = tape.gate_preactivation[:, step].to(dtype=dtype)
    gate = tape.gate[:, step].to(dtype=dtype)
    novelty = tape.novelty[:, step].to(dtype=dtype)
    plasticity = tape.plasticity[:, step].to(dtype=dtype)

    alpha = torch.sigmoid(core.logit_alpha.to(dtype))
    lam = F.softplus(core.log_lambda.to(dtype))
    decay = torch.sigmoid(core.logit_saturation_decay.to(dtype))

    gradient_next_budget = signal("next_budget")
    gradient_budget = gradient_next_budget * (1.0 - alpha)
    gradient_energy = signal("energy")
    gradient_saturation = torch.zeros_like(saturation)
    gradient_y = signal("y")
    gradient_expression = signal("expression")
    gradient_novelty = signal("novelty")
    gradient_plasticity = signal("plasticity")
    gradient_next_saturation = signal("saturation")

    direct: dict[torch.nn.Parameter, Tensor] = {}
    direct[core.beta] = gradient_next_budget.sum(dim=0)
    direct[core.logit_alpha] = (
        gradient_next_budget * (-budget) * alpha * (1.0 - alpha)
    ).sum().reshape_as(core.logit_alpha)
    direct[core.log_lambda] = (
        gradient_next_budget
        * (plasticity * expression)
        * torch.sigmoid(core.log_lambda.to(dtype))
    ).sum().reshape_as(core.log_lambda)

    gradient_write = signal("write") + gradient_next_budget * lam
    gradient_plasticity = gradient_plasticity + gradient_write * expression
    gradient_expression = gradient_expression + gradient_write * plasticity
    gradient_gate = gradient_plasticity * energy
    gradient_energy = gradient_energy + gradient_plasticity * gate

    stress = novelty * (1.0 - energy)
    gradient_saturation = gradient_saturation + gradient_next_saturation * decay
    gradient_stress = gradient_next_saturation * (1.0 - decay)
    direct[core.logit_saturation_decay] = (
        gradient_next_saturation
        * (saturation - stress)
        * decay
        * (1.0 - decay)
    ).sum().reshape_as(core.logit_saturation_decay)
    gradient_novelty = gradient_novelty + gradient_stress * (1.0 - energy)
    gradient_energy = gradient_energy - gradient_stress * novelty

    difference = expression - budget
    gradient_difference = (
        gradient_novelty
        * (1.0 - novelty.square())
        * torch.sign(difference)
    )
    gradient_expression = gradient_expression + gradient_difference
    gradient_budget = gradient_budget - gradient_difference
    # PyTorch's hardsigmoid backward uses the float32-rounded 1/6 constant even
    # for float64 inputs. Match that oracle (and the native C++ `1.0f / 6.0f`).
    hard_slope = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=device).to(dtype)
    hard_gradient = ((gate_pre > -3.0) & (gate_pre < 3.0)).to(dtype) * hard_slope
    gradient_gate_pre = gradient_gate * hard_gradient

    gradient_y = (
        gradient_y
        + F.linear(gradient_expression, core.A.weight.detach().to(dtype).T)
        + F.linear(gradient_gate_pre, core.Ug.weight.detach().to(dtype).T)
    )
    gradient_y_pre = gradient_y * (1.0 - y.square())
    incoming_state = (
        gradient_budget + gradient_y_pre,
        gradient_energy,
        gradient_saturation,
    )

    x = tape.inputs[:, step].to(dtype=dtype, device=device)
    direct[core.W.weight] = torch.einsum("bs,bi->si", gradient_y_pre, x)
    direct[core.A.weight] = torch.einsum("bs,bi->si", gradient_expression, y)
    direct[core.Wg.weight] = torch.einsum("bs,bi->si", gradient_gate_pre, x)
    direct[core.Ug.weight] = torch.einsum("bs,bi->si", gradient_gate_pre, y)
    if core.Wg.bias is None:
        raise RuntimeError("DABSNCore.Wg must retain its bias for online credit")
    direct[core.Wg.bias] = gradient_gate_pre.sum(dim=0)
    input_gradient = (
        F.linear(gradient_y_pre, core.W.weight.detach().to(dtype).T)
        + F.linear(gradient_gate_pre, core.Wg.weight.detach().to(dtype).T)
    )
    return incoming_state, direct, input_gradient


def core_online_replay_analytic(
    core: DABSNCore,
    tape: CoreOnlineTape,
    signal_grads: CoreSignalGrads,
    eligibility_state: CoreEligibilityState,
):
    """Replay online credit with a closed-form local VJP and no autograd calls."""
    return core_online_replay_reference(
        core,
        tape,
        signal_grads,
        eligibility_state,
        _local_vjp=_core_step_local_vjp_analytic,
    )


def _core_trace_scan_reference(core, tape, incoming, eligibility):
    """Small three-state traces only; matrix traces are contracted separately."""
    local = eligibility.local
    global_ = eligibility.global_
    local_gradient = torch.zeros_like(local[:, :, 0, :]).sum(0)
    global_gradient = torch.zeros_like(global_[0, 0, 0])
    transitions = []
    for step in range(tape.steps):
        local_gradient += torch.einsum("bso,bsop->sp", incoming[:, step], local)
        global_gradient += torch.einsum("bso,bsop->p", incoming[:, step], global_)
        current = CoreEligibilityState(*eligibility.tensors()[:4], local, global_)
        local, global_, jbb = _advance_core_eligibility(
            core, tape, step, current, _small_only=True
        )
        transitions.append(jbb)
    return local, global_, local_gradient, global_gradient, torch.stack(transitions, dim=1)


def _budget_trace_weights(jbb, incoming_budget):
    """Division-free adjoint/product scan of E[t+1] = j[t] E[t] + d[t]."""
    if jbb.is_cuda:
        from .kernels.online_trace import budget_trace_weights
        return budget_trace_weights(jbb, incoming_budget)
    response = torch.zeros_like(jbb[:, 0])
    multiplier = torch.ones_like(response)
    future_response = torch.empty_like(jbb)
    suffix = torch.empty_like(jbb)
    for step in range(jbb.shape[1] - 1, -1, -1):
        future_response[:, step] = response
        suffix[:, step] = multiplier
        response = incoming_budget[:, step] + jbb[:, step] * response
        multiplier = jbb[:, step] * multiplier
    return response, multiplier, future_response, suffix


def _contract_budget_trace(initial, coefficient, inputs, weights):
    """Touch a full matrix trace once per chunk, not once per time step.

    The summed rank-one updates are BLAS matrix products. No division by a
    transition/prefix product is used, so zero and negative transitions work.
    """
    boundary_response, final_multiplier, future_response, suffix = weights
    batch, steps, state = coefficient.shape
    # baddbmm avoids materializing a separate B*S*I sum of rank-one updates.
    final = torch.baddbmm(
        initial * final_multiplier.unsqueeze(-1),
        (coefficient * suffix).transpose(1, 2),
        inputs,
    )
    gradient = torch.einsum("bs,bsi->si", boundary_response, initial)
    gradient.addmm_(
        (coefficient * future_response).reshape(batch * steps, state).T,
        inputs.reshape(batch * steps, inputs.shape[-1]),
    )
    return final.contiguous(), gradient


@torch.no_grad()
@torch.autocast(device_type="cpu", enabled=False)
@torch.autocast(device_type="cuda", enabled=False)
def core_online_replay_contracted(core, tape, signal_grads, eligibility_state):
    """Exact hybrid credit with chunk-level GEMMs for large matrix traces.

    The channelwise eligibility rule is unchanged. Only the order of sums and
    products is reassociated; no past input gradient or full BPTT is introduced.
    """
    batch, steps, state = tape.batch_size, tape.steps, tape.state_dim
    if steps < 1:
        raise ValueError("online replay requires at least one time step")
    if (eligibility_state.batch_size, eligibility_state.input_dim, eligibility_state.state_dim) != (
        batch, tape.input_dim, state
    ):
        raise ValueError("eligibility dimensions do not match tape")
    for name in CoreSignalGrads.__dataclass_fields__:
        value = getattr(signal_grads, name)
        if value is not None and tuple(value.shape) != (batch, steps, state):
            raise ValueError(f"{name} cotangent must have shape {(batch, steps, state)}")

    # All current-step VJPs are independent: flatten B*T to replace T small
    # weight/input-gradient GEMMs with one large GEMM per matrix.
    flat_tape = CoreOnlineTape(**{
        name: getattr(tape, name).reshape(batch * steps, 1, -1)
        for name in CoreOnlineTape.__dataclass_fields__
    })
    flat_signals = CoreSignalGrads(**{
        name: None if getattr(signal_grads, name) is None else
        getattr(signal_grads, name).reshape(batch * steps, 1, state)
        for name in CoreSignalGrads.__dataclass_fields__
    })
    incoming, direct, input_gradient = _core_step_local_vjp_analytic(core, flat_tape, 0, flat_signals)
    incoming = torch.stack(incoming, dim=-1).reshape(batch, steps, state, 3).to(eligibility_state.dtype)
    if tape.inputs.is_cuda:
        from .kernels.online_trace import core_trace_scan
        scan = core_trace_scan
    else:
        scan = _core_trace_scan_reference
    local, global_, local_gradient, global_gradient, jbb = scan(core, tape, incoming, eligibility_state)
    weights = _budget_trace_weights(jbb, incoming[..., 0])
    dtype = eligibility_state.dtype
    y = tape.y.to(dtype)
    x = tape.inputs.to(dtype)
    expression = tape.expression.to(dtype)
    gate = tape.gate.to(dtype)
    gate_pre = tape.gate_preactivation.to(dtype)
    slope = ((gate_pre > -3) & (gate_pre < 3)).to(dtype) / 6
    common = F.softplus(core.log_lambda.detach()).to(dtype) * tape.energy.to(dtype)
    coefficient_g = common * expression * slope
    coefficient_a = common * gate
    coefficient_w = common * (1 - y.square()) * (
        expression * slope * core.Ug.weight.detach().diagonal().to(dtype)
        + gate * core.A.weight.detach().diagonal().to(dtype)
    )
    gradients = {parameter: torch.zeros_like(parameter, dtype=torch.promote_types(parameter.dtype, torch.float32))
                 for parameter in core.parameters()}
    matrices = []
    for parameter, initial, coefficient, inputs in (
        (core.W.weight, eligibility_state.w_budget, coefficient_w, x),
        (core.A.weight, eligibility_state.a_budget, coefficient_a, y),
        (core.Wg.weight, eligibility_state.wg_budget, coefficient_g, x),
        (core.Ug.weight, eligibility_state.ug_budget, coefficient_g, y),
    ):
        final, gradient = _contract_budget_trace(initial, coefficient, inputs, weights)
        matrices.append(final)
        gradients[parameter].add_(gradient.to(gradients[parameter].dtype))
    for index, name in enumerate(CORE_LOCAL_PARAMETER_NAMES):
        parameter = core.get_parameter(name)
        gradients[parameter].add_(local_gradient[:, index].to(gradients[parameter].dtype))
    for index, name in enumerate(CORE_GLOBAL_PARAMETER_NAMES):
        parameter = core.get_parameter(name)
        gradients[parameter].add_(global_gradient[index].reshape_as(parameter).to(gradients[parameter].dtype))
    for parameter, gradient in direct.items():
        gradients[parameter].add_(gradient.to(gradients[parameter].dtype))
    return (
        CoreEligibilityState(*matrices, local, global_),
        {parameter: gradient.to(parameter.dtype) for parameter, gradient in gradients.items()},
        input_gradient.reshape_as(tape.inputs).to(tape.inputs.dtype),
    )


def _core_parameter_tuple(core: DABSNCore) -> tuple[torch.nn.Parameter, ...]:
    if core.Wg.bias is None:
        raise RuntimeError("DABSNCore.Wg must retain its bias for online credit")
    return (
        core.W.weight,
        core.A.weight,
        core.Wg.weight,
        core.Ug.weight,
        core.Wg.bias,
        core.beta,
        core.logit_alpha,
        core.log_lambda,
        core.log_kappa,
        core.logit_recover,
        core.k_s,
        core.k_y,
        core.k_b,
        core.k_n,
        core.k_bias,
        core.r_s,
        core.r_y,
        core.r_b,
        core.r_n,
        core.r_bias,
        core.logit_saturation_decay,
        core.k_saturation,
        core.r_saturation,
        core.logit_saturation_suppress,
    )


class _OnlineCoreAutogradBoundary(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: Tensor,
        initial_budget: Tensor,
        initial_energy: Tensor,
        initial_saturation: Tensor,
        *args,
    ):
        parameters = args[:-3]
        core, eligibility, frame = args[-3:]
        if not isinstance(core, DABSNCore):
            raise TypeError("online core boundary requires DABSNCore")
        if not isinstance(eligibility, CoreEligibilityState):
            raise TypeError("online core boundary requires CoreEligibilityState")
        if not isinstance(frame, OnlineCoreFrame):
            raise TypeError("online core boundary requires OnlineCoreFrame")
        expected_parameters = _core_parameter_tuple(core)
        if len(parameters) != len(expected_parameters) or any(
            actual is not expected for actual, expected in zip(parameters, expected_parameters)
        ):
            raise ValueError("online core boundary parameter identity mismatch")
        frame.begin(eligibility)
        native_cpu = (
            inputs.device.type == "cpu"
            and inputs.dtype == torch.float32
            and bool(getattr(type(core), "_cpu_native_enabled", False))
        )
        native_cuda = inputs.is_cuda and bool(
            getattr(type(core), "_cuda_online_enabled", False)
        )
        if native_cpu:
            from .kernels.cpu import native_core_online_forward

            result, tape = native_core_online_forward(
                core,
                inputs,
                (initial_budget, initial_energy, initial_saturation),
                eligibility,
            )
        elif native_cuda:
            from .kernels.triton import cuda_core_online_forward

            result, tape = cuda_core_online_forward(
                core,
                inputs,
                (initial_budget, initial_energy, initial_saturation),
                eligibility,
            )
        else:
            result, tape = core_online_forward_reference(
                core,
                inputs,
                (initial_budget, initial_energy, initial_saturation),
                eligibility,
            )
        ctx.set_materialize_grads(False)
        ctx.core = core
        ctx.eligibility = eligibility
        ctx.frame = frame
        ctx.tape = tape
        ctx.parameter_count = len(parameters)
        ctx.native_cpu = native_cpu
        ctx.native_cuda = native_cuda
        return (
            result.trajectory,
            result.novelty,
            result.plasticity,
            result.expression,
            result.write,
            result.energy,
            result.saturation,
            *result.final_state,
        )

    @staticmethod
    def backward(ctx, *output_gradients):
        trajectory_gradient = output_gradients[0]
        state = ctx.tape.state_dim
        y_gradient = (
            None if trajectory_gradient is None else trajectory_gradient[..., :state]
        )
        budget_gradient = (
            None if trajectory_gradient is None else trajectory_gradient[..., state:]
        )
        final_gradients = output_gradients[7:10]
        if any(
            gradient is not None and bool(torch.count_nonzero(gradient).item())
            for gradient in final_gradients
        ):
            raise RuntimeError(
                "online temporal credit does not accept a loss on returned final recurrent state"
            )
        signals = CoreSignalGrads(
            y=y_gradient,
            next_budget=budget_gradient,
            novelty=output_gradients[1],
            plasticity=output_gradients[2],
            expression=output_gradients[3],
            write=output_gradients[4],
            energy=output_gradients[5],
            saturation=output_gradients[6],
        )
        if ctx.native_cpu:
            from .kernels.cpu import native_core_online_replay

            eligibility, parameter_gradients, input_gradient = native_core_online_replay(
                ctx.core,
                ctx.tape,
                signals,
                ctx.eligibility,
                required=True,
            )
        elif ctx.native_cuda:
            from .kernels.triton import cuda_core_online_replay

            eligibility, parameter_gradients, input_gradient = cuda_core_online_replay(
                ctx.core,
                ctx.tape,
                signals,
                ctx.eligibility,
            )
        else:
            eligibility, parameter_gradients, input_gradient = core_online_replay_reference(
                ctx.core,
                ctx.tape,
                signals,
                ctx.eligibility,
            )
        ctx.frame.complete(eligibility)
        ordered_parameter_gradients = tuple(
            parameter_gradients[parameter] for parameter in _core_parameter_tuple(ctx.core)
        )
        return (
            input_gradient,
            None,
            None,
            None,
            *ordered_parameter_gradients,
            None,
            None,
            None,
        )


def core_online_autograd(
    core: DABSNCore,
    inputs: Tensor,
    initial_state: tuple[Tensor, Tensor, Tensor],
    eligibility_state: CoreEligibilityState,
    frame: OnlineCoreFrame,
) -> tuple[Tensor, ...]:
    """Apply the reference online-credit boundary with ordinary autograd ABI."""

    return _OnlineCoreAutogradBoundary.apply(
        inputs,
        initial_state[0],
        initial_state[1],
        initial_state[2],
        *_core_parameter_tuple(core),
        core,
        eligibility_state,
        frame,
    )


def initial_core_eligibility(
    core: DABSNCore,
    batch_size: int,
    *,
    device,
    dtype: torch.dtype = torch.float32,
) -> CoreEligibilityState:
    """Create zero eligibility without coupling input and recurrent widths."""

    batch = int(batch_size)
    if batch <= 0:
        raise ValueError("batch_size must be positive")
    state = int(core.hidden_dim)
    input_dim = int(core.W.in_features)

    def zeros(*shape: int) -> Tensor:
        return torch.zeros(*shape, device=device, dtype=dtype)

    return CoreEligibilityState(
        w_budget=zeros(batch, state, input_dim),
        a_budget=zeros(batch, state, state),
        wg_budget=zeros(batch, state, input_dim),
        ug_budget=zeros(batch, state, state),
        local=zeros(batch, state, 3, len(CORE_LOCAL_PARAMETER_NAMES)),
        global_=zeros(batch, state, 3, len(CORE_GLOBAL_PARAMETER_NAMES)),
    )


def initial_long_read_eligibility(
    read,
    batch_size: int,
    *,
    device,
    dtype: torch.dtype = torch.float32,
) -> LongReadEligibilityState:
    """Create exact zero eligibility for a DABSN read's long recurrence."""

    batch = int(batch_size)
    if batch <= 0:
        raise ValueError("batch_size must be positive")
    hidden = int(read.hidden_dim)
    return LongReadEligibilityState(
        torch.zeros(batch, hidden, 3, 4, device=device, dtype=dtype)
    )


def _online_blocks(model) -> tuple:
    pending = [model]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        blocks = getattr(candidate, "blocks", None)
        if blocks is not None:
            return tuple(blocks)
        graph = getattr(candidate, "graph", None)
        bindings = () if graph is None else tuple(getattr(graph, "bindings", ()))
        canonical_backbone = getattr(candidate, "backbone", None)
        has_canonical_blocks = getattr(canonical_backbone, "blocks", None) is not None
        if bindings and not has_canonical_blocks:
            unsupported = [
                binding.component_id
                for binding in bindings
                if binding.capabilities.temporal_credit == "unsupported"
            ]
            if unsupported:
                raise TypeError(
                    "online temporal credit is unsupported by graph components "
                    f"{unsupported}"
                )
            owned = tuple(
                binding.module
                for binding in bindings
                if binding.capabilities.temporal_credit == "dabsn"
            )
            if owned:
                return owned
        pending.extend(
            getattr(candidate, name, None) for name in ("module", "body", "backbone")
        )
    raise TypeError("online temporal state requires a model with a canonical DABSN backbone")


def initial_online_temporal_state(
    model,
    batch_size: int,
    *,
    stream_policy: str,
    stream_ids: tuple[str | int, ...],
    device,
    dtype,
    eligibility_dtype: torch.dtype | None = None,
    temporal_chunk_size: int | None = None,
    eligibility_batch_mode: str = "vectorized",
    eligibility_lane_group_size: int | None = None,
) -> OnlineTemporalState:
    """Create independent recurrent and eligibility state for every DABSN block.

    ``dtype`` controls recurrent storage. Eligibility defaults to FP32 for
    FP16/BF16/FP32 recurrent state and FP64 for FP64 recurrent state. An explicit
    ``eligibility_dtype`` overrides that precision policy; backend support is
    checked when replay executes. Resets and checkpoint restore preserve the
    stored eligibility dtype rather than deriving it again from recurrent state.
    """

    batch = int(batch_size)
    if eligibility_dtype is None:
        eligibility_dtype = torch.promote_types(dtype, torch.float32)
    layers: list[OnlineLayerState] = []
    for block in _online_blocks(model):
        core_recurrent = tuple(
            value.detach()
            for value in block.core.initial_state(batch, device=device, dtype=dtype)
        )
        long_recurrent = tuple(
            value.detach()
            for value in block.read.initial_long_scan_state(
                batch,
                device=device,
                dtype=dtype,
            )
        )
        layers.append(
            OnlineLayerState(
                core_recurrent=core_recurrent,
                core_eligibility=initial_core_eligibility(
                    block.core,
                    batch,
                    device=device,
                    dtype=eligibility_dtype,
                ),
                long_recurrent=long_recurrent,
                long_eligibility=initial_long_read_eligibility(
                    block.read,
                    batch,
                    device=device,
                    dtype=eligibility_dtype,
                ),
            )
        )
    return OnlineTemporalState(
        layers=tuple(layers),
        stream_policy=stream_policy,
        stream_ids=tuple(stream_ids),
        pending_reset_mask=torch.zeros(batch, dtype=torch.bool, device=device),
        temporal_chunk_size=temporal_chunk_size,
        eligibility_batch_mode=eligibility_batch_mode,
        eligibility_lane_group_size=eligibility_lane_group_size,
    )


def _checkpoint_tensor(value: Tensor) -> Tensor:
    return value.detach().cpu().contiguous().clone()


def online_temporal_state_dict(state: OnlineTemporalState) -> dict[str, object]:
    """Return the complete tensor/metadata payload required for exact resume."""

    if not isinstance(state, OnlineTemporalState):
        raise TypeError("state must be OnlineTemporalState")
    state.assert_checkpointable()
    if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in state.stream_ids):
        raise ValueError("stream_ids must contain only string or integer identifiers")
    layers: list[dict[str, object]] = []
    for layer in state.layers:
        layers.append(
            {
                "core_recurrent": [_checkpoint_tensor(value) for value in layer.core_recurrent],
                "core_eligibility": {
                    "schema_version": layer.core_eligibility.schema_version,
                    "w_budget": _checkpoint_tensor(layer.core_eligibility.w_budget),
                    "a_budget": _checkpoint_tensor(layer.core_eligibility.a_budget),
                    "wg_budget": _checkpoint_tensor(layer.core_eligibility.wg_budget),
                    "ug_budget": _checkpoint_tensor(layer.core_eligibility.ug_budget),
                    "local": _checkpoint_tensor(layer.core_eligibility.local),
                    "global": _checkpoint_tensor(layer.core_eligibility.global_),
                },
                "long_recurrent": [_checkpoint_tensor(value) for value in layer.long_recurrent],
                "long_eligibility": {
                    "schema_version": layer.long_eligibility.schema_version,
                    "traces": _checkpoint_tensor(layer.long_eligibility.traces),
                },
                "read_bank": {
                    name: _checkpoint_tensor(value) for name, value in layer.read_bank.items()
                },
            }
        )
    return {
        "format": ONLINE_TEMPORAL_STATE_FORMAT,
        "schema_version": state.schema_version,
        "temporal_credit": "online",
        "stream_policy": state.stream_policy,
        "stream_ids": list(state.stream_ids),
        "pending_reset_mask": _checkpoint_tensor(state.pending_reset_mask),
        "chunk_position": state.chunk_position,
        "accumulation_position": state.accumulation_position,
        "temporal_chunk_size": state.temporal_chunk_size,
        "eligibility_batch_mode": state.eligibility_batch_mode,
        "eligibility_lane_group_size": state.eligibility_lane_group_size,
        "layers": layers,
    }


def _checkpoint_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _checkpoint_sequence(value: object, label: str, length: int | None = None) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a sequence")
    result = tuple(value)
    if length is not None and len(result) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return result


def _restore_tensor(value: object, label: str, device: torch.device) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{label} must be a tensor")
    return value.detach().to(device=device).contiguous().clone()


def load_online_temporal_state(
    model,
    payload: Mapping[str, object],
    *,
    device: str | torch.device | None = None,
    expected_stream_policy: str | None = None,
    expected_stream_ids: tuple[str | int, ...] | None = None,
    expected_temporal_chunk_size: int | None = None,
    expected_eligibility_batch_mode: str | None = None,
    expected_eligibility_lane_group_size: int | None = None,
    expected_accumulation_position: int | None = None,
) -> OnlineTemporalState:
    """Validate and reconstruct exact online state without mutating ``model``."""

    data = _checkpoint_mapping(payload, "online temporal state")
    if data.get("format") != ONLINE_TEMPORAL_STATE_FORMAT:
        raise ValueError("not a DABSN online temporal state payload")
    schema_version = int(data.get("schema_version", -1))
    if schema_version != ONLINE_TEMPORAL_STATE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported online temporal state schema_version "
            f"{schema_version}; expected {ONLINE_TEMPORAL_STATE_SCHEMA_VERSION}"
        )
    if data.get("temporal_credit") != "online":
        raise ValueError("online temporal state temporal_credit must be online")
    stream_policy = data.get("stream_policy")
    if stream_policy not in {"reset_each_batch", "carry_lanes"}:
        raise ValueError("online temporal state has an invalid stream policy")
    if expected_stream_policy is not None and stream_policy != expected_stream_policy:
        raise ValueError(
            f"online temporal state stream policy is {stream_policy}, expected "
            f"{expected_stream_policy}"
        )
    stream_ids = _checkpoint_sequence(data.get("stream_ids"), "stream_ids")
    if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in stream_ids):
        raise ValueError("stream_ids must contain only string or integer identifiers")
    if expected_stream_ids is not None and stream_ids != tuple(expected_stream_ids):
        raise ValueError("online temporal state stream lane identifiers do not match")
    raw_chunk_size = data.get("temporal_chunk_size")
    temporal_chunk_size = None if raw_chunk_size is None else int(raw_chunk_size)
    if (
        expected_temporal_chunk_size is not None
        and temporal_chunk_size != int(expected_temporal_chunk_size)
    ):
        raise ValueError("online temporal state chunk contract does not match")
    eligibility_batch_mode = str(data.get("eligibility_batch_mode", "vectorized"))
    raw_group_size = data.get("eligibility_lane_group_size")
    eligibility_lane_group_size = None if raw_group_size is None else int(raw_group_size)
    if (
        expected_eligibility_batch_mode is not None
        and eligibility_batch_mode != expected_eligibility_batch_mode
    ):
        raise ValueError("online temporal state eligibility batch contract does not match")
    if (
        expected_eligibility_lane_group_size is not None
        and eligibility_lane_group_size != int(expected_eligibility_lane_group_size)
    ):
        raise ValueError("online temporal state lane-group contract does not match")
    accumulation_position = int(data.get("accumulation_position", -1))
    if (
        expected_accumulation_position is not None
        and accumulation_position != int(expected_accumulation_position)
    ):
        raise ValueError("online temporal state accumulation position does not match")

    blocks = _online_blocks(model)
    raw_layers = _checkpoint_sequence(data.get("layers"), "layers")
    if len(raw_layers) != len(blocks):
        raise ValueError(
            f"online temporal state has {len(raw_layers)} layers, model has {len(blocks)}"
        )
    if device is None:
        try:
            resolved_device = next(model.parameters()).device
        except StopIteration:
            resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)

    layers: list[OnlineLayerState] = []
    for index, (block, raw_layer) in enumerate(zip(blocks, raw_layers)):
        layer_data = _checkpoint_mapping(raw_layer, f"layer {index}")
        core_recurrent_raw = _checkpoint_sequence(
            layer_data.get("core_recurrent"), f"layer {index} core_recurrent", 3
        )
        long_recurrent_raw = _checkpoint_sequence(
            layer_data.get("long_recurrent"), f"layer {index} long_recurrent", 3
        )
        core_data = _checkpoint_mapping(
            layer_data.get("core_eligibility"), f"layer {index} core_eligibility"
        )
        long_data = _checkpoint_mapping(
            layer_data.get("long_eligibility"), f"layer {index} long_eligibility"
        )
        bank_data = _checkpoint_mapping(layer_data.get("read_bank"), f"layer {index} read_bank")
        try:
            core_eligibility = CoreEligibilityState(
                w_budget=_restore_tensor(
                    core_data.get("w_budget"), f"layer {index} w_budget", resolved_device
                ),
                a_budget=_restore_tensor(
                    core_data.get("a_budget"), f"layer {index} a_budget", resolved_device
                ),
                wg_budget=_restore_tensor(
                    core_data.get("wg_budget"), f"layer {index} wg_budget", resolved_device
                ),
                ug_budget=_restore_tensor(
                    core_data.get("ug_budget"), f"layer {index} ug_budget", resolved_device
                ),
                local=_restore_tensor(
                    core_data.get("local"), f"layer {index} local", resolved_device
                ),
                global_=_restore_tensor(
                    core_data.get("global"), f"layer {index} global", resolved_device
                ),
                schema_version=int(core_data.get("schema_version", -1)),
            )
            long_eligibility = LongReadEligibilityState(
                _restore_tensor(
                    long_data.get("traces"), f"layer {index} long traces", resolved_device
                ),
                schema_version=int(long_data.get("schema_version", -1)),
            )
            layer = OnlineLayerState(
                core_recurrent=tuple(
                    _restore_tensor(value, f"layer {index} core recurrent", resolved_device)
                    for value in core_recurrent_raw
                ),
                core_eligibility=core_eligibility,
                long_recurrent=tuple(
                    _restore_tensor(value, f"layer {index} long recurrent", resolved_device)
                    for value in long_recurrent_raw
                ),
                long_eligibility=long_eligibility,
                read_bank={
                    str(name): _restore_tensor(
                        value, f"layer {index} read_bank {name}", resolved_device
                    )
                    for name, value in bank_data.items()
                },
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"online temporal state model layer {index} is invalid: {exc}") from exc
        expected_input = int(block.core.W.in_features)
        expected_state = int(block.core.hidden_dim)
        expected_hidden = int(block.read.hidden_dim)
        if (
            core_eligibility.input_dim != expected_input
            or core_eligibility.state_dim != expected_state
            or long_eligibility.hidden_dim != expected_hidden
        ):
            raise ValueError(
                f"online temporal state does not match model layer {index}: expected "
                f"core {expected_input}->{expected_state} and read width {expected_hidden}"
            )
        layers.append(layer)

    pending_reset_mask = _restore_tensor(
        data.get("pending_reset_mask"), "pending_reset_mask", resolved_device
    )
    try:
        return OnlineTemporalState(
            layers=tuple(layers),
            stream_policy=str(stream_policy),
            stream_ids=tuple(stream_ids),
            pending_reset_mask=pending_reset_mask,
            chunk_position=int(data.get("chunk_position", -1)),
            accumulation_position=accumulation_position,
            temporal_chunk_size=temporal_chunk_size,
            eligibility_batch_mode=eligibility_batch_mode,
            eligibility_lane_group_size=eligibility_lane_group_size,
            schema_version=schema_version,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid online temporal state payload: {exc}") from exc


def finalize_online_temporal_state(state: OnlineTemporalState) -> OnlineTemporalState:
    """Replace pending frames with their replayed eligibility state."""

    if len(state.active_frames) != 2 * len(state.layers):
        raise RuntimeError("online temporal state does not contain one core/read frame pair per layer")
    layers: list[OnlineLayerState] = []
    for index, layer in enumerate(state.layers):
        core_frame = state.active_frames[2 * index]
        long_frame = state.active_frames[2 * index + 1]
        if not isinstance(core_frame, OnlineCoreFrame) or not isinstance(
            long_frame, OnlineLongReadFrame
        ):
            raise RuntimeError("online temporal frame ordering is invalid")
        if (
            not core_frame.replayed
            or core_frame.resulting_eligibility is None
            or not long_frame.replayed
            or long_frame.resulting_eligibility is None
        ):
            raise RuntimeError("cannot finalize before every online replay completes")
        layers.append(
            OnlineLayerState(
                core_recurrent=layer.core_recurrent,
                core_eligibility=core_frame.resulting_eligibility,
                long_recurrent=layer.long_recurrent,
                long_eligibility=long_frame.resulting_eligibility,
                read_bank=layer.read_bank,
            )
        )
    return OnlineTemporalState(
        layers=tuple(layers),
        stream_policy=state.stream_policy,
        stream_ids=state.stream_ids,
        pending_reset_mask=state.pending_reset_mask,
        chunk_position=state.chunk_position,
        accumulation_position=state.accumulation_position,
        temporal_chunk_size=state.temporal_chunk_size,
        eligibility_batch_mode=state.eligibility_batch_mode,
        eligibility_lane_group_size=state.eligibility_lane_group_size,
        schema_version=state.schema_version,
    )


def _reset_lanes(value: Tensor, fresh: Tensor, mask: Tensor) -> Tensor:
    expanded = mask.reshape(mask.shape[0], *((1,) * (value.dim() - 1)))
    return torch.where(expanded, fresh, value).detach().contiguous()


def _slice_online_tensor(value: Tensor, start: int, stop: int) -> Tensor:
    return value[start:stop].detach().contiguous()


def _slice_online_temporal_state(
    state: OnlineTemporalState, start: int, stop: int
) -> OnlineTemporalState:
    if state.active_frames:
        raise RuntimeError("cannot group online state with pending replay")
    if not 0 <= start < stop <= state.batch_size:
        raise ValueError("online lane slice is out of range")
    layers = []
    for layer in state.layers:
        layers.append(
            OnlineLayerState(
                core_recurrent=tuple(
                    _slice_online_tensor(value, start, stop) for value in layer.core_recurrent
                ),
                core_eligibility=CoreEligibilityState(
                    *(
                        _slice_online_tensor(value, start, stop)
                        for value in layer.core_eligibility.tensors()
                    )
                ),
                long_recurrent=tuple(
                    _slice_online_tensor(value, start, stop) for value in layer.long_recurrent
                ),
                long_eligibility=LongReadEligibilityState(
                    _slice_online_tensor(layer.long_eligibility.traces, start, stop)
                ),
                read_bank={
                    name: _slice_online_tensor(value, start, stop)
                    for name, value in layer.read_bank.items()
                },
            )
        )
    return OnlineTemporalState(
        layers=tuple(layers),
        stream_policy=state.stream_policy,
        stream_ids=state.stream_ids[start:stop],
        pending_reset_mask=_slice_online_tensor(state.pending_reset_mask, start, stop),
        chunk_position=state.chunk_position,
        accumulation_position=state.accumulation_position,
        temporal_chunk_size=state.temporal_chunk_size,
        eligibility_batch_mode=state.eligibility_batch_mode,
        eligibility_lane_group_size=state.eligibility_lane_group_size,
        schema_version=state.schema_version,
    )


def _merge_online_temporal_states(states: tuple[OnlineTemporalState, ...]) -> OnlineTemporalState:
    if not states:
        raise ValueError("cannot merge an empty online lane group")
    first = states[0]
    if any(state.active_frames for state in states):
        raise RuntimeError("cannot merge online state with pending replay")
    identity = (
        first.stream_policy,
        first.chunk_position,
        first.accumulation_position,
        first.temporal_chunk_size,
        first.eligibility_batch_mode,
        first.eligibility_lane_group_size,
        first.schema_version,
        len(first.layers),
    )
    if any(
        (
            state.stream_policy,
            state.chunk_position,
            state.accumulation_position,
            state.temporal_chunk_size,
            state.eligibility_batch_mode,
            state.eligibility_lane_group_size,
            state.schema_version,
            len(state.layers),
        )
        != identity
        for state in states[1:]
    ):
        raise ValueError("online lane groups have incompatible state metadata")

    layers = []
    for layer_index in range(len(first.layers)):
        grouped = tuple(state.layers[layer_index] for state in states)

        def cat(values) -> Tensor:
            return torch.cat(tuple(values), dim=0).detach().contiguous()

        bank_names = set(grouped[0].read_bank)
        if any(set(layer.read_bank) != bank_names for layer in grouped[1:]):
            raise ValueError("online lane groups have incompatible read-bank schemas")
        layers.append(
            OnlineLayerState(
                core_recurrent=tuple(
                    cat(layer.core_recurrent[index] for layer in grouped) for index in range(3)
                ),
                core_eligibility=CoreEligibilityState(
                    *(cat(layer.core_eligibility.tensors()[index] for layer in grouped) for index in range(6))
                ),
                long_recurrent=tuple(
                    cat(layer.long_recurrent[index] for layer in grouped) for index in range(3)
                ),
                long_eligibility=LongReadEligibilityState(
                    cat(layer.long_eligibility.traces for layer in grouped)
                ),
                read_bank={name: cat(layer.read_bank[name] for layer in grouped) for name in bank_names},
            )
        )
    return OnlineTemporalState(
        layers=tuple(layers),
        stream_policy=first.stream_policy,
        stream_ids=tuple(value for state in states for value in state.stream_ids),
        pending_reset_mask=torch.cat(
            tuple(state.pending_reset_mask for state in states), dim=0
        ).contiguous(),
        chunk_position=first.chunk_position,
        accumulation_position=first.accumulation_position,
        temporal_chunk_size=first.temporal_chunk_size,
        eligibility_batch_mode=first.eligibility_batch_mode,
        eligibility_lane_group_size=first.eligibility_lane_group_size,
        schema_version=first.schema_version,
    )


def prepare_online_temporal_state(
    model,
    state: OnlineTemporalState,
    *,
    stream_ids: tuple[str | int, ...],
    stream_reset_mask: Tensor | None,
) -> OnlineTemporalState:
    """Apply batch/lane reset policy before the first chunk of an update."""

    if state.active_frames:
        raise RuntimeError("cannot prepare online state with pending replay")
    ids = tuple(stream_ids)
    if len(ids) != state.batch_size:
        raise ValueError("stream_ids do not match the online batch size")
    device = state.pending_reset_mask.device
    reset_all = state.stream_policy == "reset_each_batch"
    if reset_all:
        mask = torch.ones(state.batch_size, dtype=torch.bool, device=device)
    elif stream_reset_mask is None:
        mask = torch.zeros(state.batch_size, dtype=torch.bool, device=device)
    else:
        mask = stream_reset_mask.to(device=device)
        if mask.dtype != torch.bool or tuple(mask.shape) != (state.batch_size,):
            raise ValueError("stream_reset_mask must be bool [B]")
        mask = mask.contiguous()
    changed = torch.tensor(
        [actual != expected for actual, expected in zip(ids, state.stream_ids)],
        dtype=torch.bool,
        device=device,
    )
    if not reset_all and bool(torch.any(changed & ~mask).item()):
        raise ValueError("stream lane identity changed without a matching reset")

    blocks = _online_blocks(model)
    layers: list[OnlineLayerState] = []
    for block, layer in zip(blocks, state.layers):
        fresh_core = tuple(
            value.detach()
            for value in block.core.initial_state(
                state.batch_size,
                device=device,
                dtype=layer.core_recurrent[0].dtype,
            )
        )
        fresh_long = tuple(
            value.detach()
            for value in block.read.initial_long_scan_state(
                state.batch_size,
                device=device,
                dtype=layer.long_recurrent[0].dtype,
            )
        )
        fresh_core_eligibility = initial_core_eligibility(
            block.core,
            state.batch_size,
            device=device,
            dtype=layer.core_eligibility.dtype,
        )
        fresh_long_eligibility = initial_long_read_eligibility(
            block.read,
            state.batch_size,
            device=device,
            dtype=layer.long_eligibility.traces.dtype,
        )
        if reset_all:
            layers.append(
                OnlineLayerState(
                    core_recurrent=fresh_core,
                    core_eligibility=fresh_core_eligibility,
                    long_recurrent=fresh_long,
                    long_eligibility=fresh_long_eligibility,
                )
            )
            continue
        bank = dict(layer.read_bank)
        if bank:
            valid = bank["valid"].clone()
            valid[mask] = False
            bank["valid"] = valid.detach().contiguous()
        layers.append(
            OnlineLayerState(
                core_recurrent=tuple(
                    _reset_lanes(old, fresh, mask)
                    for old, fresh in zip(layer.core_recurrent, fresh_core)
                ),
                core_eligibility=CoreEligibilityState(
                    *(
                        _reset_lanes(old, fresh, mask)
                        for old, fresh in zip(
                            layer.core_eligibility.tensors(),
                            fresh_core_eligibility.tensors(),
                        )
                    )
                ),
                long_recurrent=tuple(
                    _reset_lanes(old, fresh, mask)
                    for old, fresh in zip(layer.long_recurrent, fresh_long)
                ),
                long_eligibility=LongReadEligibilityState(
                    _reset_lanes(
                        layer.long_eligibility.traces,
                        fresh_long_eligibility.traces,
                        mask,
                    )
                ),
                read_bank=bank,
            )
        )
    return OnlineTemporalState(
        layers=tuple(layers),
        stream_policy=state.stream_policy,
        stream_ids=ids,
        pending_reset_mask=torch.zeros_like(state.pending_reset_mask),
        chunk_position=0 if reset_all else state.chunk_position,
        accumulation_position=0 if reset_all else state.accumulation_position,
        temporal_chunk_size=state.temporal_chunk_size,
        eligibility_batch_mode=state.eligibility_batch_mode,
        eligibility_lane_group_size=state.eligibility_lane_group_size,
        schema_version=state.schema_version,
    )


def bind_online_temporal_contract(
    state: OnlineTemporalState,
    *,
    temporal_chunk_size: int,
    eligibility_batch_mode: str,
    eligibility_lane_group_size: int | None,
) -> OnlineTemporalState:
    """Bind an initial state once, then reject silent execution-contract changes."""

    requested_chunk = int(temporal_chunk_size)
    if state.temporal_chunk_size is not None and state.temporal_chunk_size != requested_chunk:
        raise ValueError("online temporal state chunk contract changed")
    if state.temporal_chunk_size is not None and (
        state.eligibility_batch_mode != eligibility_batch_mode
        or state.eligibility_lane_group_size != eligibility_lane_group_size
    ):
        raise ValueError("online temporal state lane-group contract changed")
    return OnlineTemporalState(
        layers=state.layers,
        stream_policy=state.stream_policy,
        stream_ids=state.stream_ids,
        pending_reset_mask=state.pending_reset_mask,
        chunk_position=state.chunk_position,
        accumulation_position=state.accumulation_position,
        temporal_chunk_size=requested_chunk,
        eligibility_batch_mode=eligibility_batch_mode,
        eligibility_lane_group_size=eligibility_lane_group_size,
        schema_version=state.schema_version,
    )


def online_training_state_dict(model, state: OnlineTemporalState) -> dict[str, object]:
    """Capture online temporal state plus accumulated gradients for exact resume."""

    gradients = {
        name: _checkpoint_tensor(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return {
        "format": ONLINE_TRAINING_STATE_FORMAT,
        "temporal_state": online_temporal_state_dict(state),
        "gradients": gradients,
    }


def load_online_training_state(
    model,
    payload: Mapping[str, object],
    **temporal_expectations,
) -> OnlineTemporalState:
    """Validate a full online training payload, then install accumulated gradients."""

    data = _checkpoint_mapping(payload, "online training state")
    if data.get("format") != ONLINE_TRAINING_STATE_FORMAT:
        raise ValueError("not a DABSN online training-state payload")
    temporal = load_online_temporal_state(
        model,
        _checkpoint_mapping(data.get("temporal_state"), "temporal_state"),
        **temporal_expectations,
    )
    raw_gradients = _checkpoint_mapping(data.get("gradients"), "gradients")
    parameters = dict(model.named_parameters())
    unexpected = set(raw_gradients) - set(parameters)
    if unexpected:
        raise ValueError(f"online training state has unknown gradients: {sorted(unexpected)}")
    validated: dict[str, Tensor] = {}
    for name, value in raw_gradients.items():
        parameter = parameters[name]
        gradient = _restore_tensor(value, f"gradient {name}", parameter.device)
        if gradient.shape != parameter.shape or gradient.dtype != parameter.dtype:
            raise ValueError(f"online training-state gradient {name} does not match its parameter")
        validated[name] = gradient
    for parameter in parameters.values():
        parameter.grad = None
    for name, gradient in validated.items():
        parameters[name].grad = gradient
    return temporal


def core_online_forward_reference(
    core: DABSNCore,
    inputs: Tensor,
    recurrent_state: tuple[Tensor, Tensor, Tensor],
    eligibility_state: CoreEligibilityState,
) -> tuple[CoreForwardResult, CoreOnlineTape]:
    """Run canonical values and save a compact tape without advancing eligibility."""

    if inputs.dim() != 3:
        raise ValueError("inputs must have shape [B,T,input_dim]")
    batch, steps, input_dim = (int(value) for value in inputs.shape)
    state_dim = int(core.hidden_dim)
    if input_dim != int(core.W.in_features):
        raise ValueError(
            f"inputs have width {input_dim}, expected core input width {core.W.in_features}"
        )
    if eligibility_state.batch_size != batch:
        raise ValueError("eligibility batch size does not match inputs")
    if eligibility_state.input_dim != input_dim or eligibility_state.state_dim != state_dim:
        raise ValueError("eligibility dimensions do not match core")
    if eligibility_state.device != inputs.device:
        raise ValueError("eligibility and inputs must be on the same device")
    if len(recurrent_state) != 3 or any(
        tuple(value.shape) != (batch, state_dim) for value in recurrent_state
    ):
        raise ValueError(
            f"recurrent_state tensors must have shape {(batch, state_dim)}"
        )

    with torch.no_grad():
        budget, energy, saturation = (
            value.to(device=inputs.device, dtype=inputs.dtype) for value in recurrent_state
        )
        projected = core.W(inputs)
        gate_projected = core.Wg(inputs)
        fused_recurrence = torch.cat([core.Ug.weight, core.A.weight], dim=0)
        gate_alpha = torch.sigmoid(core.logit_alpha)
        lam = F.softplus(core.log_lambda)
        kappa = F.softplus(core.log_kappa)
        recover = torch.sigmoid(core.logit_recover)
        saturation_decay = torch.sigmoid(core.logit_saturation_decay)
        saturation_suppress = torch.sigmoid(core.logit_saturation_suppress)

        trajectories: list[Tensor] = []
        novelty_trajectory: list[Tensor] = []
        plasticity_trajectory: list[Tensor] = []
        expression_trajectory: list[Tensor] = []
        write_trajectory: list[Tensor] = []
        energy_trajectory: list[Tensor] = []
        saturation_trajectory: list[Tensor] = []
        tape_budget: list[Tensor] = []
        tape_energy: list[Tensor] = []
        tape_saturation: list[Tensor] = []
        tape_y: list[Tensor] = []
        tape_expression: list[Tensor] = []
        tape_gate_preactivation: list[Tensor] = []
        tape_gate: list[Tensor] = []
        tape_novelty_effective: list[Tensor] = []
        tape_write_cost: list[Tensor] = []
        tape_recovery: list[Tensor] = []

        for step in range(steps):
            tape_budget.append(budget)
            tape_energy.append(energy)
            tape_saturation.append(saturation)

            y = torch.tanh(projected[:, step, :] + budget)
            gate_recurrence, expression = F.linear(y, fused_recurrence).chunk(2, dim=1)
            gate_preactivation = gate_projected[:, step, :] + gate_recurrence
            gate = F.hardsigmoid(gate_preactivation)
            novelty = torch.tanh((expression - budget).abs())
            stress = novelty * (1.0 - energy)
            next_saturation = saturation_decay * saturation + (
                1.0 - saturation_decay
            ) * stress
            novelty_effective = novelty * (1.0 - saturation_suppress * next_saturation)
            k_signal = (
                core.k_s * gate
                + core.k_y * y
                + core.k_b * torch.tanh(budget)
                + core.k_n * novelty_effective
                + core.k_bias
                + core.k_saturation * next_saturation
            )
            r_signal = (
                core.r_s * gate
                + core.r_y * y
                + core.r_b * torch.tanh(budget)
                + core.r_n * novelty_effective
                + core.r_bias
                + core.r_saturation * next_saturation
            )
            write_cost = kappa * torch.exp(0.5 * torch.tanh(k_signal))
            recovery = torch.clamp(
                recover * torch.exp(0.5 * torch.tanh(r_signal)),
                0.0,
                1.0,
            )
            plasticity = gate * energy
            next_budget = (
                (1.0 - gate_alpha) * budget
                + core.beta
                + lam * (plasticity * expression)
            )
            next_energy = torch.clamp(
                energy + recovery * (1.0 - energy) - write_cost * plasticity,
                0.0,
                1.0,
            )

            trajectories.append(torch.cat([y, next_budget], dim=-1))
            novelty_trajectory.append(novelty)
            plasticity_trajectory.append(plasticity)
            expression_trajectory.append(expression)
            write_trajectory.append(plasticity * expression)
            energy_trajectory.append(energy)
            saturation_trajectory.append(next_saturation)
            tape_y.append(y)
            tape_expression.append(expression)
            tape_gate_preactivation.append(gate_preactivation)
            tape_gate.append(gate)
            tape_novelty_effective.append(novelty_effective)
            tape_write_cost.append(write_cost)
            tape_recovery.append(recovery)

            budget, energy, saturation = next_budget, next_energy, next_saturation

        result = CoreForwardResult(
            trajectory=torch.stack(trajectories, dim=1),
            novelty=torch.stack(novelty_trajectory, dim=1),
            plasticity=torch.stack(plasticity_trajectory, dim=1),
            expression=torch.stack(expression_trajectory, dim=1),
            write=torch.stack(write_trajectory, dim=1),
            energy=torch.stack(energy_trajectory, dim=1),
            saturation=torch.stack(saturation_trajectory, dim=1),
            final_state=(budget, energy, saturation),
        )
        tape = CoreOnlineTape(
            inputs=inputs.detach(),
            budget=torch.stack(tape_budget, dim=1),
            energy=torch.stack(tape_energy, dim=1),
            saturation=torch.stack(tape_saturation, dim=1),
            y=torch.stack(tape_y, dim=1),
            expression=torch.stack(tape_expression, dim=1),
            gate_preactivation=torch.stack(tape_gate_preactivation, dim=1),
            gate=torch.stack(tape_gate, dim=1),
            novelty=result.novelty.detach(),
            next_saturation=result.saturation.detach(),
            novelty_effective=torch.stack(tape_novelty_effective, dim=1),
            write_cost=torch.stack(tape_write_cost, dim=1),
            recovery=torch.stack(tape_recovery, dim=1),
            plasticity=result.plasticity.detach(),
        )
    return result, tape


def _long_parameter_tuple(read) -> tuple[torch.nn.Parameter, ...]:
    return (
        read.logit_retain,
        read.logit_expect_retain,
        read.logit_retention_decay,
        read.logit_retention_strength,
    )


def long_read_online_forward(
    read,
    write: Tensor,
    plasticity: Tensor,
    novelty: Tensor,
    initial_state: tuple[Tensor, Tensor, Tensor],
    eligibility_state: LongReadEligibilityState,
) -> tuple[Tensor, LongReadOnlineTape, tuple[Tensor, Tensor, Tensor]]:
    """Run the exact long-read recurrence while saving detached pre-state values."""

    if write.dim() != 3 or plasticity.shape != write.shape or novelty.shape != write.shape:
        raise ValueError("long-read online inputs must share shape [B,T,H]")
    batch, steps, hidden = write.shape
    if (
        eligibility_state.batch_size != batch
        or eligibility_state.hidden_dim != hidden
    ):
        raise ValueError("long-read eligibility dimensions do not match inputs")
    if len(initial_state) != 3 or any(value.shape != (batch, hidden) for value in initial_state):
        raise ValueError("initial long-read state must contain three tensors shaped [B,H]")

    with torch.no_grad():
        long, expected, retention = (
            value.to(device=write.device, dtype=write.dtype) for value in initial_state
        )
        retain = torch.sigmoid(read.logit_retain)
        expect_retain = torch.sigmoid(read.logit_expect_retain)
        retention_decay = torch.sigmoid(read.logit_retention_decay)
        retention_strength = torch.sigmoid(read.logit_retention_strength)
        outputs: list[Tensor] = []
        long_tape: list[Tensor] = []
        expected_tape: list[Tensor] = []
        retention_tape: list[Tensor] = []
        for step in range(steps):
            long_tape.append(long)
            expected_tape.append(expected)
            retention_tape.append(retention)
            prediction_error = torch.tanh((write[:, step] - expected).abs())
            plastic_salience = plasticity[:, step] * prediction_error
            expected = expect_retain * expected + (1.0 - expect_retain) * write[:, step]
            retention = retention_decay * retention + (1.0 - retention_decay) * (
                1.0 - prediction_error
            )
            effective_retain = retain + (
                (1.0 - retain) * retention_strength * retention
            )
            long = effective_retain * long + (1.0 - effective_retain) * (
                plastic_salience * write[:, step]
            )
            outputs.append(long)
        output = torch.stack(outputs, dim=1)
        tape = LongReadOnlineTape(
            write=write.detach(),
            plasticity=plasticity.detach(),
            novelty=novelty.detach(),
            long=torch.stack(long_tape, dim=1).detach(),
            expected=torch.stack(expected_tape, dim=1).detach(),
            retention=torch.stack(retention_tape, dim=1).detach(),
        )
    return output, tape, (long.detach(), expected.detach(), retention.detach())


def _advance_long_read_eligibility(
    read,
    tape: LongReadOnlineTape,
    step: int,
    eligibility: LongReadEligibilityState,
) -> LongReadEligibilityState:
    dtype = eligibility.traces.dtype
    long = _dual_basis(tape.long[:, step].to(dtype), 0)
    expected = _dual_basis(tape.expected[:, step].to(dtype), 1)
    retention = _dual_basis(tape.retention[:, step].to(dtype), 2)
    parameters = [
        _dual_basis(
            parameter.detach().reshape(1, 1).expand(tape.batch_size, tape.hidden_dim).to(dtype),
            3 + index,
        )
        for index, parameter in enumerate(_long_parameter_tuple(read))
    ]
    write = _dual_constant(tape.write[:, step].to(dtype))
    plasticity = _dual_constant(tape.plasticity[:, step].to(dtype))
    prediction_error = _dual_tanh(_dual_abs(write - expected))
    plastic_salience = plasticity * prediction_error
    retain = _dual_sigmoid(parameters[0])
    expect_retain = _dual_sigmoid(parameters[1])
    retention_decay = _dual_sigmoid(parameters[2])
    retention_strength = _dual_sigmoid(parameters[3])
    next_expected = expect_retain * expected + (1.0 - expect_retain) * write
    next_retention = retention_decay * retention + (1.0 - retention_decay) * (
        1.0 - prediction_error
    )
    effective_retain = retain + (1.0 - retain) * retention_strength * next_retention
    next_long = effective_retain * long + (1.0 - effective_retain) * (
        plastic_salience * write
    )
    state_derivatives = torch.stack(
        [next_long.derivative, next_expected.derivative, next_retention.derivative],
        dim=2,
    )
    jacobian = state_derivatives[..., :3]
    direct = state_derivatives[..., 3:7]
    next_traces = torch.einsum(
        "bhoi,bhip->bhop", jacobian, eligibility.traces
    ) + direct
    return LongReadEligibilityState(next_traces.contiguous())


def _long_read_chunk_input_vjp(
    read,
    tape: LongReadOnlineTape,
    output_gradient: Tensor,
) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
    carry = (
        torch.zeros_like(tape.long[:, 0]),
        torch.zeros_like(tape.expected[:, 0]),
        torch.zeros_like(tape.retention[:, 0]),
    )
    write_gradients = torch.zeros_like(tape.write)
    plasticity_gradients = torch.zeros_like(tape.plasticity)
    novelty_gradients = torch.zeros_like(tape.novelty)
    parameter_values = tuple(parameter.detach() for parameter in _long_parameter_tuple(read))

    for step in range(tape.steps - 1, -1, -1):
        with torch.enable_grad():
            leaves = tuple(
                value[:, step].detach().requires_grad_(True)
                for value in (
                    tape.long,
                    tape.expected,
                    tape.retention,
                    tape.write,
                    tape.plasticity,
                    tape.novelty,
                )
            )
            long, expected, retention, write, plasticity, novelty = leaves
            del novelty
            retain, expect_retain, retention_decay, retention_strength = (
                torch.sigmoid(parameter) for parameter in parameter_values
            )
            prediction_error = torch.tanh((write - expected).abs())
            plastic_salience = plasticity * prediction_error
            next_expected = expect_retain * expected + (1.0 - expect_retain) * write
            next_retention = retention_decay * retention + (1.0 - retention_decay) * (
                1.0 - prediction_error
            )
            effective_retain = retain + (
                (1.0 - retain) * retention_strength * next_retention
            )
            next_long = effective_retain * long + (1.0 - effective_retain) * (
                plastic_salience * write
            )
            derivatives = torch.autograd.grad(
                (next_long, next_expected, next_retention),
                leaves[:5],
                grad_outputs=(output_gradient[:, step] + carry[0], carry[1], carry[2]),
                allow_unused=True,
            )
        carry = tuple(
            torch.zeros_like(leaves[index]) if derivatives[index] is None else derivatives[index]
            for index in range(3)
        )
        write_gradients[:, step] = derivatives[3]
        plasticity_gradients[:, step] = derivatives[4]
    return (
        write_gradients,
        plasticity_gradients,
        novelty_gradients,
    ), carry


def _long_read_chunk_input_vjp_analytic(
    read,
    tape: LongReadOnlineTape,
    output_gradient: Tensor,
) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
    """Closed-form bounded-chunk input VJP for the long recurrence."""
    carry_long = torch.zeros_like(tape.long[:, 0])
    carry_expected = torch.zeros_like(tape.expected[:, 0])
    carry_retention = torch.zeros_like(tape.retention[:, 0])
    write_gradients = torch.zeros_like(tape.write)
    plasticity_gradients = torch.zeros_like(tape.plasticity)
    novelty_gradients = torch.zeros_like(tape.novelty)
    retain = torch.sigmoid(read.logit_retain.detach()).to(tape.write.dtype)
    expect_retain = torch.sigmoid(read.logit_expect_retain.detach()).to(tape.write.dtype)
    retention_decay = torch.sigmoid(read.logit_retention_decay.detach()).to(tape.write.dtype)
    retention_strength = torch.sigmoid(read.logit_retention_strength.detach()).to(
        tape.write.dtype
    )

    for step in range(tape.steps - 1, -1, -1):
        long = tape.long[:, step]
        expected = tape.expected[:, step]
        retention = tape.retention[:, step]
        write = tape.write[:, step]
        plasticity = tape.plasticity[:, step]
        prediction_error = torch.tanh((write - expected).abs())
        plastic_salience = plasticity * prediction_error
        next_retention = retention_decay * retention + (1.0 - retention_decay) * (
            1.0 - prediction_error
        )
        effective_retain = retain + (
            (1.0 - retain) * retention_strength * next_retention
        )
        candidate = plastic_salience * write

        gradient_long_output = output_gradient[:, step] + carry_long
        gradient_effective_retain = gradient_long_output * (long - candidate)
        next_carry_long = gradient_long_output * effective_retain
        gradient_candidate = gradient_long_output * (1.0 - effective_retain)

        gradient_next_retention = carry_retention + gradient_effective_retain * (
            (1.0 - retain) * retention_strength
        )
        next_carry_retention = gradient_next_retention * retention_decay
        gradient_prediction_error = -gradient_next_retention * (1.0 - retention_decay)

        gradient_plastic_salience = gradient_candidate * write
        gradient_write = gradient_candidate * plastic_salience
        gradient_plasticity = gradient_plastic_salience * prediction_error
        gradient_prediction_error = (
            gradient_prediction_error + gradient_plastic_salience * plasticity
        )

        gradient_write = gradient_write + carry_expected * (1.0 - expect_retain)
        next_carry_expected = carry_expected * expect_retain
        difference = write - expected
        gradient_difference = (
            gradient_prediction_error
            * (1.0 - prediction_error.square())
            * torch.sign(difference)
        )
        gradient_write = gradient_write + gradient_difference
        next_carry_expected = next_carry_expected - gradient_difference

        write_gradients[:, step] = gradient_write
        plasticity_gradients[:, step] = gradient_plasticity
        carry_long = next_carry_long
        carry_expected = next_carry_expected
        carry_retention = next_carry_retention

    return (
        write_gradients,
        plasticity_gradients,
        novelty_gradients,
    ), (carry_long, carry_expected, carry_retention)


def long_read_online_replay(
    read,
    tape: LongReadOnlineTape,
    output_gradient: Tensor,
    eligibility_state: LongReadEligibilityState,
    *,
    return_incoming_state_grad: bool = False,
    _input_vjp=None,
):
    """Replay exact long-read parameter eligibility and bounded input credit."""

    expected_shape = (tape.batch_size, tape.steps, tape.hidden_dim)
    if tuple(output_gradient.shape) != expected_shape:
        raise ValueError(f"long-read output gradient must have shape {expected_shape}")
    if (
        eligibility_state.batch_size != tape.batch_size
        or eligibility_state.hidden_dim != tape.hidden_dim
    ):
        raise ValueError("long-read eligibility dimensions do not match tape")
    parameters = _long_parameter_tuple(read)
    gradients = {
        parameter: torch.zeros_like(parameter, dtype=torch.promote_types(parameter.dtype, torch.float32))
        for parameter in parameters
    }
    eligibility = eligibility_state
    for step in range(tape.steps):
        eligibility = _advance_long_read_eligibility(read, tape, step, eligibility)
        contribution = torch.einsum(
            "bh,bhp->p",
            output_gradient[:, step].to(eligibility.traces.dtype),
            eligibility.traces[:, :, 0, :],
        )
        for index, parameter in enumerate(parameters):
            gradients[parameter] += contribution[index].reshape_as(parameter).to(gradients[parameter].dtype)
    input_vjp = _long_read_chunk_input_vjp if _input_vjp is None else _input_vjp
    input_gradients, incoming_state = input_vjp(
        read, tape, output_gradient
    )
    gradients = {parameter: gradient.to(parameter.dtype) for parameter, gradient in gradients.items()}
    if return_incoming_state_grad:
        return eligibility, gradients, input_gradients, incoming_state
    return eligibility, gradients, input_gradients


def long_read_online_replay_analytic(
    read,
    tape: LongReadOnlineTape,
    output_gradient: Tensor,
    eligibility_state: LongReadEligibilityState,
):
    """Online long replay without per-timestep autograd invocation."""
    return long_read_online_replay(
        read,
        tape,
        output_gradient,
        eligibility_state,
        _input_vjp=_long_read_chunk_input_vjp_analytic,
    )


class _OnlineLongReadAutogradBoundary(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        write: Tensor,
        plasticity: Tensor,
        novelty: Tensor,
        initial_long: Tensor,
        initial_expected: Tensor,
        initial_retention: Tensor,
        *args,
    ):
        parameters = args[:-3]
        read, eligibility, frame = args[-3:]
        if not isinstance(eligibility, LongReadEligibilityState):
            raise TypeError("online long-read boundary requires LongReadEligibilityState")
        if not isinstance(frame, OnlineLongReadFrame):
            raise TypeError("online long-read boundary requires OnlineLongReadFrame")
        expected_parameters = _long_parameter_tuple(read)
        if len(parameters) != len(expected_parameters) or any(
            actual is not expected
            for actual, expected in zip(parameters, expected_parameters)
        ):
            raise ValueError("online long-read boundary parameter identity mismatch")
        frame.begin(eligibility)
        native_cuda = write.is_cuda and bool(
            getattr(type(read), "_cuda_native_enabled", False)
        )
        if native_cuda:
            from .kernels.triton import cuda_long_read_online_forward

            output, tape, final_state = cuda_long_read_online_forward(
                read,
                write,
                plasticity,
                novelty,
                (initial_long, initial_expected, initial_retention),
                eligibility,
            )
        else:
            output, tape, final_state = long_read_online_forward(
                read,
                write,
                plasticity,
                novelty,
                (initial_long, initial_expected, initial_retention),
                eligibility,
            )
        ctx.set_materialize_grads(False)
        ctx.read = read
        ctx.eligibility = eligibility
        ctx.frame = frame
        ctx.tape = tape
        ctx.parameter_count = len(parameters)
        ctx.native_cuda = native_cuda
        return output, *final_state

    @staticmethod
    def backward(ctx, output_gradient, *final_gradients):
        if any(
            gradient is not None and bool(torch.count_nonzero(gradient).item())
            for gradient in final_gradients
        ):
            raise RuntimeError(
                "online temporal credit does not accept a loss on returned final long-read state"
            )
        if output_gradient is None:
            output_gradient = torch.zeros_like(ctx.tape.write)
        if ctx.native_cuda:
            from .kernels.triton import cuda_long_read_online_replay

            eligibility, parameter_gradients, input_gradients = cuda_long_read_online_replay(
                ctx.read,
                ctx.tape,
                output_gradient,
                ctx.eligibility,
            )
        else:
            eligibility, parameter_gradients, input_gradients = long_read_online_replay(
                ctx.read,
                ctx.tape,
                output_gradient,
                ctx.eligibility,
            )
        ctx.frame.complete(eligibility)
        ordered = tuple(
            parameter_gradients[parameter]
            for parameter in _long_parameter_tuple(ctx.read)
        )
        return (
            *input_gradients,
            None,
            None,
            None,
            *ordered,
            None,
            None,
            None,
        )


def long_read_online_autograd(
    read,
    write: Tensor,
    plasticity: Tensor,
    novelty: Tensor,
    initial_state: tuple[Tensor, Tensor, Tensor],
    eligibility_state: LongReadEligibilityState,
    frame: OnlineLongReadFrame,
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
    """Apply the exact long-read online boundary with normal autograd outputs."""

    outputs = _OnlineLongReadAutogradBoundary.apply(
        write,
        plasticity,
        novelty,
        *initial_state,
        *_long_parameter_tuple(read),
        read,
        eligibility_state,
        frame,
    )
    return outputs[0], (outputs[1], outputs[2], outputs[3])


__all__ = [
    "CORE_ELIGIBILITY_SCHEMA_VERSION",
    "CORE_GLOBAL_PARAMETER_NAMES",
    "CORE_LOCAL_PARAMETER_NAMES",
    "CoreEligibilityState",
    "CoreForwardResult",
    "CoreOnlineTape",
    "CoreSignalGrads",
    "LongReadEligibilityState",
    "LongReadOnlineTape",
    "OnlineCoreFrame",
    "OnlineLongReadFrame",
    "OnlineLayerState",
    "OnlineTemporalState",
    "core_online_autograd",
    "core_online_forward_reference",
    "core_online_replay_reference",
    "finalize_online_temporal_state",
    "initial_core_eligibility",
    "initial_long_read_eligibility",
    "initial_online_temporal_state",
    "long_read_online_forward",
    "long_read_online_replay",
    "long_read_online_autograd",
    "prepare_online_temporal_state",
]
