"""DABSN novelty-budget-plasticity recurrence."""

from __future__ import annotations

import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _parameter(shape: int | tuple[int, ...], value: float) -> nn.Parameter:
    if isinstance(shape, int):
        shape = (shape,)
    return nn.Parameter(torch.full(shape, float(value)))


class DABSNCore(nn.Module):
    """Recurrent core with novelty modulation and retained saturation state."""

    W: nn.Linear
    A: nn.Linear
    Wg: nn.Linear
    Ug: nn.Linear
    beta: Tensor
    logit_alpha: Tensor
    log_lambda: Tensor
    log_kappa: Tensor
    logit_recover: Tensor
    k_s: Tensor
    k_y: Tensor
    k_b: Tensor
    k_n: Tensor
    k_bias: Tensor
    r_s: Tensor
    r_y: Tensor
    r_b: Tensor
    r_n: Tensor
    r_bias: Tensor
    logit_saturation_decay: Tensor
    k_saturation: Tensor
    r_saturation: Tensor
    logit_saturation_suppress: Tensor

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kappa_init: float = -2.25,
        recover_init: float = -4.60,
        a_gain: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.a_gain = a_gain

        self.W = nn.Linear(input_dim, hidden_dim, bias=False)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wg = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Ug = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.beta = _parameter(hidden_dim, 0.0)

        self.logit_alpha = _parameter(1, -2.0)
        self.log_lambda = _parameter(1, -1.0)
        self.log_kappa = _parameter(hidden_dim, kappa_init)
        self.logit_recover = _parameter(hidden_dim, recover_init)
        for name in (
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
        ):
            setattr(self, name, _parameter(hidden_dim, 0.0))

        self.logit_saturation_decay = _parameter(1, 2.2)
        self.k_saturation = _parameter(hidden_dim, 0.0)
        self.r_saturation = _parameter(hidden_dim, 0.0)
        self.logit_saturation_suppress = _parameter(1, -6.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.W, self.A, self.Wg, self.Ug):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.A.weight, gain=self.a_gain)
        nn.init.zeros_(self.beta)
        nn.init.constant_(self.Wg.bias, -2.0)

    def initial_state(
        self,
        batch_size: int,
        *,
        device=None,
        dtype=None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        reference = next(self.parameters())
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype
        budget = torch.zeros(batch_size, self.hidden_dim, dtype=dtype, device=device)
        energy = torch.ones_like(budget)
        saturation = torch.zeros_like(budget)
        return budget, energy, saturation

    def _reference_forward_from_state(
        self,
        inputs: Tensor,
        *,
        initial_state: tuple[Tensor, Tensor, Tensor] | None = None,
        return_writes: bool = False,
        return_cocktail: bool = False,
        return_final_state: bool = False,
    ):
        """Run the core recurrence, optionally from a carried state.

        The carried tuple contains only the core's budget, energy, and
        saturation tensors. It does not include the state owned by
        :class:`dabsn.read.DABSNRead`, so this method alone is not a complete
        chunked or streaming block/model API.

        This is the mathematical oracle retained for kernel conformance. The
        public carried-state path executes the registered operator below.
        """
        batch, steps, _ = inputs.shape
        gate_alpha = torch.sigmoid(self.logit_alpha)
        lam = F.softplus(self.log_lambda)
        kappa = F.softplus(self.log_kappa)
        recover = torch.sigmoid(self.logit_recover)

        projected = self.W(inputs)
        gate_projected = self.Wg(inputs)
        fused_recurrence = torch.cat([self.Ug.weight, self.A.weight], dim=0)

        if initial_state is None:
            budget, energy, saturation = self.initial_state(
                batch,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            budget, energy, saturation = (
                value.to(device=inputs.device, dtype=inputs.dtype) for value in initial_state
            )
            if budget.shape != (batch, self.hidden_dim) or energy.shape != (batch, self.hidden_dim):
                raise ValueError(
                    "initial_state budget/energy must have shape "
                    f"{(batch, self.hidden_dim)}, got {tuple(budget.shape)} "
                    f"and {tuple(energy.shape)}"
                )
            if saturation.shape != (batch, self.hidden_dim):
                raise ValueError(
                    "initial_state saturation must have shape "
                    f"{(batch, self.hidden_dim)}, got {tuple(saturation.shape)}"
                )

        saturation_decay = torch.sigmoid(self.logit_saturation_decay)
        trajectories: list[Tensor] = []
        novelty_trajectory: list[Tensor] = []
        plasticity_trajectory: list[Tensor] = []
        expression_trajectory: list[Tensor] = []
        write_trajectory: list[Tensor] = []
        energy_trajectory: list[Tensor] = []
        saturation_trajectory: list[Tensor] = []

        for step in range(steps):
            y = torch.tanh(projected[:, step, :] + budget)
            gate_recurrence, expression = F.linear(y, fused_recurrence).chunk(2, dim=1)
            gate = F.hardsigmoid(gate_projected[:, step, :] + gate_recurrence)
            novelty = torch.tanh((expression - budget).abs())
            stress = novelty * (1.0 - energy)
            saturation = saturation_decay * saturation + (1.0 - saturation_decay) * stress
            novelty_effective = novelty * (
                1.0 - torch.sigmoid(self.logit_saturation_suppress) * saturation
            )

            k_signal = (
                self.k_s * gate
                + self.k_y * y
                + self.k_b * torch.tanh(budget)
                + self.k_n * novelty_effective
                + self.k_bias
            )
            r_signal = (
                self.r_s * gate
                + self.r_y * y
                + self.r_b * torch.tanh(budget)
                + self.r_n * novelty_effective
                + self.r_bias
            )
            k_signal = k_signal + self.k_saturation * saturation
            r_signal = r_signal + self.r_saturation * saturation
            write_cost = kappa * torch.exp(0.5 * torch.tanh(k_signal))
            recovery = torch.clamp(
                recover * torch.exp(0.5 * torch.tanh(r_signal)),
                0.0,
                1.0,
            )

            plasticity = gate * energy
            if return_cocktail:
                energy_trajectory.append(energy)
                saturation_trajectory.append(saturation)
            budget = (1.0 - gate_alpha) * budget + self.beta + lam * (plasticity * expression)
            energy = torch.clamp(
                energy + recovery * (1.0 - energy) - write_cost * plasticity,
                0.0,
                1.0,
            )

            trajectories.append(torch.cat([y, budget], dim=-1))
            novelty_trajectory.append(novelty)
            plasticity_trajectory.append(plasticity)
            if return_writes:
                expression_trajectory.append(expression)
                write_trajectory.append(plasticity * expression)

        trajectory = torch.stack(trajectories, dim=1)
        novelty = torch.stack(novelty_trajectory, dim=1)
        plasticity = torch.stack(plasticity_trajectory, dim=1)
        result: tuple[Tensor, ...]
        if return_writes:
            expression = torch.stack(expression_trajectory, dim=1)
            write = torch.stack(write_trajectory, dim=1)
            if return_cocktail:
                result = (
                    trajectory,
                    novelty,
                    plasticity,
                    expression,
                    write,
                    torch.stack(energy_trajectory, dim=1),
                    torch.stack(saturation_trajectory, dim=1),
                )
            else:
                result = trajectory, novelty, plasticity, expression, write
        else:
            result = trajectory, novelty, plasticity
        if return_final_state:
            return result, (budget, energy, saturation)
        return result

    def forward_from_state(
        self,
        inputs: Tensor,
        *,
        initial_state: tuple[Tensor, Tensor, Tensor] | None = None,
        return_writes: bool = False,
        return_cocktail: bool = False,
        return_final_state: bool = False,
    ):
        """Run the canonical registered core scan with optional carried state."""

        if inputs.dim() != 3:
            raise ValueError("DABSNCore inputs must have shape [B,T,input_dim]")
        batch = inputs.shape[0]
        if initial_state is None:
            initial_state = self.initial_state(
                batch,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            state_values = tuple(
                value.to(device=inputs.device, dtype=inputs.dtype) for value in initial_state
            )
            expected = (batch, self.hidden_dim)
            if any(value.shape != expected for value in state_values):
                shapes = tuple(tuple(value.shape) for value in state_values)
                raise ValueError(f"initial_state tensors must have shape {expected}, got {shapes}")
            initial_state = (state_values[0], state_values[1], state_values[2])

        from .kernels.batched_runtime import dabsn_core_scan_batched

        outputs = dabsn_core_scan_batched(
            self.W(inputs),
            self.Wg(inputs),
            self.Ug.weight,
            self.A.weight,
            self.beta,
            self.log_kappa,
            self.logit_recover,
            self.k_s,
            self.k_y,
            self.k_b,
            self.k_n,
            self.k_bias,
            self.r_s,
            self.r_y,
            self.r_b,
            self.r_n,
            self.r_bias,
            self.logit_saturation_decay.expand(self.hidden_dim),
            self.k_saturation,
            self.r_saturation,
            self.logit_alpha.reshape(()),
            self.log_lambda.reshape(()),
            self.logit_saturation_suppress.reshape(()),
            return_tape=return_cocktail,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )
        if not torch.compiler.is_compiling():
            if (
                inputs.device.type == "cpu"
                and outputs[0].dtype == torch.float32
                and bool(getattr(type(self), "_cpu_native_enabled", False))
            ):
                self._last_core_backend = "cpu_native_cpp"
            elif inputs.device.type == "cuda" and bool(
                getattr(type(self), "_cuda_native_enabled", False)
            ):
                self._last_core_backend = "cuda_registered"
            else:
                self._last_core_backend = "registered_reference"
        trajectory, novelty, plasticity, expression, write = outputs[:5]
        result: tuple[Tensor, ...]
        if return_cocktail:
            result = (
                trajectory,
                novelty,
                plasticity,
                expression,
                write,
                outputs[5],
                outputs[6],
            )
            final_offset = 7
        elif return_writes:
            result = trajectory, novelty, plasticity, expression, write
            final_offset = 5
        else:
            result = trajectory, novelty, plasticity
            final_offset = 5
        if return_final_state:
            return result, tuple(outputs[final_offset : final_offset + 3])
        return result

    def forward(
        self,
        inputs: Tensor,
        return_writes: bool = False,
        return_cocktail: bool = False,
    ):
        return self.forward_from_state(
            inputs,
            return_writes=return_writes,
            return_cocktail=return_cocktail,
        )

    def step(
        self,
        input_step: Tensor,
        state: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor, Tensor]]:
        budget, energy, saturation = state
        gate_alpha = torch.sigmoid(self.logit_alpha)
        lam = F.softplus(self.log_lambda)
        kappa = F.softplus(self.log_kappa)
        recover = torch.sigmoid(self.logit_recover)

        y = torch.tanh(self.W(input_step) + budget)
        gate_recurrence, expression = F.linear(
            y,
            torch.cat([self.Ug.weight, self.A.weight], dim=0),
        ).chunk(2, dim=1)
        gate = F.hardsigmoid(self.Wg(input_step) + gate_recurrence)
        novelty = torch.tanh((expression - budget).abs())
        saturation_decay = torch.sigmoid(self.logit_saturation_decay)
        stress = novelty * (1.0 - energy)
        saturation = saturation_decay * saturation + (1.0 - saturation_decay) * stress
        novelty_effective = novelty * (
            1.0 - torch.sigmoid(self.logit_saturation_suppress) * saturation
        )
        k_signal = (
            self.k_s * gate
            + self.k_y * y
            + self.k_b * torch.tanh(budget)
            + self.k_n * novelty_effective
            + self.k_bias
            + self.k_saturation * saturation
        )
        r_signal = (
            self.r_s * gate
            + self.r_y * y
            + self.r_b * torch.tanh(budget)
            + self.r_n * novelty_effective
            + self.r_bias
            + self.r_saturation * saturation
        )
        write_cost = kappa * torch.exp(0.5 * torch.tanh(k_signal))
        recovery = torch.clamp(
            recover * torch.exp(0.5 * torch.tanh(r_signal)),
            0.0,
            1.0,
        )
        plasticity = gate * energy
        energy_before_write = energy
        next_budget = (1.0 - gate_alpha) * budget + self.beta + lam * (plasticity * expression)
        next_energy = torch.clamp(
            energy + recovery * (1.0 - energy) - write_cost * plasticity,
            0.0,
            1.0,
        )
        signals = {
            "y": y,
            "b": next_budget,
            "ay": expression,
            "write": plasticity * expression,
            "novelty": novelty,
            "p": plasticity,
            "energy": energy_before_write,
            "saturation": saturation,
        }
        return signals, (next_budget, next_energy, saturation)


class _GatherHidden(torch.autograd.Function):
    """Gather uneven hidden shards with an explicit backward reduction policy."""

    @staticmethod
    def forward(ctx, local: Tensor, group, sizes: tuple[int, ...], average: bool):
        world = len(sizes)
        rank = torch_dist.get_rank(group)
        padded_width = max(sizes)
        padded = local
        if local.shape[-1] < padded_width:
            padded = F.pad(local, (0, padded_width - local.shape[-1]))
        buffers = [torch.empty_like(padded) for _ in range(world)]
        torch_dist.all_gather(buffers, padded.contiguous(), group=group)
        ctx.group = group
        ctx.rank = rank
        ctx.sizes = sizes
        ctx.average = average
        return torch.cat(
            [buffer[..., :width] for buffer, width in zip(buffers, sizes)],
            dim=-1,
        )

    @staticmethod
    def backward(ctx, gradient: Tensor):
        reduced = gradient.contiguous()
        torch_dist.all_reduce(reduced, op=torch_dist.ReduceOp.SUM, group=ctx.group)
        if ctx.average:
            reduced.div_(len(ctx.sizes))
        start = sum(ctx.sizes[: ctx.rank])
        width = ctx.sizes[ctx.rank]
        return reduced[..., start : start + width], None, None, None


_SYMMETRIC_GROUP_CACHE: dict[int, str | None] = {}


def _symmetric_group_name(group) -> str | None:
    """The group's symmetric-memory name, or None if it cannot host one.

    Symmetric memory is a single-node, NVLink-class transport, and whether a
    given machine can actually run the fused collective is not answerable by
    inspection: on a pair of T4s ``enable_symm_mem_for_group`` returns happily
    and then the first ``_fused_all_gather_matmul`` dies with "CUDA driver
    error: invalid device ordinal". Registering a group is not the same question
    as being able to use it, so the probe is an actual tiny fused call, run once
    per group and cached. A capability nobody has exercised is a guess.

    A negative answer is ordinary, not an error -- gloo groups, builds without
    symmetric memory, and cards without the interconnect all take the explicit
    gather-then-matmul path, which is the same arithmetic through the same
    autograd node.
    """
    key = id(group)
    if key in _SYMMETRIC_GROUP_CACHE:
        return _SYMMETRIC_GROUP_CACHE[key]

    resolved: str | None = None
    try:
        from torch.distributed import _symmetric_memory as symm_mem

        name = getattr(group, "group_name", None)
        if isinstance(name, str):
            symm_mem.enable_symm_mem_for_group(name)
            world = torch_dist.get_world_size(group)
            device = torch.device("cuda", torch.cuda.current_device())
            probe = torch.zeros(8, 8, device=device, dtype=torch.bfloat16)
            weight = torch.zeros(8 * world, 8, device=device, dtype=torch.bfloat16)
            symm_mem._fused_all_gather_matmul(
                probe, [weight.t()], gather_dim=1, group_name=name
            )
            resolved = name
    except Exception:  # noqa: BLE001 - capability probe, any failure means "no"
        resolved = None

    _SYMMETRIC_GROUP_CACHE[key] = resolved
    return resolved


class _GatherHiddenRecurrent(torch.autograd.Function):
    """Gather this rank's ``y`` and apply the recurrent matrix as one node.

    These two operations are the entire per-step cost of a sharded recurrence,
    and the gathered ``y`` feeds nothing else in the step
    (``kernels/batched_runtime.py`` forms ``recurrent_out`` from it and then
    never touches it again), so there is no reason for them to be two round
    trips. Fusing them also removes the gathered [B,H] tensor from the saved
    activations of every step: the backward reconstructs what it needs from the
    product instead.

    On a symmetric-memory-capable group this dispatches to
    ``_fused_all_gather_matmul``, which micro-pipelines the transfer against the
    GEMM rather than running the transfer to completion and then starting the
    GEMM. Everywhere else -- gloo, CPU, any build without it, and any uneven
    shard split, which ``all_gather_single`` cannot express -- it runs the same
    arithmetic written out explicitly. The two forwards are the same function of
    the same inputs, so they share this one backward, and the multi-process CPU
    test exercises that backward directly.

    The backward is the part that has to be right. Every rank computed with the
    full gathered ``y``, so every rank holds a partial gradient for every unit;
    summing them and taking this rank's slice is the reduce-scatter that closes
    the loop. Omitting it yields a forward that matches the unsharded model
    exactly and gradients that are quietly incomplete -- the model trains, just
    not the model you think. That is the same failure ``_GatherHidden`` documents,
    and it is why the gradient is asserted against a single-process baseline
    rather than merely shape-checked.
    """

    @staticmethod
    def forward(ctx, local: Tensor, recurrent: Tensor, group, sizes: tuple[int, ...]):
        rank = torch_dist.get_rank(group)
        compute_dtype = recurrent.dtype
        shard = local.to(compute_dtype).contiguous()

        fused_name = None
        if len(set(sizes)) == 1 and local.is_cuda:
            fused_name = _symmetric_group_name(group)

        if fused_name is not None:
            from torch.distributed import _symmetric_memory as symm_mem

            gathered, outputs = symm_mem._fused_all_gather_matmul(
                shard,
                [recurrent.t()],
                gather_dim=shard.dim() - 1,
                group_name=fused_name,
            )
            recurrent_out = outputs[0]
        else:
            padded_width = max(sizes)
            padded = shard
            if shard.shape[-1] < padded_width:
                padded = F.pad(shard, (0, padded_width - shard.shape[-1]))
            buffers = [torch.empty_like(padded) for _ in range(len(sizes))]
            torch_dist.all_gather(buffers, padded.contiguous(), group=group)
            gathered = torch.cat(
                [buffer[..., :width] for buffer, width in zip(buffers, sizes)],
                dim=-1,
            )
            recurrent_out = F.linear(gathered, recurrent)

        ctx.group = group
        ctx.rank = rank
        ctx.sizes = sizes
        ctx.local_dtype = local.dtype
        ctx.save_for_backward(gathered, recurrent)
        return recurrent_out

    @staticmethod
    def backward(ctx, grad_recurrent_out: Tensor):
        gathered, recurrent = ctx.saved_tensors
        grad_out = grad_recurrent_out.to(recurrent.dtype).contiguous()

        # This rank owns the rows of `recurrent` that produced its own units, so
        # its parameter gradient is complete without a collective: it is formed
        # entirely from this rank's outgoing gradient and the gathered y.
        flat_out = grad_out.reshape(-1, grad_out.shape[-1])
        flat_gathered = gathered.reshape(-1, gathered.shape[-1])
        grad_recurrent = flat_out.t() @ flat_gathered

        # The activation gradient is the opposite case: this rank consumed every
        # rank's units, so every rank holds a partial for every unit. Sum, then
        # keep this rank's slice.
        grad_gathered = (flat_out @ recurrent).reshape(gathered.shape).contiguous()
        torch_dist.all_reduce(grad_gathered, op=torch_dist.ReduceOp.SUM, group=ctx.group)
        start = sum(ctx.sizes[: ctx.rank])
        width = ctx.sizes[ctx.rank]
        grad_local = grad_gathered[..., start : start + width].to(ctx.local_dtype)
        return grad_local, grad_recurrent, None, None


class _SumReplicatedInputGradient(torch.autograd.Function):
    """Identity forward whose backward combines hidden-shard input contributions."""

    @staticmethod
    def forward(ctx, value: Tensor, group):
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, gradient: Tensor):
        reduced = gradient.contiguous()
        torch_dist.all_reduce(reduced, op=torch_dist.ReduceOp.SUM, group=ctx.group)
        return reduced, None


class TensorParallelDABSNCore(DABSNCore):
    """Live hidden-dimension shard of one DABSN core.

    This is a DABSN-provider implementation detail, not a generic framework
    policy. Each worker owns a contiguous set of state units and the rows that
    produce them. The recurrent input is gathered at every step; complete
    public tapes are reconstructed once after the scan so ``DABSNBlock`` and
    carried-memory callers retain the ordinary core contract.
    """

    def __init__(self, source: DABSNCore, *, group, rank: int, world_size: int) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} outside tensor worker count {world_size}")
        base, extra = divmod(source.hidden_dim, world_size)
        sizes = tuple(base + (1 if worker < extra else 0) for worker in range(world_size))
        if not sizes[rank]:
            raise ValueError(
                f"DABSN state width {source.hidden_dim} is smaller than tensor workers "
                f"{world_size}"
            )
        start = sum(sizes[:rank])
        stop = start + sizes[rank]
        super().__init__(source.W.in_features, sizes[rank], a_gain=source.a_gain)
        self.global_hidden_dim = int(source.hidden_dim)
        self.tensor_rank = int(rank)
        self.tensor_world_size = int(world_size)
        self.tensor_slice = slice(start, stop)
        self.tensor_sizes = sizes
        object.__setattr__(self, "tensor_group", group)

        unit_names = (
            "beta",
            "log_kappa",
            "logit_recover",
            "k_s",
            "k_y",
            "k_b",
            "k_n",
            "k_bias",
            "k_saturation",
            "r_s",
            "r_y",
            "r_b",
            "r_n",
            "r_bias",
            "r_saturation",
        )
        scalar_names = (
            "logit_alpha",
            "log_lambda",
            "logit_saturation_decay",
            "logit_saturation_suppress",
        )
        for name in unit_names:
            setattr(self, name, nn.Parameter(getattr(source, name).detach()[start:stop].clone()))
        for name in scalar_names:
            parameter = nn.Parameter(getattr(source, name).detach().clone())
            setattr(self, name, parameter)

        self.W.weight = nn.Parameter(source.W.weight.detach()[start:stop].clone())
        self.Wg.weight = nn.Parameter(source.Wg.weight.detach()[start:stop].clone())
        self.Wg.bias = nn.Parameter(source.Wg.bias.detach()[start:stop].clone())
        self.Ug.weight = nn.Parameter(source.Ug.weight.detach()[start:stop].clone())
        self.A.weight = nn.Parameter(source.A.weight.detach()[start:stop].clone())
        self.Ug.in_features = self.global_hidden_dim
        self.A.in_features = self.global_hidden_dim

        self._tensor_scalar_gradient_hooks = []
        if world_size > 1:
            for name in scalar_names:
                parameter = getattr(self, name)

                def sum_gradient(gradient, *, process_group=group):
                    synchronized = gradient.clone()
                    torch_dist.all_reduce(synchronized, group=process_group)
                    return synchronized

                self._tensor_scalar_gradient_hooks.append(parameter.register_hook(sum_gradient))

    def _local_state(
        self,
        batch_size: int,
        *,
        device,
        dtype,
        initial_state: tuple[Tensor, Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        local_shape = (batch_size, self.hidden_dim)
        if initial_state is None:
            return (
                torch.zeros(local_shape, device=device, dtype=dtype),
                torch.ones(local_shape, device=device, dtype=dtype),
                torch.zeros(local_shape, device=device, dtype=dtype),
            )
        values = tuple(value.to(device=device, dtype=dtype) for value in initial_state)
        global_shape = (batch_size, self.global_hidden_dim)
        if all(value.shape == global_shape for value in values):
            cut = self.tensor_slice
            return values[0][:, cut], values[1][:, cut], values[2][:, cut]
        if all(value.shape == local_shape for value in values):
            return values[0], values[1], values[2]
        shapes = tuple(tuple(value.shape) for value in values)
        raise ValueError(
            f"tensor-parallel initial state must be three {global_shape} public tensors "
            f"or three {local_shape} local tensors, got {shapes}"
        )

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        reference = next(self.parameters())
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype
        shape = (batch_size, self.global_hidden_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.ones(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def _gather_fields(self, fields: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if self.tensor_world_size == 1:
            return fields
        packed = torch.cat(fields, dim=-1)
        packed_sizes = tuple(len(fields) * width for width in self.tensor_sizes)
        gathered = _GatherHidden.apply(packed, self.tensor_group, packed_sizes, True)
        regrouped: list[list[Tensor]] = [[] for _ in fields]
        offset = 0
        for width in self.tensor_sizes:
            worker = gathered[..., offset : offset + len(fields) * width]
            pieces = worker.split(width, dim=-1)
            for index, piece in enumerate(pieces):
                regrouped[index].append(piece)
            offset += len(fields) * width
        return tuple(torch.cat(pieces, dim=-1) for pieces in regrouped)

    def forward_from_state(
        self,
        inputs: Tensor,
        *,
        initial_state: tuple[Tensor, Tensor, Tensor] | None = None,
        return_writes: bool = False,
        return_cocktail: bool = False,
        return_final_state: bool = False,
    ):
        if inputs.dim() != 3:
            raise ValueError("DABSNCore inputs must have shape [B,T,input_dim]")
        from .kernels.batched_runtime import _forward_step, _state_dtype

        world_inputs = (
            _SumReplicatedInputGradient.apply(inputs, self.tensor_group)
            if self.tensor_world_size > 1
            else inputs
        )
        batch, steps, _ = world_inputs.shape
        state_dtype = _state_dtype(world_inputs.dtype)
        budget, energy, saturation = self._local_state(
            batch,
            device=world_inputs.device,
            dtype=state_dtype,
            initial_state=initial_state,
        )
        projected = self.W(world_inputs)
        gate_projected = self.Wg(world_inputs)
        recurrent = torch.cat((self.Ug.weight, self.A.weight), dim=0)
        sharded = self.tensor_world_size > 1
        tapes: list[list[Tensor]] = [[] for _ in range(8)]
        for step_index in range(steps):
            # Sharded: the gather and the recurrent GEMM go together as one node,
            # so the step receives the finished product instead of the gathered
            # activation. Unsharded: neither is supplied and the step forms the
            # product itself from its own local y, exactly as it always has.
            recurrent_out = (
                _GatherHiddenRecurrent.apply(
                    torch.tanh(projected[:, step_index].to(state_dtype) + budget),
                    recurrent,
                    self.tensor_group,
                    self.tensor_sizes,
                )
                if sharded
                else None
            )
            step = _forward_step(
                projected[:, step_index],
                gate_projected[:, step_index],
                recurrent,
                budget,
                energy,
                saturation,
                self.beta,
                self.log_kappa,
                self.logit_recover,
                self.k_s,
                self.k_y,
                self.k_b,
                self.k_n,
                self.k_bias,
                self.r_s,
                self.r_y,
                self.r_b,
                self.r_n,
                self.r_bias,
                self.logit_saturation_decay.expand(self.hidden_dim),
                self.k_saturation,
                self.r_saturation,
                self.logit_alpha.reshape(()),
                self.log_lambda.reshape(()),
                self.logit_saturation_suppress.reshape(()),
                recurrent_out=recurrent_out,
            )
            y, budget, energy, saturation = step[:4]
            for tape, value in zip(
                tapes,
                (y, budget, step[4], step[5], step[6], step[7], step[8], saturation),
            ):
                tape.append(value)
        local_fields = tuple(torch.stack(values, dim=1) for values in tapes)
        y, budget_tape, novelty, plasticity, expression, write, energy_tape, saturation_tape = (
            self._gather_fields(local_fields)
        )
        trajectory = torch.cat((y, budget_tape), dim=-1)
        result: tuple[Tensor, ...]
        if return_cocktail:
            result = (
                trajectory,
                novelty,
                plasticity,
                expression,
                write,
                energy_tape,
                saturation_tape,
            )
        elif return_writes:
            result = trajectory, novelty, plasticity, expression, write
        else:
            result = trajectory, novelty, plasticity
        if not torch.compiler.is_compiling():
            self._last_core_backend = "tensor_parallel_registered_step"
        if return_final_state:
            final_state = self._gather_fields((budget, energy, saturation))
            return result, final_state
        return result

    def step(
        self,
        input_step: Tensor,
        state: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor, Tensor]]:
        result, final = self.forward_from_state(
            input_step.unsqueeze(1),
            initial_state=state,
            return_writes=True,
            return_cocktail=True,
            return_final_state=True,
        )
        trajectory, novelty, plasticity, expression, write, energy, saturation = result
        y, budget = trajectory.split(self.global_hidden_dim, dim=-1)
        signals = {
            "y": y[:, 0],
            "b": budget[:, 0],
            "ay": expression[:, 0],
            "write": write[:, 0],
            "novelty": novelty[:, 0],
            "p": plasticity[:, 0],
            "energy": energy[:, 0],
            "saturation": saturation[:, 0],
        }
        return signals, final
