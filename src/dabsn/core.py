"""DABSN novelty-budget-plasticity recurrence."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _parameter(shape: int | tuple[int, ...], value: float) -> nn.Parameter:
    if isinstance(shape, int):
        shape = (shape,)
    return nn.Parameter(torch.full(shape, float(value)))


class DABSNCore(nn.Module):
    """Recurrent core with novelty modulation and retained saturation state."""

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

    def forward_from_state(
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

        Enabling the native CPU or CUDA runtime replaces this method with the
        corresponding C++/OpenMP or Triton carried-state scan.
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
                    f"{(batch, self.hidden_dim)}, got {tuple(budget.shape)} and {tuple(energy.shape)}"
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
            budget = (
                (1.0 - gate_alpha) * budget
                + self.beta
                + lam * (plasticity * expression)
            )
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
        next_budget = (
            (1.0 - gate_alpha) * budget
            + self.beta
            + lam * (plasticity * expression)
        )
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
