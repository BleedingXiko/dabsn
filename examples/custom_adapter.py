"""Custom adapters for multi-task industrial telemetry forecasting.

The input at each time step combines five continuous measurements, elapsed
time, a sensor identifier, and a missingness flag. The output predicts both the
next event class and a log-normal distribution over time to that event.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dabsn import DABSNLayerSpec, DABSNTaskModel
from dabsn.adapters import register_input_adapter, register_output_head

RAW_FEATURES = 8
EVENT_CLASSES = 4
OUTPUT_DIM = EVENT_CLASSES + 2


class TelemetryInputAdapter(nn.Module):
    """Encode mixed continuous, temporal, categorical, and missingness data."""

    def __init__(
        self,
        raw_input_dim: int,
        model_input_dim: int,
        *,
        num_sensors: int = 32,
    ) -> None:
        super().__init__()
        if raw_input_dim != RAW_FEATURES:
            raise ValueError(f"telemetry rows require {RAW_FEATURES} features")
        self.output_dim = model_input_dim
        self.num_sensors = num_sensors
        self.value_norm = nn.LayerNorm(5)
        self.sensor_embedding = nn.Embedding(num_sensors, 12)
        self.missing_embedding = nn.Embedding(2, 4)
        self.register_buffer(
            "time_frequencies",
            torch.tensor([1.0, 2.0, 4.0, 8.0]),
        )
        fused_dim = 5 + 8 + 12 + 4
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, model_input_dim * 2),
            nn.SiLU(),
            nn.Linear(model_input_dim * 2, model_input_dim),
            nn.LayerNorm(model_input_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        values = torch.nan_to_num(inputs[..., :5].float())
        elapsed = inputs[..., 5].float().clamp_min(0.0)
        sensor_ids = inputs[..., 6].long().clamp(0, self.num_sensors - 1)
        missing = inputs[..., 7].long().clamp(0, 1)

        phase = torch.log1p(elapsed).unsqueeze(-1) * self.time_frequencies
        time_features = torch.cat([phase.sin(), phase.cos()], dim=-1)
        fused = torch.cat(
            [
                self.value_norm(values),
                time_features,
                self.sensor_embedding(sensor_ids),
                self.missing_embedding(missing),
            ],
            dim=-1,
        )
        return self.fusion(fused)


class EventForecastHead(nn.Module):
    """Predict event logits and log-normal time-to-event parameters."""

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        if out_dim < 3:
            raise ValueError("event forecast output needs event classes plus two time parameters")
        self.event_classes = out_dim - 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.event_logits = nn.Linear(hidden_dim, self.event_classes)
        self.time_distribution = nn.Linear(hidden_dim, 2)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.norm(hidden)
        return torch.cat(
            [self.event_logits(hidden), self.time_distribution(hidden)],
            dim=-1,
        )

    def unpack(self, output: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = output[..., : self.event_classes]
        log_time_mean = output[..., -2]
        log_time_scale = F.softplus(output[..., -1]) + 1e-4
        return logits, log_time_mean, log_time_scale


register_input_adapter(
    "telemetry",
    lambda raw_dim, model_dim: TelemetryInputAdapter(
        raw_dim,
        raw_dim if model_dim is None else model_dim,
    ),
)
register_output_head("event_forecast", EventForecastHead)


def build_model() -> DABSNTaskModel:
    return DABSNTaskModel(
        raw_input_dim=RAW_FEATURES,
        model_input_dim=96,
        out_dim=OUTPUT_DIM,
        layers=[
            DABSNLayerSpec(hidden_dim=96, state_dim=64, read_geometry="seq"),
            DABSNLayerSpec(hidden_dim=128, state_dim=96, read_geometry="seq"),
        ],
        input_adapter="telemetry",
        output_adapter="event_forecast",
    )


def example_batch(batch: int = 8, steps: int = 32) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(batch, steps, 5, generator=generator)
    elapsed = torch.rand(batch, steps, 1, generator=generator) * 30.0
    sensor = torch.randint(0, 32, (batch, steps, 1), generator=generator).float()
    missing = (torch.rand(batch, steps, 1, generator=generator) < 0.08).float()
    values = values.masked_fill(missing.bool(), 0.0)
    inputs = torch.cat([values, elapsed, sensor, missing], dim=-1)

    event = ((values[..., 2] > 0.7).long() + 2 * (values[..., 3] > 0.9).long()).clamp_max(
        EVENT_CLASSES - 1
    )
    time_to_event = 2.0 + elapsed[..., 0] * 0.2 + values[..., 0].abs() + missing[..., 0] * 3.0
    return inputs, event, time_to_event


def main() -> None:
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    inputs, event_target, time_target = example_batch()

    output = model.forward_sequence(inputs)
    head = model.body.output_adapter
    event_logits, log_time_mean, log_time_scale = head.unpack(output)
    event_loss = F.cross_entropy(
        event_logits.flatten(0, 1),
        event_target.flatten(),
    )
    normalized_error = (torch.log(time_target) - log_time_mean) / log_time_scale
    time_nll = (log_time_scale.log() + 0.5 * normalized_error.square()).mean()
    loss = event_loss + 0.2 * time_nll

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(
        {
            "output_shape": tuple(output.shape),
            "loss": float(loss.detach()),
            "parameters": model.num_params(),
        }
    )


if __name__ == "__main__":
    main()
