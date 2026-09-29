"""One update from native Capacity's zero-output initialization."""
import torch

from dabsn import CapacityConfig, build_dabsn
from dabsn.runtime import build_training_optimizer


def main():
    torch.manual_seed(7)
    model = build_dabsn(8, 3, 12, state_dim=10,
                       capacity=CapacityConfig(mode="sparse", size=64, active_units=8))
    optimizer = build_training_optimizer(model, lr=1e-3)
    inputs = torch.randn(2, 6, 8)
    targets = torch.randn(2, 6, 3)
    optimizer.zero_grad(set_to_none=True)
    loss = (model.forward_sequence(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    print("loss", float(loss.detach()), "parameters", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
