import torch
import torch.nn.functional as F

from dabsn import DABSNLayerSpec, DABSNModel

model = DABSNModel(8, 3, [DABSNLayerSpec(32, read_geometry="seq")], output_adapter="token")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
inputs = torch.randn(4, 16, 8)
targets = torch.randint(0, 3, (4, 16))
loss = F.cross_entropy(model.forward_sequence(inputs).flatten(0, 1), targets.flatten())
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
print({"loss": float(loss.detach()), "parameters": model.num_params()})
