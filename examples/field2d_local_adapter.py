import torch.nn as nn

from dabsn.kernels import local_field_gather


class Field2DLocalInputAdapter(nn.Module):
    """Gather a fixed neighborhood table from a flattened 2D field."""

    def __init__(self, patch_indices):
        super().__init__()
        self.register_buffer("patch_indices", patch_indices.long())

    def forward(self, field):
        batch, height, width, channels = field.shape
        flat = field.reshape(batch, height * width, channels)
        gathered, _backend = local_field_gather(flat, self.patch_indices)
        return gathered
