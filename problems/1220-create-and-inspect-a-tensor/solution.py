import torch

def make_tensor():
    """Return a 2x3 float32 tensor [[1, 2, 3], [4, 5, 6]]."""
    return torch.arange(1.0, 7.0).view(2, 3)
