import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    # TODO: mean triplet loss with squared L2 distance
    d = torch.sum(torch.pow(anchor-positive, 2)-torch.pow(anchor-negative, 2), dim=-1) + margin
    l = torch.where(d > 0, d, 0)
    return torch.mean(l)
