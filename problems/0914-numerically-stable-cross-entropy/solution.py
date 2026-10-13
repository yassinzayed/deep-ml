import torch

def cross_entropy(logits, targets):
    # TODO: numerically stable mean cross-entropy
    logsumexp = torch.logsumexp(logits, dim=-1)
    target_logits = logits[torch.arange(len(logits)), targets]
    loss = logsumexp - target_logits
    return loss.mean()