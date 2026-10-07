import numpy as np

def triplet_margin_loss(anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 1.0) -> float:
    """
    Compute the triplet margin loss for metric learning.
    
    Args:
        anchor: Anchor embeddings, shape (D,) for single or (N, D) for batch
        positive: Positive embeddings (same class as anchor), same shape as anchor
        negative: Negative embeddings (different class from anchor), same shape as anchor
        margin: Minimum desired distance gap between positive and negative pairs
    
    Returns:
        Mean triplet margin loss as a float
    """
    dp = np.sqrt(np.sum(((anchor-positive)**2), axis=-1))
    dn = np.sqrt(np.sum(((anchor-negative)**2), axis=-1))
    loss = np.where(dp-dn+margin > 0, dp-dn+margin, 0)
    return np.mean(loss)