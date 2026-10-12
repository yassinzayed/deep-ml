import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    raw_scores = np.matmul(Q, K.T)
    divided = raw_scores/np.sqrt(K.shape[1])
    max_val = np.max(divided, axis=-1, keepdims=True)
    exp_scores = np.exp(divided - max_val)
    soft = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return soft
