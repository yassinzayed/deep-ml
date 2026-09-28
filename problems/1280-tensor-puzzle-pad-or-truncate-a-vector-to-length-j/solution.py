import numpy as np

def pad_to(a: np.ndarray, j: int) -> np.ndarray:
    """Pad a with zeros (or truncate) to length j."""
    # Your code here
    out = np.zeros(j)
    n = min(len(a), j)
    out[:n] = a[:n]
    return out
