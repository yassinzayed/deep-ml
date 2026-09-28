import numpy as np

def heaviside(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise heaviside with zero-value b."""
    out = np.zeros_like(a)
    for i in range(len(a)):
        if a[i] < 0:
            out[i] = 0
        elif a[i] == 0:
            out[i] = b[i]
        elif a[i] > 0:
            out[i] = 1
    return out
