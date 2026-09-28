import numpy as np

def linspace(start, stop, n: int) -> np.ndarray:
    """n evenly spaced values from start to stop inclusive."""
    if (n-1) == 0:
        return [start]
    step = (stop-start)/(n-1)
    out = [start+i*step for i in range(n)]
    return out
