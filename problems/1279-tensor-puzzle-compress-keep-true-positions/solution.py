import numpy as np

def compress(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pack v[g] into the front of a zero vector of length len(v)."""
    compress_1 = [v[i] for i in range(len(v)) if g[i]]
    compress = np.append(compress_1, np.zeros(len(v)-len(compress_1)))
    return compress
