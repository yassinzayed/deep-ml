import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""
    indices = np.arange(len(A))
    diags = A[indices, indices]
    return diags
