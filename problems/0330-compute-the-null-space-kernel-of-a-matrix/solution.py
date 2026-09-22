import numpy as np

def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis for the null space (kernel) of matrix A.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering singular values as zero
    
    Returns:
        Matrix of shape (n, k) where k is the dimension of the null space.
        Columns form an orthonormal basis for the null space.
    """
    U, S, Vt = np.linalg.svd(A)
    r = np.linalg.matrix_rank(A)
    V = Vt.T
    NS = V[:, r:len(V)]
    return NS