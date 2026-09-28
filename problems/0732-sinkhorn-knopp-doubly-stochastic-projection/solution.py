import numpy as np

def sinkhorn_knopp(B: list, t_max: int = 20) -> list:
    """
    Project a square matrix onto the set of doubly stochastic matrices.

    Args:
        B: n x n matrix as a list of lists (real-valued).
        t_max: number of normalization iterations.

    Returns:
        A nested list representing the resulting doubly stochastic matrix.
    """
    M = np.exp(B)
    for i in range(t_max):
        row_sum = np.sum(M, axis=1)
        M = M/(row_sum[:, None])
        col_sum = np.sum(M, axis=0)
        M = M/col_sum
    return M
