import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    fro_norm = np.linalg.norm(M, ord='fro')
    if fro_norm == 0:
        return M
    M = np.array(M)/np.linalg.norm(M, ord='fro')
    for i in range(num_iters):
        M = a*M + b*np.matmul(np.matmul(M, M.T), M) + c*np.matmul(np.linalg.matrix_power(np.matmul(M, M.T), 2), M)
    return M
