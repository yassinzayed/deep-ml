import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    for i in range(len(A)-1):
        if A[i][i] == 0 and (i != (len(A)-1)):
            for k in range(len(A)):
                if A[k][i] != 0:
                    A[i], A[k] = A[k], A[i]
                    break
        if A[i][i] != 0:
            if A[i][i] != 1:
                A[i] = A[i]/A[i][i]
            for k in range(len(A)-i-1):
                A[k+1+i] = A[k+1+i] - A[i]*(A[k+1+i][i]/A[i][i])
    n = 0
    for i in A:
        if all(x == 0 for x in i):
            n += 1
    return (len(A)-n)

