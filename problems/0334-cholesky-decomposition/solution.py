import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    # REVISE REVISE REVISE
    # Worry about the order in which your operations are performed
    # If the returned element should be a list then use .tolist() not list()
    # WORRY ABOUT NAN AND INF VALUSE so check whether whats under the root is less than 1 or if the denominator is 0 and learn what that means and wjay should be placed there
    dims = len(A)
    L = np.zeros((dims, dims))
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i == j:
                sums = 0
                for k in range(i):
                    sums += (L[i][k])**2
                under_root = A[i][i] - sums
                if under_root <= 0:
                    return -1
                L[i][j] = np.sqrt(under_root)
            else:
                sums = 0
                for k in range(j):
                    sums += L[i][k]*L[j][k]
                in_parent = A[i][j] - sums
                if L[j][j] == 0:
                    L[i][j] = 0
                else:
                    L[i][j] = (1/(L[j][j]))*in_parent
    return L.tolist()