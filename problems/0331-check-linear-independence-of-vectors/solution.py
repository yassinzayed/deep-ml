import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    A = np.array(vectors)
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
    if n == 0:
        return True
    else:
        return False