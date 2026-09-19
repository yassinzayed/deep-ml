import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    A_T_A = np.matmul(A.T, A)
    if A_T_A[0, 0] == A_T_A[1, 1]:
        theta = (np.pi)/4
    else:
        nom = 2*A_T_A[0, 1]
        denom = A_T_A[0, 0] - A_T_A[1, 1]
        theta = 0.5*np.arctan2(nom, denom)
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])
    D = np.matmul(np.matmul(R.T, A_T_A), R)
    trace = D[0,0] + D[1, 1]
    det = D[0,0]*D[1,1] - D[0,1]*D[1,0]
    under_root = trace**2 - 4*det
    eig_1 = (trace + (under_root)**0.5)/2
    eig_2 = (trace - (under_root)**0.5)/2
    S = [np.sqrt(eig_1), np.sqrt(eig_2)]
    Vt = R.T
    sigma = np.diag([1/np.sqrt(eig_1), 1/np.sqrt(eig_2)])
    U = np.matmul(np.matmul(A, R), sigma)
    return U, S, Vt