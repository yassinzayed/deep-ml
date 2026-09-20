import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    matrix = np.array(matrix)
    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalues = np.round(eigenvalues, 4)
    is_positive_definite = False
    if (eigenvalues > 0).all() and (matrix.T == matrix).all():
        is_positive_definite = True
    final_dict = {
        'is_positive_definite': is_positive_definite,
        'eigenvalues': sorted(list(eigenvalues))
    }
    return final_dict