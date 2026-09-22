import numpy as np

def gaussian_random_projection(X: np.ndarray, n_components: int, seed: int = 0) -> np.ndarray:
    """
    Project X into a lower-dimensional space using a Gaussian random projection.

    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Target dimensionality
        seed: Random seed for reproducibility

    Returns:
        Projected matrix of shape (n_samples, n_components)
    """
    rng = np.random.RandomState(seed)
    R = rng.standard_normal((X.shape[1], n_components))
    R = (1/np.sqrt(n_components))*R
    matrix = np.matmul(X, R)
    return matrix
