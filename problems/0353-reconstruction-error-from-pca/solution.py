import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    X_mean = X-np.mean(X, axis=0)
    C = (1/len(X))*np.matmul(X_mean.T, X_mean)
    vals, vecs = np.linalg.eig(C)
    indices = np.argsort(np.abs(vals))[::-1][:n_components]
    eigen_vecs = vecs[:, indices]
    z = np.matmul(X_mean, eigen_vecs)
    proj = np.matmul(z, eigen_vecs.T)
    reconstructed = proj+np.mean(X, axis=0)
    mse = (1/(X.shape[0]*X.shape[1]))*(sum(sum(np.square(X-reconstructed))))
    return float(mse)