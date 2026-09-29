import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    X_mean = X-np.mean(X, axis=0)
    C = (1/(len(X)-1))*np.matmul(X_mean.T, X_mean)
    eig_vals = np.linalg.eigvals(C)
    var = [float(i/sum(eig_vals)) for i in eig_vals]
    return sorted(var, reverse=True)