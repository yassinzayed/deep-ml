import numpy as np

def gaussian_mle(data: np.ndarray) -> tuple:
    """
    Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    
    Args:
        data: 1D numpy array of observations
        
    Returns:
        Tuple of (mean_mle, variance_mle)
    """
    # Your code here
    means = np.mean(data)
    mle = 0
    for i in data:
        mle += (i-means)**2
    return (means, mle/len(data))