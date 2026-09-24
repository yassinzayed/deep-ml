import numpy as np

def vector_sum(a: np.ndarray):
    """Sum elements of 1-D array a without np.sum / loops."""
    # Your code here
    ones = np.ones_like(a)
    summed = np.dot(a, ones)
    return summed
