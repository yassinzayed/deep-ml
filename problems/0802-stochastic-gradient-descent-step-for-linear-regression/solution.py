import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n_samples = len(y)
    for i in range(n_iter):
        grad = 2*(np.matmul(X[int(i % n_samples)].T, w)-y[int(i % n_samples)]) * X[int(i % n_samples)]
        weights -= learning_rate*grad
    return weights
