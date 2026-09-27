import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    weights = np.zeros(X.shape[1])
    bias = 0
    for i in range(max_iter):
        y_pred = np.matmul(X, weights)+bias
        err = y_pred - y
        grad_weights = ((1/len(X))*np.matmul(X.T, err))+alpha1*np.sign(weights)+2*alpha2*weights
        grad_bias = (1/len(X))*sum(err)
        weights -= learning_rate*grad_weights
        bias -= learning_rate*grad_bias
        if np.linalg.norm(grad_weights, ord=1) < tol:
            return weights, bias
    return weights, bias