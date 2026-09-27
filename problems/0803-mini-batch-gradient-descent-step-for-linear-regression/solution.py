import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    #X_batch = np.column_stack((np.ones(X_batch.shape[0]), X_batch))
    #weights = np.insert(weights, 0, bias)
    err = np.matmul(X_batch, weights)+bias-y_batch
    grad = (2/len(batch_indices))*np.matmul(X_batch.T, err)
    weights -=  lr*grad
    bias -= lr*(2/len(batch_indices))*sum(err)
    final_weights = np.append(weights, bias)
    return final_weights
