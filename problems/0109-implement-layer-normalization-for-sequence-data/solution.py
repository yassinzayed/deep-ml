import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	mean = np.mean(X, axis=-1, keepdims=True)
	sigma = np.var(X, axis=-1, keepdims=True)
	x_hat = (X-mean)/np.sqrt(sigma+epsilon)
	y = gamma*x_hat+beta
	return y