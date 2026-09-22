import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	mse = np.sum(np.square(y_true - np.dot(X, w)))
	reg = alpha*np.sum(np.square(w))
	ridge = (1/len(X))*mse + reg
	return ridge
