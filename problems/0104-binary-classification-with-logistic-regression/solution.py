import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	preds = []
	for i in range(len(X)):
		z = np.dot(X[i], weights)+bias
		sigmoid = 1/(1+np.exp(-z))
		if np.all(sigmoid >= 0.5):
			preds.append(1)
		else:
			preds.append(0)
	return preds