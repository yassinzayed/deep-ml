import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	X = np.column_stack((np.ones(X.shape[0]), X))
	weights = np.zeros(X.shape[1])
	losses = []
	for i in range(iterations):
		sigmoid = 1/(1+np.exp(-np.dot(X, weights)))
		loss = -sum(y*np.log(sigmoid)+(1-y)*np.log(1-sigmoid))
		losses.append(loss)
		grad = np.matmul(X.T, sigmoid-y)
		weights -= learning_rate*grad
	return (np.round(weights, 4).tolist(), np.round(losses, 4).tolist())