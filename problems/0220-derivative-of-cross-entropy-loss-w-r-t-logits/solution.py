import numpy as np
def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""

	softmax = []
	sum_denom = np.sum(np.exp(np.array(logits)))
	for i in logits:
		softmax.append(np.exp(i)/sum_denom)
	one_hot = [0 for x in logits]
	one_hot[target] = 1
	grad_vec = np.array(softmax) - np.array(one_hot)
	return grad_vec