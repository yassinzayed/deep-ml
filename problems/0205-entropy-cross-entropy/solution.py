import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	entropy = 0.0
	cross = 0.0
	for i in P:
		if i != 0:
			entropy += i*np.log(i)
	for i in range(len(P)):
		cross += P[i]*np.log(Q[i])
	return (-entropy, -cross)