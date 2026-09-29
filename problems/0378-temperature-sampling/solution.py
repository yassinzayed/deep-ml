import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
	"""
	Compute temperature-scaled softmax probabilities from logits.
	
	Args:
		logits: 1D numpy array of raw model output scores
		temperature: float controlling distribution sharpness
	
	Returns:
		List of probabilities after temperature scaling
	"""
	# Your code here
	if temperature == 0:
		exps = np.zeros_like(logits)
		exps[np.argmax(logits)] = 1.0
		return exps
	temped = (logits/temperature)-max(logits)
	exp = np.exp(temped)
	return exp/sum(exp)