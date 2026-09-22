import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	if np.all(delta_W == 0):
		return 0
	U, s, Vh = np.linalg.svd(delta_W)
	total = np.sum(np.square(s))
	energy = 0
	n = 0
	for i in s:
		if energy < energy_threshold:
			energy += (i**2)/total
			n += 1
		else:
			return n
	return n