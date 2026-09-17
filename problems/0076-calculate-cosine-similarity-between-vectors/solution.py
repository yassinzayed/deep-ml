import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	if len(v1) == len(v2) and len(v1) != 0:
		dot = np.dot(v1, v2)
		mag1 = np.linalg.norm(v1)
		mag2 = np.linalg.norm(v2)
		return dot/(mag1*mag2)