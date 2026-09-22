import numpy as np

def low_rank_approximation(delta_W: np.ndarray, r: int) -> list:
	"""
	Compute the best rank-r approximation of delta_W via truncated SVD.

	Args:
		delta_W: matrix of shape (m, n)
		r: target rank (1 <= r <= min(m, n))

	Returns:
		The rank-r approximation as a nested Python list of shape (m, n).
	"""
	U, S, Vh = np.linalg.svd(delta_W)
	S_full = np.diag(S)
	Ur = U[:, :r]
	Vr = (Vh.T)[:, :r]
	Sr = S_full[:r, :r]
	Ar = np.matmul(np.matmul(Ur, Sr), Vr.T)
	return Ar.tolist()