import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
	eigen_vals = np.linalg.eigvals(hessian)
	if (eigen_vals > 0).all():
		return -1
	elif (eigen_vals < 0).all():
		return 1
	elif (np.isclose(eigen_vals, 0, atol=tol)).any():
		return None
	else:
		return 0