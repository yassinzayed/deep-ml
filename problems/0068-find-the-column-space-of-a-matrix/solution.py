import numpy as np

def matrix_image(A):
	U, s, Vh = np.linalg.svd(A)
	rank = np.sum(s > 1e-10)
	col_space = A[:, :rank]
	return col_space
