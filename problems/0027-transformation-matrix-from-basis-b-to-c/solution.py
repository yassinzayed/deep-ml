import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	C_inv = np.linalg.inv(C)
	P = np.matmul(C_inv, B)
	return P