import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	num = len(A)
	A = np.array(A)
	L = np.eye(num)
	U = np.zeros((num, num))
	for i in range(num):
		for j in range(num):
			sums = 0
			sums_2 = 0
			for k in range(i):
				sums += L[i][k]*U[k][j]
			U[i][j] = A[i][j] - sums
			for k in range(i):
				sums_2 += L[j][k]*U[k][i]
			if U[i][i] == 0:
				L[j][i] = 0
			else:
				L[j][i] = (1/U[i][i])*(A[j][i] - sums_2)
	return L, U