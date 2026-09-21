import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	A = np.column_stack((A, b))
	for i in range(len(A)-1):
		if A[i][i] == 0 and (i != (len(A)+1)):
			for k in range(len(A)):
				if A[k][i] != 0:
					A[[i, k]] = A[[k, i]]
					break
		if A[i][i] != 0:
			if A[i][i] != 1:
				A[i] = A[i]/(A[i][i])
			for k in range(len(A)-i-1):
				A[k+1+i] = A[k+1+i] - A[i]*(A[k+1+i][i]/A[i][i])
	A[-1] = A[-1]/A[-1][-2]
	for i in range(len(A)-1, 0, -1):
		for k in range(i):
			A[k] = A[k] - A[i] * A[k][i]
	b = A[:, -1]
	return b
