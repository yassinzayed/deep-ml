import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	dims = len(A)
	if x_ini is None:
		x_ini = np.zeros((dims))
	#else:
	#	x_ini = np.array(x_ini)
	for i in range(n):
		for k in range(len(A)):
			coeff = 1/A[k][k]
			sums1 = 0
			for j in range(k):
				sums1 += A[k][j]*x_ini[j]
			sums2 = 0
			for j in range(k+1, len(A)):
				sums2 += A[k][j]*x_ini[j]
			parenthesis = b[k] - sums1 - sums2
			x_ini[k] = coeff*parenthesis
	return x_ini
