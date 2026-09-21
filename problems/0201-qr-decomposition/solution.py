import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	if not all(A):
		return [], []
	R = [[0 for i in range(len(A[0]))] for i in range(len(A[0]))]
	e = []
	for i in range(len(A[0])):
		col_i = [A[k][i] for k in range(len(A))]
		u = col_i
		for z in range(i):
			first_prod = sum(x*y for x, y in zip(col_i, e[z]))
			u = [x-y for x, y in zip(u, [(first_prod*l) for l in e[z]])]
		u_norm = (sum([l**2 for l in u]))**(0.5)
		e_i = [(l/u_norm) for l in u]
		e.append(e_i)
		for j in range(i+1):
			R[j][i] = sum(x*y for x, y in zip(col_i, e[j]))
	Q = [list(row) for row in zip(*e)]
	return Q, R