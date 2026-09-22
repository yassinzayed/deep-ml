import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	# calculate initial residual vector
	# YAAA FAGER
	# Also a few point awalan recheck this question it was a good one
	# Thaneyan dude when multiplying a matrix by a scalar don't use matmul
	if x0 is None:
		x0 = np.zeros_like(b)
	x = x0
	r0 = b - np.matmul(A, x0)
	p0 = r0
	r = [r0, r0]
	p = [p0, p0]
	for i in range(n):
		alpha_k = (np.matmul(r[1].T, r[1]))/ np.matmul(np.matmul(p[1].T, A),p[1])
		x = x + alpha_k*p[1]
		r[1] = r[0] - np.matmul(alpha_k*A, p[1])
		if np.linalg.norm(r[1]) < tol:
			break
		beta = np.matmul(r[1].T, r[1])/np.matmul(r[0].T, r[0])
		p[1] = r[1] + beta*p[0]
		p[0] = p[1]
		r[0] = r[1]
	return x
