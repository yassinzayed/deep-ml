import numpy as np
def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	x = np.exp(x)
	x /= np.sum(x)
	dims = len(x)
	Jac = np.zeros((dims, dims))
	for i in range(len(Jac)):
		for j in range(len(Jac[i])):
			if i == j:
				Jac[i][j] = x[i]*(1-x[i])
			else:
				Jac[i][j] = -x[i]*x[j]
	return Jac