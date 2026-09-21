import numpy as np
def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Don't turn x into a numpy array
	# Not sure why but it has something to do with numpy rounding
	# I think if you do turn it turn it into a float so when you add h
	# it actually gets addded
	points = len(x)
	func = len(f(x))
	Jac = np.zeros((func, points))
	for i in range(len(f(x))):
		for k in range(points):
			x_temp = x.copy()
			x_temp[k] = x[k]+h
			nom = f(x_temp)[i]-f(x)[i]
			f_dash = nom/h
			Jac[i][k] = f_dash
	return Jac