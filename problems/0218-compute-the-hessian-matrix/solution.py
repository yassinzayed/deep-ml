from typing import Callable
import numpy as np

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""
	# Your code here
	Hessian = [[0 for x in point] for x in point]
	x = np.array(point)
	for i in range(len(Hessian)):
		for j in range(len(Hessian)):
			if i == j:
				h_vec = np.array([0.0 for k in x])
				h_vec[i] = h
				Hessian[i][j] = (f(x + h_vec)-2*f(x)+f(x-h_vec))/(h**2)
			else:
				hvec1 = np.array([0.0 for k in x])
				hvec2 = np.array([0.0 for k in x])
				hvec1[i] = h
				hvec2[j] = h
				nom = f(x+hvec1+hvec2)-f(x-hvec1+hvec2)-f(x+hvec1-hvec2)+f(x-hvec1-hvec2)
				denom = 4*(h**2)
				Hessian[j][i] = nom/denom
	return Hessian