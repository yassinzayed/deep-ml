from typing import Callable
import numpy as np

def newtons_method_optimization(
	gradient_func: Callable[[list[float]], list[float]],
	hessian_func: Callable[[list[float]], list[list[float]]],
	x0: list[float],
	tol: float = 1e-6,
	max_iter: int = 100
) -> list[float]:
	"""
	Find the minimum of a function using Newton's method.
	
	Args:
		gradient_func: Function that returns gradient vector at a point
		hessian_func: Function that returns Hessian matrix at a point
		x0: Initial guess (list of coordinates)
		tol: Convergence tolerance for gradient norm
		max_iter: Maximum number of iterations
		
	Returns:
		The point that minimizes the function
	"""
	grad_norm = gradient_func(x0)
	while np.linalg.norm(np.array(grad_norm)) > tol:
		grad_hess = np.linalg.inv(np.array(hessian_func(x0)))
		newton_step = np.matmul(grad_hess, np.array(grad_norm))
		x0 -= newton_step
		grad_norm = gradient_func(x0)
	return x0.tolist()