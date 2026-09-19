import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	if len(point) == 2:
		x, y = point
		z = 0
	else:
		x, y, z = point
	func_dictionary = {
		'poly2d': (2*x*y + y**2, 2*x*y + x**2),
		'exp_sum': (np.exp(x+y), np.exp(x+y)),
		'product_sin': (np.sin(y), x*np.cos(y)),
		'poly3d': (2*x*y, x**2 + z**2, 2*z*y),
		'squared_error': (2*(x-y), -2*(x-y))
	}
	return func_dictionary[func_name]