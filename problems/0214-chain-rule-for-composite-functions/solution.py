import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	def eval_der(func, x, der=False):
		if not der:
			if func == 'square':
				return x**2
			elif func == 'sin':
				return np.sin(x)
			elif func == 'exp':
				return np.exp(x)
			elif func == 'log':
				return np.log(x)
		else:
			if func == 'square':
				return 2*x
			elif func == 'sin':
				return np.cos(x)
			elif func == 'exp':
				return np.exp(x)
			elif func == 'log':
				return 1/x
	if len(functions) == 1:
		answer = eval_der(functions[0], x, der=True)
	elif len(functions) == 2:
		answer = eval_der(functions[0] ,eval_der(functions[1], x), der=True)*eval_der(functions[1], x, der=True)
	else:
		temp_answer = eval_der(functions[1] ,eval_der(functions[2], x), der=True)*eval_der(functions[2], x, der=True)
		answer = eval_der(functions[0] , eval_der(functions[1], eval_der(functions[2], x)), der=True)*temp_answer
	return answer