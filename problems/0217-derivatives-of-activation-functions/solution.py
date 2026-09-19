import numpy as np

def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	sigmoid = 1/(1+np.exp(-x))
	sigmoid_der = sigmoid*(1-sigmoid)
	tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
	tanh_der = 1 - tanh**2
	relu = 0
	if x > 0:
		relu = 1
	act_dict = {
		'sigmoid': sigmoid_der,
		'tanh': tanh_der,
		'relu': relu
	}
	return act_dict