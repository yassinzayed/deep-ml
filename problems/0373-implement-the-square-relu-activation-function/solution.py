import numpy as np

def square_relu(x: np.ndarray) -> dict:
	"""
	Apply the Square ReLU activation function and compute its derivative.
	
	Args:
		x: Input numpy array of any shape
	
	Returns:
		Dictionary with 'output' and 'derivative' as numpy arrays
	"""
	scores = np.where(x > 0, np.round(x**2, 4), 0)
	derivatives = np.where(x > 0, np.round(2*x, 4), 0)
	return {
		'output': scores,
		'derivative': derivatives
	}