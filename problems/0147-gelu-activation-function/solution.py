import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
	tanh_val = np.sqrt(2/np.pi)*(x+0.044715*(x**3))
	scores = 0.5*x*(1+np.tanh(tanh_val))
	return scores