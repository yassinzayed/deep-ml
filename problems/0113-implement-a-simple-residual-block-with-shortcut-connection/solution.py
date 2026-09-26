import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	prod1 = w1 @ x
	relu1 = [max(0, z) for z in prod1]
	prod2 = w2 @ relu1
	final = x + prod2
	result = [max(0, z) for z in final]
	return result