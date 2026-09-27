import numpy as np
def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	'''for i in range(len(parameters)):
		if all(apply_to_all == True for i in apply_to_all):
			parameters[i] = parameters[i] - lr*gradients[i] - lr*weight_decay*parameters[i]
		else:
			if apply_to_all[i] == True:
				pass
	'''
	parameters = np.array(parameters)
	gradients = np.array(gradients)
	parameters = parameters - lr*gradients-lr*weight_decay*parameters
	return parameters.tolist()