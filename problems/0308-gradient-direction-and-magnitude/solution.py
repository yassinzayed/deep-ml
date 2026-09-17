import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	mag = np.linalg.norm(np.array(gradient))
	if np.sum(gradient) != 0:
		direction = np.array(gradient)/mag
		descent = -direction
	else:
		direction = [0.0, 0.0]
		descent = [0.0, 0.0]
	grad_dict = {
		"magnitude": mag,
		'direction': direction,
		'descent_direction': descent
	}
	return grad_dict