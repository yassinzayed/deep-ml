import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
	"""
	Compute the mutual information between two random variables.
	
	Args:
		joint_prob: 2D joint probability distribution P(X,Y)
	
	Returns:
		Mutual information I(X;Y)
	"""
	p_x = [sum(i) for i in joint_prob]
	p_y = [sum(i) for i in zip(*joint_prob)]
	mutual_info = 0
	for i in joint_prob:
		for j in i:
			if j/(p_x[0]*p_y[0]) != 0:
				mutual_info += j*np.log(j/(p_x[0]*p_y[0]))
	return mutual_info