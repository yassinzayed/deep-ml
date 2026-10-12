import numpy as np

def pos_encoding(position: int, d_model: int):
	if position == 0 or d_model <= 0:
		return -1
	pos_encoding = np.zeros((position, d_model))
	for i in range(position):
		for j in range(d_model):
			if j%2 == 0:
				pos_encoding[i][j] = np.sin(i/np.power(10000, (2*(j//2))/d_model))
			else:
				pos_encoding[i][j] = np.cos(i/np.power(10000, (2*(j//2))/d_model))
	pos_encoding = np.float16(pos_encoding)
	return pos_encoding