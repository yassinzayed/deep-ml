import numpy as np

def shuffle_data(X, y, seed=None):
	rng = np.random.seed(seed)
	indices = np.random.permutation(range(0, len(X))).tolist()
	X_new = []
	y_new = []
	for i in indices:
		X_new.append(X[i].tolist())
		y_new.append(y[i])
	return (X_new, y_new)