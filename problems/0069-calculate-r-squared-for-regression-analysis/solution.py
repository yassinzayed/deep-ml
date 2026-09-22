
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	# np.square squares all elements of a matrix!!
	ssr = np.sum(np.square(y_true - y_pred))
	sst = np.sum(np.square(y_true - np.mean(y_true)))
	r_squared = 1-(ssr/sst)
	return r_squared
