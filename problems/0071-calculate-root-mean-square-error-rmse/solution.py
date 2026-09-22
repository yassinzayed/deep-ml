
import numpy as np

def rmse(y_true, y_pred):
	coeff = 1/y_true.size
	rmse_res = np.sqrt(coeff*np.sum(np.square(y_true-y_pred)))
	return round(rmse_res,3)
