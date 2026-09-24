
import numpy as np

def dice_score(y_true, y_pred):
	if np.all(y_true == 0) and np.all(y_pred == 0):
		return 0
	tp = sum(y_true[i] == y_pred[i] and y_true[i] == 1 for i in range(len(y_true)))
	fp = sum(y_true[i] == 0 and y_pred[i] == 1 for i in range(len(y_true)))
	fn = sum(y_true[i] == 1 and y_pred[i] == 0 for i in range(len(y_true)))
	res = 2*(tp/(2*tp+fp+fn))
	return round(res, 3)
