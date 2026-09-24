
import numpy as np

def jaccard_index(y_true, y_pred):
	# intersecion is the same as the tp
	intersection = sum(y_true[i] == y_pred[i] and y_true[i] == 1 for i in range(len(y_true)))
	union = sum(y_true)+sum(y_pred)-intersection
	result = intersection/union
	return round(result, 3)
