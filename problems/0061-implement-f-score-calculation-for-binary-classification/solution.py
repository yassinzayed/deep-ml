import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	tp = sum(y_true[i] == 1 and y_pred[i] == 1 for i in range(len(y_true)))
	tot_pos = sum(y_true == 1)
	tot_samp = sum(y_pred == 1)
	recall = tp/tot_pos
	percision = tp/tot_samp
	f_score = (1+beta**2)*((percision*recall)/(((beta**2) * percision)+recall))
	return round(f_score, 3)
