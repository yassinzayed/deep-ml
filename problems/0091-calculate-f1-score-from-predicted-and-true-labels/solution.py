def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	tp = sum(y_true[i] == 1 and y_pred[i] == 1for i in range(len(y_true)))
	fp = sum(y_true[i] == 1 and y_pred[i] == 0 for i in range(len(y_true)))
	fn = sum(y_true[i] == 0 and y_pred[i] == 1 for i in range(len(y_true)))
	if (tp+fp) == 0 or (tp+fn) == 0:
		return 0
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	f1 = 2*((precision*recall)/(precision+recall))
	return round(f1,3)