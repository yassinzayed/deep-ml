def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	if not isinstance(y_true, list):
		y_true, y_pred = [y_true], [y_pred]
	huber_loss = sum([0.5*((y_true[i]-y_pred[i])**2) if abs(y_true[i]-y_pred[i])<=delta else delta*(abs(y_true[i]-y_pred[i])-0.5*delta) for i in range(len(y_true))])/len(y_true)
	return huber_loss