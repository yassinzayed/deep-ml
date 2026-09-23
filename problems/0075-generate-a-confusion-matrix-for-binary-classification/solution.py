from collections import Counter
import numpy as np
# wtf does counter do???
def confusion_matrix(data):
	data = np.array(data)
	y_true = data[:, 0]
	y_pred = data[:, 1]
	obs = len(y_true)
	tp = sum(y_true[i] == 1 and y_pred[i] == 1 for i in range(obs))
	fn = sum(y_true[i] == 1 and y_pred[i] == 0 for i in range(obs))
	fp = sum(y_true[i] == 0 and y_pred[i] == 1 for i in range(obs))
	tn = sum(y_true[i] == 0 and y_pred[i] == 0 for i in range(obs))
	confusion_matrix = [[tp, fn], [fp, tn]]
	return confusion_matrix
