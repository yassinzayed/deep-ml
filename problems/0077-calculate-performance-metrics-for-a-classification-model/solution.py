
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	# Implement your code here
	obs = len(actual)
	tp = sum(actual[i] == 1 and predicted[i] == 1 for i in range(obs))
	fn = sum(actual[i] == 1 and predicted[i] == 0 for i in range(obs))
	fp = sum(actual[i] == 0 and predicted[i] == 1 for i in range(obs))
	tn = sum(actual[i] == 0 and predicted[i] == 0 for i in range(obs))
	confusion_matrix = [[tp, fn], [fp, tn]]
	accuracy = (tp+tn)/(tp+tn+fp+fn)
	precision = tp/(tp+fp)
	negativePredictive = tn/(tn+fn)
	recall = tp/(tp+fn)
	specificity = tn/(tn+fp)
	f1 = 2*((precision*recall)/(precision+recall))
	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
