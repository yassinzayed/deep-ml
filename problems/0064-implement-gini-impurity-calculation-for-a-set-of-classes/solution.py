
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	probs = 0
	for i in set(y):
		count = y.count(i)
		prob = count/len(y)
		probs += prob**2
	val = 1-probs
	return round(val,3)