from math import exp
def thanksgiving_dish_predictor(preference_scores: list[float]) -> list[float]:
	"""
	Predict the probability of choosing each Thanksgiving dish using softmax.
	
	Args:
		preference_scores: List of preference scores for each dish
		(e.g., [turkey_score, stuffing_score, cranberry_score, pie_score])
		
	Returns:
		List of probabilities for each dish
	"""
	sums = sum([exp(i) for i in preference_scores])
	scores = [exp(i)/sums for i in preference_scores]
	return scores