def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	post = []
	for i in range(len(priors)):
		nom = priors[i]*likelihoods[i]
		sum_denom = 0
		for i in range(len(priors)):
			sum_denom += priors[i]*likelihoods[i]
		post.append(nom/sum_denom) 
	return post