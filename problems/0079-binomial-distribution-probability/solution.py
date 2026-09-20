import math

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.
    
    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial
    
    Returns:
        Probability of k successes
    """
    binomial_coeff = math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    sucess_term = p**k
    failure_term = (1-p)**(n-k)
    result = binomial_coeff*sucess_term*failure_term
    return result