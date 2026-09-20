import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    # Fucking seed bro
    # Apparently cannot use dict because of how the random seed works
    # Also btw usually u can usually just use numpy instead of a loop
    np.random.seed(seed)
    if distribution == 'uniform':
        distr = np.random.uniform(0, 1, size=(runs, n))
        mu = 0.5
        sigma = np.sqrt(1/12)
    elif distribution == 'exponential':
        distr = np.random.exponential(1.0, size=(runs, n))
        mu = 1.0
        sigma = 1.0
    elif distribution == 'bernoulli':
        distr = (np.random.rand(runs, n) < 0.3).astype(float)
        mu = 0.3
        sigma = np.sqrt(0.3*0.7)
    else:
        raise ValueError
    means = np.mean(distr, axis=1)
    z_scores = (means - mu)/(sigma/np.sqrt(n))
    final_dict = {
        'mean': np.mean(z_scores),
        'std': np.std(z_scores)
    }
    return final_dict