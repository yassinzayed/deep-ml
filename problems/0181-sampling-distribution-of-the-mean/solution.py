import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
    """
    Compute the mean of sample means to demonstrate the sampling distribution.

    Args:
        num_samples: Number of independent samples to draw
        sample_size: Size of each sample
        distribution: 'uniform' (0,1) or 'exponential' (scale=1)

    Returns:
        Mean of the sample means (float)
    """
    # Your code here
    if distribution == "uniform":
        distr = np.random.uniform(0, 1, (num_samples, sample_size))
    else:
        distr = np.random.exponential(1, (num_samples, sample_size))
    return float(np.mean(distr))