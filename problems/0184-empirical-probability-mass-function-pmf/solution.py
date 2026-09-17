import numpy as np
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    values, counts = np.unique(samples, return_counts=True)
    pmf_list = []
    for i in range(len(counts)):
        probability = counts[i]/len(samples)
        pmf_list.append((values[i], probability))
    return pmf_list