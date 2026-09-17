import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    mean = np.mean(data)
    median = np.median(data)
    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]
    variance = np.var(data)
    standard_deviation = np.std(data)
    quantiles = np.quantile(data, [0.25, 0.5, 0.75])
    stat_dict = {
        'mean': mean,
        'median': median,
        'mode': mode,
        'variance': variance,
        'standard_deviation': standard_deviation,
        '25th_percentile': quantiles[0],
        '50th_percentile': quantiles[1],
        '75th_percentile': quantiles[2],
        'interquartile_range': quantiles[2] - quantiles[0]
    }
    return stat_dict