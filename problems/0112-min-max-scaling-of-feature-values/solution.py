import numpy as np
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    x = np.array(x)
    maximum = np.max(x)
    minimum = np.min(x)
    x = (x-minimum)/(maximum-minimum)
    return x.tolist()