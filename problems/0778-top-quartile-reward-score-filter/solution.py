import numpy as np

def top_quartile_mask(scores: list) -> list:
    """
    Return a boolean list marking scores in the top quartile (>= 75th percentile).
    """
    if scores == []:
        return []
    top_quan = np.quantile(scores, 0.75)
    return [i >= top_quan for i in scores]