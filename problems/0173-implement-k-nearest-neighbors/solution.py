import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    eucs = []
    for i in points:
        eucs.append(np.sqrt(sum(np.square(np.subtract(i, query_point)))))
    finals = []
    for i in range(k):
        finals.append(points[eucs.index(min(eucs))])
        eucs[eucs.index(min(eucs))] = max(eucs)+1
    return finals