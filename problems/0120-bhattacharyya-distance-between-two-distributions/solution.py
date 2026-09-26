import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if (len(p) != len(q)) or (len(p) == 0 and len(q) == 0):
        return 0.0
    bc = sum(np.sqrt(np.array(p)*np.array(q)))
    bd = -np.log(bc)
    return bd