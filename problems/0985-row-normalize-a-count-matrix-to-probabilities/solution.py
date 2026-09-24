import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    probs_final = []
    for i in counts:
        probs = []
        for j in i:
            if sum(i) == 0:
                probs.append(0)
            else:
                probs.append(j/sum(i))
        probs_final.append(probs)
    return probs_final