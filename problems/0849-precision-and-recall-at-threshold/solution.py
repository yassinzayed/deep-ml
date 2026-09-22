import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    yscore = []
    for i in y_scores:
        if i >= threshold:
            yscore.append(1)
        else:
            yscore.append(0)
    tp = sum(y_true[i] == 1 and yscore[i] == 1 for i in range(len(y_true)))
    tot_pos = sum(np.array(y_true) == 1)
    tot_samp = sum(np.array(yscore) == 1)
    recall = 0
    percision = 0
    if tot_pos != 0:
        recall = tp/tot_pos
    if tot_samp != 0:
        percision = tp/tot_samp
    return [percision, recall]
