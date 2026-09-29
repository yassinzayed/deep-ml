import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    # Your code here
    precisions = []
    recalls = []
    for i in y_scores:
        y_preds = [1 if k >= i else 0 for k in y_scores]
        tp = sum([y_true[i] == y_preds[i] and y_true[i] == 1 for i in range(len(y_true))])
        fp = sum([y_true[i] == 0 and y_preds[i] == 1 for i in range(len(y_true))])
        fn = sum([y_true[i] == 1 and y_preds[i] == 0 for i in range(len(y_true))])
        if (tp+fp) == 0:
            presicions.append(1.0)
        else:
            precisions.append(tp/(tp+fp))
        if (tp+fn) == 0:
            recalls.append(0.0)
        else:
            recalls.append(tp/(tp+fn))
    return precisions, recalls, sorted(y_scores, reverse=True)