import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    # Your code here
    y_scores_sorted = sorted(set(y_scores), reverse=True)
    p = y_true.count(1)
    n = y_true.count(0)
    tpr = []
    fpr = []
    for i in y_scores_sorted:
        y_preds = [1 if k>=i else 0 for k in y_scores]
        tp = sum([y_preds[i] == 1 and y_true[i] == 1 for i in range(len(y_true))])
        fp = sum([y_preds[i] == 1 and y_true[i] == 0 for i in range(len(y_true))])
        tpr.append(tp/p)
        fpr.append(fp/n)
    tpr.insert(0, 0.0)
    fpr.insert(0, 0.0)
    return (fpr, tpr)