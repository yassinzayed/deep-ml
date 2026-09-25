import numpy as np

def compute_tpr_fpr(y_true, y_pred):
    """
    Compute TPR and FPR from true and predicted binary labels.

    Args:
        y_true (array-like): Ground-truth labels (0 or 1).
        y_pred (array-like): Predicted labels (0 or 1).

    Returns:
        tuple: (tpr, fpr) as Python floats.
    """
    n = len(y_true)
    tp = sum(y_true[i] == 1 and y_pred[i] == 1 for i in range(n))
    fp = sum(y_true[i] == 0 and y_pred[i] == 1 for i in range(n))
    fn = sum(y_true[i] == 1 and y_pred[i] == 0 for i in range(n))
    tn = sum(y_true[i] == 0 and y_pred[i] == 0 for i in range(n))
    if (tp+fn) == 0:
        tpr = 0.0
    else:
        tpr = tp/(tp+fn)
    if (fp+tn) == 0:
        fpr = 0.0
    else:
        fpr = fp/(fp+tn)
    return (tpr, fpr)
