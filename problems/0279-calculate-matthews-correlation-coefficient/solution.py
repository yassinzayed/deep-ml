import numpy as np

def matthews_correlation_coefficient(y_true: list, y_pred: list) -> float:
    """
    Calculate the Matthews Correlation Coefficient for binary classification.
    
    Args:
        y_true: List of actual binary labels (0 or 1)
        y_pred: List of predicted binary labels (0 or 1)
    
    Returns:
        MCC value rounded to 4 decimal places
    """
    tp = sum([y_true[i] == 1 and y_pred[i] == 1 for i in range(len(y_true))])
    tn = sum([y_true[i] == 0 and y_pred[i] == 0 for i in range(len(y_true))])
    fp = sum([y_true[i] == 0 and y_pred[i] == 1 for i in range(len(y_true))])
    fn = sum([y_true[i] == 1 and y_pred[i] == 0 for i in range(len(y_true))])
    nom = tp*tn-fp*fn
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if denom == 0:
        return 0.0
    mcc = nom/denom
    return round(float(mcc), 4)