import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
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
    tpr, fpr = np.array(tpr), np.array(fpr)
    auc = sum(((tpr[1:]+tpr[:len(tpr)-1])/2)*(fpr[1:]-fpr[:len(fpr)-1]))
    return auc