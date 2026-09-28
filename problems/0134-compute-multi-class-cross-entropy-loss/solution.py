import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    pred_logs = np.log(predicted_probs+epsilon)
    val = true_labels*pred_logs
    val = -sum(sum(val))/np.sum(true_labels == 1)
    return float(val)