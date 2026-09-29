import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    mean_preds = np.mean(predictions, axis=0)
    squared_bias = np.mean(np.square(y_true-mean_preds))
    variance = np.mean(np.var(predictions, axis=0))
    mse = squared_bias+variance
    return {
        'bias_squared': squared_bias,
        'variance': variance,
        'mse': mse
    }
