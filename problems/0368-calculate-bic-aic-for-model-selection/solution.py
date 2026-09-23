import numpy as np

def calculate_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> tuple:
    """
    Calculate AIC and BIC for model selection.
    
    Args:
        y_true: True target values
        y_pred: Predicted values from the model
        k: Number of parameters in the model
    
    Returns:
        Tuple of (AIC, BIC)
    """
    resid = sum(np.square(y_true-y_pred))
    n = len(y_true)
    AIC = n*np.log(resid/n)+2*k
    BIC = n*np.log(resid/n)+k*np.log(n)
    return (AIC, BIC)