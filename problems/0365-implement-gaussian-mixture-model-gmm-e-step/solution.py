import numpy as np

def gmm_e_step(X: np.ndarray, means: np.ndarray, variances: np.ndarray, 
               mixing_coeffs: np.ndarray) -> np.ndarray:
    """
    Compute the E-step of Gaussian Mixture Model.
    
    Args:
        X: Data points of shape (n_samples,)
        means: Component means of shape (n_components,)
        variances: Component variances of shape (n_components,)
        mixing_coeffs: Mixing coefficients of shape (n_components,)
    
    Returns:
        Responsibility matrix of shape (n_samples, n_components)
    """
    responsibility = []
    for i in X:
        exp = np.exp(-0.5*np.square((i-means)/np.sqrt(variances)))
        gaussian = (1/((np.sqrt(variances))*np.sqrt(2*np.pi)))*exp
        responsibility.append((mixing_coeffs*gaussian)/(sum(mixing_coeffs*gaussian)))
    return responsibility