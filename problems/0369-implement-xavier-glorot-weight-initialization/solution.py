import numpy as np

def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> dict:
    """
    Perform Xavier/Glorot weight initialization.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    # Your code here
    np.random.seed(seed)
    sigma = 2/(fan_in+fan_out)
    if mode == 'normal':
        param = np.sqrt(2/(fan_in+fan_out))
        weights = np.random.normal(scale=sigma, size=(fan_in, fan_out))
    elif mode == 'uniform':
        param = np.sqrt(6/(fan_in+fan_out))
        weights = np.random.uniform(low=-param, high=param, size=(fan_in, fan_out))
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")
    return {
        'weights': weights,
        'shape': weights.shape,
        'param': param
    }