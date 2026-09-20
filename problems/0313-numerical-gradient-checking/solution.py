import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    f_dash = np.array([])
    for i in range(len(x)):
        l_plus = x.copy()
        l_minus = x.copy()
        l_plus[i] += epsilon
        l_minus[i] -= epsilon
        f_dash = np.append(f_dash, (f(l_plus) - f(l_minus)) / (2 * epsilon))
    rel_error = np.linalg.norm(f_dash - analytical_grad) / (np.linalg.norm(f_dash) + np.linalg.norm(analytical_grad))
    return (f_dash, rel_error)