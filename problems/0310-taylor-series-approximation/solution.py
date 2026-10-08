import numpy as np
from math import factorial

def taylor_approximation(func_name: str, x: float, n_terms: int) -> float:
    """
    Compute Taylor series approximation for common functions.
    
    Args:
        func_name: Name of function ('exp', 'sin', 'cos')
        x: Point at which to evaluate
        n_terms: Number of terms in the series
    
    Returns:
        Taylor series approximation rounded to 6 decimal places
    """
    taylor_at_x = 0
    if func_name == "exp":
        for i in range(n_terms):
            taylor_at_x += (x**i)/(factorial(i))
    elif func_name == "sin":
        for i in range(n_terms):
            taylor_at_x += (((-1)**i)*(x**(2*i+1)))/factorial(2*i+1)
    elif func_name == "cos":
        for i in range(n_terms):
            taylor_at_x += (((-1)**i)*(x**(2*i)))/factorial(2*i)
    return taylor_at_x