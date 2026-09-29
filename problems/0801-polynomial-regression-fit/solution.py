import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    X = np.ones((len(x), 1))
    for i in range(degree):
        X = np.column_stack((X, np.power(x, i+1)))
    c = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
    return c.tolist()
