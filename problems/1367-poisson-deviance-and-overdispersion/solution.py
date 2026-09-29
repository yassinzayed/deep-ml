import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    d = 0
    for i in range(len(y)):
        if y[i] == 0:
            d += mu[i]
        else:
            d += (y[i]*np.log(y[i]/mu[i]) - (y[i]-mu[i]))
    return 2*d


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    ratio = (1/(len(y)-n_params))*sum(np.square(y-mu)/mu)
    return ratio
