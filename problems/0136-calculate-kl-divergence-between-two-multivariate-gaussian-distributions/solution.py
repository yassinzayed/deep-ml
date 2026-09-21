import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    # REDO BRO THIS SUCKS DO IT COMPLETELY OVER
    #det1 = Cov_q[0][0]*Cov_q[1][1] - Cov_q[1][0]*Cov_q[0][1]
    #det2 = Cov_p[0][0]*Cov_p[1][1] - Cov_p[1][0]*Cov_p[0][1]
    det1 = np.linalg.det(Cov_q)
    det2 = np.linalg.det(Cov_p)
    term_1 = np.log(det1/det2)
    term_2 = np.matmul(np.matmul(((mu_p-mu_q).T), np.linalg.inv(Cov_q)), (mu_p-mu_q))
    term_3 = np.trace(np.matmul(np.linalg.inv(Cov_q), Cov_p))
    KL_divergence = 0.5*(term_1+term_2+term_3-len(Cov_q))
    return KL_divergence