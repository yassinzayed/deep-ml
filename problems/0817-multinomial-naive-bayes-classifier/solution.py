import numpy as np

def multinomial_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Implements Multinomial Naive Bayes classifier.

    Args:
        X_train: Training count features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test count features (shape: N_test x D)
        alpha: Laplace smoothing parameter

    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    indices = []
    for i in set(y_train):
        indices.append(np.where(y_train == i)[0])
    counts = []
    for i in indices:
        counts.append(sum(X_train[i]))
    counts = np.array(counts)
    row_sums = counts.sum(axis=1, keepdims=True)
    counts = (counts + alpha)/(row_sums+alpha*X_train.shape[1])
    priors = np.array([])
    for i in indices:
        priors = np.append(priors, len(i)/len(y_train))
    scores = np.dot(X_test, np.log(counts).T)+np.log(priors)
    preds = np.array([list(set(y_train))[i] for i in np.argmax(scores, axis=1)])
    return preds
