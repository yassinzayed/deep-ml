import numpy as np

def lda_predict(X_train, y_train, X_test):
    """
    Fit a binary LDA classifier on (X_train, y_train) and predict labels for X_test.

    Args:
        X_train: array-like of shape (n_samples, n_features)
        y_train: array-like of shape (n_samples,) with values in {0, 1}
        X_test:  array-like of shape (n_test, n_features)

    Returns:
        List[int] of predicted labels (0 or 1) for each row in X_test.
    """
    X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)
    indices0 = np.where(y_train == 0)[0]
    indices1 = np.where(y_train == 1)[0]
    mu0 = np.mean(X_train[indices0], axis=0)
    mu1 = np.mean(X_train[indices1], axis=0)
    a = (X_train[indices0]-mu0).T@(X_train[indices0]-mu0)
    b = (X_train[indices1]-mu1).T@(X_train[indices1]-mu1)
    Sw = a+b
    w = np.linalg.inv(Sw)@(mu1-mu0)
    c = np.dot(w, ((mu0+mu1)/2))
    preds = [1 if np.dot(w, i)>c else 0 for i in X_test]
    return preds
