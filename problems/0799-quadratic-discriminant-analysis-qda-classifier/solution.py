import numpy as np

def qda_predict(X_train, y_train, X_test):
    """
    Train a QDA classifier on (X_train, y_train) and predict labels for X_test.
    Returns a list of predicted class labels (Python ints).
    """
    # Yabn el la3iba
    X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)
    indices = []
    for i in set(y_train):
        indices.append(np.where(y_train == i)[0])
    mu = []
    for i in indices:
        mu.append(np.mean(X_train[i], axis=0))
    cov = []
    for i in range(len(mu)):
        cov.append((1/len(indices[i]))*(X_train[indices[i]]-mu[i]).T@(X_train[indices[i]]-mu[i]))
    preds = []
    for i in X_test:
        preds_temp = []
        for k in range(len(mu)):
            preds_temp.append((i-mu[k]).T@np.linalg.inv(cov[k])@(i-mu[k]))
        preds.append(preds_temp.index(min(preds_temp)))
    return preds