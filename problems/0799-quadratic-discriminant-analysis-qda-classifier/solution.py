import numpy as np

def qda_predict(X_train, y_train, X_test):
    """
    Train a QDA classifier on (X_train, y_train) and predict labels for X_test.
    Returns a list of predicted class labels (Python ints).
    """
    X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)
    indices = []
    for i in set(y_train):
        indices.append(np.where(y_train == i)[0])
    #indices0 = np.where(y_train == 0)[0]
    #indices1 = np.where(y_train == 1)[0]
    mu = []
    for i in indices:
        mu.append(np.mean(X_train[i], axis=0))
    #mu0 = np.mean(X_train[indices0], axis=0)
    #mu1 = np.mean(X_train[indices1], axis=0)
    cov = []
    for i in range(len(mu)):
        cov.append((1/len(indices[i]))*(X_train[indices[i]]-mu[i]).T@(X_train[indices[i]]-mu[i]))
    #cov1 = (1/len(indices0))*(X_train[indices0]-mu0).T@(X_train[indices0]-mu0)
    #cov2 = (1/len(indices1))*(X_train[indices1]-mu1).T@(X_train[indices1]-mu1)
    #cov = [cov1, cov2]
    #mu = [mu1, mu2]
    preds = []
    for i in X_test:
        preds_temp = []
        for k in range(len(mu)):
            preds_temp.append((i-mu[k]).T@np.linalg.inv(cov[k])@(i-mu[k]))
        #i0 = (i-mu0).T@np.linalg.inv(cov1)@(i-mu0)
        #i1 = (i-mu1).T@np.linalg.inv(cov2)@(i-mu1)
        preds.append(preds_temp.index(min(preds_temp)))
        #if i0 < i1:
        #    preds.append(0)
        #else:
        #    preds.append(1)
    return preds