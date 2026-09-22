import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    for i in range(len(X_test)):
        for j in range(len(X_test[i])):
            if stds[j] == 0:
                X_test[i][j] = (X_test[i][j] - means[j])
            else:
                X_test[i][j] = (X_test[i][j] - means[j])/stds[j]
    return X_test
