import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    # Think of something more elegant than the copy method
    # I honestly didn't want to go through renaming everything
    mse_train = []
    mse_val = []
    x = X_train.copy()
    y = y_train.copy()
    xval = X_val.copy()
    yval = y_val.copy()
    for n in train_sizes:
        X_train = x[:n]
        y_train = y[:n]
        X_val = xval
        y_val = yval
        if degree == 0:
            X_train = np.ones((len(X_train), 1))

        elif degree == 1:
            X_train = np.column_stack((np.ones((len(X_train), 1)), X_train))
        else:
            for i in range(degree-1):
                X_train = np.column_stack((X_train, X_train[:, 1]**(i+2)))
        weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train)), X_train.T), y_train)
        if degree == 0:
            X_val = np.ones((len(X_val), 1))
        else:
            X_val = np.column_stack((np.ones((len(X_val), 1)), X_val))
        preds = np.matmul(X_train, weights)
        vals = np.matmul(X_val, weights)
        mse_train.append(np.mean(np.square(preds-y_train)))
        mse_val.append(np.mean(np.square(vals-y_val)))
    if mse_train[-1] > bias_threshold:
        diagnosis = 'high_bias'
    elif (mse_val[-1]-mse_train[-1]) > variance_threshold:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'
    return {
        'train_errors': mse_train,
        'val_errors': mse_val,
        'diagnosis': diagnosis
    }