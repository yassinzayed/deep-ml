import numpy as np
def rotation_layer(X, angle):
    X = np.array(X)
    R = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    y = np.matmul(X, np.array(R).T)
    return y.tolist()