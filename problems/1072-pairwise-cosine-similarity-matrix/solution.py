import numpy as np

def pairwise_cosine_similarity(X):
    S = np.zeros((np.shape(X)[0], np.shape(X)[0]))
    for i in range(len(S)):
        for j in range(len(S)):
            prod = np.dot(X[i], X[j])
            abs1 = np.linalg.norm(X[i])
            abs2 = np.linalg.norm(X[j])
            S[i][j] = 0
            if (abs1 and abs2) != 0:
                S[i][j] = prod/(abs1*abs2)
    return S