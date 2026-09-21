import numpy as np

def cramers_rule(A, b):
    A = np.array(A)
    b = np.array(b)
    det = np.linalg.det(A)
    if det == 0:
        return -1
    else:
        x = []
        for i in range(len(b)):
            A_copy = A.copy()
            A_copy[:, i] = b
            det_cram = np.linalg.det(A_copy)
            x.append(det_cram/det)
        return x