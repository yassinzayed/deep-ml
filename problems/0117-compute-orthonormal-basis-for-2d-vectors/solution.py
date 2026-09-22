import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    vectors = np.array(vectors)
    if np.all(vectors == 0):
        return []
    u = []
    u1 = vectors[0]/np.linalg.norm(vectors[0])
    u.append(u1)
    for i in range(len(vectors)-1):
        current_v = vectors[i+1]
        sums = 0
        for k in range(i+1):
            sums += np.dot(current_v, u[k])*u[k]
        wk = current_v - sums
        if np.linalg.norm(wk) > tol:
            u_new = wk/np.linalg.norm(wk)
            u.append(u_new)
    return u