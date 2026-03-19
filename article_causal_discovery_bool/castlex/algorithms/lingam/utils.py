import numpy as np


def to_causal_matrix(am: np.ndarray, tol=1e-6) -> np.ndarray:
    # am: adjacency matrix
    # cm: causal matrix
    n = am.shape[0]
    cm = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(n):
            if np.isnan(am[i,j]) or am[i, j] < -tol or am[i, j] > tol:
                cm[i,j] = 1
    return cm
