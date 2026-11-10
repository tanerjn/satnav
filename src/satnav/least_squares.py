"""Stable least-squares solvers for ELM output weights."""
import numpy as np

def solve_least_squares(A, B, method="svd"):
    A, B = np.asarray(A, float), np.asarray(B, float)
    if A.ndim != 2 or B.ndim not in (1, 2) or A.shape[0] != B.shape[0]:
        raise ValueError("Incompatible least-squares inputs")
    if method == "svd":
        return np.linalg.lstsq(A, B, rcond=None)[0]
    if method == "qr":
        if A.shape[0] < A.shape[1]:
            raise ValueError("QR solver requires rows >= columns; use SVD")
        Q, R = np.linalg.qr(A, mode="reduced")
        if np.linalg.matrix_rank(R) < R.shape[1]:
            raise ValueError("Rank-deficient design matrix; use SVD")
        return np.linalg.solve(R, Q.T @ B)
    raise ValueError("method must be 'svd' or 'qr'")
