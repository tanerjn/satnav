"""SVD/QR least squares with optional Tikhonov (ridge) regularization."""
import numpy as np

def solve_least_squares(A, B, method="svd", ridge=0.0):
    A, B = np.asarray(A,float), np.asarray(B,float)
    if A.ndim!=2 or B.ndim not in (1,2) or A.shape[0]!=B.shape[0]:
        raise ValueError("Incompatible least-squares shapes")
    if not np.isfinite(ridge) or ridge<0: raise ValueError("ridge must be >= 0")
    if method=="svd":
        U,s,Vt=np.linalg.svd(A,full_matrices=False)
        if ridge:
            factors=s/(s*s+ridge)
        else:
            factors=np.divide(1.,s,out=np.zeros_like(s),where=s>np.finfo(float).eps*max(A.shape)*s[0])
        projected=U.T@B
        return Vt.T@(factors[:,None]*projected if B.ndim==2 else factors*projected)
    if method=="qr":
        if ridge:
            n=A.shape[1]
            A0=A
            A=np.vstack([A0,np.sqrt(ridge)*np.eye(n)])
            B=np.concatenate([B,np.zeros((n,)+B.shape[1:])])
        if A.shape[0]<A.shape[1]:raise ValueError("QR requires rows >= columns")
        Q,R=np.linalg.qr(A,mode="reduced")
        if np.linalg.matrix_rank(R)<R.shape[1]:raise ValueError("Rank deficient: use SVD or ridge")
        return np.linalg.solve(R,Q.T@B)
    raise ValueError("method must be 'svd' or 'qr'")
