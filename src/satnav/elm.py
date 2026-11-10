"""Basic ELM regression, fixed random hidden layer and SVD least squares.

This prototype expects already scaled input/output arrays. For orbital use,
fit scaling on training data only and predict state increments or local states.
"""
import numpy as np

class ELM:
    def __init__(self, hidden=64, seed=42):
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.hidden = hidden
        self.seed = seed

    def _hidden(self, X):
        return np.tanh(X @ self.W + self.bias)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("Expected aligned 2D samples")
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(size=(X.shape[1], self.hidden))/np.sqrt(X.shape[1])
        self.bias = rng.uniform(-1, 1, size=self.hidden)
        H = self._hidden(X)
        self.beta, *_ = np.linalg.lstsq(H, Y, rcond=None)
        return self

    def predict(self, X):
        return self._hidden(np.asarray(X, dtype=float)) @ self.beta
