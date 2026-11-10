"""Single-hidden-layer ELM with fixed random features and least-squares output."""
import numpy as np
from .least_squares import solve_least_squares

class ELM:
    def __init__(self, hidden=64, seed=42, solver="svd"):
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.hidden, self.seed, self.solver = hidden, seed, solver

    def _hidden(self, X):
        return np.tanh(X @ self.W + self.bias)

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("Expected aligned 2D samples")
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(size=(X.shape[1], self.hidden))/np.sqrt(X.shape[1])
        self.bias = rng.uniform(-1, 1, size=self.hidden)
        self.beta = solve_least_squares(self._hidden(X), Y, method=self.solver)
        return self

    def predict(self, X):
        return self._hidden(np.asarray(X, float)) @ self.beta
