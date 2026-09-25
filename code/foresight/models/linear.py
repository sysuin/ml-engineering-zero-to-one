"""
A linear model fitted by gradient descent, in NumPy, so that every step
can be read. Chapter 6 writes it; Chapter 7 reuses descend() with the
gradient of a different loss.

    model = LinearGD(lr=0.1, steps=10_000).fit(X, y)
    model.coef_, model.intercept_       in the columns' own units
"""
from __future__ import annotations

import numpy as np


def squared_error(X, y, w, b) -> float:
    """The mean squared error of the predictions X @ w + b."""
    return float(np.mean((X @ w + b - y) ** 2))


def squared_error_gradient(X, y, w, b):
    """How the mean squared error changes with each weight and b."""
    error = X @ w + b - y
    return 2 * X.T @ error / len(y), 2 * error.mean()


def descend(X, y, lr: float, steps: int,
            gradient=squared_error_gradient, path: bool = False):
    """Start at zero and take `steps` steps downhill, each lr times
    the gradient. path=True also returns (w, b) before each step."""
    w, b = np.zeros(X.shape[1]), 0.0
    visited = []
    for _ in range(steps):
        if path:
            visited.append((w.copy(), b))
        dw, db = gradient(X, y, w, b)
        w, b = w - lr * dw, b - lr * db
    return (w, b, visited) if path else (w, b)


class Standardiser:
    """Give every column mean 0 and standard deviation 1, using the
    training rows' means and deviations for every later row too."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)  # constant column
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


class LinearGD:
    """Standardise, descend, and report weights in original units."""

    def __init__(self, lr: float = 0.1, steps: int = 10_000):
        self.lr, self.steps = lr, steps

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.scaler_ = Standardiser().fit(X)
        Z = self.scaler_.transform(X)
        w, b = descend(Z, y, self.lr, self.steps)
        self.coef_ = w / self.scaler_.scale_         # per original unit
        self.intercept_ = b - self.coef_ @ self.scaler_.mean_
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
