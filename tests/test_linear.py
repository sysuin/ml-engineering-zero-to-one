"""
The gradient-descent line from Chapter 6, checked against answers that
can be worked out without it: a line with no noise, the least-squares
solution from linear algebra, and the gradient measured numerically.
"""
from __future__ import annotations

import numpy as np
import pytest

from foresight.config import rng
from foresight.models.linear import (LinearGD, Standardiser, descend,
                                     squared_error,
                                     squared_error_gradient)


@pytest.fixture
def data():
    g = rng()
    X = g.normal(size=(200, 3)) * [1.0, 50.0, 0.01] + [0.0, 400.0, 2.0]
    y = X @ np.array([2.0, -0.3, 40.0]) + 7.0 + g.normal(size=200)
    return X, y


def test_descent_finds_a_line_with_no_noise():
    x = np.linspace(-1, 1, 21)[:, None]
    w, b = descend(x, 3 * x[:, 0] + 2, lr=0.3, steps=2000)
    assert w[0] == pytest.approx(3.0) and b == pytest.approx(2.0)


def test_the_gradient_matches_a_numerical_one(data):
    X, y = data
    w, b, h = np.array([0.5, 0.01, 1.0]), 3.0, 1e-6
    dw, db = squared_error_gradient(X, y, w, b)
    for i in range(3):
        step = np.eye(3)[i] * h
        slope = (squared_error(X, y, w + step, b)
                 - squared_error(X, y, w - step, b)) / (2 * h)
        assert dw[i] == pytest.approx(slope, rel=1e-5)
    slope = (squared_error(X, y, w, b + h)
             - squared_error(X, y, w, b - h)) / (2 * h)
    assert db == pytest.approx(slope, rel=1e-5)


def test_the_path_starts_at_zero_and_has_one_entry_a_step():
    x = np.arange(5.0)[:, None]
    _, _, path = descend(x, x[:, 0], lr=0.01, steps=7, path=True)
    assert len(path) == 7
    assert path[0][1] == 0.0 and not path[0][0].any()


def test_standardiser_uses_training_statistics_and_survives_constants():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    s = Standardiser().fit(X)
    Z = s.transform(X)
    assert Z[:, 0].mean() == pytest.approx(0.0)
    assert Z[:, 0].std() == pytest.approx(1.0)
    assert not Z[:, 1].any()                  # constant: zero, not NaN
    assert s.transform(np.array([[7.0, 5.0]]))[0, 0] == pytest.approx(
        (7 - 3) / X[:, 0].std())


def test_the_fitted_line_matches_least_squares(data):
    X, y = data
    model = LinearGD(lr=0.1, steps=5000).fit(X, y)
    A = np.column_stack([np.ones(len(X)), X])
    exact = np.linalg.lstsq(A, y, rcond=None)[0]
    assert model.intercept_ == pytest.approx(exact[0], abs=1e-6)
    np.testing.assert_allclose(model.coef_, exact[1:], rtol=1e-6)
    np.testing.assert_allclose(model.predict(X), A @ exact, rtol=1e-6)


def test_too_large_a_learning_rate_diverges(data):
    X, y = data
    with np.errstate(over="ignore", invalid="ignore"):
        w, b = descend(X, y, lr=1.0, steps=200)
    assert not np.isfinite(b) or abs(b) > 1e6
