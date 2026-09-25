"""
Chapter 18's segments: features measured point in time, k-means by
hand matching scikit-learn, and names that follow the profile rather
than k-means' numbering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from foresight.segments import (Segmenter, account_features, assign,
                                inertia, kmeans_numpy, name_segments,
                                profile, update)


@pytest.fixture
def book():
    orders = pd.DataFrame({
        "account_id": [1, 1, 1, 2, 2],
        "day": pd.to_datetime(["2023-06-01", "2024-02-01", "2024-04-01",
                               "2023-05-01", "2022-01-01"]),
        "value": [100.0, 300.0, 999.0, 50.0, 70.0]})
    accounts = pd.DataFrame({
        "since": pd.to_datetime(["2020-04-01", "2023-01-01"]),
        "crm_segment": ["Small business", "Mid-market"],
        "is_key_account": [0, 0]}, index=pd.Index([1, 2],
                                                   name="account_id"))
    return orders, accounts


def test_an_order_on_the_day_is_not_counted(book):
    f = account_features("2024-04-01", *book)
    assert f.loc[1, "orders_365"] == 2
    assert f.loc[1, "spend_365"] == 400.0
    assert f.loc[1, "recency_days"] == 60


def test_accounts_without_an_order_in_the_year_are_left_out(book):
    f = account_features("2024-06-01", *book)
    assert list(f.index) == [1]


def test_trend_compares_the_two_half_years(book):
    f = account_features("2024-04-01", *book)
    # 300 in the last 182 days, 100 in the 183 before, $100 added to each
    assert f.loc[1, "trend"] == pytest.approx(np.log(400 / 200))
    assert f.loc[1, "tenure_years"] == pytest.approx(4.0, abs=0.01)


def blobs(seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [6, 0], [0, 6]])
    return np.vstack([c + rng.normal(size=(100, 2)) for c in centres])


def test_one_step_moves_each_centre_to_its_mean():
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    labels = assign(X, np.array([[1.0], [9.0]]))
    assert labels.tolist() == [0, 0, 1, 1]
    assert update(X, labels, 2).ravel().tolist() == [1.0, 11.0]
    assert inertia(X, labels, update(X, labels, 2)) == 4.0


def test_hand_built_kmeans_matches_scikit_learn():
    X = blobs()
    start = X[[0, 100, 200]] + 0.5
    centres, labels, history = kmeans_numpy(X, start.copy())
    km = KMeans(3, init=start, n_init=1, tol=0).fit(X)
    assert (km.labels_ == labels).all()
    assert np.allclose(km.cluster_centers_, centres)
    assert len(history) >= 2


def test_names_follow_the_profile_not_the_numbering():
    prof = pd.DataFrame({"recency_days": [10, 80, 12, 9],
                         "tenure_years": [9.0, 5.0, 0.5, 2.5]},
                        index=[7, 3, 5, 1])
    assert name_segments(prof) == {3: "Drifting", 5: "Newcomers",
                                   7: "Long-standing core",
                                   1: "Established core"}
    with pytest.raises(ValueError):
        name_segments(prof.iloc[:3])


def test_the_segmenter_is_deterministic_and_profiles_add_up():
    rng = np.random.default_rng(1)
    n = 400
    f = pd.DataFrame({"orders_365": rng.integers(1, 60, n),
                      "basket": rng.uniform(200, 1500, n),
                      "recency_days": rng.integers(1, 300, n),
                      "tenure_years": rng.uniform(0, 12, n),
                      "trend": rng.normal(0, 1.5, n)})
    f["spend_365"] = f.orders_365 * f.basket
    a = Segmenter(4).fit(f).predict(f)
    b = Segmenter(4).fit(f).predict(f)
    assert (a == b).all()
    prof = profile(f, a)
    assert prof.accounts.sum() == n
    assert prof.share.sum() == pytest.approx(1.0)
    assert prof.spend_share.sum() == pytest.approx(1.0)
