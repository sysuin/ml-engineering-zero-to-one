"""
How a triage model is judged: per class, at the desk's capacity, and
against how far two labellers agree with each other.

    per_class(y, pred)          precision, recall and count per class
    urgent_at_capacity(t, s)    share of Urgent tickets among the first
                                CAPACITY read each day, ranked by s
    keyword_rule(body)          the desk's rule of thumb, as a score
    kappa(a, b)                 agreement beyond chance, Cohen's kappa
    desk_pairs(t)               each North ticket beside the South
                                ticket with the most nearly same words
    paired(a, b)                the difference in accuracy between two
                                models on the same tickets, with a
                                95% bootstrap interval
    eval_set(test)              the fixed sample every approach is
                                compared on, trained or prompted
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from foresight.config import SEED, rng
from foresight.triage.features import tokens

CAPACITY = 3            # tickets a senior agent reads first, each day
EVAL_SIZE = 1000        # tickets in the comparison set, from 2025

URGENT_WORDS = re.compile(r"urgent|asap|today|same day|immediately",
                          re.I)


def per_class(y, pred, classes) -> pd.DataFrame:
    """Precision and recall for each class, and how many there were."""
    y, pred = np.asarray(y), np.asarray(pred)
    rows = []
    for c in classes:
        hit = ((pred == c) & (y == c)).sum()
        rows.append({"class": c, "n": int((y == c).sum()),
                     "precision": hit / max((pred == c).sum(), 1),
                     "recall": hit / max((y == c).sum(), 1)})
    return pd.DataFrame(rows).set_index("class")


def urgent_at_capacity(t: pd.DataFrame, score, label: str = "priority",
                       k: int = CAPACITY) -> float:
    """Rank each day's tickets by score, read the first k, and return
    the share of that period's Urgent tickets among those read. Ties
    go to the ticket that arrived first."""
    s = pd.Series(np.asarray(score, dtype=float), index=t.index)
    order = t.assign(s=s).sort_values(["day", "s", "opened_at"],
                                      ascending=[True, False, True])
    read = order.groupby("day").head(k)
    urgent = t[label] == "Urgent"
    return float(urgent[read.index].sum() / urgent.sum())


def keyword_rule(body: str) -> float:
    """1 if the ticket uses a word the desk treats as urgent, else 0."""
    return float(bool(URGENT_WORDS.search(body)))


def kappa(a, b) -> float:
    """Cohen's kappa: observed agreement, corrected for the agreement
    two labellers with these label frequencies would reach by chance."""
    a, b = pd.Series(np.asarray(a)), pd.Series(np.asarray(b))
    observed = float((a.values == b.values).mean())
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    chance = float(pa.mul(pb, fill_value=0).sum())
    return (observed - chance) / (1 - chance)


def desk_pairs(t: pd.DataFrame, label: str = "category",
               closeness: float = 0.9) -> pd.DataFrame:
    """Pairs of near-identical tickets, one from each desk, with both
    desks' labels: for each North ticket, the South ticket nearest to
    it by TF-IDF cosine, kept if the cosine is at least closeness."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    north = t[t.desk == "North desk"].reset_index(drop=True)
    south = t[t.desk == "South desk"].reset_index(drop=True)
    vec = TfidfVectorizer(tokenizer=tokens, lowercase=False,
                          token_pattern=None).fit(t.body)
    index = NearestNeighbors(n_neighbors=1, metric="cosine")
    index.fit(vec.transform(south.body))
    distance, nearest = index.kneighbors(vec.transform(north.body))
    close = 1 - distance[:, 0] >= closeness
    return pd.DataFrame({
        "north": north[label][close].to_numpy(),
        "south": south[label].iloc[nearest[close, 0]].to_numpy(),
        "north_body": north.body[close].to_numpy(),
        "south_body": south.body.iloc[nearest[close, 0]].to_numpy()})


def paired(hit_a, hit_b, reps: int = 2000, seed: int = SEED):
    """Accuracy of a minus accuracy of b, scored on the same tickets,
    and a 95% interval from resampling the tickets."""
    d = np.asarray(hit_a, float) - np.asarray(hit_b, float)
    g = rng(seed)
    draws = [d[g.integers(0, len(d), len(d))].mean()
             for _ in range(reps)]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def eval_set(test: pd.DataFrame, n: int = EVAL_SIZE,
             seed: int = SEED) -> pd.DataFrame:
    """A fixed random sample of the test year, oldest first: the one set
    on which every approach, trained or prompted, is compared."""
    return (test.sample(n=n, random_state=seed)
            .sort_values(["opened_at", "ticket_id"])
            .reset_index(drop=True))
