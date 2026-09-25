"""
The trained triage model: TF-IDF over words and word pairs, and one
logistic regression for each label. Chapter 20's baseline, and the
model Foresight ships.

    m = TriageModel().fit(bodies, priority, category)
    m.proba(bodies, "priority")    a frame, one column per class
    m.predict(bodies)              each ticket's labels, P(Urgent)
                                   and the confidence of each label
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from foresight.triage.features import tokens


def vectoriser(min_df: int = 2) -> TfidfVectorizer:
    """Words and pairs of adjacent words, weighted by TF-IDF."""
    return TfidfVectorizer(tokenizer=tokens, lowercase=False,
                           token_pattern=None, ngram_range=(1, 2),
                           min_df=min_df, sublinear_tf=True)


class TriageModel:
    """One vocabulary, two logistic regressions: priority, category."""

    def __init__(self, C: float = 1.0, min_df: int = 2):
        self.C, self.min_df = C, min_df

    def fit(self, bodies, priority, category):
        self.vec_ = vectoriser(self.min_df)
        X = self.vec_.fit_transform(bodies)
        self.models_ = {
            name: LogisticRegression(C=self.C, max_iter=2000).fit(X, y)
            for name, y in [("priority", priority),
                            ("category", category)]}
        return self

    def proba(self, bodies, label: str) -> pd.DataFrame:
        m = self.models_[label]
        P = m.predict_proba(self.vec_.transform(bodies))
        return pd.DataFrame(P, columns=m.classes_)

    def predict(self, bodies) -> pd.DataFrame:
        """Each ticket's priority and category, the model's confidence
        in each (its largest probability), and its chance of being
        Urgent."""
        out = {}
        for label in ("priority", "category"):
            P = self.proba(bodies, label)
            out[label] = P.columns[np.argmax(P.to_numpy(), axis=1)]
            out[f"{label}_conf"] = P.max(axis=1).to_numpy()
            if label == "priority":
                out["p_urgent"] = P["Urgent"].to_numpy()
        return pd.DataFrame(out)
