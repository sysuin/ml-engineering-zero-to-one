"""
The hybrid: the trained model answers every ticket, and the ones it is
unsure of go to the language model.

    unsure(pred, threshold)     the tickets to send on
    route(bodies, model, ask)   the model's answers, with the unsure
                                ones replaced by ask()'s, and a column
                                saying which answered
"""
from __future__ import annotations

import pandas as pd

THRESHOLD = 0.7
"""Send a ticket on when the model's confidence in either label is below
this. Chosen on the 2024 validation tickets in Chapter 20, §20.10."""


def unsure(pred: pd.DataFrame, threshold: float = THRESHOLD):
    """True where the model's top probability for the priority or for
    the category is below the threshold."""
    return ((pred.priority_conf < threshold)
            | (pred.category_conf < threshold))


def route(bodies, model, ask=None,
          threshold: float = THRESHOLD) -> pd.DataFrame:
    """The trained model's answer for every ticket. Where it is unsure
    and ask is given (a function from bodies to a frame with category
    and priority), ask's answer replaces it; where ask gave no answer,
    or there is no ask, the model's stands and says so."""
    bodies = list(bodies)
    pred = model.predict(bodies)
    pred["answered_by"] = "model"
    send = unsure(pred, threshold)
    pred.loc[send, "answered_by"] = "model (unsure)"
    if ask is not None and send.any():
        where = send[send].index
        reply = ask([bodies[i] for i in where])
        reply.index = where
        ok = reply.category.notna() & reply.priority.notna()
        for col in ("category", "priority"):
            pred.loc[where[ok], col] = reply.loc[ok, col]
        pred.loc[where[ok], "answered_by"] = "llm"
    return pred
