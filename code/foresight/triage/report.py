"""
Foresight's triage, end to end: train on every ticket before 2025,
score 2025, and say how many tickets the router would send on.

    python -m foresight.triage.report           trained model only
    python -m foresight.triage.report --llm     also ask the language
                                                model about the unsure
                                                tickets (needs a key)
"""
from __future__ import annotations

import argparse

import pandas as pd

from foresight.triage.evaluate import (CAPACITY, keyword_rule,
                                       per_class, urgent_at_capacity)
from foresight.triage.features import (CATEGORIES, PRIORITIES, relabel,
                                       split, tickets)
from foresight.triage.model import TriageModel
from foresight.triage.router import THRESHOLD, route


def page(test: pd.DataFrame, pred: pd.DataFrame) -> str:
    lines = [f"Foresight triage: {len(test):,} tickets from 2025", ""]
    for label, classes in [("priority", PRIORITIES),
                           ("category", CATEGORIES)]:
        acc = (pred[label].to_numpy() == test[label].to_numpy()).mean()
        lines.append(f"{label}: accuracy {acc:.1%}")
        table = per_class(test[label], pred[label], classes)
        lines += table.to_string(
            float_format=lambda v: f"{v:.3f}").splitlines()
        lines.append("")
    lines.append(f"Urgent found in the first {CAPACITY} read each day")
    for name, score in [
            ("arrival", -test.opened_at.astype("int64")),
            ("urgent words", test.body.map(keyword_rule)),
            ("model", pred.p_urgent)]:
        lines.append(f"  {name:14}"
                     f"{urgent_at_capacity(test, score):.1%}")
    counts = pred.answered_by.value_counts()
    lines += ["", f"answered by (threshold {THRESHOLD})"]
    lines += [f"  {k:16}{v:6,}" for k, v in counts.items()]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", action="store_true",
                    help="send unsure tickets to the language model")
    args = ap.parse_args(argv)
    t = tickets()
    t["category"] = relabel(t)
    train, valid, test = split(t)
    seen = pd.concat([train, valid], ignore_index=True)
    model = TriageModel().fit(seen.body, seen.priority, seen.category)
    ask = None
    if args.llm:
        from foresight.triage.llm import classify
        ask = classify
    pred = route(test.body, model, ask=ask)
    print(page(test, pred))


if __name__ == "__main__":
    main()
