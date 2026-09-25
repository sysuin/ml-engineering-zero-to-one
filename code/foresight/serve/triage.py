"""
Ticket triage in service. Chapter 22 writes it.

    python -m foresight.serve.triage                 train, register
    python -m foresight.serve.triage promote 1 "why" promote a version

Chapter 20's report fits its model in about a second and keeps it only
in memory. Serving needs what the renewal model has: an artifact with
a manifest, a registry, and a version a person has promoted. train()
fits the model on every ticket opened before 2025, as the report does,
and registers it staged.

answer() is what the API returns for one ticket: both labels, the
chance it is Urgent, and "unsure" when either label's confidence is
below Chapter 20's threshold. The language model is not called here.
Its answer takes seconds; an unsure ticket is answered now with the
model's labels, marked, and can be sent for a second opinion outside
the request. keywords() answers when no model is loaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from foresight.config import ROOT
from foresight.pipeline.registry import Registry
from foresight.tracking import rows_hash
from foresight.triage.evaluate import URGENT_WORDS
from foresight.triage.features import TEST, relabel, split, tickets
from foresight.triage.model import TriageModel
from foresight.triage.router import THRESHOLD, unsure


def train(root: Path = ROOT) -> int:
    """Fit Chapter 20's model on the tickets before 2025; register it
    as the next version of `triage`, staged. Returns the version."""
    t = tickets()
    t["category"] = relabel(t)
    fit, valid, _ = split(t)
    seen = pd.concat([fit, valid], ignore_index=True)
    model = TriageModel().fit(seen.body, seen.priority, seen.category)
    rows = seen[["ticket_id", "body", "priority", "category"]]
    manifest = {
        "name": "triage", "run": f"triage, tickets before {TEST}",
        "inputs": {"body": "str"},
        "data": {"as_of": TEST, "rows": len(seen),
                 "rows_sha256": rows_hash(rows)},
        "metrics": {"tickets": len(seen)},
        "settings": {"threshold": THRESHOLD}}
    registry = Registry("triage", Path(root) / "artifacts")
    return registry.register(model, manifest)


def answer(model, body: str) -> dict:
    """One ticket's labels, from the model."""
    p = model.predict([body]).iloc[0]
    doubt = bool(unsure(p.to_frame().T).iloc[0])
    return {"priority": p.priority, "category": p.category,
            "p_urgent": round(float(p.p_urgent), 4),
            "confidence": {"priority": round(float(p.priority_conf), 4),
                           "category": round(float(p.category_conf),
                                             4)},
            "answered_by": "model (unsure)" if doubt else "model"}


def keywords(body: str) -> dict:
    """Chapter 20's urgent-words rule, when there is no model: Urgent
    or Normal, and no category."""
    urgent = bool(URGENT_WORDS.search(body))
    return {"priority": "Urgent" if urgent else "Normal",
            "category": None, "p_urgent": None, "confidence": None,
            "answered_by": "keywords"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", nargs="?", default="train",
                    choices=["train", "promote"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    if a.action == "promote":
        version, *why = a.args
        Registry("triage", a.root / "artifacts").promote(
            int(version), " ".join(why) or "promoted")
        print(f"triage version {version} in production")
        return
    version = train(a.root)
    print(f"triage version {version}, staged")


if __name__ == "__main__":
    main()
