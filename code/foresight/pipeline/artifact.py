"""
A model artifact: one folder holding everything needed to use a
fitted model and to know what it is. Chapter 21 writes it.

    model.joblib     the fitted object: preprocessing, lasso and map
    model_card.md    Chapter 16's card, filled in for this fit
    manifest.json    what the model reads, with the type of each
                     column; the rows it learned from, as hashes; the
                     code that made it; the libraries it ran on; its
                     settings, scores and checks; and a hash of every
                     other file in the folder

load() refuses a folder whose files do not match their hashes, and a
model saved by another version of scikit-learn: an object unpickled
by a different version of the library that made it can load without
complaint and score differently.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from foresight.tracking import environment, sha256

MODEL, CARD, MANIFEST = "model.joblib", "model_card.md", "manifest.json"


class ArtifactError(Exception):
    """The artifact is not what its manifest says it is."""


def schema(rows: pd.DataFrame, columns) -> dict:
    """Each input column and its type, as the model was fitted on."""
    return {c: str(rows[c].dtype) for c in columns}


def save(folder: Path, model, manifest: dict,
         card: str | None = None) -> dict:
    """Write the model, its card and its manifest to a new folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)      # never overwrite
    joblib.dump(model, folder / MODEL)
    if card is not None:
        (folder / CARD).write_text(card)
    files = {p.name: sha256(p.read_bytes())
             for p in sorted(folder.iterdir())}
    manifest = {**manifest, "environment": environment(),
                "files": files}
    (folder / MANIFEST).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def verify(folder: Path) -> dict:
    """The manifest, once every file matches its hash and the
    scikit-learn that saved the model is the one installed."""
    folder = Path(folder)
    manifest = json.loads((folder / MANIFEST).read_text())
    for name, digest in manifest["files"].items():
        p = folder / name
        if not p.exists() or sha256(p.read_bytes()) != digest:
            raise ArtifactError(f"{name} does not match its manifest")
    saved = manifest["environment"]["scikit-learn"]
    here = environment()["scikit-learn"]
    if saved != here:
        raise ArtifactError(f"saved with scikit-learn {saved};"
                            f" this is {here}")
    return manifest


def load(folder: Path):
    """The fitted model and its manifest, verified."""
    manifest = verify(folder)
    return joblib.load(Path(folder) / MODEL), manifest


def check_inputs(rows: pd.DataFrame, manifest: dict) -> None:
    """Refuse rows that lack a column the model reads, or carry it as
    a different type from the one it learned from."""
    for c, kind in manifest["inputs"].items():
        if c not in rows.columns:
            raise ArtifactError(f"missing input column {c!r}")
        if str(rows[c].dtype) != kind:
            raise ArtifactError(f"{c}: {rows[c].dtype},"
                                f" expected {kind}")
