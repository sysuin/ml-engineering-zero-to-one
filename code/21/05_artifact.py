# One model artifact, packed by hand: the fitted object, its card, and
# a manifest of everything needed to know what it is. Then loaded back,
# and refused when a file or the library no longer matches.
import json
import tempfile
from pathlib import Path

import pandas as pd

from foresight.card import render
from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, backtest, known_by
from foresight.pipeline import artifact, evaluation, settings
from foresight.pipeline.model import build, maker
from foresight.tracking import code_record, data_record
from foresight.train import COLUMNS

cfg = settings.load("renewal")
table = pd.read_parquet(TABLE)
rows = known_by(table[table.end_date >= HISTORY_FROM], "2025-01-01")
model = build().fit(rows, rows.not_renewed)
scored = backtest(rows, "2024-07-01", "2024-12-31", maker(build()))
scores = evaluation.scores(scored)
manifest = {
    "name": "renewal", "version": "0.6", "run": "by hand",
    "inputs": artifact.schema(rows, COLUMNS),
    "features": list(model.estimator_.named_steps["prepare"]
                     .get_feature_names_out()),
    "data": {**data_record(rows, table), "as_of": "2025-01-01"},
    "code": code_record(model), "settings": cfg, "metrics": scores,
    "checks": {"leakage": "passed"}}
card = render(evaluation.card(build(), table, scored, scores, cfg))

with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp) / "renewal-0.6"
    saved = artifact.save(folder, model, manifest, card)
    for p in sorted(folder.iterdir()):
        print(f"{p.name:<15}{p.stat().st_size / 1024:>6.1f} KB")
    d = saved["data"]
    print(f"\ninputs     {len(saved['inputs'])} columns, e.g."
          f" days_since_order: {saved['inputs']['days_since_order']}")
    print(f"features   {len(saved['features'])} numbers, from"
          f" {saved['features'][0]} to {saved['features'][-1]}")
    match = "matches" if d["matches_manifest"] else "does not match"
    print(f"data       {d['table']}, {d['content_sha256'][:8]},"
          f" {match} its manifest")
    print(f"           {d['rows']:,} rows, {d['rows_sha256'][:8]},"
          f" outcomes known on {d['as_of']}")
    print(f"code       {', '.join(saved['code']['files'])}")
    lib = saved["environment"]["scikit-learn"]
    print(f"library    scikit-learn {lib}")
    print(f"metrics    {saved['metrics']['model hits']} leavers in"
          f" {saved['metrics']['calls']} calls,"
          f" AUC {saved['metrics']['model auc']:.3f}")

    sizes = [(p.name, p.stat().st_size / 1024)
             for p in sorted(folder.iterdir())]
    with open("code/21/05_artifact.json", "w") as f:
        json.dump({**saved, "sizes": sizes}, f)

    loaded, _ = artifact.load(folder)
    same = (loaded.predict_proba(rows) == model.predict_proba(rows))
    word = "the same" if same.all() else "different"
    print(f"\nloaded again: {word} chances for all {len(rows):,} rows")
    with open(folder / "model_card.md", "a") as f:
        f.write("\nCalibrated monthly.\n")        # a quiet edit
    try:
        artifact.load(folder)
    except artifact.ArtifactError as e:
        print(f"a line added to the card: refused\n  {e}")
    listed = json.loads((folder / "manifest.json").read_text())
    listed["environment"]["scikit-learn"] = "1.5.2"
    listed["files"]["model_card.md"] = artifact.sha256(
        (folder / "model_card.md").read_bytes())
    (folder / "manifest.json").write_text(json.dumps(listed))
    try:
        artifact.load(folder)
    except artifact.ArtifactError as e:
        print(f"saved by another scikit-learn: refused\n  {e}")
