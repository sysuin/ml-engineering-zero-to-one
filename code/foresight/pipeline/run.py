"""
Foresight's training job as one command. Chapter 21 writes it, from
Chapter 13's train.py.

    make train MODEL=renewal
    python -m foresight.pipeline.run renewal
    python -m foresight.pipeline.run renewal --set data.as_of=2025-02-01

One run, in order, stopping at the first step that fails:

1. builds the training table through features.labelled() and writes
   it with Chapter 4's checks and manifest;
2. checks Chapter 5's expectations;
3. keeps the outcomes known on the day, backtests the model on the
   latest cohorts it can see, and runs Chapter 13's leakage checks
   against the cohort whose list is made that day, built the way the
   list will be built (features.at_mark());
4. fits the model on every outcome it may use;
5. measures the backtest (evaluation.scores());
6. records the run in the hand-built log and in MLflow (Chapter 15),
   with the rows it chose and how;
7. fills in the model card (Chapter 16);
8. saves the artifact and registers it, staged.

Promotion to production is a separate command, made by a person
(registry.py). Every setting comes from configs/<model>.toml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from foresight.card import render
from foresight.checks.leakage import run_checks
from foresight.config import ROOT
from foresight.data.build_table import write
from foresight.data.expectations import check_table
from foresight.evaluate import HISTORY_FROM, backtest, known_by
from foresight.features.sources import load_sources
from foresight.pipeline import evaluation, features, settings
from foresight.pipeline.artifact import schema
from foresight.pipeline.model import build, maker
from foresight.pipeline.registry import Registry
from foresight.tracking import track
from foresight.train import COLUMNS

# The code a model depends on beyond its own classes: the columns'
# definitions, Platt's map, and the job itself.
PACKAGE = Path(__file__).parents[1]
CODE = [PACKAGE / f for f in (
    "data/build_table.py", "decide.py", "pipeline/features.py",
    "pipeline/evaluation.py", "pipeline/run.py")]


def run(name: str = "renewal", overrides=(), root: Path = ROOT,
        say=print) -> dict:
    """Every step, in order. Returns what each step produced."""
    cfg = settings.load(name, overrides)
    d, m, paths = cfg["data"], cfg["model"], cfg["paths"]
    table_path = Path(root) / paths["table"]

    table = features.labelled(d["first"], d["last"])
    manifest = write(table, d["first"], d["last"], path=table_path)
    say(f"1 table     {manifest['rows']:,} rows,"
        f" {manifest['content_sha256'][:8]}; Chapter 4's checks pass")
    check_table(table)
    say("2 expect    Chapter 5's expectations hold")

    history = table[table.end_date >= HISTORY_FROM]
    rows = known_by(history, d["as_of"])
    mark = features.mark_on(d["as_of"])
    recent = features.at_mark(mark)
    model = build(m["strength"], m["calibration_months"],
                  m["no_order_days"])
    cohorts = cfg["evaluation"]["cohorts"]
    ends = sorted(rows.end_date.unique())[-cohorts:]
    first = str(ends[0].replace(day=1).date())    # months, as splits
    last = str(ends[-1].date())
    scored = backtest(rows, first, last, maker(model))
    report = run_checks(rows, recent, COLUMNS, load_sources(),
                        scored=scored)
    say(f"3 leakage   {len(COLUMNS)} columns, no failure;"
        f" {len(report.reviews())} for a person to review")

    fitted = model.fit(rows, rows.not_renewed)
    say(f"4 fit       {len(rows):,} outcomes known on {d['as_of']};"
        f" next list\n            {len(recent)} contracts marked"
        f" {mark:%Y-%m-%d}")

    scores = evaluation.scores(scored)
    say(f"5 backtest  {scores['cohorts']} cohorts ending {first} to"
        f" {last}:\n            {scores['model hits']} leavers in"
        f" {scores['calls']} calls (rule {scores['rule hits']}),"
        f" AUC {scores['model auc']:.3f} ({scores['rule auc']:.3f})")

    selection = {"history_from": HISTORY_FROM, "as_of": d["as_of"],
                 "rule": "end_date >= history_from and"
                         " end_date < as_of"}
    params = {"model": name, "version": m["version"],
              "strength": m["strength"],
              "calibration_months": m["calibration_months"],
              "no_order_days": m["no_order_days"], **selection}
    store = Path(root) / paths["mlruns"]
    rec = track(f"{name}, v{m['version']}", fitted, params, rows, table,
                scores, store=store, log=store / "runs.jsonl",
                extra_code=[*CODE, settings.path(name)],
                table_path=table_path, selection=selection)
    say("6 record    the run, its rows and how they were chosen, in"
        " the\n            hand-built log and in MLflow")

    card = render(evaluation.card(model, table, scored, scores, cfg))
    say(f"7 card      {len(card.splitlines())} lines, every number"
        " measured on this run")

    meta = {"name": name, "version": m["version"], "run": rec["run"],
            "inputs": schema(rows, COLUMNS),
            "features": list(fitted.estimator_.named_steps["prepare"]
                             .get_feature_names_out()),
            "data": {**rec["data"], **selection},
            "code": {k: v for k, v in rec["code"].items()
                     if k != "environment"},
            "settings": cfg, "metrics": scores,
            "checks": {"leakage": "passed",
                       "reviews": report.reviews()}}
    registry = Registry(name, Path(root) / paths["artifacts"])
    version = registry.register(fitted, meta, card)
    say(f"8 register  {name} version {version}, staged")
    return {"settings": cfg, "table": table, "rows": rows,
            "scored": scored, "report": report, "model": fitted,
            "scores": scores, "record": rec, "card": card,
            "registry": registry, "version": version}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", nargs="?", default="renewal")
    ap.add_argument("--set", dest="overrides", action="append",
                    default=[], metavar="SECTION.KEY=VALUE")
    args = ap.parse_args(argv)
    run(args.model, args.overrides)


if __name__ == "__main__":
    main()
