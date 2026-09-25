"""
Leakage checks, run by every Foresight training job. Chapter 13 writes
them.

    python -m foresight.checks.leakage     v0.5's columns, checked

Four checks, because each kind of leak slips past some of them:

screen()        each column alone, against the ceiling. No column can
                rank the contracts better than the process that made
                the labels, so one that does is reading the answer.
audit()         every column's source, and a deletion test: rebuild it
                with every record dated on or after the mark deleted
                from the warehouse, and the value must not move.
adversarial()   the training rows against the rows about to be scored.
                A column that tells them apart on its own is a date,
                an id, or a world that has changed.
ceiling_check() the model's own backtest against the ceiling.

run_checks() runs all four. A "fail" stops the job with LeakageError;
a "review" is printed for a person and does not. A column can be let
through only by name, with a reason, in `allow`, so that every
exception is written down.
"""
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pandas.api.types import (is_datetime64_any_dtype,
                              is_numeric_dtype)
from sklearn.model_selection import StratifiedKFold

from foresight.config import LIGHTGBM_DETERMINISTIC, ML_WAREHOUSE, SEED
from foresight.data.build_table import segment_as_of, window_features
from foresight.evaluate import auc
from foresight.features.encoding import out_of_fold
from foresight.features.registry import REGISTRY, Feature
from foresight.features.sources import Context, Sources

# ------------------------------------------------ thresholds
CEILING_AUC = 0.811
"""The best AUC any model can reach on 2023-2025's renewals, measured
once from the generator's true probabilities (DECISIONS.md). With real
data nobody knows it; use the best model you trust, plus a margin."""
REVIEW_SHARE = 0.9
"""A column that goes 90% of the way from a coin (0.5) to the ceiling
on its own is not impossible, but a person should know why."""
CLOCK = 0.95
"""A column that tells old rows from new with an AUC this high is, for
a model, a date or an id: its values in the rows to be scored lie
where no training row was."""
SHIFT = 0.75
"""Above this, a column has moved between the periods: a launch, a
price change, a calendar. A person should look."""
MARGIN = 0.025
"""How far above the ceiling a whole model's backtest may land before
the job fails. The ceiling itself moves from period to period (0.794
on the training cohorts, 0.819 on validation's), so an honest model
near it can cross it by that much; above that, it is not luck."""


class LeakageError(Exception):
    """A leakage check failed, and no model was fitted."""


# ------------------------------------------------ what columns read
EVENTS = ("orders", "lines", "tickets")
"""Logs: every record carries its date, so the deletion test applies."""

FIXED = {"accounts": "since, is_key_account, region: set when the "
                     "account opened"}
"""Current-state tables a feature may read, and the attributes that
make it safe: ones that do not change after an account opens. No
deletion test can check them; the leakage review confirms them."""

WRITTEN_LATE = {
    "contracts.outcome": "the label, written when the contract ends",
    "contracts.notice_date": "written at notice, 60 days before the "
                             "end: after the mark",
    "contracts.cancellation_reason": "written only when an account "
                                     "leaves",
    "accounts.account_manager": "today's manager; leavers go to the "
                                "Retention desk at notice",
    "accounts.segment": "today's segment; account_history has it "
                        "as it was",
    "accounts.closed_on": "written when an account closes",
}
"""Sources no feature may read: each is filled in after the mark, or
holds today's value rather than the value at the mark."""

TABLE_SOURCES = {
    "days_since_order": ("orders", "event log"),
    "orders_90d": ("orders", "event log"),
    "orders_prev_90d": ("orders", "event log"),
    "spend_365": ("orders", "event log"),
    "tickets_90d": ("tickets", "event log"),
    "segment": ("account_history", "history"),
    "tenure_days": ("accounts.since", "fixed at opening"),
    "region": ("accounts.region", "fixed at opening"),
    "term_months": ("contracts", "signed"),
    "legacy_terms": ("contracts", "signed"),
    "discount_pct": ("contracts", "signed"),
}
"""Chapter 4's columns: where each comes from and how it is dated."""


def source_of(column: str, feature: Feature | None = None
              ) -> tuple[str, str]:
    """(source, kind) for a table column or a feature. A feature's
    kind is the weakest of the sources it reads."""
    if column in TABLE_SOURCES:
        return TABLE_SOURCES[column]
    feature = REGISTRY.get(column) if feature is None else feature
    if feature is None:
        return ("?", "undeclared")
    if not feature.reads:
        return ("the mark's date", "calendar")
    kinds = {"event log" if r in EVENTS else
             "fixed at opening" if r in FIXED else
             "written late" if r in WRITTEN_LATE else "undeclared"
             for r in feature.reads}
    worst = next(k for k in ("written late", "undeclared",
                             "fixed at opening", "event log")
                 if k in kinds)
    return ("+".join(feature.reads), worst)


# ------------------------------------------------ 1. the screen
def as_score(x: pd.Series, y, groups) -> np.ndarray:
    """A column as one number per row, for ranking. Numbers stay
    numbers, with a gap ranked below everything; dates become days;
    anything else is a category, target-encoded out of fold by
    account (Chapter 12), with a gap as a category of its own."""
    if is_datetime64_any_dtype(x):
        x = (x - pd.Timestamp("2000-01-01")).dt.days
    if is_numeric_dtype(x):
        v = x.astype(float)
        low = v.min() - 1 if v.notna().any() else 0.0
        return v.fillna(low).to_numpy()
    cats = x.astype("string").fillna("(missing)")
    return out_of_fold(cats, y, groups)


def single_auc(x: pd.Series, y, groups) -> float:
    """How well the column alone ranks leavers above renewers, in
    either direction: 0.5 is nothing, 1.0 is the label itself."""
    a = auc(y, as_score(x, y, groups))
    return max(a, 1 - a)


def screen(rows: pd.DataFrame, columns, ceiling: float = CEILING_AUC
           ) -> pd.DataFrame:
    """Every column alone against the ceiling: one row a column."""
    review = 0.5 + REVIEW_SHARE * (ceiling - 0.5)
    out = []
    for c in columns:
        a = single_auc(rows[c], rows.not_renewed, rows.account_id)
        verdict = ("fail" if a >= ceiling else
                   "review" if a >= review else "pass")
        out.append({"column": c, "auc": a, "screen": verdict})
    return pd.DataFrame(out).set_index("column")


# ------------------------------------------------ 2. the audit
def truncate(sources: Sources, mark) -> Sources:
    """The sources as they stood on the morning of `mark`: every
    record dated on or after it deleted."""
    mark = pd.Timestamp(mark)
    return Sources(*(df[df.day < mark] for df in
                     (sources.orders, sources.lines, sources.tickets)),
                   sources.accounts)


def _table_columns(rows, sources, history) -> pd.DataFrame:
    """Chapter 4's dated columns, computed by its own functions."""
    con = sqlite3.connect(":memory:")
    history.to_sql("account_history", con, index=False)
    out = window_features(rows, sources.orders, sources.tickets)
    out = out.reindex(rows.contract_id)
    out["segment"] = segment_as_of(rows, con).reindex(rows.contract_id)
    con.close()
    return out


def _changed(a: pd.Series, b: pd.Series) -> int:
    """Rows where two versions of a column disagree; a gap on both
    sides is agreement."""
    both_gap = a.isna().to_numpy() & b.isna().to_numpy()
    a, b = a.to_numpy(dtype=object), b.to_numpy(dtype=object)
    same = np.array([x == y or (isinstance(x, float) and
                                isinstance(y, float) and
                                np.isclose(x, y))
                     for x, y in zip(a, b)])
    return int((~(same | both_gap)).sum())


def deletion_test(rows: pd.DataFrame, features, sources: Sources,
                  history: pd.DataFrame, contracts: pd.DataFrame,
                  marks) -> pd.Series:
    """For each column, how many rows change when every record dated
    on or after the row's mark is deleted, over the cohorts at
    `marks`. Features may be names or Feature records."""
    changed: dict[str, int] = {}
    for mark in marks:
        k = rows[rows.moment == pd.Timestamp(mark)]
        cut = truncate(sources, mark)
        before = history[pd.to_datetime(history.valid_from) < mark]
        for f in features:
            if isinstance(f, str) and f in REGISTRY:
                f = REGISTRY[f]
            if isinstance(f, Feature):
                full = f.compute(Context(k, sources))
                try:
                    then = f.compute(Context(k, cut))
                except Exception:   # cannot be computed without
                    n = len(k)      # the future: every row moved
                else:
                    n = _changed(full.reset_index(drop=True),
                                 then.reset_index(drop=True))
                changed[f.name] = changed.get(f.name, 0) + n
        table = [c for c in features if c in TABLE_SOURCES]
        dated = [c for c in table
                 if TABLE_SOURCES[c][1] in ("event log", "history")]
        if dated:
            full = _table_columns(k, sources, history)
            then = _table_columns(k, cut, before)
            for c in dated:
                changed[c] = changed.get(c, 0) + _changed(
                    full[c].reset_index(drop=True),
                    then[c].reset_index(drop=True))
        signed = [c for c in table if TABLE_SOURCES[c][1] == "signed"]
        if signed:          # the terms were on record at the mark
            start = k.contract_id.map(contracts.start_date)
            late = int((pd.to_datetime(start) >= k.moment).sum())
            for c in signed:
                changed[c] = changed.get(c, 0) + late
    return pd.Series(changed, dtype=float)


def audit(rows: pd.DataFrame, columns, sources: Sources,
          warehouse: Path = ML_WAREHOUSE, marks=None,
          extra: tuple = ()) -> pd.DataFrame:
    """Every column's source, and the deletion test on the cohorts at
    `marks` (by default the first, middle and last in `rows`).
    `extra` adds Feature records the registry does not hold."""
    if marks is None:
        m = np.sort(rows.moment.unique())
        marks = [m[0], m[len(m) // 2], m[-1]]
    con = sqlite3.connect(warehouse)
    history = pd.read_sql_query(
        "SELECT account_id, valid_from, segment FROM account_history",
        con)
    contracts = pd.read_sql_query(
        "SELECT contract_id, start_date FROM contracts",
        con).set_index("contract_id")
    con.close()
    extra = {f.name: f for f in extra}
    kinds = {c: source_of(c, extra.get(c)) for c in columns}
    dated = [extra.get(c, c) for c in columns
             if kinds[c][1] not in ("written late", "undeclared")]
    moved = deletion_test(rows, dated, sources, history, contracts,
                          marks)
    out = []
    for c in columns:
        src, kind = kinds[c]
        n = moved.get(c, np.nan)
        verdict = ("fail" if kind in ("written late", "undeclared")
                   or n > 0 else "pass")
        out.append({"column": c, "source": src, "kind": kind,
                    "moved": n, "audit": verdict})
    return pd.DataFrame(out).set_index("column")


# ------------------------------------------------ 3. adversarial
def _matrix(rows: pd.DataFrame, columns) -> pd.DataFrame:
    X = pd.DataFrame(index=rows.index)
    for c in columns:
        x = rows[c]
        if is_datetime64_any_dtype(x):
            x = (x - pd.Timestamp("2000-01-01")).dt.days
        X[c] = (x.astype(float) if is_numeric_dtype(x)
                else x.astype("category"))
    return X


def _oof_auc(X: pd.DataFrame, y: np.ndarray, seed: int = SEED
             ) -> float:
    """Out-of-fold AUC of a small booster telling y = 1 from y = 0."""
    p = np.zeros(len(y))
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fit, held in folds.split(X, y):
        m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=100,
                           learning_rate=0.1, num_leaves=7,
                           min_child_samples=50)
        m.fit(X.iloc[fit], y[fit])
        p[held] = m.predict_proba(X.iloc[held])[:, 1]
    return auc(y, p)


def adversarial(old: pd.DataFrame, new: pd.DataFrame, columns
                ) -> tuple[float, pd.DataFrame]:
    """Can a model tell the training rows from the rows to be scored?
    The AUC with every column together, and each column on its own.
    No label is read: this can run on rows whose outcome is unknown."""
    both = pd.concat([old, new], ignore_index=True)
    X = _matrix(both, columns)
    y = np.r_[np.zeros(len(old)), np.ones(len(new))]
    together = _oof_auc(X, y)
    out = []
    for c in columns:
        a = _oof_auc(X[[c]], y)
        verdict = ("fail" if a >= CLOCK else
                   "review" if a >= SHIFT else "pass")
        out.append({"column": c, "shift auc": a, "shift": verdict})
    return together, pd.DataFrame(out).set_index("column")


# ------------------------------------------------ 4. the model itself
def ceiling_check(scored: pd.DataFrame, ceiling: float = CEILING_AUC
                  ) -> tuple[float, str]:
    """A backtest's AUC: "review" at or above the ceiling, "fail" more
    than MARGIN above it."""
    a = auc(scored.not_renewed, scored.model)
    return a, ("fail" if a >= ceiling + MARGIN else
               "review" if a >= ceiling else "pass")


# ------------------------------------------------ all of them
@dataclass
class Report:
    sheet: pd.DataFrame             # one row a column, every check
    shift_auc: float                # every column together
    model_auc: float | None = None
    model_verdict: str = "not run"
    allowed: dict = field(default_factory=dict)

    def failures(self) -> list[str]:
        out = []
        for c, r in self.sheet.iterrows():
            if c in self.allowed:
                continue
            for check in ("screen", "audit", "shift"):
                if r[check] == "fail":
                    out.append(f"{c}: {check}")
        if self.model_verdict == "fail":
            out.append(f"model: backtest AUC {self.model_auc:.3f},"
                       f" above the ceiling by more than {MARGIN}")
        return out

    def reviews(self) -> list[str]:
        out = [f"{c}: {check}" for c, r in self.sheet.iterrows()
               for check in ("screen", "audit", "shift")
               if r[check] == "review" and c not in self.allowed]
        if self.model_verdict == "review":
            out.append(f"model: backtest AUC {self.model_auc:.3f}, at"
                       " or above the ceiling")
        return out

    def text(self) -> str:
        """The sheet as fixed-width text no wider than 68."""
        head = (f"{'column':<18}{'source':<16}{'moved':>6}"
                f"{'label':>7}{'period':>8}  verdict")
        lines = [head]
        for c, r in self.sheet.iterrows():
            checks = [r["audit"], r["screen"], r["shift"]]
            v = ("allowed" if c in self.allowed else
                 "fail" if "fail" in checks else
                 "review" if "review" in checks else "pass")
            moved = "-" if np.isnan(r["moved"]) else f"{r['moved']:.0f}"
            lines.append(f"{c[:17]:<18}{r['source'][:15]:<16}"
                         f"{moved:>6}{r['auc']:>7.3f}"
                         f"{r['shift auc']:>8.3f}  {v}")
        lines.append(f"\nall columns together tell the periods apart:"
                     f" AUC {self.shift_auc:.3f}")
        if self.model_auc is not None:
            lines.append(f"model's backtest AUC {self.model_auc:.3f}"
                         f" against a ceiling of {CEILING_AUC}:"
                         f" {self.model_verdict}")
        return "\n".join(lines)


def run_checks(rows: pd.DataFrame, recent: pd.DataFrame, columns,
               sources: Sources, scored: pd.DataFrame | None = None,
               allow: dict | None = None, marks=None,
               warehouse: Path = ML_WAREHOUSE) -> Report:
    """Every check on `columns`: the screen and the audit on the
    training rows, adversarial validation against `recent`, and the
    ceiling check on a backtest if one is given. Raises LeakageError
    listing every failure not allowed by name."""
    columns = list(columns)
    sheet = (audit(rows, columns, sources, warehouse, marks)
             .join(screen(rows, columns)))
    together, shift = adversarial(rows, recent, columns)
    report = Report(sheet.join(shift), together,
                    allowed=dict(allow or {}))
    if scored is not None:
        report.model_auc, report.model_verdict = ceiling_check(scored)
    failed = report.failures()
    if failed:
        raise LeakageError("leakage checks failed:\n  "
                           + "\n  ".join(failed)
                           + "\n\n" + report.text())
    return report


def main(argv=None) -> None:
    from foresight.train import train
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-of", default="2024-07-01")
    args = ap.parse_args(argv)
    print(train(args.as_of)["report"].text())


if __name__ == "__main__":
    main()
