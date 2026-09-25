"""
Where a list is wrong: its errors by slice of the contracts, each with
a bootstrap interval. Chapter 16 writes it.

    python -m foresight.slices       v0.6 on the validation cohorts

A slice is a group of contracts that somebody might treat differently
or ask about separately: a region, a segment, a size of account, a
length of tenure, the key accounts against the long tail. For each,
the table gives what the list did (how many of the slice's leavers it
called, how many of its renewers it called as well) and whether the
chances held within it (the average chance given against the share
that left). The intervals come from resampling contracts within each
monthly cohort, as Chapter 8's bootstrap does, with the list's calls
fixed: they say how much the slice's numbers could move with a
different draw of the same kind of contracts.

The bands for size and tenure are cut on the training rows, so a
slice means the same thing in every period it is measured in.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE, SEED, rng
from foresight.evaluate import REPS

SLICES = ["region", "segment", "size", "tenure", "population"]
TENURE_YEARS = [1, 3, 6]        # bands: under 1, 1-3, 3-6, over 6
SIZES = ["smallest", "second", "third", "largest"]


def key_accounts(warehouse: Path = ML_WAREHOUSE) -> set:
    """The ids of Meridian's key accounts."""
    with sqlite3.connect(warehouse) as con:
        ids = con.execute("SELECT account_id FROM accounts"
                          " WHERE is_key_account = 1").fetchall()
    return {i for (i,) in ids}


def label(scored: pd.DataFrame, train: pd.DataFrame,
          keys: set) -> pd.DataFrame:
    """`scored` with a column for every slice. Size is the quarter of
    last year's spend, cut where the training rows' quarters fall."""
    cuts = train.spend_365.quantile([0.25, 0.5, 0.75]).to_numpy()
    years = scored.tenure_days / 365.25
    tenure = np.select([years < y for y in TENURE_YEARS],
                       ["under 1 year", "1-3 years", "3-6 years"],
                       "over 6 years")
    return scored.assign(
        size=pd.Categorical(
            np.array(SIZES)[np.searchsorted(cuts, scored.spend_365)],
            categories=SIZES),
        tenure=pd.Categorical(tenure, categories=[
            "under 1 year", "1-3 years", "3-6 years", "over 6 years"]),
        population=np.where(scored.account_id.isin(keys),
                            "key accounts", "long tail"))


def _draws(scored: pd.DataFrame, reps: int, seed: int):
    """Row positions for each resample: contracts drawn with
    replacement within each cohort, as evaluate.resample() draws."""
    g = rng(seed)
    groups = list(scored.groupby("moment").indices.values())
    for _ in range(reps):
        yield np.concatenate([rows[g.integers(0, len(rows), len(rows))]
                              for rows in groups])


def _rates(code, y, called, p, n):
    """Per slice: the share of its leavers not called, the share of
    its renewers called, and the average chance minus the share
    left."""
    size = np.bincount(code, minlength=n).astype(float)
    left = np.bincount(code, weights=y, minlength=n)
    hit = np.bincount(code, weights=y * called, minlength=n)
    waste = np.bincount(code, weights=(1 - y) * called, minlength=n)
    chance = np.bincount(code, weights=p, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1 - hit / left, waste / (size - left),
                (chance - left) / size)


def table(scored: pd.DataFrame, by: str, called, reps: int = REPS,
          seed: int = SEED) -> pd.DataFrame:
    """One row per value of `by`: contracts, leavers, the share that
    left, calls, the share of leavers missed and of renewers called,
    and the chance given minus the share left, the last three with
    95% intervals. `called` marks the contracts the list called."""
    s = scored.reset_index(drop=True)
    cats = pd.Categorical(s[by])
    code, names = cats.codes, list(cats.categories)
    y = s.not_renewed.to_numpy(dtype=float)
    c = np.asarray(called, dtype=float)
    p = s.model.to_numpy(dtype=float)
    k = len(names)
    miss, fpr, gap = _rates(code, y, c, p, k)
    draws = np.array([_rates(code[i], y[i], c[i], p[i], k)
                      for i in _draws(s, reps, seed)])  # reps, 3, k
    out = []
    for j, name in enumerate(names):
        row = {"slice": name, "contracts": int((code == j).sum()),
               "leavers": int(y[code == j].sum()),
               "calls": int(c[code == j].sum())}
        row["left"] = row["leavers"] / row["contracts"]
        for m, (what, point) in enumerate(
                (("miss", miss), ("fpr", fpr), ("gap", gap))):
            vals = draws[:, m, j][~np.isnan(draws[:, m, j])]
            row[what] = point[j]
            row[what + "_lo"], row[what + "_hi"] = (
                np.percentile(vals, [2.5, 97.5]) if len(vals)
                else (np.nan, np.nan))
        out.append(row)
    return pd.DataFrame(out).set_index("slice")


def _pct(v: float) -> str:
    return "-" if np.isnan(v) else f"{v * 100:.0f}"


def _pts(v: float) -> str:
    """A share as signed points, with no sign on a zero."""
    s = f"{v * 100:+.1f}"
    return " 0.0" if s in ("+0.0", "-0.0") else s


def page(scored: pd.DataFrame, called, slices=SLICES,
         reps: int = REPS, seed: int = SEED) -> str:
    """Every slice as fixed-width text no wider than 68."""
    out = [f"{'':15}{'rows':>6}{'left':>5}{'calls':>6}{'miss':>5}"
           f"{'(95%)':>8}{'FPR':>4}{'gap':>6}{'(95%)':>13}"]
    for by in slices:
        out.append(by)
        for r in table(scored, by, called, reps, seed).itertuples():
            ci = (f"{_pct(r.miss_lo)}-{_pct(r.miss_hi)}"
                  if not np.isnan(r.miss_lo) else "")
            gi = f"{_pts(r.gap_lo)} to {_pts(r.gap_hi)}"
            out.append(f" {str(r.Index):<14}{r.contracts:>6,}"
                       f"{r.leavers:>5}{r.calls:>6}{_pct(r.miss):>5}"
                       f"{ci:>8}{_pct(r.fpr):>4}{_pts(r.gap):>6}"
                       f"{gi:>13}")
    out += ["miss: % of the slice's leavers not called. FPR: % of its",
            "renewers called. gap: average chance minus share left, in",
            "points. Intervals: 95%, resampled within each cohort."]
    return "\n".join(out)


def main(argv=None) -> None:
    from foresight.data.build_table import TABLE
    from foresight.decide import by_capacity, calibrated
    from foresight.evaluate import SPLITS, backtest
    from foresight.train import make_model
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="validation")
    args = ap.parse_args(argv)
    table_ = pd.read_parquet(TABLE)
    s = backtest(table_, *SPLITS[args.split], calibrated(make_model()))
    train = table_[table_.end_date.between("2023-01-01", "2024-06-30")]
    s = label(s, train, key_accounts())
    print(page(s, by_capacity(s)))


if __name__ == "__main__":
    main()
