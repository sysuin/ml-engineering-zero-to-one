"""
Foresight's monthly list, v0.6. Chapter 16 writes it; Chapter 22 turns
it into a scheduled job.

    python -m foresight.score --as-of 2024-10-02
    python -m foresight.score --as-of 2024-10-02 --out list.csv

One run makes the list for the cohort whose 90-day mark falls on or
just before `as_of`:

1. runs v0.5's training job (train.py), whose leakage checks stop
   everything if a column leaks;
2. fits v0.6's model on the outcomes known that morning: v0.5's lasso,
   its chances Platt-calibrated on the latest three cohorts it may see
   (Chapter 14's Calibrated);
3. ranks the cohort's long-tail contracts and keeps the top 40, each
   with its chance, a mark if the chance is below the break-even, three
   reasons in words (explain.py), and any column the model leans on
   whose value it has rarely seen;
4. lists the cohort's key accounts apart, unranked and without a
   chance, for their account managers (Chapter 16's Halloway section
   says why).
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE
from foresight.decide import COSTS, Calibrated
from foresight.evaluate import HISTORY_FROM, known_by
from foresight.explain import (SMALLEST, contributions, facts, reasons,
                               unfamiliar)
from foresight.models.logistic import CALLS
from foresight.train import make_model, train

VERSION = "0.6"


def model() -> Calibrated:
    """v0.6's model: v0.5's lasso with Platt-calibrated chances."""
    return Calibrated(make_model())


def accounts(rows: pd.DataFrame, warehouse: Path = ML_WAREHOUSE
             ) -> pd.DataFrame:
    """Each contract's account name, whether it is a key account, and
    its manager on the day before the mark (from the history, never
    today's accounts table: Chapter 13's Retention desk)."""
    with sqlite3.connect(warehouse) as con:
        names = pd.read_sql_query(
            "SELECT account_id, name, is_key_account FROM accounts",
            con)
        history = pd.read_sql_query(
            "SELECT account_id, valid_from, account_manager"
            " FROM account_history", con)
    history["valid_from"] = pd.to_datetime(history.valid_from)
    keys = rows[["contract_id", "account_id", "moment"]]
    who = pd.merge_asof(
        keys.sort_values("moment"), history.sort_values("valid_from"),
        left_on="moment", right_on="valid_from", by="account_id",
        allow_exact_matches=False)
    return (rows.merge(names, on="account_id", how="left")
                .merge(who[["contract_id", "account_manager"]],
                       on="contract_id", how="left")
                .set_index(rows.index))


def score(as_of: str, table: pd.DataFrame | None = None,
          k: int = CALLS, checks: bool = True,
          warehouse: Path = ML_WAREHOUSE) -> dict:
    """The list for the latest mark on or before `as_of`."""
    table = pd.read_parquet(TABLE) if table is None else table
    if checks:
        train(as_of, table)             # raises LeakageError on a leak
    rows = known_by(table[table.end_date >= HISTORY_FROM], as_of)
    marks = np.sort(table.moment.unique())
    mark = pd.Timestamp(marks[marks <= pd.Timestamp(as_of)][-1])
    cohort = accounts(table[table.moment == mark], warehouse)

    fitted = model().fit(rows)
    cohort["chance"] = fitted.predict_proba(cohort)
    _, parts = contributions(fitted, cohort)
    cohort["reasons"] = reasons(parts, cohort, facts(rows))
    # A rare value matters only on a column the model leans on.
    odd = unfamiliar(rows, cohort)
    odd &= parts[odd.columns].abs().to_numpy() >= SMALLEST
    cohort["unfamiliar"] = [list(odd.columns[f])
                            for f in odd.to_numpy()]
    cohort["below"] = cohort.chance <= COSTS.break_even()

    key = cohort.is_key_account == 1
    ranked = cohort[~key].sort_values(["chance", "contract_id"],
                                      ascending=[False, True])
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return {"as_of": as_of, "mark": mark, "trained_on": len(rows),
            "contracts": len(cohort), "k": k, "list": ranked.head(k),
            "keys": cohort[key].sort_values("contract_id"),
            "map": (fitted.map_.a_, fitted.map_.b_)}


def page(result: dict, ranks=None) -> str:
    """The list as fixed-width text no wider than 68: the entries at
    `ranks`, or all of them, then the key accounts."""
    listed = result["list"]
    shown = (listed if ranks is None
             else listed[listed["rank"].isin(ranks)])
    below = int(listed.below.sum())
    out = [f"Foresight v{VERSION}: the list for the cohort marked"
           f" {result['mark']:%Y-%m-%d}",
           f"{result['contracts']} contracts reach their mark;"
           f" {len(listed)} calls; break-even"
           f" {COSTS.break_even():.1%}",
           f"Model fitted on the {result['trained_on']:,} outcomes"
           f" known on {result['as_of']}", ""]
    for r in shown.itertuples():
        mark = "  below" if r.below else ""
        out.append(f"{r.rank:>2}  {r.contract_id:<6} {r.name[:24]:<24}"
                   f" {str(r.account_manager)[:14]:<14}"
                   f"{r.chance:>6.1%}{mark}")
        for why in r.reasons:
            out.append(f"      - {why}")
        if r.unfamiliar:
            out.append("      ! rarely seen: "
                       + ", ".join(r.unfamiliar))
    if len(shown) < len(listed):
        out.append(f"   ... and {len(listed) - len(shown)} more")
    expected = float(listed.chance[listed.below].sum())
    out += ["", f"{below} of the {len(listed)} calls are below the"
            f" break-even; together they", f"are expected to reach"
            f" {expected:.1f} leavers.", ""]
    keys = result["keys"]
    out += [f"Key accounts at their mark: {len(keys)}, not ranked",
            f"{'':50}{'orders':>7}{'tickets':>9}",
            f"{'':50}{'90d':>4}{'prior':>6}{'90d':>6}"]
    for r in keys.itertuples():
        out.append(f"    {r.contract_id:<6} {r.name[:22]:<22}"
                   f" {str(r.account_manager)[:14]:<15}"
                   f"{r.orders_90d:>5}{r.orders_prev_90d:>6}"
                   f"{r.tickets_90d:>6}")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-of", default="2024-10-02")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-checks", action="store_true")
    args = ap.parse_args(argv)
    result = score(args.as_of, checks=not args.no_checks)
    print(page(result))
    if args.out is not None:
        listed = result["list"].assign(
            reasons=lambda d: d.reasons.map(" | ".join),
            unfamiliar=lambda d: d.unfamiliar.map(", ".join))
        listed[["rank", "contract_id", "account_id", "name",
                "account_manager", "chance", "below", "reasons",
                "unfamiliar"]].to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
