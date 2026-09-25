"""
Foresight's training table: one row per contract renewal, as it stood on
the morning of its 90-day mark, labelled with whether it renewed.

    python -m foresight.data.build_table                  2023 to 2025
    python -m foresight.data.build_table --last 2024-06-30    training

Every feature reads only records dated strictly before the moment.
Orders and tickets filed under a legacy CRM id are moved to the
account's current id before anything is counted. The table is checked
before it is written, and a table that fails a check is not written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

from foresight.config import DATA, ML_WAREHOUSE

FORESIGHT_DATA = DATA / "foresight"
TABLE = FORESIGHT_DATA / "renewals_table.parquet"
MOMENT_DAYS = 90        # the moment: 90 days before the contract ends
LEGACY_FROM = 90_000    # ids at or above this are the old CRM's

COLUMNS = ["contract_id", "account_id", "moment", "end_date",
           "days_since_order", "orders_90d", "orders_prev_90d",
           "spend_365", "tickets_90d", "tenure_days", "segment",
           "region", "term_months", "legacy_terms", "discount_pct",
           "not_renewed"]


class TableCheckError(Exception):
    """The table failed a check, and nothing was written."""


# ------------------------------------------------ the rows
def renewals(con, first: str, last: str,
             known_by: str | None = None) -> pd.DataFrame:
    """Contracts ending in [first, last] whose outcome is recorded."""
    rows = pd.read_sql_query("""
        SELECT contract_id, account_id, end_date, term_months,
               legacy_terms, discount_pct,
               outcome = 'not_renewed' AS not_renewed
        FROM contracts
        WHERE end_date BETWEEN ? AND ?
          AND outcome IS NOT NULL""", con, params=(first, last))
    rows["end_date"] = pd.to_datetime(rows.end_date)
    if known_by is not None:            # outcome written by that day
        rows = rows[rows.end_date < pd.Timestamp(known_by)]
    rows["moment"] = rows.end_date - pd.Timedelta(days=MOMENT_DAYS)
    rows["discount_pct"] = rows.discount_pct.astype("Int64")
    return rows.reset_index(drop=True)


# ------------------------------------------------ one company, one id
def tidy(name: str) -> str:
    """A company name as the two CRMs would agree on it."""
    name = " ".join(name.lower().split())
    for tail in (" (old)", " inc"):
        name = name.removesuffix(tail)
    return name


def legacy_ids(con) -> dict[int, int]:
    """Legacy id -> current id: same postcode, start date and name."""
    pairs = pd.read_sql_query("""
        SELECT old.account_id AS legacy_id, new.account_id,
               old.name AS legacy_name, new.name
        FROM accounts old
        JOIN accounts new
          ON new.postcode = old.postcode AND new.since = old.since
         AND new.crm_source = 'Meridian CRM'
        WHERE old.crm_source = 'Legacy CRM'""", con)
    same = pairs.legacy_name.map(tidy) == pairs.name.map(tidy)
    return dict(zip(pairs.legacy_id[same], pairs.account_id[same]))


def events(con, id_map: dict[int, int]):
    """Orders with their value, and tickets, under the current id."""
    orders = pd.read_sql_query("""
        SELECT o.account_id, o.order_date AS day,
               SUM(l.qty * l.unit_price) AS value
        FROM orders o JOIN order_lines l USING (order_id)
        GROUP BY o.order_id""", con)
    tickets = pd.read_sql_query(
        "SELECT account_id, date(opened_at) AS day FROM tickets", con)
    for df in (orders, tickets):
        df["account_id"] = df.account_id.replace(id_map)
        df["day"] = pd.to_datetime(df.day)
        stranded = (df.account_id >= LEGACY_FROM).sum()
        if stranded:            # history the mapping did not reach
            raise TableCheckError(f"{stranded:,} events on legacy ids")
    return orders, tickets


# ------------------------------------------------ features
def window_features(rows, orders, tickets) -> pd.DataFrame:
    """Counts and sums over windows ending the day before the moment."""
    keys = rows[["contract_id", "account_id", "moment"]]

    o = keys.merge(orders, on="account_id")
    o["age"] = (o.moment - o.day).dt.days     # 1 = the day before
    o = o[o.age >= 1]                    # strictly before the moment
    o["in_90"] = o.age <= 90
    o["prev_90"] = (o.age > 90) & (o.age <= 180)
    o["spend"] = o.value.where(o.age <= 365, 0.0)
    by = o.groupby("contract_id")
    out = pd.DataFrame({"days_since_order": by.age.min(),
                        "orders_90d": by.in_90.sum(),
                        "orders_prev_90d": by.prev_90.sum(),
                        "spend_365": by.spend.sum().round(2)})

    t = keys.merge(tickets, on="account_id")
    age = (t.moment - t.day).dt.days
    t = t[(age >= 1) & (age <= 90)]
    tickets_90d = t.groupby("contract_id").size().rename("tickets_90d")
    out = out.join(tickets_90d, how="outer")
    return out


def segment_as_of(rows, con) -> pd.Series:
    """Each account's segment in force the day before its moment."""
    history = pd.read_sql_query("""
        SELECT account_id, valid_from, segment
        FROM account_history""", con)
    history["valid_from"] = pd.to_datetime(history.valid_from)
    keys = rows[["contract_id", "account_id", "moment"]]
    joined = pd.merge_asof(
        keys.sort_values("moment"), history.sort_values("valid_from"),
        left_on="moment", right_on="valid_from", by="account_id",
        allow_exact_matches=False)     # a change on the day is too late
    return joined.set_index("contract_id").segment


def build(first: str, last: str, known_by: str | None = None,
          warehouse: Path = ML_WAREHOUSE) -> pd.DataFrame:
    """The training table for contracts ending from first to last."""
    con = sqlite3.connect(warehouse)
    rows = renewals(con, first, last, known_by)
    orders, tickets = events(con, legacy_ids(con))
    accounts = pd.read_sql_query("""
        SELECT a.account_id, a.since, r.name AS region
        FROM accounts a JOIN regions r USING (region_id)""", con)
    accounts["since"] = pd.to_datetime(accounts.since)

    table = (rows.merge(accounts, on="account_id", how="left")
                 .join(window_features(rows, orders, tickets),
                       on="contract_id")
                 .join(segment_as_of(rows, con), on="contract_id"))
    con.close()
    table["tenure_days"] = (table.moment - table.since).dt.days
    counts = ["orders_90d", "orders_prev_90d", "tickets_90d"]
    table[counts] = table[counts].fillna(0).astype(int)
    table["spend_365"] = table.spend_365.fillna(0.0)
    table["days_since_order"] = table.days_since_order.astype("Int64")
    table["not_renewed"] = table.not_renewed.astype(int)
    return (table[COLUMNS].sort_values(["moment", "contract_id"])
                          .reset_index(drop=True))


# ------------------------------------------------ checks
def expected_rows(first, last, known_by=None, warehouse=ML_WAREHOUSE):
    """How many rows the table should have, counted a second way."""
    known = known_by or "9999-12-31"
    with sqlite3.connect(warehouse) as con:
        (n,) = con.execute("""
            SELECT COUNT(*) FROM contracts
            WHERE end_date BETWEEN ? AND ? AND end_date < ?
              AND outcome IS NOT NULL""",
            (first, last, known)).fetchone()
    return n


def check(table, first, last, known_by=None,
          expected=None) -> list[str]:
    """Everything that must be true of the table. Empty means passed."""
    problems = []
    if len(table) == 0:
        problems.append("the table is empty")
    if expected is not None and len(table) != expected:
        problems.append(f"{len(table):,} rows, expected {expected:,}")
    ids = table.contract_id
    dupes = ids[ids.duplicated()].nunique()
    if dupes:
        problems.append(f"{dupes:,} contracts appear more than once")
    lo, hi = pd.Timestamp(first), pd.Timestamp(last)
    if not table.end_date.between(lo, hi).all():
        problems.append(f"end dates outside {first} to {last}")
    if known_by and (table.end_date >= pd.Timestamp(known_by)).any():
        problems.append(f"outcomes not yet known on {known_by}")
    gap = (table.end_date - table.moment).dt.days
    if (gap != MOMENT_DAYS).any():
        problems.append(
            f"moments not {MOMENT_DAYS} days before the end")
    if (table.days_since_order < 1).any():
        problems.append("an order on or after the moment was counted")
    if (table.account_id >= LEGACY_FROM).any():
        problems.append("rows still keyed by a legacy CRM id")
    if not table.not_renewed.isin([0, 1]).all():
        problems.append("labels other than 0 and 1")
    required = ["contract_id", "account_id", "moment", "end_date",
                "region", "term_months", "legacy_terms", "not_renewed"]
    for col in required:
        if table[col].isna().any():
            problems.append(f"missing values in {col}")
    return problems


# ------------------------------------------------ writing
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(table, first, last, known_by=None, path: Path = TABLE,
          warehouse: Path = ML_WAREHOUSE) -> dict:
    """Check the table; if it passes, write it and its manifest."""
    expected = expected_rows(first, last, known_by, warehouse)
    problems = check(table, first, last, known_by, expected)
    if problems:
        raise TableCheckError("; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    rows_hash = pd.util.hash_pandas_object(table, index=False)
    manifest = {
        "table": path.name,
        "rows": len(table),
        "contracts_ending": [first, last],
        "outcomes_known_by": known_by,
        "moment_days_before_end": MOMENT_DAYS,
        "label": "not_renewed",
        "label_rate": round(float(table.not_renewed.mean()), 4),
        "columns": {c: str(t) for c, t in table.dtypes.items()},
        "missing": {c: int(n)
                    for c, n in table.isna().sum().items() if n},
        "content_sha256": hashlib.sha256(
            rows_hash.values.tobytes()).hexdigest(),
        "warehouse_sha256": sha256(warehouse),
        "builder_sha256": sha256(Path(__file__)),
        "checks": "passed",
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--first", default="2023-01-01")
    ap.add_argument("--last", default="2025-12-31")
    ap.add_argument("--known-by", default=None)
    ap.add_argument("--out", type=Path, default=TABLE)
    args = ap.parse_args(argv)
    table = build(args.first, args.last, args.known_by)
    manifest = write(table, args.first, args.last, args.known_by,
                     args.out)
    print(f"{manifest['rows']:,} rows written to {args.out}")


if __name__ == "__main__":
    main()
