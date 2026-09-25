"""
Where Foresight's answers are written: a database of its own, beside
the warehouse and never inside it. Chapter 22 writes it.

    <root>/data/foresight/scores.db

    runs       one row per run of a job: which cohort, which model,
               how far it got, and how it ended
    scores     every contract a run scored: chance, rank, whether it
               is on the list, reasons, and the holdout's arm
    inputs     the rows exactly as the model saw them
    forecasts  next quarter's demand, one row per product, region
               and month
    answers    a view: each cohort's list of record, from the latest
               finished run for its mark

Every row carries the run that wrote it, so a run is never updated in
place and any list can be read again as it was. The index one_answer
lets each cohort finish once; a second list for the same mark must
give a reason, and the first stays.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FILE = "scores.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run          INTEGER PRIMARY KEY,
    job          TEXT NOT NULL,
    mark         TEXT NOT NULL,
    run_on       TEXT NOT NULL,
    source       TEXT,
    model_as_of  TEXT,
    calibration  TEXT,
    status       TEXT NOT NULL,
    step         TEXT,
    rows         INTEGER,
    reason       TEXT,
    error        TEXT,
    started      TEXT,
    finished     TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS one_answer ON runs (job, mark)
    WHERE status = 'finished' AND reason IS NULL;
CREATE TABLE IF NOT EXISTS scores (
    run          INTEGER NOT NULL REFERENCES runs (run),
    mark         TEXT NOT NULL,
    contract_id  INTEGER NOT NULL,
    account_id   INTEGER NOT NULL,
    key_account  INTEGER NOT NULL,
    rank         INTEGER,
    chance       REAL,
    listed       INTEGER NOT NULL,
    below        INTEGER NOT NULL,
    arm          TEXT,
    reasons      TEXT,
    unfamiliar   TEXT,
    PRIMARY KEY (run, contract_id));
CREATE TABLE IF NOT EXISTS forecasts (
    run          INTEGER NOT NULL REFERENCES runs (run),
    target       TEXT NOT NULL,
    h            INTEGER NOT NULL,
    category     TEXT NOT NULL,
    sku          TEXT NOT NULL,
    region       TEXT NOT NULL,
    units        REAL NOT NULL,
    lo           REAL,
    hi           REAL,
    PRIMARY KEY (run, sku, region, target));
CREATE VIEW IF NOT EXISTS answers AS
    SELECT s.*, r.source, r.run_on FROM scores s JOIN runs r USING (run)
    WHERE r.run = (SELECT MAX(x.run) FROM runs x
                   WHERE x.job = r.job AND x.mark = r.mark
                     AND x.status = 'finished');
"""


def path(root: Path) -> Path:
    """The scores database of the installation at `root`."""
    return Path(root) / "data" / "foresight" / FILE


def connect(file: Path) -> sqlite3.Connection:
    """The database, made if it is not there."""
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(file)
    con.executescript(SCHEMA)
    return con


def read(file: Path, sql: str, params=()) -> pd.DataFrame:
    """A query against the database, read-only."""
    uri = f"{Path(file).resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as con:
        return pd.read_sql_query(sql, con, params=params)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------ a run's life
def finished(con, job: str, mark) -> int | None:
    """The first finished run for this cohort, if there is one."""
    row = con.execute(
        "SELECT MIN(run) FROM runs WHERE job = ? AND mark = ?"
        " AND status = 'finished'", (job, _day(mark))).fetchone()
    return row[0]


def start(con, job: str, mark, on, reason: str | None = None) -> int:
    """A new run, recorded before any work is done."""
    with con:
        cur = con.execute(
            "INSERT INTO runs (job, mark, run_on, status, step, reason,"
            " started) VALUES (?, ?, ?, 'started', 'start', ?, ?)",
            (job, _day(mark), _day(on), reason, _now()))
    return cur.lastrowid


def step(con, run: int, name: str, **fields) -> None:
    """Record that a run has reached `name`, with anything it knows."""
    sets = ", ".join(f"{k} = ?" for k in ["step", *fields])
    with con:
        con.execute(f"UPDATE runs SET {sets} WHERE run = ?",
                    (name, *fields.values(), run))


def fail(con, run: int, error: BaseException) -> None:
    """Mark a run failed, keeping the step it reached and why."""
    with con:
        con.execute("UPDATE runs SET status = 'failed', error = ?,"
                    " finished = ? WHERE run = ?",
                    (f"{type(error).__name__}: {error}", _now(), run))


def write(con, run: int, mark, scored: pd.DataFrame,
          rows: pd.DataFrame) -> None:
    """Scores and the rows behind them, and the run's end: all of it
    in one transaction, or none of it."""
    out = pd.DataFrame({
        "run": run, "mark": _day(mark),
        "contract_id": scored.contract_id.astype(int),
        "account_id": scored.account_id.astype(int),
        "key_account": scored.is_key_account.astype(int),
        "rank": scored["rank"].where(scored["rank"] > 0)
                              .astype("Int64"),
        "chance": scored.chance, "listed": scored.listed.astype(int),
        "below": scored.below.astype(int), "arm": scored.arm,
        "reasons": scored.reasons.map(json.dumps),
        "unfamiliar": scored.unfamiliar.map(json.dumps)})
    kept = rows.assign(run=run)
    for c in ("moment", "end_date"):
        kept[c] = kept[c].dt.strftime("%Y-%m-%d")
    with con:
        out.to_sql("scores", con, if_exists="append", index=False)
        kept.to_sql("inputs", con, if_exists="append", index=False)
        con.execute("UPDATE runs SET status = 'finished',"
                    " step = 'done', rows = ?, finished = ?"
                    " WHERE run = ?",
                    (len(out), _now(), run))


def inputs(file: Path, run: int, manifest: dict) -> pd.DataFrame:
    """The rows a run scored, with the types the model was fitted on."""
    rows = read(file, "SELECT * FROM inputs WHERE run = ?"
                " ORDER BY contract_id", (run,)).drop(columns="run")
    for c in ("moment", "end_date"):
        rows[c] = pd.to_datetime(rows[c])
    for c, kind in manifest["inputs"].items():
        rows[c] = rows[c].astype(kind)
    return rows


def _day(value) -> str:
    return f"{pd.Timestamp(value):%Y-%m-%d}"
