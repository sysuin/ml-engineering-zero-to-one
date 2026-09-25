"""
Foresight's monthly list as a job that is safe to run every night.
Chapter 22 writes it. It supersedes score.py, which fits a model of
its own each time and can score only cohorts whose outcomes are
already known.

    make score                   python -m foresight.serve.batch
    python -m foresight.serve.batch --on 2025-12-31
    python -m foresight.serve.batch --again "why"   a second list
    python -m foresight.serve.batch --rule "why"    the rule's list
    python -m foresight.serve.batch forecast        next quarter

Each run asks one question: whose mark is due this morning (the
latest on or before `on`), and has that cohort's list been made? If it
has, the run does nothing. If not, it makes it, recording each step in
the run's row of the scores database (store.py) as it goes:

1. fresh    the warehouse holds the day before the mark
2. model    the production model, verified, fitted on no outcome
            recorded after this morning; Platt's map learned again
            on the latest cohorts it never saw (Chapter 24). The
            weights are never refitted here: a new model is a
            candidate for Chapter 24's gate and a person's promotion
3. rows     the cohort's rows from at_mark(), checked against the
            model's inputs and against the same rows made the API's
            way (Chapter 21's skew test)
4. score    chances, ranks and reasons; key accounts apart
5. holdout  Chapter 25's draw, appended to the holdout file
6. write    scores, the rows behind them and the run's end, in one
            transaction

A lock keeps two runs from working at once. A failure marks the run
failed, with its step and its error, releases the lock and exits
non-zero; the next run starts again from the top. A finished list is
never overwritten: a second list for the same mark needs a reason,
keeps the first draw of the holdout, and leaves the first list on
record.
"""
from __future__ import annotations

import argparse
import os
import socket
import sqlite3
import sys
from contextlib import closing, contextmanager
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE, ROOT
from foresight.impact import holdout
from foresight.models.logistic import CALLS
from foresight.pipeline import features
from foresight.pipeline.artifact import check_inputs
from foresight.pipeline.registry import Registry
from foresight.score import accounts, page
from foresight.serve import online, renewal, store, today

LOCK = "batch.lock"


class Locked(RuntimeError):
    """Another run holds the lock."""


class StaleData(RuntimeError):
    """The warehouse does not yet hold what the list needs."""


class TooNew(RuntimeError):
    """The model learned from outcomes recorded after this morning."""


# ------------------------------------------------ one run at a time
@contextmanager
def lock(file: Path):
    """Hold `file` for the length of a run. A lock left by a process
    that no longer exists on this machine is taken over."""
    file.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        try:
            fd = os.open(file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if attempt == 2 or not stale(file):
                raise Locked(f"another run holds {file.name}")
            file.unlink(missing_ok=True)
    os.write(fd, f"{socket.gethostname()} {os.getpid()}".encode())
    os.close(fd)
    try:
        yield
    finally:
        file.unlink(missing_ok=True)


def stale(file: Path) -> bool:
    """True if the lock's holder was on this machine and has gone."""
    host, _, pid = file.read_text().partition(" ")
    if host != socket.gethostname() or not pid.isdigit():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


# ------------------------------------------------ the checks
def fresh(mark, warehouse: Path = ML_WAREHOUSE) -> None:
    """Refuse to score before the day before the mark has loaded."""
    with closing(online.open_warehouse(warehouse)) as con:
        (last,) = con.execute("SELECT MAX(order_date)"
                              " FROM orders").fetchone()
    need = pd.Timestamp(mark) - pd.Timedelta(days=1)
    if last is None or pd.Timestamp(last) < need:
        raise StaleData(f"orders on record to {last};\nthe list"
                        f" marked {mark:%Y-%m-%d} needs them to"
                        f" {need:%Y-%m-%d}")


def production(root: Path, on):
    """The production model, its manifest and its version, if it
    could have existed on the morning of `on`."""
    registry = Registry("renewal", Path(root) / "artifacts")
    model, manifest = registry.production()
    version = next(v for v, r in registry.read()["versions"].items()
                   if r["stage"] == "production")
    as_of = pd.Timestamp(manifest["data"]["as_of"])
    if as_of > pd.Timestamp(on):
        raise TooNew(f"the production model knows the outcomes to"
                     f" {as_of:%Y-%m-%d}; this list is for the morning"
                     f" of {pd.Timestamp(on):%Y-%m-%d}")
    return model, manifest, version


# ------------------------------------------------ the monthly list
def month(on=None, root: Path = ROOT, again: str | None = None,
          rule: str | None = None, warehouse: Path = ML_WAREHOUSE,
          k: int = CALLS, held: int = holdout.HELD) -> dict:
    """Make the list for the mark due on `on`, unless it is made."""
    on = pd.Timestamp(on) if on is not None else today()
    mark = features.mark_on(on, warehouse)
    folder = store.path(root).parent
    with lock(folder / LOCK), closing(
            store.connect(store.path(root))) as con:
        done = store.finished(con, "renewal", mark)
        if done is not None and again is None:
            return {"mark": mark, "run": done, "status": "done already"}
        run = store.start(con, "renewal", mark, on, again or rule)
        try:
            return _make(con, run, on, mark, root, done is None,
                         rule, warehouse, k, held, folder)
        except Exception as e:
            store.fail(con, run, e)
            raise


def _make(con, run, on, mark, root, first, rule, warehouse, k, held,
          folder) -> dict:
    store.step(con, run, "fresh")
    fresh(mark, warehouse)

    store.step(con, run, "model")
    model = manifest = calibration = None
    if rule is None:
        model, manifest, version = production(root, on)
        as_of = manifest["data"]["as_of"]
        model, span = renewal.recalibrate(
            model, renewal.known(manifest, on, warehouse), as_of)
        calibration = ("as fitted" if span is None else
                       f"refitted on the cohorts marked"
                       f" {span[0]:%Y-%m-%d} to {span[1]:%Y-%m-%d}")
        store.step(con, run, "model", source=f"renewal v{version}",
                   model_as_of=as_of, calibration=calibration)
    else:
        store.step(con, run, "model", source="rule")

    store.step(con, run, "rows")
    rows = features.at_mark(mark, warehouse)
    if manifest is not None:
        check_inputs(rows, manifest)
    features.check_skew(rows, online.rows(rows.contract_id, warehouse))

    store.step(con, run, "score")
    if rule is None:
        train = renewal.training_rows(manifest, warehouse)
        scored = renewal.score(model, manifest, rows, train, k,
                               warehouse)
    else:
        scored = renewal.rule(accounts(rows, warehouse), k)

    store.step(con, run, "holdout")
    listed = scored[scored.listed].assign(moment=mark)
    file = folder / holdout.ASSIGNMENTS.name
    if first:
        drawn = holdout.assign(listed, held)
        holdout.record(drawn, file)
    else:                               # the first draw stands
        drawn = holdout.load(file)
        drawn = drawn[pd.to_datetime(drawn.moment) == mark]
    arms = drawn.set_index("contract_id").arm
    scored["arm"] = scored.contract_id.map(arms)

    store.step(con, run, "write")
    store.write(con, run, mark, scored, rows)
    return {"run": run, "status": "finished", "on": on, "mark": mark,
            "source": "rule" if rule else "model", "scored": scored,
            "manifest": manifest, "calibration": calibration, "k": k}


def summary(result: dict, ranks=range(1, 6)) -> str:
    """What a run did, as text no wider than 68."""
    mark = f"{pd.Timestamp(result['mark']):%Y-%m-%d}"
    if result["status"] != "finished":
        return (f"cohort {mark}: list made by run {result['run']};"
                " nothing to do")
    scored, manifest = result["scored"], result["manifest"]
    tail = scored[scored.is_key_account == 0]
    arms = scored.arm.value_counts()
    end = (f"holdout: {arms.get('called', 0)} to call,"
           f" {arms.get('held out', 0)} held out\n"
           f"run {result['run']} finished: {len(scored)} rows written")
    if manifest is None:                # the rule's list
        listed = tail[tail.listed & tail["rank"].isin(ranks)]
        lines = [f"Chapter 3's rule: the list for the cohort marked"
                 f" {mark}"]
        lines += [f"{r.rank:>2}  {r.contract_id:<6} {r.name[:24]:<24}"
                  f" {r.reasons[0]}" for r in listed.itertuples()]
        return "\n".join(lines) + "\n\n" + end
    shown = page({"as_of": manifest["data"]["as_of"],
                  "mark": result["mark"],
                  "trained_on": manifest["data"]["rows"],
                  "contracts": len(scored), "k": result["k"],
                  "list": tail[tail.listed],
                  "keys": scored[scored.is_key_account == 1]},
                 ranks=ranks)
    return (f"{shown}\n\nPlatt's map: {result['calibration']}\n"
            f"{end}")


# ------------------------------------------------ the forecast
def forecast(on=None, root: Path = ROOT) -> dict:
    """Next quarter's demand from the last whole month before `on`
    (Chapter 17's report), written as one run."""
    from foresight.forecast.report import next_quarter
    on = pd.Timestamp(on) if on is not None else today()
    origin = on.to_period("M") - 1
    folder = store.path(root).parent
    with lock(folder / LOCK), closing(
            store.connect(store.path(root))) as con:
        mark = origin.to_timestamp()
        done = store.finished(con, "forecast", mark)
        if done is not None:
            return {"mark": mark, "run": done, "status": "done already"}
        run = store.start(con, "forecast", mark, on)
        try:
            store.step(con, run, "forecast", source="global model")
            f = next_quarter(origin=str(origin))
            out = pd.DataFrame({
                "run": run, "target": f.target.astype(str), "h": f.h,
                "category": f.category, "sku": f.sku,
                "region": f.region, "units": f.forecast,
                "lo": f.lo, "hi": f.hi})
            with con:
                out.to_sql("forecasts", con, if_exists="append",
                           index=False)
                con.execute("UPDATE runs SET status = 'finished',"
                            " step = 'done', rows = ? WHERE run = ?",
                            (len(out), run))
        except Exception as e:
            store.fail(con, run, e)
            raise
    return {"mark": mark, "run": run, "status": "finished",
            "rows": len(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("job", nargs="?", default="renewal",
                    choices=["renewal", "forecast"])
    ap.add_argument("--on", default=None,
                    help="the morning, YYYY-MM-DD")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--again", default=None, metavar="REASON")
    ap.add_argument("--rule", default=None, metavar="REASON")
    a = ap.parse_args(argv)
    try:
        if a.job == "forecast":
            r = forecast(a.on, a.root)
            print(f"forecast from {r['mark']:%Y-%m}: run {r['run']},"
                  f" {r['status']}")
        else:
            print(summary(month(a.on, a.root, a.again, a.rule)))
    except Locked as e:
        print(f"not run: {e}", file=sys.stderr)
        return 75                       # EX_TEMPFAIL: try later
    except sqlite3.IntegrityError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
