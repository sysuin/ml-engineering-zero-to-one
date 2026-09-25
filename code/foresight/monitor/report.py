"""
Foresight's monitoring report: one page, read weekly. Chapter 24.

    python -m foresight.monitor.report --day 2025-04-07
    python -m foresight.monitor.report          the year's last page,
                                                 written with the log

gather() lives the year once: the contracts as each list scored them
(cohorts.py), every contender's scores in shadow and the gate's
decisions (policy.py), the feed checks (feeds.py), the forecast's
record (forecast.py) and the alerts on the lists the account team
received (alerts.py). page() reads it as of one morning and prints
only what was on record by then, in the order a reader should take
it: what needs a decision, what fired this week, the latest list,
then the evidence as it arrives, notices before outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ROOT
from foresight.decide import Calibrated
from foresight.evaluate import HISTORY_FROM
from foresight.explain import contributions
from foresight.forecast.report import panels
from foresight.forecast.series import PRODUCT_REGION
from foresight.models.featured import FeaturedLasso
from foresight.monitor import alerts, feeds, forecast, policy, watch
from foresight.monitor.cohorts import (GO_LIVE, RECORD_ENDS,
                                       as_scored, known)
from foresight.monitor.policy import Contender
from foresight.monitor.psi import ACT, Baseline
from foresight.train import COLUMNS, STRENGTH, make_model

REPORT = ROOT / "docs" / "foresight-monitoring-2025.md"
ACKNOWLEDGED = {"exposure": "zero for every contract marked before the"
                            " rise, by construction"}
WEEK = pd.Timedelta(days=7)


def v06():
    """v0.6's model: the lasso, Platt-calibrated (score.model())."""
    return Calibrated(make_model())


def with_voss_share():
    """v0.6 with the library's Voss share (Chapter 12) added."""
    return Calibrated(lambda: FeaturedLasso(("supplier_voss",),
                                            STRENGTH))


def with_exposure():
    """v0.6 with the share of spend whose price rose before the mark."""
    return Calibrated(lambda: FeaturedLasso(("exposure",), STRENGTH))


CONTENDERS = (Contender("v0.6", v06, "never"),
              Contender("monthly", v06, "monthly"),
              Contender("voss share", with_voss_share, "monthly"),
              Contender("exposure", with_exposure, "monthly"))


@dataclass
class Year:
    rows: pd.DataFrame          # every contract, as scored at its mark
    shadow: pd.DataFrame        # every contender's chances
    log: pd.DataFrame           # the gate's decisions
    served: pd.Series           # mark -> the model the team received
    lists: pd.DataFrame         # the lists as received (watch.scored)
    alerts: list                # alerts on those lists
    feed: pd.DataFrame          # feed alerts
    months: pd.DataFrame        # the forecast, month by month


def gather(first=GO_LIVE, last=RECORD_ENDS,
           contenders=CONTENDERS) -> Year:
    """Run the year once."""
    rows = as_scored(HISTORY_FROM, "2026-03-31")
    s = policy.shadow(rows, contenders, first)
    names = [c.name for c in contenders]
    log, served = policy.forward(s, names[0], names[1:])
    got = policy.production(s, served)
    lists = rows.merge(got[["contract_id", "model", "name", "rule"]],
                       on="contract_id").rename(
        columns={"model": "chance"})
    feed = feeds.alerts(feeds.daily())
    fitted = {}

    def baseline(mark):
        c = next(c for c in contenders if c.name == served[mark])
        day = policy.fit_day(c, mark, first)
        if (c.name, day) not in fitted:
            ref = known(rows[rows.end_date >= HISTORY_FROM], day)
            fitted[c.name, day] = (c.make_model().fit(ref), ref)
        return fitted[c.name, day]

    found = alerts.year(baseline, lists, COLUMNS, first, last, feed,
                        acknowledged=ACKNOWLEDGED, due=rows)
    P, T = panels(PRODUCT_REGION)
    months = forecast.watch(forecast.by_month(forecast.record(P, (T,))),
                            first)
    return Year(rows, s, log, served, lists, found, feed, months)


def serving(y: Year, day) -> tuple:
    """The model serving on `day`, refitted as it was fitted then, and
    the rows it learned from."""
    marks = y.served.index[y.served.index <= pd.Timestamp(day)]
    name = y.served[marks[-1]] if len(marks) else y.served.iloc[0]
    c = next(c for c in CONTENDERS if c.name == name)
    mark = marks[-1] if len(marks) else GO_LIVE
    rows = known(y.rows[y.rows.end_date >= HISTORY_FROM],
                 policy.fit_day(c, mark))
    return name, c, c.make_model().fit(rows), rows


def _pct(x: float) -> str:
    return f"{x:.1%}"


def page(y: Year, day) -> str:
    """The page for the week ending `day`, as fixed-width text no
    wider than 68."""
    day = pd.Timestamp(day)
    name, c, model, ref = serving(y, day)
    lists = y.lists[y.lists.moment <= day]
    latest = lists.moment.max()
    now = [a for a in y.alerts if a.day <= day]
    week = [a for a in now if a.day > day - WEEK]
    out = [f"Foresight monitoring: the week to {day:%a %d %b %Y}",
           f"Serving {name} ({c.policy}): {len(ref):,} outcomes,"
           f" fitted {policy.fit_day(c, latest):%Y-%m-%d}",
           "In shadow: " + ", ".join(k.name for k in CONTENDERS
                                     if k.name != name), ""]

    out.append("Alerts this week")
    out += [f"  {a.line()}" for a in week] or ["  none"]

    parts = contributions(model, ref)[1].columns
    cols = [k for k in parts if k in ref]
    cohort = lists[lists.moment == latest]
    drift = Baseline(ref, cols).table(cohort).iloc[0]
    moved = alerts.effects(model, ref, cohort)
    sd = watch.score_drift(model.predict_proba(ref),
                           cohort.assign(
                               chance=model.predict_proba(cohort)))
    n = watch.volume(y.rows[y.rows.moment <= day]).loc[latest]
    out += ["", f"The latest list, {latest:%Y-%m-%d}",
            f"  contracts {int(n.contracts)}, usual {n.usual:.0f}",
            f"  chances: average {_pct(sd['mean'].iloc[0])},"
            f" PSI {sd.psi.iloc[0]:.2f} against training",
            "  inputs over PSI 0.25, with the drift's effect on the"
            " average", "  log-odds:"]
    over = drift[drift >= ACT].sort_values(ascending=False)
    out += [f"    {k:<18}{v:>6.2f}{moved.get(k, 0.0):>+8.3f}"
            + ("  acknowledged" if k in ACKNOWLEDGED else "")
            for k, v in over.items()] or ["    none"]

    e = watch.early(lists, day).tail(4)
    out += ["", "Notices: the last four lists whose notices are in",
            f"  {'list':<12}{'notices':>8}{'expected':>10}"
            f"{'limits':>10}"]
    for m, r in e.iterrows():
        flag = "  !" if r.alert else ""
        out.append(f"  {m:%Y-%m-%d}{int(r.noticed):>10}"
                   f"{r.expected:>10.1f}{r.low:>6.0f} to{r.high:>3.0f}"
                   f"{flag}")

    a = watch.arrived(lists, day)
    sl, n_out = watch.slope(lists, day)
    k = known(lists, day)
    out += ["", f"Outcomes on record: {len(a)} lists,"
            f" {n_out:,} contracts"]
    if len(a):
        out += [f"  left {int(a.left.sum())}, expected"
                f" {a.expected.sum():.1f}; calibration slope"
                f" {sl:.2f}",
                f"  average chance {_pct(k.chance.mean())},"
                f" share that left {_pct(k.not_renewed.mean())}",
                f"  leavers in the calls: list {int(a.hits.sum())},"
                f" rule {int(a.rule.sum())}, of {40 * len(a)}"]

    g = y.log[y.log.day <= day]
    g = g[g.day == g.day.max()] if len(g) else g
    out += ["", "The gate, this month (challenger minus champion)"]
    for r in g.itertuples():
        if r.cohorts < policy.LEAST:
            out.append(f"  {r.challenger:<11}{r.cohorts} lists with"
                       " outcomes: too few to judge")
            continue
        p, u = r.precision, r.auc
        verdict = "promoted" if r.promote else "stays in shadow"
        out.append(f"  {r.challenger:<11}{p[0] * 100:+.1f} pts"
                   f" ({p[1] * 100:+.1f} to {p[2] * 100:+.1f}),"
                   f" AUC {u[0]:+.3f}: {verdict}")

    m = y.months[y.months.index <= (day - pd.Timedelta(days=1))
                 .to_period("M") - 1]
    if len(m):
        r = m.iloc[-1]
        out += ["", f"Demand forecast, {m.index[-1]}, one month ahead",
                f"  WAPE {_pct(r.wape)} (limit {_pct(r.limit)}),"
                f" seasonal naive {_pct(r.naive)}",
                f"  80% intervals held {_pct(r.widened)}; over six"
                f" months {_pct(r.coverage_6m)}"]
    return "\n".join(out)


def log_page(y: Year) -> str:
    """Every alert of the year, one line each."""
    return "\n".join(a.line() for a in y.alerts)


def write(y: Year, days, path: Path = REPORT) -> str:
    """The pages for `days` and the year's alert log, as Markdown."""
    parts = ["# Foresight monitoring, 2025", "",
             "Written by `python -m foresight.monitor.report`. Each"
             " page shows only", "what was on record on its day.", ""]
    for d in days:
        parts += [f"## The week to {pd.Timestamp(d):%d %B %Y}", "",
                  "```text", page(y, d), "```", ""]
    parts += ["## Every alert of the year", "", "```text",
              log_page(y), "```", ""]
    text = "\n".join(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--day", default=None)
    args = ap.parse_args(argv)
    y = gather()
    if args.day:
        print(page(y, args.day))
        return
    write(y, ["2025-04-07", "2025-12-31"])
    print(page(y, "2025-12-31"))
    print(f"\nWritten to {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
