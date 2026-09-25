"""
Foresight v1.0 in ten minutes: every part, run once, in order. Chapter
27 writes it.

    make demo                 python -m foresight.demo
    python -m foresight.demo --on 2026-01-01

It works, then it fails correctly, then it shows the evidence:

    1 list      the monthly job makes the list due this morning, then
                does nothing when it runs again
    2 forecast  the forecast job writes next quarter
    3 api       the service answers for one contract and one ticket,
                and refuses a request that breaks its contract
    4 monitor   tonight's list against the rows the model learned
                from, and the order feed
    5 mcp       the server Clarity calls, asked about the Midwest
    6 failures  a stale warehouse, a contract not yet due, an empty
                registry, a region that does not exist
    7 evidence  the run record, the holdout, and the rows the model
                saw, scored again

Everything happens in a temporary installation holding a copy of the
sandbox's registry (Chapter 22), so the demonstration never touches
the project's own scores, holdout or models. If the sandbox has no
models, the demonstration trains and promotes them there first, which
adds a minute. It prints no path, no time and no run id it did not
make itself, so two runs print the same thing.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from foresight import __version__
from foresight.config import ML_WAREHOUSE
from foresight.serve import SANDBOX, batch, store

MORNING = "2026-01-01"
WIDTH = 68
MODELS = ("renewal", "triage")


@dataclass
class Demo:
    root: Path
    on: pd.Timestamp
    lines: list[str] = field(default_factory=list)
    run: dict = field(default_factory=dict)
    service: object = None
    server: object = None

    def say(self, *lines: str, indent: int = 3) -> None:
        for line in lines:
            for part in textwrap.wrap(line, WIDTH - indent) or [""]:
                self.lines.append(" " * indent + part)
                print(" " * indent + part)

    def heading(self, n: int, title: str, command: str) -> None:
        text = f"{n}  {title}"
        print(f"\n{text}{command:>{WIDTH - len(text)}}".rstrip())


# ------------------------------------------------ the installation
def ready(folder: Path) -> bool:
    """True if `folder` holds a registry with both models live."""
    return all((folder / "artifacts" / m / "registry.json").exists()
               for m in MODELS)


def workspace(sandbox: Path = SANDBOX) -> Path:
    """A new, temporary installation with the sandbox's registry, or
    with models trained for it if the sandbox has none."""
    root = Path(tempfile.mkdtemp(prefix="foresight-demo-"))
    if ready(sandbox):
        shutil.copytree(sandbox / "artifacts", root / "artifacts")
    else:
        train(root)
    return root


def train(root: Path) -> None:
    """Chapter 22's sandbox, made from nothing: the training job on
    the last morning of 2025 and the triage model, both promoted."""
    from foresight.pipeline.registry import Registry
    from foresight.pipeline.run import run
    from foresight.serve import triage
    done = run("renewal", ["data.as_of=2025-12-31"], root=root,
               say=lambda *lines: None)
    Registry("renewal", root / "artifacts").promote(
        done["version"], "the demonstration")
    Registry("triage", root / "artifacts").promote(
        triage.train(root), "the demonstration")


# ------------------------------------------------ the beats
def the_list(d: Demo) -> None:
    d.heading(1, "The monthly list", "make score")
    r = batch.month(d.on, root=d.root)
    d.run = r
    s = r["scored"]
    tail = s[s.is_key_account == 0]
    arms = s.arm.value_counts()
    d.say(f"cohort marked {r['mark']:%Y-%m-%d}: {len(s)} contracts,"
          f" {int(tail.listed.sum())} calls,"
          f" {arms.get('held out', 0)} held out at random;"
          f" {int((s.is_key_account == 1).sum())} key accounts"
          " listed apart")
    top = tail[tail.listed].sort_values("rank").head(3)
    for x in top.itertuples():
        said = x.reasons[0].split(" (")[0]
        d.say(f"{x.rank:>2}  {x.name:<20} {x.chance:>6.1%}  {said}")
    again = batch.month(d.on + pd.Timedelta(days=1), root=d.root)
    d.say(f"the next night: {again['status']}, list made by run"
          f" {again['run']}")


def the_forecast(d: Demo) -> None:
    d.heading(2, "Next quarter's demand", "make score ARGS=forecast")
    r = batch.forecast(d.on, root=d.root)
    d.say(f"from {r['mark']:%Y-%m}: {r['rows']} product-region"
          " months written, each with an 80% range")


def the_api(d: Demo) -> None:
    d.heading(3, "The API", "make serve")
    warnings.filterwarnings("ignore", message=".*httpx.*")
    from fastapi.testclient import TestClient

    from foresight.serve.api import Service, create_app
    d.service = Service(d.root, on=d.on)
    client = TestClient(create_app(d.service))
    h = client.get("/health").json()
    d.say(f"GET /health: {h['status']}; {h['renewal']},"
          f" {h['triage']}; Platt's map {h['calibration']}")
    first = int(d.run["scored"].query("rank == 1").contract_id.iloc[0])
    a = client.post("/renewal", json={"contract_id": first}).json()
    d.say(f"POST /renewal {first}: {a['answered_by']},"
          f" {a['chance']:.1%}; {a['note']}")
    bad = client.post("/renewal", json={"contract_id": str(first)})
    d.say(f"POST /renewal \"{first}\": {bad.status_code},"
          f" {bad.json()['detail'][0]['msg'].lower()}")
    t = client.post("/triage", json={
        "body": "hi, only half of order 1342207 turned up and the"
                " production line is down"}).json()
    d.say(f"POST /triage: {t['priority']}, {t['category']}"
          f" ({t['answered_by']})")
    f = client.get("/forecast/MRD-CLE-001/Midwest").json()
    m = f["months"][0]
    d.say(f"GET /forecast/MRD-CLE-001/Midwest: {m['month']},"
          f" {m['units']:,.0f} units ({m['lo']:,.0f} to"
          f" {m['hi']:,.0f})")


def the_monitor(d: Demo) -> None:
    d.heading(4, "The monitor", "the weekly page")
    from foresight.monitor import feeds
    from foresight.monitor.psi import ACT, Baseline
    from foresight.monitor.watch import score_drift
    from foresight.pipeline.model import INPUTS
    from foresight.serve import renewal
    m = d.service.renewal
    model, train = m["model"], m["train"]
    rows = store.inputs(store.path(d.root), d.run["run"], m["manifest"])
    reference = model.predict_proba(train)[:, 1]
    tonight = rows.assign(chance=model.predict_proba(rows)[:, 1])
    drift = score_drift(reference, tonight).iloc[0]
    d.say(f"chances: average {drift['mean']:.1%}, PSI"
          f" {drift.psi:.2f} against the {len(train):,} rows the"
          " model learned from")
    base = Baseline(train, INPUTS)
    effect = (renewal.contributions(model, rows)[1].mean()
              - renewal.contributions(model, train)[1].mean())
    moved = [(c, base.column(rows, c)) for c in INPUTS]
    moved = [(c, p) for c, p in moved if p >= ACT]
    d.say(f"inputs over a PSI of {ACT}, and how far they move the"
          " average log-odds:")
    for c, p in sorted(moved, key=lambda x: -x[1]):
        d.say(f"{c:<16}{p:>6.2f}{effect.get(c, 0.0):>+9.3f}",
              indent=5)
    fed = feeds.alerts(feeds.daily(ML_WAREHOUSE), last=d.on)
    days = fed.groupby("day").supplier.first()
    for day, who in days.tail(2).items():
        what = fed[fed.day == day].what.iloc[0]
        d.say(f"order feed: {what} alert {day:%Y-%m-%d},"
              f" {who.split(' / ')[0]}")
    notice = d.run["mark"] + pd.Timedelta(days=30)
    outcome = d.run["mark"] + pd.Timedelta(days=91)
    d.say(f"this list's notices are due {notice:%Y-%m-%d}, its"
          f" outcomes on {outcome:%Y-%m-%d}")


def the_mcp(d: Demo) -> None:
    d.heading(5, "The MCP server", "python -m foresight.mcp_server")
    from foresight import mcp_server as mcp
    d.server = mcp.build(d.root, on=d.on, service=d.service)
    listed, asked = mcp.exchange(d.server, [
        mcp.frame(1, "tools/list"),
        mcp.frame(2, "tools/call", name="renewal_risk",
                  arguments={"region": "Midwest"})])
    names = [t["name"] for t in listed["result"]["tools"]]
    d.say(f"tools/list: {', '.join(names)}")
    got = mcp.content(asked)["lists"][0]
    held = sum(c["call"].startswith("held") for c in got["at_risk"])
    d.say(f"renewal_risk(region=\"Midwest\"): {got['calls']} of the"
          f" {got['of']} calls, {held} of them held out; first,"
          f" {got['at_risk'][0]['account']}"
          f" ({got['at_risk'][0]['chance']:.1%}),"
          f" {got['at_risk'][0]['reasons'][0].lower()}")


def the_failures(d: Demo) -> None:
    d.heading(6, "It fails correctly", "")
    from fastapi.testclient import TestClient

    from foresight import mcp_server as mcp
    from foresight.serve.api import Service, Stopwatch, create_app
    late = d.on + pd.Timedelta(days=29)
    try:
        batch.month(late, root=d.root)
    except batch.StaleData as e:
        d.say(f"the job on {late:%Y-%m-%d}: refused, "
              + str(e).replace("\n", " "))
    client = TestClient(create_app(d.service))
    r = client.post("/renewal", json={"contract_id": 16000})
    d.say(f"POST /renewal 16000: {r.status_code},"
          f" {r.json()['detail']}")
    empty = Path(tempfile.mkdtemp(prefix="foresight-empty-"))
    try:
        first = d.run["scored"].query("rank == 1").contract_id
        a = Service(empty, on=d.on).score(int(first.iloc[0]),
                                          Stopwatch())
        d.say(f"no model in the registry: answered by the"
              f" {a['answered_by']}, \"{a['reasons'][0]}\"")
    finally:
        shutil.rmtree(empty, ignore_errors=True)
    (bad,) = mcp.exchange(d.server, [mcp.frame(
        1, "tools/call", name="renewal_risk",
        arguments={"region": "Mid-west"})])
    d.say("renewal_risk(region=\"Mid-west\"): refused by the schema:"
          " " + mcp.content(bad).split("\n")[2].split(" [")[0].strip())


def the_evidence(d: Demo) -> None:
    d.heading(7, "The evidence", "")
    from foresight.pipeline.registry import Registry
    scores = store.path(d.root)
    runs = store.read(scores, "SELECT job, mark, status, step FROM"
                      " runs ORDER BY run")
    for r in runs.itertuples():
        d.say(f"run: {r.job:<9}{r.mark}  {r.status:<9}at {r.step}")
    held = pd.read_csv(scores.parent / "holdout.csv")
    d.say(f"holdout.csv: {len(held)} contracts,"
          f" {(held.arm == 'held out').sum()} held out, drawn once")
    registry = Registry("renewal", d.root / "artifacts")
    model, manifest = registry.production()
    version = next(v for v, r in registry.read()["versions"].items()
                   if r["stage"] == "production")
    kept = store.inputs(scores, d.run["run"], manifest)
    stored = store.read(scores, "SELECT contract_id, chance FROM"
                        " scores WHERE run = ? AND key_account = 0"
                        " ORDER BY contract_id", (d.run["run"],))
    again = model.predict_proba(
        kept[kept.contract_id.isin(stored.contract_id)])[:, 1]
    d.say(f"the {len(kept)} rows the list scored, scored again:"
          f" largest change {abs(again - stored.chance).max():.1e}")
    card = registry.folder / version / "model_card.md"
    d.say(f"renewal v{version}'s model card, beside it in the"
          f" registry: {len(card.read_text().splitlines())} lines")


BEATS = [the_list, the_forecast, the_api, the_monitor, the_mcp,
         the_failures, the_evidence]


def main(argv=None) -> Demo:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--on", default=MORNING,
                    help="the morning, YYYY-MM-DD")
    a = ap.parse_args(argv)
    root = workspace()
    d = Demo(root, pd.Timestamp(a.on))
    print(f"Foresight v{__version__}: the demonstration, on the"
          f" morning of {d.on:%Y-%m-%d}")
    print("in a temporary installation with the sandbox's models")
    try:
        for beat in BEATS:
            beat(d)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return d


if __name__ == "__main__":
    main()
