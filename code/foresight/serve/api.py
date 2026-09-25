"""
Foresight's API. Chapter 22 writes it.

    make serve        uvicorn foresight.serve.api:app --port 8000

    GET  /health                    what is loaded, and the last run
    POST /renewal                   {"contract_id": 15231}
    POST /triage                    {"body": "..."}
    GET  /forecast/{sku}/{region}   next quarter, with its interval

A renewal answer comes from the first of these that can give one:

    the list    the monthly job's answer, if it scored the contract
    the model   the production model at the contract's mark, on rows
                made by online.rows()
    the rule    Chapter 3's rule, when the model cannot answer

It is never a number made up to fill a gap: if the warehouse cannot be
read the answer is 503, and a key account gets no chance, as on the
list. Each renewal answer carries a Server-Timing header saying where
its milliseconds went. Nothing here writes anywhere.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager, closing, contextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, Request
from fastapi import Path as InPath
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from foresight import __version__
from foresight.config import ML_WAREHOUSE, ROOT
from foresight.explain import facts
from foresight.pipeline.artifact import ArtifactError, check_inputs
from foresight.pipeline.registry import Registry, RegistryError
from foresight.serve import online, renewal, store, today, triage

# Targets, in milliseconds, for the 95th percentile of one renewal
# request as the CRM page sees it. network and headroom are outside
# the service's own measurements.
BUDGET_MS = {"network": 50, "lookup": 10, "list": 10, "fetch": 120,
             "score": 50, "framework": 30, "headroom": 30}
REGIONS = Literal["Midwest", "Northeast", "Southeast", "Southwest",
                  "West"]


# ------------------------------------------------ the contract
class RenewalRequest(BaseModel):
    """One contract, by its id in the warehouse. Nothing else."""
    model_config = ConfigDict(extra="forbid")
    contract_id: int = Field(gt=0, strict=True)


class RenewalAnswer(BaseModel):
    contract_id: int
    account: str
    mark: date
    answered_by: Literal["list", "model", "rule", "key account"]
    chance: float | None = None
    rank: int | None = None
    reasons: list[str] = []
    model: str | None = None
    note: str


class TicketRequest(BaseModel):
    """One support ticket's text, as the customer wrote it."""
    model_config = ConfigDict(extra="forbid")
    body: Annotated[str, StringConstraints(strip_whitespace=True,
                                           min_length=1,
                                           max_length=2000)]


class TriageAnswer(BaseModel):
    priority: Literal["Urgent", "High", "Normal", "Low"]
    category: str | None
    p_urgent: float | None
    confidence: dict[str, float] | None
    answered_by: Literal["model", "model (unsure)", "keywords"]
    model: str | None


class Month(BaseModel):
    month: str
    units: float
    lo: float
    hi: float


class ForecastAnswer(BaseModel):
    sku: str
    region: str
    run: int
    months: list[Month]


class NotFound(Exception):
    """404: there is no such thing to answer about."""


class NotYet(Exception):
    """409: the question cannot be answered for this contract now."""


class Unavailable(Exception):
    """503: the service cannot answer at the moment."""


# ------------------------------------------------ the service
class Stopwatch:
    """Milliseconds spent in each named stage of one request."""

    def __init__(self):
        self.ms: dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.ms[stage] = (self.ms.get(stage, 0.0)
                              + 1000 * (time.perf_counter() - t))

    def header(self) -> str:
        return ", ".join(f"{k};dur={v:.1f}" for k, v in self.ms.items())


class Service:
    """Everything a request needs, loaded once when the API starts."""

    def __init__(self, root: Path = ROOT,
                 warehouse: Path = ML_WAREHOUSE, on=None,
                 fetch=online.rows):
        self.root, self.warehouse = Path(root), Path(warehouse)
        self.on = pd.Timestamp(on) if on is not None else None
        self.fetch = fetch
        self.scores = store.path(self.root)
        self.problems: list[str] = []
        self.renewal = self._load("renewal", self._renewal)
        self.triage = self._load("triage", self._triage)

    def _load(self, name, loader):
        try:
            return loader()
        except (RegistryError, ArtifactError, OSError,
                sqlite3.Error) as e:
            self.problems.append(f"{name}: {e}")
            return None

    def _production(self, name: str):
        registry = Registry(name, self.root / "artifacts")
        model, manifest = registry.production()
        version = next(v for v, r in registry.read()["versions"].items()
                       if r["stage"] == "production")
        return model, manifest, f"{name} v{version}"

    def _renewal(self) -> dict:
        model, manifest, name = self._production("renewal")
        train = renewal.training_rows(manifest, self.warehouse)
        outcomes = renewal.known(manifest, self.today(), self.warehouse)
        model, span = renewal.recalibrate(model, outcomes,
                                          manifest["data"]["as_of"])
        return {"model": model, "manifest": manifest, "name": name,
                "train": train, "typical": facts(train),
                "calibration": "as fitted" if span is None else
                f"refitted to the cohort marked {span[1]:%Y-%m-%d}"}

    def _triage(self) -> dict:
        model, _, name = self._production("triage")
        return {"model": model, "name": name}

    def today(self) -> pd.Timestamp:
        return self.on if self.on is not None else today()

    # -------------------------------------------- renewal
    def contract(self, contract_id: int) -> pd.Series:
        try:
            with closing(online.open_warehouse(self.warehouse)) as con:
                found = online.contracts(con, [contract_id])
        except sqlite3.Error as e:
            raise Unavailable(f"the warehouse cannot be read: {e}")
        if found.empty:
            raise NotFound(f"no contract {contract_id}")
        c = found.iloc[0]
        if c.moment > self.today():
            raise NotYet(f"contract {contract_id} is not at its mark"
                         f" until {c.moment:%Y-%m-%d}")
        if pd.notna(c.outcome):
            raise NotYet(f"contract {contract_id} ended on"
                         f" {c.end_date:%Y-%m-%d}; its outcome is on"
                         " record")
        return c

    def from_list(self, c: pd.Series) -> dict | None:
        """The list of record's answer, if the monthly job made one."""
        if not self.scores.exists():
            return None
        found = store.read(self.scores, "SELECT * FROM answers WHERE"
                           " contract_id = ? AND mark = ?",
                           (int(c.contract_id), f"{c.moment:%Y-%m-%d}"))
        if found.empty:
            return None
        r = found.iloc[0]
        if r.listed and r.arm == "held out":
            note = "on the list, held out this month: no call"
        elif r.listed:
            note = "on the list: call"
        else:
            note = f"not on the list (rank {int(r['rank'])})"
        return {"answered_by": "list", "chance": round(r.chance, 4),
                "rank": int(r["rank"]), "model": r.source,
                "reasons": json.loads(r.reasons),
                "note": f"{note}; run {int(r.run)}"}

    def score(self, contract_id: int, clock: Stopwatch) -> dict:
        with clock("lookup"):
            c = self.contract(contract_id)
        out = {"contract_id": contract_id, "account": c["name"],
               "mark": c.moment.date()}
        if c.is_key_account == 1:
            return {**out, "answered_by": "key account",
                    "note": "no chance given; ask the account"
                            " manager"}
        with clock("list"):
            listed = self.from_list(c)
        if listed is not None:
            return {**out, **listed}
        with clock("fetch"):
            try:
                rows = self.fetch([contract_id], self.warehouse)
            except (sqlite3.Error, OSError) as e:
                raise Unavailable(f"the warehouse cannot be read: {e}")
        with clock("score"):
            answer, why = self.by_model(rows)
            if answer is None:
                answer = self.by_rule(rows, why)
        return {**out, **answer}

    def by_model(self, rows: pd.DataFrame):
        """(answer, None), or (None, why the model could not)."""
        if self.renewal is None:
            return None, "no renewal model is loaded"
        m = self.renewal
        try:
            check_inputs(rows, m["manifest"])
            s = renewal.explain(m["model"], rows, m["train"],
                                m["typical"]).iloc[0]
        except (ValueError, KeyError, ArtifactError) as e:
            return None, f"{type(e).__name__}: {e}"
        return {"answered_by": "model", "chance": round(s.chance, 4),
                "reasons": list(s.reasons), "model": m["name"],
                "note": "scored now: no monthly list holds it"}, None

    def by_rule(self, rows: pd.DataFrame, why: str) -> dict:
        gap = rows.days_since_order.iloc[0]
        said = ("No order on record" if pd.isna(gap)
                else f"Last order {int(gap)} days ago")
        return {"answered_by": "rule", "reasons": [said],
                "note": f"the model could not answer ({why});"
                        " Chapter 3's rule gives the gap alone"}

    # -------------------------------------------- the others
    def classify(self, body: str) -> dict:
        if self.triage is not None:
            try:
                return {**triage.answer(self.triage["model"], body),
                        "model": self.triage["name"]}
            except (ValueError, KeyError):
                pass
        return {**triage.keywords(body), "model": None}

    def forecast(self, sku: str, region: str) -> dict:
        if not self.scores.exists():
            raise NotFound("no forecast has been made")
        rows = store.read(self.scores, """
            SELECT f.* FROM forecasts f
            WHERE f.sku = ? AND f.region = ? AND f.run = (
                SELECT MAX(run) FROM runs
                WHERE job = 'forecast' AND status = 'finished')
            ORDER BY f.h""", (sku, region))
        if rows.empty:
            raise NotFound(f"no forecast for {sku} in the {region}")
        return {"sku": sku, "region": region,
                "run": int(rows.run.iloc[0]),
                "months": [{"month": r.target, "units": round(r.units),
                            "lo": round(r.lo), "hi": round(r.hi)}
                           for r in rows.itertuples()]}

    def health(self) -> dict:
        last = None
        if self.scores.exists():
            runs = store.read(self.scores, "SELECT run, mark, status,"
                              " step FROM runs WHERE job = 'renewal'"
                              " ORDER BY run DESC LIMIT 1")
            if len(runs):
                last = runs.iloc[0].to_dict()
                last["run"] = int(last["run"])
        return {"status": "degraded" if self.problems else "ok",
                "version": __version__,
                "today": f"{self.today():%Y-%m-%d}",
                "renewal": self.renewal and self.renewal["name"],
                "calibration": self.renewal
                and self.renewal["calibration"],
                "triage": self.triage and self.triage["name"],
                "last_run": last, "problems": self.problems}


# ------------------------------------------------ the app
def create_app(service: Service | None = None) -> FastAPI:
    """The API around `service`, or around one loaded from the project
    (or FORESIGHT_ROOT) when the server starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "service"):
            root = Path(os.getenv("FORESIGHT_ROOT", ROOT))
            app.state.service = Service(root)
        yield

    app = FastAPI(title="Foresight", version=__version__,
                  lifespan=lifespan)
    if service is not None:
        app.state.service = service

    for error, status in [(NotFound, 404), (NotYet, 409),
                          (Unavailable, 503)]:
        def handler(request, e, status=status):
            return JSONResponse({"detail": str(e)}, status_code=status)
        app.add_exception_handler(error, handler)

    @app.get("/health")
    def health(request: Request) -> dict:
        return request.app.state.service.health()

    @app.post("/renewal", response_model=RenewalAnswer)
    def score(req: RenewalRequest, request: Request):
        clock = Stopwatch()
        answer = request.app.state.service.score(req.contract_id, clock)
        body = RenewalAnswer(**answer).model_dump(mode="json")
        return JSONResponse(body,
                            headers={"Server-Timing": clock.header()})

    @app.post("/triage", response_model=TriageAnswer)
    def classify(ticket: TicketRequest, request: Request):
        return request.app.state.service.classify(ticket.body)

    @app.get("/forecast/{sku}/{region}", response_model=ForecastAnswer)
    def forecast(sku: Annotated[str, InPath(
                     pattern=r"^MRD-[A-Z]{3}-\d{3}$")],
                 region: REGIONS, request: Request):
        return request.app.state.service.forecast(sku, region)

    return app


app = create_app()
