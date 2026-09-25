"""
Every training run, recorded: the settings it was given, the rows it
read, the code that ran, what it scored, and the model it made.
Chapter 15 writes it.

    python -m foresight.tracking        the runs logged so far

A run is recorded twice. RunLog is the hand-built record, a JSON line
per run appended to a file that is never rewritten; Chapter 15 builds
it first to show that tracking is no more than this. log_mlflow()
writes the same record to MLflow, which adds a browsable store, the
model as an artifact, and a way to compare runs. The store is local,
under mlruns/, with no server: one SQLite file for the records and a
folder for the artifacts. mlruns/ is in .gitignore.

A record's run id is a hash of everything in it except the time, so
two runs that were given the same settings, read the same rows, ran
the same code in the same environment and scored the same have the
same id. A repeat that gets a different id has found a difference.

track() is the one call a training command makes: it builds the
record, appends it to the log and writes it to MLflow with the model.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import pickle
import platform
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from foresight.config import ROOT, SEED, THREADS
from foresight.data.build_table import TABLE

MLRUNS = ROOT / "mlruns"
RUN_LOG = ROOT / "mlruns" / "runs.jsonl"
EXPERIMENT = "foresight-renewals"
LIBRARIES = ("numpy", "pandas", "scikit-learn", "lightgbm", "optuna",
             "mlflow")


# ------------------------------------------------ the hand-built log
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows_hash(rows: pd.DataFrame) -> str:
    """A fingerprint of every value in the rows: Chapter 4's content
    hash when the rows are the whole table."""
    values = pd.util.hash_pandas_object(rows, index=False)
    return sha256(values.to_numpy().tobytes())


def code_hash(paths) -> str:
    """One hash over the named source files, name and contents."""
    digest = hashlib.sha256()
    for p in sorted(Path(p) for p in paths):
        digest.update(p.name.encode() + p.read_bytes())
    return digest.hexdigest()


class RunLog:
    """An append-only file of runs, one JSON object a line."""

    def __init__(self, path):
        self.path = Path(path)

    def append(self, record: dict) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def runs(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in
                self.path.read_text().splitlines() if line]


def record(name: str, params: dict, data: dict, code: dict,
           metrics: dict, **extra) -> dict:
    """A run as a dictionary. Its id hashes everything but the time."""
    body = {"name": name, "params": params, "data": data,
            "code": code, "metrics": metrics, **extra}
    run = sha256(json.dumps(body, sort_keys=True).encode())[:12]
    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {"run": run, "when": when, **body}


# ------------------------------------------------ what it read and ran
def table_version(table: pd.DataFrame | None = None,
                  path: Path = TABLE) -> dict:
    """The table's content hash, recomputed from its rows, and whether
    it matches the manifest Chapter 4's builder wrote beside it."""
    table = pd.read_parquet(path) if table is None else table
    manifest = json.loads(
        path.with_suffix(".manifest.json").read_text())
    content = rows_hash(table)
    return {"table": path.name, "content_sha256": content,
            "matches_manifest":
                content == manifest["content_sha256"]}


def data_record(rows: pd.DataFrame, table: pd.DataFrame | None = None,
                path: Path = TABLE) -> dict:
    """Which table, and which of its rows, a model was fitted on."""
    return {**table_version(table, path), "rows": len(rows),
            "rows_sha256": rows_hash(rows)}


def source_files(model) -> list[Path]:
    """The files that define a model: every Foresight module among its
    classes, and the configuration that holds the seeds."""
    package = ROOT / "code" / "foresight"
    files = {Path(inspect.getsourcefile(c)).resolve()
             for c in type(model).__mro__ if c is not object}
    files = {f for f in files if f.is_relative_to(package)}
    return sorted(files | {package / "config.py"})


def environment() -> dict:
    """What the code ran on: Python, the libraries, threads, seed."""
    out = {"python": platform.python_version(),
           "system": platform.system(), "threads": THREADS,
           "seed": SEED}
    for lib in LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


def code_record(model, extra=()) -> dict:
    files = [*source_files(model), *map(Path, extra)]
    return {"sha256": code_hash(files),
            "files": sorted(str(p.relative_to(ROOT / "code"))
                            if p.is_relative_to(ROOT / "code")
                            else p.name for p in files),
            "environment": environment()}


# ------------------------------------------------ MLflow
def quiet():
    """Import MLflow without its announcements: a hint on import, and a
    line for every database table it creates. None of it is output a
    reader needs. The levels are set after the import, which sets its
    own."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    for name in ("mlflow", "alembic"):
        logging.getLogger(name).setLevel(logging.ERROR)
    return mlflow


def open_store(store=MLRUNS, experiment: str = EXPERIMENT):
    """Point MLflow at a local store and make the experiment current."""
    mlflow = quiet()
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment,
            artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


def flat(d: dict, prefix: str = "") -> dict:
    """Nested keys joined by dots: code.environment.python."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        elif isinstance(v, list):
            out[f"{prefix}{k}"] = ",".join(map(str, v))
        else:
            out[f"{prefix}{k}"] = v
    return out


def log_mlflow(rec: dict, model=None, store=MLRUNS,
               experiment: str = EXPERIMENT) -> str:
    """Write a record to MLflow as one run: settings as parameters,
    scores as metrics, data and code as tags, the model and the whole
    record as artifacts. Returns MLflow's own id for the run."""
    mlflow = open_store(store, experiment)
    metrics = {k.replace(" ", "_"): float(v)
               for k, v in rec["metrics"].items()}
    tags = {k: str(v) for k, v in flat(
        {"data": rec["data"], "code": rec["code"]}).items()}
    with mlflow.start_run(run_name=rec["name"]) as run:
        mlflow.log_params(rec["params"])
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"run": rec["run"], **tags})
        mlflow.log_dict(rec, "record.json")
        if model is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pkl"
                path.write_bytes(pickle.dumps(model))
                mlflow.log_artifact(str(path))
        return run.info.run_id


def runs(store=MLRUNS, experiment: str = EXPERIMENT) -> pd.DataFrame:
    """Every run in the experiment, oldest first, as MLflow returns
    them: one row a run, columns params.*, metrics.*, tags.*."""
    mlflow = open_store(store, experiment)
    found = mlflow.search_runs(experiment_names=[experiment],
                               order_by=["attributes.start_time ASC"])
    return found.reset_index(drop=True)


# ------------------------------------------------ the one call
def track(name: str, model, params: dict, rows: pd.DataFrame,
          table: pd.DataFrame | None, metrics: dict, store=MLRUNS,
          log=RUN_LOG, extra_code=(), table_path: Path = TABLE,
          **extra) -> dict:
    """Record a fitted model's run in the log and in MLflow. `rows` are
    the rows it was fitted on, `table` the table they came from, read
    from `table_path`. With store=None, the log alone."""
    rec = record(name, params, data_record(rows, table, table_path),
                 code_record(model, extra_code), metrics, **extra)
    RunLog(log).append(rec)
    if store is not None:
        log_mlflow(rec, model, store)
    return rec


def main() -> None:
    for r in RunLog(RUN_LOG).runs():
        m = r["metrics"]
        print(f"{r['run']}  {r['when']}  {r['name']:<20}"
              f"  log loss {m.get('log loss', float('nan')):.5f}")


if __name__ == "__main__":
    main()
