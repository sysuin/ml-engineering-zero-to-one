"""
The enriched table: Chapter 4's rows with every feature in the library
beside them. Chapter 12 writes it.

    python -m foresight.features.build        build, or reuse the cache

build() computes the features for any rows of the table. load() does
the same for the whole table and keeps the result in
data/foresight/features_table.parquet, stamped with a hash of the
table, the warehouse and the library's own source; when any of them
changes, the stamp no longer matches and the table is rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import FORESIGHT_DATA, TABLE
from foresight.features import definitions  # noqa: F401  registers
from foresight.features.registry import REGISTRY, names
from foresight.features.sources import Context, load_sources

ENRICHED = FORESIGHT_DATA / "features_table.parquet"
LIBRARY = Path(__file__).parent


def build(table: pd.DataFrame, features=None, sources=None
          ) -> pd.DataFrame:
    """`table` with one column per feature (all of them by default)."""
    sources = load_sources() if sources is None else sources
    ctx = Context(table, sources)
    out = ctx.keys.copy()
    for name in names() if features is None else features:
        values = REGISTRY[name].compute(ctx)
        if len(values) != len(out) or values.isna().any():
            raise ValueError(f"{name}: one number per contract needed")
        out[name] = values.to_numpy(dtype=float)
    return out


def _hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def stamp(table_path: Path = TABLE,
          warehouse: Path = ML_WAREHOUSE) -> str:
    """What the enriched table was built from, as one hash."""
    code = sorted(LIBRARY.glob("*.py"))
    return _hash(table_path, warehouse, *code)


def load(path: Path = ENRICHED) -> pd.DataFrame:
    """The whole table with every feature, rebuilt only if stale."""
    meta = path.with_suffix(".json")
    now = stamp()
    if path.exists() and meta.exists():
        if json.loads(meta.read_text()).get("stamp") == now:
            return pd.read_parquet(path)
    table = build(pd.read_parquet(TABLE))
    table.to_parquet(path, index=False)
    meta.write_text(json.dumps({"stamp": now, "rows": len(table),
                                "features": names()}, indent=1) + "\n")
    return table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    table = load()
    print(f"{len(table):,} rows, {len(names())} features: {ENRICHED}")


if __name__ == "__main__":
    main()
