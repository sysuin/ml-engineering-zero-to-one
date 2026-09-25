# A registry in a folder: December's model registered and promoted,
# January's retrain registered and promoted over it, then rolled back.
import json
import tempfile

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, backtest, known_by
from foresight.pipeline import evaluation
from foresight.pipeline.model import build, maker
from foresight.pipeline.registry import Registry, RegistryError

table = pd.read_parquet(TABLE)
history = table[table.end_date >= HISTORY_FROM]


def trained(as_of: str):
    """A fitted model and the manifest fields the registry reads."""
    rows = known_by(history, as_of)
    ends = sorted(rows.end_date.unique())[-6:]
    scored = backtest(rows, f"{ends[0]:%F}", f"{ends[-1]:%F}",
                      maker(build()))
    manifest = {"run": f"by hand, {as_of}", "data": {"as_of": as_of},
                "metrics": evaluation.scores(scored)}
    return build().fit(rows, rows.not_renewed), manifest


with tempfile.TemporaryDirectory() as tmp:
    reg = Registry("renewal", root=tmp)
    v1 = reg.register(*trained("2024-12-01"))
    reg.promote(v1, "December's list")
    v2 = reg.register(*trained("2025-01-01"))
    reg.promote(v2, "January's retrain; checks passed")
    back = reg.rollback("account team queried half the list")
    model, manifest = reg.production()
    print(reg.page())
    print(f"\nin production: version {back}, fitted on the outcomes"
          f" known on {manifest['data']['as_of']}")
    try:
        reg.promote(v2, "try again")
    except RegistryError as e:
        print(f"promote version {v2} again: refused, {e}")
    state = reg.read()
    for e in state["log"]:
        del e["when"]                   # the figure needs no clock
    with open("code/21/06_registry.json", "w") as f:
        json.dump(state, f)
    kept = sorted(p.name for p in reg.folder.iterdir())
    print(f"in the folder: {', '.join(kept)}")
