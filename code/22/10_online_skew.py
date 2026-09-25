# Chapter 21's skew test pointed at the API's rows: first a scope that
# forgets legacy ids, then online.rows() as written, over every mark;
# then tonight's chances from the API's path against the list's.
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.pipeline import features
from foresight.pipeline.registry import Registry
from foresight.serve import SANDBOX, online, renewal, store

table = pd.read_parquet(TABLE)
ids = table.contract_id


def forgetful(con, accounts):
    """The scope anybody writes first: the accounts' current ids."""
    con.executescript(online.SCOPE)
    con.executemany("INSERT INTO scope VALUES (?)",
                    [(int(a),) for a in set(accounts)])


print(f"{table.moment.nunique()} marks, {len(table):,} contracts\n")
right, online.scope = online.scope, forgetful
found = features.skew(table, online.rows(ids))
online.scope = right
print("A scope that forgets legacy ids, against the training table")
for c, r in found[found.differ > 0].iterrows():
    print(f"  {c:<18}{int(r.differ):>6,} differ"
          f"   e.g. {int(r.example)}")

checked = features.check_skew(table, online.rows(ids))
print(f"\nonline.rows() against the training table: {len(checked)}"
      f" columns,\n  {checked.differ.sum()} differences")

registry = Registry("renewal", SANDBOX / "artifacts")
model, manifest = registry.production()
listed = store.read(store.path(SANDBOX), "SELECT contract_id, chance"
                    " FROM answers WHERE key_account = 0"
                    " ORDER BY contract_id").set_index("contract_id")
rows = online.rows(listed.index)
train = renewal.training_rows(manifest)
live = renewal.explain(model, rows, train).set_index("contract_id")
gap = (live.chance - listed.chance).abs().max()
print(f"\nTonight's {len(listed)} long-tail contracts, by the API's"
      " path"
      f"\n  largest gap from the list's chance: {gap:.1e}")
