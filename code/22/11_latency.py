# nondeterministic: timing
# Where a renewal request's milliseconds go, stage by stage, against
# the budget; answers from the list; and at_mark() as the fetch.
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*httpx.*")   # a notice
from fastapi.testclient import TestClient                # noqa: E402

from foresight.pipeline import features                  # noqa: E402
from foresight.serve import SANDBOX, store               # noqa: E402
from foresight.serve.api import (BUDGET_MS, Service,     # noqa: E402
                                 create_app)


def timed(client, ids):
    """Each request's stages, from Server-Timing, and its total."""
    out = []
    for cid in ids:
        t = time.perf_counter()
        r = client.post("/renewal", json={"contract_id": int(cid)})
        total = 1000 * (time.perf_counter() - t)
        parts = dict(p.split(";dur=")
                     for p in r.headers["server-timing"].split(", "))
        out.append({**{k: float(v) for k, v in parts.items()},
                    "total": total})
    return pd.DataFrame(out)


service = Service(SANDBOX, on="2025-12-31")
client = TestClient(create_app(service))
marks = features.at_marks(["2025-11-02", "2025-11-30"])
live = [c for c in marks.contract_id
        if not service.contract(int(c)).is_key_account][:200]
listed = store.read(store.path(SANDBOX), "SELECT contract_id FROM"
                    " answers WHERE key_account = 0").contract_id

t = timed(client, live)
t["framework"] = t.total - t[["lookup", "list", "fetch",
                              "score"]].sum(axis=1)
budget = {k: v for k, v in BUDGET_MS.items()
          if k not in ("network", "headroom")}      # the service's
budget["total"] = sum(budget.values())
print(f"{len(t)} requests scored now, milliseconds")
print(f"  {'stage':<11}{'budget':>7}{'median':>8}{'95th':>7}")
for stage, b in budget.items():
    q = np.percentile(t[stage], [50, 95])
    print(f"  {stage:<11}{b:>7}{q[0]:>8.1f}{q[1]:>7.1f}")

s = timed(client, listed[:200])
q = np.percentile(s.total, [50, 95])
print(f"\n{len(s)} answered from the list: median {q[0]:.1f},"
      f" 95th {q[1]:.1f}")

slow = Service(SANDBOX, on="2025-12-31",
               fetch=lambda ids, wh: features.at_marks(
                   marks[marks.contract_id.isin(ids)].moment.unique(),
                   wh).query("contract_id in @ids"))
u = timed(TestClient(create_app(slow)), live[:5])
print(f"5 with at_marks() as the fetch: median {u.total.median():.0f}")
