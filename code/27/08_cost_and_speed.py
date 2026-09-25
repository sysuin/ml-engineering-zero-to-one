# The numbers card, second half: how fast each part is, and what a
# month of it costs in core-hours, by Appendix C's method: time a job
# once, multiply by how often it runs.
# nondeterministic: timing
# timeout: 400
import shutil
import sqlite3
import statistics
import time
import warnings

warnings.filterwarnings("ignore", message=".*httpx.*")   # a notice
from fastapi.testclient import TestClient                # noqa: E402

from foresight import demo                               # noqa: E402
from foresight import mcp_server as mcp                  # noqa: E402
from foresight.config import ML_WAREHOUSE                # noqa: E402
from foresight.monitor import feeds                      # noqa: E402
from foresight.pipeline.run import run                   # noqa: E402
from foresight.serve import batch                        # noqa: E402
from foresight.serve.api import Service, create_app      # noqa: E402


def timed(job, *args, **kw):
    start = time.perf_counter()
    job(*args, **kw)
    return time.perf_counter() - start


root = demo.workspace()
s = {"list": timed(batch.month, "2026-01-01", root=root),
     "nothing due": timed(batch.month, "2026-01-02", root=root),
     "forecast": timed(batch.forecast, "2026-01-01", root=root),
     "feed check": timed(feeds.daily),
     "retrain": timed(run, "renewal", ["data.as_of=2025-12-31"],
                      root=root, say=lambda *lines: None)}

client = TestClient(create_app(Service(root, on="2026-01-01")))
listed, fresh = [], []
for cid in (15378, 15330, 15308, 15415, 15256) * 10:  # on the list
    listed.append(timed(client.post, "/renewal",
                        json={"contract_id": cid}))
with sqlite3.connect(ML_WAREHOUSE) as con:    # November's cohort: no
    unlisted = [c for (c,) in con.execute(      # list holds it
        "SELECT contract_id FROM contracts JOIN accounts USING"
        " (account_id) WHERE end_date = '2026-02-28'"
        " AND is_key_account = 0 ORDER BY contract_id LIMIT 10")]
for cid in unlisted:                                    # scored now
    fresh.append(timed(client.post, "/renewal",
                       json={"contract_id": cid}))
server = mcp.build(root, on="2026-01-01")
asks = [timed(mcp.exchange, server, [mcp.frame(
    1, "tools/call", name="renewal_risk",
    arguments={"region": "Midwest"})]) for _ in range(5)]
shutil.rmtree(root)

print("How fast, on one core of the book's laptop")
print(f"  the monthly list                 {s['list']:6.1f} s")
print(f"  a night with nothing due         {s['nothing due']:6.2f} s")
print(f"  next quarter's forecast          {s['forecast']:6.1f} s")
print(f"  a retrain, checks and all        {s['retrain']:6.1f} s")
print(f"  /renewal, from the list    "
      f"{1000 * statistics.median(listed):6.0f} ms median")
print(f"  /renewal, scored now       "
      f"{1000 * statistics.median(fresh):6.0f} ms median")
print(f"  renewal_risk over MCP      "
      f"{1000 * statistics.median(asks):6.0f} ms median")

HOURS = 24 * 365 / 12
month = [  # what, core-hours a month
    ("the job, 30 nights", (29 * s["nothing due"] + s["list"]) / 3600),
    ("the forecast, once", s["forecast"] / 3600),
    ("the feed check, daily", 30 * s["feed check"] / 3600),
    ("retrains: 3 challengers", 3 * s["retrain"] / 3600),
    ("the API, always on", HOURS),
    ("the MCP server", 0.0)]
print("\nCore-hours a month")
for what, hours in month:
    print(f"  {what:<26}{hours:>10.3f}")
batch_only = sum(h for w, h in month if "API" not in w)
print(f"  {'all of it':<26}{sum(h for _, h in month):>10.3f}")
print(f"  {'without the API':<26}{batch_only:>10.3f}")
print("The MCP server runs inside the host that starts it: its time"
      "\nis the user's, not the service's.")
