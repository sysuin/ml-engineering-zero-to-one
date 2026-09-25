# The forecast served the way the list is: made by a monthly run into
# the scores database, then read by the API. The run is made twice.
# timeout: 300
import shutil
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*httpx.*")   # a notice
from fastapi.testclient import TestClient                # noqa: E402

from foresight.serve import SANDBOX, batch, store        # noqa: E402
from foresight.serve.api import Service, create_app      # noqa: E402

root = Path(tempfile.mkdtemp())
shutil.copytree(SANDBOX / "artifacts", root / "artifacts")
store.path(root).parent.mkdir(parents=True)
shutil.copy(store.path(SANDBOX), store.path(root))

for _ in range(2):
    r = batch.forecast("2026-01-01", root=root)
    print(f"forecast from {r['mark']:%Y-%m}: run {r['run']},"
          f" {r['status']}")
print(store.read(store.path(root), "SELECT job, COUNT(*) AS runs,"
                 " SUM(rows) AS rows FROM runs GROUP BY job")
      .to_string(index=False))

client = TestClient(create_app(Service(root, on="2026-01-01")))
f = client.get("/forecast/MRD-CLE-001/Midwest").json()
print(f"\nGET /forecast/MRD-CLE-001/Midwest (run {f['run']})")
for m in f["months"]:
    print(f"  {m['month']}  {m['units']:>6,.0f} units,"
          f" 80% between {m['lo']:,.0f} and {m['hi']:,.0f}")
missing = client.get("/forecast/MRD-CLE-999/Midwest")
print(f"GET /forecast/MRD-CLE-999/Midwest -> {missing.status_code}\n"
      f"  {missing.json()['detail']}")
shutil.rmtree(root)
