# The API, called in-process with FastAPI's test client: its health,
# four renewal questions it can answer and three it cannot, and one
# ticket for triage.
import json
import warnings

warnings.filterwarnings("ignore", message=".*httpx.*")   # a notice
from fastapi.testclient import TestClient                # noqa: E402

from foresight.serve import SANDBOX                      # noqa: E402
from foresight.serve.api import Service, create_app      # noqa: E402

service = Service(SANDBOX, on="2025-12-31")
client = TestClient(create_app(service))
health = client.get("/health").json()
last = health.pop("last_run")
for k, v in health.items():
    print(f"{k:<10}{v}")
print(f"{'last_run':<10}run {last['run']}, {last['mark']},"
      f" {last['status']}")

r = client.post("/renewal", json={"contract_id": 14942})
answer = r.json()
print(f"\nPOST /renewal 14942 -> {r.status_code}")
print(json.dumps(answer, indent=1))
with open("code/22/08_api.json", "w") as f:
    json.dump(answer, f)

print()
for cid in (15330, 15378, 15229, 7424, 16000, 99999):
    r = client.post("/renewal", json={"contract_id": cid})
    a = r.json()
    said = a.get("detail") or f"{a['answered_by']}: {a['note']}"
    print(f"{cid:>5}  {r.status_code}\n       {said}")

r = client.post("/triage", json={
    "body": "hi, only half of order 1342207 turned up and the"
            " production line is down"})
t = r.json()
print(f"\nPOST /triage -> {r.status_code}: {t['priority']}"
      f" ({t['confidence']['priority']:.2f}), {t['category']}"
      f" ({t['confidence']['category']:.2f}),\n"
      f"  P(Urgent) {t['p_urgent']:.2f}; {t['answered_by']},"
      f" {t['model']}")
