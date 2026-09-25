# What the service says when the model cannot answer: no model in the
# registry, a segment the model has never seen, a warehouse it cannot
# read; and the monthly job's list made by the rule instead.
import shutil
import tempfile
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*httpx.*")   # a notice
from fastapi.testclient import TestClient                # noqa: E402

from foresight.serve import SANDBOX, batch, online       # noqa: E402
from foresight.serve.api import Service, create_app      # noqa: E402

empty = Path(tempfile.mkdtemp())


def ask(service, title, url="/renewal", body=None):
    r = TestClient(create_app(service)).post(
        url, json=body or {"contract_id": 14942})
    a = r.json()
    print(f"{title} -> {r.status_code}")
    if "detail" in a:
        lines = [a["detail"]]
    elif url == "/renewal":
        lines = [f"{a['answered_by']}: {'; '.join(a['reasons'])}",
                 a["note"]]
    else:
        lines = [f"{a['answered_by']}: {a['priority']},"
                 f" {a['category'] or 'no category'}"]
    for said in lines:
        print(textwrap.fill(said, 66, initial_indent="  ",
                            subsequent_indent="    "))


nothing = Service(empty, on="2025-12-31")
print(f"health: {nothing.health()['status']}")
for problem in nothing.problems:
    print(f"  {problem}")
print()
ask(nothing, "no model in the registry")
ask(nothing, "no triage model", "/triage",
    {"body": "hi, only half of order 1342207 turned up and the"
             " production line is down"})


def wholesale(ids, warehouse):
    """The CRM has started calling some accounts Wholesale."""
    return online.rows(ids, warehouse).assign(segment="Wholesale")


ask(Service(SANDBOX, on="2025-12-31", fetch=wholesale),
    "a segment the model has never seen")
gone = Service(SANDBOX, on="2025-12-31")
gone.warehouse = empty / "moved.db"
ask(gone, "the warehouse cannot be read")

rule = batch.month("2025-12-31", root=empty,
                   rule="no model in production")
print("\n" + batch.summary(rule, ranks=[1, 2, 3]))
shutil.rmtree(empty)
