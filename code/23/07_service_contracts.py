# Contracts on what Foresight serves. The CRM's list of what it reads,
# against the schema the API publishes; the same schema after two
# tidy-ups a developer might make; the contract tests; and the list
# Chapter 22's job wrote, against the dashboard's contract.
import copy
import warnings
from contextlib import closing

warnings.filterwarnings("ignore", message=".*httpx.*")   # a notice
from _suite import run                                   # noqa: E402
from fastapi.testclient import TestClient                # noqa: E402

from foresight import contracts                          # noqa: E402
from foresight.contracts.answers import (CRM_READS,      # noqa: E402
                                         offered)
from foresight.serve import SANDBOX, store               # noqa: E402
from foresight.serve.api import Service, create_app      # noqa: E402

client = TestClient(create_app(Service(SANDBOX, on="2025-12-31")))
spec = client.get("/openapi.json").json()
answer = spec["components"]["schemas"]["RenewalAnswer"]
print(f"{'the CRM reads':<14}{'as':<16}{'the API offers'}")
for name, kinds in CRM_READS.items():
    prop = answer["properties"][name]
    got = "|".join(p["type"] for p in prop.get("anyOf", [prop]))
    print(f"{name:<14}{'|'.join(kinds):<16}{got}")
print("broken:", offered(answer) or "nothing")

tidied = copy.deepcopy(answer)
tidied["properties"]["probability"] = tidied["properties"].pop("chance")
tidied["properties"]["rank"] = {"type": "string"}
print("after renaming chance and sending rank as text:")
for problem in offered(tidied):
    print("  " + problem)

r = run("tests/test_service_contracts.py")
print(f"\ncontract tests: {(r.outcome == 'passed').sum()} of {len(r)}"
      " pass")

with closing(contracts.read_only(store.path(SANDBOX))) as con:
    found = contracts.breaches(con, contracts.ANSWERS)
    (n,) = con.execute("SELECT COUNT(*) FROM answers").fetchone()
print(f"Chapter 22's list, {n} rows: "
      f"{contracts.clauses(contracts.ANSWERS)} clauses,"
      f" {len(found)} broken")
