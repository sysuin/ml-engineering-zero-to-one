# Requests the API's contract refuses, each with the 422 a caller
# gets back; and a count of the times the service itself was asked.
import json
import textwrap
import warnings

warnings.filterwarnings("ignore", message=".*httpx.*")   # a notice
from fastapi.testclient import TestClient                # noqa: E402

from foresight.serve import SANDBOX                      # noqa: E402
from foresight.serve.api import Service, create_app      # noqa: E402

service = Service(SANDBOX, on="2025-12-31")
asked = []
score, classify = service.score, service.classify
service.score = lambda *a: asked.append(a) or score(*a)
service.classify = lambda *a: asked.append(a) or classify(*a)
client = TestClient(create_app(service))

bad = [("/renewal", {"contract_id": "14942"}),
       ("/renewal", {"contract_id": 14942.5}),
       ("/renewal", {"contract": 14942}),
       ("/renewal", {"contract_id": -1}),
       ("/renewal", "contract_id=14942"),
       ("/triage", {"body": "   "}),
       ("/triage", {"body": "x" * 2001}),
       ("/forecast/mops/Midwest", None),
       ("/forecast/MRD-CLE-001/Mid-west", None)]
for url, body in bad:
    if body is None:
        r, shown = client.get(url), "GET " + url
    else:
        content = body if isinstance(body, str) else json.dumps(body)
        r = client.post(url, content=content,
                        headers={"content-type": "application/json"})
        what = content if len(content) < 32 else '{"body": "xx...x"}'
        shown = f"POST {url} {what}"
    print(f"{shown} -> {r.status_code}")
    for e in r.json()["detail"]:
        where = ".".join(str(p) for p in e["loc"][1:]) or "body"
        print(textwrap.fill(f"{where}: {e['msg']}", 66,
                            initial_indent="    ",
                            subsequent_indent="      "))
print(f"\nThe service was asked {len(asked)} times.")
