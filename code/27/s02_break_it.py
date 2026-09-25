# Exercise 2: three things broken on purpose in a copy of the
# sandbox, and what the job, the API and the MCP server do with each.
import json
import shutil
import textwrap

from foresight import demo
from foresight import mcp_server as mcp
from foresight.pipeline.registry import Registry
from foresight.serve import batch
from foresight.serve.api import Service, Stopwatch

root = demo.workspace()
batch.month("2026-01-01", root=root)        # December's list, made
live = Registry("renewal", root / "artifacts").folder / "1"
top = 15378                                  # first on December's list

print("1. the production model's file deleted")
(live / "model.joblib").rename(live / "model.kept")
again = batch.month("2026-01-02", root=root)
print(f"   the job the next night: {again['status']}")
blind = Service(root, on="2026-01-01")
print(f"   the API: {blind.health()['status']};"
      f" {blind.problems[0].split(':')[0]} model missing")
a = blind.score(top, Stopwatch())
print(f"   /renewal {top}: {a['answered_by']}, chance {a['chance']}")
a = blind.score(14942, Stopwatch())    # November: no list holds it
print(f"   /renewal 14942: {a['answered_by']}, {a['reasons'][0]}")
(live / "model.kept").rename(live / "model.joblib")

print("\n2. the manifest says the model knows the future")
manifest = json.loads((live / "manifest.json").read_text())
kept = dict(manifest["data"])
manifest["data"]["as_of"] = "2026-02-01"
(live / "manifest.json").write_text(json.dumps(manifest))
try:
    batch.production(root, "2026-01-01")
except batch.TooNew as e:
    print(textwrap.fill(f"the job: refused, {e}", 64,
                        initial_indent="   ", subsequent_indent="   "))
manifest["data"] = kept
(live / "manifest.json").write_text(json.dumps(manifest))

print("\n3. the MCP server asked for a month not yet reached")
(reply,) = mcp.exchange(mcp.build(root, on="2026-01-01"), [mcp.frame(
    1, "tools/call", name="renewal_risk",
    arguments={"region": "West", "month": "2026-02"})])
print(textwrap.fill(f"isError {reply['result']['isError']}:"
                    f" {mcp.content(reply)}", 64, initial_indent="   ",
                    subsequent_indent="   "))
shutil.rmtree(root)
