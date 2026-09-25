# "Which Midwest accounts are at risk, and why?" asked of Foresight's
# MCP server the way Clarity's host asks it: JSON-RPC frames, sent to
# the server in this process. No network, no language model.
import json
import shutil
import textwrap

from foresight import demo
from foresight import mcp_server as mcp
from foresight.serve import batch

root = demo.workspace()
batch.month("2026-01-01", root=root)        # December's list, made
server = mcp.build(root, on="2026-01-01")

frames = [
    mcp.frame(1, "server/discover"),
    mcp.frame(2, "tools/list"),
    mcp.frame(3, "tools/call", name="renewal_risk",
              arguments={"region": "Midwest"}),
    mcp.frame(4, "tools/call", name="why",
              arguments={"contract_id": 15458}),
    mcp.frame(5, "tools/call", name="renewal_risk",
              arguments={"region": "Mid-west"})]
found, tools, risk, why, wrong = mcp.exchange(server, frames)

info = found["result"]
(me,) = info["_meta"].values()          # the server's name and version
print(f"-> server/discover\n<- {me['name']}, protocol"
      f" {', '.join(info['supportedVersions'])}")
print("-> tools/list")
for t in tools["result"]["tools"]:
    need = t["inputSchema"].get("required", [])
    args = ", ".join(a + ("" if a in need else "?")
                     for a in t["inputSchema"]["properties"])
    print(f"<- {t['name']}({args})")

sent = {k: v for k, v in frames[2].items() if k != "params"}
sent["params"] = {k: v for k, v in frames[2]["params"].items()
                  if k != "_meta"}
print()
print(textwrap.fill(json.dumps(sent), 66, initial_indent="-> ",
                    subsequent_indent="   "))
keys = ", ".join(k.split("/")[1] for k in mcp.ENVELOPE)
print(f"   (and _meta: {keys})")
answer = mcp.content(risk)
(one,) = answer["lists"]
print(f"<- the {one['answered_by']} of record, marked {one['mark']}"
      f" ({one['model']}, run {one['run']}):\n   {one['contracts']}"
      f" Midwest contracts, {one['calls']} of the {one['of']} calls,"
      f" {len(one['key_accounts'])} key accounts")
for c in one["at_risk"]:
    arm = "hold out" if c["call"].startswith("held") else "call"
    print(f"   {c['rank']:>2}  {c['account']:<20}{c['chance']:>6.1%}"
          f"  {arm:<9}{c['manager']}")
for c in one["at_risk"][:2]:            # every call carries its three
    print(f"   {c['account']}:")
    for r in c["reasons"]:
        print(f"     {r}")

w = mcp.content(why)
print(f"\n-> tools/call why {{\"contract_id\": {w['contract_id']}}}")
print(f"<- {w['account']}, {w['answered_by']}: {w['chance']:.1%};"
      f" {w['note']}")
for r in w["reasons"]:
    print(f"   {r}")

print("\n-> tools/call renewal_risk {\"region\": \"Mid-west\"}")
print(f"<- isError: {str(wrong['result']['isError']).lower()}")
for line in mcp.content(wrong).split(" [")[0].splitlines():
    print(textwrap.fill(line.strip(), 64, initial_indent="   ",
                        subsequent_indent="     "))
shutil.rmtree(root)
