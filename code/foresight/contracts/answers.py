"""
Foresight's own side of a contract: what the account team's dashboard
may rely on in the `answers` view of the scores database (Chapter
22). Chapter 23 writes it.

The same clauses as the feeds Foresight reads, pointed the other way.
Here Foresight is the owner, and a change to the monthly job that
breaks one of these breaks somebody else's screen.
"""
from __future__ import annotations

from foresight.contracts.clauses import Column, Contract, Rule
from foresight.contracts.orders import DAY

ANSWERS = Contract(
    table="answers",
    owner="Foresight",
    version="1.0",
    columns=(
        Column("run", "integer", low=1),
        Column("mark", "text", form=DAY),
        Column("contract_id", "integer", low=1),
        Column("account_id", "integer", low=1),
        Column("key_account", "integer", values=(0, 1)),
        Column("rank", "integer", required=False, low=1,
               why="empty for a key account"),
        Column("chance", "real", required=False, low=0, high=1,
               why="empty for a key account or the rule's list"),
        Column("listed", "integer", values=(0, 1)),
        Column("arm", "text", required=False,
               values=("called", "held out")),
        Column("reasons", "text", form="[[]*[]]",
               why="a JSON list of up to three sentences"),
        Column("run_on", "text", form=DAY),
    ),
    key=("mark", "contract_id"),
    rules=(
        Rule("a key account has no chance and no rank",
             "SELECT COUNT(*) FROM {rows} WHERE key_account = 1"
             " AND (chance IS NOT NULL OR rank IS NOT NULL)"),
        Rule("every listed contract has an arm",
             "SELECT COUNT(*) FROM {rows} WHERE listed = 1"
             " AND arm IS NULL"),
    ),
    words=("a list, once made, is never changed; a second list for"
           " the same mark is a new run with a reason",),
    readers=("the account team's dashboard", "the API's /renewal"),
)

# What the CRM's account page reads from POST /renewal, and the JSON
# types it can handle in each field. The CRM team wrote this list; a
# field Foresight adds is no concern of theirs, and a field it drops,
# renames or retypes is.
CRM_READS = {
    "contract_id": ("integer",),
    "account": ("string",),
    "mark": ("string",),
    "answered_by": ("string",),
    "chance": ("number", "null"),
    "rank": ("integer", "null"),
    "reasons": ("array",),
    "note": ("string",),
}
JSON = {"integer": int, "number": (int, float), "string": str,
        "array": list, "null": type(None)}


def _types(prop: dict) -> set:
    return {p["type"] for p in prop.get("anyOf", [prop])}


def offered(schema: dict, reads: dict = CRM_READS) -> list[str]:
    """What the API's published schema no longer promises a reader:
    a field gone, or of a type the reader cannot handle."""
    props, out = schema["properties"], []
    for name, kinds in reads.items():
        if name not in props:
            out.append(f"{name}: gone")
        elif not _types(props[name]) <= set(kinds):
            out.append(f"{name}: now {sorted(_types(props[name]))}")
        elif ("null" not in kinds and name not in schema["required"]
              and "default" not in props[name]):
            out.append(f"{name}: no longer always sent")
    return out


def answered(body: dict, reads: dict = CRM_READS) -> list[str]:
    """The same, for one answer as it was actually sent."""
    out = []
    for name, kinds in reads.items():
        ok = tuple(JSON[k] for k in kinds)
        value = body.get(name, KeyError)
        if value is KeyError:
            out.append(f"{name}: missing")
        elif isinstance(value, bool) or not isinstance(value, ok):
            out.append(f"{name}: {type(value).__name__}")
    return out
