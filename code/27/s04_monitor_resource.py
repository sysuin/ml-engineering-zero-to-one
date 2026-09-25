# Exercise 4: the weekly monitoring page offered as an MCP resource,
# a noun the host can show a person, beside the three tools.
import shutil

from foresight import demo
from foresight import mcp_server as mcp
from foresight.config import ROOT

PAGE = ROOT / "docs" / "foresight-monitoring-2025.md"


def with_page(server):
    """Add the latest monitoring page to `server` as a resource."""
    @server.resource("foresight://monitor/latest",
                     mime_type="text/plain",
                     description="The latest weekly monitoring page,"
                                 " as Chapter 24's report wrote it.")
    def latest() -> str:
        text = PAGE.read_text()
        last = text.rindex("## The week")    # the last page written
        return text[last:text.index("\n## ", last + 1)]
    return server


root = demo.workspace()
before = PAGE.read_bytes()
server = with_page(mcp.build(root, on="2026-01-01"))
listed, read = mcp.exchange(server, [
    mcp.frame(1, "resources/list"),
    mcp.frame(2, "resources/read", uri="foresight://monitor/latest")])
for r in listed["result"]["resources"]:
    print(f"{r['uri']}  ({r['mimeType']})")
page = read["result"]["contents"][0]["text"].splitlines()
print("\n".join(page[:12]))
print(f"... {len(page)} lines; the page on disk unchanged:"
      f" {PAGE.read_bytes() == before}")
shutil.rmtree(root)
