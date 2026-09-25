# `make demo`: Foresight v1.0 from the list to the MCP server, then
# four failures on purpose, then the evidence. About a minute.
# timeout: 300
from foresight import demo

demo.main(["--on", "2026-01-01"])
