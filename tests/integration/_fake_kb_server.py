"""Minimal in-process MCP server used by integration tests.

Exposes a single ``kb_search`` tool with canned results keyed by query
substring.  Run as a subprocess by :mod:`tests.integration.test_mcp_grounding`
via ``MCPClient(MCPServerConfig(command=sys.executable, args=(path,)))``.

Leading underscore keeps pytest from auto-collecting this file as a test module.
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("fake-kb")


@mcp.tool()
def kb_search(query: str, limit: int = 5) -> str:
    """Search the fake KB. Returns canned results keyed by query substring.

    Args:
        query: Search query string.
        limit: Maximum number of results (ignored — only one result returned).

    Returns:
        Canned result string matching the first keyword found in the query,
        or ``"No results."`` if no keyword matches.
    """
    corpus = {
        "roadmap": "Q2 roadmap: finish V3 grounding. Deadline 2026-05-15.",
        "team": "Team: Jason (SDM), Alice (SDE), Bob (SDE-II).",
    }
    for key, val in corpus.items():
        if key in query.lower():
            return val
    return "No results."


if __name__ == "__main__":
    mcp.run()
