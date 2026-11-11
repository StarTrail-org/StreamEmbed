"""An MCP server that lets Claude Code (or any MCP client) search your screen history with StreamEmbed.

    claude mcp add streamembed -- uv run --python 3.12 --with streamembed --with mcp python /path/to/mcp_server.py

It reads the index that `streamembed start` (or run.py) keeps, ~/screenshot_index unless you configured another. While that service
runs, the question is embedded by its model, so this process stays small; otherwise the model is loaded here on the first search.
"""
import io

import streamembed as se
from PIL import Image as PILImage

try:                                                    # mcp 2.x
    from mcp.server.mcpserver import Image, MCPServer
except ImportError:                                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer, Image

mcp = MCPServer("streamembed")


def _jpeg(path: str, width: int = 1280) -> Image:
    im = PILImage.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, im.height * width // im.width), PILImage.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return Image(data=buf.getvalue(), format="jpeg")


@mcp.tool()
def search_screens(question: str, since: str | None = None, until: str | None = None, top_k: int = 3) -> list:
    """Find past screens of this computer that match a description, and return them as images with their capture time.

    question: what was on the screen, e.g. "the smartphone I added to my Amazon cart".
    since / until: optional local time limits, "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" (a date alone as `until` includes that day).
    top_k: how many screens to return (1-10), best match first.
    """
    results = se.query_index(question, top_k=max(1, min(int(top_k), 10)), since=since, until=until)
    if not results:
        return ["No screens found. The index is empty, or nothing was captured in that time."]
    out = []
    for i, r in enumerate(results, 1):
        out.append(f"{i}. {r.time} (similarity {r.score:.3f}) {r.path}")
        out.append(_jpeg(r.path))
    return out


if __name__ == "__main__":
    mcp.run()
