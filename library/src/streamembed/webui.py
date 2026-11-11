"""The search page the background capture serves: the results page with a search box, answered by the process that has the model
loaded, so a question takes under a second.

It listens on 127.0.0.1 only, on a free port, and every request needs a random token; the port and token are written to
<index folder>/web.json (owner-only) for `streamembed search` to open the page. A request whose Host is not 127.0.0.1 or localhost is
refused (a web site cannot reach it through DNS rebinding), and images are served by index row only, never by path.

    GET /?t=TOKEN[&q=QUESTION]          the page (it runs the search in the URL)
    GET /api/ping?t=TOKEN               {"ok": true, "screens": N}
    GET /api/search?t=TOKEN&q=...       results as the page's JSON; also since=, until= ("YYYY-MM-DD[ HH:MM]") and k=
    GET /img/ROW?t=TOKEN                the image of an index row
    GET /thumb/ROW?t=TOKEN              its 320 px thumbnail
    GET /api/front?t=TOKEN&id=&x=&y=&w=&h=&dpr=&focus=   page `id` is shown, in this window (screen coordinates), with or
                                        without focus; &off=1: it is not shown (another tab, minimized, covered, closed)

While a page is shown, the capture of this process paints its window gray (front(), capture.page_masks).
"""
from __future__ import annotations

import json
import math
import mimetypes
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

WEB_FILE = "web.json"
FRONT_S = 4.0           # a shown page says so every second; without word for this long, it is taken to be closed
mimetypes.add_type("image/webp", ".webp")

_front: dict = {}       # page id -> (until, {"rect", "dpr", "focused"})
_front_lock = threading.Lock()


def front() -> list[dict]:
    """The search pages on screen: {"rect": (left, top, width, height) of the browser window, "dpr": devicePixelRatio, "focused"}."""
    now = time.monotonic()
    with _front_lock:
        for k in [k for k, (until, _) in _front.items() if until <= now]:
            del _front[k]
        return [p for _, p in _front.values()]


def _set_front(qs: dict) -> None:
    pid = qs.get("id", "")
    if not (0 < len(pid) <= 40 and pid.isalnum()):
        raise ValueError("bad page id")
    if qs.get("off"):
        with _front_lock:
            _front.pop(pid, None)
        return
    x, y, w, h = (float(qs.get(k) or "nan") for k in ("x", "y", "w", "h"))
    dpr = float(qs.get("dpr") or 1)
    if not (w > 0 and h > 0 and 0 < dpr < 16) or not all(map(math.isfinite, (x, y, w, h))):
        raise ValueError("bad window")
    with _front_lock:
        if pid not in _front and len(_front) >= 32:
            raise ValueError("too many pages")
        _front[pid] = (time.monotonic() + FRONT_S, {"rect": (x, y, w, h), "dpr": dpr, "focused": qs.get("focus") == "1"})


def serve(root: Path, search: Callable[..., dict], screens: Callable[[], int], image: Callable[[int], Path],
          thumb: Callable[[int], Path], port: int = 0) -> tuple[str, Callable[[], None]]:
    """Start the page server for the index at `root`. search(q, since, until, k, link) -> page data (visualize.page_data), where
    link(kind, row) gives an image link ("img" or "thumb"); screens() -> number of stored screens; image(row) and thumb(row) -> files.
    Returns (URL with the token, stop)."""
    from .capture import page_title
    from .visualize import render_page
    token = secrets.token_urlsafe(18)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # quiet: the service log is for the capture
            pass

        def _send(self, code: int, body: bytes, kind: str, cache: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=86400" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            if host not in ("127.0.0.1", "localhost"):
                return self._send(403, b"forbidden", "text/plain")
            url = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(url.query).items()}
            if not secrets.compare_digest(qs.get("t", ""), token):
                return self._send(403, b"forbidden: open the page with streamembed search", "text/plain")
            try:
                if url.path == "/":
                    page = render_page(None, {"token": token, "screens": screens()}, title=page_title())
                    return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                if url.path == "/api/front":
                    _set_front(qs)
                    return self._json(200, {"ok": True})
                if url.path == "/api/ping":
                    return self._json(200, {"ok": True, "screens": screens()})
                if url.path == "/api/search":
                    k = int(qs["k"]) if qs.get("k", "").isdigit() else None
                    data = search(qs.get("q", ""), qs.get("since") or None, qs.get("until") or None, k,
                                  lambda kind, row: f"{kind}/{row}?t={token}")
                    return self._json(200, data)
                if url.path.startswith(("/img/", "/thumb/")):
                    row = url.path.rsplit("/", 1)[1]
                    if not row.isdigit():
                        return self._send(404, b"not found", "text/plain")
                    p = (image if url.path.startswith("/img/") else thumb)(int(row))
                    kind = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
                    return self._send(200, Path(p).read_bytes(), kind, cache=True)
                return self._send(404, b"not found", "text/plain")
            except (IndexError, FileNotFoundError, IsADirectoryError):
                return self._send(404, b"not found", "text/plain")
            except ValueError as e:                 # a bad question, date or number: the page shows the message
                return self._json(400, {"error": str(e)})
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="streamembed-web", daemon=True).start()
    address = f"http://127.0.0.1:{server.server_address[1]}/"
    marker = Path(root) / WEB_FILE
    old = os.umask(0o177)
    try:
        marker.write_text(json.dumps({"url": address, "token": token, "pid": os.getpid()}), encoding="utf-8")
    finally:
        os.umask(old)

    def stop() -> None:
        with _front_lock:
            _front.clear()
        server.shutdown()
        server.server_close()
        marker.unlink(missing_ok=True)

    return f"{address}?t={token}", stop


def find(root: Path) -> str | None:
    """The page URL (with its token) of the process serving the index at `root`, if it answers."""
    import urllib.request
    try:
        info = json.loads((Path(root) / WEB_FILE).read_text(encoding="utf-8"))
        with urllib.request.urlopen(f"{info['url']}api/ping?t={info['token']}", timeout=3) as r:
            if json.loads(r.read()).get("ok"):
                return f"{info['url']}?t={info['token']}"
    except Exception:
        pass
    return None
