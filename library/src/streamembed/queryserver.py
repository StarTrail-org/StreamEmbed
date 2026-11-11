"""Questions encoded by the process that already has the model loaded.

The background capture keeps the model in memory. It also listens on a local channel tied to its index folder, and query_index in
another process (`streamembed search`) sends it the question instead of loading a second copy of the model (20-60 s, about 8 GB).
Only the question's vector comes back; the search itself runs in the asking process, on the index opened read-only.

    macOS, Linux   a Unix socket, <index folder>/query.sock (owner-only); a path too long for a socket uses one in the temp folder
                   named after the index folder
    Windows        127.0.0.1 on a free port, written with a random token to <index folder>/query.port

One JSON line each way: {"text": ..., "token": ...} -> {"encoder": name, "vector": base64 float32} or {"error": message}.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import socket
import socketserver
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable

import numpy as np

SOCK = "query.sock"
PORT_FILE = "query.port"
WINDOWS = sys.platform == "win32"


def _sock_path(root: Path) -> Path:
    p = Path(root).resolve() / SOCK
    if len(str(p)) < 100:                       # AF_UNIX paths are limited to about 104 bytes on macOS
        return p
    h = hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"streamembed-{os.getuid()}-{h}.sock"


def serve(root: Path, encode: Callable[[str], np.ndarray], name: str) -> Callable[[], None]:
    """Answer questions for the index at `root` with `encode` (Encoder.encode_query) from a daemon thread. Returns a function that
    stops it and removes the socket or port file."""
    token = secrets.token_hex(16) if WINDOWS else ""

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                req = json.loads(self.rfile.readline(1 << 20).decode("utf-8"))
                if WINDOWS and not secrets.compare_digest(str(req.get("token", "")), token):
                    raise PermissionError("wrong token")
                v = np.asarray(encode(str(req["text"])), np.float32)
                out = {"encoder": name, "vector": base64.b64encode(v.tobytes()).decode("ascii")}
            except Exception as e:              # one bad request never stops the server (or the capture)
                out = {"error": f"{type(e).__name__}: {e}"}
            self.wfile.write((json.dumps(out) + "\n").encode("utf-8"))

    if WINDOWS:
        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        marker = Path(root) / PORT_FILE
        marker.write_text(json.dumps({"port": server.server_address[1], "token": token, "pid": os.getpid()}), encoding="utf-8")
    else:
        marker = _sock_path(root)
        if marker.exists():                     # left by a process that did not stop cleanly (only the writer serves)
            marker.unlink()
        old = os.umask(0o177)                   # the socket is created owner-only
        try:
            server = socketserver.ThreadingUnixStreamServer(str(marker), Handler)
        finally:
            os.umask(old)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="streamembed-queries", daemon=True).start()

    def stop() -> None:
        server.shutdown()
        server.server_close()
        Path(marker).unlink(missing_ok=True)

    return stop


def ask(root: Path, text: str, timeout: float = 60.0) -> tuple[np.ndarray, str] | None:
    """(question vector, encoder name) from the process serving the index at `root`, or None if none answers."""
    try:
        if WINDOWS:
            info = json.loads((Path(root) / PORT_FILE).read_text(encoding="utf-8"))
            s = socket.create_connection(("127.0.0.1", int(info["port"])), timeout=timeout)
            token = info["token"]
        else:
            path = _sock_path(root)
            if not path.exists():
                return None
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(str(path))
            token = ""
        with s:
            s.sendall((json.dumps({"text": text, "token": token}) + "\n").encode("utf-8"))
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(1 << 16)
                if not chunk:
                    break
                data += chunk
        out = json.loads(data.decode("utf-8"))
        if "vector" not in out:
            return None
        return np.frombuffer(base64.b64decode(out["vector"]), np.float32).copy(), out["encoder"]
    except (OSError, ValueError, KeyError):
        return None
