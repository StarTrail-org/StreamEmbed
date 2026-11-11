"""Results -> one HTML page. Frame data is embedded as JSON.

A page written to a file (write_page) links images as file:// URLs: a folder screenshot from its original file, a live capture from
capture_dir, and match thumbnails from thumbs/ in the index folder. It needs no server and keeps working after Python exits, on this
machine, while those files are there. The page the background capture serves (webui) is the same page, with a search box, and gets
its images and new results from that server."""
from __future__ import annotations

import html
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

from PIL import Image

from .capture import page_title

TEMPLATE = Path(__file__).parent / "static" / "results.html"


def _thumb(idx, row: int) -> Path:
    """A 320 px JPEG, made once per frame. Written to a temporary file first, so a crash or a second page being built at the same time
    never leaves a broken one."""
    out = idx.root / "thumbs" / f"{row}.jpg"
    if not out.is_file():
        out.parent.mkdir(exist_ok=True)
        tmp = out.with_name(f"{row}.{os.getpid()}.{threading.get_ident()}.tmp")
        with Image.open(idx.image(row)) as img:
            img.thumbnail((320, 1280))
            img.convert("RGB").save(tmp, "JPEG", quality=80)
        os.replace(tmp, out)
    return out


def _file_urls(idx) -> tuple[Callable[[int], str], Callable[[int, bool], str | None]]:
    def url(row: int) -> str:
        return idx.image(row).resolve().as_uri()

    def thumb(row: int, match: bool) -> str | None:
        cached = idx.root / "thumbs" / f"{row}.jpg"            # thumbnails are made for the matches only; the timeline uses
        try:                                                   # the image itself (loaded lazily)
            p = _thumb(idx, row) if match else cached if cached.is_file() else None
        except OSError:
            p = None                    # the original of a folder screenshot was moved or deleted: the page shows its path
        return p.resolve().as_uri() if p else None

    return url, thumb


def page_data(idx, results: list, context: int, query: str | None, window: tuple | None = None,
              url: Callable[[int], str] | None = None, thumb: Callable[[int, bool], str | None] | None = None) -> dict:
    """Each result with the `context` frames of its screen before and after it, in time order. `pos`: a frame's place among the
    frames of its screen (the page marks where frames are left out between two results it shows together). `fold`: results with
    the same value within 5 minutes are shown as one card, whose viewer holds the frames of each (null: never folded, see
    Index._dedupe_key). `source`: the original file of a folder screenshot, null for a live capture. url(row) and thumb(row, is_match)
    give the image links (default: file:// URLs)."""
    if url is None or thumb is None:
        url, thumb = _file_urls(idx)
    items = []
    for r in results:
        frames = []
        rows = idx.around(r.id, context)
        first = idx.position(r.id) - rows.index(r.id)
        for k, row in enumerate(rows):
            f = idx.frames[row]
            u = url(row)
            frames.append(dict(id=row, pos=first + k, ts=f["ts"], ts_last=f["ts_last"], source=idx.source(row), url=u,
                               thumb=thumb(row, row == r.id) or u))
        key = idx._dedupe_key(r.id)
        items.append(dict(id=r.id, score=r.score, frames=frames, fold=None if key is None else "|".join(key),
                          time=getattr(r, "time", None), path=str(idx.image(r.id))))
    return dict(query=query, made=time.strftime("%Y-%m-%d %H:%M:%S"), window=window, results=items)


def render_page(data: dict | None, live: dict | None = None, title: str | None = None) -> str:
    """The HTML page. live: {"token": ...} for the page the background capture serves (search box on); None for a file."""
    blob = lambda v: json.dumps(v).replace("</", "<\\/")    # noqa: E731
    return (TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", blob(data)).replace("/*LIVE*/null", blob(live))
            .replace("__TITLE__", html.escape(title or page_title((data or {}).get("query")))))


def write_page(idx, results: list, out: Path, context: int, query: str | None, window: tuple | None = None) -> Path:
    page = render_page(page_data(idx, results, context, query, window))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
