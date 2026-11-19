"""Index screenshots from a folder or from the screen, query them, look at the results.

    import streamembed as se
    se.build_index_from_directory("screenshots")
    capture = se.build_index_from_now(interval=10)          # keeps capturing in the background
    results = se.query_index("the flight search page with fares to Denver", top_k=10)
    se.visualize_results(results)

Everything acts on one index folder (default ~/screenshot_index, or $STREAMEMBED_INDEX); `configure()` changes it and the other
settings. The `streamembed` command does the same from a terminal (streamembed --help).
"""
from __future__ import annotations

import bisect
import json
import math
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from typing import Callable, Iterator

from .ann import require_hnswlib
from .capture import (fit_width, folder_files, load_frames, open_screen_settings, screen_frames, screen_permission,
                      screen_permission_help)
from .gate import Gate
from .index import INDEX_TYPES, Index
from . import models as _models
from . import queryserver
from .config import _coerce, get_config
from .models import resolve

JPEG_QUALITY = 85
WEBP_QUALITY = 80       # about 82 KB for a 1280-wide screenshot (JPEG 85: 150-200 KB), text still readable
PAUSED = "paused"       # this file in the index folder pauses capture and indexing in every process (pause(), resume())
MIN_LIVE_INTERVAL = 0.5     # seconds: live capture never runs faster than this (interval 0 means "every screenshot" for folders only)


@dataclass
class Result:
    id: int             # row in the index
    score: float        # cosine similarity
    time: str           # capture time, local, "YYYY-MM-DD HH:MM:SS"
    ts: float           # the same, unix seconds
    path: str           # the image file: the original screenshot (folder), or the live capture in capture_dir
    source: str | None  # the original file for a screenshot indexed from a folder (then equal to path); None for a live capture

    def to_dict(self) -> dict:
        return asdict(self)


class Results(list):
    """What query_index returns: a list of Result, best first, that also carries its question (`query`; `text` is the same) and time
    window (`window`: (start, end) in unix seconds from since/until, None for an open end). A slice keeps them."""
    query: str | None = None
    text: str | None = None
    window: tuple | None = None

    def __getitem__(self, i):
        out = super().__getitem__(i)
        if isinstance(i, slice):
            out = Results(out)
            out.query, out.text, out.window = self.query, self.text, self.window
        return out


STALL_S = 60.0      # a screen grab that has not returned for this long is reported (Capture.stalled)


class Capture:
    """Screen capture in the background: `stop()`, `pause()`, `resume()`, `wait()`, `running`, `stalled`, and `stats` (captured,
    encoded, idle, minor, skipped, errors)."""

    def __init__(self, interval: float, duration: float | None, frames: Iterator, skip_unchanged: bool = True):
        self.stats = dict(captured=0, encoded=0, idle=0, minor=0, skipped=0, errors=0)
        self._stop, self._pause, self._done = threading.Event(), threading.Event(), threading.Event()
        self._beat = dict(waiting=None)     # when the frame source (the screen grab) was last asked for a frame; None once it returned
        self.abandoned = False              # stop() gave up on a thread stuck in a screen grab
        self._noted = False
        self._thread = threading.Thread(target=self._run, args=(frames, self.stats, self._stop, self._pause, interval, skip_unchanged,
                                                                self._beat, True, True), daemon=True)
        self._thread.start()
        threading.Thread(target=self._watch, daemon=True).start()
        self._timer = None
        if duration:
            self._timer = threading.Timer(duration, self._stop.set)
            self._timer.daemon = True           # a script that stops earlier is not kept alive until `duration`
            self._timer.start()

    def _run(self, *args) -> None:
        try:
            _ingest(*args)
        finally:
            if _state["capture"] is self and not self.abandoned:    # a capture started since writes on: keep its lock
                _release_index()
            self._done.set()

    @property
    def running(self) -> bool:
        return not self._done.is_set()

    @property
    def stalled(self) -> bool:
        """True while the screen grab has not returned for STALL_S seconds (60): capture is stuck outside StreamEmbed."""
        w = self._beat["waiting"]
        return self.running and w is not None and time.monotonic() - w >= STALL_S

    def _watch(self) -> None:
        """Say on stderr (the service's log) when the screen grab is stuck, and when it works again."""
        while not self._done.wait(min(5.0, STALL_S / 4)):
            if self.stalled and not self._noted:
                self._noted = True
                print(f"streamembed: screen capture has not returned for {STALL_S:.0f} s; nothing is captured meanwhile. stop() it "
                      "and start a new capture (the background service restarts by itself)", file=sys.stderr, flush=True)
            elif self._noted and not self.stalled:
                self._noted = False
                print("streamembed: screen capture returned again", file=sys.stderr, flush=True)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    # Waits use the _done event, not Thread.join: a Ctrl-C inside join() makes Python 3.12 mark a thread that is still running as
    # stopped, so the process would exit in the middle of an encode (and abort in torch).
    def stop(self, timeout: float = 30.0) -> bool:
        """Stop after the screenshot in progress. Returns True once the capture has ended. A screenshot being encoded is always
        finished; a screen grab that has not returned after `timeout` seconds is given up on: stop() prints so and returns False,
        nothing more is stored from this capture, and a new one can start. Once given up on, stop() returns False at once."""
        if self.abandoned:
            return False
        if self._timer is not None:
            self._timer.cancel()
        self._stop.set()
        t0 = time.monotonic()
        while not self._done.wait(0.25):
            if self._beat["waiting"] is not None and time.monotonic() - t0 >= timeout:
                self.abandoned = True
                print("streamembed: the capture is stuck in a screen grab and was left behind; nothing more is stored from it",
                      file=sys.stderr, flush=True)
                if _state["capture"] is self:
                    _release_index()        # the stuck thread stores nothing more once its grab returns
                return False
        return True

    def wait(self) -> None:
        """Block until the capture stops (after `duration`, or Ctrl-C), or until it is `stalled`."""
        try:
            while not self._done.wait(1.0):
                if self.stalled:
                    return
        except KeyboardInterrupt:
            self.stop()


# ---- settings and the lazily loaded pieces (the model takes ~20 s and ~8 GB, so it loads once, on first use)
_SESSION = ("index_dir", "index_type", "model", "models_repo", "device", "width",       # what configure() sets for this process
            "capture_dir", "capture_format", "capture_width")
_explicit: dict = {}    # the values given to configure(); every other session setting follows the config (set_config) as it changes
_cfg: dict = {}         # the session settings as last read by _settings()
_state: dict = dict(encoder=None, encoder_key=None, index=None, index_key=None, capture=None, query=None, text=None, window=None)


def _settings() -> dict:
    """The session settings now: configure()'s values, the rest from the config. Read on every use (nothing is read at import), so a
    bad value in the config file raises here, naming the file and the setting, and set_config() applies to this process too."""
    values = {k: _explicit[k] if k in _explicit else get_config(k) for k in _SESSION}
    if values["index_type"] not in INDEX_TYPES:
        raise ValueError(f"Invalid index type: {values['index_type']} not in {available_index_types()}")
    if values["index_type"] == "hnsw":
        require_hnswlib()                   # raises ImportError naming the extra to install
    _cfg.update(values, index_dir=Path(values["index_dir"]).expanduser(), models_repo=str(values["models_repo"]),
                width=int(values["width"]))
    return _cfg


def _capturing() -> bool:
    c = _state["capture"]
    return c is not None and c.running and not c.abandoned


def available_index_types() -> list[str]:
    return list(INDEX_TYPES)


def available_models() -> list[str]:
    """Model names from the configured model repo, then the built-in base models."""
    return _models.available_models(_settings()["models_repo"])


def configure(index_dir: str | Path | None = None, index_type: str | None = None, model: str | None = None,
              models_repo: str | Path | None = None, device: str | None = None, width: int | None = None,
              capture_dir: str | Path | None = None, capture_format: str | None = None, capture_width: int | None = None,
              encoder=None) -> None:
    """Settings for this process; any left as None come from the config (get_config(); set_config() keeps changes for later).
    index_dir: where frames, vectors and images go. index_type: "flat" (exact), "binary" (1-bit codes + re-rank), "hnsw" (graph,
    needs the hnsw extra), "ivf" (k-means lists) or "pq" (k-means lists + product-quantized codes + re-rank); see the README.
    model: a name from `available_models()`. models_repo: the Hugging Face repo or local folder with models.json.
    width: screenshots are fitted to this width for the model (1280). How live captures are saved: capture_dir (where; "" =
    captures/ in the index folder), capture_format ("webp" or "jpeg") and capture_width (pixels, 1280 by default; 0 = the
    screen's own size). encoder: a stand-in, for tests. A loaded model is kept while model, models_repo and device stay the same."""
    given = {k: v for k, v in dict(index_dir=index_dir, index_type=index_type, model=model, models_repo=models_repo, device=device,
                                   width=width, capture_dir=capture_dir, capture_format=capture_format,
                                   capture_width=capture_width).items() if v is not None}
    for k in ("index_dir", "models_repo", "model", "index_type", "device", "width", "capture_dir", "capture_format", "capture_width"):
        if k in given:
            given[k] = _coerce(k, given[k], strict=False) if k == "model" else _coerce(k, given[k])
    if _state["capture"] is not None:
        _state["capture"].stop()
    if _state["index"] is not None:
        _state["index"].close()             # release the write lock
    _explicit.clear()
    _explicit.update({k: str(v) if k in ("index_dir", "models_repo", "capture_dir") else v for k, v in given.items()})
    if encoder is not None or _state["encoder_key"] is None:      # a stand-in is used until the next configure()
        _state.update(encoder=encoder, encoder_key=None)
    _state.update(index=None, index_key=None, capture=None, query=None, text=None, window=None)
    _settings()                             # raises now for a bad setting


def _model_key(s: dict) -> tuple:
    return s["model"], s["models_repo"], s["device"]


def _encoder():
    s = _settings()
    enc = _state["encoder"]
    stale = enc is not None and _state["encoder_key"] is not None and _state["encoder_key"] != _model_key(s) and not _capturing()
    if enc is None or stale:
        from .encoder import Encoder
        _state["encoder"] = None
        spec = resolve(s["model"], s["models_repo"])
        size = {"qwen3-vl-2b": "about 4 GB", "qwen3-vl-2b-base": "about 4 GB", "qwen3-vl-8b": "about 17 GB"}.get(s["model"])
        note = f"downloading it first, {size}" if size and not _cached(spec.base, spec.revision) else "about 20-60 s"
        print(f"streamembed: loading {s['model']} ({note})", file=sys.stderr, flush=True)
        _quiet_transformers()
        _state["encoder"] = Encoder(spec.weights, s["device"], spec.base, spec.revision, (spec.scale, spec.scale), spec.encoder_name)
        _state["encoder_key"] = _model_key(s)
    return _state["encoder"]


def _cached(repo: str, revision: str | None) -> bool:
    """Whether the base model's weights are in the Hugging Face cache (a local folder counts as cached). Never downloads."""
    if Path(repo).expanduser().is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        config = try_to_load_from_cache(repo, "config.json", revision=revision)
    except Exception:
        return False
    return isinstance(config, str) and any(Path(config).parent.glob("*.safetensors"))


def _release_index() -> None:
    """Give up the write lock: the index is opened again, read-only, on next use (it takes the lock again on a write)."""
    idx = _state["index"]
    if idx is not None and not idx.readonly:
        idx.close()
        _state["index"] = None


def _quiet_transformers() -> None:
    """No "Loading weights" progress bar on every load: it repeats on every search and fills service.log. The download bars of the
    first run stay."""
    try:
        from huggingface_hub import utils as hub
        from transformers.utils import logging as hf_logging
        hub_off = hub.are_progress_bars_disabled()
        hf_logging.disable_progress_bar()       # this turns off the download bars too
        if not hub_off:
            hub.enable_progress_bars()
    except Exception:
        pass


def _encoder_name(s: dict) -> str:
    """The name the encoder records in an index, without loading the model when it is not loaded yet."""
    enc = _state["encoder"]
    if enc is not None and (_state["encoder_key"] is None or _state["encoder_key"] == _model_key(s)):
        return enc.name
    return resolve(s["model"], s["models_repo"]).encoder_name


def _index() -> Index:
    """The index, opened read-only; it takes the write lock on its first write. An index folder that already records its vector size
    opens without loading the model."""
    s = _settings()
    key = (s["index_dir"], s["index_type"], s["model"], s["models_repo"])
    if _state["index"] is not None and (_state["index_key"] == key or _capturing()):
        return _state["index"]
    if _state["index"] is not None:
        _state["index"].close()
        _state["index"] = None
    try:
        dim = int(json.loads((s["index_dir"] / "index.json").read_text(encoding="utf-8"))["dim"])
        name = _encoder_name(s)
    except (OSError, ValueError, KeyError, TypeError):
        enc = _encoder()
        name, dim = enc.name, getattr(enc, "dim", 2048)
    _state["index"] = Index(s["index_dir"], s["index_type"], name, dim, write=False)
    _state["index_key"] = key
    return _state["index"]


def _query_vector(idx: Index, text: str) -> np.ndarray:
    """The question's vector: from the process that captures into this index (it has the model loaded; queryserver), else from a
    model loaded here. The served vector is used only when it comes from the encoder the index was built with."""
    if _state["encoder"] is None and not _capturing():
        got = queryserver.ask(idx.root, text)
        if got is not None and got[1] == idx.encoder_name and got[0].shape[0] == idx.dim:
            return got[0]
    return _encoder().encode_query(text)


def serve_page(port: int = 0) -> tuple[str, Callable[[], None]]:
    """Serve the search page for the configured index from this process (the background capture does), with this process's model.
    Returns (URL with its token, stop). See webui."""
    from . import webui
    from .visualize import _thumb, page_data

    def search(q, since, until, k, link):
        if not str(q or "").strip():
            raise ValueError("a question is needed")
        res = query_index(q, top_k=k or get_config("page_results"), since=since, until=until)
        return page_data(_index(), res, get_config("context"), q, res.window, url=lambda row: link("img", row),
                         thumb=lambda row, match: link("thumb", row) if match else None)

    return webui.serve(_settings()["index_dir"], search, lambda: len(_index()), lambda row: _index().image(row),
                       lambda row: _thumb(_index(), row), port)


def _has_rows(root: Path) -> bool:
    try:
        return (root / "frames.jsonl").stat().st_size > 0 and (root / "vectors.f16").stat().st_size > 0
    except OSError:
        return False


def _say_paused(root: Path) -> None:
    print(f"streamembed: {root} is paused: nothing is captured or indexed until `streamembed resume` (se.resume())",
          file=sys.stderr, flush=True)


def _ingest(frames: Iterator, stats: dict, stop: threading.Event | None = None, pause: threading.Event | None = None,
            interval: float | None = None, skip_unchanged: bool = True, beat: dict | None = None, prune: bool = False,
            defer: bool = False) -> dict:
    """frames -> change gate -> JPEG -> StreamEmbed -> index. Each item of `frames` is one capture moment: None (nothing captured), one
    frame (img, ts[, screen[, source[, tag]]]) or a list of frames of different screens (source: the original file, for folders; tag: the
    display suffix of the stored file name, "1" -> 1794026938031_1.jpg). With a list, a screen missing from it (a display disconnected)
    loses its stream and change gate. Every screen has its own change gate and its own
    stream (temporal reuse never crosses screens). skip_unchanged=False stores and encodes unchanged screens too. One call starts new
    streams. A folder screenshot that is not stored is recorded as handled (Index.mark_handled). beat["waiting"]: when `frames` was
    asked for the next item, None once it returned (Capture.stalled). prune: keep live captures under capture_max_gb (checked at the
    start and every PRUNE_EVERY_S; prune_captures). defer: while a search page has focus (webui.front), while macOS reports critical
    memory pressure, and while the model loads again, live captures are stored but encoded later, in order (the first MAX_DEFERRED
    kept in memory, the next ones read back from their files, at most MAX_PENDING; the rest when the capture ends). It also frees the
    model once nothing has been captured for IDLE_UNLOAD_S (the screen locked) and loads it again, in the background, when capture
    resumes."""
    enc, idx = _encoder(), _index()
    gates: dict = {}
    last_row: dict = {}
    if hasattr(enc, "reset_stream"):
        enc.reset_stream()          # nothing is reused from the previous source's frames
    settings = _settings()
    heartbeat = get_config("heartbeat")
    width = settings["width"]
    fmt, save_width = str(settings["capture_format"]).lower(), int(settings["capture_width"])
    captures = Path(settings["capture_dir"]).expanduser().resolve() if settings["capture_dir"] else idx.root.resolve() / "captures"
    it = iter(frames)
    paused_file = idx.root / PAUSED
    noted = paused_file.exists()          # build_index_* said so before the model loaded
    reported = dict(at=None, since=0)     # errors are logged to stderr (the service's log): the first in full, then a count a minute
    next_prune = time.monotonic() if prune else math.inf
    pending: list = []                    # deferred live captures, oldest first
    waiting: dict = {}                    # screen -> its newest deferred capture (its row is not known yet)

    loading: dict = {"thread": None}
    quiet_since = None                    # when the capture last gave nothing (screen locked), for IDLE_UNLOAD_S

    def drain() -> None:
        while pending:
            job = pending.pop(0)
            img = job["img"]
            if img is None:                   # kept as a file only
                from PIL import Image
                f = Path(job["path"])
                img = fit_width(Image.open(f if f.is_absolute() else idx.root / f).convert("RGB"), width)
            vec, _ = enc.encode_frame(img, stream=job["screen"])
            row = idx.add(vec, dict(ts=job["ts"], path=job["path"]))
            stats["encoded"] += 1
            if waiting.get(job["screen"]) is job:
                del waiting[job["screen"]]
                last_row[job["screen"]] = row
            for t in job["seen"]:
                idx.still_on_screen(t, row)

    def searching() -> bool:
        if not defer:
            return False
        if loading["thread"] is not None and loading["thread"].is_alive():
            return True                       # the model is loading again: keep the captures for it
        try:
            from . import webui
            return any(p.get("focused") for p in webui.front()) or _memory_critical()
        except Exception:
            return False

    while stop is None or not stop.is_set():
        tick = time.monotonic()
        if tick >= next_prune:
            next_prune = tick + PRUNE_EVERY_S
            try:
                gone = prune_captures(idx)
                if gone:        # the rows after them moved up: so do the rows of the screens still on screen
                    gone_set = set(gone)
                    for k, r in list(last_row.items()):
                        if r in gone_set:
                            del last_row[k]
                        else:
                            last_row[k] = r - bisect.bisect_left(gone, r)
            except Exception as e:
                stats["errors"] += 1
                _report(e, reported)
        if (pause is not None and pause.is_set()) or paused_file.exists():   # before taking a frame: nothing is captured while paused
            if not noted and paused_file.exists():
                _say_paused(idx.root)
                noted = True
            stop.wait(0.5) if stop is not None else time.sleep(0.5)
            continue
        noted = False
        if beat is not None:
            beat["waiting"] = time.monotonic()
        try:
            item = next(it)
        except StopIteration:
            break
        finally:
            if beat is not None:
                beat["waiting"] = None
        if stop is not None and stop.is_set():
            break                   # stop() may have given up waiting for this grab: store nothing more
        deferring = searching()
        if pending and not deferring:
            try:
                drain()
            except Exception as e:
                stats["errors"] += 1
                _report(e, reported)
        if defer and hasattr(enc, "unload"):
            if item is None:
                quiet_since = quiet_since or tick
                if tick - quiet_since >= IDLE_UNLOAD_S and enc.loaded and not pending:
                    enc.unload()
                    print(f"streamembed: nothing captured for {IDLE_UNLOAD_S / 60:.0f} min (screen locked): the model is freed until "
                          "capture resumes", file=sys.stderr, flush=True)
            else:
                quiet_since = None
                if not enc.loaded and (loading["thread"] is None or not loading["thread"].is_alive()):
                    loading["thread"] = threading.Thread(target=enc.ensure_loaded, name="streamembed-load", daemon=True)
                    loading["thread"].start()
                    deferring = True
        if item is None:
            stats["skipped"] += 1
        elif isinstance(item, list):
            present = {f[2] if len(f) > 2 else 0 for f in item}
            for gone in [k for k in gates if k not in present]:
                del gates[gone]
                last_row.pop(gone, None)
                waiting.pop(gone, None)
                if hasattr(enc, "reset_stream"):
                    enc.reset_stream(gone)
        for frame in [] if item is None else [item] if isinstance(item, tuple) else item:
            try:
                full, ts, *rest = frame
                img = fit_width(full, width)        # what the gate compares and the encoder sees; a folder frame is fitted already
                screen = rest[0] if rest else 0
                source = rest[1] if len(rest) > 1 else None
                tag = rest[2] if len(rest) > 2 else ""
                stats["captured"] += 1
                d = gates.setdefault(screen, Gate(heartbeat))(img, ts)
                if d.encode or not skip_unchanged:
                    if source is not None:      # a folder screenshot: not copied, the index keeps the original's path
                        vec, _ = enc.encode_frame(img, stream=screen)
                        last_row[screen] = idx.add(vec, dict(ts=ts, path=None, source=source))
                        stats["encoded"] += 1
                        continue
                    # a live capture: written to capture_dir, named as OpenRecall names them (the time, "_1" for display 2, ...), and
                    # recorded by its path (Index.handled_sources lists it, so indexing that folder later leaves it out)
                    name = f"{int(ts * 1000)}{'_' + tag if tag else ''}"
                    ext = ".webp" if fmt == "webp" else ".jpg"
                    out = captures / time.strftime("%Y-%m-%d", time.localtime(ts)) / f"{name}{ext}"
                    if out.exists():
                        out = out.with_name(f"{name}-{len(idx)}{ext}")
                    if not captures.exists():
                        captures.mkdir(parents=True, mode=0o700)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    keep = full.convert("RGB") if save_width == 0 else img if save_width == img.width else fit_width(full, save_width)
                    if fmt == "webp":
                        keep.save(out, "WEBP", quality=WEBP_QUALITY, method=4)
                    else:
                        keep.save(out, "JPEG", quality=JPEG_QUALITY)
                    root = idx.root.resolve()
                    stored = out.relative_to(root).as_posix() if out.is_relative_to(root) else str(out)   # inside the index: relative,
                    if deferring and len(pending) < MAX_PENDING:                                         # so the folder can move
                        job = dict(img=img if len(pending) < MAX_DEFERRED else None, ts=ts, screen=screen, path=stored, seen=[])
                        pending.append(job)
                        waiting[screen] = job
                        continue
                    drain()                                           # the deferred ones first: rows stay in time order
                    vec, _ = enc.encode_frame(img, stream=screen)     # from memory: the same input whatever is saved
                    last_row[screen] = idx.add(vec, dict(ts=ts, path=stored))
                    stats["encoded"] += 1
                    continue
                stats["idle" if d.idle else "minor"] += 1
                if d.idle and screen in waiting:
                    waiting[screen]["seen"].append(ts)        # its stored capture is deferred: marked once it has a row
                elif d.idle and screen in last_row:
                    idx.still_on_screen(ts, last_row[screen])
                if source is not None:
                    idx.mark_handled(source)
            except Exception as e:
                stats["errors"] += 1
                _report(e, reported)
        if interval and stop is not None:
            stop.wait(max(0.0, interval - (time.monotonic() - tick)))
    try:
        drain()
    except Exception as e:
        stats["errors"] += 1
        _report(e, reported)
    return stats


IDLE_UNLOAD_S = 600.0   # the live capture frees the model once nothing has been captured (screen locked) this long
MAX_PENDING = 2000      # deferred live captures at most (past MAX_DEFERRED they wait as files only); then they are encoded at once
PRUNE_EVERY_S = 3600.0
MAX_DEFERRED = 24       # live captures held unencoded while a search page has focus (about 3 MB each at 1280 px)
PRUNE_TO = 0.9          # past capture_max_gb, remove the oldest until the captures take this share of it


def prune_captures(idx: Index, max_gb: float | None = None, keep_days: float | None = None) -> list[int]:
    """Remove live captures of `idx` (image files and index rows) older than keep_days (default: the capture_keep_days setting), then,
    if the rest take more than max_gb (default: capture_max_gb), the oldest until they take PRUNE_TO of it; 0 turns either off.
    Screenshots indexed from a folder are neither counted nor removed. Returns the removed rows (as numbered before), in order.
    Needs the write lock."""
    max_gb = get_config("capture_max_gb") if max_gb is None else max_gb
    keep_days = get_config("capture_keep_days") if keep_days is None else keep_days
    if not (max_gb and max_gb > 0) and not (keep_days and keep_days > 0):
        return []
    before = time.time() - keep_days * 86400 if keep_days and keep_days > 0 else -math.inf
    live = []
    for r, f in enumerate(idx.frames):
        if f.get("path") and not f.get("source"):
            try:
                size = idx.image(r).stat().st_size
            except OSError:
                size = 0
            live.append((f["ts"], r, size))
    total = sum(s for _, _, s in live)
    cap = max_gb * 1e9 if max_gb and max_gb > 0 else math.inf
    gone, left, old = [], total, 0
    for ts, r, size in sorted(live):
        if ts < before:
            old += 1
        elif left <= (cap * PRUNE_TO if total > cap else math.inf):
            break
        gone.append(r)
        left -= size
    if not gone:
        return []
    first, last = (time.strftime("%Y-%m-%d %H:%M", time.localtime(idx.frames[r]["ts"])) for r in (gone[0], gone[-1]))
    files = idx.remove(gone)
    for p in files:
        p.unlink(missing_ok=True)
        try:
            p.parent.rmdir()            # its day folder, once empty
        except OSError:
            pass
    why = ([f"{old} older than capture_keep_days ({keep_days:g})"] if old else []) + \
          ([f"{len(gone) - old} more as they took {total / 1e9:.1f} GB, over capture_max_gb ({max_gb:g})"] if len(gone) > old else [])
    print(f"streamembed: removed {len(gone)} live captures ({first} to {last}, {(total - left) / 1e9:.1f} GB): {'; '.join(why)}",
          file=sys.stderr, flush=True)
    return sorted(gone)


def _memory_critical() -> bool:
    """macOS: the system reports critical memory pressure (kern.memorystatus_vm_pressure_level 4)."""
    if sys.platform != "darwin":
        return False
    import ctypes
    level, size = ctypes.c_int(0), ctypes.c_size_t(4)
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    if libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(level), ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
        return False
    return level.value >= 4


def _report(e: Exception, reported: dict) -> None:
    now = time.monotonic()
    if reported["at"] is None:
        print("streamembed: a frame could not be stored:", file=sys.stderr)
        traceback.print_exception(e, file=sys.stderr)
        reported["at"] = now
    elif now - reported["at"] >= 60:
        print(f"streamembed: {reported['since'] + 1} more frames could not be stored; the last: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        reported.update(at=now, since=0)
    else:
        reported["since"] += 1


# ---- the API
def build_index_from_directory(directory: str | Path, interval: float | None = None, skip_unchanged: bool | None = None,
                               progress: Callable[[dict], None] | None = None) -> dict:
    """Index the screenshots you already have in `directory` (searched recursively), with StreamEmbed as for live capture.

    Each screenshot's time comes from its file name, else its EXIF data, else the file's modification time. Screenshots of one size are
    one screen, and so is each display number after the time in the file name ("_1", " (2)"); each screen is its own stream, and
    screens are indexed one after another, each in time order. Per screen, at most one screenshot every `interval` seconds is used
    (0: all), and unchanged screens are skipped (skip_unchanged=False: stored and encoded too). None: from the config.
    Ctrl-C stops it after the screenshot in progress; run it again to continue: screenshots read before (stored or not) are left out,
    and ones added to the folder since are read, whatever their time. The folder is recorded by its full path, so another spelling of
    it continues too. The index folder is left out of the scan; the index folder itself, or a folder inside it, is refused
    (ValueError). The index's write lock is released when it returns.
    The folder, the index and what is left to do are checked before the model loads: nothing new to index returns at once, and a
    folder with no screenshots returns zero counts. A missing folder raises FileNotFoundError, a file NotADirectoryError, and a folder
    that cannot be read PermissionError.
    progress: called with the counts after each screenshot (the `streamembed index` command prints them).
    Returns the counts: found (files), skipped (dropped by `interval`), done (read in earlier runs), todo (screenshots this run
    reads), captured (read), encoded (stored), idle (unchanged, not stored), minor (changed too little to store), errors (not
    readable or not stored), and stopped (True after Ctrl-C). found = skipped + done + encoded + idle + minor + errors once it ends."""
    if _capturing():
        raise RuntimeError("Stop the running capture first: both feed one stream encoder.")
    interval = _coerce("interval", get_config("interval") if interval is None else interval)
    skip_unchanged = get_config("skip_unchanged") if skip_unchanged is None else skip_unchanged
    folder = Path(directory).expanduser().resolve()
    root = Path(_settings()["index_dir"]).expanduser().resolve()
    _check_folder(folder, root)
    # what earlier runs read (and live captures written here) is left out first; a screenshot added later is read, even an older one
    handled = _index().handled_sources() if _state["index"] is not None or _has_rows(root) else set()
    items, found, unreadable, done = folder_files(folder, interval or None, skip=root, exclude=handled)
    screens: dict[tuple, list] = {}
    for item in items:
        screens.setdefault(item[2], []).append(item)
    stats = dict(found=found, skipped=found - len(items) - unreadable - done, done=done, todo=len(items), captured=0, encoded=0,
                 idle=0, minor=0, errors=unreadable, stopped=False)
    if not items:
        return stats
    idx = _index()
    idx._writable()
    try:
        return _index_folder(idx, screens, stats, skip_unchanged, progress)
    finally:
        _release_index()                        # a session left open does not keep `streamembed start` from writing


def _check_folder(folder: Path, root: Path) -> None:
    """A folder to index must not be the index folder or inside it (ValueError). Both are full paths."""
    if folder == root:
        raise ValueError(f"{folder} is the index folder; give the folder your screenshots are in.")
    if root in folder.parents:
        raise ValueError(f"{folder} is inside the index folder {root}: its images are already indexed. Index the screenshots "
                         "where they are, or use another index_dir.")


def _index_folder(idx: Index, screens: dict, stats: dict, skip_unchanged: bool, progress) -> dict:
    if not stats["todo"]:
        return stats                            # nothing new: the model is not loaded
    if is_paused():
        _say_paused(idx.root)
    width = _settings()["width"]

    def frames(group):
        for img, ts, screen, path in load_frames(group, width):
            if img is None:
                stats["errors"] += 1
                idx.mark_handled(path)
            else:
                yield img, ts, 0, path, screen[0]
            if progress is not None:
                progress(stats)

    try:
        for group in screens.values():
            if group:
                _ingest(frames(group), stats, skip_unchanged=skip_unchanged)    # one screen at a time: one stream, one change gate
    except KeyboardInterrupt:
        stats["stopped"] = True         # the index is reopened on next use, which drops a vector written without its line
    return stats


def build_index_from_now(interval: float | None = None, duration: float | None = None, block: bool = False,
                         monitors: str | list[int] | None = None, skip_unchanged: bool | None = None, *,
                         frames: Iterator | None = None) -> Capture:
    """Capture the screen every `interval` seconds from now on, in the background (or until it stops, with block=True).
    monitors: "all" (default; a display connected meanwhile is captured from then on), "primary", or display numbers (1 = primary,
    2, ...). Each display is its own stream, stored as <time>.jpg (primary) and <time>_1.jpg, <time>_2.jpg (others). Unchanged screens are
    skipped (skip_unchanged=False: stored and encoded too). Nothing is captured while the screen is locked or a password manager or
    private browsing window is in front (exclude_apps, exclude_titles in the config). Settings left as None come from the config.
    Live capture waits at least 0.5 s between captures. The index's write lock is held until the capture ends. A screen grab that has not returned for 60 s is reported on stderr and in
    `Capture.stalled`; stop() then gives up on it, and a new capture can start. `frames` replaces the screen (tests)."""
    interval = _coerce("interval", get_config("interval") if interval is None else interval)
    monitors = _coerce("monitors", get_config("monitors") if monitors is None else monitors)     # checked before the thread starts
    skip_unchanged = get_config("skip_unchanged") if skip_unchanged is None else skip_unchanged
    if _capturing():
        raise RuntimeError("A capture is already running: stop() it first." + (" It is stuck in a screen grab; stop() gives up on it "
                           "after 30 s." if _state["capture"].stalled else ""))
    if frames is None:
        interval = max(interval, MIN_LIVE_INTERVAL)
        if not screen_permission():         # opens the pane too: the system prompt shows only the first time
            raise PermissionError(screen_permission_help(opened=open_screen_settings()))
        frames = screen_frames(_settings()["width"], monitors)
    _index()._writable()                    # check the lock and load the model before the thread starts, so errors surface here
    try:
        if is_paused():
            _say_paused(_index().root)
        _encoder()
        _state["capture"] = Capture(interval, duration, frames, skip_unchanged)
    except BaseException:
        _release_index()
        raise
    if block:
        _state["capture"].wait()
    return _state["capture"]


def _index_dir(index_dir: str | Path | None) -> Path:
    return Path(index_dir).expanduser() if index_dir else _settings()["index_dir"]


def pause(index_dir: str | Path | None = None) -> None:
    """Pause capturing and indexing into the index folder (default: the configured one), in this process and any other, such as the
    background service. Nothing is captured until resume()."""
    d = _index_dir(index_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / PAUSED).touch()


def resume(index_dir: str | Path | None = None) -> None:
    """Undo pause()."""
    (_index_dir(index_dir) / PAUSED).unlink(missing_ok=True)


def is_paused(index_dir: str | Path | None = None) -> bool:
    return (_index_dir(index_dir) / PAUSED).exists()


def query_index(query: str, top_k: int | None = None, since: float | str | None = None, until: float | str | None = None) -> Results:
    """Screens that best match a description. A match within 30 s of a better one of the same screen is dropped (see the README).

    since / until (unix seconds or "YYYY-MM-DD[ HH:MM]"; a date alone as `until` includes that day) limit the search to that time.
    top_k: None = from the config. An empty index returns [] without loading the model.
    The list also carries the question and the window (Results.query, .text, .window), which visualize_results shows."""
    if not str(query or "").strip():
        raise ValueError('a question is needed, such as "the flight search page to Denver"')
    top_k = _coerce("top_k", get_config("top_k") if top_k is None else top_k)
    text = query
    start = _when(since, "since") if since is not None else None
    end = _when(until, "until") if until is not None else None
    if start is not None and end is not None and start >= end:
        raise ValueError(f"since ({since}) must be before until ({until})")
    _state.update(window=(start, end), query=query, text=text)
    out = Results()
    out.query, out.text, out.window = query, text, (start, end)
    if _state["index"] is None and not _has_rows(_settings()["index_dir"]):
        return out
    idx = _index()
    idx.refresh()                           # read-only (another process writes): pick up its new rows
    if len(idx) == 0 or (start is not None or end is not None) and len(idx.rows_between(start, end)) == 0:
        return out                          # nothing to search: the model is not loaded
    q = _query_vector(idx, text)
    hits = idx.search(q, top_k) if start is None and end is None else idx.search_between(q, top_k, start, end)
    for row, score in hits:
        f = idx.frames[row]
        out.append(Result(row, round(score, 4), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f["ts"])), f["ts"],
                          str(idx.image(row)), idx.source(row)))
    return out


def _when(t: float | str, which: str = "since") -> float:
    """Unix seconds, or "YYYY-MM-DD[ HH:MM[:SS]]" in local time. A date alone as `until` means the end of that day."""
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t)
    try:
        d = datetime.fromisoformat(str(t).strip())
    except ValueError:
        raise ValueError(f'{which}: a date such as 2026-11-03 or "2026-11-03 14:00", not {t!r}') from None
    if which == "until" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(t).strip()):
        d += timedelta(days=1)
    return d.timestamp()


def visualize_results(results: list[Result], path: str | Path | None = None, open_browser: bool = True, context: int | None = None,
                      query: str | None = None) -> Path:
    """Write an HTML page of the results and open it. Each result opens full size with the `context` frames before and after it
    (None: from the config). The page is headed by `query`, else by the question and time window the results came with (the list
    query_index returns, or a slice of it); a list of your own has no heading unless `query` is given."""
    from .visualize import write_page
    context = _coerce("context", get_config("context") if context is None else context)
    idx = _index() if results or _state["index"] is not None else None
    root = idx.root if idx is not None else _settings()["index_dir"]
    query = query if query is not None else getattr(results, "query", None)
    window = getattr(results, "window", None) if query == getattr(results, "query", None) else None
    out = write_page(idx, results, Path(path) if path else root / "results.html", context, query, window)
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out
