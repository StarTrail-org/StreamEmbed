"""The index folder: one fp16 vector and one JSON line per stored frame, appended in stream order.

    index.json      encoder name and vector size; vectors from different encoders must not share an index
    vectors.f16     `dim` fp16 per frame (2048 for 2B, 4096 for 8B); row i belongs to line i of frames.jsonl
    frames.jsonl    per frame {"ts", "source"} for a screenshot indexed from a folder (its original file), {"ts", "path"} for a live
                    capture (its file in capture_dir, relative when inside the index folder); a last line without its newline is one
                    still being written (or cut by a crash) and is left out
    seen.jsonl      [row, ts]: the screen of that row was still on screen at ts (unchanged captures)
    handled.jsonl   one JSON string per line: a folder screenshot that was read but not stored (unchanged, changed too little, or
                    not readable), so that indexing the folder again continues after it
    captures/       live captures (when capture_dir is left empty); folder screenshots are not copied
    mu.npy, codes.bin   "binary" only: centring vector and 64-B sign codes of the first 512 dimensions
    hnsw.*, ivf*, pq*   "hnsw", "ivf", "pq" only: their structures (see ann.py)
    writer.lock     held by the one process that writes (a capture service, or a script indexing). It is taken on the first write
                    (or at open with write=True) and released by close(); a process that cannot take it reads, and sees new rows on
                    refresh()

Search is exact ("flat"), or for "binary" above BINARY_FROM frames: Hamming top 1,000 over the codes, then exact re-rank.
"hnsw", "ivf" and "pq" search exactly up to ann.FROM frames; beyond that their candidates are scored exactly (ann.py).
"""
from __future__ import annotations

import bisect
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import ann
from .capture import _name_time
from .config import INDEX_TYPES

try:
    import fcntl
except ImportError:                         # Windows
    fcntl = None
    import msvcrt

DIM, CODE_DIM, CANDIDATES = 2048, 512, 1000   # DIM: default vector size (indexes written before the size was recorded)
BINARY_FROM = 1000


def _codes(v: np.ndarray, mu: np.ndarray) -> np.ndarray:
    t = v[:, :CODE_DIM].astype(np.float32)
    t /= np.linalg.norm(t, axis=1, keepdims=True).clip(1e-9)
    return np.packbits(t >= mu, axis=1)


_TAKEN = re.compile(r"^\d{13}(?:_(\d{1,2}))?-\d+$")   # a live capture whose name was taken: 1794026938031_1-17.jpg (display tag "1")
_LIVE = re.compile(r"(?:^|[\\/])\d{4}-\d{2}-\d{2}[\\/]\d{13}(?:_\d{1,2})?(?:-\d+)?\.jpg$")   # <capture_dir>/2026-11-07/1794026938031_1.jpg


def _lines(path: Path) -> tuple[list[str], int, int]:
    """(complete non-empty lines, bytes they take up, file size). Bytes after the last newline are a line still being written."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return [], 0, 0
    end = data.rfind(b"\n") + 1
    return [l for l in data[:end].decode("utf-8").splitlines() if l.strip()], end, len(data)


def _replace_text(path: Path, text: str) -> None:
    """Write a file whole or not at all: the new text is on disk before it replaces the old file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _set_aside(path: Path, data: bytes, what: str) -> Path:
    """Keep bytes the repair would otherwise drop in <name>.orphan-<time> next to `path`, and say so on stderr."""
    out = path.with_name(f"{path.name}.orphan-{int(time.time())}")
    with open(out, "ab") as f:
        f.write(data)
    print(f"streamembed: {path} had {what}; they were moved to {out}", file=sys.stderr, flush=True)
    return out


def _write_at(path: Path, offset: int, data: bytes) -> None:
    """Write `data` at `offset` and end the file there: whatever a failed earlier write left beyond it is dropped."""
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(data)
        f.truncate()


class Index:
    def __init__(self, root: Path, index_type: str, encoder_name: str, dim: int = DIM, write: bool = True):
        """write=True takes the write lock now if it is free; write=False opens read-only and takes it on the first write."""
        if index_type not in INDEX_TYPES:
            raise ValueError(f"Invalid index type: {index_type} not in {list(INDEX_TYPES)}")
        if index_type == "hnsw":
            ann.require_hnswlib()
        self.root, self.type, self.encoder_name, self.lock = Path(root), index_type, encoder_name, threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self._lockfile = None
        self.readonly = not (write and self._take_lock())   # another process writes: read only, nothing on disk is changed
        old = self._info()
        if old is not None and old["encoder"] != encoder_name:
            raise ValueError(f"{root} was built with encoder {old['encoder']!r}, not {encoder_name!r}. Use another index_dir.")
        self.dim = int(old.get("dim", DIM)) if old is not None else dim
        self.vec_path, self.meta_path, self.seen_path = self.root / "vectors.f16", self.root / "frames.jsonl", self.root / "seen.jsonl"
        self.handled_path = self.root / "handled.jsonl"
        self.ann = None
        if not self.readonly:
            self._setup()
        self._load()

    def _take_lock(self) -> bool:
        if self._lockfile is None or self._lockfile.closed:
            self._lockfile = open(self.root / "writer.lock", "a+")
        try:
            if fcntl is not None:
                fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                self._lockfile.seek(0)
                msvcrt.locking(self._lockfile.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _info(self) -> dict | None:
        """index.json, or None when it is missing or unreadable (the writer then writes it again)."""
        try:
            info = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return info if isinstance(info, dict) and "encoder" in info else None

    def _setup(self) -> None:
        """The writer: record the encoder and vector size (only when that changes the file) and create the files."""
        info = {"encoder": self.encoder_name, "dim": self.dim}
        if self._info() != info:
            _replace_text(self.root / "index.json", json.dumps(info))
        for p in (self.vec_path, self.meta_path, self.seen_path):
            if not p.exists():
                p.touch()

    def _load(self) -> None:
        """Read the folder. The writer also repairs it (only when something needs repairing); a reader leaves out a row that is still
        being written."""
        lines, used, size = _lines(self.meta_path)
        frames = []
        for i, l in enumerate(lines):
            try:
                frames.append(json.loads(l))
            except ValueError:
                raise ValueError(f"{self.meta_path}: line {i + 1} is not valid JSON; the index cannot be opened") from None
        vec_size = self.vec_path.stat().st_size if self.vec_path.exists() else 0
        row_bytes = self.dim * 2
        n = min(len(frames), vec_size // row_bytes)           # a crash between the two writes leaves one vector more
        if not self.readonly:
            # A crash leaves at most one vector (or part of one) without its line. More than that means frames.jsonl lost lines (an
            # older copy restored, a sync conflict): those vectors, or lines without a vector, are kept aside rather than dropped.
            if vec_size != n * row_bytes:
                if vec_size - n * row_bytes > row_bytes:
                    with open(self.vec_path, "rb") as f:
                        f.seek(n * row_bytes)
                        _set_aside(self.vec_path, f.read(), f"{vec_size // row_bytes - n} vectors more than frames.jsonl has lines")
                with open(self.vec_path, "r+b") as f:
                    f.truncate(n * row_bytes)
            if len(frames) > n or used != size:                # rows without a vector, or a cut last line
                if len(frames) > n + 1:
                    _set_aside(self.meta_path, "".join(l + "\n" for l in lines[n:]).encode("utf-8"),
                               f"{len(frames) - n} lines without a vector")
                _replace_text(self.meta_path, "".join(l + "\n" for l in lines[:n]))
                used = size = sum(len(l.encode("utf-8")) + 1 for l in lines[:n])
            _, h_used, h_size = _lines(self.handled_path)
            if h_used != h_size:                                # a line cut by a crash: the next one would be joined to it
                with open(self.handled_path, "r+b") as f:
                    f.truncate(h_used)
        self._all = frames                                      # every complete line; rows 0..n-1 have their vector
        self._meta = (self._stamp(self.meta_path), used)
        self._seen_max: dict[int, float] = {}
        self._seen = (self._stamp(self.seen_path), 0)
        for m in frames:
            m["ts_last"] = m["ts"]
        self._read_seen()
        self.frames = frames[:n]
        if not self.readonly:
            self._all = self.frames
        self._groups = None
        self.mu = self.codes = None
        if self.type == "binary" and n > BINARY_FROM:
            self._load_codes()
        self._sync_ann()

    @staticmethod
    def _stamp(path: Path):
        try:
            st = path.stat()
        except FileNotFoundError:
            return None
        return st.st_ino, st.st_size, st.st_mtime_ns

    def _read_seen(self) -> None:
        """seen.jsonl from where the last read stopped. Lines that do not parse are skipped; the writer cuts a partial last one."""
        stamp, pos = self._seen
        try:
            with open(self.seen_path, "rb") as f:
                f.seek(pos)
                data = f.read()
        except FileNotFoundError:
            return
        end = data.rfind(b"\n") + 1
        if not self.readonly and end < len(data):
            with open(self.seen_path, "r+b") as f:
                f.truncate(pos + end)
        for l in data[:end].decode("utf-8", "replace").splitlines():
            try:
                row, ts = json.loads(l)
                row, ts = int(row), float(ts)
            except (ValueError, TypeError):
                continue
            self._seen_max[row] = max(self._seen_max.get(row, ts), ts)
            if 0 <= row < len(self._all):
                self._all[row]["ts_last"] = max(self._all[row]["ts_last"], ts)
        self._seen = (self._stamp(self.seen_path), pos + end)

    def _sync_ann(self) -> None:
        """hnsw, ivf, pq: past ann.FROM rows, bring the structure up to the rows (it is built, loaded or caught up)."""
        if self.type in ann.TYPES and len(self) > ann.FROM:
            if self.ann is None:
                self.ann = ann.TYPES[self.type](self)
            self.ann.sync(len(self))

    def refresh(self) -> None:
        """Read-only: pick up the rows the writing process added since. Only what was appended is read; a file the writer replaced
        (or cut) is read again in full."""
        if not self.readonly:
            return
        with self.lock:
            meta, seen = self._stamp(self.meta_path), self._stamp(self.seen_path)
            vec_rows = (self.vec_path.stat().st_size if self.vec_path.exists() else 0) // (self.dim * 2)
            if meta == self._meta[0] and seen == self._seen[0] and vec_rows == len(self):
                return                                          # nothing changed
            old_meta, pos = self._meta
            if meta is None or old_meta is None or meta[0] != old_meta[0] or meta[1] < pos or \
                    (seen is not None and self._seen[0] is not None and (seen[0] != self._seen[0][0] or seen[1] < self._seen[1])):
                self._load()
                return
            with open(self.meta_path, "rb") as f:
                f.seek(pos)
                data = f.read()
            end = data.rfind(b"\n") + 1
            for l in data[:end].decode("utf-8").splitlines():
                if l.strip():
                    m = json.loads(l)
                    m["ts_last"] = max(m["ts"], self._seen_max.get(len(self._all), m["ts"]))
                    self._all.append(m)
            self._meta = (meta, pos + end)
            self._read_seen()
            n = min(len(self._all), vec_rows)
            self.frames = self._all[:n]
            self._groups = None
            if self.codes is not None:
                if len(self.codes) < n:
                    self.codes = np.concatenate([self.codes, _codes(np.asarray(self._rows()[len(self.codes):n]), self.mu)])
                self.codes = self.codes[:n]
            elif self.type == "binary" and n > BINARY_FROM:
                self._load_codes()
            self._sync_ann()

    def close(self) -> None:
        """Save what is pending (hnsw saves its graph every ann.SAVE_EVERY rows) and release the write lock (also released when the
        process ends)."""
        if self.ann is not None and not self.readonly and self._lockfile is not None and not self._lockfile.closed:
            with self.lock:
                self.ann.close()
        if self._lockfile is not None:
            self._lockfile.close()

    def _writable(self) -> None:
        """Become the writer: take the write lock if this process does not hold it yet (then read the folder again, as the writer), or
        raise RuntimeError while another process holds it."""
        if not self.readonly:
            return
        with self.lock:
            if not self._take_lock():
                raise RuntimeError(f"{self.root} is being written by another process: the capture service, a `streamembed index` or "
                                   "`capture` command, or a Python session that is capturing or indexing into it. Stop that first "
                                   "(`streamembed stop` stops the service), or use another index_dir.")
            self.readonly = False
            self.ann = None
            self._setup()
            self._load()

    def __len__(self) -> int:
        return len(self.frames)

    def _rows(self) -> np.ndarray:
        return np.memmap(self.vec_path, np.float16, "r", shape=(len(self.frames), self.dim))

    def _load_codes(self) -> None:
        mu_path, code_path = self.root / "mu.npy", self.root / "codes.bin"
        if mu_path.exists():
            self.mu = np.load(mu_path)
            stored = np.fromfile(code_path, np.uint8) if code_path.exists() else np.zeros(0, np.uint8)
            self.codes = stored[:len(stored) // (CODE_DIM // 8) * (CODE_DIM // 8)].reshape(-1, CODE_DIM // 8)[:len(self)]
        else:
            sample = self._rows()[np.linspace(0, len(self) - 1, min(len(self), 50_000), dtype=np.int64)]
            t = np.asarray(sample[:, :CODE_DIM], np.float32)
            self.mu = (t / np.linalg.norm(t, axis=1, keepdims=True).clip(1e-9)).mean(0)
            self.codes = np.zeros((0, CODE_DIM // 8), np.uint8)
        if len(self.codes) < len(self):                                          # codes lag after a crash or a "flat" session
            self.codes = np.concatenate([self.codes, _codes(np.asarray(self._rows()[len(self.codes):]), self.mu)])
        if not self.readonly:
            np.save(mu_path, self.mu)
            self.codes.tofile(code_path)

    def add(self, vector: np.ndarray, meta: dict) -> int:
        """Store one frame. The vector is written at its row's offset, so a vector left by a failed add is overwritten; if the line
        cannot be written, frames.jsonl is cut back to where it was and the error is raised."""
        self._writable()
        with self.lock:
            v = vector.astype(np.float16).reshape(1, self.dim)
            row = len(self.frames)
            line = {"ts": meta["ts"], **({"path": meta["path"]} if meta.get("path") else {}),
                    **({"source": meta["source"]} if meta.get("source") else {})}
            size = self.meta_path.stat().st_size
            try:
                _write_at(self.vec_path, row * self.dim * 2, v.tobytes())
                with open(self.meta_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line) + "\n")
            except BaseException:
                with open(self.meta_path, "r+b") as f:
                    f.truncate(size)
                raise
            self.frames.append({**line, "ts_last": line["ts"]})
            if self._all is not self.frames:
                self._all = self.frames
            self._groups = None
            if self.codes is not None:
                c = _codes(v, self.mu)
                _write_at(self.root / "codes.bin", row * (CODE_DIM // 8), c.tobytes())
                self.codes = np.concatenate([self.codes[:row], c])
            elif self.type == "binary" and len(self) > BINARY_FROM:
                self._load_codes()
            self._sync_ann()
            return row

    def remove(self, rows) -> list[Path]:
        """Take rows out of the index (the writer only; stop the capture first). The rows after them move up, so seen.jsonl is
        renumbered, and what is kept by row number is dropped to be made again (thumbs/, codes.bin, the hnsw/ivf/pq structures). The
        removed rows' files are recorded as handled, so indexing their folder again leaves them out. Returns those files (live
        captures and folder screenshots alike); they are not deleted here."""
        self._writable()
        with self.lock:
            gone = {int(r) for r in rows}
            if not gone:
                return []
            if min(gone) < 0 or max(gone) >= len(self):
                raise IndexError(f"rows must be in 0..{len(self) - 1}")
            files = [self.image(r) for r in sorted(gone)]
            keep = [r for r in range(len(self)) if r not in gone]
            new = {old: i for i, old in enumerate(keep)}
            lines, _, _ = _lines(self.meta_path)
            seen, _, _ = _lines(self.seen_path)
            kept_seen = []
            for l in seen:
                try:
                    row, ts = json.loads(l)
                except ValueError:
                    continue
                if row in new:
                    kept_seen.append(json.dumps([new[row], ts]))
            with open(self.handled_path, "a", encoding="utf-8") as f:
                for p in files:
                    f.write(json.dumps(str(p)) + "\n")
            # each file is written whole before it replaces the old one (a crash between the replaces can still leave them apart:
            # keep a copy of the index folder when that matters)
            tmp = self.vec_path.with_name(self.vec_path.name + ".tmp")
            rows = self._rows()
            with open(tmp, "wb") as f:
                for s in range(0, len(keep), 8192):        # a part at a time: the vectors of a large index need not fit in memory
                    f.write(np.ascontiguousarray(rows[keep[s:s + 8192]], np.float16).tobytes())
                f.flush()
                os.fsync(f.fileno())
            del rows
            _replace_text(self.meta_path.with_name(self.meta_path.name + ".new"), "".join(lines[r] + "\n" for r in keep))
            os.replace(tmp, self.vec_path)
            os.replace(self.meta_path.with_name(self.meta_path.name + ".new"), self.meta_path)
            _replace_text(self.seen_path, "".join(l + "\n" for l in kept_seen))
            for p in [self.root / "codes.bin", self.root / "mu.npy", self.root / "hnsw.bin", self.root / "hnsw.json",
                      *self.root.glob("ivf*"), *self.root.glob("pq*")]:
                p.unlink(missing_ok=True)
            for p in (self.root / "thumbs").glob("*.jpg"):
                p.unlink()
            self.ann = None
            self._load()
            return files

    def still_on_screen(self, ts: float, row: int | None = None) -> None:
        """Stored screen `row` (default: the last one) is unchanged at ts."""
        self._writable()
        with self.lock:
            if self.frames:
                row = len(self) - 1 if row is None else row
                self.frames[row]["ts_last"] = max(self.frames[row]["ts_last"], ts)
                with open(self.seen_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps([row, ts]) + "\n")

    def mark_handled(self, source: str) -> None:
        """A folder screenshot was read but not stored: indexing the folder again continues after it."""
        self._writable()
        with self.lock, open(self.handled_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(source) + "\n")

    def image(self, row: int) -> Path:
        """The image file of a row: a screenshot from a folder, or a live capture in capture_dir (a path relative to the index
        folder when it is inside it). It may have been moved or deleted since. "path": rows written before captures were recorded
        by their source."""
        f = self.frames[row]
        p = Path(f.get("source") or f.get("path") or f.get("title") or "")
        return p if p.is_absolute() else (self.root / p).resolve()

    def handled_sources(self) -> set[str]:
        """The image files of every stored frame (live captures too, so that indexing capture_dir leaves them out) and every handled
        folder screenshot (mark_handled). A relative source (indexes written before folders were resolved) is also given resolved
        against the working directory; a relative live capture path is resolved against the index folder."""
        out = {f.get("source") or f.get("title") for f in self.frames} - {None}      # "title": indexes written before 2026-11-07
        root = self.root.resolve()
        out |= {p if os.path.isabs(p) else str(root / p) for p in (f.get("path") for f in self.frames if not f.get("source")) if p}
        lines, _, _ = _lines(self.handled_path)
        for l in lines:
            try:
                out.add(str(json.loads(l)))
            except ValueError:
                continue
        return out | {str(Path(s).resolve()) for s in out if not os.path.isabs(s)}

    def live(self, row: int) -> bool:
        """Whether a row is a live capture (recorded under "path"; StreamEmbed 0.3.0 recorded it under "source", by its capture_dir
        name), not a screenshot indexed from a folder."""
        f = self.frames[row]
        if f.get("source"):
            return bool(_LIVE.search(f["source"]))
        return not f.get("title")           # "title": the original file, in indexes written before 2026-11-07

    def source(self, row: int) -> str | None:
        """The original file of a screenshot indexed from a folder; None for a live capture."""
        return None if self.live(row) else self.frames[row].get("source") or self.frames[row].get("title")

    # ---- screens: which stream a stored frame belongs to
    def screen(self, row: int) -> tuple[str, str]:
        """(folder, display tag) of a stored frame: frames of one screen share it. The display tag is the number after the time in the
        file name, as indexing reads it (capture._name_time): " (2)" (macOS), "_1" (OpenRecall and live captures), "" for none. A
        live capture's folder is "": all live captures of one display are one screen. Screens that indexing told apart by image
        size alone (displays of different sizes with no number in the name) share a key here."""
        f = self.frames[row]
        key = f.get("_screen")
        if key is None:
            src = self.source(row) or ""
            stem = Path(f.get("source") or f.get("path") or f.get("title") or "").stem
            ts, tag = _name_time(stem)
            if ts is None:
                taken = _TAKEN.match(stem)
                tag = (taken[1] or "") if taken else ""
            folder = re.match(r"(.*)[\\/]", src)
            folder = folder[1] if folder else ""
            key = f["_screen"] = (folder, tag)
        return key

    def _dedupe_key(self, row: int):
        """Matches with the same key within dedupe_s seconds are one result. None: never folded, for screenshots whose file name has
        no time (their time may be the time the file was copied)."""
        src = self.source(row)
        if src and _name_time(re.split(r"[\\/]", src)[-1].rsplit(".", 1)[0])[0] is None:
            return None
        return self.screen(row)

    def _best(self, ids: np.ndarray, scores: np.ndarray, k: int, dedupe_s: float) -> list[tuple[int, float]]:
        out, kept = [], {}
        for j in np.argsort(-scores):
            row = int(ids[j])
            key, ts = self._dedupe_key(row), self.frames[row]["ts"]
            if key is not None:
                near = kept.setdefault(key, [])
                if any(abs(ts - t) < dedupe_s for t in near):
                    continue
                near.append(ts)
            out.append((row, float(scores[j])))
            if len(out) == k:
                break
        return out

    def search(self, q: np.ndarray, k: int, dedupe_s: float = 30.0) -> list[tuple[int, float]]:
        """(row, cosine) best first. A match within dedupe_s seconds of a better one of the same screen is dropped (screens stay up for
        a while)."""
        with self.lock:
            n = len(self)
            if n == 0:
                return []
            rows, q = self._rows(), q.astype(np.float32)
            if self.ann is not None:
                ids = np.sort(self.ann.candidates(q))
                scores = np.asarray(rows[ids], np.float32) @ q
            elif self.codes is None:
                ids = np.arange(n)
                scores = np.concatenate([np.asarray(rows[s:s + 8192], np.float32) @ q for s in range(0, n, 8192)])
            else:
                t = q[:CODE_DIM] / max(float(np.linalg.norm(q[:CODE_DIM])), 1e-9)
                qc = np.packbits(t >= self.mu).view(np.uint64)
                dist = np.bitwise_count(self.codes.view(np.uint64) ^ qc).sum(1)
                ids = np.sort(np.argpartition(dist, min(CANDIDATES, n) - 1)[:CANDIDATES])
                scores = np.asarray(rows[ids], np.float32) @ q
            return self._best(ids, scores, k, dedupe_s)

    def _group(self, row: int) -> tuple[list, int]:
        """The (ts, row) list of the screen of `row`, in time order, and the position of `row` in it."""
        if self._groups is None:
            groups: dict = {}
            for i, f in enumerate(self.frames):
                groups.setdefault(self.screen(i), []).append((f["ts"], i))
            for g in groups.values():
                g.sort()
            self._groups = groups
        g = self._groups[self.screen(row)]
        return g, bisect.bisect_left(g, (self.frames[row]["ts"], row))

    def around(self, row: int, n: int) -> list[int]:
        """Rows of the same screen as `row` (see screen()) recorded just before and after it, including it, in time order."""
        with self.lock:
            g, pos = self._group(row)
            return [i for _, i in g[max(0, pos - n):pos + n + 1]]

    def position(self, row: int) -> int:
        """Where `row` is among the rows of its screen, in time order (0: the first)."""
        with self.lock:
            return self._group(row)[1]

    def rows_between(self, start: float | None, end: float | None) -> np.ndarray:
        """The rows on screen at some time in [start, end)."""
        lo, hi = -np.inf if start is None else start, np.inf if end is None else end
        with self.lock:
            return np.array([i for i, f in enumerate(self.frames) if f["ts"] < hi and f["ts_last"] >= lo], np.int64)

    def search_between(self, q: np.ndarray, k: int, start: float | None, end: float | None,
                       dedupe_s: float = 30.0) -> list[tuple[int, float]]:
        """As search, over the screens on screen at some time in [start, end) only (exact; a time window is usually small)."""
        with self.lock:
            ids = self.rows_between(start, end)
            if len(ids) == 0:
                return []
            rows, q = self._rows(), q.astype(np.float32)
            scores = np.concatenate([np.asarray(rows[ids[s:s + 8192]], np.float32) @ q for s in range(0, len(ids), 8192)])
            return self._best(ids, scores, k, dedupe_s)
