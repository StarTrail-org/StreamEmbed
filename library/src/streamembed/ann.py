"""Approximate search for the "hnsw", "ivf" and "pq" index types. Each gives candidate rows; Index.search scores them exactly over
vectors.f16, so the returned scores are true cosines. Up to FROM rows every type searches exactly.

    hnsw    hnswlib graph over the first PREFIX dims (renormalised, as the binary codes use), M=32, efConstruction=500; a query takes
            the graph's top efSearch=800 rows as candidates. Optional dependency: pip install "streamembed[hnsw]".
            hnsw.bin (the graph and its PREFIX-dim float32 vectors), hnsw.json (its parameters). Saved every SAVE_EVERY added rows
            and on close; rows added since are caught up from vectors.f16 on open. Never retrained: the graph grows with the index.
    ivf     IVF over the full vectors: spherical k-means, nlist = round(sqrt(n)) lists; a query scores every row of the nprobe=32
            nearest lists exactly. ivf.json, ivf_centroids.npy, ivf_codes.bin (the list of each row, int32).
    pq      IVF-PQ: the same coarse lists (pq_centroids.npy), plus 64 sub-quantizers x 8 bits over the first PREFIX dims
            (pq_codebooks.npy; 64 B per row). A query takes the nprobe=64 nearest lists, ranks their rows by the PQ estimate, and
            re-ranks the top 2,000 exactly. pq.json, pq_codes.bin (per row: list int32, then the 64 codes).

ivf and pq train on TRAIN_SAMPLE evenly spaced rows (fixed seed) when the index passes FROM rows, and train again from scratch
(new lists and codebooks, every row re-encoded) once the index has grown RETRAIN_GROWTH (4) times past the size they were trained at.

Files may lag vectors.f16 (a crash, or sessions with another type): on open the missing rows are added from vectors.f16. A
read-only process never writes; it uses the files when they are there and builds or catches up the rest in memory. A writer
retraining first empties the codes file, then replaces the model files, then the .json, so codes on disk always belong to the
model on disk; a reader takes the files only if the .json did not change while it read them.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

FROM = 10_000                   # exact search up to this many rows (a few ms)
PREFIX = 512                    # hnsw and pq index the first 512 dims, renormalised
TRAIN_SAMPLE = 50_000           # rows k-means trains on
RETRAIN_GROWTH = 4              # ivf and pq retrain when the index has grown 4x since training
KMEANS_ITERS = 10
HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, SAVE_EVERY = 32, 500, 800, 1000
IVF_NPROBE = 32
PQ_M, PQ_NPROBE, PQ_CANDIDATES, PQ_TRAIN = 64, 64, 2000, 20_000
HNSW_MISSING = ('index_type="hnsw" needs hnswlib. Install it with: pip install "streamembed[hnsw]" (chroma-hnswlib, a build of '
                'hnswlib with wheels).')


def require_hnswlib():
    try:
        import hnswlib
    except ImportError as e:
        raise ImportError(HNSW_MISSING) from e
    return hnswlib


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)


def _prefix(x: np.ndarray) -> np.ndarray:
    return _unit(np.asarray(x)[..., :PREFIX])


def _chunks(rows: np.ndarray, start: int, stop: int, size: int = 8192):
    for s in range(start, stop, size):
        yield s, np.asarray(rows[s:min(s + size, stop)], np.float32)


def _replace(path: Path, write) -> None:
    """Write a file whole or not at all."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class _Grow:
    """An array that rows are appended to, in amortised O(1)."""

    def __init__(self, shape: tuple = (), dtype=np.int32):
        self.a, self.n = np.empty((16, *shape), dtype), 0

    def append(self, x: np.ndarray) -> None:
        if self.n + len(x) > len(self.a):
            b = np.empty((max(2 * len(self.a), self.n + len(x)), *self.a.shape[1:]), self.a.dtype)
            b[:self.n] = self.a[:self.n]
            self.a = b
        self.a[self.n:self.n + len(x)] = x
        self.n += len(x)

    def view(self) -> np.ndarray:
        return self.a[:self.n]


def kmeans(x: np.ndarray, k: int, spherical: bool, iters: int = KMEANS_ITERS, seed: int = 0) -> np.ndarray:
    """k centroids of the rows of x: spherical (unit centroids, nearest by inner product) or Euclidean. Fixed seed."""
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        a = assign(x, c, spherical)
        counts = np.bincount(a, minlength=k)
        if x.shape[1] <= 64:
            sums = np.stack([np.bincount(a, weights=x[:, d], minlength=k) for d in range(x.shape[1])], 1).astype(np.float32)
        else:
            sums = np.zeros_like(c)
            for j in np.flatnonzero(counts):
                sums[j] = x[a == j].sum(0)
        empty = counts == 0
        c[~empty] = sums[~empty] / (1 if spherical else counts[~empty, None])
        c[empty] = x[rng.integers(len(x), size=int(empty.sum()))]
        if spherical:
            c = _unit(c)
    return c


def assign(x: np.ndarray, c: np.ndarray, spherical: bool) -> np.ndarray:
    out = np.empty(len(x), np.int32)
    cc = None if spherical else (c * c).sum(1)
    for s in range(0, len(x), 8192):
        d = x[s:s + 8192] @ c.T
        out[s:s + 8192] = d.argmax(1) if spherical else (cc - 2 * d).argmin(1)
    return out


class HNSW:
    name = "hnsw"

    def __init__(self, idx):
        self.hnswlib, self.idx, self.p, self.unsaved = require_hnswlib(), idx, None, 0
        self.path, self.info = idx.root / "hnsw.bin", idx.root / "hnsw.json"
        self.params = dict(dim=PREFIX, M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION)

    def _open(self, n: int):
        cap = max(2 * n, 1024)
        if _read_json(self.info) == self.params and self.path.exists():
            p = self.hnswlib.Index(space="ip", dim=PREFIX)
            try:
                p.load_index(str(self.path), max_elements=cap)
                if p.get_current_count() <= n:
                    return p                    # rows added since the last save are caught up by sync
            except (RuntimeError, ValueError, OSError):
                pass
        p = self.hnswlib.Index(space="ip", dim=PREFIX)
        p.init_index(max_elements=cap, M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, random_seed=0)
        if not self.idx.readonly:
            self.path.unlink(missing_ok=True)
            _replace(self.info, lambda f: f.write(json.dumps(self.params).encode()))
        return p

    def sync(self, n: int) -> None:
        if self.p is not None and self.p.get_current_count() > n:     # more nodes than rows (vectors.f16 was replaced)
            self.p = None
        if self.p is None:
            self.p = self._open(n)
        have = self.p.get_current_count()
        if n > self.p.get_max_elements():
            self.p.resize_index(max(2 * n, 1024))
        for s, x in _chunks(self.idx._rows(), have, n, 16384):
            self.p.add_items(_prefix(x), np.arange(s, s + len(x)))
        if not self.idx.readonly:
            self.unsaved += n - have
            if self.unsaved >= SAVE_EVERY:
                self.save()

    def save(self) -> None:
        if self.p is not None and self.unsaved and not self.idx.readonly:
            tmp = self.path.with_name(self.path.name + ".tmp")
            self.p.save_index(str(tmp))
            os.replace(tmp, self.path)
            self.unsaved = 0

    close = save

    def candidates(self, q: np.ndarray) -> np.ndarray:
        k = min(HNSW_EF_SEARCH, self.p.get_current_count())
        self.p.set_ef(max(HNSW_EF_SEARCH, k))
        labels, _ = self.p.knn_query(_prefix(q)[None], k=k)
        return labels[0].astype(np.int64)

    def memory_bytes(self) -> int:
        """hnswlib's layout: per row the PREFIX-dim float32 vector, 2M level-0 links, a label; upper levels hold ~1/M of the rows."""
        n = self.p.get_current_count()
        return n * (PREFIX * 4 + 2 * HNSW_M * 4 + 4 + 8) + n // HNSW_M * (HNSW_M * 4 + 4)


class IVF:
    """Coarse lists; per row, `width` bytes in {name}_codes.bin (the list as int32, then any codes)."""
    name, width, keys = "ivf", 4, ("centroids",)

    def __init__(self, idx):
        self.idx, self.model, self.meta, self.count, self.disk = idx, None, None, 0, None
        self.info, self.codes_path = idx.root / f"{self.name}.json", idx.root / f"{self.name}_codes.bin"

    # ---- what differs between ivf and pq
    def params(self) -> dict:
        return dict(dim=self.idx.dim)

    def train(self, sample: np.ndarray, n: int) -> dict:
        return dict(centroids=kmeans(_unit(sample), max(1, min(len(sample), round(np.sqrt(n)))), spherical=True))

    def encode(self, x: np.ndarray) -> np.ndarray:
        return assign(_unit(x), self.model["centroids"], True).reshape(-1, 1).view(np.uint8)

    def valid(self, model: dict, codes: np.ndarray) -> bool:
        c = model["centroids"]
        return c.ndim == 2 and c.shape[1] == self.idx.dim and (len(codes) == 0 or int(self._lists_of(codes).max()) < len(c))

    def _reset(self) -> None:
        self.lists = [_Grow() for _ in range(len(self.model["centroids"]))]
        self.count = 0

    def _insert(self, codes: np.ndarray) -> None:
        lists = self._lists_of(codes)
        rows = np.arange(self.count, self.count + len(codes), dtype=np.int32)
        order = np.argsort(lists, kind="stable")
        bounds = np.flatnonzero(np.diff(lists[order])) + 1
        for part in np.split(order, bounds):
            self._append(int(lists[part[0]]), rows[part], codes[part])
        self.count += len(codes)

    def _append(self, lst: int, rows: np.ndarray, codes: np.ndarray) -> None:
        self.lists[lst].append(rows)

    def _probe(self, q: np.ndarray, nprobe: int) -> np.ndarray:
        s = self.model["centroids"] @ _unit(q)
        return np.argpartition(-s, min(nprobe, len(s)) - 1)[:nprobe]

    def candidates(self, q: np.ndarray) -> np.ndarray:
        return np.concatenate([self.lists[j].view() for j in self._probe(q, IVF_NPROBE)]).astype(np.int64)

    def memory_bytes(self) -> int:
        return sum(a.nbytes for a in self.model.values()) + 4 * self.count

    # ---- shared
    @staticmethod
    def _lists_of(codes: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(codes[:, :4]).view(np.int32).ravel()

    def sync(self, n: int) -> None:
        if self.model is not None and self.count > n:      # more codes than rows (vectors.f16 was replaced)
            self.model = None
        if self.model is None:
            self.disk = _read_json(self.info)
            if not self._load(n):
                self._train(n)
        elif self.idx.readonly and _read_json(self.info) != self.disk:     # the writer trained again: take its files
            self.disk = _read_json(self.info)
            self._load(n)                       # if they do not load (still being written), keep the model in memory
        if n >= RETRAIN_GROWTH * self.meta["trained_n"]:
            self._train(n)
        if self.count < n:
            for _, x in _chunks(self.idx._rows(), self.count, n, 32768):
                codes = self.encode(x)
                self._insert(codes)
                if not self.idx.readonly:
                    with open(self.codes_path, "ab") as f:
                        codes.tofile(f)

    def _load(self, n: int) -> bool:
        meta = _read_json(self.info)
        if meta is None or {k: meta.get(k) for k in self.params()} != self.params():
            return False
        try:
            model = {k: np.load(self.idx.root / f"{self.name}_{k}.npy") for k in self.keys}
            raw = np.fromfile(self.codes_path, np.uint8) if self.codes_path.exists() else np.zeros(0, np.uint8)
        except (OSError, ValueError):
            return False
        codes = raw[:len(raw) // self.width * self.width].reshape(-1, self.width)[:n]
        if _read_json(self.info) != meta or not self.valid(model, codes):     # changed while read, or does not fit
            return False
        self.model, self.meta = model, meta
        self._reset()
        self._insert(codes)
        if not self.idx.readonly and len(raw) != len(codes) * self.width:   # a partial record, or rows that went away
            with open(self.codes_path, "r+b") as f:
                f.truncate(len(codes) * self.width)
        return True

    def _train(self, n: int) -> None:
        rows = self.idx._rows()
        sample = np.asarray(rows[np.linspace(0, n - 1, min(n, TRAIN_SAMPLE), dtype=np.int64)], np.float32)
        self.model = self.train(sample, n)
        self.meta = dict(self.params(), trained_n=n, version=time.time_ns())
        if not self.idx.readonly:
            open(self.codes_path, "wb").close()             # codes first: they never outlive their model
            for k, a in self.model.items():
                _replace(self.idx.root / f"{self.name}_{k}.npy", lambda f, a=a: np.save(f, a))
            _replace(self.info, lambda f: f.write(json.dumps(self.meta).encode()))
        self._reset()

    def close(self) -> None:
        pass


class PQ(IVF):
    name, width, keys = "pq", 4 + PQ_M, ("centroids", "codebooks")

    def params(self) -> dict:
        return dict(dim=self.idx.dim, prefix=PREFIX, m=PQ_M)

    def train(self, sample: np.ndarray, n: int) -> dict:
        model = super().train(sample, n)
        x = _prefix(sample[np.linspace(0, len(sample) - 1, min(len(sample), PQ_TRAIN), dtype=np.int64)])
        k = min(256, len(x))
        ds = PREFIX // PQ_M
        model["codebooks"] = np.stack([kmeans(x[:, j * ds:(j + 1) * ds], k, spherical=False, seed=j) for j in range(PQ_M)])
        return model

    def encode(self, x: np.ndarray) -> np.ndarray:
        lists = super().encode(x)
        xp, ds = _prefix(x), PREFIX // PQ_M
        codes = np.stack([assign(xp[:, j * ds:(j + 1) * ds], b, False) for j, b in enumerate(self.model["codebooks"])], 1)
        return np.concatenate([lists, codes.astype(np.uint8)], 1)

    def valid(self, model: dict, codes: np.ndarray) -> bool:
        b = model["codebooks"]
        return super().valid(model, codes) and b.ndim == 3 and b.shape[0] == PQ_M and b.shape[2] == PREFIX // PQ_M

    def _reset(self) -> None:
        super()._reset()
        self.list_codes = [_Grow((PQ_M,), np.uint8) for _ in self.lists]

    def _append(self, lst: int, rows: np.ndarray, codes: np.ndarray) -> None:
        self.lists[lst].append(rows)
        self.list_codes[lst].append(codes[:, 4:])

    def candidates(self, q: np.ndarray) -> np.ndarray:
        probe = self._probe(q, PQ_NPROBE)
        ids = np.concatenate([self.lists[j].view() for j in probe])
        codes = np.concatenate([self.list_codes[j].view() for j in probe])
        b = self.model["codebooks"]
        table = np.einsum("mkd,md->mk", b, _prefix(q).reshape(PQ_M, -1))      # inner product of each code with the query
        est = np.zeros(len(ids), np.float32)
        for j in range(PQ_M):
            est += table[j].take(codes[:, j])
        if len(ids) > PQ_CANDIDATES:
            ids = ids[np.argpartition(-est, PQ_CANDIDATES - 1)[:PQ_CANDIDATES]]
        return ids.astype(np.int64)

    def memory_bytes(self) -> int:
        return super().memory_bytes() + PQ_M * self.count


TYPES = {"hnsw": HNSW, "ivf": IVF, "pq": PQ}
