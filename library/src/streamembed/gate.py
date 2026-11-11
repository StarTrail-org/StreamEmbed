"""The change gate: which captured screenshots are encoded and stored.

A captured screenshot is encoded if it is the first one, or if more than 10 % of its kept blocks (32x32 RGB; the blocks spatial pruning
with Grid keep 3x3 keeps) differ by more than 32/255 from the last encoded frame, or, when it changed less than that, as a heartbeat
`heartbeat` seconds after the last encode (the setting of that name; 300 by default). Any resolution divisible by 32 works. A
heartbeat is skipped when no block differs by more than 8/255 at all: an idle screen would otherwise add identical rows, and
near-duplicates crowd the top results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

BLOCK = 32
TAU = 8             # a block is flat if it matches its left or upper neighbour within this
DIFF = 32           # a block changed if any value differs from the last encoded frame by more than this
FRACTION = 0.1      # encode if more than this fraction of the kept blocks changed
HEARTBEAT_S = 300.0     # default; Gate(heartbeat=...) and the setting `heartbeat` change it
TILE = 3
IDLE_DIFF = 8       # below this everywhere, the screen did not change


def blocks(img: Image.Image) -> np.ndarray:
    """(blocks_h, blocks_w, 3 * 32 * 32) uint8 array of an RGB image whose sides are multiples of 32."""
    a = np.asarray(img.convert("RGB"))
    h, w = a.shape[0] // BLOCK, a.shape[1] // BLOCK
    assert a.shape[0] == h * BLOCK and a.shape[1] == w * BLOCK, f"sides must be multiples of {BLOCK}: {a.shape}"
    return a.reshape(h, BLOCK, w, BLOCK, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, -1)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-block max |a - b| on the last axis, without widening the uint8 values."""
    return (np.maximum(a, b) - np.minimum(a, b)).max(-1)


def _components(eq_left: np.ndarray, eq_up: np.ndarray) -> np.ndarray:
    """Connected components of the flat-block graph (min-label propagation, with pointer jumping)."""
    h, w = eq_left.shape
    lab = np.arange(h * w).reshape(h, w)
    while True:
        old = lab.copy()
        m = np.minimum(lab[:, 1:], lab[:, :-1])
        lab[:, 1:] = np.where(eq_left[:, 1:], m, lab[:, 1:])
        lab[:, :-1] = np.where(eq_left[:, 1:], np.minimum(lab[:, :-1], m), lab[:, :-1])
        m = np.minimum(lab[1:], lab[:-1])
        lab[1:] = np.where(eq_up[1:], m, lab[1:])
        lab[:-1] = np.where(eq_up[1:], np.minimum(lab[:-1], m), lab[:-1])
        lab = lab.reshape(-1)
        lab = lab[lab].reshape(h, w)
        if np.array_equal(lab, old):
            return lab


def kept_blocks(b: np.ndarray) -> np.ndarray:
    """Block ids (raster order) kept by spatial pruning with Grid keep: non-flat blocks, plus one block per (flat component, 3x3 tile)."""
    h, w = b.shape[:2]
    eq_left, eq_up = np.zeros((h, w), bool), np.zeros((h, w), bool)
    eq_left[:, 1:] = _max_abs_diff(b[:, 1:], b[:, :-1]) <= TAU
    eq_up[1:, :] = _max_abs_diff(b[1:], b[:-1]) <= TAU
    keep = ~(eq_left | eq_up).reshape(-1)
    if not keep.any():
        keep[0] = True
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    tiles_per_row = -(-w // TILE)
    tile = ((rows // TILE) * tiles_per_row + cols // TILE).reshape(-1)
    key = _components(eq_left, eq_up).reshape(-1) * (-(-h // TILE) * tiles_per_row) + tile
    covered = np.isin(key, np.unique(key[keep]))
    todo = np.flatnonzero(~keep & ~covered)
    if len(todo):
        uniq, inv = np.unique(key[todo], return_inverse=True)
        first = np.full(len(uniq), -1)
        first[inv[::-1]] = todo[::-1]
        keep[first] = True
    return np.flatnonzero(keep)


@dataclass
class Decision:
    encode: bool
    idle: bool          # nothing on screen changed since the last encoded frame


class Gate:
    def __init__(self, heartbeat: float = HEARTBEAT_S):
        self.ref: np.ndarray | None = None
        self.last_ts = 0.0
        self.heartbeat = float(heartbeat)

    def __call__(self, img: Image.Image, ts: float) -> Decision:
        b = blocks(img)
        if self.ref is None or self.ref.shape != b.shape:
            encode, idle = True, False
        else:
            diff = _max_abs_diff(b, self.ref).reshape(-1)
            idle = bool((diff <= IDLE_DIFF).all())
            heartbeat = ts - self.last_ts >= self.heartbeat
            encode = not idle and (heartbeat or float((diff > DIFF)[kept_blocks(b)].mean()) > FRACTION)
        if encode:
            self.ref, self.last_ts = b, ts
        return Decision(encode, idle)
