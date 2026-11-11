"""Points between the layers of a frame's forward where a waiting question can go first (streamembed.encoder sets the hook).

Every PAUSE_EVERY layers, when a hook is set, the device is synchronized (so the frame's work queued so far is done: the CPU cannot run
ahead of the device to the end of the frame) and the hook is called; the encoder's hook lets waiting questions run, then the frame goes
on from where it was. Without a hook (the paper's benchmarks), nothing changes.
"""
from __future__ import annotations

from typing import Callable, Optional

PAUSE_EVERY = 4
_hook: Optional[Callable[[], None]] = None


def set_hook(fn: Optional[Callable[[], None]]) -> None:
    global _hook
    _hook = fn


def point(m, layer: int) -> None:
    if _hook is not None and layer % PAUSE_EVERY == PAUSE_EVERY - 1:
        m.sync()
        _hook()
