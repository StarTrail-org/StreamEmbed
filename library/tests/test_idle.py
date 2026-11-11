"""Under critical memory pressure captures wait (in memory, then as files); after the screen has been locked a while the model is freed
and loaded again, in the background, when capture resumes."""
import threading
import time

import numpy as np

import streamembed as se
from streamembed import api, webui
from test_api import FakeEncoder, enc, noise  # noqa: F401  (enc: the fixture)


class Unloadable(FakeEncoder):
    def __init__(self):
        super().__init__()
        self.loaded, self.events = True, []

    def unload(self):
        self.loaded = False
        self.events.append("unload")

    def ensure_loaded(self):
        time.sleep(0.2)                       # loading takes a while: captures meanwhile wait
        self.loaded = True
        self.events.append("load")

    def encode_frame(self, img, stream=0):
        assert self.loaded, "encoded while the model was freed"
        return super().encode_frame(img, stream)


def test_pressure_defers_and_spills_to_files(tmp_path, enc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(webui, "front", lambda: [])
    state = {"critical": False}
    monkeypatch.setattr(api, "_memory_critical", lambda: state["critical"])
    monkeypatch.setattr(api, "MAX_DEFERRED", 2)
    rows = []
    t = 1_800_000_000

    def frames():
        for k in range(7):
            state["critical"] = 1 <= k <= 5
            rows.append(len(api._index()))
            yield (noise(k), t + 5 * k)

    cap = se.build_index_from_now(interval=0.01, frames=frames())
    cap.wait()
    idx = api._index()
    assert rows == [0, 1, 1, 1, 1, 1, 1], "nothing encoded under critical pressure"
    assert [f["ts"] for f in idx.frames] == [t + 5 * k for k in range(7)], "every capture stored, in order (3 read back from files)"


def test_the_model_is_freed_while_locked(tmp_path, monkeypatch):
    e = Unloadable()
    se.configure(index_dir=tmp_path / "idx", width=256, encoder=e)
    monkeypatch.setattr(webui, "front", lambda: [])
    monkeypatch.setattr(api, "IDLE_UNLOAD_S", 0.05)
    t = 1_800_000_000

    def frames():
        yield (noise(1), t)
        for _ in range(4):                     # the screen is locked
            time.sleep(0.03)
            yield None
        yield (noise(2), t + 100)              # unlocked: loads in the background; these wait for it
        yield (noise(3), t + 105)
        time.sleep(0.4)
        yield (noise(4), t + 110)

    try:
        cap = se.build_index_from_now(interval=0.01, frames=frames())
        cap.wait()
        assert e.events == ["unload", "load"]
        assert [f["ts"] for f in api._index().frames] == [t, t + 100, t + 105, t + 110]
    finally:
        se.configure(index_dir=tmp_path / "idx")
