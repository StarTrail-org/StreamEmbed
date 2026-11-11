"""While a search page has focus, live captures are stored but encoded after it has not: a search waits for no frame, and nothing is lost."""
import json
import time

import streamembed as se
from streamembed import api, webui
from test_api import enc, noise  # noqa: F401  (enc: the fixture)


def _run(monkeypatch, plan, focus):
    """plan: [(seed, ts)]; focus: the indexes of the frames taken while a search page has focus. Returns (rows seen when each frame was
    taken, the capture's index)."""
    state = {"focus": False}
    monkeypatch.setattr(webui, "front", lambda: [{"rect": (0, 0, 10, 10), "dpr": 1, "focused": True}] if state["focus"] else [])
    rows_then = []

    def frames():
        for k, (seed, ts) in enumerate(plan):
            state["focus"] = k in focus
            rows_then.append(len(api._index()))
            yield (noise(seed), ts)
        state["focus"] = False

    cap = se.build_index_from_now(interval=0.01, frames=frames())
    cap.wait()
    return rows_then, api._index()


def test_captures_wait_while_a_search_page_has_focus(tmp_path, enc, monkeypatch):  # noqa: F811
    t = 1_800_000_000
    plan = [(1, t), (2, t + 5), (2, t + 10), (3, t + 15), (3, t + 20), (4, t + 25)]
    rows_then, idx = _run(monkeypatch, plan, focus={1, 2, 3})
    assert rows_then == [0, 1, 1, 1, 1, 3], "nothing is encoded while the page has focus; the deferred ones first, after"
    assert [f["ts"] for f in idx.frames] == [t, t + 5, t + 15, t + 25], "every stored screen is there, in time order"
    seen = [json.loads(l) for l in (idx.root / "seen.jsonl").read_text().splitlines()]
    assert seen == [[1, t + 10], [2, t + 20]], "unchanged screens are marked on the right rows"


def test_at_most_max_pending_wait(tmp_path, enc, monkeypatch):  # noqa: F811
    monkeypatch.setattr(api, "MAX_PENDING", 1)
    t = 1_800_000_000
    rows_then, idx = _run(monkeypatch, [(1, t), (2, t + 5), (3, t + 10), (4, t + 15)], focus={1, 2, 3})
    assert len(idx) == 4 and [f["ts"] for f in idx.frames] == [t, t + 5, t + 10, t + 15]
    assert rows_then[3] == 3, "past the limit, the deferred one and the new one are encoded at once"


def test_deferred_captures_are_stored_when_the_capture_ends(tmp_path, enc, monkeypatch):  # noqa: F811
    t = 1_800_000_000
    rows_then, idx = _run(monkeypatch, [(1, t), (2, t + 5), (3, t + 10)], focus={1, 2})
    assert len(idx) == 3 and [f["ts"] for f in idx.frames] == [t, t + 5, t + 10]
