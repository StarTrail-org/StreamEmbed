"""While the search page is shown, its window is painted gray in the captures (not the windows in front of it), so a search does not
find earlier results pages; everything else on the screen is captured as before."""
import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

import numpy as np
import pytest
from PIL import Image

import streamembed as se
from streamembed import capture, webui
from test_api import _FakeMSS, enc, screenshots  # noqa: F401  (enc: the fixture)

LEFT, RIGHT = (0, 0, 1440, 900), (1440, 0, 1920, 1080)
GRAY = capture.MASK_GRAY


def _page(rect, focused=False, dpr=2.0):
    return {"rect": rect, "dpr": dpr, "focused": focused}


def _win(bounds, layer=0, alpha=1.0):
    return {"layer": layer, "alpha": alpha, "bounds": bounds}


def test_page_masks():
    page = (1500, 40, 1200, 800)
    windows = [_win((1600, 100, 300, 200)), _win(page), _win((0, 0, 1440, 900))]
    assert capture.page_masks([LEFT, RIGHT], [], windows) == {}
    assert capture.page_masks([LEFT, RIGHT], [_page(page)], windows) == {1: ([page], [(1600, 100, 300, 200)])}, \
        "the window in front of the page is captured"
    assert capture.page_masks([LEFT, RIGHT], [_page((1300, 40, 800, 800))], [_win((1300, 40, 800, 800))]) == \
        {0: ([(1300, 40, 140, 800)], []), 1: ([(1440, 40, 660, 800)], [])}, "across two displays: its part on each"
    assert capture.page_masks([LEFT, RIGHT], [_page(page)], [_win((10, 10, 50, 50))]) == {}, \
        "its window is not listed and it has no focus: what covers it is not known, so it is captured"
    assert capture.page_masks([LEFT, RIGHT], [_page(page, focused=True)], None) == {1: ([page], [])}, "no window list: in front"
    overlay = _win((1440, 0, 1920, 1080), layer=25)
    assert capture.page_masks([LEFT, RIGHT], [_page(page)], [overlay, _win(page)]) == {1: ([page], [])}, \
        "a display-wide overlay hides nothing"
    assert capture.page_masks([LEFT, RIGHT], [_page((-5000, 0, 100, 100))], None) == {}, "on no display: not painted"
    windows_px = [(0, 0, 2880, 1800), (2880, 0, 3840, 2160)]
    assert capture.page_masks(windows_px, [_page((1500, 100, 800, 600), focused=True)], None, pixels=True) == \
        {1: ([(3000, 200, 1600, 1200)], [])}, "Windows: displays in pixels, the window in points"


def test_a_page_is_found_by_its_window_title():
    """A results page written to a file has no server to tell where it is: its window is found by its title."""
    page, chat = (1500, 40, 1200, 800), (1600, 100, 300, 200)
    titled = {"layer": 0, "alpha": 1.0, "bounds": page, "name": "the flight search page – StreamEmbed"}
    assert capture.page_masks([LEFT, RIGHT], [], [_win(chat), titled]) == {1: ([page], [chat])}, "the window in front is captured"
    assert capture.page_masks([LEFT, RIGHT], [], [dict(titled, name="StreamEmbed")]) == {1: ([page], [])}, "the empty search page"
    for other in ("StreamEmbed docs", "dayoon/streamembed: Search your screen", "notes on StreamEmbed.md"):
        assert capture.page_masks([LEFT, RIGHT], [], [dict(titled, name=other)]) == {}, other
    assert capture.page_title("a slide") == "a slide – StreamEmbed" and capture.is_page_title(capture.page_title("a slide"))


def test_the_page_is_painted_and_the_rest_is_captured(monkeypatch):
    class Shot:
        def __init__(self, size):
            self.size = size
            self.rgb = np.full((size[1], size[0], 3), 200, np.uint8).tobytes()

    class TwoDisplays(_FakeMSS):
        monitors = [dict(left=0, top=0, width=3360, height=1080), dict(left=0, top=0, width=1440, height=900),
                    dict(left=1440, top=0, width=1920, height=1080)]

        def grab(self, m):
            return Shot((m["width"] * 2, m["height"] * 2))          # Retina: pixels are twice the points

    monkeypatch.setattr(capture, "_grabber", lambda: TwoDisplays)
    monkeypatch.setattr(capture, "_front_window", lambda: ("Safari", "News"))
    monkeypatch.setattr(capture, "_locked", lambda: False)
    page, chat = (1500, 40, 1200, 800), (1600, 100, 300, 200)
    state = {"pages": [], "windows": [_win(chat), _win(page)]}
    monkeypatch.setattr(capture, "_window_list", lambda: state["windows"])
    frames = capture.screen_frames(64, pages=lambda: state["pages"])
    assert [f[4] for f in next(frames)] == ["", "1"]
    state["pages"] = [_page(page)]
    shots = next(frames)
    assert [f[4] for f in shots] == ["", "1"], "both displays are still captured"
    left, right = (np.asarray(f[0]) for f in shots)
    assert (left == 200).all(), "nothing painted on the other display"
    x0, y0 = (page[0] - 1440) * 2, page[1] * 2
    assert (right[y0 + 10, x0 + 10] == GRAY).all() and (right[y0 + 1590, x0 + 2390] == GRAY).all(), "the page is painted"
    cx, cy = (chat[0] - 1440) * 2, chat[1] * 2
    assert (right[cy + 5, cx + 5] == 200).all(), "the window in front of it is not"
    assert (right[5, 5] == 200).all() and (right[2000, 3500] == 200).all(), "nor the rest of the display"
    state["pages"] = [_page((1440, 0, 1920, 1080))]
    state["windows"] = [_win((1440, 0, 1920, 1080))]
    assert [f[4] for f in next(frames)] == [""], "a display with nothing on it but the page is not stored"
    state["pages"] = []
    assert [f[4] for f in next(frames)] == ["", "1"]


def test_painting_a_retina_display_is_quick():
    img = Image.new("RGB", (3456, 2234), (200, 200, 200))
    t = time.perf_counter()
    share = capture._mask(img, (0, 0, 1728, 1117), [(100, 100, 1200, 800)], [(200, 200, 300, 200)])
    took = time.perf_counter() - t
    assert 0.4 < share < 0.5 and took < 0.1, took


def test_the_search_page_says_where_it_is(tmp_path, enc):  # noqa: F811
    se.build_index_from_directory(screenshots(tmp_path / "shots", 2))
    url, stop = se.api.serve_page()
    try:
        u = urlparse(url)
        token = u.query.split("=", 1)[1]
        get = lambda path: urllib.request.urlopen(f"http://{u.netloc}{path}", timeout=10)     # noqa: E731
        assert "api/front" in get(f"/?t={token}").read().decode()
        assert webui.front() == []
        get(f"/api/front?t={token}&id=a1&x=1500&y=40&w=1200&h=800&dpr=2&focus=1")
        get(f"/api/front?t={token}&id=b2&x=0&y=0&w=800&h=600&dpr=2&focus=0")
        assert sorted(webui.front(), key=lambda p: p["rect"]) == [
            {"rect": (0.0, 0.0, 800.0, 600.0), "dpr": 2.0, "focused": False},
            {"rect": (1500.0, 40.0, 1200.0, 800.0), "dpr": 2.0, "focused": True}], "two pages, each kept"
        get(f"/api/front?t={token}&id=a1&off=1")
        assert [p["rect"] for p in webui.front()] == [(0.0, 0.0, 800.0, 600.0)], "hidden: painted no more"
        for bad in ("&id=a1&x=1&y=1&w=0&h=5", "&id=a1&x=nan&y=1&w=5&h=5", "&id=a1&x=1", "&x=1&y=1&w=5&h=5", "&id=a/b&x=1&y=1&w=5&h=5"):
            with pytest.raises(urllib.error.HTTPError):
                get(f"/api/front?t={token}{bad}")
        with pytest.raises(urllib.error.HTTPError):
            get("/api/front?t=wrong&id=a1&x=1&y=1&w=5&h=5")
    finally:
        stop()
    assert webui.front() == [], "the page server stopped"


def test_a_closed_page_is_forgotten(monkeypatch):
    webui._set_front({"id": "c3", "x": "1", "y": "1", "w": "5", "h": "5"})
    assert len(webui.front()) == 1
    monkeypatch.setattr(webui, "FRONT_S", 0.0)
    webui._set_front({"id": "c3", "x": "1", "y": "1", "w": "5", "h": "5"})
    assert webui.front() == []


def test_rows_can_be_removed(tmp_path, enc):  # noqa: F811
    from streamembed.index import Index
    shots = screenshots(tmp_path / "shots", 6)
    se.build_index_from_directory(shots, interval=0)
    idx = se.api._index()
    before = np.asarray(idx._rows()).copy()
    srcs = [f["source"] for f in idx.frames]
    idx.still_on_screen(idx.frames[4]["ts"] + 30, row=4)
    (idx.root / "thumbs").mkdir(exist_ok=True)
    (idx.root / "thumbs" / "2.jpg").write_bytes(b"x")
    files = idx.remove([1, 3])
    assert [str(f) for f in files] == [srcs[1], srcs[3]]
    assert len(idx) == 4 and [f["source"] for f in idx.frames] == [srcs[i] for i in (0, 2, 4, 5)]
    assert np.array_equal(np.asarray(idx._rows()), before[[0, 2, 4, 5]])
    assert [json.loads(l)[0] for l in (idx.root / "seen.jsonl").read_text().splitlines()] == [2], "row 4 is row 2 now"
    assert idx.frames[2]["ts_last"] == idx.frames[2]["ts"] + 30
    assert not list((idx.root / "thumbs").glob("*.jpg")), "thumbnails are by row: made again"
    assert se.build_index_from_directory(shots, interval=0)["encoded"] == 0, "the removed screenshots are not indexed again"
    se.api._release_index()
    again = Index(idx.root, "flat", enc.name, write=False)
    assert len(again) == 4 and np.array_equal(np.asarray(again._rows()), before[[0, 2, 4, 5]])
    again.close()
