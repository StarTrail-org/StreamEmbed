"""capture_max_gb: live captures are kept under a size, the oldest removed first; folder screenshots are never removed."""
import json
import time

import pytest

import streamembed as se
from streamembed import api
from test_api import enc, noise, screenshots  # noqa: F401  (enc: the fixture)


def _capture(n, start, seed=0):
    cap = se.build_index_from_now(interval=0.01, frames=iter([(noise(seed + i), start + i) for i in range(n)]))
    cap.wait()


def _live(idx):
    return [(r, f) for r, f in enumerate(idx.frames) if f.get("path")]


def test_the_oldest_captures_go_first(tmp_path, enc, capsys):  # noqa: F811
    se.set_config(capture_max_gb=0)
    se.build_index_from_directory(screenshots(tmp_path / "shots", 2), interval=0)
    _capture(6, 1_800_000_000)
    idx = api._index()
    live = _live(idx)
    sizes = [idx.image(r).stat().st_size for r, _ in live]
    files = [idx.image(r) for r, _ in live]
    assert api.prune_captures(idx) == [], "0: no limit"
    cap_gb = sum(sizes[-3:]) * 1.05 / 1e9                   # room for the newest 3 and a bit
    assert api.prune_captures(idx, cap_gb * 10) == [], "under the limit: nothing removed"
    gone = api.prune_captures(idx, cap_gb)
    assert gone == [r for r, _ in live[:len(gone)]] and len(gone) >= 3, "the oldest first"
    left = _live(idx)
    assert sum(idx.image(r).stat().st_size for r, _ in left) <= cap_gb * 1e9 * api.PRUNE_TO
    assert [f["ts"] for _, f in left] == sorted(f["ts"] for _, f in left) and left[-1][1]["ts"] == 1_800_000_005
    assert not any(p.exists() for p in files[:len(gone)]) and all(p.exists() for p in files[len(gone):]), "their images are deleted"
    assert len([f for f in idx.frames if f.get("source")]) == 2, "folder screenshots are kept"
    assert "over capture_max_gb" in capsys.readouterr().err


def test_captures_older_than_keep_days_go(tmp_path, enc, capsys):  # noqa: F811
    se.set_config(capture_max_gb=0)
    now = time.time()
    se.build_index_from_directory(screenshots(tmp_path / "shots", 2, start=int(now) - 90 * 86400), interval=0)
    _capture(3, now - 40 * 86400)                       # 40 days ago
    _capture(2, now - 86400, seed=10)                   # yesterday
    idx = api._index()
    assert api.prune_captures(idx) == [], "no limits by default"
    old = [idx.image(r) for r, f in _live(idx) if f["ts"] < now - 30 * 86400]
    gone = api.prune_captures(idx, keep_days=30)
    assert len(gone) == 3 and not any(p.exists() for p in old)
    assert [f["ts"] > now - 2 * 86400 for _, f in _live(idx)] == [True, True]
    assert len([f for f in idx.frames if f.get("source")]) == 2, "folder screenshots are kept however old"
    assert "3 older than capture_keep_days (30)" in capsys.readouterr().err
    assert api.prune_captures(idx, keep_days=30) == []


def test_the_capture_prunes_and_keeps_its_rows_right(tmp_path, enc, monkeypatch):  # noqa: F811
    se.set_config(capture_max_gb=0)
    _capture(4, 1_800_000_000)
    idx = api._index()
    one = idx.image(0).stat().st_size
    se.set_config(capture_max_gb=one * 2.5 / 1e9)             # room for about two
    monkeypatch.setattr(api, "PRUNE_EVERY_S", 0.0)            # check before every capture
    api._release_index()
    same = noise(99)
    cap = se.build_index_from_now(interval=0.01, frames=iter([(same, 1_800_001_000), (same.copy(), 1_800_001_005),
                                                              (same.copy(), 1_800_001_010)]))
    cap.wait()
    idx = api._index()
    rows = [f["ts"] for f in idx.frames]
    assert 1_800_000_000 not in rows and rows[-1] == 1_800_001_000, "old ones removed, the new one kept"
    seen = [json.loads(l) for l in (idx.root / "seen.jsonl").read_text().splitlines()]
    assert seen and all(row == len(idx) - 1 for row, _ in seen), "still-on-screen marks the new screen's row after the rows moved"
    assert idx.frames[-1]["ts_last"] == 1_800_001_010


def test_the_setting_is_checked():
    se.set_config(capture_max_gb="30")
    assert se.get_config("capture_max_gb") == 30.0
    with pytest.raises(ValueError, match="capture_max_gb"):
        se.set_config(capture_max_gb=-1)
    with pytest.raises(ValueError, match="capture_max_gb"):
        se.set_config(capture_max_gb="lots")
