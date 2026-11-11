"""The API without the model: a stand-in encoder gives random unit vectors."""
import importlib.util
import json
import os
import time

import numpy as np
import pytest
from PIL import Image

import streamembed as se
from streamembed import index as index_mod
from streamembed.capture import private

DIM = 2048


class FakeEncoder:
    name = "fake"

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.resets = 0
        self.streams = []

    def reset_stream(self, stream=None):
        self.resets += stream is None
        self.dropped = getattr(self, "dropped", []) + ([stream] if stream is not None else [])

    def encode_frame(self, path, stream=0):
        assert hasattr(path, "size") or os.path.isfile(path), "a PIL image or a stored file"
        self.streams.append(stream)
        v = self.rng.standard_normal(DIM).astype(np.float32)
        return v / np.linalg.norm(v), 1.0

    def encode_query(self, text):
        v = np.random.default_rng(len(text)).standard_normal(DIM).astype(np.float32)
        return v / np.linalg.norm(v)


def noise(seed, w=256, h=128):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), np.uint8))


def screenshots(folder, n, start=1_700_000_000):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        p = folder / f"shot_{n - i:03d}.png"          # names in reverse time order, to tell "mtime" from "name"
        noise(i).save(p)
        os.utime(p, (start + 100 * i, start + 100 * i))
    return folder


@pytest.fixture
def enc(tmp_path):
    e = FakeEncoder()
    se.configure(index_dir=tmp_path / "idx", width=256, encoder=e)
    yield e
    se.configure(index_dir=tmp_path / "idx")      # stops any capture left running


def test_index_types_and_validation(tmp_path):
    assert se.available_index_types() == ["flat", "binary", "hnsw", "ivf", "pq"]
    with pytest.raises(ValueError, match="index_type: one of flat"):
        se.configure(index_dir=tmp_path, index_type="annoy")


def test_directory_then_query(tmp_path, enc):
    stats = se.build_index_from_directory(screenshots(tmp_path / "shots", 6))
    assert stats["encoded"] == 6 and stats["errors"] == 0
    res = se.query_index("a page with a form", top_k=3)
    assert len(res) == 3 and res[0].score >= res[1].score >= res[2].score
    assert all(os.path.isfile(r.path) and r.path == r.source for r in res), "a folder screenshot's image is its original"
    lines = [json.loads(l) for l in (tmp_path / "idx" / "frames.jsonl").read_text().splitlines()]
    assert [m["ts"] for m in lines] == [1_700_000_000 + 100 * i for i in range(6)], "file times, in time order"
    assert lines[0]["source"].endswith("shot_006.png") and set(lines[0]) == {"ts", "source"}, "time and original file, no copy"
    assert not (tmp_path / "idx" / "captures").exists(), "folder screenshots are not copied into the index"


def test_unchanged_screens_and_missing_folder(tmp_path, enc):
    folder = screenshots(tmp_path / "shots", 3)
    noise(2).save(folder / "again.png")                         # same pixels as the last screenshot, 50 s later
    os.utime(folder / "again.png", (1_700_000_250, 1_700_000_250))
    stats = se.build_index_from_directory(folder)
    assert stats["found"] == 4 and stats["captured"] == 4 and stats["encoded"] == 3 and stats["idle"] == 1
    idx = index_mod.Index(tmp_path / "idx", "flat", "fake")
    assert idx.frames[-1]["ts_last"] == 1_700_000_250, "the unchanged screen extends the last row's time on screen"
    with pytest.raises(FileNotFoundError):
        se.build_index_from_directory(tmp_path / "empty")


def test_copied_folder_is_not_thinned(tmp_path, enc):
    """Files with no time in their name, copied within a second: their file times are copy times, so the interval keeps them all."""
    folder = tmp_path / "shots"
    folder.mkdir()
    for i in range(6):
        noise(i).save(folder / f"shot_{i}.png")
        os.utime(folder / f"shot_{i}.png", (1_700_000_000 + 0.1 * i,) * 2)
    stats = se.build_index_from_directory(folder, interval=5)
    assert stats["skipped"] == 0 and stats["encoded"] == 6


def test_capture_time_from_names():
    from datetime import datetime
    from streamembed.capture import _name_time
    t = datetime(2026, 11, 6, 15, 20, 11).timestamp()
    assert _name_time("Screenshot 2026-11-06 at 15.20.11") == (t, "")
    assert _name_time("스크린샷 2026-11-06 오후 3.20.11") == (t, "")
    assert _name_time("Screen Shot 2026-11-06 at 3.20.11 PM") == (t, "")
    assert _name_time("Screenshot 2026-11-06 at 15.20.11 (2)") == (t, "2"), "macOS: second display"
    assert _name_time("20261106_152011") == (t, "")
    assert _name_time("1793413554_1") == (1793413554.0, "1") and _name_time("1793413554123") == (1793413554.123, "")
    assert _name_time("shot_003") == (None, "")


def test_same_size_displays_by_name(tmp_path, enc):
    """Two displays of one size, told apart by the number after the time: each keeps its own interval and change gate."""
    folder = tmp_path / "shots"
    folder.mkdir()
    for i in range(4):
        t = 1_700_000_000 + 10 * i
        noise(i).save(folder / f"{t}.png")                       # display 0: changes every time
        noise(7).save(folder / f"{t}_1.png")                     # display 1, same size: never changes
    stats = se.build_index_from_directory(folder)
    assert stats["captured"] == 8 and stats["encoded"] == 5 and stats["idle"] == 3 and enc.resets == 2


def test_interval_and_screens(tmp_path, enc):
    """Two displays (two sizes) interleaved every 2 s: each is its own screen for the interval and the change gate."""
    folder = tmp_path / "shots"
    folder.mkdir()
    for i in range(10):
        t = 1_700_000_000 + 2 * i
        noise(i, 256, 128).save(folder / f"{t}.png")             # display 0: a new screen every time
        noise(7, 192, 128).save(folder / f"{t}_1.png")           # display 1: never changes
    stats = se.build_index_from_directory(folder, interval=5.0)
    # display 0 keeps t = 0, 6, 12, 18 (2 s apart, at least 5 s between kept ones); display 1 the same four, encoded once
    assert stats["found"] == 20 and stats["skipped"] == 12 and stats["captured"] == 8
    assert stats["encoded"] == 5 and stats["idle"] == 3 and enc.resets == 2, "one stream per screen"
    se.configure(index_dir=tmp_path / "idx2", width=256, encoder=enc)
    assert se.build_index_from_directory(folder, interval=0)["captured"] == 20


def test_stop_and_continue(tmp_path):
    """Ctrl-C stops a folder run; the next run continues after the last indexed screenshot, without duplicates."""
    class Interrupted(FakeEncoder):
        def __init__(self, after):
            super().__init__()
            self.after = after

        def encode_frame(self, path, stream=0):
            if len(self.streams) == self.after:
                raise KeyboardInterrupt
            return super().encode_frame(path, stream)

    folder = screenshots(tmp_path / "shots", 6)
    se.configure(index_dir=tmp_path / "idx", width=256, encoder=Interrupted(after=3))
    first = se.build_index_from_directory(folder)
    assert first["stopped"] and first["encoded"] == 3
    se.configure(index_dir=tmp_path / "idx", width=256, encoder=FakeEncoder())
    again = se.build_index_from_directory(folder)
    assert not again["stopped"] and again["done"] == 3 and again["encoded"] == 3
    titles = [json.loads(l)["source"] for l in (tmp_path / "idx" / "frames.jsonl").read_text().splitlines()]
    assert len(titles) == 6 == len(set(titles))
    assert se.build_index_from_directory(folder)["encoded"] == 0, "nothing new"
    se.configure(index_dir=tmp_path / "idx")


def test_small_images_are_not_enlarged():
    from streamembed.capture import fit_width
    assert fit_width(noise(0, 100, 50), 1280).size == (96, 64)
    assert fit_width(noise(0, 3456, 2234), 1280).size == (1280, 832)


def test_one_writer_others_read(tmp_path, enc):
    """While one process writes an index (the capture service), another opens it read-only and sees new rows on refresh."""
    se.build_index_from_directory(screenshots(tmp_path / "shots", 3))
    assert se.api._state["index"] is None, "indexing a folder releases the write lock when it returns"
    writer = index_mod.Index(tmp_path / "idx", "flat", "fake")      # the capture service
    with open(tmp_path / "idx" / "vectors.f16", "ab") as f:      # the writer is in the middle of a row: vector, no line yet
        np.zeros(DIM, np.float16).tofile(f)
    reader = index_mod.Index(tmp_path / "idx", "flat", "fake")
    assert reader.readonly and not writer.readonly and len(reader) == 3
    assert (tmp_path / "idx" / "vectors.f16").stat().st_size == 4 * DIM * 2, "a reader does not repair the writer's files"
    with pytest.raises(RuntimeError, match="another process"):
        reader.add(np.zeros(DIM, np.float32), dict(ts=1.0, path="x.jpg"))
    with open(tmp_path / "idx" / "frames.jsonl", "a") as f:      # the writer finishes the row
        f.write(json.dumps(dict(ts=2e9, path="images/x.jpg")) + "\n")
    reader.refresh()
    assert len(reader) == 4
    reader.close()
    writer.close()


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """The service module with the service managers replaced by a recorder; returns (module, calls, set_platform)."""
    import sys
    from streamembed import service
    calls = []

    def run(*args):
        calls.append(args)
        out = {"launchctl": "\tstate = running\n\tpid = 4242\n", "systemctl": "ActiveState=active\nMainPID=4242\n",
               "schtasks": "Status:        Running\n", "whoami": "pc-1\\me\r\n"}.get(args[0], "")
        return type("R", (), dict(returncode=0, stdout=out, stderr=""))()

    monkeypatch.setattr(service, "_run", run)
    monkeypatch.setattr(service, "BOOTOUT_WAIT_S", 0)
    monkeypatch.setattr(service.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(service.os, "getuid", lambda: 501, raising=False)     # macOS simulated on Windows, which has no getuid
    return service, calls, lambda name: monkeypatch.setattr(sys, "platform", name)


def _service_round_trip(service, tmp_path):
    (tmp_path / "shots").mkdir(exist_ok=True)
    path = service.install_service(interval=10, monitors=[1, 2], skip_unchanged=False, index_dir=tmp_path / "idx",
                                   directory=tmp_path / "shots")
    (tmp_path / "idx" / "paused").touch()
    st = service.service_status()
    args = st["args"]
    assert args[args.index("--folder") + 1] == str((tmp_path / "shots").resolve()) and st["paused"]
    (tmp_path / "idx" / "paused").unlink()
    assert st["installed"] and st["running"] and args[1:3] == ["-m", "streamembed"]
    assert args[args.index("--interval") + 1] == "10" and args[args.index("--monitors") + 1] == "1,2" and "--keep-unchanged" in args
    assert args[args.index("--index-dir") + 1] == str((tmp_path / "idx").resolve())
    assert st["log"] == str((tmp_path / "idx" / "service.log").resolve())
    service.uninstall_service()
    assert not path.exists() and service.service_status()["installed"] is False
    return path


def test_service_macos(tmp_path, svc):
    import plistlib
    service, calls, platform = svc
    platform("darwin")
    path = service.install_service(index_dir=tmp_path / "idx")
    p = plistlib.loads(path.read_bytes())
    assert path.name == "org.streamembed.capture.plist" and p["RunAtLoad"] and p["KeepAlive"] == {"SuccessfulExit": False}
    assert [c[:2] for c in calls] == [("launchctl", "bootout"), ("launchctl", "bootstrap")]
    _service_round_trip(service, tmp_path)


def test_service_linux(tmp_path, svc, monkeypatch):
    service, calls, platform = svc
    platform("linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="DISPLAY"):
        service.install_service(index_dir=tmp_path / "idx")
    monkeypatch.setenv("DISPLAY", ":0")
    path = service.install_service(index_dir=tmp_path / "idx")
    unit = path.read_text()
    assert path.name == "streamembed-capture.service" and "Restart=on-failure" in unit and "Environment=DISPLAY=:0" in unit
    assert ("systemctl", "--user", "enable", "streamembed-capture.service") in calls
    _service_round_trip(service, tmp_path)
    assert ("systemctl", "--user", "disable", "--now", "streamembed-capture.service") in calls


def test_service_windows(tmp_path, svc, monkeypatch):
    service, calls, platform = svc
    platform("win32")
    monkeypatch.setenv("HF_HOME", "C:\\hf cache")
    path = service.install_service(index_dir=tmp_path / "idx")
    xml = path.read_text(encoding="utf-16")
    assert "<LogonTrigger>" in xml and "<RestartOnFailure>" in xml and "--env" in xml
    assert "<UserId>pc-1\\me</UserId>" in xml, "the account comes from whoami ($USERDOMAIN is WORKGROUP over SSH)"
    assert [c[:2] for c in calls if c[0] == "schtasks"] == [("schtasks", "/End"), ("schtasks", "/Create"), ("schtasks", "/Run")]
    _service_round_trip(service, tmp_path)
    assert ("schtasks", "/Delete", "/TN", "StreamEmbedCapture", "/F") in calls


def test_service_process_arguments():
    from streamembed import service
    with pytest.raises(SystemExit):
        service.main(["--help"])


def test_command_line(tmp_path, svc, capsys, monkeypatch):
    from streamembed import cli
    service, calls, platform = svc
    platform("darwin")
    idx = str(tmp_path / "idx")
    assert cli.main(["pause", "--index-dir", idx]) == 0 and (tmp_path / "idx" / "paused").exists()
    assert cli.main(["status", "--index-dir", idx]) == 0 and "not installed" in capsys.readouterr().out
    assert cli.main(["start", "--index-dir", idx, "--interval", "10", "--monitors", "1,2", "--folder", str(tmp_path)]) == 0
    st = service.service_status()
    assert st["installed"] and st["args"][3] == "run" and float(st["args"][st["args"].index("--interval") + 1]) == 10
    cli.main(["status", "--index-dir", idx])
    assert "paused" in capsys.readouterr().out.splitlines()[-2]
    assert cli.main(["resume", "--index-dir", idx]) == 0 and not (tmp_path / "idx" / "paused").exists()
    assert cli.main(["stop", "--index-dir", idx]) == 0 and not service.service_status()["installed"]
    seen = []
    monkeypatch.setattr(service, "main", lambda argv: seen.append(argv) or 0)
    old = ["--index-dir", idx, "--interval", "5", "--log", str(tmp_path / "service.log")]
    assert cli.main(old) == 0 and seen == [old], \
        "services installed before the subcommands still start"
    se.configure(index_dir=tmp_path / "other")


def test_index_records_vector_size(tmp_path):
    idx = index_mod.Index(tmp_path / "i8", "flat", "big", dim=4096)
    v = np.ones(4096, np.float32) / 64
    idx.add(v, dict(ts=1.0, path="a.jpg"))
    again = index_mod.Index(tmp_path / "i8", "flat", "big")             # the size comes from index.json
    assert again.dim == 4096 and len(again) == 1 and again.search(v, 1)[0][0] == 0


def test_reopen_and_crash_recovery(tmp_path, enc):
    se.build_index_from_directory(screenshots(tmp_path / "shots", 4))
    with open(tmp_path / "idx" / "vectors.f16", "ab") as f:      # a vector written, then a crash before its line
        np.zeros(DIM, np.float16).tofile(f)
    se.configure(index_dir=tmp_path / "other", width=256, encoder=enc)     # the crashed writer is gone: its lock is released
    idx = index_mod.Index(tmp_path / "idx", "flat", "fake")
    assert len(idx) == 4 and (tmp_path / "idx" / "vectors.f16").stat().st_size == 4 * DIM * 2
    with pytest.raises(ValueError, match="encoder"):
        index_mod.Index(tmp_path / "idx", "flat", "another-model")


def test_binary_switches_to_codes(tmp_path, enc, monkeypatch):
    monkeypatch.setattr(index_mod, "BINARY_FROM", 3)
    se.configure(index_dir=tmp_path / "idx", index_type="binary", width=256, encoder=enc)
    se.build_index_from_directory(screenshots(tmp_path / "shots", 5))
    idx = index_mod.Index(tmp_path / "idx", "binary", "fake")
    assert idx.codes is not None and len(idx.codes) == 5
    assert len(se.query_index("x", top_k=2)) == 2
    assert index_mod.Index(tmp_path / "idx", "flat", "fake").codes is None, "the same folder opened as flat searches exactly"


def test_capture_in_background(tmp_path, enc):
    se.build_index_from_directory(screenshots(tmp_path / "a", 2))
    frames = ((noise(100 + i), time.time() + i) for i in range(3))
    cap = se.build_index_from_now(interval=0.01, frames=frames)
    cap.wait()
    assert not cap.running and cap.stats["encoded"] == 3
    assert enc.resets == 2, "each source starts a new stream"


def test_capture_several_displays(tmp_path, enc):
    """Each capture moment gives one frame per display; each display has its own stream and change gate."""
    def ticks():
        for i in range(4):
            ts = time.time() + i
            yield [(noise(200 + i), ts, 1, None, ""), (noise(7, 192, 128), ts, 2, None, "1")]   # display 2 never changes

    cap = se.build_index_from_now(interval=0.01, frames=ticks())
    cap.wait()
    assert cap.stats["captured"] == 8 and cap.stats["encoded"] == 5 and cap.stats["idle"] == 3
    assert enc.streams == [1, 2, 1, 1, 1], "frames go to their display's stream"
    idx = index_mod.Index(tmp_path / "idx", "flat", "fake")
    row2 = 1                                                            # stream order 1, 2, 1, 1, 1
    assert idx.frames[row2]["ts_last"] > idx.frames[row2]["ts"], "display 2 stayed on screen, recorded on its own row"
    assert idx.frames[0]["path"].endswith(f"{int(idx.frames[0]['ts'] * 1000)}.webp") and idx.frames[1]["path"].endswith("_1.webp")
    assert idx.frames[0]["path"].startswith("captures/") and set(idx.frames[0]) == {"ts", "ts_last", "path"}
    res = se.query_index("x")
    assert res and all(r.source is None and os.path.isfile(r.path) for r in res), "live captures have no original file"


def test_display_connected_and_disconnected(tmp_path, enc):
    """A display that appears is captured from then on; one that disappears loses its stream."""
    def ticks():
        t = time.time()
        yield [(noise(1), t, 1, None, "")]
        yield [(noise(2), t + 1, 1, None, ""), (noise(3, 192, 128), t + 1, 2, None, "1")]     # a monitor is connected
        yield [(noise(4), t + 2, 1, None, "")]                                                 # and disconnected

    cap = se.build_index_from_now(interval=0.01, frames=ticks())
    cap.wait()
    assert cap.stats["encoded"] == 4 and enc.streams == [1, 1, 2, 1] and enc.dropped == [2]
    names = sorted(p.name for p in (tmp_path / "idx" / "captures").rglob("*.webp"))
    assert sum(n.endswith("_1.webp") for n in names) == 1
    se.configure(index_dir=tmp_path / "idx2", width=256, encoder=enc)
    again = se.build_index_from_directory(tmp_path / "idx" / "captures", interval=0)     # captures are a valid folder
    assert again["encoded"] == 4 and enc.resets == 3, "two displays -> two streams (plus the capture's one)"


def test_pause_from_anywhere(tmp_path, enc):
    """pause() writes a file in the index folder; every process (the background service too) stops capturing until resume()."""
    def endless():
        i = 0
        while True:
            i += 1
            yield (noise(i), time.time())

    cap = se.build_index_from_now(interval=0.02, frames=endless())
    time.sleep(0.3)
    se.pause()
    assert se.is_paused()
    time.sleep(0.2)
    n = cap.stats["captured"]
    time.sleep(0.4)
    assert cap.stats["captured"] == n, "nothing captured while paused"
    se.resume()
    time.sleep(1.0)                                     # paused capture looks again every 0.5 s
    cap.stop()
    assert cap.stats["captured"] > n and not se.is_paused()


def test_keep_unchanged_screens(tmp_path, enc):
    folder = tmp_path / "shots"
    folder.mkdir()
    for i in range(3):
        noise(1).save(folder / f"{1_700_000_000 + 10 * i}.png")          # the same screen three times
    assert se.build_index_from_directory(folder)["encoded"] == 1
    se.configure(index_dir=tmp_path / "idx2", width=256, encoder=enc)
    assert se.build_index_from_directory(folder, skip_unchanged=False)["encoded"] == 3


def test_capture_skips_locked_or_private_and_stops(tmp_path, enc):
    def endless():
        i = 0
        while True:
            i += 1
            yield None if i % 3 == 0 else (noise(i), time.time())

    cap = se.build_index_from_now(interval=0.02, duration=0.5, frames=endless())
    with pytest.raises(RuntimeError):
        se.build_index_from_now(interval=0.02, frames=endless())
    with pytest.raises(RuntimeError):
        se.build_index_from_directory(tmp_path)
    cap.wait()
    assert not cap.running and cap.stats["encoded"] >= 2 and cap.stats["skipped"] >= 1
    assert private("1Password 8", None) and private("Safari", "Private Browsing - x") and not private("Safari", "News")


def test_visualize_page(tmp_path, enc):
    se.build_index_from_directory(screenshots(tmp_path / "shots", 8))
    meta = tmp_path / "idx" / "frames.jsonl"
    lines = meta.read_text().splitlines()
    first = json.loads(lines[0])
    first["source"] = "evil </script><script>alert(1)</script>"
    meta.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    se.configure(index_dir=tmp_path / "idx", width=256, encoder=enc)          # reopen with the edited file name
    res = se.query_index("x", top_k=8)
    out = se.visualize_results(res, open_browser=False, context=3)
    page = out.read_text()
    assert out == tmp_path / "idx" / "results.html"
    assert "/*DATA*/" not in page and '"x"' in page
    assert "</script><script>alert" not in page, "file names cannot close the script element"
    assert page.count("file://") >= 2 * len(res)
    assert len(list((tmp_path / "idx" / "thumbs").glob("*.jpg"))) == 7, "the edited first row has no original: no thumbnail"
    data = json.loads(page.split("let DATA = ", 1)[1].split(";\n", 1)[0].replace("<\\/", "</"))
    for r in data["results"]:
        ts = [f["ts"] for f in r["frames"]]
        assert ts == sorted(ts) and len(ts) <= 7 and r["id"] in [f["id"] for f in r["frames"]], "context: time order, 3 each side"
        pos = [f["pos"] for f in r["frames"]]
        assert pos == list(range(pos[0], pos[0] + len(pos))), "pos: the place among the screen's frames (folded cards join them)"
    assert all(r["fold"] is None for r in data["results"]), "shot_001.png has no time in its name: never folded"
    assert 'role", "button"' in page and "showModal(); show()" in page, "cards open from the keyboard; the strip scrolls to the match"
    assert "vorig" not in page and "f.id in V.r.hits" in page, "no second link to the same file; every folded match is marked"


# ---- "hnsw", "ivf", "pq": small indexes, with the exact-search threshold lowered
ANN = [pytest.param("hnsw", marks=pytest.mark.skipif(importlib.util.find_spec("hnswlib") is None,
                                                       reason='needs the hnsw extra: pip install "streamembed[hnsw]"')),
       "ivf", "pq"]
ANN_FILES = {"hnsw": ("hnsw.bin", "hnsw.json"), "ivf": ("ivf.json", "ivf_centroids.npy", "ivf_codes.bin"),
             "pq": ("pq.json", "pq_centroids.npy", "pq_codebooks.npy", "pq_codes.bin")}


def clustered(n, seed=0):
    """Unit vectors around 6 centres, so the lists and the graph have structure."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((6, DIM))
    v = centres[rng.integers(0, 6, n)] + 0.5 * rng.standard_normal((n, DIM))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def fill(idx, vectors, start=0):
    for i, v in enumerate(vectors):
        idx.add(v, dict(ts=1_700_000_000 + 100.0 * (start + i), path=f"images/{start + i}.jpg", app=None, title=None))


def snapshot(folder):
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in folder.iterdir() if p.is_file()}


def ann_count(idx):
    return idx.ann.p.get_current_count() if idx.type == "hnsw" else idx.ann.count


@pytest.fixture
def small(monkeypatch):
    from streamembed import ann
    monkeypatch.setattr(ann, "FROM", 20)
    return ann


@pytest.mark.parametrize("kind", ANN)
def test_ann_switches_from_exact_and_scores_exactly(tmp_path, small, kind):
    v = clustered(60)
    idx = index_mod.Index(tmp_path / "idx", kind, "fake")
    fill(idx, v[:20])
    assert idx.ann is None, "exact search up to ann.FROM rows"
    fill(idx, v[20:], 20)
    assert idx.ann is not None and ann_count(idx) == 60, "every new row is searchable"
    for row in (5, 37, 59):
        hits = idx.search(v[row], 5)
        assert hits[0][0] == row
        exact = v[[r for r, _ in hits]] @ v[row]
        assert np.allclose([s for _, s in hits], exact, atol=2e-3), "scores are true cosines"
    idx.close()
    assert all((tmp_path / "idx" / f).exists() for f in ANN_FILES[kind])


@pytest.mark.parametrize("kind", ANN)
def test_ann_reopen_with_each_type(tmp_path, small, kind):
    v = clustered(50)
    idx = index_mod.Index(tmp_path / "idx", kind, "fake")
    fill(idx, v)
    idx.close()
    for other in ["flat", "binary", "ivf", "pq"] + (["hnsw"] if importlib.util.find_spec("hnswlib") else []) + [kind]:
        again = index_mod.Index(tmp_path / "idx", other, "fake")
        assert len(again) == 50 and again.search(v[33], 1)[0][0] == 33, other
        again.close()
    again = index_mod.Index(tmp_path / "idx", kind, "fake")
    assert ann_count(again) == 50, "loaded from its files"


@pytest.mark.parametrize("kind", ANN)
def test_ann_reader_sees_new_rows_and_writes_nothing(tmp_path, small, kind):
    v = clustered(70)
    writer = index_mod.Index(tmp_path / "idx", kind, "fake")
    fill(writer, v[:40])
    reader = index_mod.Index(tmp_path / "idx", kind, "fake")
    assert reader.readonly and ann_count(reader) == 40
    fill(writer, v[40:], 40)                    # hnsw.bin now lags (saved every ann.SAVE_EVERY rows): the reader catches up
    before = snapshot(tmp_path / "idx")
    reader.refresh()
    assert len(reader) == 70 and ann_count(reader) == 70 and reader.search(v[66], 1)[0][0] == 66
    assert snapshot(tmp_path / "idx") == before, "a reader changes no file"
    reader.close()
    writer.close()


@pytest.mark.parametrize("kind", ANN)
def test_ann_reader_builds_in_memory_when_files_are_missing(tmp_path, small, kind):
    v = clustered(40)
    writer = index_mod.Index(tmp_path / "idx", "flat", "fake")         # e.g. a capture service writing a "flat" index
    fill(writer, v)
    before = snapshot(tmp_path / "idx")
    reader = index_mod.Index(tmp_path / "idx", kind, "fake")
    assert reader.readonly and ann_count(reader) == 40 and reader.search(v[12], 1)[0][0] == 12
    assert snapshot(tmp_path / "idx") == before and not any((tmp_path / "idx" / f).exists() for f in ANN_FILES[kind])
    writer.close()


@pytest.mark.parametrize("kind", ANN)
def test_ann_catches_up_after_a_crash(tmp_path, small, kind):
    v = clustered(70)
    folder = tmp_path / "idx"
    idx = index_mod.Index(folder, kind, "fake")
    fill(idx, v[:40])
    idx.close()
    idx = index_mod.Index(folder, kind, "fake")
    fill(idx, v[40:60], 40)
    idx._lockfile.close()                       # the writer dies: no close(), so hnsw.bin misses rows 40-59
    if kind != "hnsw":
        with open(folder / f"{kind}_codes.bin", "r+b") as f:     # codes lag too, and the last record is half written
            f.truncate((50 * idx.ann.width) + 3)
    other = index_mod.Index(folder, "flat", "fake")              # rows added while another type was in use
    fill(other, v[60:], 60)
    with open(folder / "vectors.f16", "ab") as f:                # and a vector without its line
        np.zeros(DIM, np.float16).tofile(f)
    other._lockfile.close()
    idx = index_mod.Index(folder, kind, "fake")
    assert len(idx) == 70 and ann_count(idx) == 70
    assert idx.search(v[45], 1)[0][0] == 45 and idx.search(v[65], 1)[0][0] == 65
    if kind != "hnsw":
        assert (folder / f"{kind}_codes.bin").stat().st_size == 70 * idx.ann.width
    idx.close()
    if kind == "hnsw":
        assert ann_count(index_mod.Index(folder, kind, "fake")) == 70, "the caught-up graph was saved"


@pytest.mark.parametrize("kind", ["ivf", "pq"])
def test_ivf_pq_retrain_after_4x_growth(tmp_path, small, kind):
    v = clustered(100)
    idx = index_mod.Index(tmp_path / "idx", kind, "fake")
    fill(idx, v[:21])
    assert idx.ann.meta["trained_n"] == 21 and len(idx.ann.model["centroids"]) == 5      # round(sqrt(21)) lists
    fill(idx, v[21:83], 21)
    assert idx.ann.meta["trained_n"] == 21
    fill(idx, v[83:], 83)
    assert idx.ann.meta["trained_n"] == 84 and len(idx.ann.model["centroids"]) == 9
    assert json.loads((tmp_path / "idx" / f"{kind}.json").read_text())["trained_n"] == 84
    assert (tmp_path / "idx" / f"{kind}_codes.bin").stat().st_size == 100 * idx.ann.width
    assert idx.search(v[90], 1)[0][0] == 90


@pytest.mark.parametrize("kind", ANN)
def test_ann_through_the_api(tmp_path, enc, small, kind):
    se.configure(index_dir=tmp_path / "idx", index_type=kind, width=256, encoder=enc)
    small.FROM = 3
    se.build_index_from_directory(screenshots(tmp_path / "shots", 6))
    assert len(se.query_index("x", top_k=2)) == 2 and se.api._state["index"].ann is not None


def test_hnsw_without_hnswlib_says_what_to_install(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "hnswlib", None)           # import hnswlib raises ImportError
    with pytest.raises(ImportError, match=r"streamembed\[hnsw\]"):
        se.configure(index_dir=tmp_path, index_type="hnsw")
    with pytest.raises(ImportError, match=r"streamembed\[hnsw\]"):
        index_mod.Index(tmp_path / "idx", "hnsw", "fake")


def test_screen_frames_survives_a_failed_grab(monkeypatch):
    """No screen to grab (a lock screen, a display gone) gives None and is tried again, instead of ending the capture."""
    from streamembed import capture
    tries = []

    def no_screen():
        tries.append(1)
        raise RuntimeError("Cannot connect to display")

    monkeypatch.setattr(capture, "_grabber", lambda: no_screen)
    monkeypatch.setattr(capture, "_locked", lambda: False)
    monkeypatch.setattr(capture, "_front_window", lambda: (None, None))
    frames = capture.screen_frames(256, "primary")
    assert next(frames) is None and next(frames) is None and len(tries) == 2


# ---- the index folder survives crashes and partial writes
def test_torn_last_lines_are_left_out(tmp_path):
    v = clustered(4)
    idx = index_mod.Index(tmp_path / "idx", "flat", "fake")
    fill(idx, v[:3])
    idx.still_on_screen(1_700_000_050.0, 0)
    reader = index_mod.Index(tmp_path / "idx", "flat", "fake")          # while the writer is open
    with open(tmp_path / "idx" / "frames.jsonl", "a") as f:
        f.write('{"ts": 3.0, "pa')                                     # a line still being written
    with open(tmp_path / "idx" / "seen.jsonl", "a") as f:
        f.write('not json\n[1, 1700000')
    reader.refresh()
    assert len(reader) == 3 and reader.readonly
    assert len(index_mod.Index(tmp_path / "idx", "flat", "fake")) == 3, "a reader opens it"
    idx.close()
    again = index_mod.Index(tmp_path / "idx", "flat", "fake")          # the writer cuts the partial lines
    assert len(again) == 3 and again.frames[0]["ts_last"] == 1_700_000_050.0
    assert (tmp_path / "idx" / "frames.jsonl").read_text().endswith("}\n")
    assert (tmp_path / "idx" / "seen.jsonl").read_text().endswith("\n")
    again.add(v[3], dict(ts=1_700_000_300.0, path="images/3.jpg"))
    assert len(index_mod.Index(tmp_path / "idx", "flat", "fake")) == 4


def test_opening_rewrites_nothing_that_is_whole(tmp_path):
    idx = index_mod.Index(tmp_path / "idx", "flat", "fake")
    fill(idx, clustered(5))
    idx.close()
    before = snapshot(tmp_path / "idx")
    index_mod.Index(tmp_path / "idx", "flat", "fake").close()
    after = snapshot(tmp_path / "idx")
    assert {k: v for k, v in after.items() if k != "writer.lock"} == {k: v for k, v in before.items() if k != "writer.lock"}
    (tmp_path / "idx" / "index.json").write_text("")                   # cut by a crash: written again
    idx = index_mod.Index(tmp_path / "idx", "flat", "fake")
    assert len(idx) == 5 and json.loads((tmp_path / "idx" / "index.json").read_text()) == {"encoder": "fake", "dim": DIM}


def test_a_failed_add_does_not_shift_later_rows(tmp_path, monkeypatch):
    import builtins
    v = clustered(4)
    idx = index_mod.Index(tmp_path / "idx", "flat", "fake")
    fill(idx, v[:1])
    real_open, fail = builtins.open, [True]

    def full_disk(path, mode="r", *a, **k):
        if str(path).endswith("frames.jsonl") and mode == "a" and fail:
            fail.clear()
            raise OSError(28, "No space left on device")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(index_mod, "open", full_disk, raising=False)
    with pytest.raises(OSError):
        idx.add(v[1], dict(ts=1_700_000_100.0, path="images/1.jpg"))
    fill(idx, v[2:], 2)
    assert len(idx) == 3 and idx.search(v[3], 1)[0][0] == 2 and idx.search(v[3], 1)[0][1] > 0.99, "row 2 holds its own vector"
    idx.close()
    again = index_mod.Index(tmp_path / "idx", "flat", "fake")
    assert len(again) == 3 and again.search(v[3], 1)[0][0] == 2 and again.search(v[0], 1)[0][0] == 0


def test_a_query_does_not_hold_the_write_lock(tmp_path, enc):
    se.build_index_from_directory(screenshots(tmp_path / "shots", 3))
    se.configure(index_dir=tmp_path / "idx", width=256, encoder=enc)
    assert len(se.query_index("x")) == 3 and se.api._state["index"].readonly
    other = index_mod.Index(tmp_path / "idx", "flat", "fake")
    assert not other.readonly, "a process that only queried leaves the lock free"
    assert len(se.query_index("x")) == 3
    with pytest.raises(RuntimeError, match="another process"):
        se.build_index_from_directory(screenshots(tmp_path / "more", 2, start=1_800_000_000))
    other.close()
    stats = se.build_index_from_directory(tmp_path / "more")           # the lock is free again: taken now
    assert stats["encoded"] == 2 and se.api._state["index"] is None, "taken while indexing, released after"
    assert len(se.query_index("x")) == 5 and se.api._state["index"].readonly


def test_inputs_are_checked_before_the_model_loads(tmp_path, enc, monkeypatch):
    folder = screenshots(tmp_path / "shots", 3)
    se.build_index_from_directory(folder)
    se.configure(index_dir=tmp_path / "idx")
    from types import SimpleNamespace
    monkeypatch.setattr(se.api, "_encoder", lambda: pytest.fail("the model was loaded"))
    monkeypatch.setattr(se.api, "resolve", lambda *a: SimpleNamespace(encoder_name="fake"))
    again = se.build_index_from_directory(folder)
    assert again["encoded"] == 0 and again["done"] == 3 and again["todo"] == 0, "nothing new"
    with pytest.raises(FileNotFoundError, match="Folder not found"):
        se.build_index_from_directory(tmp_path / "nonexist")
    with pytest.raises(ValueError, match="since"):
        se.query_index("x", since="yesterday")
    se.configure(index_dir=tmp_path / "empty")
    assert se.query_index("anything") == [] and not (tmp_path / "empty").exists(), "an empty index: no model, no folder"


def test_progress_and_until_a_date(tmp_path, enc):
    seen = []
    stats = se.build_index_from_directory(screenshots(tmp_path / "shots", 4), progress=lambda st: seen.append(st["captured"]))
    assert seen == [1, 2, 3, 4] and stats["todo"] == 4
    from datetime import datetime
    assert se.api._when("2026-11-03", "until") == datetime(2026, 11, 4).timestamp(), "a date alone includes that day"
    assert se.api._when("2026-11-03 14:00", "until") == datetime(2026, 11, 3, 14).timestamp()


def test_dedupe_and_context_stay_within_one_screen(tmp_path, enc):
    """Two recordings of one size in two subfolders, over the same seconds: two streams, and a match from each."""
    t = 1_700_000_000
    for rec in ("rec1", "rec2"):
        (tmp_path / "shots" / rec).mkdir(parents=True)
        for i in range(3):
            noise(10 * (rec == "rec2") + i).save(tmp_path / "shots" / rec / f"{t + 4 * i}.png")
    stats = se.build_index_from_directory(tmp_path / "shots", interval=0)
    assert stats["encoded"] == 6 and enc.resets == 2, "each subfolder is its own stream"
    res = se.query_index("x", top_k=10)
    assert len(res) == 2 and {os.path.basename(os.path.dirname(r.source)) for r in res} == {"rec1", "rec2"}
    # a folder copied in one go (no time in the names): file times are copy times, so nothing is folded
    copied = tmp_path / "copied"
    copied.mkdir()
    for i in range(4):
        noise(50 + i).save(copied / f"shot_{i}.png")
        os.utime(copied / f"shot_{i}.png", (t + 1000 + 0.1 * i,) * 2)
    se.build_index_from_directory(copied)
    assert len(se.query_index("x", top_k=10)) == 2 + 4
    # live frames afterwards: the viewer's context for a live match holds live frames only, in time order
    live = [(noise(70 + i), time.time() + i) for i in range(3)]
    se.build_index_from_now(interval=0.01, frames=iter(live)).wait()
    idx = se.api._index()
    last = len(idx) - 1
    assert idx.around(last, 15) == [last - 2, last - 1, last]
    rec1 = [i for i, f in enumerate(idx.frames) if "rec1" in (f.get("source") or "")]
    ctx = idx.around(rec1[1], 15)
    assert ctx == rec1 and [idx.frames[i]["ts"] for i in ctx] == sorted(idx.frames[i]["ts"] for i in ctx)


def test_capture_timer_errors_and_monitors(tmp_path, enc, capsys):
    def endless():
        i = 0
        while True:
            i += 1
            yield (noise(i), time.time())

    with pytest.raises(ValueError, match="monitors"):
        se.build_index_from_now(monitors=[0], frames=endless())        # checked before the thread starts
    cap = se.build_index_from_now(interval=0.02, duration=3600, frames=endless())
    cap.stop()
    time.sleep(0.05)
    assert cap._timer.daemon and not cap._timer.is_alive(), "stop() cancels the duration timer"

    class Broken(FakeEncoder):
        def encode_frame(self, path, stream=0):
            raise RuntimeError("disk full")

    se.configure(index_dir=tmp_path / "idx", width=256, encoder=Broken())
    cap = se.build_index_from_now(interval=0.01, frames=iter([(noise(1), time.time()), (noise(2), time.time() + 1)]))
    cap.wait()
    err = capsys.readouterr().err
    assert cap.stats["errors"] == 2 and "could not be stored" in err and "disk full" in err


def test_command_line_search_and_pause(tmp_path, svc, capsys, monkeypatch, enc):
    from streamembed import cli
    service, calls, platform = svc
    platform("darwin")
    idx = str(tmp_path / "idx")
    assert cli.main(["search", "anything", "--index-dir", str(tmp_path / "none"), "--no-open"]) == 0
    assert "does not exist" in capsys.readouterr().out
    (tmp_path / "empty").mkdir()
    assert cli.main(["search", "anything", "--index-dir", str(tmp_path / "empty"), "--no-open"]) == 0
    assert "No screens in" in capsys.readouterr().out
    assert not (tmp_path / "none").exists()
    se.configure(index_dir=tmp_path / "idx", width=256, encoder=enc)
    se.build_index_from_directory(screenshots(tmp_path / "shots", 3))      # in 2023
    se.configure(index_dir=tmp_path / "other")
    monkeypatch.setattr(se.api, "_encoder", lambda: enc)
    monkeypatch.setattr(se.api, "resolve", lambda *a: type("S", (), dict(encoder_name="fake")))
    assert cli.main(["search", "the page", "--since", "2099-01-01", "--index-dir", idx, "--no-open"]) == 0
    out = capsys.readouterr().out
    assert "No screens from" in out and "'the page'" in out
    assert cli.main(["search", "the page yesterday", "--index-dir", idx, "--no-open"]) == 0
    assert "page:" in capsys.readouterr().out, "time words are part of the question, not a filter"
    assert cli.main(["index", str(tmp_path / "shots"), "--index-dir", idx]) == 0
    assert "Nothing new: all 3 screenshots were read in earlier runs." in capsys.readouterr().out
    (tmp_path / "rec").mkdir()
    assert cli.main(["start", "--index-dir", idx]) == 0
    se.configure(index_dir=tmp_path / "other")
    assert cli.main(["pause"]) == 0 and (tmp_path / "idx" / "paused").exists(), "the service's folder, not the configured one"
    assert cli.main(["resume"]) == 0 and not (tmp_path / "idx" / "paused").exists()
    with pytest.raises(FileNotFoundError, match="Folder not found"):
        service.install_service(index_dir=idx, directory=tmp_path / "nonexist")
    cli.main(["stop"])


# ---- indexing a folder again, and folders that cannot be indexed
def _truncated_png(path, seed=9):
    import io
    buf = io.BytesIO()
    noise(seed).save(buf, "PNG")
    path.write_bytes(buf.getvalue()[:200])           # the header reads, the pixels do not


def test_a_second_run_reads_nothing_and_counts_add_up(tmp_path, enc, monkeypatch, capsys):
    """A folder that ends in unchanged, slightly changed and broken screenshots: the second run has nothing to do and loads no model."""
    folder = screenshots(tmp_path / "shots", 3)                          # shot_001 (the last) is noise(2)
    tail = [("same_a.png", noise(2))]
    little = np.asarray(noise(2)).copy()
    little[:32, :32] = 0                                                 # 1 block of 32 changed: below the gate's 10 %
    tail += [("little.png", Image.fromarray(little)), ("same_b.png", noise(2))]
    for i, (name, img) in enumerate(tail):
        img.save(folder / name)
        os.utime(folder / name, (1_700_000_205 + 5 * i,) * 2)          # within the gate's 30-s heartbeat
    _truncated_png(folder / "broken.png")
    os.utime(folder / "broken.png", (1_700_000_900,) * 2)
    (folder / "notes.png").write_text("not an image")
    first = se.build_index_from_directory(folder)
    assert (first["found"], first["encoded"], first["idle"], first["minor"], first["errors"]) == (8, 3, 2, 1, 2), first
    assert first["found"] == first["skipped"] + first["done"] + first["encoded"] + first["idle"] + first["minor"] + first["errors"]
    rows = (tmp_path / "idx" / "frames.jsonl").read_text()
    se.configure(index_dir=tmp_path / "idx", width=256)
    from types import SimpleNamespace
    monkeypatch.setattr(se.api, "_encoder", lambda: pytest.fail("the model was loaded"))
    monkeypatch.setattr(se.api, "resolve", lambda *a: SimpleNamespace(encoder_name="fake"))
    monkeypatch.chdir(tmp_path)
    again = se.build_index_from_directory("shots")                       # another spelling of the folder
    assert (again["todo"], again["done"], again["encoded"], again["errors"]) == (0, 7, 0, 1), again
    assert again["found"] == again["skipped"] + again["done"] + again["errors"]
    assert (tmp_path / "idx" / "frames.jsonl").read_text() == rows
    from streamembed import cli
    assert cli.main(["index", "shots", "--index-dir", str(tmp_path / "idx")]) == 0
    out = capsys.readouterr().out
    assert "0 screens indexed, 1 could not be read or stored, 7 read in earlier runs (8 files)" in out, out


def test_relative_and_absolute_folder_are_one_folder(tmp_path, enc, monkeypatch):
    """Sources are stored as full paths; an index written with relative ones (before) continues too."""
    screenshots(tmp_path / "shots", 3)
    monkeypatch.chdir(tmp_path)
    se.build_index_from_directory("shots")
    meta = tmp_path / "idx" / "frames.jsonl"
    lines = [json.loads(l) for l in meta.read_text().splitlines()]
    assert all(os.path.isabs(l["source"]) for l in lines)
    assert se.build_index_from_directory(tmp_path / "shots")["encoded"] == 0
    for l in lines:                                                      # as written before folders were resolved
        l["source"] = os.path.relpath(l["source"], tmp_path)
    meta.write_text("".join(json.dumps(l) + "\n" for l in lines))
    se.configure(index_dir=tmp_path / "idx", width=256, encoder=enc)
    again = se.build_index_from_directory(tmp_path / "shots")
    assert again["done"] == 3 and again["encoded"] == 0


def test_empty_file_and_unreadable_folders(tmp_path, enc, capsys):
    (tmp_path / "empty").mkdir()
    st = se.build_index_from_directory(tmp_path / "empty")
    assert st["found"] == 0 and st["encoded"] == 0 and not st["stopped"], "a folder with no screenshots is not an error"
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(NotADirectoryError, match="is a file, not a folder"):
        se.build_index_from_directory(tmp_path / "a.txt")
    from streamembed import cli
    assert cli.main(["index", str(tmp_path / "empty"), "--index-dir", str(tmp_path / "idx")]) == 0
    assert "No screenshots" in capsys.readouterr().out
    assert cli.main(["search", "x", "--index-dir", str(tmp_path / "a.txt")]) == 2
    assert "is a file, not a folder" in capsys.readouterr().err
    se.configure(index_dir=tmp_path / "idx", width=256, encoder=enc)      # the commands above configured their own
    if os.name != "posix" or os.geteuid() == 0:
        return
    locked = tmp_path / "locked"
    screenshots(locked / "sub", 2)
    screenshots(locked, 1)
    (locked / "sub").chmod(0)
    try:
        st = se.build_index_from_directory(locked)
        assert st["encoded"] == 1 and "could not be read" in capsys.readouterr().err, "an unreadable subfolder is named"
        locked.chmod(0)
        with pytest.raises(PermissionError, match="permission denied"):
            se.build_index_from_directory(locked)
    finally:
        locked.chmod(0o755)
        (locked / "sub").chmod(0o755)


def test_a_paused_index_says_so(tmp_path, enc, capsys):
    import threading
    se.pause()
    threading.Timer(0.3, se.resume).start()
    st = se.build_index_from_directory(screenshots(tmp_path / "shots", 2))
    assert st["encoded"] == 2 and "is paused" in capsys.readouterr().err


# ---- a screen grab that does not return
def test_a_stuck_grab_is_reported_and_left_behind(tmp_path, enc, monkeypatch, capsys):
    import threading
    monkeypatch.setattr(se.api, "STALL_S", 0.4)
    release = threading.Event()

    def stuck():
        yield (noise(1), time.time())
        release.wait(10)                                                # the grab hangs
        yield (noise(2), time.time())

    cap = se.build_index_from_now(interval=0.01, frames=stuck())
    cap.wait()                                                          # returns once the grab has not returned for STALL_S
    assert cap.stalled and cap.running and cap.stats["encoded"] == 1
    time.sleep(0.3)
    assert "has not returned" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="stuck in a screen grab"):
        se.build_index_from_now(interval=0.01, frames=iter([]))
    assert cap.stop(timeout=0.2) is False and cap.abandoned
    new = se.build_index_from_now(interval=0.01, frames=iter([(noise(3), time.time() + 5)]))
    new.wait()
    release.set()                                                       # the old grab returns: it stores nothing
    time.sleep(0.2)
    assert not cap.running and cap.stats["encoded"] == 1 and new.stats["encoded"] == 1
    assert len(index_mod.Index(tmp_path / "idx", "flat", "fake", write=False)) == 2


def test_stop_waits_for_an_encode(tmp_path, monkeypatch):
    """stop() gives up only on a grab; a screenshot being encoded is always finished (the index stays locked meanwhile)."""
    class Slow(FakeEncoder):
        def encode_frame(self, path, stream=0):
            time.sleep(0.5)
            return super().encode_frame(path, stream)

    se.configure(index_dir=tmp_path / "idx", width=256, encoder=Slow())
    cap = se.build_index_from_now(interval=0.01, frames=iter([(noise(1), time.time())]))
    time.sleep(0.1)
    assert cap.stop(timeout=0.05) is True and cap.stats["encoded"] == 1
    se.configure(index_dir=tmp_path / "idx")


# ---- the service process
def _service_env(monkeypatch, tmp_path, frames):
    """service.main with the stand-in encoder and `frames` as the screen."""
    from streamembed import service
    enc = FakeEncoder()
    real = se.configure

    def configure(**kw):
        real(**{**kw, "encoder": enc, "width": 256})

    monkeypatch.setattr(se, "configure", configure)
    monkeypatch.setattr(se.api, "screen_permission", lambda: True)
    monkeypatch.setattr(se.api, "screen_frames", lambda width, monitors: frames)
    return service, enc


def test_service_main_folder_then_capture(tmp_path, monkeypatch, capsys):
    import signal
    import threading
    def endless():
        i = 0
        while True:
            i += 1
            yield [(noise(100 + i), time.time(), 1, None, "")]

    service, enc = _service_env(monkeypatch, tmp_path, endless())
    screenshots(tmp_path / "shots", 2)
    threading.Timer(1.0, signal.raise_signal, (signal.SIGTERM,)).start()
    code = service.main(["--index-dir", str(tmp_path / "idx"), "--interval", "0.5", "--folder", str(tmp_path / "shots")])
    out = capsys.readouterr().out
    assert code == 0 and "folder" in out and "capturing every 0.5 s" in out and "stopped" in out
    assert enc.resets >= 2, "the folder, then the screen"
    se.configure(index_dir=tmp_path / "idx")


def test_service_main_exit_codes(tmp_path, monkeypatch, capsys):
    service, enc = _service_env(monkeypatch, tmp_path, iter([]))       # a screen that ends at once: as an error would
    idx = str(tmp_path / "idx")
    assert service.main(["--index-dir", idx, "--index-type", "annoy"]) == 0, "a bad setting: not restarted"
    assert "Fix the setting" in capsys.readouterr().out
    assert service.main(["--index-dir", idx, "--folder", str(tmp_path / "gone")]) == 1, "capture ended by itself: restarted"
    out = capsys.readouterr().out
    assert "capturing without it" in out and "ended with an error" in out
    monkeypatch.setattr(se.api, "STALL_S", 0.3)
    import threading
    hang = threading.Event()

    def stuck():
        hang.wait(5)
        yield None

    monkeypatch.setattr(se.api, "screen_frames", lambda width, monitors: stuck())
    assert service.main(["--index-dir", idx]) == 1, "a stuck grab: exit so the service manager starts it again"
    assert "has not returned" in capsys.readouterr().out
    hang.set()
    se.configure(index_dir=tmp_path / "idx")


def test_service_gets_the_settings_file(tmp_path, svc, monkeypatch):
    import plistlib
    service, calls, platform = svc
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STREAMEMBED_CONFIG", "mine.toml")
    platform("darwin")
    p = plistlib.loads(service.install_service(index_dir=tmp_path / "idx").read_bytes())
    assert p["EnvironmentVariables"]["STREAMEMBED_CONFIG"] == str((tmp_path / "mine.toml").resolve())
    monkeypatch.setenv("DISPLAY", ":0")
    platform("linux")
    assert "STREAMEMBED_CONFIG=" in service.install_service(index_dir=tmp_path / "idx").read_text()
    platform("win32")
    assert "STREAMEMBED_CONFIG=" in service.install_service(index_dir=tmp_path / "idx").read_text(encoding="utf-16")


# ---- the results page and the command line
def test_the_page_shows_its_own_question(tmp_path, enc):
    se.build_index_from_directory(screenshots(tmp_path / "shots", 3))
    a = se.query_index("flight page", since="2020-01-01")
    se.query_index("slides")
    page = se.visualize_results(a, open_browser=False).read_text()
    assert '"query": "flight page"' in page and "<title>flight page" in page
    assert a[:2].query == "flight page" and a.text == "flight page" and a.window[0] is not None
    page = se.visualize_results(list(a), open_browser=False, query="mine").read_text()
    assert '"query": "mine"' in page and '"window": null' in page
    assert "[hidden] { display: none !important; }" in page, "Show more and the date bar hide"


def test_command_line_checks_top_k_and_config_words(tmp_path, capsys):
    from streamembed import cli
    for argv in (["search", "x", "--top-k", "0"], ["search", "x", "--top-k", "-1"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert "--top-k" in capsys.readouterr().err
    for argv, says in ((["config", "set", "interval"], "'interval' has no value"), (["config", "show", "extra"], "no more words"),
                       (["config", "path", "foo"], "no more words")):
        assert cli.main(argv) == 2
        err = capsys.readouterr().err
        assert err.startswith("streamembed: error:") and says in err, err


def test_capture_time_from_chinese_names():
    from datetime import datetime
    from streamembed.capture import _name_time
    assert _name_time("截屏2026-11-06 下午3.20.11") == (datetime(2026, 11, 6, 15, 20, 11).timestamp(), "")
    assert _name_time("截屏2026-11-06 上午3.20.11") == (datetime(2026, 11, 6, 3, 20, 11).timestamp(), "")
    assert _name_time("截屏2026-11-06 上午12.05.00")[0] == datetime(2026, 11, 6, 0, 5).timestamp()


def test_an_empty_question_is_refused_before_the_model_loads(tmp_path, enc):
    se.build_index_from_directory(screenshots(tmp_path / "shots", 2))
    for q in ("", "   "):
        with pytest.raises(ValueError, match="a question is needed"):
            se.query_index(q)
    assert se.query_index("the page I saw yesterday").text == "the page I saw yesterday", "the question is used as it is"


class _FakeShot:
    size = (64, 32)
    rgb = bytes(64 * 32 * 3)


class _FakeMSS:
    monitors = [dict(left=0, top=0, width=64, height=32)] * 2

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def grab(self, m):
        return _FakeShot()


def test_index_folder_inside_the_scanned_folder_is_left_out(tmp_path, enc):
    """~/ (scanned) holds ~/screenshot_index: the index's own stored captures are not indexed again, and the index folder itself
    cannot be given as the folder."""
    home = tmp_path / "home"
    screenshots(home / "Pictures" / "Screenshots", 3)
    se.configure(index_dir=home / "screenshot_index", width=256, encoder=enc)
    cap = se.build_index_from_now(interval=0.01, frames=iter([(noise(50 + i), time.time() + i) for i in range(2)]))
    cap.wait()
    assert len(list((home / "screenshot_index" / "captures").rglob("*.webp"))) == 2, "live captures are stored in the index"
    stats = se.build_index_from_directory(home)
    assert stats["found"] == 3 and stats["encoded"] == 3, "the index's own stored captures are not screenshots to index"
    with pytest.raises(ValueError, match="inside the index folder"):
        se.build_index_from_directory(home / "screenshot_index" / "captures")


def test_capture_format_and_width(tmp_path, enc):
    """WebP at 1280 by default; jpeg and the screen's own size on request. The encoder always gets the image fitted to `width`."""
    from PIL import Image
    big = noise(1, 2560, 1600)
    se.configure(index_dir=tmp_path / "a", width=256, encoder=enc)
    se.build_index_from_now(interval=0.01, frames=iter([(big, time.time())])).wait()
    saved = next((tmp_path / "a" / "captures").rglob("*.*"))
    assert saved.suffix == ".webp" and Image.open(saved).width == 1280
    se.set_config(capture_format="jpeg", capture_width=0)
    se.configure(index_dir=tmp_path / "b", width=256, encoder=enc)
    se.build_index_from_now(interval=0.01, frames=iter([(big, time.time())])).wait()
    saved = next((tmp_path / "b" / "captures").rglob("*.*"))
    assert saved.suffix == ".jpg" and Image.open(saved).size == (2560, 1600)
    with pytest.raises(ValueError, match="capture_format"):
        se.set_config(capture_format="png")
    with pytest.raises(ValueError, match="capture_width"):
        se.set_config(capture_width=10)


def test_configure_sets_how_captures_are_saved(tmp_path, enc):
    from PIL import Image
    big = noise(2, 2560, 1600)
    se.configure(index_dir=tmp_path / "a", width=256, capture_width=640, capture_format="jpeg", encoder=enc)
    se.build_index_from_now(interval=0.01, frames=iter([(big, time.time())])).wait()
    saved = next((tmp_path / "a" / "captures").rglob("*.*"))
    assert saved.suffix == ".jpg" and Image.open(saved).width == 640
    assert se.get_config("capture_width") == 1280, "configure() changes this process only; the saved setting stays"
    with pytest.raises(ValueError, match="capture_width"):
        se.configure(index_dir=tmp_path / "b", capture_width=10, encoder=enc)
    se.configure(index_dir=tmp_path / "c", encoder=enc)
    se.build_index_from_now(interval=0.01, frames=iter([(big, time.time())])).wait()
    assert Image.open(next((tmp_path / "c" / "captures").rglob("*.*"))).width == 1280, "back to the setting (1280)"


def test_heartbeat_stores_a_slightly_changed_screen_at_most_that_often():
    import numpy as np
    from PIL import Image
    from streamembed.gate import Gate
    base = np.random.default_rng(0).integers(0, 255, (128, 256, 3), np.uint8)
    def frame(k):                                   # one 32x32 block changed: under 10 % of the blocks
        a = base.copy()
        a[:32, :32] = (k * 50) % 255
        return Image.fromarray(a)
    for beat, want in ((30, [True, False, True, False, True]), (300, [True, False, False, False, False])):
        g = Gate(heartbeat=beat)
        got = [g(frame(k), t).encode for k, t in enumerate((0, 20, 40, 60, 80))]
        assert got == want, (beat, got)


def test_query_server_round_trip(tmp_path):
    from streamembed import queryserver
    root = tmp_path / ("a-long-folder-name-" * 6)                     # longer than a Unix socket path allows: temp-folder socket
    root.mkdir()
    stop = queryserver.serve(root, lambda text: np.full(8, len(text), np.float32), "fake")
    try:
        vec, name = queryserver.ask(root, "hello")
        assert name == "fake" and vec.tolist() == [5.0] * 8
    finally:
        stop()
    assert queryserver.ask(root, "hello") is None, "nothing answers once stopped"


def test_search_uses_the_capturing_process_model(tmp_path, enc, monkeypatch):
    """Another process (the service) serves questions for the index: search uses its vector and loads no model of its own."""
    from streamembed import queryserver
    se.build_index_from_directory(screenshots(tmp_path / "shots", 4))
    se.configure(index_dir=tmp_path / "idx", width=256)                  # no stand-in encoder: a search would load the model
    monkeypatch.setattr(se.api, "resolve", lambda *a: type("S", (), dict(encoder_name="fake")))
    def no_model():
        raise AssertionError("the model was loaded")
    monkeypatch.setattr(se.api, "_encoder", no_model)
    stop = queryserver.serve(tmp_path / "idx", enc.encode_query, "fake")
    try:
        assert len(se.query_index("a page", top_k=3)) == 3
    finally:
        stop()
    with pytest.raises(AssertionError, match="model was loaded"):
        se.query_index("a page", top_k=3)                                 # nobody serves: the model loads here
    stop = queryserver.serve(tmp_path / "idx", enc.encode_query, "another-model")
    try:
        with pytest.raises(AssertionError, match="model was loaded"):
            se.query_index("a page", top_k=3)                             # a different model's vector is not used
    finally:
        stop()


def test_the_served_search_page(tmp_path, enc, monkeypatch, capsys):
    """The page the background capture serves: search with this process's model, images by row, a token on every request, and a
    Host check; `streamembed search` uses it without loading a model."""
    import http.client
    import urllib.request
    from urllib.parse import urlparse
    from streamembed import cli
    se.build_index_from_directory(screenshots(tmp_path / "shots", 4))
    url, stop = se.api.serve_page()
    try:
        u = urlparse(url)
        token = u.query.split("=", 1)[1]
        get = lambda path: urllib.request.urlopen(f"http://{u.netloc}{path}", timeout=10)     # noqa: E731
        page = get(f"/?t={token}").read().decode()
        assert '"token": "' in page and "id=\"qbox\"" in page and '"screens": 4' in page
        assert json.loads(get(f"/api/ping?t={token}").read()) == {"ok": True, "screens": 4}
        data = json.loads(get(f"/api/search?t={token}&q=a+page&k=3").read())
        assert len(data["results"]) == 3 and data["results"][0]["frames"][0]["url"].startswith("img/")
        assert get(f"/{data['results'][0]['frames'][0]['url']}").read()[:4] == b"\x89PNG", "the original image, by row"
        assert get(f"/thumb/{data['results'][0]['id']}?t={token}").read()[:2] == b"\xff\xd8"
        for path in (f"/api/search?t=wrong&q=x", "/img/0", f"/img/../../etc/passwd?t={token}"):
            with pytest.raises(urllib.error.HTTPError):
                get(path)
        c = http.client.HTTPConnection(u.netloc, timeout=10)
        c.request("GET", f"/?t={token}", headers={"Host": "evil.example:80"})
        assert c.getresponse().status == 403, "another Host (DNS rebinding) is refused"
        opened = []
        monkeypatch.setattr("webbrowser.open", lambda link: opened.append(link))
        capsys.readouterr()
        asked = []
        real_query = enc.encode_query
        monkeypatch.setattr(enc, "encode_query", lambda text: asked.append(text) or real_query(text))
        assert cli.main(["search", "a page", "--since", "2020-01-01"]) == 0
        out = capsys.readouterr().out
        assert ".png" not in out and "served by the background capture" in out
        assert opened and "q=a+page" in opened[-1] and "since=2020-01-01" in opened[-1] and token in opened[-1]
        assert asked == [], "the page searches, once: the command does not search too"
        assert cli.main(["search", "a page", "--top-k", "2", "--no-open"]) == 0
        assert capsys.readouterr().out.count(".png") == 2 and asked == ["a page"], "--no-open: searched and printed here"
        assert cli.main(["search"]) == 0 and "q=" not in opened[-1], "no question: the empty search page"
    finally:
        stop()
    assert not (tmp_path / "idx" / "web.json").exists()


def test_long_digit_names_are_not_read_as_dates():
    from datetime import datetime
    from streamembed.capture import _name_time
    assert _name_time("1794030871211_120") == (1794030871.211, ""), "<ms>_<row> from captures stored before 0.3.0"
    assert _name_time("1794030871211_1") == (1794030871.211, "1")
    assert _name_time("17940308712119") == (None, ""), "a long number is not a date in 1794"
    assert _name_time("Screenshot 2026-11-06 at 15.20.11")[0] == datetime(2026, 11, 6, 15, 20, 11).timestamp()


def test_screenshots_added_later_are_indexed_even_if_older(tmp_path, enc):
    folder = tmp_path / "shots"
    folder.mkdir()
    for t in (1_700_000_100, 1_700_000_200, 1_700_000_300):
        noise(t % 97).save(folder / f"{t}.png")
    assert se.build_index_from_directory(folder)["encoded"] == 3
    noise(5).save(folder / "1700000050.png")                    # copied in later, but taken before the others
    again = se.build_index_from_directory(folder)
    assert again["done"] == 3 and again["encoded"] == 1


def test_captures_in_the_screenshots_folder_are_not_indexed_twice(tmp_path, enc):
    """capture_dir set to the user's screenshots folder: captures land next to the screenshots, and indexing that folder later
    reads only the screenshots that are not in the index yet."""
    shots = tmp_path / "Pictures" / "Screenshots"
    screenshots(shots, 2)
    se.set_config(capture_dir=str(shots))
    se.configure(index_dir=tmp_path / "idx", width=256, encoder=enc)
    cap = se.build_index_from_now(interval=0.01, frames=iter([(noise(60 + i), time.time() + i) for i in range(3)]))
    cap.wait()
    assert cap.stats["encoded"] == 3 and len(list(shots.rglob("*.webp"))) == 3, "captures written into the screenshots folder"
    stats = se.build_index_from_directory(shots)
    assert stats["found"] == 5 and stats["done"] == 3 and stats["encoded"] == 2, "only the 2 screenshots are new"
    lines = [json.loads(l) for l in (tmp_path / "idx" / "frames.jsonl").read_text().splitlines()]
    assert sorted(tuple(sorted(l)) for l in lines) == [("path", "ts")] * 3 + [("source", "ts")] * 2, "only times and paths"


def test_a_slow_screen_grab_is_reported_once(monkeypatch, capsys):
    from streamembed import capture

    class SlowMSS(_FakeMSS):
        def grab(self, m):
            time.sleep(0.05)
            return _FakeShot()

    monkeypatch.setattr(capture, "_grabber", lambda: SlowMSS)
    monkeypatch.setattr(capture, "SLOW_GRAB_S", 0.01)
    monkeypatch.setattr(capture, "_front_window", lambda: ("Safari", "News"))
    monkeypatch.setattr(capture, "_locked", lambda: False)
    frames = capture.screen_frames(64)
    next(frames), next(frames)
    err = capsys.readouterr().err
    assert err.count("a screen grab took") == 1


def test_screen_capture_survives_a_bad_config_file_and_failing_checks(monkeypatch, capsys):
    from streamembed import capture, config as C
    monkeypatch.setattr(capture, "_grabber", lambda: _FakeMSS)
    monkeypatch.setattr(capture, "_excluded", {})
    front = {"app": ("Safari", "News")}

    def window():
        if front["app"] is None:
            raise RuntimeError("no window server")
        return front["app"]

    monkeypatch.setattr(capture, "_front_window", window)
    frames = capture.screen_frames(64)
    assert isinstance(next(frames), list)
    C.config_path().write_text('top_k = "ten"\n')                       # a typo made by hand while it captures
    assert isinstance(next(frames), list), "the exclude lists read before are used"
    front["app"] = ("1Password", None)
    assert next(frames) is None, "they still apply"
    assert "exclude lists read before are used" in capsys.readouterr().err
    front["app"] = None
    assert next(frames) is None, "a failing window check skips one capture"
    front["app"] = ("Safari", "News")
    assert isinstance(next(frames), list), "and capture goes on"
    monkeypatch.setattr(capture, "_excluded", {})
    assert capture.private("Safari", "News"), "lists never read: nothing is captured"
    C.config_path().unlink()
    assert not capture.private("Safari", "News")


def test_service_exits_cleanly_when_the_index_needs_another_model(tmp_path, monkeypatch, capsys):
    service, enc = _service_env(monkeypatch, tmp_path, iter([]))
    idx = str(tmp_path / "idx")
    for error in (ValueError("was built with encoder other@abc. Use another index_dir"), NotImplementedError("siglip2")):
        def fail(*a, error=error, **k):
            raise error
        monkeypatch.setattr(se, "build_index_from_now", fail)
        assert service.main(["--index-dir", idx]) == 0, error
        assert "Fix the setting" in capsys.readouterr().out
        monkeypatch.setattr(se, "build_index_from_directory", fail)
        screenshots(tmp_path / "shots", 1)
        assert service.main(["--index-dir", idx, "--folder", str(tmp_path / "shots")]) == 0
        capsys.readouterr()
    assert service.main(["--index-dir", idx, "--interval", "nan"]) == 0 and "not nan" in capsys.readouterr().out
    se.configure(index_dir=tmp_path / "idx")


def test_stop_after_giving_up_returns_at_once(tmp_path, enc, monkeypatch):
    import threading
    monkeypatch.setattr(se.api, "STALL_S", 0.3)
    release = threading.Event()

    def stuck():
        release.wait(10)
        yield None

    cap = se.build_index_from_now(interval=0.01, frames=stuck())
    time.sleep(0.1)
    assert cap.stop(timeout=0.1) is False
    t = time.monotonic()
    assert cap.stop() is False and time.monotonic() - t < 1, "not another 30 s"
    se.configure(index_dir=tmp_path / "idx")                            # also returns at once
    assert time.monotonic() - t < 1
    release.set()


def test_command_line_index_dir_and_environment_notes(tmp_path, monkeypatch, capsys):
    from streamembed import cli
    assert cli.main(["search", "headphones", "--index-dir", str(tmp_path / "nonexistent"), "--no-open"]) == 0
    assert "does not exist" in capsys.readouterr().out
    assert cli.main(["search", "  ", "--index-dir", str(tmp_path), "--no-open"]) == 2
    assert "a question is needed" in capsys.readouterr().err
    monkeypatch.setenv("STREAMEMBED_INDEX", str(tmp_path / "env"))
    assert cli.main(["config", "set", "index_dir", str(tmp_path / "other")]) == 0
    assert "$STREAMEMBED_INDEX is set and takes precedence" in capsys.readouterr().out
    assert cli.main(["config", "set", "top_k", "5"]) == 0
    assert "takes precedence" not in capsys.readouterr().out
    for argv in (["capture", "--interval", "nan"], ["index", str(tmp_path), "--interval", "inf"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert "--interval" in capsys.readouterr().err


def test_a_wrong_models_repo_is_one_line(tmp_path):
    from huggingface_hub.errors import HFValidationError
    from streamembed.models import _repo_error
    for repo, e in ((str(tmp_path), FileNotFoundError("x")), ("nobody/nothing-here", HFValidationError("Repo id must be ..."))):
        msg = _repo_error(repo, e)
        assert "\n" not in msg and repr(repo) in msg and "config reset models_repo" in msg, msg
    msg = _repo_error("org/repo", OSError("first line\nsecond line\nthird"))
    assert "first line" in msg and "second" not in msg


# ---- regressions
def test_macos_display_numbers_are_screens_at_search_time(tmp_path, enc):
    """'Screenshot ... (2)' is display 2: the viewer's context and the 30-s dedupe keep the two displays apart, as indexing does."""
    folder = tmp_path / "shots"
    folder.mkdir()
    for i in range(3):
        name = f"Screenshot 2026-11-06 at 15.20.{10 + 10 * i:02d}"
        noise(i).save(folder / f"{name}.png")
        noise(20 + i).save(folder / f"{name} (2).png")
    assert se.build_index_from_directory(folder, interval=0)["encoded"] == 6
    idx = se.api._index()
    second = [i for i, f in enumerate(idx.frames) if f["source"].endswith(" (2).png")]
    first = [i for i in range(len(idx)) if i not in second]
    assert idx.around(first[0], 15) == first and idx.around(second[1], 15) == second, "context stays on one display"
    assert [idx.position(i) for i in second] == [0, 1, 2]
    hits = idx.search(np.asarray(idx._rows()[first[0]], np.float32), 10)
    assert {r in second for r, _ in hits} == {True, False}, "a match on display 2 is not a repeat of display 1"


def test_a_capture_releases_the_write_lock_when_it_ends(tmp_path, enc):
    cap = se.build_index_from_now(interval=0.01, frames=iter([(noise(80 + i), time.time() + i) for i in range(2)]))
    cap.wait()
    assert cap.stats["encoded"] == 2 and se.api._state["index"] is None
    other = index_mod.Index(tmp_path / "idx", "flat", "fake")
    assert not other.readonly and len(other) == 2, "another process can write once the capture ended"
    other.close()


def test_service_main_when_another_process_writes(tmp_path, monkeypatch, capsys):
    service, enc = _service_env(monkeypatch, tmp_path, iter([]))
    holder = index_mod.Index(tmp_path / "idx", "flat", "fake")          # a notebook that indexed and was left open
    assert service.main(["--index-dir", str(tmp_path / "idx")]) == 1, "restarted later, when the lock may be free"
    out = capsys.readouterr()
    assert "another process" in out.out and "Python session" in out.out and "Traceback" not in out.out + out.err
    screenshots(tmp_path / "shots", 2)
    assert service.main(["--index-dir", str(tmp_path / "idx"), "--folder", str(tmp_path / "shots")]) == 1
    holder.close()
    se.configure(index_dir=tmp_path / "idx")


def test_capture_dir_must_be_a_full_path(tmp_path, monkeypatch):
    from streamembed.config import config_path
    with pytest.raises(ValueError, match="capture_dir: a full path"):
        se.set_config(capture_dir="Screenshots")
    se.set_config(capture_dir="~/Pictures/Screenshots")                # ~ is a full path
    se.set_config(capture_dir="")
    config_path().write_text('capture_dir = "Screenshots"\n')
    with pytest.raises(ValueError, match="capture_dir"):
        se.get_config("capture_dir")


def test_a_live_capture_has_no_source_also_in_older_indexes(tmp_path):
    """StreamEmbed 0.3.0 recorded live captures under "source"; they are still live captures, not folder screenshots."""
    root = tmp_path / "idx"
    idx = index_mod.Index(root, "flat", "fake")
    v = np.ones(DIM, np.float32) / np.sqrt(DIM)
    idx.add(v, dict(ts=1.0, source="captures/2026-11-07/1794071628117.jpg"))          # 0.3.0
    idx.add(v, dict(ts=2.0, path="captures/2026-11-07/1794071633117_1.jpg"))          # now
    idx.add(v, dict(ts=3.0, source="/Users/me/Desktop/Screenshot 2026-11-07 at 10.00.00.png"))
    assert [idx.live(i) for i in range(3)] == [True, True, False]
    assert [idx.source(i) for i in range(3)] == [None, None, "/Users/me/Desktop/Screenshot 2026-11-07 at 10.00.00.png"]
    assert idx.screen(0) == ("", "") and idx.screen(1) == ("", "1")
    assert str(root.resolve() / "captures/2026-11-07/1794071633117_1.jpg") in idx.handled_sources()
    idx.close()


def test_many_extra_vectors_are_kept_aside_not_dropped(tmp_path, capsys):
    root = tmp_path / "idx"
    idx = index_mod.Index(root, "flat", "fake")
    for i in range(50):
        idx.add(np.full(DIM, i, np.float32), dict(ts=float(i), path=f"{i}.jpg"))
    idx.close()
    lines = (root / "frames.jsonl").read_text().splitlines()
    (root / "frames.jsonl").write_text("\n".join(lines[:10]) + "\n")    # an older frames.jsonl restored
    idx = index_mod.Index(root, "flat", "fake")
    assert len(idx) == 10 and (root / "vectors.f16").stat().st_size == 10 * DIM * 2
    orphan = list(root.glob("vectors.f16.orphan-*"))
    assert len(orphan) == 1 and orphan[0].stat().st_size == 40 * DIM * 2, "the 40 vectors are kept"
    assert np.fromfile(orphan[0], np.float16)[0] == 10 and "40 vectors" in capsys.readouterr().err
    idx.close()
    with open(root / "vectors.f16", "ab") as f:                         # one vector without its line: the crash case, cut quietly
        np.zeros(DIM, np.float16).tofile(f)
    idx = index_mod.Index(root, "flat", "fake")
    assert len(idx) == 10 and len(list(root.glob("*.orphan-*"))) == 1 and capsys.readouterr().err == ""
    idx.close()


def test_a_cut_handled_line_is_dropped_when_the_writer_opens(tmp_path):
    root = tmp_path / "idx"
    idx = index_mod.Index(root, "flat", "fake")
    idx.mark_handled("/shots/a.png")
    idx.close()
    with open(root / "handled.jsonl", "a") as f:
        f.write('"/shots/b.p')                                          # cut by SIGTERM
    idx = index_mod.Index(root, "flat", "fake")
    idx.mark_handled("/shots/c.png")
    assert idx.handled_sources() == {"/shots/a.png", "/shots/c.png"}
    idx.close()


def test_since_after_until_and_an_empty_window(tmp_path, enc, monkeypatch):
    se.build_index_from_directory(screenshots(tmp_path / "shots", 3))              # 2023
    with pytest.raises(ValueError, match="since .* must be before until"):
        se.query_index("x", since="2026-09-22", until="2026-09-20")
    monkeypatch.setattr(se.api, "_encoder", lambda: (_ for _ in ()).throw(AssertionError("the model was loaded")))
    assert se.query_index("x", since="2026-01-01") == [], "no frame in the window: the model is not loaded"


def test_command_line_options_before_the_command(capsys, monkeypatch):
    from streamembed import cli, service
    monkeypatch.setattr(service, "main", lambda argv: pytest.fail("the service process was started"))
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"streamembed {se.__version__}"
    for argv in (["--interval", "10"], ["--index-dir", "x", "search", "excel"], ["--anything"]):
        assert cli.main(argv) == 2
        err = capsys.readouterr().err
        assert err.startswith(f"streamembed: error: {argv[0]} goes after the command") and err.count("\n") == 1


def test_excluded_defaults_dropped_by_set_are_named(capsys):
    se.set_config(exclude_apps="1password,bitwarden,dashlane,lastpass,keychain access,keepass,signal")
    assert capsys.readouterr().err == "", "the defaults are kept: nothing to say"
    se.set_config(exclude_apps="signal")
    err = capsys.readouterr().err
    assert "replaces the whole list" in err and "1password, bitwarden" in err
    se.set_config(exclude_titles=[])
    assert "incognito" in capsys.readouterr().err
    assert se.get_config("exclude_apps") == ["signal"] and se.get_config("exclude_titles") == []


def test_configure_checks_the_index_folder_and_model_names(tmp_path):
    for k in ("index_dir", "models_repo", "model"):
        with pytest.raises(ValueError, match=f"{k}: a .*not an empty text"):
            se.configure(**{k: ""})
    se.configure(index_dir=tmp_path / "idx")


def test_the_index_folder_itself_and_a_folder_inside_it(tmp_path, enc, svc):
    with pytest.raises(ValueError, match="is the index folder; give the folder your screenshots are in"):
        se.build_index_from_directory(tmp_path / "idx")
    service, calls, platform = svc
    platform("darwin")
    (tmp_path / "idx" / "captures").mkdir(parents=True)
    with pytest.raises(ValueError, match="inside the index folder"):
        service.install_service(index_dir=tmp_path / "idx", directory=tmp_path / "idx" / "captures")
    assert not calls and not service.service_status()["installed"], "refused before anything is installed"


def test_the_model_download_note_only_when_not_cached(tmp_path, monkeypatch):
    from streamembed import api
    assert api._cached(str(tmp_path), None), "a local folder needs no download"
    snap = tmp_path / "hub" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name, revision=None: str(snap / name))
    assert not api._cached("Org/model", "abc"), "config only: the weights are not downloaded yet"
    (snap / "model.safetensors").write_bytes(b"")
    assert api._cached("Org/model", "abc")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name, revision=None: None)
    assert not api._cached("Org/model", "abc")


def test_the_windows_lock(tmp_path, monkeypatch):
    """The msvcrt branch of the write lock, with a stand-in msvcrt: one writer, then a reader."""
    held = set()

    class FakeMsvcrt:
        LK_NBLCK = 2

        @staticmethod
        def locking(fd, mode, n):
            key = os.fstat(fd).st_ino
            if key in held:
                raise OSError("locked")
            held.add(key)

    monkeypatch.setattr(index_mod, "fcntl", None)
    monkeypatch.setattr(index_mod, "msvcrt", FakeMsvcrt, raising=False)
    first = index_mod.Index(tmp_path / "idx", "flat", "fake")
    second = index_mod.Index(tmp_path / "idx", "flat", "fake")
    assert not first.readonly and second.readonly
    with pytest.raises(RuntimeError, match="another process"):
        second.add(np.zeros(DIM, np.float32), dict(ts=1.0, path="x.jpg"))
    first.close()
    second.close()
