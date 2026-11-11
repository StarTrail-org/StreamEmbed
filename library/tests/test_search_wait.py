"""`streamembed search` waits for a background capture that is still loading its model, instead of loading a second one."""
import threading
import time

from streamembed import cli, service, webui


def _service(monkeypatch, tmp_path, log_lines, running=True):
    log = tmp_path / "service.log"
    log.write_text("\n".join(log_lines) + "\n")
    monkeypatch.setattr(service, "service_status",
                        lambda: {"running": running, "index_dir": str(tmp_path), "log": str(log), "installed": True})
    return log


def test_it_waits_while_the_capture_loads(monkeypatch, tmp_path, capsys):
    log = _service(monkeypatch, tmp_path, ["streamembed service: search page at http://127.0.0.1:1/ (old)",
                                           "streamembed: loading qwen3-vl-2b (about 20-60 s)"])
    ready = {"url": None}
    monkeypatch.setattr(webui, "find", lambda root: ready["url"])

    def serve_soon():
        time.sleep(1.5)
        with open(log, "a") as f:
            f.write("streamembed service: search page at http://127.0.0.1:2/ (streamembed search opens it)\n")
        ready["url"] = "http://127.0.0.1:2/?t=x"

    threading.Thread(target=serve_soon).start()
    t = time.monotonic()
    assert cli._wait_for_page(tmp_path) == "http://127.0.0.1:2/?t=x"
    assert 1.0 < time.monotonic() - t < 5
    assert "loading its model; waiting" in capsys.readouterr().err


def test_it_does_not_wait_otherwise(monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "find", lambda root: None)
    _service(monkeypatch, tmp_path, ["streamembed: loading qwen3-vl-2b (about 20-60 s)",
                                     "streamembed service: search page at http://127.0.0.1:2/ (streamembed search opens it)"])
    t = time.monotonic()
    assert cli._wait_for_page(tmp_path) is None, "it loaded already: its page is gone, not coming"
    _service(monkeypatch, tmp_path, ["streamembed: loading qwen3-vl-2b (about 20-60 s)"], running=False)
    assert cli._wait_for_page(tmp_path) is None, "not running"
    _service(monkeypatch, tmp_path, ["streamembed: loading qwen3-vl-2b (about 20-60 s)"])
    assert cli._wait_for_page(tmp_path / "other") is None, "another index"
    assert time.monotonic() - t < 1


def test_it_stops_waiting_when_the_capture_stops(monkeypatch, tmp_path):
    _service(monkeypatch, tmp_path, ["streamembed: loading qwen3-vl-2b (about 20-60 s)"])
    monkeypatch.setattr(webui, "find", lambda root: None)
    threading.Timer(1.0, lambda: _service(monkeypatch, tmp_path, [], running=False)).start()
    t = time.monotonic()
    assert cli._wait_for_page(tmp_path) is None and time.monotonic() - t < 5
