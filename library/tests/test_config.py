"""Settings: package defaults, the user's file, the environment, and arguments, in that order of precedence."""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import streamembed as se
from streamembed import config as C
from test_api import FakeEncoder, noise, svc  # noqa: F401  (svc: a fixture)


def test_defaults_and_changes():
    assert se.get_config("interval") == 5.0 and se.get_config("monitors") == "all" and se.get_config("skip_unchanged") is True
    path = se.set_config(interval="10", skip_unchanged="false", monitors="1,2", exclude_apps="Signal, Slack")
    assert se.get_config("interval") == 10.0 and se.get_config("skip_unchanged") is False and se.get_config("monitors") == [1, 2]
    assert se.get_config("exclude_apps") == ["Signal", "Slack"]
    text = path.read_text()
    assert "interval = 10.0" in text and "top_k" not in text, "the file holds only what was changed"
    assert C.sources()["interval"] == "config file" and C.sources()["top_k"] == "default"
    se.set_config(interval=5.0)
    assert "interval" not in path.read_text(), "back at the default: removed from the file"
    se.reset_config()
    assert se.get_config() == C.defaults()
    with pytest.raises(KeyError):
        se.set_config(intervall=3)
    with pytest.raises(ValueError):
        se.set_config(skip_unchanged="maybe")


def test_environment_wins_over_the_file(tmp_path, monkeypatch):
    se.set_config(index_dir=str(tmp_path / "a"))
    monkeypatch.setenv("STREAMEMBED_INDEX", str(tmp_path / "b"))
    assert se.get_config("index_dir") == str(tmp_path / "b") and C.sources()["index_dir"] == "STREAMEMBED_INDEX"


def test_api_follows_the_settings(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    now = int(time.time())
    for i in range(6):
        noise(i).save(folder / f"{now - 40 * i}.png")                   # 40 s apart (more than the 30 s dedupe)
    se.set_config(index_dir=str(tmp_path / "idx"), width=256, top_k=2, interval=60)
    se.configure(encoder=FakeEncoder())                                  # everything else from the settings
    assert se.api._cfg["index_dir"] == tmp_path / "idx" and se.api._cfg["width"] == 256
    assert se.build_index_from_directory(folder)["captured"] == 3, "interval 60 s from the settings: every other one"
    assert len(se.query_index("x")) == 2, "top_k from the settings"
    se.set_config(interval=0)
    se.configure(index_dir=tmp_path / "idx2", encoder=FakeEncoder())
    assert se.build_index_from_directory(folder)["captured"] == 6, "interval 0: every screenshot"
    from streamembed.capture import private
    assert private("1Password 8", None) and not private("Signal", None)
    se.set_config(exclude_apps=["Signal"])
    assert private("Signal", None) and not private("1Password 8", None)
    se.configure(index_dir=tmp_path / "other")


def test_command_line_config(capsys):
    from streamembed import cli
    assert cli.main(["config", "set", "interval", "7", "skip_unchanged", "false"]) == 0
    out = capsys.readouterr().out
    assert any(l.startswith("interval") and "7.0" in l and "config file" in l for l in out.splitlines())
    assert se.get_config("skip_unchanged") is False
    cli.main(["config", "reset", "interval"])
    assert se.get_config("interval") == 5.0 and se.get_config("skip_unchanged") is False
    cli.main(["config", "path"])
    assert os.environ["STREAMEMBED_CONFIG"] in capsys.readouterr().out


def test_bad_values_are_refused():
    for name, value, says in [("index_type", "faiss", "index_type: one of flat"), ("width", -5, "width: at least 32"),
                              ("device", "gpu", "device: one of auto"), ("interval", "ten", "interval: a number of seconds"),
                              ("interval", -3, "interval: at least 0"), ("top_k", 3.5, "top_k: a number of results (a whole number)"),
                              ("top_k", "3.5", "top_k"), ("monitors", "left", "monitors"), ("page_results", 0, "page_results: at least 1"),
                              ("context", "x", "context")]:
        with pytest.raises(ValueError, match=re.escape(says)):
            se.set_config(**{name: value})
    assert not C.config_path().exists() or "faiss" not in C.config_path().read_text(), "nothing was saved"
    se.set_config(device="cuda:1", width="640")
    assert se.get_config("width") == 640


def test_a_bad_file_breaks_no_command(tmp_path, capsys):
    """A wrong value in the file (written by hand, or by an older version) never breaks the import, --help or `config reset`."""
    from streamembed import cli
    path = C.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "STREAMEMBED_CONFIG": str(path), "PYTHONPATH": str(Path(se.__file__).parents[1])}
    for text, says in [('index_type = "faiss"\n', "index_type: one of flat"), ('width = "wide"\n', "width: a width in pixels"),
                       ("[x\n", "is not valid TOML")]:
        path.write_text(text)
        r = subprocess.run([sys.executable, "-c", "import streamembed, streamembed.cli as c; c.main(['--help'])"], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert cli.main(["config"]) == 2
        err = capsys.readouterr().err
        assert err.startswith("streamembed: error:") and says in err and str(path) in err and "Traceback" not in err
        assert cli.main(["config", "reset"] + ([] if text.startswith("[") else [text.split()[0]])) == 0
        assert se.get_config() == C.defaults()
        capsys.readouterr()


def test_command_line_errors_are_one_line(tmp_path, capsys):
    from streamembed import cli
    idx = str(tmp_path / "idx")
    for argv, says in [(["config", "set", "intervall", "10"], "Unknown setting 'intervall'"),
                       (["config", "set", "interval", "ten"], "interval: a number of seconds, not 'ten'"),
                       (["config", "set", "monitors", "left"], "monitors"),
                       (["config", "reset", "foo"], "Unknown setting 'foo'"),
                       (["search", "x", "--since", "yesterday", "--index-dir", idx], "since: a date such as 2026-11-03"),
                       (["index", str(tmp_path / "nonexist"), "--index-dir", idx], "Folder not found")]:
        assert cli.main(argv) == 2, argv
        err = capsys.readouterr().err
        assert err.startswith("streamembed: error:") and says in err and len(err.strip().splitlines()) == 1, err
    with pytest.raises(SystemExit):
        cli.main(["capture", "--monitors", "a,b"])            # argparse names the option
    assert "--monitors" in capsys.readouterr().err
    assert not (tmp_path / "idx").exists(), "nothing was created for a mistake"


def test_set_config_applies_in_this_process(tmp_path, svc):
    service, calls, platform = svc
    platform("darwin")
    se.configure(encoder=FakeEncoder())
    se.set_config(index_dir=str(tmp_path / "a"), width=640)
    assert se.api._settings()["index_dir"] == tmp_path / "a" and se.api._settings()["width"] == 640
    args = service.service_status()["args"] if service.install_service(index_dir=tmp_path / "a") else None
    assert "--width" not in args and "--model" not in args, "the saved settings are read by the service itself"
    se.configure(width=320, encoder=FakeEncoder())
    service.install_service(index_dir=tmp_path / "a")
    args = service.service_status()["args"]
    assert args[args.index("--width") + 1] == "320", "what configure() set for this process is passed on"
    se.set_config(width=800)
    assert se.api._settings()["width"] == 320, "configure() wins over the config"


def test_configure_keeps_the_loaded_model(tmp_path):
    model = object()
    se.configure(index_dir=tmp_path / "a")
    se.api._state.update(encoder=model, encoder_key=se.api._model_key(se.api._settings()))
    se.configure(index_dir=tmp_path / "b", index_type="pq")
    assert se.api._state["encoder"] is model, "only the index changed"
    se.configure(index_dir=tmp_path / "b", device="cpu")
    assert se.api._state["encoder_key"] != se.api._model_key(se.api._settings()), "another device: loaded again on next use"
    se.api._state.update(encoder=None, encoder_key=None)
    se.configure(index_dir=tmp_path / "b")


def test_empty_text_settings_are_refused():
    for name in ("index_dir", "models_repo", "model"):
        with pytest.raises(ValueError, match=f"{name}: a .*not an empty text"):
            se.set_config(**{name: " "})
    assert not C.config_path().exists()


def test_interval_must_be_a_finite_number_of_seconds():
    for value in ("nan", "inf", "-inf", "1e400", float("nan"), 86401):
        with pytest.raises(ValueError, match="interval"):
            se.set_config(interval=value)
    se.set_config(interval=86400)
    assert se.get_config("interval") == 86400.0
    with pytest.raises(ValueError, match="interval"):
        se.build_index_from_now(interval=float("nan"), frames=iter([]))
    C.config_path().write_text("interval = nan\n")
    with pytest.raises(ValueError, match="interval"):
        se.get_config("interval")
    se.reset_config()
