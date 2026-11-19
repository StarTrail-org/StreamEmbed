"""`streamembed start` on macOS reads the service's first log line to warn only when screen capture is not allowed."""
from streamembed.cli import _screen_allowed, _warn_screen_permission
from streamembed.service import SCREEN_ALLOWED, SCREEN_NOT_ALLOWED


def _log(tmp_path, *lines):
    p = tmp_path / "service.log"
    p.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return p


def test_allowed_and_not_allowed(tmp_path):
    assert _screen_allowed(_log(tmp_path, SCREEN_ALLOWED), 0, timeout=1) is True
    assert _screen_allowed(_log(tmp_path, SCREEN_NOT_ALLOWED, "streamembed service: Screen capture needs ..."), 0, timeout=1) is False


def test_reads_only_after_offset(tmp_path):
    old = SCREEN_NOT_ALLOWED + "\n"                     # an earlier run's line stays in the log
    p = _log(tmp_path, SCREEN_NOT_ALLOWED, SCREEN_ALLOWED)
    assert _screen_allowed(p, len(old.encode()), timeout=1) is True


def test_other_first_line_or_nothing(tmp_path):
    assert _screen_allowed(_log(tmp_path, "streamembed service: unknown model x. Fix the setting"), 0, timeout=5) is None
    assert _screen_allowed(tmp_path / "missing.log", 0, timeout=0.3) is None


def test_warning_text(capsys):
    _warn_screen_permission()
    err = capsys.readouterr().err
    assert err.startswith("Warning: ") and "Screen & System Audio Recording" in err and "streamembed start" in err
