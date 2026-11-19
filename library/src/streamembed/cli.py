"""The `streamembed` command: everything the Python API does, from a terminal.

    streamembed start [--folder DIR] [--interval 5] [--monitors all]    capture in the background from now and after every login
    streamembed status | pause | resume | stop                          the background capture
    streamembed index DIR                                               index a folder of screenshots now (Ctrl-C stops; run again to continue)
    streamembed capture                                                 capture in this terminal until Ctrl-C
    streamembed search "the flight search page to Denver"               search and open the results page
    streamembed config [show | set KEY VALUE ... | reset [KEY ...] | path]   the settings every command starts from

Options go after the command. Options left out come from the settings (`streamembed config`); --index-dir defaults to the index_dir
setting. A mistake in a setting, a path or a date prints one line; a mistake in an option prints the usage and the error. Both exit
with status 2.
"""
from __future__ import annotations

import argparse
import sys
import time


ERRORS = (KeyError, ValueError, FileNotFoundError, NotADirectoryError, PermissionError, ImportError, RuntimeError, NotImplementedError)


def _setting(name: str):
    """An argparse type that checks the value as the setting `name` is checked."""
    def parse(s: str):
        from .config import _coerce
        try:
            return _coerce(name, s)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e).removeprefix(f"{name}: ")) from None
    return parse


def _n(count: int, one: str, many: str | None = None) -> str:
    return f"{count:,} {one if count == 1 else many or one + 's'}"


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index-dir", default=None, help="index folder (setting: index_dir)")


def _unchanged(p: argparse.ArgumentParser) -> None:
    p.add_argument("--keep-unchanged", dest="skip_unchanged", action="store_const", const=False, default=None,
                   help="also store screens that did not change")
    p.add_argument("--skip-unchanged", dest="skip_unchanged", action="store_const", const=True, help="store changed screens only")


def _capture_args(p: argparse.ArgumentParser) -> None:
    from .config import INDEX_TYPES
    p.add_argument("--interval", type=_setting("interval"), default=None,
                   help="seconds between captures; below 0.5 means 0.5 (setting: interval)")
    p.add_argument("--monitors", type=_setting("monitors"), default=None, help='"all", "primary", or display numbers such as 1,2 (setting: monitors)')
    _unchanged(p)
    p.add_argument("--model", default=None, help="qwen3-vl-2b (default), qwen3-vl-8b or qwen3-vl-2b-base (setting: model)")
    p.add_argument("--index-type", default=None, choices=INDEX_TYPES, help="(setting: index_type)")


EXAMPLES = """examples:
  streamembed start                          capture from now on, in the background (search a minute later)
  streamembed search "the flight search page to Denver"
  streamembed search "a slide about pricing" --since 2026-11-01 --until 2026-11-06
  streamembed index ~/Pictures/Screenshots   add screenshots you already have
  streamembed config set capture_max_gb 30

Run "streamembed COMMAND -h" for its options."""


def _parser() -> argparse.ArgumentParser:
    from . import __version__
    ap = argparse.ArgumentParser(prog="streamembed", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EXAMPLES,
                                 description="Search your own screen history by describing what you saw.\nOptions left out come "
                                 "from the settings (streamembed config).\nOptions go after the command.")
    ap.add_argument("--version", action="version", version=f"streamembed {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def command(name: str, text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=text, description=text[0].upper() + text[1:] + ".")

    p = command("start", "capture in the background from now and after every login")
    _common(p), _capture_args(p)
    p.add_argument("--folder", default=None, help="index the screenshots in this folder first")
    for name, text in (("stop", "stop the background capture and remove it"), ("status", "is it running, paused, how many screens"),
                       ("pause", "pause capturing (the background capture too)"), ("resume", "resume capturing")):
        _common(command(name, text))
    p = command("index", "index a folder of screenshots now (Ctrl-C stops; run it again to continue)")
    _common(p)
    p.add_argument("folder", help="the folder with your screenshots, searched recursively")
    p.add_argument("--interval", type=_setting("interval"), default=None,
                   help="at most one screenshot per screen every this many seconds, 0: all (setting: interval)")
    _unchanged(p)
    p = command("capture", "capture in this terminal until Ctrl-C")
    _common(p), _capture_args(p)
    p = command("search", "search in the search page (with --no-open: print the results here)")
    _common(p)
    p.add_argument("query", nargs="?", default=None,
                   help="a description of the screen, such as \"the flight search page to Denver\"; without one, the search page opens "
                        "(while the background capture runs)")
    p.add_argument("--top-k", type=_setting("top_k"), default=None, help="results to print, at least 1 (setting: top_k)")
    p.add_argument("--since", default=None, help='from this time: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"')
    p.add_argument("--until", default=None, help='up to this time; a date alone includes that day')
    p.add_argument("--no-open", action="store_true", help="do not open the results page")
    p = command("config", "show or change the settings")
    p.add_argument("action", nargs="?", choices=("show", "set", "reset", "path"), default="show",
                   help="show (default), set, reset, or path (where the settings files are)")
    p.add_argument("items", nargs="*", help="set: KEY VALUE [KEY VALUE ...]; reset: KEY [KEY ...] (none: all)")
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    for stream in (sys.stdout, sys.stderr):     # Windows pipes and files use the ANSI code page: print "?" rather than fail
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    if argv and argv[0].startswith("--") and "--log" in argv:
        argv = ["run", *argv]               # services installed before the subcommands: python -m streamembed --index-dir ... --log ...
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--version"):
        print(f"streamembed: error: {argv[0]} goes after the command, such as `streamembed search {argv[0]} ...`; "
              "`streamembed --help` lists the commands", file=sys.stderr)
        return 2
    if argv and argv[0] == "run":           # the background capture process itself, started by the service manager
        from .service import main as run
        return run(argv[1:])
    ap = _parser()
    if not argv:
        ap.print_help()
        return 0
    a = ap.parse_args(argv)
    try:
        return _config(a) if a.command == "config" else _run(a)
    except ERRORS as e:                     # a mistake in a setting, a path or a date: one line, not a traceback
        print(f"streamembed: error: {e.args[0] if isinstance(e, KeyError) and e.args else e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:               # Ctrl-C outside a capture or an indexing loop (while the model loads, for example)
        print("Stopped.", file=sys.stderr)
        return 130


def _service_index_dir(se, given: str | None):
    """--index-dir, else the background service's index folder when it is installed, else the configured one."""
    if given:
        return given
    try:
        return se.service_status()["index_dir"]
    except NotImplementedError:
        return None


def _screen_allowed(log, offset: int, timeout: float = 20.0) -> bool | None:
    """Whether the service just started may record the screen (macOS): its first line in `log` after `offset` says so.
    None when its first line is something else (a bad setting stops it before the check) or it wrote nothing within `timeout` s."""
    from .service import SCREEN_ALLOWED, SCREEN_NOT_ALLOWED
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with open(log, encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                lines = f.read().splitlines()
        except FileNotFoundError:
            lines = []
        for line in lines:
            if line in (SCREEN_ALLOWED, SCREEN_NOT_ALLOWED):
                return line == SCREEN_ALLOWED
            if line.startswith("streamembed service:"):
                return None
        time.sleep(0.2)
    return None


def _warn_screen_permission() -> None:
    """The service cannot record the screen: say what to turn on, as a warning on stderr."""
    import os
    from .capture import SCREEN_PANE, python_app, turn_on
    head = "Warning: the background service cannot record the screen yet, so it stopped."
    if sys.stderr.isatty() and not os.environ.get("NO_COLOR"):
        head = f"\033[1;33m{head}\033[0m"
    print(f"{head}\n"
          f"  In {SCREEN_PANE},\n"
          f"  {turn_on(python_app())}.\n"
          "  Then run `streamembed start` again. Only the pane's name mentions audio: nothing records sound.", file=sys.stderr)


def _stamp(ts: float | None, default: str) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts is not None else default


def _progress():
    """Print the counts to stderr after the first screenshot, then every 10 s."""
    state = dict(t0=None, n0=0, last=-1e9)

    def show(st: dict) -> None:
        now, read = time.monotonic(), st["captured"] + st["errors"]
        if state["t0"] is None:
            state.update(t0=now, n0=read)
        if now - state["last"] < 10 and read < st["todo"]:
            return
        state["last"] = now
        each = (now - state["t0"]) / (read - state["n0"]) if read - state["n0"] >= 2 else None
        print(f"  {read:,}/{st['todo']:,} screenshots read, {st['encoded']:,} indexed" + (f", {each:.1f} s each" if each is not None else ""),
              file=sys.stderr, flush=True)
    return show


def _run(a) -> int:
    import streamembed as se
    from pathlib import Path
    from .api import _has_rows, _settings
    if a.index_dir and Path(a.index_dir).expanduser().is_file():
        raise NotADirectoryError(f"{a.index_dir} is a file, not a folder")
    settings = {k: v for k, v in (("index_dir", a.index_dir), ("model", getattr(a, "model", None)),
                                  ("index_type", getattr(a, "index_type", None))) if v}
    if settings:
        se.configure(**settings)
    if a.command == "start":
        log = Path(_settings()["index_dir"]).expanduser().resolve() / "service.log"     # where install_service points the service
        offset = log.stat().st_size if log.is_file() else 0
        path = se.install_service(interval=a.interval, monitors=a.monitors, skip_unchanged=a.skip_unchanged, directory=a.folder)
        if sys.platform == "darwin" and _screen_allowed(log, offset) is False:
            _warn_screen_permission()
            return 1
        print(f"Started. It runs from now on and after every login ({path}).\nstreamembed status | pause | resume | stop")
    elif a.command == "stop":
        paused = se.service_status()["paused"]
        se.uninstall_service()
        print("Stopped and removed. The index stays." + (" It is still paused: `streamembed resume` lets `index` and `capture` "
                                                          "write to it again." if paused else ""))
    elif a.command == "status":
        st = se.service_status()
        index_dir = Path(st["index_dir"] or _settings()["index_dir"])
        frames = index_dir / "frames.jsonl"
        n = sum(1 for _ in open(frames, encoding="utf-8")) if frames.exists() else 0
        state = "not installed" if not st["installed"] else "paused" if st["paused"] else "running" if st["running"] else "stopped"
        pid = f" (pid {st['pid']})" if st["pid"] else ""
        print(f"{state}{pid}, {n} screens in {index_dir}" + (f"\nlog: {st['log']}" if st["log"] else ""))
    elif a.command == "pause":
        se.pause(_service_index_dir(se, a.index_dir))
        print("Paused. streamembed resume to continue.")
    elif a.command == "resume":
        se.resume(_service_index_dir(se, a.index_dir))
        print("Resumed.")
    elif a.command == "index":
        t = time.time()
        st = se.build_index_from_directory(a.folder, interval=a.interval, skip_unchanged=a.skip_unchanged, progress=_progress())
        if not st["found"]:
            print(f"No screenshots (.jpg, .jpeg, .png, .webp) in {Path(a.folder).expanduser().resolve()}.")
            return 0
        if not st["todo"] and not st["errors"] and st["done"]:
            print(f"Nothing new: {'' if st['skipped'] else 'all '}{_n(st['done'], 'screenshot')} were read in earlier runs"
                  + (f", {st['skipped']:,} left out by the interval." if st["skipped"] else "."))
            return 0
        parts = [_n(st["encoded"], "screen") + " indexed"]
        for key, text in (("idle", "unchanged"), ("minor", "changed too little to store"), ("errors", "could not be read or stored"),
                          ("done", "read in earlier runs"), ("skipped", "left out by the interval")):
            if st[key]:
                parts.append(f"{st[key]:,} {text}")
        print(f"{'Stopped' if st['stopped'] else 'Done'} in {time.time() - t:.0f} s: {', '.join(parts)} ({_n(st['found'], 'file')})."
              + (" Run it again to continue." if st["stopped"] else ""))
    elif a.command == "capture":
        print("Capturing; Ctrl-C stops.")
        cap = se.build_index_from_now(interval=a.interval, monitors=a.monitors, skip_unchanged=a.skip_unchanged, block=True)
        if cap.stalled:
            cap.stop(timeout=0)
            from .capture import SCREEN_PANE
            print(f"streamembed: error: screen capture has not returned for {se.api.STALL_S:.0f} s ({_n(cap.stats['encoded'], 'screen')} "
                  f"indexed). Check {SCREEN_PANE}, then run it again.", file=sys.stderr)
            return 1
        print(f"Stopped: {_n(cap.stats['encoded'], 'screen')} indexed.")
    elif a.command == "search":
        from .config import get_config
        top_k = a.top_k if a.top_k is not None else get_config("top_k")
        if _search_served(a, top_k):
            return 0
        if not a.query:
            raise ValueError("a question is needed, such as `streamembed search \"the flight search page to Denver\"` (the search "
                             "page without one opens only while the background capture runs: streamembed start)")
        res = se.query_index(a.query, top_k=max(top_k, get_config("page_results")), since=a.since,
                             until=a.until)                                    # the page holds page_results, to pick dates among
        start, end = res.window or (None, None)
        window = f"{_stamp(start, 'the start')} to {_stamp(end, 'now')}" if start is not None or end is not None else None
        if not res:
            index_dir = _settings()["index_dir"]
            if not index_dir.exists():
                print(f"{index_dir} does not exist (no index there yet). Check --index-dir, or add screens with "
                      "`streamembed index FOLDER` or `streamembed start`.")
            elif not _has_rows(index_dir):
                print(f"No screens in {index_dir} yet. Add some with `streamembed index FOLDER` or `streamembed start`.")
            elif window:
                print(f"No screens from {window} for {res.text!r}.")
            else:
                print("No screens found.")
            return 0
        if window:
            print(f"Searched {window} for {res.text!r}.")
        for r in res[:top_k]:
            print(f"{r.score:.3f}  {r.time}  {r.path}")
        page = se.visualize_results(res, open_browser=not a.no_open)
        print(f"page: {page}")
    return 0


SERVICE_WAIT_S = 150.0      # how long `search` waits for a background capture that is loading its model


def _starting(root) -> bool:
    """Whether the background capture of index folder `root` runs and is still loading its model: its log says it is loading, and
    has not given its search page since."""
    from pathlib import Path
    from .service import service_status
    try:
        st = service_status()
        if not st.get("running") or not st.get("index_dir") or not st.get("log"):
            return False
        if Path(st["index_dir"]).expanduser().resolve() != Path(root).expanduser().resolve():
            return False
        lines = Path(st["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    except Exception:
        return False
    loading = max((i for i, l in enumerate(lines) if l.startswith("streamembed: loading ")), default=-1)
    served = max((i for i, l in enumerate(lines) if "search page at " in l), default=-1)
    return loading > served


def _wait_for_page(root):
    """The page URL once a background capture that is loading its model serves it (None: none is loading, or it took too long)."""
    import time
    from .webui import find
    if not _starting(root):
        return None
    print("streamembed: the background capture is loading its model; waiting for it (about 20-60 s)", file=sys.stderr, flush=True)
    end = time.monotonic() + SERVICE_WAIT_S
    while time.monotonic() < end:
        url = find(root)
        if url is not None:
            return url
        if not _starting(root) and find(root) is None:
            return None             # it stopped, or it serves no page
        time.sleep(1.0)
    return None


def _search_served(a, top_k: int) -> bool:
    """Search through the page the background capture serves, if it runs: no model is loaded here. The page opens with the question
    in its search box and searches it, as if it was typed there (without a question: the empty page); with --no-open the matches
    are searched for and printed here instead. False when no capture serves this index."""
    import json
    import urllib.parse
    import urllib.error
    import urllib.request
    import webbrowser
    from .api import _settings
    from .webui import find
    root = _settings()["index_dir"]
    url = find(root) or _wait_for_page(root)
    if url is None:
        return False
    when = {k: v for k, v in (("since", a.since), ("until", a.until)) if v}
    if a.query and not a.no_open:               # the page searches: one search, the same as typing it there
        url += "&" + urllib.parse.urlencode({"q": a.query, **when})
        webbrowser.open(url)
        print(f"page: {url.split('?')[0]} (served by the background capture; the question is searched there)")
        return True
    if a.query:
        from .config import get_config
        params = {"q": a.query, "k": max(top_k, get_config("page_results"))}
        params.update(when)
        api = url.replace("/?t=", "/api/search?t=", 1) + "&" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(api, timeout=120) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise ValueError(json.loads(e.read()).get("error", str(e))) from None
        if not data["results"]:
            print("No screens found.")
        for item in data["results"][:top_k]:
            print(f"{item['score']:.3f}  {item['time']}  {item['path']}")
    if not a.no_open:
        webbrowser.open(url)
    print(f"page: {url.split('?')[0]} (served by the background capture)")
    return True


def _config(a) -> int:
    from . import config as C
    if a.action in ("show", "path") and a.items:
        raise ValueError(f"config {a.action} takes no more words, not {' '.join(a.items)!r}")
    if a.action == "path":
        print(f"your settings: {C.config_path()}\ndefaults:      {C.DEFAULTS_FILE}")
        return 0
    if a.action == "set":
        if not a.items or len(a.items) % 2:
            raise ValueError("config set needs KEY VALUE pairs, such as `streamembed config set interval 10`"
                             + (f"; {a.items[-1]!r} has no value" if a.items else ""))
        C.set_config(**dict(zip(a.items[::2], a.items[1::2])))
    elif a.action == "reset":
        C.reset_config(*a.items)
    values, src = C.get_config(), C.sources()
    width, src_width = max(len(k) for k in values), max(len(v) for v in src.values())
    print(f"{'setting':<{width}}  {'from':<{src_width}}  value")
    for k, v in values.items():
        print(f"{k:<{width}}  {src[k]:<{src_width}}  {C._toml(v)}")
    if a.action in ("set", "reset"):
        print(f"\nSaved in {C.config_path()}. A running background capture uses new settings after `streamembed start`.")
        import os
        for k, var in C.ENV.items():
            named = a.items[::2] if a.action == "set" else a.items or list(C.ENV)
            if os.environ.get(var) and k in named:
                print(f"{k}: ${var} is set and takes precedence over this value.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
