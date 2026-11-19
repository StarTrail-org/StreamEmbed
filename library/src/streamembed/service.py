"""Capture in the background from login on, also with no terminal open.

    import streamembed as se
    se.install_service(interval=5)      # starts now and at every login; restarted if it stops with an error
    se.service_status()                 # {"installed": True, "running": True, "pid": ..., "log": ...}
    se.uninstall_service()              # stops it and removes it

One API, the system's own service manager underneath:
    macOS     a launchd user agent        ~/Library/LaunchAgents/org.streamembed.capture.plist
    Linux     a systemd user service      ~/.config/systemd/user/streamembed-capture.service (X11; needs DISPLAY at install time)
    Windows   a Task Scheduler task        "StreamEmbedCapture", at logon, with pythonw (no console window)

The service runs `python -m streamembed` with this Python and the settings given here, and logs to `service.log` in the index folder.
It holds the index folder's write lock, so while it runs, other processes open that index read-only: `query_index` and
`visualize_results` work (and see new screens), while `build_index_*` on the same folder raises until the service is stopped.
"""
from __future__ import annotations

import argparse
import getpass
import os
import plistlib
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape, unescape

LABEL = "org.streamembed.capture"           # launchd
UNIT = "streamembed-capture.service"        # systemd
TASK = "StreamEmbedCapture"                 # Task Scheduler
ENV_KEYS = ("HF_HOME", "HF_HUB_CACHE", "STREAMEMBED_MODELS")     # passed on to the service, which starts with no shell environment
# the service's first line on macOS; `streamembed start` waits for it to warn only when capture cannot work
SCREEN_ALLOWED, SCREEN_NOT_ALLOWED = "streamembed service: screen capture allowed", "streamembed service: screen capture not allowed"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True)


BOOTOUT_WAIT_S = 60.0      # launchd stops a job asynchronously: wait for it to be gone before registering it again


def _bootout(domain: str) -> None:
    """Stop and unregister the launchd job, and wait until launchd no longer knows it. bootstrap right after bootout fails with
    "Bootstrap failed: 5: Input/output error" while the old process is still stopping (it ends the screenshot in progress first)."""
    _launchctl("bootout", f"{domain}/{LABEL}")
    end = time.monotonic() + BOOTOUT_WAIT_S
    while time.monotonic() < end and _launchctl("print", f"{domain}/{LABEL}").returncode == 0:
        time.sleep(0.5)


def _bootstrap(domain: str, path: Path) -> subprocess.CompletedProcess:
    """Register the job; retried a few times while launchd still reports the old one (error 5)."""
    for attempt in range(5):
        r = _launchctl("bootstrap", domain, str(path))
        if r.returncode == 0 or "5:" not in (r.stderr or "") or BOOTOUT_WAIT_S == 0:
            return r
        time.sleep(2)
    return r


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return _run("launchctl", *args)


def _system() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "windows"
    raise NotImplementedError(f"No service manager for {sys.platform}. Run build_index_from_now(block=True) under your own.")


def _paths() -> dict:
    home = Path.home()
    return dict(macos=home / "Library" / "LaunchAgents" / f"{LABEL}.plist", linux=home / ".config" / "systemd" / "user" / UNIT,
                windows=home / "AppData" / "Local" / "StreamEmbed" / f"{TASK}.xml")


def _python(system: str) -> str:
    if system == "windows":                 # pythonw: no console window at logon
        w = Path(sys.executable).with_name("pythonw.exe")
        return str(w) if w.exists() else sys.executable
    return sys.executable


# ---- one file per service manager
def _plist(cmd: list[str], log: str, env: dict) -> bytes:
    return plistlib.dumps(dict(Label=LABEL, ProgramArguments=cmd, RunAtLoad=True, KeepAlive=dict(SuccessfulExit=False),
                               ThrottleInterval=60, ProcessType="Background", StandardOutPath=log, StandardErrorPath=log,
                               EnvironmentVariables=env))


def _unit(cmd: list[str], log: str, env: dict) -> str:
    envs = "".join(f"Environment={shlex.quote(f'{k}={v}')}\n" for k, v in env.items())
    return (f"[Unit]\nDescription=StreamEmbed screen capture\n\n[Service]\nExecStart={shlex.join(cmd)}\n{envs}"
            f"Restart=on-failure\nRestartSec=60\nStandardOutput=append:{log}\nStandardError=append:{log}\n\n"
            f"[Install]\nWantedBy=default.target\n")


def _task_xml(cmd: list[str], env: dict) -> str:
    # whoami gives DOMAIN\user (COMPUTER\user for a local account); $USERDOMAIN is "WORKGROUP" in an SSH session, which no account has
    who = _run("whoami").stdout.strip()
    user = escape(who if "\\" in who else os.environ.get("USERDOMAIN", "") + "\\" + getpass.getuser() if os.environ.get("USERDOMAIN")
                  else getpass.getuser())
    args = escape(subprocess.list2cmdline(cmd[1:]))
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>StreamEmbed screen capture</Description></RegistrationInfo>
  <Triggers><LogonTrigger><Enabled>true</Enabled><UserId>{user}</UserId></LogonTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>{user}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure><Interval>PT1M</Interval><Count>999</Count></RestartOnFailure>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author"><Exec><Command>{escape(cmd[0])}</Command><Arguments>{args}</Arguments></Exec></Actions>
</Task>
"""


def install_service(interval: float | None = None, monitors: str | list[int] | None = None, skip_unchanged: bool | None = None,
                    index_dir: str | Path | None = None, index_type: str | None = None, model: str | None = None,
                    models_repo: str | None = None, device: str | None = None, width: int | None = None,
                    directory: str | Path | None = None) -> Path:
    """Capture every `interval` seconds from now on and after every login, as build_index_from_now does. With `directory`, the
    screenshots you already have there are indexed first (as build_index_from_directory; after a restart it continues where it
    stopped), then capture starts. Settings left as None come from the config when the service starts (so `set_config` and a
    restart change them), or from `configure()` where this process changed them. Installing again replaces the running service.
    A `directory` that is missing, a file, the index folder or inside it raises before anything is installed.
    Returns the service file. On macOS the first start asks for Screen Recording permission for this Python; after allowing it,
    call install_service() again."""
    system = _system()
    from .api import _explicit, _settings
    from .config import _coerce
    if directory:
        folder = Path(directory).expanduser()
        if not folder.is_dir():
            raise NotADirectoryError(f"{folder} is a file, not a folder") if folder.exists() else FileNotFoundError(
                f"Folder not found: {folder}")
    if monitors is not None:
        monitors = _coerce("monitors", monitors)
    for k, v in dict(index_type=index_type, device=device, width=width, interval=interval).items():
        if v is not None:
            _coerce(k, v)
    index_dir = Path(index_dir or _settings()["index_dir"]).expanduser().resolve()
    if directory:
        from .api import _check_folder
        _check_folder(Path(directory).expanduser().resolve(), index_dir)     # the service could never index it
    index_dir.mkdir(parents=True, exist_ok=True)
    log = str(index_dir / "service.log")
    cmd = [_python(system), "-m", "streamembed", "run", "--index-dir", str(index_dir), "--log", log]
    # only what configure() set in this process; every other setting is read from the config when the service starts
    session = {k: _explicit[k] for k in ("index_type", "model", "models_repo", "device", "width") if k in _explicit}
    given = dict(index_type=index_type, model=model, models_repo=models_repo, device=device, width=width, interval=interval)
    for k, v in {**session, **{k: v for k, v in given.items() if v is not None}}.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    if monitors is not None:
        cmd += ["--monitors", monitors if isinstance(monitors, str) else ",".join(str(int(m)) for m in monitors)]
    if skip_unchanged is not None:
        cmd.append("--skip-unchanged" if skip_unchanged else "--keep-unchanged")
    if directory:
        cmd += ["--folder", str(Path(directory).expanduser().resolve())]
    env = {k: os.environ[k] for k in ENV_KEYS if k in os.environ}
    if os.environ.get("STREAMEMBED_CONFIG"):        # your settings file, as a full path (the service starts in another folder)
        from .config import config_path
        env["STREAMEMBED_CONFIG"] = str(config_path().resolve())
    path = _paths()[system]
    path.parent.mkdir(parents=True, exist_ok=True)
    if system == "macos":
        path.write_bytes(_plist(cmd, log, env))
        domain = f"gui/{os.getuid()}"
        _bootout(domain)
        r = _bootstrap(domain, path)
    elif system == "linux":
        env.update({k: os.environ[k] for k in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY") if k in os.environ})
        if "DISPLAY" not in env:
            raise RuntimeError("DISPLAY is not set: install the service from a terminal in your desktop session (screen capture "
                               "needs X11).")
        path.write_text(_unit(cmd, log, env), encoding="utf-8")
        _run("systemctl", "--user", "daemon-reload")
        r = _run("systemctl", "--user", "enable", UNIT)
        if r.returncode == 0:
            r = _run("systemctl", "--user", "restart", UNIT)
    else:
        if env:                             # Task Scheduler has no per-task environment: pass it through the command line
            cmd += ["--env", *(f"{k}={v}" for k, v in env.items())]
        path.write_text(_task_xml(cmd, env), encoding="utf-16")
        _run("schtasks", "/End", "/TN", TASK)
        r = _run("schtasks", "/Create", "/TN", TASK, "/XML", str(path), "/F")
        if r.returncode == 0:
            r = _run("schtasks", "/Run", "/TN", TASK)
    if r.returncode:
        raise RuntimeError(f"Could not start the service: {(r.stderr or r.stdout).strip()}")
    return path


def uninstall_service() -> None:
    """Stop the service and remove it (the index stays)."""
    system = _system()
    if system == "macos":
        _bootout(f"gui/{os.getuid()}")
    elif system == "linux":
        _run("systemctl", "--user", "disable", "--now", UNIT)
    else:
        _run("schtasks", "/End", "/TN", TASK)
        _run("schtasks", "/Delete", "/TN", TASK, "/F")
    _paths()[system].unlink(missing_ok=True)
    if system == "linux":
        _run("systemctl", "--user", "daemon-reload")


def service_status() -> dict:
    """installed, running, pid (None where the system does not report it), log (the service's log file), args (its command line)."""
    system = _system()
    path = _paths()[system]
    out = dict(installed=path.exists(), running=False, paused=False, pid=None, log=None, index_dir=None, args=None)
    if not path.exists():
        return out
    if system == "macos":
        args = plistlib.loads(path.read_bytes())["ProgramArguments"]
        r = _launchctl("print", f"gui/{os.getuid()}/{LABEL}")
        pid = re.search(r"^\s*pid = (\d+)", r.stdout, re.M) if r.returncode == 0 else None
        out.update(running=pid is not None, pid=int(pid.group(1)) if pid else None)
    elif system == "linux":
        line = re.search(r"^ExecStart=(.*)$", path.read_text(encoding="utf-8"), re.M)
        args = shlex.split(line.group(1)) if line else None
        r = _run("systemctl", "--user", "show", UNIT, "--property=ActiveState,MainPID")
        props = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
        pid = int(props.get("MainPID", "0") or 0)
        out.update(running=props.get("ActiveState") == "active", pid=pid or None)
    else:
        xml = path.read_text(encoding="utf-16")
        cmd, arg = re.search(r"<Command>(.*?)</Command>", xml), re.search(r"<Arguments>(.*?)</Arguments>", xml)
        args = [unescape(cmd.group(1))] + [t.strip('"') for t in shlex.split(unescape(arg.group(1)), posix=False)] if cmd and arg else None
        r = _run("schtasks", "/Query", "/TN", TASK, "/FO", "LIST")
        out.update(running=bool(re.search(r":\s*Running", r.stdout)))
    if args and "--log" in args:
        out["log"] = args[args.index("--log") + 1]
    if args and "--index-dir" in args:
        out["index_dir"] = args[args.index("--index-dir") + 1]
        out["paused"] = (Path(out["index_dir"]) / "paused").exists()
    out["args"] = args
    return out


# ---- the service process
def main(argv: list[str] | None = None) -> int:
    """Capture until the service manager stops it (SIGTERM ends the screenshot in progress, then exits)."""
    ap = argparse.ArgumentParser(prog="python -m streamembed", description="StreamEmbed capture service (see install_service)")
    ap.add_argument("--index-dir", default=None)        # every setting left out comes from the config
    ap.add_argument("--index-type", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--models-repo", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--interval", type=float, default=None)
    ap.add_argument("--monitors", default=None)
    ap.add_argument("--keep-unchanged", dest="skip_unchanged", action="store_const", const=False, default=None)
    ap.add_argument("--skip-unchanged", dest="skip_unchanged", action="store_const", const=True)
    ap.add_argument("--folder", default=None, help="index the screenshots in this folder first, then capture")
    ap.add_argument("--log", default=None, help="append output to this file (Task Scheduler has no log of its own)")
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs to set before loading anything")
    a = ap.parse_args(argv)
    if a.log and sys.platform == "win32":
        sys.stdout = sys.stderr = open(a.log, "a", buffering=1, encoding="utf-8")
    for kv in a.env:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    import streamembed as se
    from .config import _coerce, get_config

    def _bad_setting(e: Exception) -> int:
        print(f"streamembed service: {e.args[0] if isinstance(e, KeyError) else e}. Fix the setting, then `streamembed start` again.",
              flush=True)
        return 0                            # a clean exit: restarting would not help

    def _busy(e: Exception) -> int:
        print(f"streamembed service: {e} The service manager starts it again in a minute.", flush=True)
        return 1                            # another writer may have stopped by then, a network error may be gone

    monitors = None if a.monitors is None else a.monitors if a.monitors in ("primary", "all") else [int(m) for m in a.monitors.split(",")]
    try:
        se.configure(index_dir=a.index_dir, index_type=a.index_type, model=a.model, models_repo=a.models_repo, device=a.device,
                     width=a.width)
        interval = max(_coerce("interval", get_config("interval") if a.interval is None else a.interval), se.api.MIN_LIVE_INTERVAL)
    except (KeyError, ValueError, ImportError) as e:     # a bad setting: restarting would not help
        return _bad_setting(e)
    if sys.platform == "darwin":            # first thing, so that `streamembed start` can say at once whether capture will work
        from .capture import _preflight
        print(SCREEN_ALLOWED if _preflight() else SCREEN_NOT_ALLOWED, flush=True)
    if a.folder:
        def interrupt(*_):
            raise KeyboardInterrupt         # build_index_from_directory stops after the screenshot in progress
        signal.signal(signal.SIGTERM, interrupt)
        try:
            stats = se.build_index_from_directory(a.folder, interval=a.interval, skip_unchanged=a.skip_unchanged)
        except (FileNotFoundError, NotADirectoryError, PermissionError) as e:   # capture anyway (exiting would restart every minute)
            print(f"streamembed service: {e}; capturing without it", flush=True)
        except (ValueError, NotImplementedError) as e:    # the index was built with another model, a model not supported: the same
            return _bad_setting(e)                        # after a restart
        except RuntimeError as e:                         # another process writes the index
            return _busy(e)
        else:
            print(f"streamembed service: folder {a.folder}: {stats}", flush=True)
            if stats["stopped"]:
                return 0
    try:
        cap = se.build_index_from_now(interval=interval, monitors=monitors, skip_unchanged=a.skip_unchanged)
    except PermissionError as e:
        print(f"streamembed service: {e} To run it again: `streamembed start` (or se.install_service()).", flush=True)
        return 0                            # a clean exit: not restarted until it is installed again
    except (ValueError, NotImplementedError) as e:
        return _bad_setting(e)
    except RuntimeError as e:
        return _busy(e)
    print(f"streamembed service: capturing every {interval:g} s into {se.api._settings()['index_dir']} (pid {os.getpid()})", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: cap._stop.set())
    from . import queryserver
    enc = se.api._encoder()
    if hasattr(enc, "warm_up"):         # before the page opens: the first searches are as fast as the next
        enc.warm_up()
    try:                                # searches from other processes use this model instead of loading their own
        stop_queries = queryserver.serve(se.api._settings()["index_dir"], enc.encode_query, enc.name)
    except OSError as e:
        print(f"streamembed service: searches will load their own model ({e})", flush=True)
        stop_queries = lambda: None     # noqa: E731
    try:                                # the search page: `streamembed search` opens it
        page_url, stop_page = se.api.serve_page()
        print(f"streamembed service: search page at {page_url.split('?')[0]} (streamembed search opens it)", flush=True)
    except OSError as e:
        print(f"streamembed service: no search page ({e})", flush=True)
        stop_page = lambda: None        # noqa: E731
    try:
        cap.wait()
    finally:
        stop_page()
        stop_queries()
    if cap.stalled:                     # stuck in a screen grab: a new process is the only way out
        print(f"streamembed service: screen capture has not returned for {se.api.STALL_S:.0f} s; exiting so that the service "
              f"manager starts it again, {cap.stats}", flush=True)
        return 1
    ended = not cap._stop.is_set()      # the capture ended by itself (an error), not by SIGTERM or Ctrl-C
    cap.stop()
    if ended:                           # exit non-zero so the service manager restarts it
        print(f"streamembed service: capture ended with an error, {cap.stats}", flush=True)
        return 1
    print(f"streamembed service: stopped, {cap.stats}", flush=True)
    return 0


if __name__ == "__main__":         # services installed before python -m streamembed
    sys.exit(main())
