"""Frames to index: the displays every few seconds, or a folder of screenshots."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image

BLOCK = 32
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MACOS = sys.platform == "darwin"


def fit_width(img: Image.Image, width: int) -> Image.Image:
    """Shrink to at most `width` keeping the aspect ratio, both sides multiples of 32 (one block is 32x32 px). Narrower images are not
    enlarged: that would only add blocks without adding detail."""
    w = max(BLOCK, min(width, img.width) // BLOCK * BLOCK)
    h = max(BLOCK, round(img.height * w / img.width / BLOCK) * BLOCK)
    img = img.convert("RGB")
    return img if img.size == (w, h) else img.resize((w, h), Image.BICUBIC)


# ---- a folder of existing screenshots
_EPOCH = re.compile(r"^(\d{10}|\d{13})(?:[._-](\d{1,2}))?$")   # 1793413519.webp, 1793413554_1.webp (display 1), milliseconds
_EPOCH_ROW = re.compile(r"^(\d{13})_\d{3,}$")    # <ms>_<row>.jpg: captures stored by StreamEmbed before 0.3.0
_DATETIME = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_.]?(\d{2})[-_.]?(\d{2})\D{0,6}?(?:(AM|PM|오전|오후|上午|下午)\s*)?(\d{1,2})[.:_-]?(\d{2})[.:_-]?(\d{2})"
                       r"(?:\s*(AM|PM|오전|오후|上午|下午))?", re.IGNORECASE)
# "Screenshot 2026-11-06 at 15.20.11", "스크린샷 2026-11-06 오후 3.20.11", "截屏2026-11-06 下午3.20.11"
_DISPLAY = re.compile(r"^\s*(?:\((\d{1,2})\)|[_-](\d{1,2}))\s*$")       # what follows the time: " (2)" (macOS), "_1" (OpenRecall)


def _name_time(stem: str) -> tuple[float | None, str]:
    """(capture time from the file name or None, display tag: the number right after the time, "" if none)."""
    m = _EPOCH_ROW.match(stem)
    if m:                                       # the row number is not a display number
        return int(m.group(1)) / 1000, ""
    m = _EPOCH.match(stem)
    if m:
        t = int(m.group(1))
        return (t / 1000 if len(m.group(1)) == 13 else float(t)), m.group(2) or ""
    m = _DATETIME.search(stem)
    if not m:
        return None, ""
    y, mo, d, ap1, h, mi, s, ap2 = m.groups()
    h, ap = int(h), (ap1 or ap2 or "").upper()
    if ap in ("PM", "오후", "下午") and h < 12:
        h += 12
    elif ap in ("AM", "오전", "上午") and h == 12:
        h = 0
    tag = _DISPLAY.match(stem[m.end():])
    try:
        return datetime(int(y), int(mo), int(d), h, int(mi), int(s)).timestamp(), (tag.group(1) or tag.group(2)) if tag else ""
    except ValueError:
        return None, ""


def _exif_time(im: Image.Image) -> float | None:
    try:
        exif = im.getexif()
        v = exif.get_ifd(0x8769).get(36867) or exif.get(306)     # DateTimeOriginal, else DateTime
        return datetime.strptime(str(v).strip(), "%Y:%m:%d %H:%M:%S").timestamp() if v else None
    except Exception:
        return None


def _cannot_read(folder: Path) -> PermissionError:
    note = (" On macOS, allow it for the app that runs Python in System Settings > Privacy & Security > Files and Folders (or Full "
            "Disk Access); the background service runs this Python") if MACOS else ""
    return PermissionError(f"Cannot read the folder {folder}: permission denied.{note}")


def folder_files(folder: Path, interval: float | None = 5.0, skip: Path | None = None,
                 exclude: set[str] | None = None) -> tuple[list[tuple[float, str, tuple]], int, int, int]:
    """Screenshots under `folder` (recursive) -> ([(capture time, path, screen)] in time order, files found, files that are not
    readable images). Capture time: from the file name, else EXIF, else the file's modification time (not thinned by `interval`: it
    may be a copy time). Screen = (display number in the file name, image size, subfolder): the number right after the time ("_1",
    " (2)") tells displays of one size apart, the size tells apart displays the names do not number, and each subfolder (a recording
    of its own) is its own screen. With `interval`, at most one screenshot per screen every `interval` seconds is kept. Only file
    headers are read here. A folder with no screenshots gives ([], 0, 0). Raises FileNotFoundError (missing), NotADirectoryError (a
    file) or PermissionError (a folder that cannot be listed, such as ~/Desktop on macOS without Files and Folders access); a
    subfolder that cannot be listed is named on stderr and left out. `skip`: a folder left out of the scan (the index folder, whose
    stored images would otherwise be indexed again). `exclude`: paths read before (the index's); they are left out before the
    interval thins the rest, and counted in the fourth value."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is a file, not a folder")
    try:
        os.listdir(folder)
    except PermissionError:
        raise _cannot_read(folder) from None
    unlisted: list[str] = []
    files = []
    skip = Path(skip).resolve() if skip else None
    for top, dirs, names in os.walk(folder, onerror=lambda e: unlisted.append(e.filename)):
        if skip is not None:
            dirs[:] = [d for d in dirs if (Path(top) / d).resolve() != skip]
        files += [p for p in (Path(top) / n for n in names) if p.suffix.lower() in EXTS and p.is_file()]
    if unlisted:
        print(f"streamembed: {len(unlisted)} subfolder(s) of {folder} could not be read and are left out, such as {unlisted[0]}",
              file=sys.stderr, flush=True)
    items, unreadable = [], 0
    for p in files:
        try:
            with Image.open(p) as im:            # lazy: reads the header, not the pixels
                ts, display = _name_time(p.stem)
                taken = ts or _exif_time(im)
                items.append((taken or p.stat().st_mtime, str(p), (display, *im.size, str(p.parent)), taken is not None))
        except Exception:
            unreadable += 1                      # not an image after all, or not readable
    items.sort()
    done = 0
    if exclude:
        done = sum(1 for it in items if it[1] in exclude)
        items = [it for it in items if it[1] not in exclude]
    if interval:
        # Only times read from a name or EXIF are capture times. A file time may be when the file was copied (a copied folder has all
        # its files within seconds), so those screenshots are all kept; the change gate still drops the unchanged ones.
        last: dict[tuple, float] = {}
        kept = []
        for ts, p, screen, taken in items:
            if not taken or ts - last.get(screen, -float("inf")) >= interval:
                kept.append((ts, p, screen, taken))
                last[screen] = ts if taken else last.get(screen, -float("inf"))
        items = kept
    return [(ts, p, screen) for ts, p, screen, _ in items], len(files), unreadable, done


def load_frames(items: list[tuple[float, str, tuple]], width: int, workers: int | None = None) -> Iterator[tuple]:
    """(image fitted to `width`, ts, screen, path) per item, in order, decoded by `workers` threads a little ahead of the consumer.
    A file that cannot be read gives (None, ts, screen, path)."""
    def load(item):
        ts, p, screen = item
        try:
            with Image.open(p) as im:
                return fit_width(im, width), ts, screen, p
        except Exception:
            return None, ts, screen, p

    workers = workers or min(8, os.cpu_count() or 1)
    it = iter(items)
    with ThreadPoolExecutor(workers) as pool:
        ahead = deque(pool.submit(load, x) for x in islice(it, 4 * workers))
        while ahead:
            done = ahead.popleft()
            nxt = next(it, None)
            if nxt is not None:
                ahead.append(pool.submit(load, nxt))
            yield done.result()


# ---- the screen. macOS reports permission, lock state and the front window; elsewhere these checks pass.
def screen_permission() -> bool:
    """Without Screen Recording permission macOS returns only the wallpaper."""
    if not MACOS:
        return True
    import Quartz
    if Quartz.CGPreflightScreenCaptureAccess():
        return True
    Quartz.CGRequestScreenCaptureAccess()      # shows the system prompt
    return False


# macOS 15 renamed the pane "Screen & System Audio Recording"; only the name changed (nothing here records audio). No command can
# turn the permission on (it is kept in the SIP-protected TCC database), so we open the pane and say what to turn on in it.
SCREEN_PANE = "System Settings > Privacy & Security > Screen & System Audio Recording (Screen Recording before macOS 15)"
SCREEN_SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
TERMINALS = {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm", "vscode": "Visual Studio Code (or Cursor)", "WezTerm": "WezTerm",
             "ghostty": "Ghostty", "WarpTerminal": "Warp", "tmux": "the terminal app that runs tmux"}


def python_app() -> str:
    """The Python macOS lists in the pane when Python itself is asked (the background service, started by launchd): Python.app for a
    framework build (python.org, Homebrew), else the interpreter with symlinks resolved (a venv's python links to it)."""
    app = Path(sys.base_prefix) / "Resources" / "Python.app"
    return str(app if app.is_dir() else Path(os.path.realpath(sys.executable)))


def permission_app() -> str:
    """What to turn on in the pane: macOS asks for the app that started this process, the terminal app when it runs in one."""
    term = os.environ.get("TERM_PROGRAM")
    return TERMINALS.get(term, term) if term else python_app()


def turn_on(app: str) -> str:
    """'turn on X' for the pane; a path is added with + when it is not listed yet."""
    if not app.startswith("/"):
        return f"turn on {app}"
    name = Path(app).stem if app.endswith(".app") else Path(app).name       # as the pane lists it
    return f"turn on {name}, or add it with + (press Cmd-Shift-G and paste {app})"


def open_screen_settings() -> bool:
    """Open the pane in System Settings (macOS). False when it could not be opened, or elsewhere."""
    if not MACOS:
        return False
    try:
        return subprocess.run(["open", SCREEN_SETTINGS_URL], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def screen_permission_help(opened: bool = False) -> str:
    """How to allow screen capture, in one message. `opened`: the pane was opened (open_screen_settings)."""
    return (f"Screen capture needs a permission that only you can give: in {SCREEN_PANE}, {turn_on(permission_app())}, then run "
            "it again. Nothing records audio; only the pane's name mentions it. "
            + ("System Settings is open at that pane now." if opened else f"To open the pane: open '{SCREEN_SETTINGS_URL}'"))


def _preflight() -> bool:
    """Whether this process may record the screen (macOS), without asking for it."""
    try:
        import Quartz
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        return True


def _locked() -> bool:
    if not MACOS:
        return False
    import Quartz
    session = Quartz.CGSessionCopyCurrentDictionary()
    return bool(session and session.get("CGSSessionScreenIsLocked", 0))


def _front_window() -> tuple[str | None, str | None]:
    """(app, window title) of the frontmost normal window."""
    if not MACOS:
        return None, None
    import Quartz
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    for win in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []:
        if win.get("kCGWindowLayer", 1) == 0:
            return win.get("kCGWindowOwnerName"), win.get("kCGWindowName") or None
    return None, None


_excluded: dict = {}     # the exclude lists as last read: kept while the config file has a mistake in it


def private(app: str | None, title: str | None) -> bool:
    """Whether the front window is one not to capture (exclude_apps, exclude_titles). The lists are read at every call, so a change
    applies at once. While the config file cannot be read (a typo in any setting), the lists read last are used and this is said once
    on stderr; if they were never read, every window counts as private."""
    from .config import get_config
    try:
        lists = (get_config("exclude_apps"), get_config("exclude_titles"))
        if _excluded.get("error"):
            print("streamembed: the settings can be read again", file=sys.stderr, flush=True)
        _excluded.update(lists=lists, error=None)
    except Exception as e:
        if not _excluded.get("error"):
            print(f"streamembed: {e} Until then, the exclude lists read before are used"
                  + ("" if "lists" in _excluded else " (none were: nothing is captured)") + ".", file=sys.stderr, flush=True)
        _excluded["error"] = str(e)
        if "lists" not in _excluded:
            return True                     # never read: do not capture
        lists = _excluded["lists"]
    a, t = (app or "").casefold(), (title or "").casefold()
    return any(x.casefold() in a for x in lists[0]) or any(x.casefold() in t for x in lists[1])


def _grabber():
    """mss.MSS (mss 10.2 and later; mss.mss is deprecated there), else mss.mss."""
    import mss
    return getattr(mss, "MSS", None) or mss.mss


SLOW_GRAB_S = 10.0     # a screen grab slower than this is reported once (a missing Screen Recording permission on macOS, or contention)


MASK_GRAY = (128, 128, 128)
PAGE_TITLE = "StreamEmbed"      # the results page's title ends with it: "<question> – StreamEmbed", or is it


def page_title(question: str | None = None) -> str:
    return f"{question} – {PAGE_TITLE}" if question else PAGE_TITLE


def is_page_title(name: str | None) -> bool:
    return bool(name) and (name == PAGE_TITLE or name.endswith(f" – {PAGE_TITLE}"))
SKIP_COVERED = 0.9      # a display this much covered by the search page is not stored (nothing on it but the page)


def _clip(a: tuple, b: tuple) -> tuple | None:
    """The part of rectangle a (left, top, width, height) inside b, or None."""
    l, t = max(a[0], b[0]), max(a[1], b[1])
    r, btm = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return (l, t, r - l, btm - t) if r > l and btm > t else None


def _window_list() -> list[dict] | None:
    """macOS: the windows on screen, front to back, as {"layer", "alpha", "bounds": (left, top, width, height) in points, "name": the
    window title (read with the Screen Recording permission the capture has)}. None elsewhere."""
    if not MACOS:
        return None
    import Quartz
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    out = []
    for w in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []:
        b = w.get("kCGWindowBounds") or {}
        out.append({"layer": int(w.get("kCGWindowLayer", 0)), "alpha": float(w.get("kCGWindowAlpha", 1)), "name": w.get("kCGWindowName") or "",
                    "bounds": (float(b.get("X", 0)), float(b.get("Y", 0)), float(b.get("Width", 0)), float(b.get("Height", 0)))})
    return out


def page_masks(displays: list[tuple], pages: list[dict], windows: list[dict] | None,
               pixels: bool = sys.platform == "win32") -> dict[int, tuple[list, list]]:
    """Where the search page is on each display: {display index: (rectangles to paint over, rectangles above them to leave as they
    are)}, in the displays' units. pages: {"rect": (left, top, width, height) in points, "dpr", "focused"} (webui.front). The
    displays are in points on macOS and in pixels on Windows (pixels=True: the rectangle scaled by the pixel ratio is tried first);
    a page that overlaps no display in either is not painted.
    windows (macOS, _window_list): the page's window is the one with its bounds; the windows in front of it are left as they are, so
    a window over part of the page is still captured. Without the list, or when the page's window is not in it, a focused page is
    painted whole (it is in front) and one without focus is not painted (what covers it is not known).
    A window in the list whose title is the page's (is_page_title: the browser shows it while the page is the window's tab) is
    painted too, so a page written to a file, which has no server to tell where it is, is not captured either."""
    out: dict[int, tuple[list, list]] = {}

    def paint(r: tuple, above: list) -> None:
        for j, d in enumerate(displays):
            part = _clip(r, d)
            if part:
                p, keep = out.setdefault(j, ([], []))
                p.append(part)
                keep.extend(k for k in (_clip(a, part) for a in above) if k)

    def in_front(at: int) -> list:
        return [w["bounds"] for w in windows[:at] if w["alpha"] > 0 and w["bounds"][2] > 0 and w["bounds"][3] > 0
                and not any(_clip(w["bounds"], d) == d for d in displays)]     # a display-wide overlay hides nothing

    for at, w in enumerate(windows or []):
        if w["layer"] == 0 and is_page_title(w.get("name")):
            paint(w["bounds"], in_front(at))
    for page in pages:
        rect, dpr = tuple(page["rect"]), page.get("dpr") or 1.0
        scaled = tuple(v * dpr for v in rect)
        r = next((c for c in ((scaled, rect) if pixels else (rect, scaled)) if any(_clip(c, d) for d in displays)), None)
        if r is None:
            continue
        above = None
        if windows is not None and r is rect:
            at = next((i for i, w in enumerate(windows)
                       if w["layer"] == 0 and all(abs(a - b) <= 4 for a, b in zip(w["bounds"], rect))), None)
            if at is not None:
                above = in_front(at)
        if above is None:
            if not page.get("focused"):
                continue
            above = []
        paint(r, above)
    return out


def _mask(img: Image.Image, display: tuple, paint: list, keep: list) -> float:
    """Paint the page's rectangles gray in a display's image (the rectangles in `keep` left as they are); returns the share of the
    image painted."""
    from PIL import ImageDraw
    sx, sy = img.width / display[2], img.height / display[3]

    def box(r):
        return (round((r[0] - display[0]) * sx), round((r[1] - display[1]) * sy),
                round((r[0] + r[2] - display[0]) * sx) - 1, round((r[1] + r[3] - display[1]) * sy) - 1)

    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    for r in paint:
        draw.rectangle(box(r), fill=255)
    for r in keep:
        draw.rectangle(box(r), fill=0)
    img.paste(MASK_GRAY, (0, 0, img.width, img.height), mask)
    return mask.histogram()[255] / (img.width * img.height)


def screen_frames(width: int, monitors: str | list[int] = "all",
                  pages: Callable[[], list] | None = None) -> Iterator[list | None]:
    """One item per call: the frames of the chosen displays [(image, ts, screen, None, tag)], or None while the screen is locked or a
    private window is in front. monitors: "all", "primary", or display numbers (1 = primary, as in mss). Displays are listed again at
    every capture, so one connected meanwhile is captured from then on. screen = the display's (left, top, width, height), its stream
    key; tag = "" for the primary display, "1", "2", ... for the others (the stored file name suffix, as "_1"). pages() -> the
    search pages on screen (webui.front, the default): their windows are painted gray (page_masks: not the windows in front of
    them), so a search does not find earlier results; a display covered SKIP_COVERED or more is left out."""
    if pages is None:
        from .webui import front as pages
    if not (monitors in ("all", "primary") or (isinstance(monitors, (list, tuple)) and monitors and all(int(m) >= 1 for m in monitors))):
        raise ValueError(f"monitors must be 'all', 'primary' or display numbers from 1, not {monitors!r}")
    failing = missing = slow = False
    while True:
        try:
            app, title = _front_window()
            hidden = _locked() or private(app, title)
        except Exception as e:      # a failing window or lock check: capture nothing this time, rather than end the capture
            if not failing:
                print(f"streamembed: the lock and window checks failed, retrying every interval: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
            failing = True
            yield None
            continue
        if hidden:
            yield None
            continue
        try:
            with _grabber()() as sct:                   # a new list of displays every time
                n = len(sct.monitors) - 1               # sct.monitors[0] is all displays together
                ids = range(1, n + 1) if monitors == "all" else [1] if monitors == "primary" else [int(m) for m in monitors if int(m) <= n]
                if not ids and not missing:
                    print(f"streamembed: none of displays {monitors} is connected ({n} connected); nothing is captured until one is",
                          file=sys.stderr, flush=True)
                missing = not ids
                ts = time.time()
                out = []
                ids = list(ids)
                displays = [tuple(sct.monitors[i][k] for k in ("left", "top", "width", "height")) for i in ids]
                try:
                    on = pages()
                    masks = page_masks(displays, on, _window_list())
                except Exception:       # the page's window could not be placed: capture as it is
                    masks = {}
                for j, i in enumerate(ids):
                    m = sct.monitors[i]
                    t0 = time.monotonic()
                    shot = sct.grab(m)
                    took = time.monotonic() - t0
                    if took > SLOW_GRAB_S and not slow:
                        slow = True
                        why = (f" The app that started Python has no permission to record the screen: {SCREEN_PANE}."
                               if MACOS and not _preflight() else
                               " Another app that records the screen at the same time can slow it down.")
                        print(f"streamembed: a screen grab took {took:.0f} s.{why}", file=sys.stderr, flush=True)
                    img = Image.frombytes("RGB", shot.size, shot.rgb)                     # full size: _ingest fits it to `width`
                    if j in masks and _mask(img, displays[j], *masks[j]) >= SKIP_COVERED:
                        continue
                    out.append((img, ts, displays[j], None, "" if i == 1 else str(i - 1)))
        except Exception as e:      # no screen to grab (Windows lock screen or UAC prompt, X display gone): try again next time
            if not failing:
                print(f"streamembed: screen capture failed, retrying every interval: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            failing = True
            yield None
            continue
        if failing:
            print("streamembed: screen capture works again", file=sys.stderr, flush=True)
            failing = False
        yield out
