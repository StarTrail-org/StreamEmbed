"""Settings: our defaults (defaults.toml in the package) and your changes (a config file of your own).

    import streamembed as se
    se.get_config()                         # every setting
    se.set_config(interval=10, skip_unchanged=False)
    se.reset_config("interval")             # back to the default (no names: all)

    streamembed config                      # every setting and where it comes from
    streamembed config set interval 10 skip_unchanged false
    streamembed config reset interval

Precedence: a value given to a function or command, then configure() (for its process), then $STREAMEMBED_INDEX (index_dir) /
$STREAMEMBED_MODELS (models_repo), then your file, then the defaults. Your file is ~/.config/streamembed/config.toml
(%APPDATA%/streamembed/config.toml on Windows), or $STREAMEMBED_CONFIG; it holds only the settings you changed, and every value in it is
checked when it is set and when it is read. The background capture reads the settings when it starts.
"""
from __future__ import annotations

import math
import os
import re
import sys
import tomllib
from pathlib import Path

DEFAULTS_FILE = Path(__file__).parent / "defaults.toml"
ENV = {"index_dir": "STREAMEMBED_INDEX", "models_repo": "STREAMEMBED_MODELS"}
_cache: dict = {}


def config_path() -> Path:
    if os.environ.get("STREAMEMBED_CONFIG"):
        return Path(os.environ["STREAMEMBED_CONFIG"]).expanduser()
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "streamembed" / "config.toml"
    return Path.home() / ".config" / "streamembed" / "config.toml"


def _read(path: Path) -> dict:
    """Parsed TOML, cached until the file changes. A file that is not valid TOML raises ValueError naming it and the line."""
    if not path.is_file():
        return {}
    key, mtime = str(path), path.stat().st_mtime_ns
    if _cache.get(key, (None,))[0] != mtime:
        try:
            _cache[key] = (mtime, tomllib.loads(path.read_text(encoding="utf-8")))
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path} is not valid TOML ({e}). Fix or delete the file, or run `streamembed config reset`.") from None
    return _cache[key][1]


def defaults() -> dict:
    return dict(_read(DEFAULTS_FILE))


def _user() -> dict:
    """Your file as written (unknown keys left out)."""
    return {k: v for k, v in _read(config_path()).items() if k in defaults()}


def _checked_user() -> dict:
    """Your file, each value checked: one of the wrong type or out of range raises ValueError naming the file and the setting."""
    out = {}
    for k, v in _user().items():
        try:
            out[k] = _coerce(k, v, strict=False)
        except ValueError as e:
            raise ValueError(f"{config_path()}: {e}. Change it with `streamembed config set {k} VALUE` or "
                             f"`streamembed config reset {k}`.") from None
    return out


def get_config(name: str | None = None):
    """One setting, or all of them as a dict, after your file and the environment."""
    if name is not None and name not in defaults():
        raise KeyError(f"Unknown setting {name!r}; settings: {', '.join(defaults())}")
    merged = {**defaults(), **_checked_user()}
    for k, var in ENV.items():
        if os.environ.get(var):
            merged[k] = os.environ[var]
    return merged[name] if name else merged


def sources() -> dict:
    """Where each setting comes from: "default", "config file" or the environment variable."""
    user = _user()
    return {k: (ENV[k] if k in ENV and os.environ.get(ENV[k]) else "config file" if k in user else "default") for k in defaults()}


INDEX_TYPES = ("flat", "binary", "hnsw", "ivf", "pq")
DEVICES = ("auto", "mps", "cuda", "cpu")
_MIN = {"width": 32, "interval": 0, "heartbeat": 5, "top_k": 1, "page_results": 1, "context": 0, "capture_max_gb": 0, "capture_keep_days": 0}       # the smallest value each number may take
_MAX = {"interval": 86400, "heartbeat": 86400}                                                              # the largest (a day)
_WHAT = {"width": "a width in pixels", "interval": "a number of seconds", "heartbeat": "a number of seconds", "top_k": "a number of results",
         "page_results": "a number of results", "context": "a number of frames", "capture_max_gb": "a size in GB (0: no limit)",
         "capture_keep_days": "a number of days (0: no limit)"}


def _number(name: str, value, kind: type):
    what = _WHAT.get(name, "a number")
    if isinstance(value, bool):
        raise ValueError(f"{name}: {what}, not {value!r}")
    try:
        if kind is int and isinstance(value, float) and not value.is_integer():
            raise ValueError
        v = kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: {what}{' (a whole number)' if kind is int else ''}, not {value!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name}: {what}, not {value!r}")
    if name in _MAX and v > _MAX[name]:
        raise ValueError(f"{name}: at most {_MAX[name]}, not {value!r}")
    if name in _MIN and v < _MIN[name]:
        raise ValueError(f"{name}: at least {_MIN[name]}, not {value!r}")
    return v


def _coerce(name: str, value, strict: bool = True, repo: str | None = None):
    """A value of the setting's type; strings (from the command line) are parsed. Raises KeyError for an unknown setting and ValueError
    naming the setting for a bad value. strict: also check that the hnsw extra is installed for index_type "hnsw", and that a model
    name is in the model list of `repo` (default: the models_repo setting) and supported by the stream encoder, when that list is on
    disk."""
    base = defaults()
    if name not in base:
        raise KeyError(f"Unknown setting {name!r}; settings: {', '.join(base)}")
    d = base[name]
    if name == "monitors":
        original = value
        if isinstance(value, str) and value not in ("all", "primary"):
            try:
                value = [int(m) for m in value.replace(" ", "").strip("[]").split(",") if m]
            except ValueError:
                value = None
        if not (value in ("all", "primary") or (isinstance(value, list) and value and
                                                all(isinstance(m, int) and not isinstance(m, bool) and m >= 1 for m in value))):
            raise ValueError(f'monitors: "all", "primary" or display numbers such as 1,2, not {original!r}')
        return value
    if name == "index_type":
        if value not in INDEX_TYPES:
            raise ValueError(f"index_type: one of {', '.join(INDEX_TYPES)}, not {value!r}")
        if strict and value == "hnsw":
            from .ann import require_hnswlib
            require_hnswlib()                   # ImportError naming the extra to install
        return value
    if name == "capture_format" and str(value).lower() not in ("webp", "jpeg"):
        raise ValueError(f'capture_format: "webp" or "jpeg", not {value!r}')
    if name == "capture_width" and isinstance(value, (int, float, str)) and str(value).strip().lstrip("-").isdigit() \
            and (int(value) < 0 or 0 < int(value) < 32):
        raise ValueError(f"capture_width: 0 (the screen's own size) or at least 32 pixels, not {value!r}")
    if name == "device":
        v = str(value)
        if v not in DEVICES and not re.fullmatch(r"cuda:\d+", v):
            raise ValueError(f"device: one of {', '.join(DEVICES)} (or cuda:N), not {value!r}")
        return v
    if name in ("index_dir", "models_repo", "model") and isinstance(value, (str, Path)) and not str(value).strip():
        raise ValueError(f"{name}: a {'folder' if name == 'index_dir' else 'name'}, not an empty text")
    if name == "capture_dir" and isinstance(value, (str, Path)) and str(value).strip() and \
            not Path(str(value).strip()).expanduser().is_absolute():
        # the service runs in "/", a terminal in any folder: a relative folder would mean a different place for each
        raise ValueError(f"capture_dir: a full path such as ~/Pictures/Screenshots (or empty for captures/ in the index folder), "
                         f"not {value!r}")
    if name == "model" and strict:
        from .models import cached_models
        repo = repo or get_config("models_repo")
        known, usable = cached_models(repo), cached_models(repo, stream_only=True)
        if known is not None and str(value) not in usable:
            why = " (listed, but the stream encoder supports Qwen3-VL models only)" if str(value) in known else ""
            raise ValueError(f"model: one of {', '.join(usable)}, not {value!r}{why}")
    if isinstance(d, bool):
        if isinstance(value, str):
            if value.lower() not in ("true", "false", "yes", "no", "1", "0", "on", "off"):
                raise ValueError(f"{name}: true or false, not {value!r}")
            return value.lower() in ("true", "yes", "1", "on")
        if not isinstance(value, (bool, int)):
            raise ValueError(f"{name}: true or false, not {value!r}")
        return bool(value)
    if isinstance(d, int):
        return _number(name, value, int)
    if isinstance(d, float):
        return _number(name, value, float)
    if isinstance(d, list):
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name}: a list of names, not {value!r}")
        return [str(s) for s in value]
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name}: a text value, not {value!r}")
    return str(value)


def _toml(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write(values: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Your StreamEmbed settings (only the ones you changed). `streamembed config` lists every setting; defaults.toml in the",
             "# package explains them. Edit this file or use `streamembed config set KEY VALUE`."]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines + [f"{k} = {_toml(v)}" for k, v in values.items()]) + "\n", encoding="utf-8")
    os.replace(tmp, path)                   # whole or not at all
    _cache.pop(str(path), None)
    return path


def set_config(**values) -> Path:
    """Change settings and keep them in your config file. A value equal to the default is removed from the file."""
    user, base = _user(), defaults()
    if "models_repo" in values:                     # a model is checked against the model repo given in the same call
        values = {"models_repo": values["models_repo"], **values}
    repo = None
    for k, v in values.items():
        v = _coerce(k, v, repo=repo)
        if k == "models_repo":
            repo = os.environ.get(ENV["models_repo"]) or v
        if v == base[k]:
            user.pop(k, None)
        else:
            user[k] = v
        if k in ("exclude_apps", "exclude_titles"):
            kept = {x.casefold() for x in v}
            dropped = [x for x in base[k] if x.casefold() not in kept]
            if dropped:
                print(f"streamembed: {k} replaces the whole list; these defaults are no longer excluded: {', '.join(dropped)}. "
                      f"To keep them, list them too.", file=sys.stderr)
    return _write(user)


def reset_config(*names: str) -> Path:
    """Back to the defaults: the named settings, or all of them (this also replaces a file that is not valid TOML)."""
    for k in names:
        if k not in defaults():
            raise KeyError(f"Unknown setting {k!r}; settings: {', '.join(defaults())}")
    if not names:
        return _write({})
    user = _user()
    for k in names:
        user.pop(k, None)
    return _write(user)
