# StreamEmbed

Search your own screen history by describing what you saw. StreamEmbed indexes the screenshots you already have and keeps capturing your screen as you work, then finds a past screen from a sentence such as "the flight search page to Denver". Everything runs on your machine.

## Installation

```bash
uv tool install streamembed      # the streamembed command, in an environment of its own
uv pip install streamembed       # or into your project's environment
```

Python 3.12 or later. The first run downloads the model (about 4 GB). On macOS, allow screen capture for Python when asked (System Settings > Privacy & Security > Screen & System Audio Recording; nothing records audio, only the name mentions it). StreamEmbed opens that pane and prints what to turn on; then run `streamembed start` again.

## Quick Start

### From the terminal

```bash
streamembed start --folder ~/Desktop                   # index the screenshots there, then capture every 5 s, now and after every login
streamembed search "the flight search page to Denver"  # a minute or more after start: open the search page with this question
streamembed status                                     # also: pause, resume, stop
```

`--folder` is where your screenshots are saved: `~/Desktop` on macOS, `~/Pictures/Screenshots` on Windows and Ubuntu. `streamembed --help` shows more examples, and `streamembed COMMAND -h` the options of a command.

### From Python

```python
import time
import streamembed as se

se.build_index_from_directory("~/Desktop")     # screenshots you already have
capture = se.build_index_from_now()            # capture the screen every 5 s, in the background
time.sleep(60)                                 # let it capture for a minute

results = se.query_index("the flight search page to Denver")
se.visualize_results(results)                  # each match with the screens before and after it
capture.stop()
```

## Settings

`streamembed config` lists every setting and its value; `streamembed config set KEY VALUE` changes one. The background capture reads them when it starts, so run `streamembed start` again after a change.

| Setting | Default | What it does |
|---|---|---|
| `index_dir` | `~/screenshot_index` | The index folder; live captures go in `captures/` inside it |
| `interval` | 5 | Seconds between captures. A screen that did not change is not stored |
| `capture_max_gb` | 20 | Live captures are kept under this size; past it, the oldest are removed (0: no limit) |
| `capture_keep_days` | 0 | Live captures older than this many days are removed (0: kept however old) |
| `exclude_apps` | password managers | Nothing is captured while one of these apps is in front (macOS) |
| `model` | `qwen3-vl-2b` | `qwen3-vl-8b` needs a GPU with about 20 GB |

## Privacy

- The index and the captures stay in the index folder, readable only by you and not encrypted. Nothing is sent over the network after the model download.
- On macOS, nothing is captured while the screen is locked or a password manager or private browsing window is in front. On Linux and Windows these checks are not implemented yet.
- The search page is painted gray in the captures, so a search does not find earlier results.

## Requirements

Tested on macOS 15 (Apple silicon), Ubuntu 24.04 under X11 and Windows Server 2025; Wayland is not supported. Encoding needs about 8 GB of memory. StreamEmbed pins torch 2.8.0 and transformers 5.17.0, so install it in an environment of its own if your project needs other versions.

## Reference

[`docs/reference.md`](docs/reference.md): every command and function, all settings, the index folder, the index types and the models.

## Development

```bash
uv venv && uv pip install -e ".[dev,hnsw]"
.venv/bin/python -m pytest          # tests use a stand-in encoder; no model needed
```

## License

Apache-2.0. See `LICENSE` and `NOTICE`; the model weights are downloaded under their own licenses.
