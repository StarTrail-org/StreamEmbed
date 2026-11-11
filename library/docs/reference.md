# StreamEmbed reference

Everything the README leaves out: every command and function, all settings, the index folder, the index types and the models.

## Requirements

Python 3.12 or later. Tested on macOS 15 (Apple silicon, GPU), Ubuntu 24.04 under X11 (Xvfb; CPU) and Windows Server 2025 (CPU): installing, indexing a folder, searching, capturing, and the background service (start, pause, resume, stop, restart after a reboot). Not tested: Wayland (not supported), Windows with several monitors, CUDA GPUs. On macOS, capturing the screen needs Screen Recording permission for the app that runs Python; `build_index_from_now` raises `PermissionError` until it is granted. On Windows, capture needs a logged-in desktop session: while the screen is locked, captures fail and are retried. Indexing `~/Desktop` on macOS needs access to it for the terminal (or, for `streamembed start --folder`, the service's Python): allow it in System Settings > Privacy & Security > Files and Folders; without it, indexing stops with a permission error.

Dependencies are pinned to torch 2.8.0 and transformers 5.17.0: the stream encoder re-implements parts of the Qwen3-VL forward pass against transformers internals, and its outputs were checked for equality with these versions. Install StreamEmbed in its own environment (for example `uv tool`, `pipx`, or a venv) if your project needs other versions.

The `hnsw` index type needs the `hnsw` extra (`pip install "streamembed[hnsw]"`), which installs chroma-hnswlib 0.7.6: hnswlib with prebuilt wheels (the `hnswlib` package itself is a source package that needs a C++ compiler). Its wheels cover Python 3.12; on later Python versions it is built from source. It does not use OpenMP and runs in one process with torch (checked on macOS with torch 2.8.0 on MPS). The other types need only numpy. On Windows, the `hnsw` extra needs the Microsoft C++ Build Tools: it has no prebuilt wheel there. The index type is a setting, not an install choice: `streamembed config set index_type pq` switches an existing index without indexing again.

## Privacy

- Vectors and metadata stay in the index folder (owner-only). Live captures are written to `capture_dir` (by default `captures/` in the index folder, also owner-only) as plain image files (WebP by default). If you set `capture_dir` to a folder that a cloud service syncs (iCloud Desktop & Documents, Dropbox, OneDrive), your screen history is uploaded with it.
- On macOS, capture is skipped when the screen is locked, when the frontmost app is a password manager, and when the frontmost window is a private browsing window. Other windows on screen are not checked; use `capture.pause()` for sensitive work. On Linux and Windows these checks are not implemented yet: everything on screen is captured.
- While the search page or a results page written to a file is shown, its browser window is painted gray in the captures, so a search does not find the results of earlier ones. On macOS the window is also found by its title ("… – StreamEmbed"), which the browser shows while the page is the window's tab. Everything else is captured as usual, including a window over part of the page (macOS). A display with nothing on it but the page is not stored. On Windows and Linux, the window is painted only while the page has focus.
- Nothing is sent over the network after the model download.


## Commands

| Command | What it does |
|---|---|
| `streamembed start [--folder DIR] [--interval S] [--monitors all\|primary\|1,2] [--keep-unchanged] [--model M] [--index-type T]` | Capture in the background from now and after every login (see below); with `--folder`, index those screenshots first |
| `streamembed status`, `pause`, `resume`, `stop` | The background capture: its state and number of screens; pause and resume (in every process, in the service's index folder); stop and remove |
| `streamembed index DIR [--interval S] [--keep-unchanged]` | Index a folder now and print the progress. Ctrl-C stops it; running it again reads only screenshots not read before (also ones added since). The index folder is left out of the scan |
| `streamembed capture [--interval S] [--monitors ...] [--keep-unchanged] [--model M] [--index-type T]` | Capture in this terminal until Ctrl-C |
| `streamembed search ["QUESTION"] [--top-k N] [--since DATE] [--until DATE] [--no-open]` | While the background capture runs: open its search page with the question in the search box, searched there as if typed (one search; no second model); `streamembed search` alone opens the page empty; with `--no-open`, search here and print the best matches. While the background capture is still loading its model, it waits for it (up to 150 s) rather than load a second one. With no background capture, the model loads here (20-60 s) and the page is written to `results.html` |
| `streamembed config [show \| set KEY VALUE ... \| reset [KEY ...] \| path]` | Show (the default), change or reset the settings, or print where the settings files are. After `set` or `reset`, a line says when `$STREAMEMBED_INDEX` or `$STREAMEMBED_MODELS` overrides the value |

Options go after the command: `streamembed search --index-dir DIR "QUESTION"`. Every command except `config` takes `--index-dir`. `streamembed COMMAND --help` lists all the options of a command, and `streamembed --version` prints the version. A mistake in a setting, a path or a date prints one line that starts with `streamembed: error:`; a mistake in an option prints the command's usage and an error naming the option. Both exit with status 2. `python -m streamembed` is the same command.

## Python

`examples/main.py` is a complete script.

| Function | What it does |
|---|---|
| `available_index_types()` | `["flat", "binary", "hnsw", "ivf", "pq"]` |
| `available_models()` | Model names: those in the model repo, then the built-in base models |
| `configure(index_dir=None, index_type=None, model=None, models_repo=None, device=None, width=None, capture_dir=None, capture_format=None, capture_width=None)` | Settings for this process only (see Settings), such as `configure(capture_width=0)` to save live captures at the screen's own size |
| `build_index_from_directory(directory, interval=None, skip_unchanged=None, progress=None) -> dict` | Index screenshots you already have |
| `build_index_from_now(interval=None, duration=None, block=False, monitors=None, skip_unchanged=None) -> Capture` | Capture the screen in the background |
| `query_index(query, top_k=None, since=None, until=None) -> Results` | The screens that best match a description; an empty question raises `ValueError` |
| `visualize_results(results, path=None, open_browser=True, context=None, query=None) -> Path` | Write the results page and open it in the browser; returns the page's path |
| `get_config()`, `set_config(**values)`, `reset_config(*names)` | The settings (below) |
| `pause()`, `resume()`, `is_paused()` | Pause capturing into the index, in every process (the background service too) |

Arguments left as None come from the settings. Screens that did not change are not stored again: a capture is encoded when more than 10 % of its kept blocks changed since the last stored one, or, when it changed less than that, once `heartbeat` seconds (300 by default) have passed since then.

### Indexing a folder

- Every `.jpg`, `.jpeg`, `.png` and `.webp` file in the folder and its subfolders is read.
- A screenshot's time comes from its file name (macOS screenshot names, Unix times, dates), else its EXIF data, else the file time.
- Each subfolder, each display number after the time in the name (`_1`, ` (2)`) and each image size is a screen of its own. A screen is its own stream with its own change gate, and screens are indexed one after another, each in time order.
- Per screen, at most one screenshot every `interval` seconds is used (0: all). Screenshots dated only by their file time are all used, since that time may be the time they were copied.
- Ctrl-C stops it. Running it again reads only the screenshots not read before, stored or not (`handled.jsonl` records those that were not), and also ones added since, whatever their time. When there is nothing new it returns before the model loads. The folder is recorded by its full path, so `shots` and `/Users/me/shots` are the same folder.
- The index folder is left out of the scan. Giving the index folder itself, or a folder inside it, raises `ValueError`.
- The index's write lock is taken while it indexes and released when it returns, so a Python session left open does not keep `streamembed start` from writing.
- A folder with no screenshots returns zero counts. A missing folder raises `FileNotFoundError`, a file `NotADirectoryError`, and a folder that cannot be read `PermissionError` (on macOS, allow it in System Settings > Privacy & Security > Files and Folders). A subfolder that cannot be read is named on stderr and left out.
- It returns the counts: `found` (files), `skipped` (left out by `interval`), `done` (read in earlier runs), `todo` (read in this run), `captured`, `encoded` (stored), `idle` (unchanged, not stored), `minor` (changed too little to store), `errors` (not readable as images, or not stored) and `stopped`. When it ends, `found` is the sum of `skipped`, `done`, `encoded`, `idle`, `minor` and `errors`. `progress`, if given, is called with these counts after each screenshot.

### Capturing the screen

- `monitors`: `"all"` (a display connected meanwhile is captured from then on), `"primary"` or display numbers (1 = primary). Each display is its own stream.
- At least 0.5 s passes between captures, also with `interval=0`. `duration` stops the capture after that many seconds. The index's write lock is held until the capture ends.
- A frame that cannot be stored (a full disk, for example) is counted in `stats["errors"]` and printed to stderr, which is `service.log` for the background service.
- It returns a `Capture`: `stop()`, `pause()`, `resume()`, `wait()` (until it stops, or Ctrl-C), `running`, `stalled`, and `stats` (`captured`, `encoded`, `idle`, `minor`, `skipped`, `errors`).
- A screen grab that has not returned for 60 s (seen on macOS, outside StreamEmbed) sets `stalled` and is printed to stderr. `wait()` then returns. `stop()` finishes a screenshot being encoded, gives up on a grab that has not returned after 30 s and returns `False`; nothing more is stored from that capture, and `build_index_from_now()` can start a new one. The background service exits in this case, and the service manager starts it again.
- While the index is paused (`streamembed pause`), capturing and folder indexing wait and say so once on stderr.
- A question does not wait for a frame being encoded: every 4 layers the frame lets a waiting question go first (75 ms median, 211 ms at most, while frames were encoded back to back on an M2 Pro).
- While macOS reports critical memory pressure, and while the model loads again, captures are stored but encoded later, in order (24 kept in memory, then read back from their files, at most 2,000).
- After 10 minutes with nothing captured (the screen locked), the background capture frees the model (5.8 GB to 0.7 GB) and loads it again, in the background, when capture resumes (about 6-20 s; captures wait meanwhile). A question while it is freed loads it.
- While a search page has focus, new captures are stored but encoded only after it has not (at most 24 wait; then they are encoded at once), so a search does not wait for a screen being encoded. They are encoded in order, and any left when the capture stops are encoded before it ends.

### Searching

`query_index` returns `Results`, a list of `Result`, best first. It also carries the question (`query`; `text` is the same) and the time window from `since`/`until` (`window`), and a slice of it keeps them:

| Field | Type | Content |
|---|---|---|
| `id` | int | Row in the index |
| `score` | float | Cosine similarity |
| `time` | str | Capture time, local, `YYYY-MM-DD HH:MM:SS` |
| `ts` | float | The same, in Unix seconds |
| `path` | str | The image file: the original screenshot for a folder screenshot, the file in `capture_dir` for a live capture |
| `source` | str or None | The original file of a folder screenshot (the same as `path`); `None` for a live capture |

`to_dict()` gives the fields as a dict.

- A match within 30 s of a better one from the same screen (the same subfolder and display) is dropped: a screen stays up for a while. Folder screenshots with no time in their file name are never dropped this way.
- `since` and `until` (Unix seconds, `"YYYY-MM-DD"` or `"YYYY-MM-DD HH:MM"`) limit the search to that time; the question is searched as written. A date alone as `until` includes that day. `since` at or after `until` raises `ValueError`.
- An empty index, or a time window with no screens in it, returns `[]` without loading the model.

### The results page

`visualize_results` writes an HTML page, `results.html` in the index folder unless `path` is given.

The background capture also serves this page with a search box, at `http://127.0.0.1:<port>/` with a token that `streamembed search` puts in the link (the port and token are in `web.json` in the index folder, owner-only). It answers only on this machine, only with the token, and serves images by index row, never by path. The page's language (English, 한국어, 中文) is chosen at the top right, or with `?lang=ko` in the link, and kept in the browser; dates and times follow it.

- It shows each screenshot from where it is: a folder screenshot from its original file, a live capture from `capture_dir`. Match thumbnails are made in `thumbs/` in the index folder. The page needs no server and works on this machine while those files are there; a moved or deleted original shows its path instead of an image.
- Its heading is `query` if given, else the question and the time window the results came with. A list you built yourself has no heading unless you give `query`.
- It shows the best 10, with Show more for the next 10. A card gives the day and time of its screen (the match score shows on hover). A match within 5 minutes of a better one from the same screen is left out: only the best of them is shown.
- A date chip (the 7 most recent days with results; the others behind a toggle) keeps the results from that day, and From/to keeps a range of days; both list only days with results, in the page's language. `--since`/`--until` search a range before the results are picked.
- Sort shows the results by best match (the default), newest first or oldest first; the choice is kept in the browser. Pass more results (`query_index(..., top_k=100)`, as `streamembed search` does) to have more to pick from.
- A card opens full size, with up to `context` frames of the same screen before and after it in time order. The match is marked with a dot and its score; a gap in the strip marks a pause, a new day, or frames left out. Arrow keys, or a swipe on a phone, step through the frames. Full size opens the image file.
- Cards open from the keyboard (Tab, then Enter).

## Settings

Every setting has a default in `defaults.toml` in the package.

- `streamembed config set KEY VALUE` (or `se.set_config(KEY=VALUE)`) changes a setting for good. It is kept in your own file (`streamembed config path` shows where), and a value that does not fit the setting is refused.
- `streamembed config reset KEY` goes back to the default; `streamembed config reset` resets every setting.
- `streamembed config` lists every setting and where its value comes from.
- The background capture reads the settings when it starts: run `streamembed start` again to apply a change. When `$STREAMEMBED_CONFIG` names your file, `streamembed start` passes that path on to the service.

A value given to a function or a command takes precedence for that call. Then come `configure()` (for its process), `$STREAMEMBED_INDEX` / `$STREAMEMBED_MODELS`, your file, and the defaults, in that order.

| Setting | Default | What it does |
|---|---|---|
| `index_dir` | `~/screenshot_index` | The index folder |
| `capture_dir` | `""` | Where live captures are written, as a full path (`~` is allowed); empty: `captures/` in the index folder. It can be your screenshots folder: indexing it later leaves the captures out. Screenshots indexed from a folder are never copied; the index keeps only vectors and each image's time and path |
| `capture_format` | `webp` | `webp` or `jpeg`. WebP is about half the size at the same quality (a 1280-wide screenshot: about 80 KB, JPEG 150-200 KB) |
| `capture_width` | 1280 | Width of a saved capture in pixels; 0 keeps the screen's own size (sharper when zoomed in, about 3.5 times larger). The model always sees the capture fitted to `width` |
| `capture_max_gb` | 20 | Live captures are kept under this many GB (images; 0: no limit). Past it, the oldest are removed with their screens in the index until they take 90 % of it; the capture checks when it starts and every hour. Screenshots indexed from a folder are never removed |
| `capture_keep_days` | 0 | Live captures older than this many days are removed with their screens in the index (0: kept however old). Checked with `capture_max_gb` |
| `index_type` | `flat` | `flat`, `binary`, `hnsw`, `ivf` or `pq` (see Index types) |
| `model`, `models_repo`, `device` | `qwen3-vl-2b`, `dayoon/streamembed`, `auto` | The model, where its LoRAs come from, and the device (`auto`, `mps`, `cuda`, `cuda:N`, `cpu`) |
| `width` | 1280 | Wider screenshots are shrunk to it; narrower ones are left as they are |
| `interval` | 5 | Seconds between captures, 0 to 86,400; live capture uses at least 0.5. For a folder, at most one screenshot per screen this often (0: all) |
| `monitors` | `all` | `all`, `primary`, or display numbers such as `[1, 2]` |
| `skip_unchanged` | true | Skip screens that did not change |
| `heartbeat` | 300 | A screen that changed a little (under 10 % of its blocks, such as a few typed characters) is stored at most once per this many seconds; one that changed more is stored at once |
| `exclude_apps`, `exclude_titles` | password managers; private browsing windows | Nothing is captured while the frontmost app or window title contains one of these (macOS). A value replaces the whole list, and `set` names the defaults it drops. To add an app and keep the defaults, list them all: `streamembed config set exclude_apps "1password,bitwarden,dashlane,lastpass,keychain access,keepass,signal"` |
| `top_k` | 10 | Results per question |
| `page_results`, `context` | 100, 15 | Results put on the results page (to pick dates among); frames shown before and after each |

## Capturing in the background

`build_index_from_now` captures while its Python process runs: `examples/main.py` keeps capturing until Ctrl-C or until the terminal closes. To keep capturing with no terminal open and after every login, install it as a service. It uses the system's own service manager:

```python
se.install_service(interval=5, monitors="all")   # starts now and at every login; restarted if it stops with an error
se.service_status()                              # {"installed": True, "running": True, "pid": ..., "log": ".../service.log", ...}
se.uninstall_service()                           # stops it and removes it; the index stays
```

| System | Service | Notes |
|---|---|---|
| macOS | launchd user agent `org.streamembed.capture` (`~/Library/LaunchAgents/`) | The first start asks for Screen Recording permission for this Python (System Settings > Privacy & Security > Screen Recording); allow it, then run `streamembed start` (or `install_service()`) again. With `--folder ~/Desktop`, this Python also needs access to the Desktop folder (Files and Folders); without it, `service.log` says so and the service captures without the folder |
| Linux | systemd user service `streamembed-capture.service` (`~/.config/systemd/user/`) | X11 only. Install from a terminal in your desktop session: the service keeps that session's `DISPLAY` |
| Windows | Task Scheduler task `StreamEmbedCapture`, at logon, run with `pythonw` (no console window) | Stopping it ends the process at once; the index drops a half-written row when it is next opened |

| Function | What it does |
|---|---|
| `install_service(interval=None, monitors=None, skip_unchanged=None, index_dir=None, index_type=None, model=None, models_repo=None, device=None, width=None, directory=None)` | Run `build_index_from_now` as a service (`streamembed start` does this); with `directory` (it must exist), index those screenshots first. Settings left out are read from the settings when the service starts. Installing again replaces it |
| `service_status()` | Whether it is installed, running and paused, its pid, its index folder, its log file (`service.log` in the index folder) and its command line |
| `uninstall_service()` | Stop it and remove it |

One process writes an index at a time. While the service runs, other processes open its index read-only: `query_index` and `visualize_results` work and see newly captured screens, and `build_index_*` on that folder raises until the service is stopped. The other way round, while a Python session is indexing or capturing into the folder, the service writes a line saying so to `service.log`, and the service manager starts it again a minute later. A `directory` that is the index folder or inside it is refused before the service is installed.

## The index folder

| File | Content |
|---|---|
| `index.json` | The encoder name and the vector size; the index refuses vectors from another encoder |
| `frames.jsonl` | One line per stored screen: its time (`ts`) and the path of its image, as `source` for a folder screenshot and as `path` for a live capture in `capture_dir` (relative when it is inside the index folder) |
| `vectors.f16` | One fp16 vector (2048-d for 2B, 4096-d for 8B) per line of `frames.jsonl` |
| `captures/YYYY-MM-DD/` | Live captures (unless `capture_dir` puts them elsewhere), in `capture_format` at `capture_width`, named `<ms>.webp` (primary display) and `<ms>_1.webp`, `<ms>_2.webp` (other displays: display 2, as in `--monitors 1,2`, is stored as `_1`, display 3 as `_2`), with `-<n>` added when the name is taken. Screenshots indexed from a folder are not copied here; the results page shows the originals, so moving or deleting one leaves its result without an image |
| `paused` | Present while capture is paused |
| `seen.jsonl` | When an unchanged screen was still on screen |
| `handled.jsonl` | Folder screenshots that were read but not stored (unchanged, changed too little, or not readable), so that indexing the folder again leaves them out |
| `writer.lock` | Held by the process that writes; others read |
| `query.sock` (`query.port` on Windows) | While the background capture runs: where searches from other processes send their question (owner-only; a token on Windows) |
| `web.json` | While the background capture runs: the address and token of its search page |
| `service.log` | The background service's log |
| `codes.bin`, `mu.npy` | `binary` only |
| `hnsw.bin`, `hnsw.json` | `hnsw` only: the graph with its 512-d float32 vectors, and its parameters |
| `ivf.json`, `ivf_centroids.npy`, `ivf_codes.bin` | `ivf` only: training size, list centroids, the list of each frame (4 B) |
| `pq.json`, `pq_centroids.npy`, `pq_codebooks.npy`, `pq_codes.bin` | `pq` only: training size, list centroids, PQ codebooks, the list and the 64 PQ codes of each frame (68 B) |
| `thumbs/`, `results.html` | Made by `visualize_results` |

A process takes `writer.lock` when it first writes (indexing or capturing), not when it only searches, and releases it when the indexing or the capture ends. A line cut short by a crash is left out when the folder is next opened, and a vector written just before a crash without its line is dropped. If `vectors.f16` holds more than one vector beyond the lines of `frames.jsonl` (an older `frames.jsonl` restored, for example), the extra vectors are moved to `vectors.f16.orphan-<time>` instead, and stderr says so.

## Index types

| Type | Search |
|---|---|
| `flat` | Exact inner product over every stored vector (fp16, 4 KB per frame) |
| `binary` | Exact up to 1,000 frames. Beyond that, Hamming top 1,000 over 1-bit codes of the first 512 dimensions (64 B per frame, kept in memory), then exact re-rank. About 0.13 GB of memory per year of 5-s captures |
| `hnsw` | Exact up to 10,000 frames. Beyond that, an HNSW graph (hnswlib, M = 32, efConstruction = 500) over the first 512 dimensions; a query takes the graph's top 800 (efSearch) and re-ranks them exactly. About 2.3 KB per frame in memory and on disk. Needs `pip install "streamembed[hnsw]"` |
| `ivf` | Exact up to 10,000 frames. Beyond that, the frames are split into round(sqrt(n)) lists by spherical k-means (10 iterations, fixed seed, on up to 50,000 evenly spaced frames); a query scores every frame of its 32 nearest lists exactly. 4 B per frame plus the centroids |
| `pq` | IVF-PQ. Exact up to 10,000 frames. Beyond that, the same kind of lists, plus a product-quantized code of the first 512 dimensions per frame (64 sub-quantizers x 8 bits, trained on up to 20,000 frames); a query ranks the frames of its 64 nearest lists by their codes and re-ranks the top 2,000 exactly. 68 B per frame plus the centroids and codebooks |

Every type returns exact cosines: candidates are re-scored over `vectors.f16`. New frames are searchable as soon as they are added: `hnsw` inserts them into the graph, `ivf` and `pq` put them in their nearest list. `ivf` and `pq` train when the index passes 10,000 frames and train again from scratch (new lists and codebooks, every frame re-encoded, a few seconds) each time the index has grown 4x since the last training. The `hnsw` graph is never rebuilt; it is saved every 1,000 new frames and when the index is closed. If a structure lags the vectors (a crash, or frames added while the folder was open with another type), the missing frames are added when the index is opened. A process that opens the folder read-only uses these files but never writes them; if they are missing or behind, it builds or catches up in memory. Defaults are constants in `src/streamembed/ann.py`.

Measured on an M2 Pro (16 GB) with 200,000 2048-d frames (132,221 stored Qwen3-VL-Embedding-2B frame vectors plus noisy copies of them) and 200 stored text-query vectors, top 10 with the 30-s dedupe: the share of `flat`'s top 10 that each type also returns is 0.93 for `binary`, 0.95 for `hnsw`, 0.97 for `ivf` and 0.96 for `pq`; median query time is 295 ms for `flat`, 8 ms for `binary`, 13 ms for `hnsw`, 53 ms for `ivf` and 22 ms for `pq`. Building from `vectors.f16` takes 1 s (`binary`), 61 s (`hnsw`), 5 s (`ivf`) and 19 s (`pq`). At 50,000 frames recall is 0.91, 0.99, 0.95 and 0.97.

An index folder can be reopened with any other type. It also records the encoder, and refuses vectors from a different one.

## Models

A model is a base model from its own Hugging Face repository, pinned to a commit, plus two LoRAs from the StreamEmbed model repository: one for screenshots (a consistency adapter that keeps pruned, streamed frames close to full frames) and one for questions. Only the LoRAs live in the StreamEmbed repository; its `models.json` names each model's base repository, commit and files.

| Model | What it is |
|---|---|
| `qwen3-vl-2b` (default) | Qwen3-VL-Embedding-2B with the StreamEmbed LoRAs |
| `qwen3-vl-8b` | Qwen3-VL-Embedding-8B with the StreamEmbed LoRAs; needs a GPU with about 20 GB |
| `qwen3-vl-2b-base` | Qwen3-VL-Embedding-2B without LoRAs; built in, needs no model repo |

The model repository also holds LoRAs for SigLIP2 So400m, SigLIP2 Base and VultronRetrieverFlash 0.8B. `available_models()` lists them, but the stream encoder supports Qwen3-VL models only: `set_config` and `streamembed config set` refuse them, and `configure(model=...)` with one raises `NotImplementedError` when StreamEmbed first encodes.

`configure(models_repo=...)` or `$STREAMEMBED_MODELS` points to another Hugging Face repository or a local folder with the same layout. An index records its model and refuses vectors from another one.

The first run of `qwen3-vl-2b` downloads the base model (about 4 GB) and the LoRAs (70 MB each); after that nothing is downloaded. Encoding needs about 8 GB of memory and takes 0.7 to 1.3 s per 1280-px frame on an M2 Pro GPU that nothing else uses; while another process (the background capture, for example) encodes on the same GPU, expect about twice that. Loading the model takes about 20 to 60 s (stderr says when it loads, and says it downloads only when the model is not cached yet), so `build_index_from_now()`, `streamembed index` and the first search in a process wait that long before they start. The change gate skips screens that did not change.
