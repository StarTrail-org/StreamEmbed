"""Seconds per screenshot of every row, as the paper measures it (run it on an idle Mac), then evaluate.py's latency table.

    python latency.py <data> [--cpu] [--out results]

Each measurement runs in its own process: 5 warm-up screenshots, then every recording of latency_sample.json (its first 50
screenshots) as a fresh stream, 0.1 s apart. Each screenshot is timed from reading the image to its vector; the mean is reported.
LoRAs do not change the computation, so the base weights are timed. The text index (ocr) is timed as Apple Vision OCR plus the text
embedding, on the GPU only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

PRESETS = ("ocr", "full", "full+pe", "S", "S+G", "S+G+T+L")


def measure(data: str, config: str, preset: str, device: str, out: Path) -> None:
    import encode as E
    import numpy as np
    root = Path(data) / config
    sample = json.load(open(root / "latency_sample.json"))
    image = lambda f: str(root / "images" / f)  # noqa: E731
    if device == "cpu":
        import torch
        torch.set_num_threads(8)
    enc = E.encoder("base", device)
    enc._trim = lambda: None        # the service frees GPU memory after every screenshot; the paper did not
    if preset == "ocr":
        import ocr as O
        step = lambda f, stream: E.embed_text(enc, O.text(image(f)))  # noqa: E731
    else:
        E.preset(enc, preset)
        step = lambda f, stream: enc.encode_frame(image(f), stream=stream)  # noqa: E731
    for f in sample["warmup"]:
        step(f, "warmup")
    times = []
    for k, rec in enumerate(sample["recordings"]):
        enc.reset_stream()
        for f in rec:
            t = time.perf_counter()
            step(f, k)
            times.append(time.perf_counter() - t)
            time.sleep(0.1)
    json.dump(dict(mean=float(np.mean(times)), screenshots=len(times)), open(out / f"latency_{config}_{device}_{preset}.json", "w"))
    print(f"{config} {preset} {device}: {np.mean(times):.2f} s per screenshot")


def main() -> int:
    ap = argparse.ArgumentParser(description="Seconds per screenshot of every row.")
    ap.add_argument("data")
    ap.add_argument("--cpu", action="store_true", help="also time the CPU rows")
    ap.add_argument("--out", default="results")
    ap.add_argument("--one", nargs=3, metavar=("CONFIG", "PRESET", "DEVICE"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.one:
        measure(a.data, *a.one, out)
        return 0
    for c in D.SEGMENTS:
        for p in PRESETS:
            for dev in ("mps", "cpu") if a.cpu else ("mps",):
                if dev == "cpu" and p in ("full", "ocr"):     # the CPU rows start from full+pe (the same vectors); OCR has no CPU time
                    continue
                if not (out / f"latency_{c}_{dev}_{p}.json").is_file():
                    subprocess.run([sys.executable, __file__, a.data, "--out", a.out, "--one", c, p, dev], check=True)
    return subprocess.run([sys.executable, str(HERE / "evaluate.py"), a.data, "--out", a.out]).returncode


if __name__ == "__main__":
    sys.exit(main())
