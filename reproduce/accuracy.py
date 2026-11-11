"""Recall@10 of every row: encode the questions and every stream of every row (one stream per GPU at a time), then evaluate.py.
A rerun continues where it stopped.

    python accuracy.py <data> [--device cuda|mps|cpu] [--vectors vectors] [--out results]
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

PASSES = [("base", "full"), ("lora", "full,S,S+G,S+G+T+L"), ("adapter", "S+G+T+L")]     # (arm, presets encoded together)


def main() -> int:
    ap = argparse.ArgumentParser(description="Recall@10 of every row on ScreenMemEval.")
    ap.add_argument("data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vectors", default="vectors")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    encode = [sys.executable, str(HERE / "encode.py")]
    common = ["--out", a.vectors, "--device", a.device]
    configs = [c for c in D.SEGMENTS if (Path(a.data) / c / "frames.jsonl").is_file()]
    for c in configs:
        for arm in ("base", "lora"):
            subprocess.run(encode + ["questions", a.data, c, arm] + common, check=True)
    size = {c: len(D.load(a.data, c).frames) / D.SEGMENTS[c] for c in configs}
    jobs: queue.Queue = queue.Queue()
    for c in configs:
        if (Path(a.data) / c / "ocr.jsonl").is_file():       # the text index, once ocr.py has run
            jobs.put(["texts", a.data, c])
    for _, c, arm, presets, s in sorted(((-size[c] * len(p.split(",")), c, arm, p, s) for c in configs for arm, p in PASSES
                                         for s in range(D.SEGMENTS[c]))):       # the longest first
        jobs.put(["frames", a.data, c, arm, presets, str(s)])
    failed = []

    def worker(gpu: int) -> None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)) if a.device == "cuda" else None
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            if subprocess.run(encode + job + common, env=env).returncode:
                failed.append(" ".join(job[2:]))

    n = 1
    if a.device == "cuda":
        import torch
        n = torch.cuda.device_count()
    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        print("failed (rerun to continue):", ", ".join(failed))
        return 1
    return subprocess.run([sys.executable, str(HERE / "evaluate.py"), a.data, "--vectors", a.vectors, "--out", a.out]).returncode


if __name__ == "__main__":
    sys.exit(main())
