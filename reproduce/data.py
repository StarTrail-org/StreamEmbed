"""Download ScreenMemEval from Hugging Face and read it.

    python data.py <data> [sme-h1 sme-h2 sme-agent]      download and unpack the images (about 18 GB on disk)
"""
from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

REPO = "dayoon/ScreenMemEval"
SEGMENTS = {"sme-h1": 8, "sme-h2": 4, "sme-agent": 4}   # the paper encodes each screen history as this many streams


@dataclass
class Bench:
    name: str
    root: Path
    frames: list            # frames.jsonl, in order
    questions: list         # questions.jsonl

    def image(self, frame: dict) -> Path:
        return self.root / self.name / "images" / frame["file"]

    def segment(self, i: int) -> list:
        n, k = len(self.frames), SEGMENTS[self.name]
        return self.frames[n * i // k: n * (i + 1) // k]


def load(root, name: str) -> Bench:
    rows = lambda f: [json.loads(line) for line in open(Path(root) / name / f)]  # noqa: E731
    return Bench(name, Path(root), rows("frames.jsonl"), rows("questions.jsonl"))


def download(root, names=tuple(SEGMENTS)) -> None:
    from huggingface_hub import snapshot_download
    root = Path(root)
    done = [p.with_suffix(".tar").relative_to(root).as_posix() for p in root.glob("*/frames/*.unpacked")]
    snapshot_download(REPO, repo_type="dataset", local_dir=root, ignore_patterns=done or None,
                      allow_patterns=[f"{n}/*" for n in names])
    for n in names:
        for shard in sorted((root / n / "frames").glob("*.tar")):
            with tarfile.open(shard) as tf:
                tf.extractall(root / n / "images", filter="data")
            shard.with_suffix(".unpacked").touch()
            shard.unlink()                          # keep only the unpacked images
        print(f"{n}: {sum(1 for _ in (root / n / 'images').rglob('*.jpg'))} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download ScreenMemEval.")
    ap.add_argument("data")
    ap.add_argument("configs", nargs="*", default=list(SEGMENTS))
    a = ap.parse_args()
    download(a.data, a.configs)
