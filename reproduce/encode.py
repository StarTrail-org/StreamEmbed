"""Encode ScreenMemEval with streamembed (accuracy.py runs it for every row).

    python encode.py questions <data> <config> <base|lora>
    python encode.py frames <data> <config> <arm> <presets> <stream>
    python encode.py texts <data> <config>          the text index: OCR texts (ocr.py) and questions, with the OCR LoRA

arm: the weights for screenshots. base (none), lora (the fine-tuned LoRA), adapter (the Consistency adapter). Questions use lora,
or base for the zero-shot row. presets (comma-separated): full, full+pe, S (spatial pruning), S+G (+ Grid Keep), S+G+T+L (+ temporal
reuse). Each preset is its own stream, started fresh at the first frame of the stream. A rerun continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402

PART = 1000         # frames per saved part
REPO = os.environ.get("STREAMEMBED_MODELS", "dayoon/streamembed")


def encoder(arm: str, device: str = "auto"):
    """arm base | lora | adapter, or ocr: the OCR LoRA for both texts and questions."""
    from streamembed.encoder import Encoder
    from streamembed.engine import pause
    from streamembed.models import resolve
    spec = resolve("qwen3-vl-2b", REPO)
    adapter, lora = spec.weights
    if arm == "ocr":
        from huggingface_hub import hf_hub_download
        name = "qwen3-vl-2b/ocr_lora.safetensors"
        lora = adapter = Path(REPO) / name if Path(REPO).is_dir() else hf_hub_download(REPO, name)
    weights = {"base": None, "lora": (lora, lora), "adapter": (adapter, lora), "ocr": (lora, lora)}[arm]
    enc = Encoder(weights, device, spec.base, spec.revision, (spec.scale, spec.scale), f"{spec.encoder_name}:{arm}")
    pause.set_hook(None)        # the service lets questions run between a frame's layers; off for measurements
    return enc


def preset(enc, name: str):
    from streamembed.engine.config import PRESETS
    enc.cfg = PRESETS[name]
    return enc.cfg


def frames(a) -> None:
    b = D.load(a.data, a.config)
    items = b.segment(a.stream)
    presets = a.presets.split(",")
    out = Path(a.out) / a.config / a.arm / f"seg{a.stream}"
    out.mkdir(parents=True, exist_ok=True)
    parts = sorted(out.glob("part_*.npz"))
    done = sum(len(np.load(p)["fids"]) for p in parts)
    if done >= len(items):
        return
    enc = encoder(a.arm, a.device)
    replay = [p for p in presets if preset(enc, p).temporal_vit]     # temporal reuse: re-encode the frames before `done` for its cache
    vecs, fids = {p: [] for p in presets}, []
    for k, f in enumerate(items):
        for p in presets if k >= done else replay:
            preset(enc, p)
            v, _ = enc.encode_frame(str(b.image(f)), stream=p)
            if k >= done:
                vecs[p].append(np.asarray(v, np.float16))
        if k < done:
            continue
        fids.append(f["fid"])
        if len(fids) == PART or k == len(items) - 1:
            np.savez(out / "writing.npz", fids=np.array(fids), **{f"{p}_emb": np.stack(vecs[p]) for p in presets})
            os.replace(out / "writing.npz", out / f"part_{len(parts):04d}.npz")
            parts.append(None)
            print(f"{a.config} {a.arm} stream {a.stream}: {k + 1}/{len(items)}", flush=True)
            vecs, fids = {p: [] for p in presets}, []


def embed_text(enc, text: str) -> np.ndarray:
    """A document as the paper embeds OCR text: the model's default instruction, at most 12,000 characters."""
    import torch
    m = enc.m
    m.core.rope_deltas = None
    with torch.inference_mode():
        e = m.embedder.embed([m.embedder.format_model_input(text=(text or "(no text)")[:12000])], normalize=False)
    m.core.rope_deltas = None
    v = e[0].float().cpu().numpy()
    return v / np.linalg.norm(v)


def texts(a) -> None:
    out = Path(a.out) / a.config / "ocr"
    out.mkdir(parents=True, exist_ok=True)
    b = D.load(a.data, a.config)
    ocr = {}
    for line in open(Path(a.data) / a.config / "ocr.jsonl"):
        r = json.loads(line)
        ocr[r["fid"]] = r["text"]
    parts = sorted(out.glob("part_*.npz"))
    done = sum(len(np.load(p)["fids"]) for p in parts)
    todo = b.frames[done:]
    if not todo and (out / "questions.npy").is_file():
        return
    enc = encoder("ocr", a.device)
    for s in range(0, len(todo), PART):
        fids = [f["fid"] for f in todo[s:s + PART]]
        np.savez(out / "writing.npz", fids=np.array(fids), emb=np.stack([embed_text(enc, ocr[f]) for f in fids]).astype(np.float16))
        os.replace(out / "writing.npz", out / f"part_{len(parts):04d}.npz")
        parts.append(None)
        print(f"{a.config} OCR texts: {done + s + len(fids)}/{len(b.frames)}", flush=True)
    np.save(out / "questions.npy", np.stack([np.asarray(enc.encode_query(q["question"]), np.float32) for q in b.questions]))


def questions(a) -> None:
    out = Path(a.out) / a.config
    if (out / f"questions_{a.arm}.npy").is_file():
        return
    b = D.load(a.data, a.config)
    enc = encoder(a.arm, a.device)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"questions_{a.arm}.npy", np.stack([np.asarray(enc.encode_query(q["question"]), np.float32) for q in b.questions]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Encode ScreenMemEval with streamembed.")
    sub = ap.add_subparsers(dest="what", required=True)
    q = sub.add_parser("questions")
    q.add_argument("data"), q.add_argument("config", choices=D.SEGMENTS), q.add_argument("arm", choices=("base", "lora"))
    f = sub.add_parser("frames")
    f.add_argument("data"), f.add_argument("config", choices=D.SEGMENTS), f.add_argument("arm", choices=("base", "lora", "adapter"))
    f.add_argument("presets"), f.add_argument("stream", type=int)
    t = sub.add_parser("texts")
    t.add_argument("data"), t.add_argument("config", choices=D.SEGMENTS)
    for p in (q, f, t):
        p.add_argument("--out", default="vectors"), p.add_argument("--device", default="auto")
    a = ap.parse_args()
    {"frames": frames, "questions": questions, "texts": texts}[a.what](a)
