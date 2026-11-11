"""Apple Vision OCR of every screenshot, for the text index (macOS only; uv pip install pyobjc-framework-Vision).

    python ocr.py <data> [--workers 4]

Accurate recognition with language correction, lines in reading order, as in the paper. Writes <data>/<config>/ocr.jsonl, one
{"fid", "text"} per screenshot; a rerun continues where it stopped. accuracy.py embeds these texts, and latency.py times OCR itself.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402


def text(path: str) -> str:
    import objc
    import Vision
    from Foundation import NSURL
    from PIL import Image
    W, H = Image.open(path).size
    with objc.autorelease_pool():
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(0)                         # accurate
        req.setUsesLanguageCorrection_(True)
        Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(path), {}).performRequests_error_([req], None)
        lines = []
        for o in req.results() or []:
            cands = o.topCandidates_(1)
            if cands:
                bb = o.boundingBox()                        # normalised, origin bottom-left
                lines.append(((1 - bb.origin.y - bb.size.height) * H, bb.origin.x * W, str(cands[0].string())))
    lines.sort(key=lambda r: (round(r[0] / 12), r[1]))      # rows of 12 px, then left to right
    return "\n".join(t for _, _, t in lines if t.strip())


def job(item: tuple) -> str:
    fid, path = item
    return json.dumps({"fid": fid, "text": text(path)}, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Apple Vision OCR of ScreenMemEval.")
    ap.add_argument("data")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for c in D.SEGMENTS:
            if not (Path(a.data) / c / "frames.jsonl").is_file():
                continue
            b = D.load(a.data, c)
            out = Path(a.data) / c / "ocr.jsonl"
            lines = out.read_text().splitlines(keepends=True) if out.is_file() else []
            if lines and not lines[-1].endswith("\n"):          # an interrupted write: drop the partial line
                lines.pop()
                out.write_text("".join(lines))
            done = {json.loads(line)["fid"] for line in lines}
            todo = [(f["fid"], str(b.image(f))) for f in b.frames if f["fid"] not in done]
            with open(out, "a") as fh:
                for k, line in enumerate(pool.imap(job, todo, chunksize=8)):
                    fh.write(line + "\n")
                    fh.flush()
                    if (k + 1) % 1000 == 0 or k + 1 == len(todo):
                        print(f"{c}: {len(done) + k + 1}/{len(b.frames)}", flush=True)
