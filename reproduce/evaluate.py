"""Score the vectors encode.py wrote and the times latency.py wrote, next to the paper's numbers (paper_numbers.json).

    python evaluate.py <data> [--vectors vectors] [--out results]

The rank of a group of recordings is 1 + the number of screenshots that score higher than its best screenshot (cosine). A single
question counts for Recall@10 if one of its groups ranks in the top 10, and a multi question counts the share of its groups that do.
Recall@10 is the mean over all questions, and nDCG@10 is over groups. Writes accuracy.md, accuracy.json (with every question's ranks)
and latency.md to --out. A row is scored once all its streams are encoded.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def ranks(score, b) -> dict:
    """score(s, e): the scores of questions s..e-1 against every screenshot (in order). Returns {qid: [rank of each group]}."""
    rows: dict = {}
    for i, f in enumerate(b.frames):
        rows.setdefault(f["recording"], []).append(i)
    out = {}
    for s in range(0, len(b.questions), 256):
        S = score(s, min(s + 256, len(b.questions)))
        for j, q in enumerate(b.questions[s:s + 256]):
            out[q["qid"]] = [int((S[j] > S[j][[i for u in g for i in rows[u]]].max()).sum()) + 1 for g in q["groups"]]
    return out


def dense(qv, X):
    qv, X = unit(qv), unit(X)
    return lambda s, e: qv[s:e] @ X.T


def text_index(qv, X, data, b):
    """The paper's text index: dense scores and BM25 scores of the OCR texts, fused by RRF (k 10, BM25 weight 0.5). Each list ranks
    every screenshot (ties share the best rank); a screenshot without a query term is not in the BM25 list."""
    import re
    import bm25s
    tok = lambda t: re.findall(r"[a-z0-9]+", (t or "").lower())  # noqa: E731
    ocr = {}
    for line in open(Path(data) / b.name / "ocr.jsonl"):
        r = json.loads(line)
        ocr[r["fid"]] = r["text"]
    bm = bm25s.BM25()
    bm.index([tok(ocr[f["fid"]]) for f in b.frames], show_progress=False)
    d = dense(qv, X)

    def rank(S):
        srt = np.sort(S, axis=1)
        return np.stack([S.shape[1] - np.searchsorted(srt[i], S[i], side="right") + 1 for i in range(len(S))])

    def score(s, e):
        B = np.stack([bm.get_scores(tok(q["question"])) for q in b.questions[s:e]])
        return 1 / (10 + rank(d(s, e))) + 0.5 * (B > 0) / (10 + rank(B))
    return score


def metrics(r: dict, b) -> dict:
    single, multi = [], []
    for q in b.questions:
        x = [min(r[q["qid"]])] if q["kind"] == "single" else r[q["qid"]]
        rec = sum(y <= 10 for y in x) / len(x)
        ndcg = sum(1 / math.log2(y + 1) for y in x if y <= 10) / sum(1 / math.log2(i + 2) for i in range(min(len(x), 10)))
        (single if q["kind"] == "single" else multi).append((rec, ndcg))
    mean = lambda z: 100 * sum(z) / len(z)  # noqa: E731
    return dict(R10=mean([x[0] for x in single + multi]), single_R10=mean([x[0] for x in single]),
                multi_R10=mean([x[0] for x in multi]), nDCG=mean([x[1] for x in single + multi]))


def vectors(vdir: Path, b, arm: str, preset: str):
    """One row per screenshot, or None until every stream is encoded. arm ocr: the OCR text vectors."""
    at = {f["fid"]: k for k, f in enumerate(b.frames)}
    X, seen = None, 0
    key = "emb" if arm == "ocr" else f"{preset}_emb"
    for p in sorted((vdir / b.name).glob("ocr/part_*.npz" if arm == "ocr" else f"{arm}/seg*/part_*.npz")):
        z = np.load(p)
        if X is None:
            X = np.zeros((len(b.frames), z[key].shape[1]), np.float32)
        X[[at[int(f)] for f in z["fids"]]] = z[key]
        seen += len(z["fids"])
    return X if seen == len(b.frames) else None


def table(title: str, configs: list, rows: list) -> str:
    return "\n".join([f"| {title} | " + " | ".join(f"{c} (paper)" for c in configs) + " |", "|---|" + "---|" * len(configs)]
                     + [f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows])


def main() -> int:
    ap = argparse.ArgumentParser(description="Score ScreenMemEval vectors and times next to the paper's.")
    ap.add_argument("data")
    ap.add_argument("--vectors", default="vectors")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    vdir, out = Path(a.vectors), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paper = json.load(open(HERE / "paper_numbers.json"))
    configs = paper["configs"]
    benches = {c: D.load(a.data, c) for c in configs if (Path(a.data) / c / "frames.jsonl").is_file()}
    res, acc = [], []
    for row in paper["rows"]:
        if "R10" not in row:
            continue
        r, cells = dict(row=row["row"]), []
        for j, c in enumerate(configs):
            b = benches.get(c)
            qf = vdir / c / {"base": "questions_base.npy", "ocr": "ocr/questions.npy"}.get(row["arm"], "questions_lora.npy")
            X = vectors(vdir, b, row["arm"], row["preset"]) if b and qf.is_file() else None
            if X is None:
                cells.append("missing")
                continue
            rk = ranks(text_index(np.load(qf), X, a.data, b) if row["arm"] == "ocr" else dense(np.load(qf), X), b)
            r[c] = dict(metrics(rk, b), ranks=rk)
            cells.append(f"{r[c]['R10']:.2f} ({row['R10'][j]:.2f})")
        res.append(r)
        acc.append((row["row"], cells))
    lat = []
    for dev, key in (("mps", "latency_gpu"), ("cpu", "latency_cpu")):
        rows = []
        for row in paper["rows"]:
            files = [out / f"latency_{c}_{dev}_{row['preset']}.json" for c in configs]
            if key in row and any(f.is_file() for f in files):
                rows.append((row["row"], [f"{json.load(open(f))['mean']:.2f} ({row[key][j]:.2f})" if f.is_file() else "missing"
                                          for j, f in enumerate(files)]))
        if rows:
            lat.append(table(f"{'GPU' if dev == 'mps' else 'CPU'}, s per screenshot", configs, rows))
    if any(c != "missing" for _, cells in acc for c in cells):
        json.dump(res, open(out / "accuracy.json", "w"))
        (out / "accuracy.md").write_text(table("Recall@10", configs, acc) + "\n")
        print(table("Recall@10", configs, acc) + "\n")
    if lat:
        (out / "latency.md").write_text("\n\n".join(lat) + "\n")
        print("\n\n".join(lat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
