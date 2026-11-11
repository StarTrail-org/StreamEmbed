"""One model for frames and questions.

Frames are encoded with the consistency adapter and questions with the question LoRA. Both are the same base model plus a low-rank update
of the same LLM attention layers. The adapter is merged into the weights (W + B @ A in fp32, cast back, as `merge_lora` does). For a
question, each of those layers adds, in fp32, the question LoRA's low-rank term minus the adapter's, so its output is the base layer plus
the question LoRA, without a second copy of the weights (0.7 GB for 2B). The ViT has no question path: a text question never runs it.
"""
from __future__ import annotations

import ctypes
import gc
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .engine.config import MODEL_ID, PRESETS, QUERY_INSTRUCTION
from .engine import pause
from .engine.engine import Engine, preprocess
from .engine.model import Model, _lora_pairs, pick_device
from .engine.qwen_embedder import Qwen3VLEmbedder

METHOD = "S+G+T+L"      # StreamEmbed: spatial pruning with Grid Keep + temporal reuse (ViT and LLM)
QUESTION_WAIT_S = 30.0
# Questions of a few lengths, run once after the model loads: Metal builds its kernels for a shape the first time it runs, so the
# first questions after a start took seconds (5.4 s, then 2.1 s, measured) before settling at 0.3 s.
WARM_UP = ("a terminal", "a chat app with a photo", "the flight search page with the prices to Denver in May",
           "the slide in yesterday's meeting with a pink and blue chart that compared the cost of the three plans")  # a frame waits at most this long for questions ahead of it (a stream of them cannot stop the capture)


def _updated(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return (w.data.float() + scale * (b.float() @ a.float())).to(w.dtype)


def _give_back() -> None:
    """Return the allocator's free memory to the system (macOS: malloc_zone_pressure_relief; Linux: malloc_trim)."""
    try:
        if sys.platform == "darwin":
            ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
        elif sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_size_t(0))
    except Exception:
        pass


class Encoder:
    def __init__(self, weights: tuple[Path, Path] | None, device: str = "auto", model_id: str = MODEL_ID, revision: str | None = None,
                 scales: tuple[float, float] = (1.0, 1.0), name: str | None = None):
        """weights: (frame adapter, question LoRA) PEFT state dicts, or None for the base model. revision: the base model commit.
        scales: LoRA alpha / rank of each. The model loads now; unload() frees it, and the next frame or question loads it again."""
        self._spec = (weights, device, model_id, revision, scales)
        self.cfg = PRESETS[METHOD]
        self.m = None
        self.engines: dict = {}         # stream (one per screen) -> Engine with that stream's temporal reuse state
        self.lock = threading.Lock()    # one forward at a time
        self._load()
        # Questions go before frames: a frame waits while a question is waiting, so a search waits for at most the one frame being
        # encoded (with two displays, frames come in pairs, and the lock alone would let the second go first).
        self._questions = threading.Condition()
        self._asking = 0
        self.name = name or f"{Path(model_id).name}:{weights[0].name if weights else 'zero-shot'}:{METHOD}"
        self._trim()                    # give back what loading and merging left in the Metal cache
        self._frame_thread = None
        pause.set_hook(self._let_questions_in)   # a frame being encoded lets a waiting question go first, between its layers

    def _load(self) -> None:
        weights, device, model_id, revision, scales = self._spec
        device = pick_device(device)
        dtype = torch.float32 if device == "cpu" else torch.float16
        try:
            emb = Qwen3VLEmbedder(model_id, dtype=dtype, local_files_only=True, revision=revision)     # once cached, never touch the network
        except OSError:
            emb = Qwen3VLEmbedder(model_id, dtype=dtype, revision=revision)                            # first run: download
        question = self._merge(emb.model, *weights, *scales) if weights is not None else []
        emb.model.to(device).eval()
        self._asking_now = False         # True for the length of a question's forward: the hooks add the question LoRA
        self._hooks = []
        for mod, terms in question:
            terms = [(a.float().to(device), b.float().to(device), sc) for a, b, sc in terms]
            self._hooks.append(mod.register_forward_hook(self._question_hook(terms)))
        del question
        self.m = Model(embedder=emb, device=torch.device(device), dtype=dtype)
        self.dim = self.m.tcfg.hidden_size     # vector size: 2048 (2B), 4096 (8B)
        _give_back()                    # the CPU copies loading made: freed, but kept by the allocator (and then swapped) until now

    @property
    def loaded(self) -> bool:
        return self.m is not None

    def warm_up(self) -> None:
        """Run a few questions of different lengths so the first real ones do not wait for Metal to build its kernels."""
        for text in WARM_UP:
            self.encode_query(text)

    def ensure_loaded(self) -> None:
        """Load the model if unload() freed it (about 20 s), and warm it up."""
        with self.lock:
            if self.m is not None:
                return
            print("streamembed: loading the model again", file=sys.stderr, flush=True)
            self._load()
        self.warm_up()

    def unload(self) -> None:
        """Free the model (about 7 GB for 2B); the next frame or question loads it again. Temporal reuse starts over."""
        with self.lock:
            if self.m is None:
                return
            for h in self._hooks:
                h.remove()
            device = self.m.device
            self._hooks, self.engines, self.m = [], {}, None
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            _give_back()

    @staticmethod
    def _merge(model: torch.nn.Module, adapter: Path, query: Path, frame_scale: float = 1.0,
               query_scale: float = 1.0) -> list[tuple[torch.nn.Module, list]]:
        """Merge the adapter into the model. Returns, for every LLM layer either LoRA touches, (module, [(A, B, scale), ...]): the
        low-rank terms that turn the merged layer into the base layer plus the question LoRA (the question's, and the adapter's with
        the opposite sign)."""
        frame_pairs = {name: (a, b) for name, a, b in _lora_pairs(str(adapter))}
        query_pairs = {name: (a, b) for name, a, b in _lora_pairs(str(query)) if "language_model" in name}
        out = []
        for name in sorted(frame_pairs.keys() | query_pairs.keys()):
            mod = model.get_submodule(name)
            if "language_model" in name:
                terms = ([(*query_pairs[name], query_scale)] if name in query_pairs else []) + \
                        ([(*frame_pairs[name], -frame_scale)] if name in frame_pairs else [])
                out.append((mod, terms))
            if name in frame_pairs:
                mod.weight.data = _updated(mod.weight, *frame_pairs[name], frame_scale)
        return out

    def _question_hook(self, terms: list):
        def hook(mod, inputs, output):
            if not self._asking_now:
                return None
            x = inputs[0].float()
            extra = sum(((x @ a.T) @ b.T) * sc for a, b, sc in terms)
            return (output.float() + extra).to(output.dtype)
        return hook

    def reset_stream(self, stream=None) -> None:
        """The next frame of `stream` (default: every stream) starts a new stream: nothing is reused from frames of another source."""
        with self.lock:
            if stream is None:
                self.engines.clear()            # also frees the cached state
            else:
                self.engines.pop(stream, None)

    def encode_frame(self, image_path, stream=0) -> tuple[np.ndarray, float]:
        """Next frame of `stream` (one stream per screen, each with its own temporal reuse) -> (unit vector, encode milliseconds)."""
        t = time.perf_counter()
        if self.m is None:
            self.ensure_loaded()
        frame = preprocess(self.m, str(image_path) if isinstance(image_path, (str, Path)) else image_path, self.cfg)   # a file or a PIL image
        with self._questions:
            self._questions.wait_for(lambda: self._asking == 0, timeout=QUESTION_WAIT_S)
        with self.lock:
            self._frame_thread = threading.get_ident()
            engine = self.engines.get(stream)
            first = engine is None
            if first:
                engine = self.engines[stream] = Engine(self.m, self.cfg)
            emb, _ = engine.step(frame, first=first)
            v = emb.float().cpu().numpy()
            self._trim()
        return v, (time.perf_counter() - t) * 1000

    def _let_questions_in(self) -> None:
        """Between a frame's layers (engine.pause): if a question waits, give it the model, and go on once it is done."""
        if self._asking == 0 or self._frame_thread != threading.get_ident():
            return
        self.lock.release()
        try:
            with self._questions:
                self._questions.wait_for(lambda: self._asking == 0, timeout=QUESTION_WAIT_S)
        finally:
            self.lock.acquire()

    def _trim(self) -> None:
        """Metal keeps freed buffers unless asked to release them. On an M2 Pro over 245 frames, releasing after every frame cut peak Metal
        memory from 10.3 GB (every 50 frames) to 7.1 GB, with no slowdown (mean 675 vs 695 ms)."""
        if self.m is not None and self.m.device.type == "mps":
            torch.mps.empty_cache()

    def encode_query(self, text: str) -> np.ndarray:
        with self._questions:
            self._asking += 1
        try:
            with self.lock:
                if self.m is None:
                    print("streamembed: loading the model again", file=sys.stderr, flush=True)
                    self._load()
                self._asking_now = True
                try:
                    self.m.core.rope_deltas = None      # cached by the stock forward; must not leak between prompts
                    e = self.m.embedder.embed([self.m.embedder.format_model_input(text=text, instruction=QUERY_INSTRUCTION)],
                                              normalize=False)
                    self.m.core.rope_deltas = None
                finally:
                    self._asking_now = False
                v = F.normalize(e.float(), p=2, dim=-1)[0].cpu().numpy()
                self._trim()
        finally:
            with self._questions:
                self._asking -= 1
                self._questions.notify_all()
        return v
