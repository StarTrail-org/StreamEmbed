"""A question waits for at most the frame being encoded: frames queued behind it go after it."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from streamembed import encoder as E


def _encoder(monkeypatch, order):
    class FakeEngine:
        def __init__(self, m, cfg):
            pass

        def step(self, frame, first):
            time.sleep(0.3)
            order.append(f"frame {frame}")
            return torch.zeros(4), None

    class FakeEmbedder:
        def format_model_input(self, text, instruction):
            return text

        def embed(self, inputs, normalize):
            order.append("question")
            return torch.ones(1, 4)

    monkeypatch.setattr(E, "Engine", FakeEngine)
    monkeypatch.setattr(E, "preprocess", lambda m, img, cfg: img)
    enc = E.Encoder.__new__(E.Encoder)
    enc.m = SimpleNamespace(embedder=FakeEmbedder(), core=SimpleNamespace(rope_deltas=None), device=torch.device("cpu"))
    enc.cfg, enc.engines, enc._swap = None, {}, []
    enc.lock = threading.Lock()
    enc._questions = threading.Condition()
    enc._asking = 0
    return enc


def test_a_question_goes_before_queued_frames(monkeypatch):
    order = []
    enc = _encoder(monkeypatch, order)

    def frames():                       # the capture: two displays, one after the other
        enc.encode_frame(1, stream=1)
        enc.encode_frame(2, stream=2)

    t = threading.Thread(target=frames)
    t.start()
    time.sleep(0.1)                     # frame 1 is being encoded
    started = time.monotonic()
    v = enc.encode_query("a terminal")
    waited = time.monotonic() - started
    t.join()
    assert order == ["frame 1", "question", "frame 2"]
    assert waited < 0.5, "only the frame being encoded"
    assert np.allclose(np.linalg.norm(v), 1.0)


def test_frames_are_not_held_forever(monkeypatch):
    order = []
    enc = _encoder(monkeypatch, order)
    monkeypatch.setattr(E, "QUESTION_WAIT_S", 0.2)
    enc._asking = 1                     # a question that never finishes
    started = time.monotonic()
    enc.encode_frame(1)
    assert order == ["frame 1"] and time.monotonic() - started < 1.0
