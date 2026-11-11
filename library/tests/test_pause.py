"""A question that arrives while a frame is being encoded goes first at the frame's next pause point (every engine.pause.PAUSE_EVERY
layers), and the frame goes on after it."""
import threading
import time
from types import SimpleNamespace

import torch

from streamembed import encoder as E
from streamembed.engine import pause
from test_encoder_order import _encoder


def test_a_question_goes_between_a_frames_layers(monkeypatch):
    order = []
    enc = _encoder(monkeypatch, order)
    enc.m.sync = lambda: None
    enc._frame_thread = None
    pause.set_hook(enc._let_questions_in)

    class LayeredEngine:
        def __init__(self, m, cfg):
            pass

        def step(self, frame, first):
            for layer in range(12):
                time.sleep(0.05)
                order.append(f"layer {layer}")
                pause.point(enc.m, layer)
            return torch.zeros(4), None

    monkeypatch.setattr(E, "Engine", LayeredEngine)
    try:
        t = threading.Thread(target=lambda: enc.encode_frame(1))
        t.start()
        time.sleep(0.12)                     # the frame is at layer 2
        started = time.monotonic()
        enc.encode_query("a terminal")
        waited = time.monotonic() - started
        t.join()
    finally:
        pause.set_hook(None)
    q = order.index("question")
    assert order[q - 1] == "layer 3" and order[-1] == "layer 11", order
    assert waited < 0.3, "it waited for a few layers, not for the frame (0.6 s)"


def test_no_hook_no_pause():
    calls = []
    m = SimpleNamespace(sync=lambda: calls.append("sync"))
    pause.set_hook(None)
    for layer in range(8):
        pause.point(m, layer)
    assert calls == []
