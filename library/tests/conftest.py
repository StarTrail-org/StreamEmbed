import pytest


@pytest.fixture(autouse=True)
def own_settings(tmp_path_factory, monkeypatch):
    """Every test reads and writes its own settings file, never the user's."""
    monkeypatch.setenv("STREAMEMBED_CONFIG", str(tmp_path_factory.mktemp("config") / "config.toml"))
    monkeypatch.delenv("STREAMEMBED_INDEX", raising=False)
    monkeypatch.delenv("STREAMEMBED_MODELS", raising=False)


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    """A test that would load the real model (about 8 GB) fails instead; tests use a stand-in encoder."""
    import streamembed.api

    def refuse(*a, **k):
        raise AssertionError("a test tried to load the real model")

    monkeypatch.setattr(streamembed.api, "resolve", refuse)
