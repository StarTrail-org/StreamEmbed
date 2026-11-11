"""models.json resolution without the model: a local folder stands in for the Hugging Face model repo."""
import json

import pytest
import torch
from safetensors.torch import save_file

import streamembed as se
from streamembed.models import BUILTIN, resolve


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "m").mkdir()
    for f in ("frame.safetensors", "query.safetensors"):
        save_file({"base_model.model.x.lora_A.weight": torch.zeros(2, 4), "base_model.model.x.lora_B.weight": torch.zeros(4, 2)},
                  tmp_path / "m" / f)
    (tmp_path / "models.json").write_text(json.dumps({
        "m-2b": dict(base="Org/Base", revision="abc123", frame_lora="m/frame.safetensors", query_lora="m/query.safetensors", scale=0.5),
        "other": dict(base="Org/Siglip", architecture="siglip2"),
        "broken": dict(base="Org/Base", frame_lora="m/missing.safetensors", query_lora="m/query.safetensors"),
    }))
    return tmp_path


def test_resolve_from_a_local_repo(repo):
    s = resolve("m-2b", str(repo))
    assert (s.base, s.revision, s.scale, s.architecture) == ("Org/Base", "abc123", 0.5, "qwen3-vl")
    assert s.weights == (repo / "m" / "frame.safetensors", repo / "m" / "query.safetensors")
    assert s.encoder_name == "m-2b@abc123"


def test_unknown_unsupported_and_missing(repo):
    with pytest.raises(ValueError, match="Unknown model"):
        resolve("nope", str(repo))
    with pytest.raises(NotImplementedError, match="siglip2"):
        resolve("other", str(repo))
    with pytest.raises(FileNotFoundError):
        resolve("broken", str(repo))


def test_builtin_needs_no_repo(tmp_path):
    s = resolve("qwen3-vl-2b-base", str(tmp_path / "no_repo_here"))
    assert s.weights is None and s.base == BUILTIN["qwen3-vl-2b-base"]["base"] and len(s.revision) == 40


def test_available_models_follows_configure(repo, tmp_path):
    se.configure(index_dir=tmp_path / "idx", models_repo=repo)
    assert se.available_models() == ["m-2b", "other", "broken", "qwen3-vl-2b-base"]
    se.configure(index_dir=tmp_path / "idx", models_repo=tmp_path / "nothing")
    assert se.available_models() == ["qwen3-vl-2b-base"], "an unreachable repo leaves the built-in models"
    with pytest.raises(RuntimeError, match="models.json"):
        resolve("m-2b", str(tmp_path / "nothing"))


def test_set_config_checks_the_model_against_the_repo_in_the_same_call(repo):
    for values in (dict(model="m-2b", models_repo=str(repo)), dict(models_repo=str(repo), model="m-2b")):
        se.reset_config()
        se.set_config(**values)
        assert se.get_config("model") == "m-2b" and se.get_config("models_repo") == str(repo)
    with pytest.raises(ValueError, match=r"one of m-2b, broken, qwen3-vl-2b-base, not 'other' \(listed, but"):
        se.set_config(model="other")                    # the stream encoder supports Qwen3-VL models only
    with pytest.raises(ValueError, match="one of m-2b"):
        se.set_config(model="nope")
    se.reset_config()
