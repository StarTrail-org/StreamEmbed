"""Which model encodes screens and questions. A model is a base model from its own Hugging Face repo, pinned to a commit, plus two
optional LoRAs (frame adapter, question LoRA) from the StreamEmbed model repo. `models.json` in that repo lists them:

    {"qwen3-vl-2b": {"base": "Qwen/Qwen3-VL-Embedding-2B", "revision": "<commit>",
                     "frame_lora": "qwen3-vl-2b/frame_adapter.safetensors", "query_lora": "qwen3-vl-2b/query_lora.safetensors",
                     "scale": 1.0, "architecture": "qwen3-vl", "description": "..."}}

The model repo is a Hugging Face repo id, or a local folder with the same layout (set by `configure(models_repo=...)` or
$STREAMEMBED_MODELS). Base models without LoRAs are built in, so they work without the model repo.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO = "dayoon/streamembed"
MODELS_REPO = os.environ.get("STREAMEMBED_MODELS", DEFAULT_REPO)
DEFAULT_MODEL = "qwen3-vl-2b"
STREAM_ARCHITECTURES = ("qwen3-vl",)      # the stream encoder re-implements the Qwen3-VL forward pass

BUILTIN = {
    "qwen3-vl-2b-base": dict(base="Qwen/Qwen3-VL-Embedding-2B", revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda", architecture="qwen3-vl",
                             description="Qwen3-VL-Embedding-2B without LoRAs"),
}


@dataclass
class ModelSpec:
    name: str
    base: str
    revision: str | None
    frame_lora: Path | None
    query_lora: Path | None
    scale: float
    architecture: str

    @property
    def weights(self) -> tuple[Path, Path] | None:
        return (self.frame_lora, self.query_lora) if self.frame_lora and self.query_lora else None

    @property
    def encoder_name(self) -> str:
        """Recorded in each index: vectors from another model or base commit must not share it."""
        return f"{self.name}@{(self.revision or 'main')[:12]}"


def _fetch(repo: str, filename: str) -> Path:
    if Path(repo).expanduser().is_dir():
        p = Path(repo).expanduser() / filename
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found")
        return p
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(repo, filename, local_files_only=True))     # once cached, no network
    except Exception:
        return Path(hf_hub_download(repo, filename))


def _manifest(repo: str) -> dict:
    return json.loads(_fetch(repo, "models.json").read_text(encoding="utf-8"))


def _cached_listing(repo: str | None) -> dict | None:
    """models.json plus the built-in models when the model list is on disk (a local folder, or the Hugging Face cache), else None.
    Never downloads."""
    repo = repo or MODELS_REPO
    try:
        if Path(repo).expanduser().is_dir():
            listing = _manifest(repo)
        else:
            from huggingface_hub import hf_hub_download
            listing = json.loads(Path(hf_hub_download(repo, "models.json", local_files_only=True)).read_text(encoding="utf-8"))
    except Exception:
        return None
    return {**listing, **{k: v for k, v in BUILTIN.items() if k not in listing}}


def cached_models(repo: str | None = None, stream_only: bool = False) -> list[str] | None:
    """Model names when the model list is on disk, else None (never downloads). stream_only: only those the stream encoder supports."""
    listing = _cached_listing(repo)
    if listing is None:
        return None
    return [k for k, v in listing.items() if not stream_only or v.get("architecture", "qwen3-vl") in STREAM_ARCHITECTURES]


def available_models(repo: str | None = None) -> list[str]:
    """Model names: the model repo's, then the built-in base models. The repo's are left out if it cannot be reached."""
    try:
        names = list(_manifest(repo or MODELS_REPO))
    except Exception:
        names = []
    return names + [n for n in BUILTIN if n not in names]


def _repo_error(repo: str, e: Exception) -> str:
    """One line on why models.json could not be read from `repo`."""
    back = f"`streamembed config reset models_repo` goes back to {DEFAULT_REPO}."
    if Path(repo).expanduser().is_dir():
        return f"models_repo {repo!r}: the folder has no readable models.json ({type(e).__name__}). {back}"
    try:
        from huggingface_hub.errors import (GatedRepoError, HFValidationError, LocalEntryNotFoundError, RemoteEntryNotFoundError,
                                            RepositoryNotFoundError)
    except ImportError:
        pass
    else:
        if isinstance(e, GatedRepoError):
            return f"models_repo {repo!r}: this Hugging Face repo needs access granted to your account. {back}"
        if isinstance(e, (RepositoryNotFoundError, RemoteEntryNotFoundError, HFValidationError)):
            return f"models_repo {repo!r}: no such Hugging Face repo or local folder (or no models.json in it). {back}"
        if isinstance(e, LocalEntryNotFoundError):
            return (f"models_repo {repo!r}: models.json is not downloaded yet and Hugging Face cannot be reached. Connect to the "
                    f"internet once, or use a built-in model ({', '.join(BUILTIN)}).")
    first = (str(e).strip().splitlines() or [type(e).__name__])[0]
    return f"Could not read models.json from {repo!r} ({first}). Built-in models: {', '.join(BUILTIN)}. {back}"


def resolve(name: str, repo: str | None = None) -> ModelSpec:
    repo = repo or MODELS_REPO
    if name in BUILTIN:
        entry = BUILTIN[name]
    else:
        try:
            listing = _manifest(repo)
        except Exception as e:
            raise RuntimeError(_repo_error(repo, e)) from None
        if name not in listing:
            raise ValueError(f"Unknown model {name!r}; available: {list(listing) + list(BUILTIN)}")
        entry = listing[name]
    arch = entry.get("architecture", "qwen3-vl")
    if arch not in STREAM_ARCHITECTURES:
        raise NotImplementedError(f"{name}: architecture {arch!r} is not supported yet; the stream encoder supports {STREAM_ARCHITECTURES}")
    loras = [_fetch(repo, entry[k]) if entry.get(k) else None for k in ("frame_lora", "query_lora")]
    return ModelSpec(name, entry["base"], entry.get("revision"), loras[0], loras[1], float(entry.get("scale", 1.0)), arch)
