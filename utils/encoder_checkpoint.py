"""Shared strict loading for training, embedding extraction and activation maps."""

import hashlib
import warnings
from importlib.metadata import version

import torch
import yaml

from .embedding_normalization import validate_embedding_normalization


def _encoder_class():
    try:
        from ..models.encoder import TCCEncoder
    except ImportError:
        from models.encoder import TCCEncoder
    return TCCEncoder


def encoder_config(encoder):
    return {
        "clip_len": encoder.clip_len, "context_size": encoder.context_size,
        "context_stride": encoder.context_stride, "embedding_dim": encoder.embedding_dim,
        "embedding_normalization": encoder.embedding_normalization, "pretrained": False,
        "train_base": encoder._train_base, "train_embedding": encoder._train_embedding,
        "memory_config": dict(encoder.memory_config),
    }


def checkpoint_dict(encoder):
    payload = {"model_state_dict": encoder.state_dict(),
               "embedding_normalization": encoder.embedding_normalization,
               "encoder_config": encoder_config(encoder)}
    if encoder.memory is not None:
        payload["full_episode"] = True
        payload["versions"] = {name: version(name) for name in ("mamba-ssm", "causal-conv1d")}
    return payload


def _load_weights(model, checkpoint, path):
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    wrapped = "model_state_dict" in checkpoint or "state_dict" in checkpoint
    if not wrapped or "embedding_normalization" not in checkpoint:
        warnings.warn(f"{path}: legacy checkpoint has no embedding_normalization metadata; "
                      "treating it as 'none'.", UserWarning, stacklevel=2)
        normalization = "none"
    else:
        normalization = checkpoint["embedding_normalization"]
    normalization = validate_embedding_normalization(normalization, str(path))
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    model.embedding_normalization = normalization
    return normalization


def load_checkpoint(model, ckpt_path, device):
    return _load_weights(model, torch.load(ckpt_path, map_location=device, weights_only=True), ckpt_path)


def _kwargs(config):
    keys = ("clip_len", "context_size", "context_stride", "embedding_dim", "pretrained",
            "embedding_normalization", "train_base", "train_embedding")
    return {**{k: config[k] for k in keys if k in config},
            "memory_config": dict(config.get("memory", {}))}


def build_encoder_from_checkpoint(path, device, fallback_config=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    kwargs = checkpoint.get("encoder_config")
    if kwargs is None:
        kwargs = _kwargs(fallback_config or {})
        kwargs.update(pretrained=False, memory_config={})
    model = _encoder_class()(**kwargs)
    _load_weights(model, checkpoint, path)
    return model.to(device)


def build_training_encoder(config, device):
    """Warm-start only the local weights; this run's flags decide trainability."""
    config = dict(config)
    memory = dict(config.get("memory", {}))
    enabled = memory.get("enabled", False)
    local_path = memory.get("local_checkpoint_path") if enabled else None
    checkpoint = None
    if local_path:
        checkpoint = torch.load(local_path, map_location="cpu", weights_only=True)
        source = checkpoint.get("encoder_config")
        if source is None:
            if not memory.get("local_config_path"):
                raise ValueError("A legacy local checkpoint needs memory.local_config_path (original train YAML).")
            with open(memory["local_config_path"]) as f:
                source = yaml.safe_load(f)
        for key in ("context_size", "context_stride", "embedding_dim"):
            config[key] = source[key]
        digest = hashlib.sha256()
        with open(local_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        memory["local_checkpoint_sha256"] = digest.hexdigest()
    elif enabled and config.get("train_base") == "frozen" and not config.get("train_embedding", True):
        raise ValueError("Freezing the entire local encoder requires memory.local_checkpoint_path.")
    if enabled and config.get("extract_backbone_cache", False) and config.get("train_base") != "frozen":
        raise ValueError("Memory backbone caching requires train_base=frozen; otherwise gradients would be lost.")

    kwargs = _kwargs(config)
    kwargs["memory_config"] = {}
    if checkpoint is not None:
        kwargs["pretrained"] = False
    encoder = _encoder_class()(**kwargs)
    if checkpoint is not None:
        _load_weights(encoder, checkpoint, local_path)
    encoder.enable_memory(memory)
    encoder.configure_trainability()
    config.update({k: getattr(encoder, k) for k in
                   ("context_size", "context_stride", "embedding_dim", "embedding_normalization")})
    config["memory"] = dict(encoder.memory_config)
    return encoder.to(device), config
