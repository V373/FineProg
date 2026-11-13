"""Shared strict loading for training, embedding extraction and activation maps."""

import hashlib
import warnings
from copy import deepcopy
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
        "single_frame_context": encoder.single_frame_context,
        "embedding_normalization": encoder.embedding_normalization, "pretrained": False,
        "train_base": encoder._train_base, "train_embedding": encoder._train_embedding,
        "memory_config": dict(encoder.memory_config),
    }


def checkpoint_dict(encoder, training_provenance=None):
    payload = {"model_state_dict": encoder.state_dict(),
               "embedding_normalization": encoder.embedding_normalization,
               "encoder_config": encoder_config(encoder)}
    if encoder.memory is not None:
        payload["full_episode"] = True
        payload["versions"] = {name: version(name) for name in ("mamba-ssm", "causal-conv1d")}
    provenance = (getattr(encoder, "training_provenance", None)
                  if training_provenance is None else training_provenance)
    if provenance:
        payload["training_provenance"] = deepcopy(provenance)
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
    if hasattr(model, "single_frame_context"):
        # This input contract belongs to the saved weights, including old weights
        # that predate the flag and must keep their original temporal context.
        single_frame_context = model._resolve_single_frame_context(
            (checkpoint.get("encoder_config") or {}).get("single_frame_context", False)
        )
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    model.embedding_normalization = normalization
    if hasattr(model, "single_frame_context"):
        model.single_frame_context = single_frame_context
    model.training_provenance = deepcopy(checkpoint.get("training_provenance", {}))
    return normalization


def load_checkpoint(model, ckpt_path, device):
    return _load_weights(model, torch.load(ckpt_path, map_location=device, weights_only=True), ckpt_path)


def _kwargs(config):
    keys = ("clip_len", "context_size", "context_stride", "embedding_dim", "pretrained",
            "embedding_normalization", "train_base", "train_embedding", "single_frame_context")
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


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_cross_attention_training_encoder(config, memory, device):
    source_path = memory.get("source_checkpoint_path")
    if not source_path:
        raise ValueError("Cross-attention training requires memory.source_checkpoint_path "
                         "for a complete single-input, no-residual memory checkpoint.")
    if memory.get("local_checkpoint_path"):
        raise ValueError("Use memory.source_checkpoint_path alone for cross-attention migration; "
                         "memory.local_checkpoint_path loads only a local encoder.")
    source = build_encoder_from_checkpoint(source_path, "cpu")
    source_memory = source.memory
    if (source_memory is None or type(source_memory).__name__ != "StreamingMamba2"
            or source_memory.history_frames != 1 or source_memory.use_residual):
        raise ValueError("memory.source_checkpoint_path must contain a single-branch "
                         "StreamingMamba2 with history_frames=1 and use_residual=false.")

    # The source determines the complete numerical short path. clip_len remains
    # this run's loss-target budget, not a restriction on the memory sequence.
    for key in ("context_size", "context_stride", "embedding_dim", "embedding_normalization",
                "single_frame_context"):
        config[key] = getattr(source, key)
    config.setdefault("clip_len", source.clip_len)
    config.setdefault("train_base", "frozen")
    config.setdefault("train_embedding", False)
    if config.get("extract_backbone_cache", False) and config["train_base"] != "frozen":
        raise ValueError("Memory backbone caching requires train_base=frozen; otherwise gradients would be lost.")
    for key in ("d_model", "d_state", "d_conv", "expand", "headdim", "ngroups", "chunk_size"):
        memory[key] = source_memory.config[key]
    memory.update(history_frames=1, history_stride=source_memory.history_stride,
                  normalize_input=source_memory.normalize_input, use_residual=False,
                  dual_branch=False, enabled=True, mode="cross_attention")
    if memory.get("branch_output_l2") is None:
        memory["branch_output_l2"] = source.embedding_normalization == "l2"
    memory["source_checkpoint_sha256"] = _checkpoint_sha256(source_path)
    memory["source_encoder_config"] = encoder_config(source)
    if getattr(source, "training_provenance", None):
        memory["source_training_provenance"] = deepcopy(source.training_provenance)
    else:
        memory.pop("source_training_provenance", None)
    config["memory"] = memory
    kwargs = _kwargs(config)
    kwargs["pretrained"] = False
    encoder = _encoder_class()(**kwargs)
    # Explicit module copies keep strict loading for every inherited parameter;
    # only the attention, input mixing and gate modules remain newly initialized.
    for name in ("backbone", "temporal_embedder"):
        getattr(encoder, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
    for name in ("block", "proj", "history_proj", "pre_norm"):
        getattr(encoder.memory, name).load_state_dict(
            getattr(source_memory, name).state_dict(), strict=True)
    with torch.no_grad():
        encoder.memory.alpha.copy_(source_memory.alpha)
    encoder.configure_trainability()
    config["memory"] = dict(encoder.memory_config)
    return encoder.to(device), config


def build_training_encoder(config, device):
    """Load a local warm start or explicitly migrate a complete memory source."""
    config = dict(config)
    memory = dict(config.get("memory", {}))
    enabled = memory.get("enabled", False)
    if enabled and memory.get("mode") == "cross_attention":
        return _build_cross_attention_training_encoder(config, memory, device)
    if enabled and memory.get("source_checkpoint_path"):
        raise ValueError("memory.source_checkpoint_path requires memory.mode=cross_attention.")
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
        config["single_frame_context"] = source.get("single_frame_context", False)
        memory["local_checkpoint_sha256"] = _checkpoint_sha256(local_path)
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
                   ("context_size", "context_stride", "embedding_dim", "embedding_normalization",
                    "single_frame_context")})
    config["memory"] = dict(encoder.memory_config)
    return encoder.to(device), config
