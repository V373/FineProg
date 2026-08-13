"""Tests for the TCCEncoder embedding normalization output contract.

The ResNet50 backbone and the real temporal embedder are replaced by tiny
stand-ins so these tests stay fast and focus purely on the encoder contract:
config resolution, the single normalization point, dtype behaviour and
gradient flow.
"""

from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


_PROJECTS_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECTS_ROOT))

from fineprog.models import encoder as encoder_module
from fineprog.models.encoder import TCCEncoder


CLIP_LEN = 3
CONTEXT_SIZE = 2
CONTEXT_STRIDE = 1
FEAT_CHANNELS = 1024
EMBED_DIM = 128
FEAT_SPATIAL = 2


class _FakeBackbone(nn.Module):
    """Cheap stand-in for ResNet50Conv4cBackbone."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.pretrained = pretrained
        self.conv = nn.Conv2d(3, FEAT_CHANNELS, kernel_size=1)
        self.train_base_mode = None

    def set_train_base_mode(self, mode: str) -> None:
        self.train_base_mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, 3, 224, 224] -> [N, FEAT_CHANNELS, FEAT_SPATIAL, FEAT_SPATIAL]
        x = F.adaptive_avg_pool2d(x, FEAT_SPATIAL)
        return self.conv(x)


class _FakeTemporalEmbedder(nn.Module):
    """Cheap stand-in that still owns a trainable projection."""

    def __init__(self, in_channels=FEAT_CHANNELS, hidden_channels=512,
                 embed_dim=EMBED_DIM, debug=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # [B, clip_len, ctx, C, H, W] -> [B, clip_len, C] -> [B, clip_len, D]
        pooled = feats.mean(dim=(2, 4, 5))
        return self.proj(pooled)


@pytest.fixture(autouse=True)
def _patch_submodules(monkeypatch):
    """Swap the heavy sub-modules for the fakes in every test."""
    monkeypatch.setattr(encoder_module, "ResNet50Conv4cBackbone", _FakeBackbone)
    monkeypatch.setattr(encoder_module, "TCCTemporalEmbedder", _FakeTemporalEmbedder)
    torch.manual_seed(0)


def _write_config(tmp_path: Path, **extra) -> str:
    """Write a minimal train-style YAML, mirroring real encoder usage."""
    cfg = {
        "clip_len": CLIP_LEN,
        "context_size": CONTEXT_SIZE,
        "context_stride": CONTEXT_STRIDE,
        "backbone_name": "resnet50_conv4c",
        "pretrained": False,
        "train_base": "only_bn",
        "train_embedding": True,
    }
    cfg.update(extra)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _build_encoder(tmp_path: Path, yaml_extra: dict | None = None, **kwargs) -> TCCEncoder:
    config_path = _write_config(tmp_path, **(yaml_extra or {}))
    return TCCEncoder(
        config_path=config_path,
        train_config_path=config_path,
        **kwargs,
    )


def _make_frames(batch_size: int = 2, **kwargs) -> torch.Tensor:
    return torch.randn(batch_size, CLIP_LEN, CONTEXT_SIZE, 3, 224, 224, **kwargs)


def _make_feats(batch_size: int = 2, **kwargs) -> torch.Tensor:
    return torch.randn(
        batch_size, CLIP_LEN, CONTEXT_SIZE, FEAT_CHANNELS,
        FEAT_SPATIAL, FEAT_SPATIAL, **kwargs
    )


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #

def test_default_mode_is_none(tmp_path):
    """No YAML key and no explicit argument must keep legacy behaviour."""
    enc = _build_encoder(tmp_path)
    assert enc.embedding_normalization == "none"


def test_default_embedding_dim_is_128(tmp_path):
    enc = _build_encoder(tmp_path)
    assert enc.embedding_dim == 128
    assert enc.temporal_embedder.proj.out_features == 128


@pytest.mark.parametrize("embedding_dim", [32, 64, 128])
def test_yaml_embedding_dim_controls_all_output_paths(tmp_path, embedding_dim):
    enc = _build_encoder(
        tmp_path,
        yaml_extra={
            "embedding_dim": embedding_dim,
            "embedding_normalization": "l2",
        },
    )
    enc.eval()
    frames = _make_frames()

    with torch.no_grad():
        regular = enc(frames)
        cached = enc.forward_from_feats(
            enc(
                frames,
                return_backbone_feats=True,
            )["grouped_backbone_feats"]
        )

    expected_shape = (2, CLIP_LEN, embedding_dim)
    assert enc.embedding_dim == embedding_dim
    assert enc.temporal_embedder.proj.out_features == embedding_dim
    assert regular.shape == expected_shape
    assert cached.shape == expected_shape
    assert torch.allclose(
        regular.norm(dim=-1),
        torch.ones(2, CLIP_LEN),
        atol=1e-5,
    )
    assert torch.allclose(
        cached.norm(dim=-1),
        torch.ones(2, CLIP_LEN),
        atol=1e-5,
    )


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, "32", True, None])
def test_invalid_yaml_embedding_dim_is_rejected(tmp_path, bad_value):
    with pytest.raises(ValueError, match="embedding_dim"):
        _build_encoder(tmp_path, yaml_extra={"embedding_dim": bad_value})


def test_yaml_enables_l2(tmp_path):
    enc = _build_encoder(tmp_path, yaml_extra={"embedding_normalization": "l2"})
    assert enc.embedding_normalization == "l2"


def test_yaml_none_is_accepted(tmp_path):
    enc = _build_encoder(tmp_path, yaml_extra={"embedding_normalization": "none"})
    assert enc.embedding_normalization == "none"


def test_explicit_argument_overrides_yaml(tmp_path):
    """An explicit value always wins, so a loader can pass the authoritative mode."""
    enc = _build_encoder(
        tmp_path,
        yaml_extra={"embedding_normalization": "l2"},
        embedding_normalization="none",
    )
    assert enc.embedding_normalization == "none"

    enc = _build_encoder(tmp_path, embedding_normalization="l2")
    assert enc.embedding_normalization == "l2"


@pytest.mark.parametrize("bad_value", ["L2", "unit", "", "l2_norm", True, False, None, 1, 1.0])
def test_invalid_explicit_value_is_rejected(tmp_path, bad_value):
    with pytest.raises(ValueError, match="embedding_normalization"):
        _build_encoder(tmp_path, embedding_normalization=bad_value)


@pytest.mark.parametrize("bad_value", ["L2", True, None])
def test_invalid_yaml_value_is_rejected(tmp_path, bad_value):
    """A misconfigured YAML (including an explicit null) must fail fast."""
    with pytest.raises(ValueError, match="embedding_normalization"):
        _build_encoder(tmp_path, yaml_extra={"embedding_normalization": bad_value})


def test_state_dict_keys_are_mode_independent(tmp_path):
    """The contract must not add parameters/buffers, so legacy checkpoints still load."""
    enc_none = _build_encoder(tmp_path, embedding_normalization="none")
    enc_l2 = _build_encoder(tmp_path, embedding_normalization="l2")
    assert list(enc_none.state_dict().keys()) == list(enc_l2.state_dict().keys())
    enc_l2.load_state_dict(enc_none.state_dict(), strict=True)


# --------------------------------------------------------------------------- #
# none mode: unchanged legacy behaviour
# --------------------------------------------------------------------------- #

def test_none_mode_returns_raw_projection(tmp_path):
    enc = _build_encoder(tmp_path, embedding_normalization="none")
    enc.eval()
    feats = _make_feats()

    with torch.no_grad():
        raw = enc.temporal_embedder(feats)
        out = enc.forward_from_feats(feats)

    assert out.dtype == raw.dtype
    assert torch.equal(out, raw)
    # Raw projections are not unit vectors, so the test would be vacuous otherwise.
    assert not torch.allclose(out.norm(dim=-1), torch.ones_like(out.norm(dim=-1)), atol=1e-3)


# --------------------------------------------------------------------------- #
# l2 mode: unit norm on every public path
# --------------------------------------------------------------------------- #

def test_l2_forward_returns_unit_norm_fp32(tmp_path):
    enc = _build_encoder(tmp_path, embedding_normalization="l2")
    enc.eval()
    frames = _make_frames()

    with torch.no_grad():
        embeddings = enc(frames)

    assert embeddings.shape == (2, CLIP_LEN, EMBED_DIM)
    assert embeddings.dtype == torch.float32
    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_l2_cache_path_returns_unit_norm(tmp_path):
    enc = _build_encoder(tmp_path, embedding_normalization="l2")
    enc.eval()
    feats = _make_feats()

    with torch.no_grad():
        embeddings = enc.forward_from_feats(feats)

    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_dict_path_normalizes_embeddings_but_not_backbone_feats(tmp_path):
    enc = _build_encoder(tmp_path, embedding_normalization="l2")
    enc.eval()
    frames = _make_frames()

    with torch.no_grad():
        out = enc(frames, return_backbone_feats=True)

    embeddings = out["embeddings"]
    backbone_feats = out["grouped_backbone_feats"]

    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Backbone features must be handed back raw.
    with torch.no_grad():
        expected_feats = enc.backbone(
            frames.reshape(-1, 3, 224, 224)
        ).reshape(backbone_feats.shape)
    assert torch.equal(backbone_feats, expected_feats)


def test_all_paths_apply_normalization_exactly_once(tmp_path):
    """Regular forward, dict forward and the cache path must agree numerically."""
    enc = _build_encoder(tmp_path, embedding_normalization="l2")
    enc.eval()
    frames = _make_frames()

    with torch.no_grad():
        plain = enc(frames)
        dict_out = enc(frames, return_backbone_feats=True)
        from_feats = enc.forward_from_feats(dict_out["grouped_backbone_feats"])

    assert torch.allclose(plain, dict_out["embeddings"], atol=1e-6)
    assert torch.allclose(plain, from_feats, atol=1e-6)


# --------------------------------------------------------------------------- #
# Gradients / AMP
# --------------------------------------------------------------------------- #

def test_l2_gradients_are_finite(tmp_path):
    enc = _build_encoder(tmp_path, embedding_normalization="l2")
    enc.train()
    enc.configure_trainability()

    feats = _make_feats(requires_grad=True)
    embeddings = enc.forward_from_feats(feats)

    weights = torch.randn_like(embeddings)
    (embeddings * weights).sum().backward()

    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()

    proj_grad = enc.temporal_embedder.proj.weight.grad
    assert proj_grad is not None
    assert torch.isfinite(proj_grad).all()
    assert proj_grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP path requires CUDA")
def test_l2_under_autocast_returns_finite_fp32(tmp_path):
    enc = _build_encoder(tmp_path, embedding_normalization="l2").to("cuda")
    enc.train()
    enc.configure_trainability()

    feats = _make_feats(device="cuda", requires_grad=True)

    with torch.amp.autocast(device_type="cuda", enabled=True):
        embeddings = enc.forward_from_feats(feats)

    assert embeddings.dtype == torch.float32
    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    embeddings.float().pow(2).sum().backward()
    assert torch.isfinite(feats.grad).all()
    assert torch.isfinite(enc.temporal_embedder.proj.weight.grad).all()
