import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fineprog import models


class TinyBackbone(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.bn1 = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class TinyEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(4, 4, 1)
        self.proj = nn.Linear(4, 128)

    def forward(self, features):
        x = self.conv3d_1(features.permute(0, 2, 1, 3, 4))
        return self.proj(x.mean((2, 3, 4)))


@pytest.fixture
def tiny_models(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(models, "ResNet50Conv4cBackbone", TinyBackbone)
    monkeypatch.setattr(models, "TCCTemporalEmbedder", TinyEmbedder)
    def efficientnet(weights=None):
        return SimpleNamespace(features=nn.Sequential(
            nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(),
            nn.Identity(), nn.Identity(), nn.Identity(),
            nn.Conv2d(4, 1280, 1), nn.BatchNorm2d(1280), nn.ReLU()))
    monkeypatch.setattr(models, "efficientnet_b0", efficientnet)


def test_local_repeats_before_backbone_and_preserves_bn_and_gradients(tiny_models):
    torch.manual_seed(42)
    model = models.LocalEncoder().train()
    reference = copy.deepcopy(model)
    frames = torch.rand(3, 3, 6, 6, requires_grad=True)
    repeated = frames[:, None].expand(-1, 2, -1, -1, -1).reshape(-1, 3, 6, 6)
    features = reference.backbone(repeated).reshape(3, 2, 4, 6, 6)
    expected = F.normalize(reference.temporal_embedder(features).float(), dim=-1)
    actual = model(frames)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.backbone.bn1.running_var, reference.backbone.bn1.running_var, rtol=0, atol=0)
    assert model.backbone.bn1.num_batches_tracked == 1
    (actual * torch.linspace(-1, 1, 128)).sum().backward()
    assert frames.grad.abs().sum() > 0
    assert model.backbone.conv1.weight.grad is None
    assert model.backbone.bn1.weight.grad.abs().sum() > 0
    assert model.temporal_embedder.proj.weight.grad.abs().sum() > 0
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(3))


def test_local_selected_checkpoint_roundtrip_and_input_scaling(tiny_models, tmp_path):
    model = models.LocalEncoder().eval()
    frames = torch.randint(256, (3, 3, 6, 6), dtype=torch.uint8)
    path = tmp_path / "local.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    loaded = models.load_local(path)
    torch.testing.assert_close(loaded(frames), model(frames.float() / 255), rtol=0, atol=0)
    assert not loaded.training


def test_upper_keeps_feature_statistics_frozen_and_old_state_schema(tiny_models, tmp_path):
    model = models.SemanticClassifier().train()
    bn = model.features[7]
    before = bn.running_mean.clone()
    frames = torch.randint(256, (3, 3, 6, 6), dtype=torch.uint8)
    F.cross_entropy(model(frames), torch.tensor([0, 1, 2])).backward()
    torch.testing.assert_close(bn.running_mean, before, rtol=0, atol=0)
    assert model.features[0].weight.grad is None
    assert model.features[6].weight.grad.abs().sum() > 0
    assert model.head.weight.grad.abs().sum() > 0
    assert not model.features.training
    path = tmp_path / "upper.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {"l2_normalize": False}}, path)
    loaded = models.load_semantic(path)
    torch.testing.assert_close(model(frames), loaded(frames), rtol=0, atol=0)
