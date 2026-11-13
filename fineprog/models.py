"""Independent semantic classifier and the single-frame local progress encoder."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0, resnet50


LABELS = ("R", "M", "G")


class ResNet50Conv4cBackbone(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        model = resnet50(weights=None)
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"):
            setattr(self, name, getattr(model, name))

    def forward(self, frames):
        x = self.maxpool(self.relu(self.bn1(self.conv1(frames))))
        return self.layer3(self.layer2(self.layer1(x)))


class TCCTemporalEmbedder(nn.Module):
    """Keep the selected local checkpoints' parameter names and two-slot head."""

    def __init__(self):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1024, 512, 3, padding=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv3d_2 = nn.Conv3d(512, 512, 3, padding=1)
        self.relu_2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512, 512)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 512)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.proj = nn.Linear(512, 128)

    def forward(self, features):
        x = features.permute(0, 2, 1, 3, 4).contiguous(memory_format=torch.channels_last_3d)
        x = self.relu_2(self.conv3d_2(self.relu_1(self.conv3d_1(x))))
        x = F.adaptive_max_pool3d(x, 1).flatten(1)
        return self.proj(self.relu_fc2(self.fc2(self.relu_fc1(self.fc1(x)))))


class LocalEncoder(nn.Module):
    """RGB [N,3,H,W] uint8/[0,1] -> unit-length [N,128] embeddings.

    The current frame is repeated before the backbone, preserving the original
    two-slot training batch and BN statistics. Only backbone BN affine parameters
    and the complete embedding head train. The lower model uses RGB / 255 only.
    """

    def __init__(self, pretrained_path=None):
        super().__init__()
        self.backbone = ResNet50Conv4cBackbone(pretrained_path)
        self.temporal_embedder = TCCTemporalEmbedder()
        self.backbone.to(memory_format=torch.channels_last)
        self.temporal_embedder.to(memory_format=torch.channels_last_3d)
        self.backbone.requires_grad_(False)
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.requires_grad_(True)

    def forward(self, frames):
        frames = frames.float().div(255) if frames.dtype == torch.uint8 else frames.float()
        contexts = frames[:, None].expand(-1, 2, -1, -1, -1)
        features = self.backbone(contexts.reshape(-1, *frames.shape[1:]))
        features = features.reshape(len(frames), 2, *features.shape[1:])
        return F.normalize(self.temporal_embedder(features).float(), dim=-1, eps=1e-12)


class SemanticClassifier(nn.Module):
    """Single-frame EfficientNet-B0 -> 32D MLP -> R/M/G logits."""

    def __init__(self, pretrained_path=None, l2_normalize=False):
        super().__init__()
        self.l2_normalize = l2_normalize
        model = efficientnet_b0(weights=None)
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path, map_location="cpu", weights_only=True))
        self.features = model.features
        self.features[:6].requires_grad_(False)
        self.mlp = nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Linear(128, 32))
        self.head = nn.Linear(32, 3)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def embed(self, frames):
        frames = frames.float().div(255) if frames.dtype == torch.uint8 else frames.float()
        latent = self.mlp(self.features((frames - self.mean) / self.std).mean((2, 3)))
        return F.normalize(latent, dim=1) if self.l2_normalize else latent

    def forward(self, frames):
        return self.head(self.embed(frames))

    def param_groups(self, backbone_lr, head_lr):
        return [{"params": self.features[6:].parameters(), "lr": backbone_lr},
                {"params": [*self.mlp.parameters(), *self.head.parameters()], "lr": head_lr}]


def load_local(path, device="cpu"):
    """Load a selected single-frame local/Stage2 checkpoint, without its old project."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = LocalEncoder()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def load_semantic(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SemanticClassifier(l2_normalize=checkpoint["config"]["l2_normalize"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()
