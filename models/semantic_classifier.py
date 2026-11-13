"""Single-frame R/M/G semantic classifier: EfficientNet-B0 features, 32D MLP, linear head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


LABELS = ("R", "M", "G")


class SemanticClassifier(nn.Module):
    """RGB frames [B,3,224,224] (uint8, or float in [0,1]) -> logits [B,3].

    Partial fine-tuning: features[:6] (stem and the first five MBConv stages) are
    frozen; features[6:] (last two MBConv stages and the final 1x1 conv), the MLP
    and the head are trained. Every BN keeps its pretrained running statistics.
    l2_normalize=False feeds the raw 32D MLP output to the head, so the logit scale
    is no longer bounded by the head weight norm.
    """

    def __init__(self, pretrained=True, l2_normalize=True):
        super().__init__()
        self.l2_normalize = l2_normalize
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = efficientnet_b0(weights=weights).features
        self.mlp = nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Linear(128, 32))
        self.head = nn.Linear(32, 3)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for parameter in self.features[:6].parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # eval() fixes the BN running statistics; it also disables stochastic depth.
        self.features.eval()
        return self

    def embed(self, frames):
        """32D latent [B,32], unit-norm when l2_normalize."""
        frames = frames.float().div(255) if frames.dtype == torch.uint8 else frames.float()
        latent = self.mlp(self.features((frames - self.mean) / self.std).mean((2, 3)))
        return F.normalize(latent, dim=1) if self.l2_normalize else latent

    def forward(self, frames):
        return self.head(self.embed(frames))

    def param_groups(self, backbone_lr, head_lr):
        return [{"params": list(self.features[6:].parameters()), "lr": backbone_lr},
                {"params": [*self.mlp.parameters(), *self.head.parameters()], "lr": head_lr}]

    @torch.no_grad()
    def predict(self, frames, batch_size=256):
        """Independent per-frame inference: [T,3,224,224] -> probabilities [T,3], labels [T]."""
        device = self.mean.device
        probabilities = torch.cat([self(frames[start:start + batch_size].to(device)).softmax(1)
                                   for start in range(0, len(frames), batch_size)])
        return probabilities, probabilities.argmax(1)


def load_semantic_classifier(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # Checkpoints saved before the l2_normalize option always normalized.
    model = SemanticClassifier(pretrained=False, l2_normalize=checkpoint["config"].get("l2_normalize", True))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
