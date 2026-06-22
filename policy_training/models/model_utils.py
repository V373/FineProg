"""Standalone neural network utilities for policy_training.

Provides self-contained re-implementations of building blocks that originate
from robomimic (e.g. SpatialSoftmax), so policy_training has no hard runtime
dependency on the robomimic package.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenPretrainedResNet18Conv(nn.Module):
    """Frozen pretrained ResNet18Conv feature extractor with robomimic parity."""

    def __init__(self, device: torch.device | str | None = None):
        super().__init__()

        from envs import ensure_vendored_robomimic

        ensure_vendored_robomimic()

        from envs.robomimic_runtime.robomimic.models.base_nets import ResNet18Conv

        self.backbone = ResNet18Conv(
            input_channel=3,
            pretrained=True,
            freeze=True,
            imagenet_norm=True,
        )
        self.backbone.eval()
        if device is not None:
            self.to(device)

    @staticmethod
    def preprocess_images(images: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, bool]:
        """Convert RGB inputs to batched BCHW float tensors."""
        tensor = images if isinstance(images, torch.Tensor) else torch.as_tensor(images)
        squeeze_batch = False

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_batch = True
        if tensor.ndim != 4:
            raise ValueError(f"Expected image tensor rank 3 or 4, got shape {tuple(tensor.shape)}")

        if tensor.shape[-1] in (1, 3) and tensor.shape[1] not in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] not in (1, 3):
            raise ValueError(
                "Expected images in HWC or CHW format with 1 or 3 channels, "
                f"got shape {tuple(tensor.shape)}"
            )

        tensor = tensor.to(dtype=torch.float32)
        if torch.max(tensor).item() > 1.0:
            tensor = tensor / 255.0

        return tensor, squeeze_batch

    @torch.no_grad()
    def forward(self, images: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor, squeeze_batch = self.preprocess_images(images)
        tensor = tensor.to(device=next(self.backbone.parameters()).device)
        features = self.backbone(tensor)
        if squeeze_batch:
            features = features.squeeze(0)
        return features

    @torch.no_grad()
    def extract_feature_map(self, images: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Alias for forward() to make call-sites self-documenting."""
        return self.forward(images)


class SpatialSoftmax(nn.Module):
    """Spatial Softmax pooling layer.

    Converts a spatial feature map ``[B, C, H, W]`` into a set of 2-D keypoint
    coordinates ``[B, num_kp, 2]`` by treating each channel (after an optional
    1×1 conv that reduces C → num_kp) as a spatial probability distribution and
    computing its expected (x, y) position.

    Re-implemented from robomimic ``SpatialSoftmax`` (Finn et al., DSAE 2016)
    without any robomimic dependency.

    Architecture
    ------------
    Input  ``[B, C, H, W]``
        └─ Conv2d(C, num_kp, 1)  ← trainable channel mixer (only when num_kp ≠ C)
        └─ reshape → [B*num_kp, H*W]
        └─ softmax / temperature → attention weights
        └─ weighted sum of pos_x, pos_y meshgrid → expected (x, y) per keypoint
    Output ``[B, num_kp, 2]``   (x ∈ [−1,1], y ∈ [−1,1])

    Parameters
    ----------
    input_shape:
        ``[C, H, W]`` of the incoming feature map.
    num_kp:
        Number of keypoints (output channels). A ``Conv2d(C, num_kp, 1)``
        is added when ``num_kp`` is given; otherwise C channels are used directly.
    temperature:
        Softmax temperature τ — lower values make the distribution sharper.
    learnable_temperature:
        If ``True``, τ is an ``nn.Parameter`` updated by backprop.
    output_variance:
        If ``True``, also return per-keypoint covariance (not used by
        ``resnet18convFeaturesExtractor``; kept for interface parity).
    noise_std:
        Gaussian noise added to keypoints **during training only** (set 0 to disable).
    """

    def __init__(
        self,
        input_shape: List[int],
        num_kp: int = 32,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        output_variance: bool = False,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()

        assert len(input_shape) == 3, "input_shape must be [C, H, W]"
        self._in_c, self._in_h, self._in_w = input_shape  # e.g. 512, 3, 3

        # ── 1×1 conv: [B, C, H, W] → [B, num_kp, H, W] ──────────────────────────
        # Trainable channel-mixing keypoint extractor.
        if num_kp is not None:
            self.nets: Optional[nn.Module] = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp: int = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c

        self.output_variance = output_variance
        self.noise_std = noise_std

        # ── Temperature (scalar) ─────────────────────────────────────────────────
        _t = torch.ones(1) * temperature
        if learnable_temperature:
            self.register_parameter("temperature", nn.Parameter(_t, requires_grad=True))
        else:
            self.register_buffer("temperature", nn.Parameter(_t, requires_grad=False))

        # ── Pre-computed (x, y) position grids ───────────────────────────────────
        # pos_x / pos_y: [1, H*W]  with values linearly spaced in [−1, 1]
        pos_x_np, pos_y_np = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w),   # x axis  (width)
            np.linspace(-1.0, 1.0, self._in_h),   # y axis  (height)
        )
        # Flatten spatial dims and register as non-trainable buffers
        pos_x = torch.from_numpy(pos_x_np.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y_np.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    # ------------------------------------------------------------------
    def output_shape(self, input_shape: List[int]) -> List[int]:
        """Return output shape ``[num_kp, 2]`` (excluding batch dim)."""
        return [self._num_kp, 2]

    # ------------------------------------------------------------------
    def forward(
        self, feature: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute spatial keypoints from a feature map.

        Args:
            feature: ``[B, C, H, W]``

        Returns:
            keypoints: ``[B, num_kp, 2]``  (or tuple with covariance if
            ``output_variance=True``)
        """
        # ── shape guards ──────────────────────────────────────────────────────────
        assert feature.shape[1] == self._in_c, (
            f"Expected {self._in_c} input channels, got {feature.shape[1]}"
        )
        assert feature.shape[2] == self._in_h and feature.shape[3] == self._in_w, (
            f"Expected spatial size ({self._in_h},{self._in_w}), "
            f"got ({feature.shape[2]},{feature.shape[3]})"
        )

        # ── optional channel reduction ────────────────────────────────────────────
        # [B, C, H, W] → [B, num_kp, H, W]
        if self.nets is not None:
            feature = self.nets(feature)

        B = feature.shape[0]

        # ── flatten spatial dims ──────────────────────────────────────────────────
        # [B, num_kp, H, W] → [B*num_kp, H*W]
        feature = feature.reshape(-1, self._in_h * self._in_w)

        # ── softmax attention ─────────────────────────────────────────────────────
        # attention: [B*num_kp, H*W]  (each row sums to 1)
        attention = F.softmax(feature / self.temperature, dim=-1)

        # ── expected (x, y) position per keypoint ────────────────────────────────
        # pos_x / pos_y: [1, H*W] broadcast over [B*num_kp, H*W]
        # expected_x/y : [B*num_kp, 1]
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)

        # [B*num_kp, 2] → [B, num_kp, 2]
        expected_xy = torch.cat([expected_x, expected_y], dim=1)
        keypoints = expected_xy.view(B, self._num_kp, 2)   # [B, num_kp, 2]

        # ── optional training noise ───────────────────────────────────────────────
        if self.training and self.noise_std > 0.0:
            keypoints = keypoints + torch.randn_like(keypoints) * self.noise_std

        # ── optional variance output ──────────────────────────────────────────────
        if self.output_variance:
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy_cov = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x  = expected_xx - expected_x * expected_x
            var_y  = expected_yy - expected_y * expected_y
            var_xy = expected_xy_cov - expected_x * expected_y
            # [B*num_kp, 4] → [B, num_kp, 2, 2]
            covariance = torch.cat([var_x, var_xy, var_xy, var_y], dim=1).reshape(
                B, self._num_kp, 2, 2
            )
            return keypoints, covariance

        return keypoints   # [B, num_kp, 2]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_shape=[{self._in_c},{self._in_h},{self._in_w}], "
            f"num_kp={self._num_kp}, "
            f"temperature={self.temperature.item():.3f}, "
            f"noise_std={self.noise_std})"
        )
