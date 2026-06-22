"""Feature extractor modules for policy networks."""

from __future__ import annotations

from typing import Dict, List, Tuple, Type

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FlatRangeFeaturesExtractor(BaseFeaturesExtractor):
    """Project contiguous observation ranges into a concatenated latent vector.

    Splits a flat observation vector into contiguous intervals, projects each
    interval through an independent linear layer (``nn.Linear``), and
    concatenates the results into a fixed-dimensional latent vector for
    downstream policy / value networks.

    Data dimensions
    ---------------
    Input ``observations``:
        A 2D tensor of shape ``(batch_size, D_in)``, where
        ``D_in = sum(dim_ranges)``. The observation is laid out as a single
        contiguous vector in feature order, and ``dim_ranges`` describes how
        to slice it into several contiguous sub-segments.

    Output ``features``:
        A 2D tensor of shape ``(batch_size, D_out)``, where
        ``D_out = sum(projection_dims)``. Each sub-segment is projected
        independently and concatenated along ``dim=1``.

    Network dimensions
    ------------------
    - ``dim_ranges``: a list of length ``n`` that describes how the input
      vector is split, e.g. ``[10, 6, 3]`` means a 19-dimensional
      observation is cut into 3 segments of lengths 10 / 6 / 3.
    - ``projection_dims``: a list of length ``n`` that describes the output
      dimension of each segment's projection, e.g. ``[64, 32, 16]``. The
      final feature dimension is their sum (112).
    - Each sub-segment maps to a single ``nn.Linear(in_dim, out_dim)`` (a
      plain linear layer with no hidden layers; if a more complex structure
      is needed, wrap it in a larger module externally).

    Example
    -------
    ``dim_ranges=[10, 6, 3]``, ``projection_dims=[64, 32, 16]``::

        Input observation x:  (B, 19) = (B, 10+6+3)
            ├─ x[:,  0:10] ──► Linear(10, 64) ──► (B, 64)
            ├─ x[:, 10:16] ──► Linear( 6, 32) ──► (B, 32)
            └─ x[:, 16:19] ──► Linear( 3, 16) ──► (B, 16)
        Concatenated output features:  (B, 112) = (B, 64+32+16)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        dim_ranges: List[int],
        projection_dims: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU, # Currently unused
        normalize_images: bool = True,
    ):
        if len(dim_ranges) != len(projection_dims):
            raise ValueError("dim_ranges and projection_dims must have the same length")

        self.activation_fn = activation_fn
        self.normalize_images = normalize_images
        self.dim_ranges = dim_ranges
        self.projection_dims = projection_dims

        super().__init__(observation_space, sum(projection_dims))

        self.projectors = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dim_ranges, projection_dims)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = []
        start = 0
        for in_dim, projector in zip(self.dim_ranges, self.projectors):
            end = start + in_dim
            projected.append(projector(x[:, start:end]))
            start = end
        return torch.cat(projected, dim=1)


class resnet18convFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for precomputed ResNet18Conv feature maps + low-dim observations.

    Mirrors the post-backbone trainable stack of robomimic ``VisualCore``—without
    the ResNet18Conv backbone itself—directly consuming precomputed conv feature
    maps stored flat inside the replay buffer’s observation vector.

    Data flow
    ---------
    Input ``x``:
        Flat observation tensor ``(B, D_total)`` from ``RobomimicReplayBuffer``.
        ``obs_slices`` maps every obs key to its ``(start, end)`` position.

    Low-dim branch (each key in ``low_dim_keys``, in order)::

        x[:, start:end]  →  [B, D_key]          (identity passthrough)

    Visual branch (each key in ``visual_specs``, in order)::

        x[:, start:end]                              [B, C*H*W]   ← flat ResNet18 feat
            └─ .reshape(B, C, H, W)                [B, C, H, W]
            └─ SpatialSoftmax(input_shape=[C,H,W]) [B, num_kp, 2]  ← (x,y) keypoint coords
            └─ nn.Flatten(start_dim=1)             [B, num_kp * 2]
            └─ nn.Linear(num_kp*2, feat_dim)       [B, feature_dimension]

    Concatenated output::

        cat([low_dim_0, …, low_dim_n, visual_0, …, visual_m], dim=1)
        →  [B, features_dim]
        features_dim = Σ D_key  +  Σ feature_dimension

    Trainable parameters
    --------------------
    Per visual key:

    * ``SpatialSoftmax.nets``: ``Conv2d(C, num_kp, kernel_size=1)``  (channel-mixing 1×1 conv)
    * ``nn.Linear(num_kp*2, feature_dimension)``                     (keypoint projection)

    No ResNet18Conv parameters exist in this module.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        obs_slices: Dict[str, Tuple[int, int]],
        low_dim_keys: List[str],
        visual_specs: Dict[str, dict],
        normalize_images: bool = False,
    ):
        """
        Args:
            observation_space: Flat ``spaces.Box`` as output by ``RobomimicReplayBuffer``.
            obs_slices: ``{obs_key: (start, end)}`` — offsets into the flat obs vector.
                Must contain entries for every key in ``low_dim_keys`` and ``visual_specs``.
                Populated at runtime by ``train_policy.py`` from ``replay_buffer.obs_slices``.
            low_dim_keys: Ordered list of low-dim obs keys passed through unchanged.
                E.g. ``["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]``.
            visual_specs: Ordered ``dict`` mapping visual obs key → spec dict::

                    {
                        "agentview_image": {
                            "input_shape": [512, 3, 3],  # [C, H’, W’] of ResNet18 feat map
                                                         #   C=512, H’=ceil(H_img/32),
                                                         #          W’=ceil(W_img/32)
                            "num_kp": 32,                # SpatialSoftmax keypoints
                            "temperature": 1.0,          # softmax temperature
                            "feature_dimension": 64,     # linear projection output dim
                        },
                        ...
                    }

            normalize_images: Unused; kept for SB3 ``BaseFeaturesExtractor`` interface.
        """
        from models.model_utils import SpatialSoftmax as _SpatialSoftmax

        # ── Validate obs_slices coverage and spatial shape consistency ───────────
        for key in low_dim_keys:
            if key not in obs_slices:
                raise ValueError(
                    f"low_dim_key '{key}' is missing from obs_slices. "
                    f"Available keys: {list(obs_slices.keys())}"
                )
        for key, spec in visual_specs.items():
            if key not in obs_slices:
                raise ValueError(
                    f"visual_specs key '{key}' is missing from obs_slices. "
                    f"Available keys: {list(obs_slices.keys())}"
                )
            start, end = obs_slices[key]
            C, H, W = spec["input_shape"]           # e.g. C=512, H'=3, W'=3
            expected_flat: int = C * H * W           # e.g. 512*3*3 = 4608
            actual_flat: int = end - start
            if actual_flat != expected_flat:
                raise ValueError(
                    f"Visual key '{key}': obs_slices covers {actual_flat} dims but "
                    f"input_shape {spec['input_shape']} implies C×H×W = {expected_flat}. "
                    f"Ensure the HDF5 feature map shape matches input_shape in visual_specs."
                )

        # ── Compute total output dimension ─────────────────────────────────────────
        # Σ D_key  for all low-dim obs keys
        low_dim_total: int = sum(
            obs_slices[k][1] - obs_slices[k][0] for k in low_dim_keys
        )  # e.g. 3 + 4 + 2 = 9 for eef_pos + eef_quat + gripper_qpos

        # Σ feature_dimension  for all visual obs keys
        visual_total: int = sum(
            int(spec["feature_dimension"]) for spec in visual_specs.values()
        )  # e.g. 64 + 64 = 128 for two camera views

        features_dim: int = low_dim_total + visual_total  # final output dim [B, features_dim]

        super().__init__(observation_space, features_dim)

        # Persist for forward()
        self.obs_slices: Dict[str, Tuple[int, int]] = dict(obs_slices)
        self.low_dim_keys: List[str] = list(low_dim_keys)
        self._visual_specs: Dict[str, dict] = dict(visual_specs)  # preserve insertion order

        # ── Build per-visual-key sub-modules ─────────────────────────────────────
        # Each maps [B, C, H, W] → [B, feature_dimension]
        visual_modules: Dict[str, nn.Module] = {}
        for key, spec in visual_specs.items():
            C, H, W = spec["input_shape"]           # ResNet18Conv feat spatial dims
            num_kp: int = int(spec.get("num_kp", 32))
            temperature: float = float(spec.get("temperature", 1.0))
            feat_dim: int = int(spec["feature_dimension"])

            # SpatialSoftmax: [B, C, H, W] → [B, num_kp, 2]
            #   Internal Conv2d(C, num_kp, 1) is trainable (channel-mixing keypoint extractor)
            pool = _SpatialSoftmax(
                input_shape=[C, H, W],
                num_kp=num_kp,
                temperature=temperature,
                learnable_temperature=False,
                output_variance=False,   # return plain tensor, not (tensor, variance) tuple
                noise_std=0.0,
            )
            visual_modules[key] = nn.Sequential(
                pool,                                # [B, C, H, W]   → [B, num_kp, 2]
                nn.Flatten(start_dim=1),             # [B, num_kp, 2] → [B, num_kp * 2]
                nn.Linear(num_kp * 2, feat_dim),     # [B, num_kp * 2] → [B, feat_dim]
            )

        self.visual_modules = nn.ModuleDict(visual_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from all obs keys and concatenate.

        Args:
            x: Flat observation tensor of shape ``(B, D_total)``.

        Returns:
            Feature tensor of shape ``(B, features_dim)``.
        """
        # x: [B, D_total]  — flat obs from RobomimicReplayBuffer
        parts: List[torch.Tensor] = []

        # ── Low-dim branch: identity passthrough ──────────────────────────────────
        # Contributes Σ D_key dims to the output.
        for key in self.low_dim_keys:
            start, end = self.obs_slices[key]
            parts.append(x[:, start:end])           # [B, D_key]

        # ── Visual branch: reshape → SpatialSoftmax → flatten → project ──────────
        # Contributes Σ feature_dimension dims to the output.
        for key, spec in self._visual_specs.items():
            start, end = self.obs_slices[key]
            C, H, W = spec["input_shape"]
            # Recover 4-D feature map from flat slice
            feat = x[:, start:end].reshape(-1, C, H, W)    # [B, C*H*W] → [B, C, H, W]
            feat = self.visual_modules[key](feat)            # [B, C, H, W] → [B, feat_dim]
            parts.append(feat)

        # [B, low_dim_total + visual_total]  ==  [B, features_dim]
        return torch.cat(parts, dim=1)
