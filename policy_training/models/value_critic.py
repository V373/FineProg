"""Value critic module used by IQL."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from .policies import create_mlp


class ValueCritic(BaseModel):
    """Single value network V(s) for IQL."""

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        share_features_extractor: bool = False,
        lr_schedule: Optional[Schedule] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_layer_norm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if lr_schedule is None:
            lr_schedule = lambda _: 3e-4
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.share_features_extractor = share_features_extractor
        v_net_list = create_mlp(
            features_dim,
            1,
            net_arch,
            activation_fn,
            use_layer_norm=use_layer_norm,
        )
        self.vf = nn.Sequential(*v_net_list)
        optimizer_params = list(self.vf.parameters())
        if not self.share_features_extractor:
            # Train the value feature extractor together with the V head.
            optimizer_params.extend(self.features_extractor.parameters())
        self.optimizer = optimizer_class(
            optimizer_params,
            lr=lr_schedule(1.0),
            **optimizer_kwargs,
        )

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        value_input = th.cat([features], dim=1)
        return self.vf(value_input)
