"""Custom SAC policy components used by IQL."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import BaseModel, BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy, get_actor_critic_arch
from torch import nn


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
    use_layer_norm: bool = False,
) -> List[nn.Module]:
    """Create an MLP with optional layer normalization."""
    modules: List[nn.Module] = []
    last_dim = input_dim

    for hidden_dim in net_arch:
        modules.append(nn.Linear(last_dim, hidden_dim, bias=with_bias))
        if use_layer_norm:
            modules.append(nn.LayerNorm(hidden_dim))
        modules.append(activation_fn())
        last_dim = hidden_dim

    if output_dim > 0:
        modules.append(nn.Linear(last_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class CustomActor(Actor):
    """SAC actor with optional layer normalization in the policy MLP."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        use_layer_norm: bool = False,
        squash_output: bool = True,
    ):
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=squash_output,
        )

        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        self.use_layer_norm = use_layer_norm
        self.actor_squash_output = squash_output

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(
            features_dim,
            -1,
            net_arch,
            activation_fn,
            use_layer_norm=use_layer_norm,
        )
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim,
                full_std=full_std,
                use_expln=use_expln,
                learn_features=True,
                squash_output=True,
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim,
                latent_sde_dim=last_layer_dim,
                log_std_init=log_std_init,
            )
            if clip_mean > 0.0:
                self.mu = nn.Sequential(
                    self.mu,
                    nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean),
                )
        elif self.actor_squash_output:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)
        else:
            # No tanh squashing: use a standard diagonal Gaussian with
            # state-independent log_std (a learnable parameter vector).
            self.action_dist = DiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Parameter(
                th.ones(action_dim) * log_std_init, requires_grad=True
            )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(dict(use_layer_norm=self.use_layer_norm, squash_output=self.actor_squash_output))
        return data


class CustomContinuousCritic(ContinuousCritic):
    """Continuous critic with optional layer normalization."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        use_layer_norm: bool = True,
    ):
        BaseModel.__init__(
            self,
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []

        for idx in range(n_critics):
            q_net = create_mlp(
                features_dim + action_dim,
                1,
                net_arch,
                activation_fn,
                use_layer_norm=use_layer_norm,
            )
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        critic_indices: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)

        if critic_indices is None:
            return tuple(q_net(qvalue_input) for q_net in self.q_networks)
        return tuple(self.q_networks[idx](qvalue_input) for idx in critic_indices)


class CustomSACPolicy(SACPolicy):
    """SAC policy that wires custom actor and critic modules."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        policy_layer_norm: bool = False,
        critic_layer_norm: bool = False,
        actor_squash_output: bool = True,
    ):
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=actor_squash_output,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs.update(
            {
                "use_sde": use_sde,
                "log_std_init": log_std_init,
                "use_expln": use_expln,
                "clip_mean": clip_mean,
                "use_layer_norm": policy_layer_norm,
                "squash_output": actor_squash_output,
            }
        )

        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
                "use_layer_norm": critic_layer_norm,
            }
        )

        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                share_features_extractor=self.share_features_extractor,
                critic_layer_norm=self.critic_kwargs["use_layer_norm"],
                policy_layer_norm=self.actor_kwargs["use_layer_norm"],
                actor_squash_output=self.actor_kwargs["squash_output"],
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)


class CustomMlpPolicy(CustomSACPolicy):
    """Alias for consistency with SB3-style policy names."""
