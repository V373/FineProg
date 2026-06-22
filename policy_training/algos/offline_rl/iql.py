"""Implicit Q-Learning offline algorithm."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import math

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.distributions import TanhBijector
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from .base_offline_rl import OfflineRLBase
from models.feature_extractors import FlatRangeFeaturesExtractor, resnet18convFeaturesExtractor
from models.policies import CustomMlpPolicy
from models.value_critic import ValueCritic


def _ns_to_plain(obj: Any) -> Any:
    """Recursively convert ``SimpleNamespace`` objects to plain Python dicts / lists."""
    if hasattr(obj, "__dict__") and type(obj).__name__ == "SimpleNamespace":
        return {k: _ns_to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_ns_to_plain(v) for v in obj]
    return obj


class IQL(OfflineRLBase):
    """Standalone IQL trainer built on top of custom SB3 policy modules."""

    def _setup_model(self) -> None:
        lr_vector = getattr(self.cfg, "learning_rates", None)
        if lr_vector is None:
            # Backward compatibility with older single-LR configs.
            fallback_lr = float(getattr(self.cfg, "learning_rate", 3e-4))
            actor_lr, critic_lr, v_lr = fallback_lr, fallback_lr, fallback_lr
        else:
            if not isinstance(lr_vector, (list, tuple)) or len(lr_vector) != 3:
                raise ValueError(
                    "learning_rates must be a 3-element list/tuple: "
                    "[actor_lr, critic_lr, v_lr]."
                )
            actor_lr = float(lr_vector[0])
            critic_lr = float(lr_vector[1])
            v_lr = float(lr_vector[2])

        self.lr_schedule = lambda _: actor_lr

        # ── Select feature extractor ──────────────────────────────────────────────
        extractor_type = str(getattr(self.cfg, "features_extractor_type", "flat_range"))

        if extractor_type == "flat_range":
            # Default: project the entire flat obs vector through a single linear layer.
            obs_dim = int(self.observation_space.shape[0])  # [B, obs_dim]
            extractor_class = FlatRangeFeaturesExtractor
            extractor_kwargs: Dict[str, Any] = {
                "dim_ranges": [obs_dim],
                "projection_dims": [obs_dim],
                "normalize_images": True,
            }
        elif extractor_type == "resnet18conv":
            # Post-backbone stack for precomputed ResNet18Conv feature maps + low-dim obs.
            # obs_slices is injected at runtime by train_policy.py after buffer construction.
            fe_cfg = getattr(self.cfg, "features_extractor_kwargs", None)
            if fe_cfg is None:
                raise ValueError(
                    "features_extractor_type='resnet18conv' requires "
                    "'features_extractor_kwargs' to be present in the iql config."
                )
            fe_dict = _ns_to_plain(fe_cfg)
            extractor_class = resnet18convFeaturesExtractor
            extractor_kwargs = {
                # {key: (start, end)} injected from replay_buffer.obs_slices by train_policy.py
                "obs_slices": fe_dict["obs_slices"],
                # e.g. ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
                "low_dim_keys": fe_dict["low_dim_keys"],
                # {key: {input_shape:[C,H,W], num_kp, temperature, feature_dimension}}
                "visual_specs": fe_dict["visual_specs"],
                "normalize_images": fe_dict.get("normalize_images", False),
            }
        else:
            raise ValueError(
                f"Unknown features_extractor_type: '{extractor_type}'. "
                f"Supported values: 'flat_range', 'resnet18conv'."
            )

        policy_kwargs = {
            "net_arch": {
                "pi": list(getattr(self.cfg, "pi_net_arch", [256, 256])),
                "qf": list(getattr(self.cfg, "qf_net_arch", [256, 256])),
            },
            "n_critics": int(getattr(self.cfg, "n_critics", 2)),
            "policy_layer_norm": bool(getattr(self.cfg, "policy_layer_norm", False)),
            "critic_layer_norm": bool(getattr(self.cfg, "critic_layer_norm", False)),
            "actor_squash_output": bool(getattr(self.cfg, "actor_squash_output", True)),
            "features_extractor_class": extractor_class,
            "features_extractor_kwargs": extractor_kwargs,
        }

        self.policy = CustomMlpPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            **policy_kwargs,
        ).to(self.device)

        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        # Keep actor/critic optimizers separate while taking LR values from config.
        for param_group in self.actor.optimizer.param_groups:
            param_group["lr"] = actor_lr
        for param_group in self.critic.optimizer.param_groups:
            param_group["lr"] = critic_lr

        self.v_net = ValueCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=list(getattr(self.cfg, "qf_net_arch", [256, 256])),
            features_extractor=deepcopy(self.policy.critic.features_extractor),
            features_dim=self.policy.actor.latent_pi[0].in_features,
            activation_fn=self.policy.net_args["activation_fn"],
            normalize_images=self.policy.critic.normalize_images,
            share_features_extractor=self.policy.critic.share_features_extractor,
            lr_schedule=lambda _: v_lr,
            optimizer_class=self.policy.optimizer_class,
            optimizer_kwargs=self.policy.optimizer_kwargs,
            use_layer_norm=bool(getattr(self.cfg, "critic_layer_norm", False)),
        ).to(self.device)

        for param_group in self.v_net.optimizer.param_groups:
            param_group["lr"] = v_lr

        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        self.gamma = float(getattr(self.cfg, "gamma", 0.99))
        self.tau = float(getattr(self.cfg, "tau", 0.005))
        self.target_update_interval = int(getattr(self.cfg, "target_update_interval", 1))

        self.advantage_temp = float(getattr(self.cfg, "advantage_temp", 5.0))
        self.expectile = float(getattr(self.cfg, "expectile", 0.7))
        self.clip_score = float(getattr(self.cfg, "clip_score", 100.0))
        self.policy_extraction = str(getattr(self.cfg, "policy_extraction", "awr"))
        self.ddpg_bc_weight = float(getattr(self.cfg, "ddpg_bc_weight", 0.1))

        self.offline_critic_update_ratio = int(getattr(self.cfg, "offline_critic_update_ratio", 1))
        self.current_critic_update_ratio = self.offline_critic_update_ratio

        self.n_critics = int(getattr(self.cfg, "n_critics", 2))
        self.n_critics_to_sample = int(getattr(self.cfg, "n_critics_to_sample", min(2, self.n_critics)))

        self.actor_debug = bool(getattr(self.cfg, "actor_debug", False))

    def _get_log_prob(self, distribution, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() > 2:
            actions = actions.reshape(actions.shape[0], -1)
        return distribution.log_prob(actions)

    def train_step(self, batch) -> Dict[str, float]:
        self.policy.set_training_mode(True)

        actor_losses = []
        q_losses = []
        v_losses = []
        actor_log_pis = []
        actor_gaussian_log_probs = []
        actor_tanh_corrections = []
        advantage_values = []
        advantage_weight_values = []
        q_values = []
        v_next_values = []
        v_values = []
        q_target_values = []
        reward_values = []

        obs = batch.observations
        actions = batch.actions
        next_obs = batch.next_observations
        rewards = batch.rewards
        dones = batch.dones

        for _ in range(int(self.current_critic_update_ratio)):
            # Step 1: Update critic and value networks
            q_preds = torch.cat(self.critic(obs, actions), dim=1)
            with torch.no_grad():
                critic_indices = torch.randperm(self.n_critics, device=obs.device)[: self.n_critics_to_sample]
                target_q_preds = torch.cat(
                    self.critic_target(obs, actions, critic_indices=critic_indices), dim=1
                )
                target_q_pred, _ = torch.min(target_q_preds, dim=1)
                target_q_pred = target_q_pred.reshape(-1, 1)
                next_vf_pred = self.v_net(next_obs)

            vf_pred = self.v_net(obs)
            target_q_values = rewards + (1.0 - dones) * self.gamma * next_vf_pred
            q_loss = F.mse_loss(q_preds, target_q_values.expand_as(q_preds))

            vf_err = vf_pred - target_q_pred
            vf_sign = (vf_err > 0).float()
            vf_weight = (1.0 - vf_sign) * self.expectile + vf_sign * (1.0 - self.expectile)
            vf_loss = (vf_weight * (vf_err ** 2)).mean()

            self.critic.optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
            self.critic.optimizer.step()

            self.v_net.optimizer.zero_grad()
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), max_norm=10.0)
            self.v_net.optimizer.step()

            # Step 2: Update target networks
            if self.global_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            q_values.append(float(q_preds.mean().item()))
            v_values.append(float(vf_pred.mean().item()))
            q_target_values.append(float(target_q_values.mean().item()))
            v_next_values.append(float(next_vf_pred.mean().item()))
            q_losses.append(float(q_loss.item()))
            v_losses.append(float(vf_loss.item()))

        # Step 3: Update actor policy with advantage-weighted regression or DDPG-style updates
        if self.policy_extraction == "awr":
            advantage = target_q_pred - vf_pred.detach()
            weights = torch.clamp(torch.exp(advantage * self.advantage_temp), 0, self.clip_score)
            advantage_values.append(float(advantage.mean().item()))
            advantage_weight_values.append(float(weights.mean().item()))
            mean_actions, log_std, _ = self.actor.get_action_dist_params(obs)
            distribution = self.actor.action_dist.proba_distribution(mean_actions, log_std)
            log_prob = self._get_log_prob(distribution, actions).reshape(-1, 1)
            z_scaled_error_mean = None
            std_error_mean = None
            if self.actor_debug:
                with torch.no_grad():
                    gaussian_actions = TanhBijector.inverse(actions)
                    gaussian_log_prob = distribution.distribution.log_prob(gaussian_actions).sum(dim=1)
                    tanh_correction = -torch.sum(
                        torch.log(1.0 - torch.tanh(gaussian_actions).pow(2) + distribution.epsilon),
                        dim=1,
                    )
                    std = torch.exp(log_std)
                    z_scaled_error = (((gaussian_actions - mean_actions) / std).pow(2)).sum(dim=-1)
                    std_error = log_std
                actor_gaussian_log_probs.append(float(gaussian_log_prob.mean().item()))
                actor_tanh_corrections.append(float(tanh_correction.mean().item()))
                z_scaled_error_mean = z_scaled_error.mean().item()
                std_error_mean = std_error.mean().item()
            policy_loss = -torch.mean(weights * log_prob)
        elif self.policy_extraction == "ddpg":
            with torch.no_grad():
                average_q_value = torch.abs(torch.min(q_preds, dim=1).values).mean()
                scaled_ddpg_bc_weight = self.ddpg_bc_weight / (average_q_value + 1e-8)

            mean_actions, log_std, _ = self.actor.get_action_dist_params(obs)
            distribution = self.actor.action_dist.proba_distribution(mean_actions, log_std)
            log_prob = self._get_log_prob(distribution, actions)
            actions_pi = distribution.actions_from_params(mean_actions, log_std)

            critic_indices = torch.randperm(self.n_critics, device=obs.device)[: self.n_critics_to_sample]
            q_values_pi = self.critic(obs, actions_pi, critic_indices=critic_indices)
            min_qf_pi = torch.min(torch.cat(q_values_pi, dim=1), dim=1).values
            if min_qf_pi.shape != log_prob.shape:
                log_prob = log_prob.reshape_as(min_qf_pi)
            policy_loss = -torch.mean(min_qf_pi + scaled_ddpg_bc_weight * log_prob)
        else:
            raise ValueError(f"Unsupported policy_extraction: {self.policy_extraction}")

        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        reward_values.append(float(rewards.mean().item()))
        actor_losses.append(float(policy_loss.item()))
        actor_log_pis.append(float(log_prob.mean().item()))

        metrics = {
            "train/actor_loss": float(np.mean(actor_losses)),
            "train/q_loss": float(np.mean(q_losses)),
            "train/v_loss": float(np.mean(v_losses)),
            "train/average_q_values": float(np.mean(q_values)),
            # "train/average_v_next_values": float(np.mean(v_next_values)),
            "train/average_reward": float(np.mean(reward_values)),
            "train/average_v_values": float(np.mean(v_values)),
            "train/q_target": float(np.mean(q_target_values)),
            "train/average_actor_log_prob": float(np.mean(actor_log_pis)),
        }
        if advantage_values:
            metrics["train/average_advantage"] = float(np.mean(advantage_values))
        if advantage_weight_values:
            metrics["train/average_advantage_weight"] = float(np.mean(advantage_weight_values))
        if actor_gaussian_log_probs:
            metrics["actor_diag/gaussian_log_prob_mean"] = float(np.mean(actor_gaussian_log_probs))
        if actor_tanh_corrections:
            metrics["actor_diag/tanh_correction_mean"] = float(np.mean(actor_tanh_corrections))
        if self.actor_debug:
            if z_scaled_error_mean is not None:
                metrics["actor_diag/z_scaled_error_mean"] = float(z_scaled_error_mean)
            if std_error_mean is not None:
                metrics["actor_diag/std_error_mean"] = float(std_error_mean)

        return metrics

    # save and load functions for checkpointing: including model and optimizer parameters
    def _module_state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "policy": self.policy.state_dict(),
            "v_net": self.v_net.state_dict(),
        }

    def _optimizer_state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "actor_optimizer": self.actor.optimizer.state_dict(),
            "critic_optimizer": self.critic.optimizer.state_dict(),
            "v_optimizer": self.v_net.optimizer.state_dict(),
        }

    def _load_module_state_dict(self, modules: Dict[str, Dict[str, Any]]) -> None:
        if "policy" in modules:
            self.policy.load_state_dict(modules["policy"])
        if "v_net" in modules:
            self.v_net.load_state_dict(modules["v_net"])

    def _load_optimizer_state_dict(self, optimizers: Dict[str, Dict[str, Any]]) -> None:
        if "actor_optimizer" in optimizers:
            self.actor.optimizer.load_state_dict(optimizers["actor_optimizer"])
        if "critic_optimizer" in optimizers:
            self.critic.optimizer.load_state_dict(optimizers["critic_optimizer"])
        if "v_optimizer" in optimizers:
            self.v_net.optimizer.load_state_dict(optimizers["v_optimizer"])
