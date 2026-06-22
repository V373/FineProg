"""Online SAC trainer for robomimic online interaction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from algos.online_rl.base_online_rl import OnlineRLBase
from models.feature_extractors import FlatRangeFeaturesExtractor, resnet18convFeaturesExtractor
from models.policies import CustomMlpPolicy


def _ns_to_plain(obj: Any) -> Any:
	"""Recursively convert SimpleNamespace objects to plain Python containers."""
	if isinstance(obj, SimpleNamespace):
		return {k: _ns_to_plain(v) for k, v in vars(obj).items()}
	if isinstance(obj, dict):
		return {str(k): _ns_to_plain(v) for k, v in obj.items()}
	if isinstance(obj, list):
		return [_ns_to_plain(v) for v in obj]
	return obj


class OnlineSAC(OnlineRLBase):
	"""Standalone online SAC trainer built on the same policy modules as IQL."""

	def _setup_model(self) -> None:
		lr_vector = getattr(self.cfg, "learning_rates", None)
		if lr_vector is None:
			fallback_lr = float(getattr(self.cfg, "learning_rate", 3e-4))
			actor_lr, critic_lr, alpha_lr = fallback_lr, fallback_lr, fallback_lr
		else:
			if not isinstance(lr_vector, (list, tuple)) or len(lr_vector) != 3:
				raise ValueError(
					"learning_rates must be a 3-element list/tuple: "
					"[actor_lr, critic_lr, alpha_lr]."
				)
			actor_lr = float(lr_vector[0])
			critic_lr = float(lr_vector[1])
			alpha_lr = float(lr_vector[2])

		self.lr_schedule = lambda _: actor_lr

		extractor_type = str(getattr(self.cfg, "features_extractor_type", "flat_range"))
		if extractor_type == "flat_range":
			obs_dim = int(self.observation_space.shape[0])
			extractor_class = FlatRangeFeaturesExtractor
			extractor_kwargs: Dict[str, Any] = {
				"dim_ranges": [obs_dim],
				"projection_dims": [obs_dim],
				"normalize_images": True,
			}
		elif extractor_type == "resnet18conv":
			fe_cfg = getattr(self.cfg, "features_extractor_kwargs", None)
			if fe_cfg is None:
				raise ValueError(
					"features_extractor_type='resnet18conv' requires "
					"features_extractor_kwargs to be present in the sac config."
				)
			fe_dict = _ns_to_plain(fe_cfg)
			extractor_class = resnet18convFeaturesExtractor
			extractor_kwargs = {
				"obs_slices": fe_dict["obs_slices"],
				"low_dim_keys": fe_dict["low_dim_keys"],
				"visual_specs": fe_dict["visual_specs"],
				"normalize_images": fe_dict.get("normalize_images", False),
			}
		else:
			raise ValueError(
				f"Unknown features_extractor_type: {extractor_type!r}. "
				"Supported values: 'flat_range', 'resnet18conv'."
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

		for param_group in self.actor.optimizer.param_groups:
			param_group["lr"] = actor_lr
		for param_group in self.critic.optimizer.param_groups:
			param_group["lr"] = critic_lr

		self.gamma = float(getattr(self.cfg, "gamma", 0.99))
		self.tau = float(getattr(self.cfg, "tau", 0.005))
		self.target_update_interval = int(getattr(self.cfg, "target_update_interval", 1))
		self.n_critics = int(getattr(self.cfg, "n_critics", 2))
		self.n_critics_to_sample = int(getattr(self.cfg, "n_critics_to_sample", min(2, self.n_critics)))
		self.target_critic_reduction = str(getattr(self.cfg, "target_critic_reduction", "mean"))
		if self.target_critic_reduction not in {"mean", "min"}:
			raise ValueError("target_critic_reduction must be either 'mean' or 'min'.")
		self.train_critic_with_entropy = bool(getattr(self.cfg, "train_critic_with_entropy", True))

		self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
		self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

		action_dim = int(np.prod(self.action_space.shape))
		target_entropy = getattr(self.cfg, "target_entropy", "auto")
		self.target_entropy = -float(action_dim) if str(target_entropy) == "auto" else float(target_entropy)
		self.ent_coef = getattr(self.cfg, "ent_coef", "auto")
		self.log_ent_coef: Optional[torch.Tensor] = None
		self.ent_coef_optimizer: Optional[torch.optim.Optimizer] = None
		if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
			init_value = 1.0
			if "_" in self.ent_coef:
				init_value = float(self.ent_coef.split("_", 1)[1])
				if init_value <= 0.0:
					raise ValueError("The initial ent_coef value must be > 0.")
			self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
			self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=alpha_lr)
		else:
			self.ent_coef_tensor = torch.tensor(float(self.ent_coef), device=self.device)

	def _current_ent_coef(self) -> torch.Tensor:
		if self.log_ent_coef is not None:
			return torch.exp(self.log_ent_coef.detach())
		return self.ent_coef_tensor

	def _reduce_target_q(self, q_values: torch.Tensor) -> torch.Tensor:
		if self.target_critic_reduction == "min":
			return torch.min(q_values, dim=1, keepdim=True).values
		return torch.mean(q_values, dim=1, keepdim=True)

	def train_step(self, batch: Any) -> Dict[str, float]:
		self.policy.set_training_mode(True)

		obs = batch.observations
		actions = batch.actions
		next_obs = batch.next_observations
		rewards = batch.rewards
		dones = batch.dones

		if rewards.dim() == 1:
			rewards = rewards.reshape(-1, 1)
		if dones.dim() == 1:
			dones = dones.reshape(-1, 1)

		with torch.no_grad():
			next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
			next_log_prob = next_log_prob.reshape(-1, 1)
			critic_indices = torch.randperm(self.n_critics, device=obs.device)[: self.n_critics_to_sample]
			next_q_values = torch.cat(
				self.critic_target(next_obs, next_actions, critic_indices=critic_indices),
				dim=1,
			)
			target_q = self._reduce_target_q(next_q_values)
			ent_coef = self._current_ent_coef()
			if self.train_critic_with_entropy:
				target_q = target_q - ent_coef * next_log_prob
			target_q_values = rewards + (1.0 - dones) * self.gamma * target_q

		current_q_values = torch.cat(self.critic(obs, actions), dim=1)
		critic_loss = F.mse_loss(current_q_values, target_q_values.expand_as(current_q_values))
		self.critic.optimizer.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
		self.critic.optimizer.step()

		if self.global_step % self.target_update_interval == 0:
			polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
			polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

		actions_pi, log_prob = self.actor.action_log_prob(obs)
		log_prob = log_prob.reshape(-1, 1)

		ent_coef_loss = None
		if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
			ent_coef_loss = -(
				self.log_ent_coef * (log_prob + self.target_entropy).detach()
			).mean()
			self.ent_coef_optimizer.zero_grad()
			ent_coef_loss.backward()
			self.ent_coef_optimizer.step()

		ent_coef = self._current_ent_coef()
		q_values_pi = torch.cat(self.critic(obs, actions_pi), dim=1)
		mean_qf_pi = torch.mean(q_values_pi, dim=1, keepdim=True)
		actor_loss = (ent_coef * log_prob - mean_qf_pi).mean()
		self.actor.optimizer.zero_grad()
		actor_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
		self.actor.optimizer.step()

		metrics = {
			"train/actor_loss": float(actor_loss.item()),
			"train/critic_loss": float(critic_loss.item()),
			"train/ent_coef": float(ent_coef.item()),
			"train/average_q_values": float(current_q_values.mean().item()),
			"train/average_q_next_values": float(next_q_values.mean().item()),
			"train/average_reward": float(rewards.mean().item()),
			"train/average_actor_log_prob": float(log_prob.mean().item()),
		}
		if ent_coef_loss is not None:
			metrics["train/ent_coef_loss"] = float(ent_coef_loss.item())
		return metrics

	def load_offline_checkpoint(self, *_args: Any, **_kwargs: Any) -> None:
		raise NotImplementedError("online_sac offline_warmstart is reserved for a later port.")

	def set_mixed_buffer(self, *_args: Any, **_kwargs: Any) -> None:
		raise NotImplementedError("online_sac mixed offline/online replay is reserved for a later port.")

	def set_reward_provider(self, *_args: Any, **_kwargs: Any) -> None:
		raise NotImplementedError("online_sac learned reward inference is reserved for a later port.")

	def _module_state_dict(self) -> Dict[str, Dict[str, Any]]:
		modules: Dict[str, Dict[str, Any]] = {"policy": self.policy.state_dict()}
		if self.log_ent_coef is not None:
			modules["ent_coef"] = {"log_ent_coef": self.log_ent_coef.detach().clone()}
		return modules

	def _optimizer_state_dict(self) -> Dict[str, Dict[str, Any]]:
		optimizers = {
			"actor_optimizer": self.actor.optimizer.state_dict(),
			"critic_optimizer": self.critic.optimizer.state_dict(),
		}
		if self.ent_coef_optimizer is not None:
			optimizers["ent_coef_optimizer"] = self.ent_coef_optimizer.state_dict()
		return optimizers

	def _load_module_state_dict(self, modules: Dict[str, Dict[str, Any]]) -> None:
		if "policy" in modules:
			self.policy.load_state_dict(modules["policy"])
		if "ent_coef" in modules and self.log_ent_coef is not None:
			stored = modules["ent_coef"].get("log_ent_coef")
			if stored is not None:
				self.log_ent_coef.data.copy_(stored.to(self.device))

	def _load_optimizer_state_dict(self, optimizers: Dict[str, Dict[str, Any]]) -> None:
		if "actor_optimizer" in optimizers:
			self.actor.optimizer.load_state_dict(optimizers["actor_optimizer"])
		if "critic_optimizer" in optimizers:
			self.critic.optimizer.load_state_dict(optimizers["critic_optimizer"])
		if "ent_coef_optimizer" in optimizers and self.ent_coef_optimizer is not None:
			self.ent_coef_optimizer.load_state_dict(optimizers["ent_coef_optimizer"])