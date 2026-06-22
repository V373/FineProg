"""Base class for online RL algorithms."""

from __future__ import annotations

from typing import Any, Dict, Optional

from algos.offline_rl.base_offline_rl import OfflineRLBase


class OnlineRLBase(OfflineRLBase):
	"""Online RL base class with a built-in interaction and training loop."""

	def learn_online(
		self,
		collector: Any,
		replay_buffer: Any,
		n_steps: int,
		batch_size: int,
		learning_starts: int,
		train_freq: int,
		gradient_steps: int,
		log_every: int,
		save_every: int,
		save_dir: str,
		logger: Optional[Any] = None,
		checkpoint_metadata: Optional[Dict[str, Any]] = None,
		rollout_evaluator: Optional[Any] = None,
	) -> None:
		from tqdm import tqdm

		if checkpoint_metadata is not None:
			self.set_checkpoint_metadata(checkpoint_metadata)

		n_steps = int(n_steps)
		batch_size = int(batch_size)
		learning_starts = int(learning_starts)
		train_freq = max(1, int(train_freq))
		gradient_steps = int(gradient_steps)
		log_every = max(1, int(log_every))
		save_every = max(1, int(save_every))

		collector.reset()
		pbar = tqdm(range(1, n_steps + 1), desc="Online Training", dynamic_ncols=True)
		for env_step in pbar:
			rollout_metrics: Dict[str, float] = collector.collect_step(learning_starts=learning_starts)

			train_metrics: Dict[str, float] = {}
			should_train = env_step % train_freq == 0 or env_step == n_steps
			if should_train and replay_buffer.can_sample(batch_size) and replay_buffer.size() >= learning_starts:
				for _ in range(gradient_steps):
					batch = replay_buffer.sample(batch_size)
					train_metrics = self.train_step(batch)
					self.global_step += 1

			if logger is not None:
				if train_metrics:
					logger.record_dict(train_metrics)
				if rollout_metrics:
					logger.record_dict(rollout_metrics)

			_eval_cfg: Dict[str, Any] = {}
			if rollout_evaluator is not None:
				_eval_cfg = getattr(rollout_evaluator, "eval_cfg", {}) or {}

			if _eval_cfg:
				eval_enabled = bool(_eval_cfg.get("enabled", False))
				eval_every = max(1, int(_eval_cfg.get("every_n_steps", 1)))
				warmstart_steps = max(0, int(_eval_cfg.get("warmstart_steps", 0)))

				should_eval = (
					eval_enabled
					and env_step >= warmstart_steps
					and (env_step % eval_every == 0)
				)
				if should_eval:
					modules = {}
					for name in ["policy", "actor", "critic", "critic_target", "v_net"]:
						module = getattr(self, name, None)
						if module is not None and hasattr(module, "training"):
							modules[name] = bool(module.training)
					try:
						if hasattr(self, "policy") and hasattr(self.policy, "set_training_mode"):
							self.policy.set_training_mode(False)
						for name in modules:
							getattr(self, name).eval()

						eval_metrics: Dict[str, Any] = rollout_evaluator.run(global_step=int(env_step))
					finally:
						if hasattr(self, "policy") and hasattr(self.policy, "set_training_mode"):
							self.policy.set_training_mode(bool(modules.get("policy", True)))
						for name, was_training in modules.items():
							module = getattr(self, name)
							if was_training:
								module.train()
							else:
								module.eval()

					if logger is not None:
						if eval_metrics:
							logger.record_dict(eval_metrics)
						for idx, video_path in enumerate(getattr(rollout_evaluator, "last_video_paths", [])):
							key = "eval/video" if idx == 0 else f"eval/video_{idx}"
							logger.record_video(key=key, path=video_path, step=env_step)
						progress_path = getattr(rollout_evaluator, "last_progress_path", None)
						if progress_path and hasattr(logger, "record_image"):
							logger.record_image(key="eval/progress", path=progress_path, step=env_step)
						logger.dump(step=env_step)

			if logger is not None and env_step % log_every == 0:
				logger.dump(step=max(self.global_step, env_step))
				postfix = {k: f"{v:.4f}" for k, v in train_metrics.items() if isinstance(v, (int, float))}
				if postfix:
					pbar.set_postfix(postfix)

			if env_step % save_every == 0:
				self.save(save_dir, f"step_{max(self.global_step, env_step)}")

		self.save(save_dir, "final")