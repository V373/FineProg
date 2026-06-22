"""Base class for offline RL algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from utils.checkpoints import validate_checkpoint_payload


class OfflineRLBase(ABC):
    """Offline RL base class with a built-in training loop."""

    def __init__(self, observation_space, action_space, cfg, device: torch.device):
        self.observation_space = observation_space
        self.action_space = action_space
        self.cfg = cfg
        self.device = device
        self.global_step = 0
        self._checkpoint_metadata: Dict[str, Any] = {}
        self._setup_model()

    @abstractmethod
    def _setup_model(self) -> None:
        """Initialize model components and optimizers."""

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Run one update step and return metrics."""

    @abstractmethod
    def _module_state_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return module state_dict mapping."""

    @abstractmethod
    def _optimizer_state_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return optimizer state_dict mapping."""

    @abstractmethod
    def _load_module_state_dict(self, modules: Dict[str, Dict[str, Any]]) -> None:
        """Load module state_dict mapping."""

    @abstractmethod
    def _load_optimizer_state_dict(self, optimizers: Dict[str, Dict[str, Any]]) -> None:
        """Load optimizer state_dict mapping."""

    def set_checkpoint_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        """Attach extra metadata that should be saved with every checkpoint."""
        self._checkpoint_metadata = dict(metadata or {})

    def _extra_checkpoint_payload(self) -> Dict[str, Any]:
        """Return any additional checkpoint payload fields."""
        return dict(self._checkpoint_metadata)

    def save(self, save_dir: str, tag: str) -> str:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"{tag}.pt"

        payload = {
            "global_step": self.global_step,
            "modules": self._module_state_dict(),
            "optimizers": self._optimizer_state_dict(),
        }
        payload.update(self._extra_checkpoint_payload())
        validate_checkpoint_payload(payload, context="OfflineRLBase.save")
        torch.save(payload, ckpt_path)
        return str(ckpt_path)

    def load(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.global_step = int(payload.get("global_step", 0))
        self._load_module_state_dict(payload.get("modules", {}))
        self._load_optimizer_state_dict(payload.get("optimizers", {}))

    def learn_offline(
        self,
        replay_buffer,
        n_steps: int,
        batch_size: int,
        log_every: int,
        save_every: int,
        save_dir: str,
        logger: Optional[Any] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
        rollout_evaluator: Optional[Any] = None,
        value_evaluator: Optional[Any] = None,
    ) -> None:
        from tqdm import tqdm

        if checkpoint_metadata is not None:
            self.set_checkpoint_metadata(checkpoint_metadata)

        pbar = tqdm(range(1, n_steps + 1), desc="Training", dynamic_ncols=True)
        for step in pbar:
            batch = replay_buffer.sample(batch_size)
            metrics = self.train_step(batch)
            self.global_step += 1

            # Determine eval trigger from whichever evaluator is active.
            _eval_cfg: Dict[str, Any] = {}
            if rollout_evaluator is not None:
                _eval_cfg = getattr(rollout_evaluator, "eval_cfg", {}) or {}
            elif value_evaluator is not None:
                _eval_cfg = getattr(value_evaluator, "eval_cfg", {}) or {}

            if _eval_cfg:
                eval_enabled = bool(_eval_cfg.get("enabled", False))
                eval_every = max(1, int(_eval_cfg.get("every_n_steps", 1)))
                warmstart_steps = max(0, int(_eval_cfg.get("warmstart_steps", 0)))

                should_eval = (
                    eval_enabled
                    and self.global_step >= warmstart_steps
                    and (self.global_step % eval_every == 0)
                )
                if should_eval:
                    # Switch all live modules to eval mode.
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

                        rollout_metrics: Dict[str, Any] = {}
                        if rollout_evaluator is not None:
                            rollout_metrics = rollout_evaluator.run(global_step=int(self.global_step))

                        value_result = None
                        if value_evaluator is not None:
                            value_result = value_evaluator.run(global_step=int(self.global_step))
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
                        if rollout_metrics:
                            logger.record_dict(rollout_metrics)
                        for idx, video_path in enumerate(
                            getattr(rollout_evaluator, "last_video_paths", [])
                            if rollout_evaluator is not None else []
                        ):
                            key = "eval/video" if idx == 0 else f"eval/video_{idx}"
                            logger.record_video(key=key, path=video_path, step=self.global_step)
                        if value_result is not None:
                            value_metrics = {}
                            q_hist_path = None
                            v_hist_path = None
                            adv_hist_path = None
                            adv_curve_path = None
                            if isinstance(value_result, dict):
                                value_metrics = value_result
                            elif isinstance(value_result, (tuple, list)):
                                if len(value_result) >= 1:
                                    value_metrics = value_result[0]
                                if len(value_result) >= 2:
                                    q_hist_path = value_result[1]
                                if len(value_result) >= 3:
                                    # Backward-compatible: 3rd item is advantage hist in old API,
                                    # but V-value hist in newer APIs.
                                    if len(value_result) >= 5:
                                        v_hist_path = value_result[2]
                                        adv_hist_path = value_result[3]
                                        adv_curve_path = value_result[4]
                                    elif len(value_result) >= 4:
                                        v_hist_path = value_result[2]
                                        adv_hist_path = value_result[3]
                                    else:
                                        adv_hist_path = value_result[2]
                            logger.record_dict(value_metrics)
                            if q_hist_path is not None and hasattr(logger, "record_image"):
                                logger.record_image("eval/q_value_distribution", q_hist_path, step=self.global_step)
                            if v_hist_path is not None and hasattr(logger, "record_image"):
                                logger.record_image("eval/v_value_distribution", v_hist_path, step=self.global_step)
                            if adv_hist_path is not None and hasattr(logger, "record_image"):
                                logger.record_image("eval/advantage_distribution", adv_hist_path, step=self.global_step)
                            if adv_curve_path is not None and hasattr(logger, "record_image"):
                                logger.record_image("eval/advantage_curve", adv_curve_path, step=self.global_step)
                        logger.dump(step=self.global_step)

            if logger is not None and step % max(1, log_every) == 0:
                logger.record_dict(metrics)
                logger.dump(step=self.global_step)
                postfix = {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))}
                if postfix:
                    pbar.set_postfix(postfix)

            if step % max(1, save_every) == 0:
                self.save(save_dir, f"step_{self.global_step}")

        self.save(save_dir, "final")
