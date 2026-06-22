"""Configuration loader for policy training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


class PolicyTrainingConfig:
    """Load and merge policy-training YAML configs."""

    DEFAULTS: Dict[str, Any] = {
        "eval": {
            "enabled": False,
            "every_n_steps": 5000,
            "warmstart_steps": 0,
            "n_rollouts": 20,
            "horizon": 400,
            "stochastic": False,
            "terminate_on_success": True,
            "env_name_override": None,
            "num_workers": 1,
            "worker_device": "cpu",
            "video": {
                "enabled": False,
                "max_episodes": 1,
                "dir": None,
                "skip": 5,
                "fps": 20,
                "frame_height": 512,
                "frame_width": 512,
                "camera_names": ["agentview", "robot0_eye_in_hand"],
            },
            "output": {
                "json_dir": None,
            },
            "value": {
                "enabled": True,
                "masks": None,
                "batch_size": 4096,
                "histogram_bins": 80,
                "output_dir": None,
            },
        },
    }

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = PolicyTrainingConfig._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _to_namespace(obj: Any) -> Any:
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: PolicyTrainingConfig._to_namespace(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [PolicyTrainingConfig._to_namespace(v) for v in obj]
        return obj

    @staticmethod
    def load(config_path: str) -> SimpleNamespace:
        root = Path(__file__).resolve().parents[1]
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

        with open(target_path, "r", encoding="utf-8") as f:
            target_cfg = yaml.safe_load(f) or {}

        merged = PolicyTrainingConfig._deep_merge(PolicyTrainingConfig.DEFAULTS, target_cfg)

        dataset_h5 = merged.get("dataset", {}).get("h5_path")
        if dataset_h5:
            dataset_h5_path = Path(dataset_h5)
            if not dataset_h5_path.is_absolute():
                merged.setdefault("dataset", {})["h5_path"] = str((root / dataset_h5_path).resolve())

        save_dir = merged.get("train", {}).get("save_dir")
        if save_dir:
            save_dir_path = Path(save_dir)
            if not save_dir_path.is_absolute():
                merged.setdefault("train", {})["save_dir"] = str((root / save_dir_path).resolve())

        # Resolve the new-style save_dir_root (parent of auto-generated per-run dirs).
        save_dir_root = merged.get("train", {}).get("save_dir_root")
        if save_dir_root:
            save_dir_root_path = Path(save_dir_root)
            if not save_dir_root_path.is_absolute():
                merged.setdefault("train", {})["save_dir_root"] = str(
                    (root / save_dir_root_path).resolve()
                )

        if merged.get("device") is None:
            merged["device"] = "auto"

        merged["project_root"] = str(root)
        merged["config_path"] = str(target_path.resolve())
        return PolicyTrainingConfig._to_namespace(merged)
