"""Lightweight logger with optional Weights & Biases support."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch


def parse_policy_train_args() -> argparse.Namespace:
    """Parse CLI arguments for policy training entrypoint."""
    parser = argparse.ArgumentParser(description="Offline policy training entrypoint")
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Optional algorithm override (e.g., iql or online_sac).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config file (absolute or relative to policy_training/). "
            "If omitted, defaults to configs/{algo}.yaml when --algo is set, else configs/iql.yaml."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override: cpu | cuda | auto",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short smoke training by overriding n_steps to 50",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume from",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def namespace_to_dict(obj: Any):
    """Convert a nested namespace/list structure to plain Python containers."""
    if hasattr(obj, "__dict__"):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    return obj


def resolve_device(device_cfg: str, override: str | None) -> torch.device:
    """Resolve runtime device from config and CLI override."""
    chosen = override if override is not None else device_cfg
    if chosen == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(chosen)


def _sanitize_token(value: str, fallback: str, *, lowercase: bool = True) -> str:
    """Normalize a token for directory / wandb naming."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip()).strip("_")
    if not cleaned:
        return fallback
    return cleaned.lower() if lowercase else cleaned


def _derive_env_task_from_dataset_path(h5_path: str) -> tuple[str | None, str | None]:
    """Infer env/task from dataset path pattern under datasets/{env}/{task}/..."""
    p = Path(h5_path)
    parts = [part for part in p.parts if part and part != "/"]
    parts_lower = [part.lower() for part in parts]

    env_name = None
    task_name = None

    if "datasets" in parts_lower:
        i = parts_lower.index("datasets")
        if i + 2 < len(parts_lower):
            env_name = _sanitize_token(parts_lower[i + 1], "unknown_env")
            task_name = _sanitize_token(parts_lower[i + 2], "unknown_task")

    return env_name, task_name


def _derive_env_task_from_h5_attrs(h5_path: str) -> tuple[str | None, str | None]:
    """Infer env/task from robomimic-style HDF5 attributes when available."""
    try:
        with h5py.File(h5_path, "r") as f:
            data_grp = f.get("data", None)
            if data_grp is None:
                return None, None

            env_args_raw = data_grp.attrs.get("env_args", None)
            if env_args_raw is None:
                return None, None

            if isinstance(env_args_raw, bytes):
                env_args_raw = env_args_raw.decode("utf-8", errors="ignore")
            env_args = json.loads(str(env_args_raw))

            env_value = env_args.get("env_name", None)
            if not env_value:
                return None, None

            env_norm = _sanitize_token(str(env_value), "unknown_env")
            task_norm = _sanitize_token(str(env_value), "unknown_task")
            return env_norm, task_norm
    except Exception:
        return None, None


def _derive_reward_type_from_h5_path(h5_path: str) -> str:
    """Infer reward type from the HDF5 filename."""
    filename = Path(h5_path).name
    match = re.search(r"_reward_labeled_(.+?)(?:_resnet18feats)?\.hdf5$", filename, flags=re.IGNORECASE)
    if match:
        return _sanitize_token(match.group(1), "all")

    stem = Path(filename).stem
    if "reward_labeled" in stem:
        suffix = stem.split("reward_labeled", 1)[-1].strip("_-")
        if suffix:
            return _sanitize_token(suffix, "all")

    return "all"


def derive_run_metadata(cfg: Any) -> dict:
    """Extract env/task/algo/mask/seed and build canonical run identifiers.

    Metadata priority for env/task is dataset-driven by default:
    1) parse from dataset h5 path under datasets/{env}/{task}/...
    2) parse from HDF5 attrs (data/env_args)
    3) fallback to YAML env_name/task_name or unknown_*
    """
    algo_name = str(cfg.algo_name).lower()
    seed = int(cfg.seed)
    h5_path = str(cfg.dataset.h5_path)
    reward_type = _derive_reward_type_from_h5_path(h5_path)
    if algo_name == "online_sac":
        online_cfg = getattr(cfg, "online", None)
        reward_cfg = getattr(online_cfg, "reward", None) if online_cfg is not None else None
        reward_type = _sanitize_token(str(getattr(reward_cfg, "type", "sparse_done")), "sparse_done")

    path_env, path_task = _derive_env_task_from_dataset_path(h5_path)
    attr_env, attr_task = _derive_env_task_from_h5_attrs(h5_path)

    env_name = path_env or attr_env or _sanitize_token(str(getattr(cfg, "env_name", None) or ""), "unknown_env")
    task_name = path_task or attr_task or _sanitize_token(str(getattr(cfg, "task_name", None) or ""), "unknown_task")

    filter_key = str(getattr(cfg.dataset, "filter_key", "") or "")
    explicit_mask = getattr(cfg, "mask_name", None)
    if explicit_mask:
        mask_name = _sanitize_token(str(explicit_mask), "all")
    elif filter_key.lower().startswith(algo_name + "_"):
        mask_name = _sanitize_token(filter_key[len(algo_name) + 1 :], "all")
    elif filter_key:
        mask_name = _sanitize_token(filter_key, "all")
    else:
        mask_name = "all"

    algo_label = _sanitize_token(str(cfg.algo_name), "algo", lowercase=False).upper()
    filter_label = _sanitize_token(filter_key, "ALL", lowercase=False) if filter_key else "ALL"
    algo_mask_name = f"{algo_label}__{filter_label}"
    is_scale_run = bool(getattr(cfg, "scale_run_dir_name", None))
    scale_run_dir_name = None
    if is_scale_run:
        scale_run_dir_name = _sanitize_token(
            str(getattr(cfg, "scale_run_dir_name", "scale_run")),
            "scale_run",
            lowercase=True,
        )

    run_timestamp = getattr(cfg, "run_timestamp", None)
    if run_timestamp:
        run_timestamp = _sanitize_token(str(run_timestamp), datetime.now().strftime("%Y%m%d_%H%M%S"), lowercase=False)
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    seed_label = f"seed{seed}"
    run_tag = f"{algo_mask_name}_{seed_label}"
    return {
        "env_name": env_name,
        "task_name": task_name,
        "algo_name": algo_name,
        "algo_mask_name": algo_mask_name,
        "reward_type": reward_type,
        "mask_name": mask_name,
        "seed": seed,
        "seed_label": seed_label,
        "run_timestamp": run_timestamp,
        "is_scale_run": is_scale_run,
        "scale_run_dir_name": scale_run_dir_name,
        "run_tag": run_tag,
    }


def resolve_save_dir(cfg: Any, meta: dict, project_root: str) -> str:
    """Build checkpoint save directory for scale and non-scale runs.

    train.save_dir_root controls the root (defaults to checkpoint).
    train.save_dir remains a manual hard override for backward compatibility.
    """
    root = Path(project_root)
    explicit_root = getattr(cfg.train, "save_dir_root", None)
    explicit_dir = getattr(cfg.train, "save_dir", None)
    if explicit_dir:
        p = Path(explicit_dir)
        if not p.is_absolute():
            p = root / p
        return str(p)

    base = Path(explicit_root) if explicit_root else root / "checkpoint"
    if not base.is_absolute():
        base = root / base
    save_dir = base / meta["env_name"] / meta["task_name"]
    if meta.get("is_scale_run") and meta.get("scale_run_dir_name"):
        save_dir = save_dir / meta["scale_run_dir_name"]
    save_dir = (
        save_dir
        / meta["algo_mask_name"]
        / meta["reward_type"]
        / meta["run_timestamp"]
        / meta["seed_label"]
        / "ckpt"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    return str(save_dir)


def build_wandb_run_id(meta: dict) -> tuple[str, str]:
    """Build wandb (group, name) from dataset-derived run metadata.

    group: {env}/{task}/{algo_mask}/{reward_type}[/{scale_run}]
    name:  {env}-{task}-[{SCALE}-]{algo_mask}-{reward_type}-{seed}
    """
    group = f"{meta['env_name']}/{meta['task_name']}/{meta['algo_mask_name']}/{meta['reward_type']}"
    if meta.get("is_scale_run"):
        group = f"{group}/SCALE"

    name_parts = [meta["env_name"], meta["task_name"]]
    if meta.get("is_scale_run"):
        name_parts.append("SCALE")
    name_parts.extend([meta["algo_mask_name"], meta["reward_type"], meta["seed_label"]])
    name = "-".join(name_parts)
    return group, name


class WandBLogger:
    """A minimal key-value logger with optional wandb backend."""

    def __init__(self, enabled: bool = False, project: Optional[str] = None, group: Optional[str] = None, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.enabled = bool(enabled)
        self._buffer = defaultdict(float)
        self._wandb = None

        if self.enabled:
            try:
                import wandb

                self._wandb = wandb
                self._wandb.init(project=project, group=group, name=name, config=config)
            except Exception as exc:
                print(f"[logger] wandb disabled due to init error: {exc}")
                self.enabled = False
                self._wandb = None

    def record(self, key: str, value: Any) -> None: # record data here
        self._buffer[key] = value

    def record_dict(self, metrics: Dict[str, Any]) -> None:
        for key, value in metrics.items():
            self._buffer[key] = value

    def record_video(self, key: str, path: str, step: int, fps: int = 20) -> None:
        if not (self.enabled and self._wandb is not None):
            return
        if not path or not os.path.isfile(path):
            return
        try:
            self._wandb.log(
                {str(key): self._wandb.Video(str(path), fps=int(fps), format="mp4")},
                step=int(step),
            )
        except Exception as exc:
            print(f"[logger] wandb video logging failed for {path}: {exc}")

    def record_image(self, key: str, path: str, step: int) -> None:
        if not (self.enabled and self._wandb is not None):
            return
        if not path or not os.path.isfile(path):
            return
        try:
            self._wandb.log(
                {str(key): self._wandb.Image(str(path))},
                step=int(step),
            )
        except Exception as exc:
            print(f"[logger] wandb image logging failed for {path}: {exc}")

    def dump(self, step: int) -> None: # log data to wandb or print
        if not self._buffer:
            return
        payload = dict(self._buffer)
        if self.enabled and self._wandb is not None:
            self._wandb.log(payload, step=step)
        else:
            print(f"[step={step}] {payload}")
        self._buffer.clear()

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
