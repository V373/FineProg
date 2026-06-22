"""Evaluation helpers for policy_training – loaded by evaluate_policy.py."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import re
from types import SimpleNamespace
import time
from typing import Any

import imageio_ffmpeg
from gymnasium import spaces
from tqdm import tqdm
import numpy as np
import torch
import yaml

from algos import build_algo
from envs.robomimic import create_robomimic_env
from models.model_utils import FrozenPretrainedResNet18Conv
from reward_model import TCCExpertProjectionDenseRewardProvider
from utils.algo_config import get_algo_cfg
from utils.checkpoints import normalize_obs_slices, validate_checkpoint_payload
from utils.config import PolicyTrainingConfig
from utils.logger import (
    _derive_env_task_from_dataset_path,
    _sanitize_token,
    derive_run_metadata,
    resolve_device,
)
from utils.progress_viz import save_progress_curve


# Root of the policy_training package (parent of this utils/ directory).
POLICY_TRAINING_ROOT = Path(__file__).resolve().parent.parent

# Repo-relative default directory for manually-evaluated runs.
DEFAULT_EVAL_OUTPUT_DIR = "outputs/eval"

DEFAULT_EVAL_CONFIG = {
    "agent": None,
    "device": "auto",
    "seed": 0,
    "policy": {"stochastic": False},
    "rollout": {"n_rollouts": 27, "horizon": 400, "num_workers": 1},
    "env": {"name_override": None},
    "video": {
        "enabled": False,
        "path": None,
        "skip": 5,
        "fps": 20,
        "frame_height": 512,
        "frame_width": 512,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    },
    "output": {"dir": DEFAULT_EVAL_OUTPUT_DIR, "json_path": None},
}


@dataclass
class EvalRuntimeConfig:
    config_path: str
    config_dir: Path
    values: dict[str, Any]


@dataclass
class CheckpointRuntime:
    checkpoint_path: str
    payload: dict[str, Any]
    cfg: SimpleNamespace
    env_metadata: dict[str, Any]
    shape_metadata: dict[str, Any]
    obs_slices: dict[str, tuple[int, int]]
    algo: Any
    device: torch.device


# ---------------------------------------------------------------------------
# Path / config helpers
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str, *, base_dir: Path) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    if path.exists():
        return str(path.resolve())
    return str((base_dir / path).resolve())


def _to_namespace(data: dict[str, Any]) -> SimpleNamespace:
    return PolicyTrainingConfig._to_namespace(data)


def _plain_data(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _plain_data(val) for key, val in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _plain_data(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_eval_config(config_path: str) -> EvalRuntimeConfig:
    resolved = _resolve_path(config_path, base_dir=POLICY_TRAINING_ROOT)
    with open(resolved, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    values = _deep_merge(deepcopy(DEFAULT_EVAL_CONFIG), loaded)
    return EvalRuntimeConfig(config_path=resolved, config_dir=Path(resolved).parent, values=values)


def _apply_cli_overrides(runtime_cfg: EvalRuntimeConfig, args: Any) -> dict[str, Any]:
    values = deepcopy(runtime_cfg.values)

    if args.agent is not None:
        values["agent"] = args.agent
    if args.device is not None:
        values["device"] = args.device
    if args.seed is not None:
        values["seed"] = args.seed
    if args.n_rollouts is not None:
        values["rollout"]["n_rollouts"] = args.n_rollouts
    if args.horizon is not None:
        values["rollout"]["horizon"] = args.horizon
    if args.env is not None:
        values["env"]["name_override"] = args.env
    if getattr(args, "no_video", False):
        values["video"]["enabled"] = False
    if args.video_path is not None:
        values["video"]["path"] = args.video_path
        values["video"]["enabled"] = True
    if args.video_skip is not None:
        values["video"]["skip"] = args.video_skip
    if args.video_fps is not None:
        values["video"]["fps"] = args.video_fps
    if args.frame_height is not None:
        values["video"]["frame_height"] = args.frame_height
    if args.frame_width is not None:
        values["video"]["frame_width"] = args.frame_width
    if args.camera_names is not None:
        values["video"]["camera_names"] = args.camera_names
    if args.json_path is not None:
        values["output"]["json_path"] = args.json_path
    if args.output_dir is not None:
        values["output"]["dir"] = args.output_dir
    if args.stochastic is not None:
        values["policy"]["stochastic"] = bool(args.stochastic)

    if values.get("agent") is None:
        raise ValueError("Evaluation config must define 'agent' or override it via --agent.")

    return values


def _resolve_eval_output_path(path_value: str | None, config_dir: Path) -> str | None:
    if path_value is None:
        return None
    return _resolve_path(path_value, base_dir=config_dir)


# ---------------------------------------------------------------------------
# Auto output-path inference (manual evaluate_policy.py)
# ---------------------------------------------------------------------------

_RUN_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def _coerce_cfg(payload: dict[str, Any]) -> SimpleNamespace:
    """Best-effort conversion of a checkpoint's saved config to a SimpleNamespace.

    The checkpoint may store the training config in either of two shapes:

    1) A nested dict (PolicyTrainingConfig.to_dict style) — the modern format.
    2) A SimpleNamespace directly serialized by torch.save — the legacy format.

    For path inference we only need attribute-style access, so a top-level
    SimpleNamespace is sufficient in both cases.
    """
    cfg_value = payload.get("config", {})
    if isinstance(cfg_value, SimpleNamespace):
        return cfg_value
    return _to_namespace(deepcopy(cfg_value))


def _derive_run_timestamp_from_checkpoint_path(checkpoint_path: str) -> str | None:
    """Look for a YYYYMMDD_HHMMSS token in the checkpoint path components."""
    if not checkpoint_path:
        return None
    parts = Path(checkpoint_path).parts
    for part in parts:
        match = _RUN_TIMESTAMP_RE.search(part)
        if match:
            return match.group(1)
    return None


def _derive_run_id_from_checkpoint_path(checkpoint_path: str) -> str:
    """Pick a stable run id from a checkpoint path.

    Prefers a YYYYMMDD_HHMMSS timestamp anywhere in the path; otherwise
    falls back to the parent directory name (which is typically a meaningful
    run identifier for both old and new layouts).
    """
    ts = _derive_run_timestamp_from_checkpoint_path(checkpoint_path)
    if ts:
        return ts
    return Path(checkpoint_path).parent.name or "run"


def _checkpoint_tag_from_path(checkpoint_path: str) -> str:
    """Return a human-readable tag for the checkpoint file itself."""
    stem = Path(checkpoint_path).stem
    return stem or "ckpt"


def _infer_eval_output_metadata(
    loaded: CheckpointRuntime, checkpoint_path: str
) -> dict[str, Any]:
    """Infer env/task/split/algo/seed/run metadata for the manual eval run.

    All fields are best-effort and fall back to ``"unknown_*"`` sentinels when
    the checkpoint lacks the corresponding information.
    """
    cfg = _coerce_cfg(loaded.payload)
    dataset = getattr(cfg, "dataset", None)
    h5_path = str(getattr(dataset, "h5_path", "") or "")

    path_env, path_task = _derive_env_task_from_dataset_path(h5_path)
    env_name = path_env or _sanitize_token(str(getattr(cfg, "env_name", None) or ""), "unknown_env")
    task_name = path_task or _sanitize_token(str(getattr(cfg, "task_name", None) or ""), "unknown_task")

    # Split: walk the h5_path components and pick the first non-trivial
    # segment after env/task that looks like a meaningful sub-folder. For
    # robomimic datasets, "mh" / "mg" / "ph" / "paired" is always at index
    # env+1 so we just take the first component after task_name.
    split_name = "unknown_split"
    if h5_path:
        parts_lower = [part.lower() for part in Path(h5_path).parts if part and part != "/"]
        try:
            datasets_idx = parts_lower.index("datasets")
        except ValueError:
            datasets_idx = -1
        if datasets_idx >= 0:
            # env, task, split, ...
            split_idx = datasets_idx + 3
            if split_idx < len(parts_lower):
                candidate = _sanitize_token(parts_lower[split_idx], "unknown_split")
                if candidate not in {"reward_labeled", "resnet18feats", "videos"}:
                    split_name = candidate
                else:
                    # Common sub-folders for the can/mh dataset: try next
                    # segment if it exists and is non-trivial.
                    if split_idx + 1 < len(parts_lower):
                        nested = _sanitize_token(parts_lower[split_idx + 1], "unknown_split")
                        split_name = nested

    # Reuse the canonical algo/mask naming convention from the training
    # pipeline so the eval output dir matches the train checkpoint dir.
    try:
        run_meta = derive_run_metadata(cfg)
        algo_mask_name = str(run_meta["algo_mask_name"])
    except Exception:
        algo_name = str(getattr(cfg, "algo_name", "") or "algo")
        filter_key = str(getattr(dataset, "filter_key", "") or "") if dataset is not None else ""
        algo_label = _sanitize_token(algo_name, "ALGO", lowercase=False).upper()
        filter_label = _sanitize_token(filter_key, "ALL", lowercase=False) if filter_key else "ALL"
        algo_mask_name = f"{algo_label}__{filter_label}"

    try:
        train_seed = int(getattr(cfg, "seed", 0) or 0)
    except (TypeError, ValueError):
        train_seed = 0
    seed_label = f"seed{int(train_seed)}"

    run_id = _derive_run_id_from_checkpoint_path(checkpoint_path)
    checkpoint_tag = _checkpoint_tag_from_path(checkpoint_path)

    return {
        "env_name": env_name,
        "task_name": task_name,
        "split_name": split_name,
        "algo_mask_name": algo_mask_name,
        "train_seed": int(train_seed),
        "seed_label": seed_label,
        "run_id": run_id,
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_path": str(checkpoint_path),
        "h5_path": h5_path,
    }


def _resolve_auto_eval_paths(
    eval_cfg: dict[str, Any],
    loaded: CheckpointRuntime,
    config_dir: Path,
    *,
    n_rollouts: int,
    horizon: int,
) -> dict[str, Any]:
    """Compute the auto output dir / video path / json path for a manual eval run.

    Honours the following precedence:
    - ``output.json_path`` non-null  ->  use that as the JSON path
    - ``video.path`` non-null         ->  use that as the video path
    - ``--no_video`` / ``video.enabled=False``  ->  skip video entirely
    - Otherwise                      ->  auto-infer from checkpoint metadata
    """
    meta = _infer_eval_output_metadata(loaded, loaded.checkpoint_path)

    output_cfg = eval_cfg.get("output", {}) or {}
    video_cfg = eval_cfg.get("video", {}) or {}

    # Root dir: ``output.dir`` from config (relative -> under policy_training/).
    output_dir_value = output_cfg.get("dir") or DEFAULT_EVAL_OUTPUT_DIR
    output_dir = _resolve_path(str(output_dir_value), base_dir=POLICY_TRAINING_ROOT)
    auto_dir = Path(output_dir) / meta["env_name"] / meta["task_name"] / meta["split_name"] / meta["algo_mask_name"] / meta["seed_label"] / meta["run_id"] / meta["checkpoint_tag"]

    seed_value = int(eval_cfg.get("seed", 0))
    eval_seed_token = f"evalseed{int(seed_value)}"
    n_rollouts_token = f"n{int(max(0, n_rollouts)):03d}"
    horizon_token = f"h{int(max(1, horizon)):04d}"
    global_step = int(loaded.payload.get("global_step", 0) or 0)
    base_name = f"eval_step_{global_step:09d}_{eval_seed_token}_{n_rollouts_token}_{horizon_token}"

    auto_video_path = str((auto_dir / f"{base_name}.mp4").resolve())
    auto_json_path = str((auto_dir / f"{base_name}.json").resolve())

    video_enabled = _is_video_enabled(video_cfg)
    video_path_value = video_cfg.get("path")
    if video_path_value:
        video_path = _resolve_path(str(video_path_value), base_dir=config_dir)
    elif video_enabled:
        video_path = auto_video_path
    else:
        video_path = None

    json_path_value = output_cfg.get("json_path")
    if json_path_value:
        json_path = _resolve_path(str(json_path_value), base_dir=config_dir)
    else:
        json_path = auto_json_path

    return {
        "output_dir": str(auto_dir.resolve()),
        "video": {
            "enabled": bool(video_enabled),
            "path": video_path,
            "auto_path": auto_video_path,
        },
        "json": {
            "path": json_path,
            "auto_path": auto_json_path,
        },
        "metadata": meta,
        "base_name": base_name,
    }


def _is_video_enabled(video_cfg: dict[str, Any] | None) -> bool:
    """Return True iff a video writer should be created.

    Honours the new ``video.enabled`` flag while staying compatible with the
    legacy convention of treating a non-null ``video.path`` as an opt-in.
    """
    if not video_cfg:
        return False
    if "enabled" in video_cfg:
        return bool(video_cfg.get("enabled"))
    return video_cfg.get("path") is not None


def _resolve_policy_training_path(path_value: object, project_root: str) -> str | None:
    if path_value is None:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = Path(project_root) / path
    return str(path.resolve())


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_payload(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _load_modules_for_eval(algo: Any, payload: dict[str, Any]) -> None:
    algo.global_step = int(payload["global_step"])
    algo._load_module_state_dict(payload["modules"])


def _normalization_stats_from_payload(payload: dict[str, Any]) -> tuple[Any, Any]:
    stats = payload.get("normalization_stats") or {}
    return stats.get("obs"), stats.get("actions")


def _ensure_normalization_support(cfg: SimpleNamespace, payload: dict[str, Any]) -> tuple[Any, Any]:
    obs_stats, action_stats = _normalization_stats_from_payload(payload)
    if bool(getattr(cfg.dataset, "normalize_obs", False)) and obs_stats is None:
        raise ValueError(
            "Checkpoint config requests normalize_obs=true but checkpoint has no observation stats."
        )
    if bool(getattr(cfg.dataset, "normalize_actions", False)) and action_stats is None:
        raise ValueError(
            "Checkpoint config requests normalize_actions=true but checkpoint has no action stats."
        )
    return obs_stats, action_stats


def _visual_keys_from_checkpoint(cfg: SimpleNamespace, shape_metadata: dict[str, Any]) -> list[str]:
    algo_cfg = get_algo_cfg(cfg)
    visual_keys = list(shape_metadata.get("visual_obs_keys", []))
    if visual_keys:
        return visual_keys
    fe_type = str(getattr(algo_cfg, "features_extractor_type", "flat_range"))
    if fe_type == "resnet18conv":
        fe_kwargs = getattr(algo_cfg, "features_extractor_kwargs", None)
        if fe_kwargs is None:
            return []
        visual_specs = _plain_data(getattr(fe_kwargs, "visual_specs", {}))
        return list(visual_specs.keys())
    return [key for key in cfg.dataset.obs_keys if key.endswith("_image")]


def load_checkpoint_for_eval(checkpoint_path: str, device_setting: str) -> CheckpointRuntime:
    resolved_checkpoint = _resolve_path(checkpoint_path, base_dir=POLICY_TRAINING_ROOT)
    device = resolve_device("auto", device_setting)
    payload = _load_payload(resolved_checkpoint, device)
    validate_checkpoint_payload(payload, context="evaluate_policy.load_checkpoint_for_eval")

    cfg = _to_namespace(deepcopy(payload["config"]))
    env_metadata = deepcopy(payload["env_metadata"])
    shape_metadata = deepcopy(payload["shape_metadata"])
    obs_slices = normalize_obs_slices(payload["obs_slices"])

    observation_dim = int(shape_metadata["observation_dim"])
    action_dim = int(shape_metadata["action_dim"])

    obs_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(observation_dim,),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(action_dim,),
        dtype=np.float32,
    )

    algo_cfg = get_algo_cfg(cfg, algo_name=str(payload["algo_name"]))
    if str(getattr(algo_cfg, "features_extractor_type", "flat_range")) == "resnet18conv":
        fe_kwargs = getattr(algo_cfg, "features_extractor_kwargs", None)
        if fe_kwargs is None:
            raise ValueError("Checkpoint config is missing features_extractor_kwargs for resnet18conv.")
        fe_kwargs.obs_slices = dict(obs_slices)

    algo = build_algo(
        algo_name=str(payload["algo_name"]),
        observation_space=obs_space,
        action_space=action_space,
        cfg=algo_cfg,
        device=device,
    )
    _load_modules_for_eval(algo, payload)
    algo.policy.set_training_mode(False)
    algo.policy.eval()
    algo.actor.eval()
    for attr in ("critic", "critic_target", "v_net"):
        module = getattr(algo, attr, None)
        if module is not None and hasattr(module, "eval"):
            module.eval()

    _ensure_normalization_support(cfg, payload)

    return CheckpointRuntime(
        checkpoint_path=resolved_checkpoint,
        payload=payload,
        cfg=cfg,
        env_metadata=env_metadata,
        shape_metadata=shape_metadata,
        obs_slices=obs_slices,
        algo=algo,
        device=device,
    )


# ---------------------------------------------------------------------------
# Observation adapter and rollout policy
# ---------------------------------------------------------------------------

class ObservationAdapter:
    """Convert env observations into the flat checkpoint observation layout."""

    def __init__(
        self,
        cfg: SimpleNamespace,
        shape_metadata: dict[str, Any],
        obs_slices: dict[str, tuple[int, int]],
        device: torch.device,
        obs_normalization_stats: Any = None,
    ):
        self.cfg = cfg
        self.shape_metadata = shape_metadata
        self.obs_slices = obs_slices
        self.device = device
        self.obs_keys = list(cfg.dataset.obs_keys)
        self.visual_keys = _visual_keys_from_checkpoint(cfg, shape_metadata)
        self.visual_key_set = set(self.visual_keys)
        self.obs_normalization_stats = obs_normalization_stats
        algo_cfg = get_algo_cfg(cfg)
        self.feature_extractor_type = str(getattr(algo_cfg, "features_extractor_type", "flat_range"))
        self.visual_specs = _plain_data(
            getattr(getattr(algo_cfg, "features_extractor_kwargs", None), "visual_specs", {})
        )
        self.image_encoder = None
        if self.visual_keys and self.feature_extractor_type == "resnet18conv":
            self.image_encoder = FrozenPretrainedResNet18Conv(device=device)
            self.image_encoder.eval()
        # Pre-cache normalization arrays to avoid per-step dict lookups.
        self._norm_offset: np.ndarray | None = None
        self._norm_scale: np.ndarray | None = None
        if obs_normalization_stats is not None:
            _obs_stats = obs_normalization_stats.get("observations")
            if _obs_stats is not None:
                self._norm_offset = np.asarray(_obs_stats["offset"], dtype=np.float32).reshape(-1)
                self._norm_scale = np.asarray(_obs_stats["scale"], dtype=np.float32).reshape(-1)

    def _normalize_flat_obs(self, flat_obs: np.ndarray) -> np.ndarray:
        if self._norm_offset is None:
            return flat_obs
        return ((flat_obs - self._norm_offset) / self._norm_scale).astype(np.float32)

    def _prepare_low_dim(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _prepare_visual_feature(self, key: str, value: Any) -> np.ndarray:
        spec = self.visual_specs.get(key)
        if spec is None:
            raise KeyError(f"Visual obs key '{key}' is missing from checkpoint config visual_specs.")

        expected_shape = tuple(int(dim) for dim in spec["input_shape"])
        expected_size = int(np.prod(expected_shape, dtype=np.int64))
        array = np.asarray(value)

        if array.shape == expected_shape:
            return array.astype(np.float32).reshape(-1)
        if array.ndim == 1 and array.size == expected_size:
            return array.astype(np.float32)
        if self.image_encoder is None:
            raise ValueError(f"Visual obs key '{key}' requires a ResNet18Conv encoder.")

        feature = self.image_encoder.extract_feature_map(array)
        if tuple(int(dim) for dim in feature.shape) != expected_shape:
            raise ValueError(
                f"Visual obs key '{key}' produced feature shape {tuple(feature.shape)} but expected {expected_shape}."
            )
        return feature.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def flatten(self, obs: dict[str, Any]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key in self.obs_keys:
            if key not in obs:
                raise KeyError(f"Observation key '{key}' missing from env observation.")
            value = obs[key]
            if key in self.visual_key_set:
                parts.append(self._prepare_visual_feature(key, value))
            else:
                parts.append(self._prepare_low_dim(value))

        flat_obs = np.concatenate(parts, axis=0).astype(np.float32)
        expected_dim = int(self.shape_metadata["observation_dim"])
        if flat_obs.shape[0] != expected_dim:
            raise ValueError(
                f"Flat observation dim mismatch: got {flat_obs.shape[0]}, expected {expected_dim}."
            )
        return self._normalize_flat_obs(flat_obs)


class SB3IQLRolloutPolicy:
    """Rollout wrapper around the policy_training IQL actor."""

    def __init__(
        self,
        algo: Any,
        obs_adapter: ObservationAdapter,
        stochastic: bool,
        action_normalization_stats: Any = None,
    ):
        self.algo = algo
        self.obs_adapter = obs_adapter
        self.stochastic = stochastic
        self.device = algo.device
        self.action_normalization_stats = action_normalization_stats

    def start_episode(self) -> None:
        self.algo.policy.set_training_mode(False)
        self.algo.policy.eval()
        self.algo.actor.eval()

    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_normalization_stats is None:
            return action
        stats = self.action_normalization_stats.get("actions")
        if stats is None:
            return action
        offset = np.asarray(stats["offset"], dtype=np.float32).reshape(-1)
        scale = np.asarray(stats["scale"], dtype=np.float32).reshape(-1)
        return (action * scale + offset).astype(np.float32)

    def __call__(self, ob: dict[str, Any]) -> np.ndarray:
        flat_obs = self.obs_adapter.flatten(ob)
        obs_tensor = torch.as_tensor(flat_obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            mean_actions, log_std, _ = self.algo.actor.get_action_dist_params(obs_tensor)
            actions = self.algo.actor.action_dist.actions_from_params(
                mean_actions,
                log_std,
                deterministic=not self.stochastic,
            )
        action = actions.squeeze(0).cpu().numpy().astype(np.float32)
        if np.isnan(action).any():
            raise ValueError("Policy produced NaN action values.")
        return self._unnormalize_action(action)


# ---------------------------------------------------------------------------
# Video, rollout, and output helpers
# ---------------------------------------------------------------------------

def _make_video_writer(
    video_path: str | None,
    video_cfg: dict[str, Any],
) -> tuple[str, Any] | tuple[None, None]:
    """Create an imageio video writer for a fully-resolved path.

    ``video_path`` is the absolute path that the caller (typically
    :func:`_resolve_auto_eval_paths`) already resolved — this helper no
    longer performs any auto-path inference, it just opens the writer.
    """
    if not video_path:
        return None, None

    _, ext = os.path.splitext(video_path)
    if not ext:
        os.makedirs(video_path, exist_ok=True)
        video_path = os.path.join(video_path, "rollout.mp4")

    frame_height = int(video_cfg["frame_height"])
    frame_width = int(video_cfg["frame_width"])
    camera_names = list(video_cfg["camera_names"])
    fps = int(video_cfg["fps"])

    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    video_writer = imageio_ffmpeg.write_frames(
        video_path,
        size=(frame_width * len(camera_names), frame_height),
        fps=fps,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
    )
    video_writer.send(None)
    return video_path, video_writer


def rollout(
    policy: SB3IQLRolloutPolicy,
    env: Any,
    horizon: int,
    video_writer: Any = None,
    video_skip: int = 5,
    camera_names: list[str] | None = None,
    frame_height: int = 512,
    frame_width: int = 512,
    terminate_on_success: bool = True,
) -> dict[str, Any]:
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    total_reward = 0.0
    video_count = 0
    success = False
    step_count = 0

    try:
        for step_idx in range(horizon):
            action = policy(ob=obs)
            next_obs, reward, done, _ = env.step(action)
            total_reward += float(reward)
            success_dict = env.is_success()
            success = bool(success_dict.get("task", False)) if isinstance(success_dict, dict) else bool(success_dict)

            if video_writer is not None and camera_names is not None:
                if video_count % max(1, video_skip) == 0:
                    frames = [
                        env.render(mode="rgb_array", height=frame_height, width=frame_width, camera_name=cam_name)
                        for cam_name in camera_names
                    ]
                    video_img = np.concatenate(frames, axis=1)
                    video_writer.send(np.ascontiguousarray(video_img, dtype=np.uint8))
                video_count += 1

            step_count = step_idx + 1
            if done or (success and terminate_on_success):
                break

            # env.step() returns a new dict each call; no deepcopy needed.
            obs = next_obs
            state_dict = env.get_state()
    except env.rollout_exceptions as exc:
        print(f"WARNING: got rollout exception {exc}")

    return {
        "Return": float(total_reward),
        "Horizon": float(step_count),
        "Success_Rate": float(success),
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _to_jsonable(val) for key, val in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2)


def _summary_from_rollouts(
    rollout_stats: list[dict[str, Any]], checkpoint_path: str, global_step: int
) -> dict[str, Any]:
    if rollout_stats:
        returns = np.asarray([item["Return"] for item in rollout_stats], dtype=np.float32)
        horizons = np.asarray([item["Horizon"] for item in rollout_stats], dtype=np.float32)
        successes = np.asarray([item["Success_Rate"] for item in rollout_stats], dtype=np.float32)
        avg_return = float(np.mean(returns))
        avg_horizon = float(np.mean(horizons))
        success_rate = float(np.mean(successes))
        num_success = int(np.sum(successes))
    else:
        avg_return = 0.0
        avg_horizon = 0.0
        success_rate = 0.0
        num_success = 0

    return {
        "Return": avg_return,
        "Horizon": avg_horizon,
        "Success_Rate": success_rate,
        "Num_Success": num_success,
        "Num_Rollouts": int(len(rollout_stats)),
        "Checkpoint": checkpoint_path,
        "Global_Step": int(global_step),
    }


# ---------------------------------------------------------------------------
# Parallel rollout helpers
# ---------------------------------------------------------------------------

def _to_cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Copy every tensor in a state dict to CPU so it can be pickled."""
    return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}


import contextlib as _contextlib


@_contextlib.contextmanager
def _suppress_worker_output():
    """Redirect stdout/stderr to /dev/null for noisy 3rd-party import/init messages."""
    import sys
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()


# Per-process state populated once by _worker_initializer, reused by every
# _rollout_worker_fn call dispatched to the same worker process.
_WORKER_STATE: dict = {}


def _worker_initializer(init_args: dict) -> None:
    """One-time setup per worker process: rebuild policy and env from serialized args.

    Called by ``multiprocessing.Pool`` before any tasks are dispatched to this
    worker. Stores ready objects in ``_WORKER_STATE`` so that subsequent
    ``_rollout_worker_fn`` calls can reuse them without re-creation overhead.
    """
    import sys

    project_root = init_args["project_root"]
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    device_str = init_args.get("worker_device", "cpu")
    if device_str in ("auto", "cuda"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    shape_metadata = init_args["shape_metadata"]
    obs_slices = normalize_obs_slices(init_args["obs_slices"])
    obs_space = spaces.Box(-np.inf, np.inf, shape=(int(shape_metadata["observation_dim"]),), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(int(shape_metadata["action_dim"]),), dtype=np.float32)

    cfg = PolicyTrainingConfig._to_namespace(init_args["cfg_dict"])
    algo_cfg = get_algo_cfg(cfg, algo_name=str(init_args["algo_name"]))
    if str(getattr(algo_cfg, "features_extractor_type", "flat_range")) == "resnet18conv":
        fe_kwargs = getattr(algo_cfg, "features_extractor_kwargs", None)
        if fe_kwargs is not None:
            fe_kwargs.obs_slices = dict(obs_slices)

    with _suppress_worker_output():
        algo = build_algo(
            algo_name=init_args["algo_name"],
            observation_space=obs_space,
            action_space=action_space,
            cfg=algo_cfg,
            device=device,
        )
    loaded_modules = {
        mod: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sd.items()}
        for mod, sd in init_args["module_snapshot"].items()
    }
    algo._load_module_state_dict(loaded_modules)
    algo.policy.set_training_mode(False)
    algo.policy.eval()
    algo.actor.eval()
    for attr in ("critic", "critic_target", "v_net"):
        m = getattr(algo, attr, None)
        if m is not None and hasattr(m, "eval"):
            m.eval()

    obs_adapter = ObservationAdapter(
        cfg=cfg,
        shape_metadata=shape_metadata,
        obs_slices=obs_slices,
        device=device,
        obs_normalization_stats=init_args.get("obs_normalization_stats"),
    )
    policy_wrapper = SB3IQLRolloutPolicy(
        algo=algo,
        obs_adapter=obs_adapter,
        stochastic=bool(init_args.get("stochastic", False)),
        action_normalization_stats=init_args.get("action_normalization_stats"),
    )

    visual_keys = _visual_keys_from_checkpoint(cfg, shape_metadata)
    video_cfg = init_args.get("video_cfg") or {}
    video_enabled = bool(init_args.get("video_enabled", False))
    with _suppress_worker_output():
        env = create_robomimic_env(
            env_meta=init_args["env_metadata"],
            obs_keys=list(cfg.dataset.obs_keys),
            visual_keys=visual_keys,
            env_name=init_args.get("env_name_override"),
            render=False,
            render_offscreen=video_enabled,
        )

    _WORKER_STATE.update({
        "policy": policy_wrapper,
        "env": env,
        "horizon": int(init_args["horizon"]),
        "terminate_on_success": bool(init_args.get("terminate_on_success", True)),
        "video_enabled": video_enabled,
        "video_cfg": video_cfg,
        "max_video_episodes": int(init_args.get("max_video_episodes", 0)),
        "global_step": int(init_args.get("global_step", 0)),
        "save_dir": Path(init_args.get("save_dir", ".")),
    })


def _rollout_worker_fn(rollout_idx: int) -> dict:
    """Per-rollout eval task; reuses env + policy built by ``_worker_initializer``."""
    state = _WORKER_STATE
    video_cfg = state.get("video_cfg") or {}
    video_writer = None
    video_path = None

    if state["video_enabled"] and rollout_idx < state["max_video_episodes"]:
        save_dir: Path = state["save_dir"]
        global_step: int = state["global_step"]
        video_dir = video_cfg.get("dir")
        if video_dir:
            out_dir = Path(str(video_dir)).expanduser()
            if not out_dir.is_absolute():
                out_dir = (save_dir / out_dir).resolve()
        else:
            out_dir = save_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(out_dir / f"eval_step_{global_step:09d}_rollout_{rollout_idx:03d}.mp4")
        _, video_writer = _make_video_writer(video_path, {**video_cfg, "path": video_path})

    try:
        stat = rollout(
            policy=state["policy"],
            env=state["env"],
            horizon=state["horizon"],
            video_writer=video_writer,
            video_skip=int(video_cfg.get("skip", 5)),
            camera_names=list(video_cfg.get("camera_names", ["agentview"])) if video_writer else None,
            frame_height=int(video_cfg.get("frame_height", 512)),
            frame_width=int(video_cfg.get("frame_width", 512)),
            terminate_on_success=state["terminate_on_success"],
        )
    finally:
        if video_writer is not None:
            try:
                video_writer.close()
            except Exception:
                pass

    stat["rollout_idx"] = rollout_idx
    if video_path and os.path.isfile(video_path):
        stat["video_path"] = video_path
    return stat


class TrainingRolloutEvaluator:
    """Run periodic rollout evaluation during training with the live in-memory algo."""


    def __init__(
        self,
        algo: Any,
        cfg: SimpleNamespace,
        env_metadata: dict[str, Any],
        shape_metadata: dict[str, Any],
        obs_slices: dict[str, tuple[int, int]] | dict[str, list[int]],
        obs_normalization_stats: Any,
        action_normalization_stats: Any,
        eval_cfg: Any,
        save_dir: str,
    ):
        self.algo = algo
        self.cfg = cfg
        self.env_metadata = deepcopy(env_metadata)
        self.shape_metadata = deepcopy(shape_metadata)
        self.obs_slices = normalize_obs_slices(obs_slices)
        self.eval_cfg = _plain_data(eval_cfg)
        self.save_dir = Path(save_dir)
        self.last_video_paths: list[str] = []
        self.last_progress_path: str | None = None

        # Keep raw stats for passing to parallel workers.
        self._obs_normalization_stats = obs_normalization_stats
        self._action_normalization_stats = action_normalization_stats
        self._progress_provider: TCCExpertProjectionDenseRewardProvider | None = None

        # Parallel-worker settings.
        self.num_workers = max(1, int(self.eval_cfg.get("num_workers", 1)))
        self.worker_device = str(self.eval_cfg.get("worker_device", "cpu"))

        self._validate_cfg()

        self.obs_adapter = ObservationAdapter(
            cfg=cfg,
            shape_metadata=self.shape_metadata,
            obs_slices=self.obs_slices,
            device=algo.device,
            obs_normalization_stats=obs_normalization_stats,
        )
        self.policy = SB3IQLRolloutPolicy(
            algo=algo,
            obs_adapter=self.obs_adapter,
            stochastic=bool(self.eval_cfg.get("stochastic", False)),
            action_normalization_stats=action_normalization_stats,
        )

        visual_keys = _visual_keys_from_checkpoint(cfg, self.shape_metadata)
        video_cfg = dict(self.eval_cfg.get("video", {}))
        render_offscreen = bool(video_cfg.get("enabled", False))
        self.env = create_robomimic_env(
            env_meta=self.env_metadata,
            obs_keys=list(cfg.dataset.obs_keys),
            visual_keys=visual_keys,
            env_name=self.eval_cfg.get("env_name_override"),
            render=False,
            render_offscreen=render_offscreen,
        )

    def _progress_model_cfg(self) -> SimpleNamespace | None:
        online_cfg = getattr(self.cfg, "online", None)
        if online_cfg is None:
            return None
        reward_cfg = getattr(online_cfg, "reward", SimpleNamespace())
        reward_type = str(getattr(reward_cfg, "type", "sparse_done"))
        if reward_type not in {"dense", "pbrs"}:
            return None
        reward_model_cfg = getattr(online_cfg, "reward_model", SimpleNamespace(enabled=False))
        if not bool(getattr(reward_model_cfg, "enabled", False)):
            return None
        kind = str(getattr(reward_model_cfg, "kind", "tcc_expert_projection"))
        if kind != "tcc_expert_projection":
            return None
        return reward_model_cfg

    def _build_progress_provider(self) -> TCCExpertProjectionDenseRewardProvider | None:
        if self._progress_provider is not None:
            return self._progress_provider

        reward_model_cfg = self._progress_model_cfg()
        if reward_model_cfg is None:
            return None

        project_root = str(getattr(self.cfg, "project_root", POLICY_TRAINING_ROOT))
        checkpoint_path = _resolve_policy_training_path(getattr(reward_model_cfg, "checkpoint_path", None), project_root)
        expert_path_h5 = _resolve_policy_training_path(getattr(reward_model_cfg, "expert_path_h5", None), project_root)
        train_config_path = _resolve_policy_training_path(getattr(reward_model_cfg, "train_config_path", None), project_root)

        self._progress_provider = TCCExpertProjectionDenseRewardProvider(
            checkpoint_path=checkpoint_path,
            expert_path_h5=expert_path_h5,
            device=self.algo.device,
            expert_group=str(getattr(reward_model_cfg, "expert_group", "videos/mean")),
            projection_temperature=float(getattr(reward_model_cfg, "projection_temperature", 0.1)),
            clip_len=int(getattr(reward_model_cfg, "clip_len", 20)),
            context_size=int(getattr(reward_model_cfg, "context_size", 2)),
            context_stride=int(getattr(reward_model_cfg, "context_stride", 15)),
            pretrained=bool(getattr(reward_model_cfg, "pretrained", True)),
            train_config_path=train_config_path,
            image_key=str(getattr(reward_model_cfg, "image_key", "agentview_image")),
            image_height=int(getattr(reward_model_cfg, "image_height", 224)),
            image_width=int(getattr(reward_model_cfg, "image_width", 224)),
        )
        return self._progress_provider

    def _build_progress_plot(self, global_step: int, video_path: str, video_cfg: dict[str, Any]) -> str | None:
        provider = self._build_progress_provider()
        if provider is None:
            return None

        camera_names = list(video_cfg.get("camera_names", ["agentview"]))
        camera_index = 0
        image_key = str(getattr(provider, "image_key", "agentview_image"))
        if camera_names:
            camera_name = image_key.removesuffix("_image")
            if camera_name in camera_names:
                camera_index = int(camera_names.index(camera_name))

        try:
            progress = provider.infer_progress_trace_from_video(
                video_path,
                camera_names=camera_names,
                camera_index=camera_index,
            )
        except Exception as exc:
            print(f"[eval] progress computation failed for {video_path}: {exc}")
            return None

        out_dir = self.save_dir / "progress"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eval_step_{int(global_step):09d}_{Path(video_path).stem}_progress.png"
        try:
            return save_progress_curve(progress, out_path, title="TCC progress", subtitle=Path(video_path).name)
        except Exception as exc:
            print(f"[eval] progress plot failed for {video_path}: {exc}")
            return None

    def _validate_cfg(self) -> None:
        n_rollouts = int(self.eval_cfg.get("n_rollouts", 0))
        horizon = int(self.eval_cfg.get("horizon", 0))
        every_n_steps = int(self.eval_cfg.get("every_n_steps", 0))
        warmstart_steps = int(self.eval_cfg.get("warmstart_steps", 0))
        if n_rollouts < 0:
            raise ValueError("eval.n_rollouts must be >= 0")
        if horizon <= 0:
            raise ValueError("eval.horizon must be > 0")
        if every_n_steps <= 0:
            raise ValueError("eval.every_n_steps must be > 0")
        if warmstart_steps < 0:
            raise ValueError("eval.warmstart_steps must be >= 0")

    def _video_path_for_rollout(self, global_step: int, rollout_idx: int) -> str:
        video_cfg = dict(self.eval_cfg.get("video", {}))
        video_dir = video_cfg.get("dir")
        if video_dir:
            out_dir = Path(str(video_dir)).expanduser()
            if not out_dir.is_absolute():
                out_dir = (self.save_dir / out_dir).resolve()
        else:
            out_dir = self.save_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"eval_step_{int(global_step):09d}_rollout_{int(rollout_idx):03d}.mp4")

    def _write_json_output(self, global_step: int, summary: dict[str, Any], rollout_stats: list[dict[str, Any]]) -> None:
        output_cfg = dict(self.eval_cfg.get("output", {}))
        json_dir = output_cfg.get("json_dir")
        if not json_dir:
            return
        out_dir = Path(str(json_dir)).expanduser()
        if not out_dir.is_absolute():
            out_dir = (self.save_dir / out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eval_step_{int(global_step):09d}.json"
        _write_json(
            str(out_path),
            {
                "summary": summary,
                "per_rollout_stats": rollout_stats,
                "global_step": int(global_step),
            },
        )

    def _run_parallel(
        self,
        global_step: int,
        n_rollouts: int,
        horizon: int,
        terminate_on_success: bool,
        video_enabled: bool,
        max_video_episodes: int,
        video_cfg: dict,
    ) -> tuple[list[dict], list[str]]:
        """Run rollouts in parallel. Per-rollout tqdm progress; env built once per worker."""
        init_args = {
            "project_root": str(POLICY_TRAINING_ROOT),
            "cfg_dict": _plain_data(self.cfg),
            "algo_name": str(getattr(self.cfg, "algo_name", "iql")),
            "env_metadata": self.env_metadata,
            "shape_metadata": self.shape_metadata,
            "obs_slices": {k: list(v) for k, v in self.obs_slices.items()},
            "obs_normalization_stats": self._obs_normalization_stats,
            "action_normalization_stats": self._action_normalization_stats,
            "module_snapshot": {mod: _to_cpu_state_dict(sd) for mod, sd in self.algo._module_state_dict().items()},
            "horizon": horizon,
            "terminate_on_success": terminate_on_success,
            "stochastic": bool(self.eval_cfg.get("stochastic", False)),
            "video_enabled": video_enabled,
            "video_cfg": video_cfg,
            "max_video_episodes": max_video_episodes,
            "global_step": global_step,
            "save_dir": str(self.save_dir),
            "worker_device": self.worker_device,
            "env_name_override": self.eval_cfg.get("env_name_override"),
        }
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=self.num_workers, initializer=_worker_initializer, initargs=(init_args,)) as pool:
            all_stats: list[dict] = []
            for stat in tqdm(
                pool.imap_unordered(_rollout_worker_fn, range(n_rollouts)),
                total=n_rollouts,
                desc=f"[eval step={global_step}]",
                unit="rollout",
                leave=False,
            ):
                all_stats.append(stat)
        all_stats.sort(key=lambda x: x["rollout_idx"])
        rollout_stats = [{k: v for k, v in s.items() if k not in ("rollout_idx", "video_path")} for s in all_stats]
        video_paths = [s["video_path"] for s in all_stats if "video_path" in s]
        return rollout_stats, video_paths

    def _run_serial(
        self,
        global_step: int,
        n_rollouts: int,
        horizon: int,
        terminate_on_success: bool,
        video_enabled: bool,
        max_video_episodes: int,
        video_cfg: dict,
    ) -> tuple[list[dict], list[str]]:
        """Run rollouts serially on the live algo (existing behaviour)."""
        rollout_stats: list[dict] = []
        video_paths: list[str] = []

        for rollout_idx in tqdm(range(n_rollouts), desc=f"[eval step={global_step}]", unit="rollout", leave=False):
            video_writer = None
            video_path = None
            if video_enabled and rollout_idx < max_video_episodes:
                video_path = self._video_path_for_rollout(global_step=global_step, rollout_idx=rollout_idx)
                _, video_writer = _make_video_writer(video_path, {
                    "path": video_path,
                    "skip": int(video_cfg.get("skip", 5)),
                    "fps": int(video_cfg.get("fps", 20)),
                    "frame_height": int(video_cfg.get("frame_height", 512)),
                    "frame_width": int(video_cfg.get("frame_width", 512)),
                    "camera_names": list(video_cfg.get("camera_names", ["agentview"])),
                })

            try:
                rollout_stats.append(
                    rollout(
                        policy=self.policy,
                        env=self.env,
                        horizon=horizon,
                        video_writer=video_writer,
                        video_skip=int(video_cfg.get("skip", 5)),
                        camera_names=list(video_cfg.get("camera_names", ["agentview"])),
                        frame_height=int(video_cfg.get("frame_height", 512)),
                        frame_width=int(video_cfg.get("frame_width", 512)),
                        terminate_on_success=terminate_on_success,
                    )
                )
            finally:
                if video_writer is not None:
                    try:
                        video_writer.close()
                    except Exception as exc:
                        print(f"WARNING: failed to finalize video writer cleanly: {exc}")
                    if video_path is not None and os.path.isfile(video_path):
                        video_paths.append(video_path)

        return rollout_stats, video_paths

    def run(self, global_step: int) -> dict[str, float]:
        n_rollouts = int(self.eval_cfg.get("n_rollouts", 0))
        horizon = int(self.eval_cfg.get("horizon", 0))
        terminate_on_success = bool(self.eval_cfg.get("terminate_on_success", True))
        video_cfg = dict(self.eval_cfg.get("video", {}))
        video_enabled = bool(video_cfg.get("enabled", False))
        max_video_episodes = max(0, int(video_cfg.get("max_episodes", 1)))

        started = time.time()

        if self.num_workers > 1:
            rollout_stats, video_paths = self._run_parallel(
                global_step=global_step,
                n_rollouts=n_rollouts,
                horizon=horizon,
                terminate_on_success=terminate_on_success,
                video_enabled=video_enabled,
                max_video_episodes=max_video_episodes,
                video_cfg=video_cfg,
            )
        else:
            rollout_stats, video_paths = self._run_serial(
                global_step=global_step,
                n_rollouts=n_rollouts,
                horizon=horizon,
                terminate_on_success=terminate_on_success,
                video_enabled=video_enabled,
                max_video_episodes=max_video_episodes,
                video_cfg=video_cfg,
            )

        self.last_video_paths = video_paths
        self.last_progress_path = None
        if video_enabled and video_paths:
            self.last_progress_path = self._build_progress_plot(
                global_step=global_step,
                video_path=video_paths[-1],
                video_cfg=video_cfg,
            )

        elapsed_minutes = float((time.time() - started) / 60.0)
        summary = _summary_from_rollouts(
            rollout_stats=rollout_stats,
            checkpoint_path="<live_algo>",
            global_step=int(global_step),
        )
        self._write_json_output(global_step=global_step, summary=summary, rollout_stats=rollout_stats)

        return {
            "eval/return": float(summary["Return"]),
            "eval/horizon": float(summary["Horizon"]),
            "eval/success_rate": float(summary["Success_Rate"]),
            "eval/num_success": float(summary["Num_Success"]),
            "eval/num_rollouts": float(summary["Num_Rollouts"]),
            "eval/time_minutes": elapsed_minutes,
        }


# ---------------------------------------------------------------------------
# Dataset value evaluator
# ---------------------------------------------------------------------------

class DatasetValueEvaluator:
    """Compute dataset-based IQL value diagnostics for multiple mask subsets.

    For each configured mask, builds a RobomimicReplayBuffer (without its own
    normalisation) and then applies the training buffer's normalisation stats so
    obs/action scales match exactly what the critic and V-net were trained on.

    Metric semantics
    ----------------
    q          = mean(current critic heads) per transition
    advantage  = min(target critic heads) - V(obs)  per transition

    Returns scalar metrics and saves shared-bin, normalised-density histograms.
    """

    def __init__(
        self,
        algo: Any,
        cfg: SimpleNamespace,
        obs_normalization_stats: Any,
        action_normalization_stats: Any,
        eval_cfg: Any,
        save_dir: str,
    ) -> None:
        self.algo = algo
        self.cfg = cfg
        self._obs_normalization_stats = obs_normalization_stats
        self._action_normalization_stats = action_normalization_stats
        # Expose as a plain dict so learn_offline can read the trigger cadence.
        self.eval_cfg = _plain_data(eval_cfg)
        self.save_dir = Path(save_dir)

        value_cfg = (self.eval_cfg.get("value") or {})

        configured_masks = value_cfg.get("masks") or None
        if not configured_masks:
            filter_key = str(getattr(cfg.dataset, "filter_key", "") or "")
            configured_masks = [filter_key] if filter_key else []
        self.masks: list[str] = [str(m) for m in configured_masks if m]

        self.batch_size = int(value_cfg.get("batch_size", 4096))
        self.histogram_bins = int(value_cfg.get("histogram_bins", 80))
        self.advantage_temp = float(getattr(self.algo, "advantage_temp", 5.0))
        self.clip_score = float(getattr(self.algo, "clip_score", 100.0))

        output_dir_cfg = value_cfg.get("output_dir") or None
        if output_dir_cfg:
            out = Path(str(output_dir_cfg)).expanduser()
            if not out.is_absolute():
                out = (self.save_dir / out).resolve()
            self.output_dir = out
        else:
            self.output_dir = self.save_dir / "value_eval"

        self._eval_index = 0
        self._adv_weight_curve_steps: list[int] = []
        self._adv_weight_curve_by_mask: dict[str, list[float]] = {
            mask: [] for mask in self.masks
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_eval_buffer(self, mask: str):
        """Load mask data into a buffer and apply training normalisation."""
        from datasets.robomimic import RobomimicReplayBuffer

        buf = RobomimicReplayBuffer(
            h5_path=str(self.cfg.dataset.h5_path),
            obs_keys=list(self.cfg.dataset.obs_keys),
            filter_key=mask,
            device="cpu",
            action_keys=getattr(self.cfg.dataset, "action_keys", None),
            strict_next_obs=bool(getattr(self.cfg.dataset, "strict_next_obs", True)),
            normalize_obs=False,
            normalize_actions=False,
        )

        norm_clip = getattr(self.cfg.dataset, "normalization_clip", None)

        if self._obs_normalization_stats is not None:
            stats = self._obs_normalization_stats.get("observations") or {}
            if stats:
                offset = stats.get("offset", 0)
                scale = stats.get("scale", 1)
                buf.observations = (buf.observations - offset) / scale
                buf.next_observations = (buf.next_observations - offset) / scale
                if norm_clip is not None:
                    buf.observations = np.clip(buf.observations, -norm_clip, norm_clip)
                    buf.next_observations = np.clip(buf.next_observations, -norm_clip, norm_clip)

        if self._action_normalization_stats is not None:
            stats = self._action_normalization_stats.get("actions") or {}
            if stats:
                offset = stats.get("offset", 0)
                scale = stats.get("scale", 1)
                buf.actions = (buf.actions - offset) / scale
                if norm_clip is not None:
                    buf.actions = np.clip(buf.actions, -norm_clip, norm_clip)

        return buf

    def _evaluate_buffer(
        self, buf, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Iterate all transitions exactly once and return q / v / advantage / advantage-weight."""
        n = buf.size()
        q_all: list[np.ndarray] = []
        v_all: list[np.ndarray] = []
        adv_all: list[np.ndarray] = []
        adv_weight_all: list[np.ndarray] = []

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            obs = torch.as_tensor(buf.observations[start:end], device=device)
            actions = torch.as_tensor(buf.actions[start:end], device=device)

            with torch.no_grad():
                # q = mean(current critic heads) per transition
                q_heads = self.algo.critic(obs, actions)          # list of [B,1]
                q_stack = torch.cat(q_heads, dim=1)               # [B, n_critics]
                q_per_transition = q_stack.mean(dim=1)            # [B]

                # advantage = min(target critic heads) - V(obs)
                qt_heads = self.algo.critic_target(obs, actions)  # list of [B,1]
                qt_stack = torch.cat(qt_heads, dim=1)             # [B, n_critics]
                min_target_q = qt_stack.min(dim=1).values         # [B]
                v_pred = self.algo.v_net(obs).squeeze(-1)         # [B]
                adv_per_transition = min_target_q - v_pred        # [B]
                adv_weight = torch.clamp(
                    torch.exp(adv_per_transition * self.advantage_temp),
                    0.0,
                    self.clip_score,
                )

            q_all.append(q_per_transition.cpu().numpy())
            v_all.append(v_pred.cpu().numpy())
            adv_all.append(adv_per_transition.cpu().numpy())
            adv_weight_all.append(adv_weight.cpu().numpy())

        return (
            np.concatenate(q_all),
            np.concatenate(v_all),
            np.concatenate(adv_all),
            np.concatenate(adv_weight_all),
        )

    def _save_advantage_curve(self, global_step: int) -> str | None:
        """Save a multi-mask curve of average advantage weight vs eval index."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._adv_weight_curve_steps:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        x = np.asarray(self._adv_weight_curve_steps, dtype=np.int32)

        fig, ax = plt.subplots(figsize=(8, 4))
        for mask_name in self.masks:
            y_list = self._adv_weight_curve_by_mask.get(mask_name, [])
            if not y_list:
                continue
            y = np.asarray(y_list, dtype=np.float32)
            valid = np.isfinite(y)
            if not np.any(valid):
                continue
            ax.plot(x[valid], y[valid], marker="o", linewidth=1.5, markersize=3, label=mask_name)

        ax.set_title(f"Average Advantage Weight by Mask (step={global_step})")
        ax.set_xlabel("Eval Step Index")
        ax.set_ylabel("Average Advantage Weight")
        ax.grid(alpha=0.3)
        if self.masks:
            ax.legend()

        out_path = str(self.output_dir / f"advantage_curve_step_{global_step:09d}.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _save_histograms(
        self,
        q_data: dict[str, np.ndarray],
        v_data: dict[str, np.ndarray],
        adv_data: dict[str, np.ndarray],
        global_step: int,
    ) -> tuple[str, str, str]:
        """Save shared-bin, density-normalised multi-mask histograms as PNGs."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.output_dir.mkdir(parents=True, exist_ok=True)

        def _plot_hist(
            data_by_mask: dict[str, np.ndarray],
            title: str,
            xlabel: str,
            filename: str,
        ) -> str:
            all_vals = np.concatenate(list(data_by_mask.values()))
            v_min, v_max = float(all_vals.min()), float(all_vals.max())
            if v_min == v_max:
                v_max = v_min + 1.0
            bins = np.linspace(v_min, v_max, self.histogram_bins + 1)

            fig, ax = plt.subplots(figsize=(8, 4))
            for mask_name, vals in data_by_mask.items():
                ax.hist(vals, bins=bins, density=True, alpha=0.6, label=mask_name)
            ax.set_title(f"{title} (step={global_step})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.legend()
            out_path = str(self.output_dir / filename)
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return out_path

        q_path = _plot_hist(
            q_data,
            "Q-Value Distribution",
            "Q-value",
            f"q_value_dist_step_{global_step:09d}.png",
        )
        v_path = _plot_hist(
            v_data,
            "V-Value Distribution",
            "V-value",
            f"v_value_dist_step_{global_step:09d}.png",
        )
        adv_path = _plot_hist(
            adv_data,
            "Advantage Distribution",
            "Advantage",
            f"advantage_dist_step_{global_step:09d}.png",
        )
        return q_path, v_path, adv_path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self, global_step: int
    ) -> tuple[dict[str, float], str | None, str | None, str | None, str | None]:
        """Evaluate all configured masks.

        Returns
        -------
        metrics : dict[str, float]
            Scalar metrics keyed as ``eval/average_q_values/{mask}``,
            ``eval/average_advantage/{mask}``, ``eval/value_num_transitions/{mask}``.
        q_hist_path : str | None
            Path to the saved Q-value histogram PNG, or None on failure.
        v_hist_path : str | None
            Path to the saved V-value histogram PNG, or None on failure.
        adv_hist_path : str | None
            Path to the saved advantage histogram PNG, or None on failure.
        adv_curve_path : str | None
            Path to the saved multi-mask advantage-weight curve PNG, or None on failure.
        """
        if not self.masks:
            return {}, None, None, None, None

        device = self.algo.device
        metrics: dict[str, float] = {}
        q_data: dict[str, np.ndarray] = {}
        v_data: dict[str, np.ndarray] = {}
        adv_data: dict[str, np.ndarray] = {}

        for mask in self.masks:
            try:
                buf = self._build_eval_buffer(mask)
                q_vals, v_vals, adv_vals, adv_weight_vals = self._evaluate_buffer(buf, device)
                metrics[f"eval/average_q_values/{mask}"] = float(q_vals.mean())
                metrics[f"eval/average_v_values/{mask}"] = float(v_vals.mean())
                metrics[f"eval/average_advantage/{mask}"] = float(adv_vals.mean())
                metrics[f"eval/average_advantage_weight/{mask}"] = float(adv_weight_vals.mean())
                metrics[f"eval/value_num_transitions/{mask}"] = float(len(q_vals))
                q_data[mask] = q_vals
                v_data[mask] = v_vals
                adv_data[mask] = adv_vals
            except Exception as exc:
                print(f"[DatasetValueEvaluator] Failed to evaluate mask '{mask}': {exc}")

        q_hist_path: str | None = None
        v_hist_path: str | None = None
        adv_hist_path: str | None = None
        adv_curve_path: str | None = None
        if q_data:
            try:
                q_hist_path, v_hist_path, adv_hist_path = self._save_histograms(
                    q_data, v_data, adv_data, global_step
                )

                self._eval_index += 1
                self._adv_weight_curve_steps.append(self._eval_index)
                for mask in self.masks:
                    metric_key = f"eval/average_advantage_weight/{mask}"
                    self._adv_weight_curve_by_mask.setdefault(mask, []).append(
                        float(metrics.get(metric_key, float("nan")))
                    )
                adv_curve_path = self._save_advantage_curve(global_step)
            except Exception as exc:
                print(f"[DatasetValueEvaluator] Failed to save histograms: {exc}")

        return metrics, q_hist_path, v_hist_path, adv_hist_path, adv_curve_path
