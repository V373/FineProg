"""Training helpers for policy_training – loaded by train_policy.py."""

from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from utils.checkpoints import CHECKPOINT_SCHEMA_VERSION
from utils.algo_config import get_algo_cfg
from utils.logger import namespace_to_dict


def _select_first_demo_id(data_grp: Any, filter_key: str | None) -> str:
    demo_ids = sorted(list(data_grp.keys()))
    if filter_key:
        if "mask" not in data_grp.file or filter_key not in data_grp.file["mask"]:
            raise KeyError(f"mask/{filter_key} not found in {data_grp.file.filename}")
        selected = data_grp.file["mask"][filter_key][()]
        selected_ids = {
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in selected
        }
        demo_ids = [demo_id for demo_id in demo_ids if demo_id in selected_ids]
    if not demo_ids:
        raise ValueError(f"No demos available in {data_grp.file.filename}")
    return demo_ids[0]


def extract_dataset_metadata(cfg: Any) -> dict:
    """Extract dataset structure needed for checkpoint metadata without loading replay data."""
    with h5py.File(str(cfg.dataset.h5_path), "r") as h5_file:
        data_grp = h5_file["data"]
        first_demo_id = _select_first_demo_id(data_grp, cfg.dataset.filter_key)
        demo_grp = data_grp[first_demo_id]
        obs_shapes = {
            key: list(demo_grp["obs"][key].shape[1:])
            for key in cfg.dataset.obs_keys
        }
        if getattr(cfg.dataset, "action_keys", None):
            action_dim = sum(int(demo_grp[key].shape[1]) for key in cfg.dataset.action_keys)
        else:
            action_dim = int(demo_grp["actions"].shape[1])
        env_args_raw = data_grp.attrs.get("env_args")
        env_metadata = None
        if env_args_raw is not None:
            if isinstance(env_args_raw, bytes):
                env_args_raw = env_args_raw.decode("utf-8", errors="ignore")
            env_metadata = json.loads(str(env_args_raw))
        if env_metadata is None:
            raise ValueError(
                "Training dataset is missing data.attrs['env_args']; cannot build self-contained eval checkpoint."
            )

        obs_slices = {}
        cursor = 0
        for key in cfg.dataset.obs_keys:
            shape = demo_grp["obs"][key].shape[1:]
            dim = int(np.prod(shape, dtype=np.int64)) if len(shape) > 0 else 1
            obs_slices[key] = [cursor, cursor + dim]
            cursor += dim

    visual_obs_keys: list[str] = []
    algo_cfg = get_algo_cfg(cfg)
    if getattr(algo_cfg, "features_extractor_type", "flat_range") == "resnet18conv":
        visual_specs = getattr(getattr(algo_cfg, "features_extractor_kwargs", None), "visual_specs", {})
        if hasattr(visual_specs, "keys"):
            visual_obs_keys = list(visual_specs.keys())
        else:
            visual_obs_keys = list(vars(visual_specs).keys())
    low_dim_obs_keys = [key for key in cfg.dataset.obs_keys if key not in set(visual_obs_keys)]

    return {
        "env_metadata": env_metadata,
        "shape_metadata": {
            "action_dim": action_dim,
            "action_shape": [action_dim],
            "observation_dim": int(sum(int(v[1] - v[0]) for v in obs_slices.values())),
            "obs_shapes": obs_shapes,
            "low_dim_obs_keys": low_dim_obs_keys,
            "visual_obs_keys": visual_obs_keys,
            "use_image_obs": bool(visual_obs_keys),
            "first_demo_id": first_demo_id,
        },
        "obs_slices": obs_slices,
    }


def build_checkpoint_metadata(cfg: Any, replay_buffer: Any | None = None, dataset_metadata: dict | None = None) -> dict:
    algo_cfg = get_algo_cfg(cfg)
    if dataset_metadata is None:
        dataset_metadata = extract_dataset_metadata(cfg)

    env_metadata = dataset_metadata["env_metadata"]
    shape_metadata = dict(dataset_metadata["shape_metadata"])
    obs_slices = dict(dataset_metadata["obs_slices"])

    if replay_buffer is not None:
        shape_metadata["observation_dim"] = int(replay_buffer.observation_space.shape[0])
        if replay_buffer.obs_slices:
            obs_slices = {
                key: [int(obs_slice.start), int(obs_slice.stop)]
                for key, obs_slice in replay_buffer.obs_slices.items()
            }

    visual_obs_keys: list[str] = []
    if getattr(algo_cfg, "features_extractor_type", "flat_range") == "resnet18conv":
        visual_specs = getattr(getattr(algo_cfg, "features_extractor_kwargs", None), "visual_specs", {})
        if hasattr(visual_specs, "keys"):
            visual_obs_keys = list(visual_specs.keys())
        else:
            visual_obs_keys = list(vars(visual_specs).keys())
    low_dim_obs_keys = [key for key in cfg.dataset.obs_keys if key not in set(visual_obs_keys)]

    normalization_stats = {}
    if replay_buffer is not None and replay_buffer.obs_normalization_stats is not None:
        normalization_stats["obs"] = replay_buffer.obs_normalization_stats
    if replay_buffer is not None and replay_buffer.action_normalization_stats is not None:
        normalization_stats["actions"] = replay_buffer.action_normalization_stats

    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algo_name": str(cfg.algo_name),
        "config": namespace_to_dict(cfg),
        "env_metadata": env_metadata,
        "shape_metadata": shape_metadata,
        "obs_slices": obs_slices,
        **({"normalization_stats": normalization_stats} if normalization_stats else {}),
    }
