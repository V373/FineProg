#!/usr/bin/env python3
"""Evaluate a strict-schema policy_training IQL checkpoint in robomimic / robosuite."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from envs.robomimic import create_robomimic_env
from utils.eval_utils import (
    ObservationAdapter,
    SB3IQLRolloutPolicy,
    _apply_cli_overrides,
    _is_video_enabled,
    _load_eval_config,
    _make_video_writer,
    _normalization_stats_from_payload,
    _resolve_auto_eval_paths,
    _resolve_path,
    _summary_from_rollouts,
    _visual_keys_from_checkpoint,
    _write_json,
    load_checkpoint_for_eval,
    rollout,
)
from utils.logger import namespace_to_dict, seed_everything


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a strict-schema policy_training checkpoint")
    parser.add_argument("--config", type=str, default="configs/evaluate_policy.yaml", help="Eval YAML path")
    parser.add_argument("--agent", type=str, default=None, help="Checkpoint path override")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    parser.add_argument("--seed", type=int, default=None, help="Rollout seed override")
    parser.add_argument("--n_rollouts", type=int, default=None, help="Number of rollout episodes override")
    parser.add_argument("--horizon", type=int, default=None, help="Rollout horizon override")
    parser.add_argument("--env", type=str, default=None, help="Env name override")
    parser.add_argument("--video_path", type=str, default=None, help="Single mp4 output path override")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording for this run")
    parser.add_argument("--video_skip", type=int, default=None, help="Video frame skip override")
    parser.add_argument("--video_fps", type=int, default=None, help="Video fps override")
    parser.add_argument("--frame_height", type=int, default=None, help="Rendered frame height override")
    parser.add_argument("--frame_width", type=int, default=None, help="Rendered frame width override")
    parser.add_argument("--camera_names", nargs="+", default=None, help="Camera names override")
    parser.add_argument("--json_path", type=str, default=None, help="JSON output path override")
    parser.add_argument("--output_dir", type=str, default=None, help="Root directory for auto-inferred outputs")
    parser.add_argument("--stochastic", action="store_true", default=None, help="Use stochastic policy actions")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime_cfg = _load_eval_config(args.config)
    eval_cfg = _apply_cli_overrides(runtime_cfg, args)

    agent_path = _resolve_path(str(eval_cfg["agent"]), base_dir=runtime_cfg.config_dir)
    loaded = load_checkpoint_for_eval(agent_path, str(eval_cfg["device"]))

    seed_value = int(eval_cfg["seed"])
    seed_everything(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    obs_stats, action_stats = _normalization_stats_from_payload(loaded.payload)
    obs_adapter = ObservationAdapter(
        cfg=loaded.cfg,
        shape_metadata=loaded.shape_metadata,
        obs_slices=loaded.obs_slices,
        device=loaded.device,
        obs_normalization_stats=obs_stats,
    )
    policy = SB3IQLRolloutPolicy(
        algo=loaded.algo,
        obs_adapter=obs_adapter,
        stochastic=bool(eval_cfg["policy"]["stochastic"]),
        action_normalization_stats=action_stats,
    )

    video_cfg = eval_cfg["video"]
    rollout_cfg = eval_cfg["rollout"]
    n_rollouts = int(rollout_cfg["n_rollouts"])
    horizon = int(rollout_cfg["horizon"])
    if n_rollouts < 0:
        raise ValueError("rollout.n_rollouts must be >= 0")
    if horizon <= 0:
        raise ValueError("rollout.horizon must be > 0")

    # Resolve output paths now that the checkpoint is loaded — we need
    # `loaded.payload["config"]` and `loaded.payload["global_step"]` to infer
    # env/task/split/run/checkpoint metadata for the auto path.
    output_plan = _resolve_auto_eval_paths(
        eval_cfg=eval_cfg,
        loaded=loaded,
        config_dir=runtime_cfg.config_dir,
        n_rollouts=n_rollouts,
        horizon=horizon,
    )
    video_path = output_plan["video"]["path"]
    json_path = output_plan["json"]["path"]
    output_metadata = output_plan["metadata"]
    output_paths_record = {
        "output_dir": output_plan["output_dir"],
        "video_path": video_path,
        "video_auto_path": output_plan["video"]["auto_path"],
        "json_path": json_path,
        "json_auto_path": output_plan["json"]["auto_path"],
    }
    print(f"[evaluate_policy] video_path = {video_path}")
    print(f"[evaluate_policy] json_path  = {json_path}")

    video_path, video_writer = _make_video_writer(video_path, video_cfg)

    visual_keys = _visual_keys_from_checkpoint(loaded.cfg, loaded.shape_metadata)
    env = create_robomimic_env(
        env_meta=loaded.env_metadata,
        obs_keys=list(loaded.cfg.dataset.obs_keys),
        visual_keys=visual_keys,
        env_name=eval_cfg["env"]["name_override"],
        render=False,
        render_offscreen=(video_writer is not None),
    )

    rollout_stats: list[dict[str, Any]] = []
    for _ in range(n_rollouts):
        rollout_stats.append(
            rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                video_writer=video_writer,
                video_skip=int(video_cfg["skip"]),
                camera_names=list(video_cfg["camera_names"]),
                frame_height=int(video_cfg["frame_height"]),
                frame_width=int(video_cfg["frame_width"]),
            )
        )

    if video_writer is not None:
        try:
            video_writer.close()
        except Exception as exc:
            print(f"WARNING: failed to finalize video writer cleanly: {exc}")

    summary = _summary_from_rollouts(
        rollout_stats=rollout_stats,
        checkpoint_path=loaded.checkpoint_path,
        global_step=int(loaded.payload["global_step"]),
    )
    summary["Num_Rollouts"] = n_rollouts
    print(json.dumps(summary, indent=2))

    if json_path is not None:
        _write_json(
            json_path,
            {
                "summary": summary,
                "per_rollout_stats": rollout_stats,
                "checkpoint": {
                    "path": loaded.checkpoint_path,
                    "schema_version": loaded.payload["checkpoint_schema_version"],
                    "global_step": int(loaded.payload["global_step"]),
                    "algo_name": loaded.payload["algo_name"],
                },
                "eval_config": eval_cfg,
                "training_config": namespace_to_dict(loaded.cfg),
                "env_metadata": loaded.env_metadata,
                "shape_metadata": loaded.shape_metadata,
                "obs_slices": loaded.obs_slices,
                "video_path": video_path,
                "output_metadata": output_metadata,
                "output_paths": output_paths_record,
            },
        )


if __name__ == "__main__":
    main()
