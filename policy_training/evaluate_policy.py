#!/usr/bin/env python3
"""Evaluate a strict-schema policy_training IQL checkpoint in robomimic / robosuite."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import imageio_ffmpeg
import numpy as np
import torch
from tqdm import tqdm

from envs.robomimic import create_robomimic_env
from utils.eval_utils import (
    POLICY_TRAINING_ROOT,
    ObservationAdapter,
    SB3IQLRolloutPolicy,
    _plain_data,
    _apply_cli_overrides,
    _is_video_enabled,
    _load_eval_config,
    _make_video_writer,
    _normalization_stats_from_payload,
    _resolve_parallel_worker_device,
    _rollout_worker_fn,
    _resolve_auto_eval_paths,
    _resolve_path,
    _summary_from_rollouts,
    _to_cpu_state_dict,
    _visual_keys_from_checkpoint,
    _worker_initializer,
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
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seed list override (e.g. --seeds 0 1 2)")
    parser.add_argument("--n_rollouts", type=int, default=None, help="Number of rollout episodes override")
    parser.add_argument("--horizon", type=int, default=None, help="Rollout horizon override")
    parser.add_argument("--num_workers", type=int, default=None, help="Parallel rollout workers (1 disables parallel)")
    parser.add_argument("--worker_device", type=str, default=None, help="Worker device for parallel eval (cpu | cuda | auto)")
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


def _normalize_eval_seeds(seed_value: Any) -> list[int]:
    if isinstance(seed_value, (list, tuple)):
        seeds = [int(value) for value in seed_value]
    else:
        seeds = [int(seed_value)]
    if not seeds:
        raise ValueError("Evaluation seed list is empty.")
    return seeds


def _run_rollouts_serial(
    *,
    policy: SB3IQLRolloutPolicy,
    env: Any,
    n_rollouts: int,
    horizon: int,
    video_writer: Any,
    video_cfg: dict[str, Any],
    progress_desc: str,
) -> list[dict[str, Any]]:
    rollout_stats: list[dict[str, Any]] = []
    for _ in tqdm(range(n_rollouts), desc=progress_desc, unit="rollout", dynamic_ncols=True):
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
    return rollout_stats


def _run_rollouts_parallel(
    *,
    loaded,
    eval_cfg: dict[str, Any],
    n_rollouts: int,
    horizon: int,
    output_dir: str,
    video_enabled: bool,
    video_dir: str | None,
    max_video_episodes: int,
    progress_desc: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    if n_rollouts <= 0:
        return [], []

    rollout_cfg = eval_cfg.get("rollout", {}) or {}
    num_workers = max(1, int(rollout_cfg.get("num_workers", 1)))
    obs_stats, action_stats = _normalization_stats_from_payload(loaded.payload)
    init_args = {
        "project_root": str(POLICY_TRAINING_ROOT),
        "cfg_dict": _plain_data(loaded.cfg),
        "algo_name": str(loaded.payload["algo_name"]),
        "env_metadata": loaded.env_metadata,
        "shape_metadata": loaded.shape_metadata,
        "obs_slices": {k: list(v) for k, v in loaded.obs_slices.items()},
        "obs_normalization_stats": obs_stats,
        "action_normalization_stats": action_stats,
        "module_snapshot": {
            mod: _to_cpu_state_dict(sd)
            for mod, sd in loaded.algo._module_state_dict().items()
        },
        "horizon": int(horizon),
        "terminate_on_success": bool(rollout_cfg.get("terminate_on_success", True)),
        "stochastic": bool(eval_cfg["policy"]["stochastic"]),
        "video_enabled": bool(video_enabled),
        "video_cfg": {
            "dir": video_dir,
            "skip": int(eval_cfg["video"]["skip"]),
            "fps": int(eval_cfg["video"]["fps"]),
            "frame_height": int(eval_cfg["video"]["frame_height"]),
            "frame_width": int(eval_cfg["video"]["frame_width"]),
            "camera_names": list(eval_cfg["video"]["camera_names"]),
        },
        "max_video_episodes": int(max_video_episodes),
        "global_step": int(loaded.payload["global_step"]),
        "save_dir": str(output_dir),
        "worker_device": _resolve_parallel_worker_device(
            num_workers,
            rollout_cfg.get("worker_device", "auto"),
        ),
        "env_name_override": eval_cfg["env"]["name_override"],
    }
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=_worker_initializer, initargs=(init_args,)) as pool:
        all_stats: list[dict[str, Any]] = []
        for stat in tqdm(
            pool.imap_unordered(_rollout_worker_fn, range(n_rollouts)),
            total=n_rollouts,
            desc=progress_desc,
            unit="rollout",
            dynamic_ncols=True,
        ):
            all_stats.append(stat)

    all_stats.sort(key=lambda item: item["rollout_idx"])
    rollout_stats = [{k: v for k, v in stat.items() if k not in {"rollout_idx", "video_path"}} for stat in all_stats]
    video_paths = [str(stat["video_path"]) for stat in all_stats if "video_path" in stat]
    return rollout_stats, video_paths


def _resolve_parallel_video_dir(video_path: str | None, output_dir: str) -> str:
    if video_path:
        candidate = Path(video_path)
        if candidate.suffix:
            return str(candidate.resolve().parent)
        return str(candidate.resolve())
    return str((Path(output_dir) / "videos").resolve())


def _resolve_max_video_episodes(max_episodes_value: Any, n_rollouts: int) -> int:
    if isinstance(max_episodes_value, str):
        token = max_episodes_value.strip().lower()
        if token == "all":
            return int(max(0, n_rollouts))
        raise ValueError("video.max_episodes must be an integer >= 0 or 'all'.")
    max_episodes = int(max_episodes_value)
    if max_episodes < 0:
        raise ValueError("video.max_episodes must be >= 0.")
    return int(min(max_episodes, max(0, n_rollouts)))


def _concatenate_videos_serial(video_paths: list[str], output_path: str) -> str | None:
    if not video_paths:
        return None

    resolved_output = str(Path(output_path).resolve())
    Path(resolved_output).parent.mkdir(parents=True, exist_ok=True)

    if len(video_paths) == 1:
        src = str(Path(video_paths[0]).resolve())
        if src != resolved_output:
            shutil.copy2(src, resolved_output)
        return resolved_output

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        concat_list_path = handle.name
        for path in video_paths:
            escaped = str(Path(path).resolve()).replace("'", r"'\\''")
            handle.write(f"file '{escaped}'\n")

    try:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-c",
            "copy",
            resolved_output,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"Failed to concatenate rollout videos with ffmpeg: {stderr}") from exc
    finally:
        try:
            Path(concat_list_path).unlink(missing_ok=True)
        except Exception:
            pass

    return resolved_output


def main() -> None:
    args = _parse_args()
    runtime_cfg = _load_eval_config(args.config)
    eval_cfg = _apply_cli_overrides(runtime_cfg, args)

    agent_path = _resolve_path(str(eval_cfg["agent"]), base_dir=runtime_cfg.config_dir)
    loaded = load_checkpoint_for_eval(agent_path, str(eval_cfg["device"]))

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
    num_workers = max(1, int(rollout_cfg.get("num_workers", 1)))
    if n_rollouts < 0:
        raise ValueError("rollout.n_rollouts must be >= 0")
    if horizon <= 0:
        raise ValueError("rollout.horizon must be > 0")
    if num_workers <= 0:
        raise ValueError("rollout.num_workers must be >= 1")

    seed_values = _normalize_eval_seeds(eval_cfg["seed"])
    all_seed_summaries: list[dict[str, Any]] = []

    for seed_value in seed_values:
        run_eval_cfg = deepcopy(eval_cfg)
        run_eval_cfg["seed"] = int(seed_value)

        seed_everything(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

        # Resolve output paths now that the checkpoint is loaded — we need
        # `loaded.payload["config"]` and `loaded.payload["global_step"]` to infer
        # env/task/split/run/checkpoint metadata for the auto path.
        output_plan = _resolve_auto_eval_paths(
            eval_cfg=run_eval_cfg,
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
        print(f"[evaluate_policy][seed={seed_value}] video_path = {video_path}")
        print(f"[evaluate_policy][seed={seed_value}] json_path  = {json_path}")

        parallel_enabled = num_workers > 1
        video_enabled = _is_video_enabled(video_cfg)

        if parallel_enabled:
            max_video_episodes = _resolve_max_video_episodes(video_cfg.get("max_episodes", 1), n_rollouts) if video_enabled else 0
            parallel_video_dir = _resolve_parallel_video_dir(video_path, output_plan["output_dir"])
            rollout_stats, video_paths = _run_rollouts_parallel(
                loaded=loaded,
                eval_cfg=run_eval_cfg,
                n_rollouts=n_rollouts,
                horizon=horizon,
                output_dir=output_plan["output_dir"],
                video_enabled=video_enabled,
                video_dir=parallel_video_dir,
                max_video_episodes=max_video_episodes,
                progress_desc=f"Eval seed={seed_value}",
            )
            if video_enabled and video_path is not None:
                print(
                    "[evaluate_policy] parallel+video writes per-rollout mp4 files; "
                    f"using directory: {parallel_video_dir}"
                )
            video_path_record = None
            if video_enabled and video_path is not None and video_paths:
                try:
                    video_path_record = _concatenate_videos_serial(video_paths, video_path)
                    print(
                        "[evaluate_policy] merged rollout videos into: "
                        f"{video_path_record}"
                    )
                except Exception as exc:
                    print(
                        "WARNING: failed to concatenate parallel rollout videos; "
                        f"keeping per-rollout files only. error={exc}"
                    )
            video_paths_record = video_paths
        else:
            video_path, video_writer = _make_video_writer(video_path, video_cfg)
            visual_keys = _visual_keys_from_checkpoint(loaded.cfg, loaded.shape_metadata)
            env = create_robomimic_env(
                env_meta=loaded.env_metadata,
                obs_keys=list(loaded.cfg.dataset.obs_keys),
                visual_keys=visual_keys,
                env_name=run_eval_cfg["env"]["name_override"],
                render=False,
                render_offscreen=(video_writer is not None),
            )

            try:
                rollout_stats = _run_rollouts_serial(
                    policy=policy,
                    env=env,
                    n_rollouts=n_rollouts,
                    horizon=horizon,
                    video_writer=video_writer,
                    video_cfg=video_cfg,
                    progress_desc=f"Eval seed={seed_value}",
                )
            finally:
                if video_writer is not None:
                    try:
                        video_writer.close()
                    except Exception as exc:
                        print(f"WARNING: failed to finalize video writer cleanly: {exc}")
            video_path_record = video_path
            video_paths_record = [video_path] if video_path is not None else []

        summary = _summary_from_rollouts(
            rollout_stats=rollout_stats,
            checkpoint_path=loaded.checkpoint_path,
            global_step=int(loaded.payload["global_step"]),
        )
        summary["Num_Rollouts"] = n_rollouts
        summary["Eval_Seed"] = int(seed_value)
        print(json.dumps(summary, indent=2))
        all_seed_summaries.append(summary)

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
                    "eval_config": run_eval_cfg,
                    "training_config": namespace_to_dict(loaded.cfg),
                    "env_metadata": loaded.env_metadata,
                    "shape_metadata": loaded.shape_metadata,
                    "obs_slices": loaded.obs_slices,
                    "video_path": video_path_record,
                    "video_paths": video_paths_record,
                    "output_metadata": output_metadata,
                    "output_paths": output_paths_record,
                },
            )

    if len(all_seed_summaries) > 1:
        print("[evaluate_policy] finished all seeds")
        print(json.dumps({"seed_summaries": all_seed_summaries}, indent=2))


if __name__ == "__main__":
    main()
