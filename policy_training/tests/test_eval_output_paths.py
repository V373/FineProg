"""Unit tests for the auto output-path inference in evaluate_policy.py."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from utils.eval_utils import (
    CheckpointRuntime,
    _apply_cli_overrides,
    _checkpoint_tag_from_path,
    _coerce_cfg,
    _derive_run_id_from_checkpoint_path,
    _infer_eval_output_metadata,
    _is_video_enabled,
    _resolve_auto_eval_paths,
    _resolve_eval_output_path,
)


def _make_loaded(payload: dict, checkpoint_path: str) -> CheckpointRuntime:
    return CheckpointRuntime(
        checkpoint_path=checkpoint_path,
        payload=payload,
        cfg=_coerce_cfg(payload),
        env_metadata={},
        shape_metadata={"observation_dim": 1, "action_dim": 1, "visual_obs_keys": []},
        obs_slices={},
        algo=SimpleNamespace(),
        device=None,
    )


def _build_old_layout_payload(h5_path: str | None = None) -> dict:
    if h5_path is None:
        h5_path = (
            "/home/user/zhangzk/projects/fineprog/policy_training/"
            "datasets/robomimic/can/mh/reward_labeled/resnet18feats/"
            "image_2view_v15_reward_labeled_PBRS_resnet18feats.hdf5"
        )
    return {
        "algo_name": "iql",
        "global_step": 100000,
        "config": {
            "algo_name": "iql",
            "seed": 1,
            "dataset": {
                "h5_path": h5_path,
                "filter_key": "IQL_expert",
                "obs_keys": ["agentview_image"],
            },
        },
    }


def _build_new_layout_payload() -> dict:
    return {
        "algo_name": "iql",
        "global_step": 250000,
        "config": {
            "algo_name": "iql",
            "seed": 3,
            "dataset": {
                "h5_path": (
                    "/home/user/zhangzk/projects/fineprog/policy_training/"
                    "datasets/robomimic/can/mg/demo_v15.hdf5"
                ),
                "filter_key": "IQL_expert",
            },
        },
    }


def test_infer_metadata_old_layout():
    ckpt = "checkpoint/robomimic/can_mh/iql_expert_seed1_20260613_042513/step_100000.pt"
    payload = _build_old_layout_payload()
    loaded = _make_loaded(payload, ckpt)

    meta = _infer_eval_output_metadata(loaded, ckpt)

    assert meta["env_name"] == "robomimic"
    assert meta["task_name"] == "can"
    assert meta["split_name"] == "mh"
    assert meta["algo_mask_name"] == "IQL__IQL_expert"
    assert meta["train_seed"] == 1
    assert meta["seed_label"] == "seed1"
    assert meta["run_id"] == "20260613_042513"
    assert meta["checkpoint_tag"] == "step_100000"


def test_infer_metadata_new_layout():
    ckpt = (
        "checkpoint/robomimic/can/IQL__IQL_expert/"
        "20260613_134706/seed3/ckpt/final.pt"
    )
    payload = _build_new_layout_payload()
    loaded = _make_loaded(payload, ckpt)

    meta = _infer_eval_output_metadata(loaded, ckpt)

    assert meta["env_name"] == "robomimic"
    assert meta["task_name"] == "can"
    assert meta["split_name"] == "mg"
    assert meta["algo_mask_name"] == "IQL__IQL_expert"
    assert meta["train_seed"] == 3
    assert meta["seed_label"] == "seed3"
    assert meta["run_id"] == "20260613_134706"
    assert meta["checkpoint_tag"] == "final"


def test_infer_metadata_split_falls_through_known_subfolders():
    """``reward_labeled`` and ``resnet18feats`` are sub-folders, not splits."""
    ckpt = "checkpoint/robomimic/can_mh/iql_expert_seed1_20260613_042513/step_100000.pt"
    payload = _build_old_layout_payload()
    loaded = _make_loaded(payload, ckpt)

    meta = _infer_eval_output_metadata(loaded, ckpt)

    assert meta["split_name"] == "mh"


def test_infer_metadata_run_id_uses_parent_when_no_timestamp():
    """When no YYYYMMDD_HHMMSS token is in the path, fall back to parent dir name."""
    ckpt = "checkpoint/iql_can_mh_resnet18conv_feats_seed42/step_80000.pt"
    payload = _build_old_layout_payload()
    loaded = _make_loaded(payload, ckpt)

    meta = _infer_eval_output_metadata(loaded, ckpt)

    assert meta["run_id"] == "iql_can_mh_resnet18conv_feats_seed42"


def test_resolve_auto_paths_video_enabled_no_path(tmp_path):
    ckpt = "checkpoint/robomimic/can_mh/iql_expert_seed1_20260613_042513/step_100000.pt"
    payload = _build_old_layout_payload()
    loaded = _make_loaded(payload, ckpt)
    eval_cfg = {
        "video": {
            "enabled": True,
            "path": None,
            "skip": 5,
            "fps": 20,
            "frame_height": 512,
            "frame_width": 512,
            "camera_names": ["agentview"],
        },
        "output": {"dir": "outputs/eval", "json_path": None},
        "seed": 0,
    }

    plan = _resolve_auto_eval_paths(
        eval_cfg=eval_cfg,
        loaded=loaded,
        config_dir=tmp_path,
        n_rollouts=20,
        horizon=400,
    )

    expected_dir = (
        PROJECT_ROOT
        / "outputs/eval/robomimic/can/mh/IQL__IQL_expert/seed1/20260613_042513/step_100000"
    )
    assert Path(plan["output_dir"]) == expected_dir
    assert plan["video"]["enabled"] is True
    assert Path(plan["video"]["path"]) == (
        expected_dir / "eval_step_000100000_evalseed0_n020_h0400.mp4"
    )
    assert Path(plan["json"]["path"]) == (
        expected_dir / "eval_step_000100000_evalseed0_n020_h0400.json"
    )


def test_resolve_auto_paths_video_path_override(tmp_path):
    ckpt = "checkpoint/robomimic/can_mh/iql_expert_seed1_20260613_042513/step_100000.pt"
    payload = _build_old_layout_payload()
    loaded = _make_loaded(payload, ckpt)
    override_mp4 = str(tmp_path / "manual.mp4")
    eval_cfg = {
        "video": {
            "enabled": True,
            "path": override_mp4,
            "skip": 5,
            "fps": 20,
            "frame_height": 512,
            "frame_width": 512,
            "camera_names": ["agentview"],
        },
        "output": {"dir": "outputs/eval", "json_path": None},
        "seed": 0,
    }

    plan = _resolve_auto_eval_paths(
        eval_cfg=eval_cfg,
        loaded=loaded,
        config_dir=tmp_path,
        n_rollouts=20,
        horizon=400,
    )

    assert plan["video"]["path"] == override_mp4
    # JSON should still use the auto path even when video is overridden.
    assert plan["json"]["path"].endswith("/eval_step_000100000_evalseed0_n020_h0400.json")


def test_resolve_auto_paths_json_path_override(tmp_path):
    ckpt = "checkpoint/robomimic/can_mh/iql_expert_seed1_20260613_042513/step_100000.pt"
    payload = _build_old_layout_payload()
    loaded = _make_loaded(payload, ckpt)
    override_json = str(tmp_path / "manual.json")
    eval_cfg = {
        "video": {
            "enabled": True,
            "path": None,
            "skip": 5,
            "fps": 20,
            "frame_height": 512,
            "frame_width": 512,
            "camera_names": ["agentview"],
        },
        "output": {"dir": "outputs/eval", "json_path": override_json},
        "seed": 0,
    }

    plan = _resolve_auto_eval_paths(
        eval_cfg=eval_cfg,
        loaded=loaded,
        config_dir=tmp_path,
        n_rollouts=20,
        horizon=400,
    )

    assert plan["json"]["path"] == override_json
    # Video should still use the auto path even when JSON is overridden.
    assert plan["video"]["path"].endswith("/eval_step_000100000_evalseed0_n020_h0400.mp4")


def test_resolve_auto_paths_no_video_still_writes_json(tmp_path):
    ckpt = "checkpoint/robomimic/can_mh/iql_expert_seed1_20260613_042513/step_100000.pt"
    payload = _build_old_layout_payload()
    loaded = _make_loaded(payload, ckpt)
    eval_cfg = {
        "video": {
            "enabled": False,
            "path": None,
            "skip": 5,
            "fps": 20,
            "frame_height": 512,
            "frame_width": 512,
            "camera_names": ["agentview"],
        },
        "output": {"dir": "outputs/eval", "json_path": None},
        "seed": 0,
    }

    plan = _resolve_auto_eval_paths(
        eval_cfg=eval_cfg,
        loaded=loaded,
        config_dir=tmp_path,
        n_rollouts=20,
        horizon=400,
    )

    assert plan["video"]["enabled"] is False
    assert plan["video"]["path"] is None
    assert plan["json"]["path"].endswith(
        "/eval_step_000100000_evalseed0_n020_h0400.json"
    )


def test_resolve_auto_paths_legacy_video_path_treated_as_enabled(tmp_path):
    """Backwards-compat: legacy configs without ``video.enabled`` still work."""
    ckpt = "checkpoint/robomimic/can_mh/iql_expert_seed1_20260613_042513/step_100000.pt"
    payload = _build_old_layout_payload()
    loaded = _make_loaded(payload, ckpt)
    eval_cfg = {
        "video": {
            "path": str(tmp_path / "legacy.mp4"),
            "skip": 5,
            "fps": 20,
            "frame_height": 512,
            "frame_width": 512,
            "camera_names": ["agentview"],
        },
        "output": {"json_path": str(tmp_path / "legacy.json")},
        "seed": 0,
    }

    plan = _resolve_auto_eval_paths(
        eval_cfg=eval_cfg,
        loaded=loaded,
        config_dir=tmp_path,
        n_rollouts=20,
        horizon=400,
    )

    assert plan["video"]["enabled"] is True
    assert plan["video"]["path"] == str(tmp_path / "legacy.mp4")
    assert plan["json"]["path"] == str(tmp_path / "legacy.json")


def test_is_video_enabled_respects_explicit_flag():
    assert _is_video_enabled({"enabled": True, "path": None}) is True
    assert _is_video_enabled({"enabled": False, "path": "/some/path.mp4"}) is False
    assert _is_video_enabled({"enabled": False, "path": None}) is False


def test_is_video_enabled_legacy_uses_path():
    assert _is_video_enabled({"path": "/some/path.mp4"}) is True
    assert _is_video_enabled({"path": None}) is False
    assert _is_video_enabled(None) is False
    assert _is_video_enabled({}) is False


def test_derive_run_id_picks_first_timestamp():
    ckpt = "checkpoint/robomimic/can/IQL__IQL_expert/20260613_134706/seed3/ckpt/final.pt"
    assert _derive_run_id_from_checkpoint_path(ckpt) == "20260613_134706"


def test_derive_run_id_falls_back_to_parent():
    ckpt = "checkpoint/iql_can_mh_resnet18conv_feats_seed42/step_80000.pt"
    assert _derive_run_id_from_checkpoint_path(ckpt) == "iql_can_mh_resnet18conv_feats_seed42"


def test_checkpoint_tag_from_path():
    assert _checkpoint_tag_from_path("/x/y/step_100000.pt") == "step_100000"
    assert _checkpoint_tag_from_path("/x/y/final.pt") == "final"


def test_apply_cli_overrides_no_video_disables_video():
    """``--no_video`` should force ``video.enabled=False`` even if config sets it True."""
    runtime = SimpleNamespace(values={
        "agent": "ckpt.pt",
        "video": {"enabled": True, "path": None, "skip": 5, "fps": 20, "frame_height": 512, "frame_width": 512, "camera_names": ["agentview"]},
        "rollout": {"n_rollouts": 5, "horizon": 100},
        "env": {"name_override": None},
        "policy": {"stochastic": False},
        "output": {"dir": "outputs/eval", "json_path": None},
        "seed": 0,
    })
    args = SimpleNamespace(
        agent=None, device=None, seed=None, n_rollouts=None, horizon=None,
        env=None, video_path=None, no_video=True, video_skip=None, video_fps=None,
        frame_height=None, frame_width=None, camera_names=None,
        json_path=None, output_dir=None, stochastic=None,
    )

    overrides = _apply_cli_overrides(runtime, args)

    assert overrides["video"]["enabled"] is False


def test_apply_cli_overrides_video_path_implies_enabled():
    runtime = SimpleNamespace(values={
        "agent": "ckpt.pt",
        "video": {"enabled": False, "path": None, "skip": 5, "fps": 20, "frame_height": 512, "frame_width": 512, "camera_names": ["agentview"]},
        "rollout": {"n_rollouts": 5, "horizon": 100},
        "env": {"name_override": None},
        "policy": {"stochastic": False},
        "output": {"dir": "outputs/eval", "json_path": None},
        "seed": 0,
    })
    args = SimpleNamespace(
        agent=None, device=None, seed=None, n_rollouts=None, horizon=None,
        env=None, video_path="/tmp/foo.mp4", no_video=False, video_skip=None, video_fps=None,
        frame_height=None, frame_width=None, camera_names=None,
        json_path=None, output_dir=None, stochastic=None,
    )

    overrides = _apply_cli_overrides(runtime, args)

    assert overrides["video"]["enabled"] is True
    assert overrides["video"]["path"] == "/tmp/foo.mp4"


def test_apply_cli_overrides_output_dir():
    runtime = SimpleNamespace(values={
        "agent": "ckpt.pt",
        "video": {"enabled": False, "path": None, "skip": 5, "fps": 20, "frame_height": 512, "frame_width": 512, "camera_names": ["agentview"]},
        "rollout": {"n_rollouts": 5, "horizon": 100},
        "env": {"name_override": None},
        "policy": {"stochastic": False},
        "output": {"dir": "outputs/eval", "json_path": None},
        "seed": 0,
    })
    args = SimpleNamespace(
        agent=None, device=None, seed=None, n_rollouts=None, horizon=None,
        env=None, video_path=None, no_video=False, video_skip=None, video_fps=None,
        frame_height=None, frame_width=None, camera_names=None,
        json_path=None, output_dir="/tmp/custom_eval", stochastic=None,
    )

    overrides = _apply_cli_overrides(runtime, args)

    assert overrides["output"]["dir"] == "/tmp/custom_eval"


def test_resolve_eval_output_path_none_returns_none(tmp_path):
    assert _resolve_eval_output_path(None, tmp_path) is None


def test_resolve_eval_output_path_relative(tmp_path):
    rel = "subdir/file.json"
    resolved = _resolve_eval_output_path(rel, tmp_path)
    assert Path(resolved) == (tmp_path / "subdir/file.json").resolve()


def test_apply_cli_overrides_seed_list_precedence_over_single_seed():
    runtime = SimpleNamespace(values={
        "agent": "ckpt.pt",
        "video": {"enabled": False, "path": None, "skip": 5, "fps": 20, "frame_height": 512, "frame_width": 512, "camera_names": ["agentview"]},
        "rollout": {"n_rollouts": 5, "horizon": 100, "num_workers": 1, "worker_device": "cpu"},
        "env": {"name_override": None},
        "policy": {"stochastic": False},
        "output": {"dir": "outputs/eval", "json_path": None},
        "seed": 0,
    })
    args = SimpleNamespace(
        agent=None, device=None, seed=999, seeds=[0, 1, 2],
        n_rollouts=None, horizon=None, num_workers=None, worker_device=None,
        env=None, video_path=None, no_video=False, video_skip=None, video_fps=None,
        frame_height=None, frame_width=None, camera_names=None,
        json_path=None, output_dir=None, stochastic=None,
    )

    overrides = _apply_cli_overrides(runtime, args)

    assert overrides["seed"] == [0, 1, 2]


def test_apply_cli_overrides_rollout_workers():
    runtime = SimpleNamespace(values={
        "agent": "ckpt.pt",
        "video": {"enabled": False, "path": None, "skip": 5, "fps": 20, "frame_height": 512, "frame_width": 512, "camera_names": ["agentview"]},
        "rollout": {"n_rollouts": 5, "horizon": 100, "num_workers": 1, "worker_device": "cpu"},
        "env": {"name_override": None},
        "policy": {"stochastic": False},
        "output": {"dir": "outputs/eval", "json_path": None},
        "seed": 0,
    })
    args = SimpleNamespace(
        agent=None, device=None, seed=None, seeds=None,
        n_rollouts=None, horizon=None, num_workers=4, worker_device="cuda",
        env=None, video_path=None, no_video=False, video_skip=None, video_fps=None,
        frame_height=None, frame_width=None, camera_names=None,
        json_path=None, output_dir=None, stochastic=None,
    )

    overrides = _apply_cli_overrides(runtime, args)

    assert overrides["rollout"]["num_workers"] == 4
    assert overrides["rollout"]["worker_device"] == "cuda"
