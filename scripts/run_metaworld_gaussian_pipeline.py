#!/usr/bin/env python3
"""Run the remaining MetaWorld Gaussian-progress pipeline serially.

The script is deliberately self-contained: it snapshots ``configs_v2`` into a
per-invocation work directory and changes only those private YAML copies.  It
therefore does not alter the repository's shared experiment configuration.

For each task it:

1. extracts embeddings for train, valid4, valid20, noise01, and noise02;
2. fits the Gaussian progress model on train embeddings;
3. calibrates with valid20 embeddings; and
4. predicts only valid4 and noise02 for six OOD p-value thresholds.

``noise01`` is exported but intentionally not used for prediction.  Prediction
artifacts contain H5 files and MP4s only; PNG figures are disabled.

Examples:

    conda run --no-capture-output -n fineprog \
        python scripts/run_metaworld_gaussian_pipeline.py --gpu 0

    conda run -n fineprog python scripts/run_metaworld_gaussian_pipeline.py \
        --task hammer --dry-run --work-dir /tmp/fineprog_metaworld_hammer_dryrun
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoint"
PROCESSED_ROOT = PROJECT_ROOT / "datasets" / "processed" / "metaworld"
EMBEDDING_ROOT = PROJECT_ROOT / "datasets" / "embeddings"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CONFIGS_ENV_VAR = "FINEPROG_CONFIGS_V2_DIR"

EXTRACT_ENTRYPOINT = PROJECT_ROOT / "extract_embeddings.py"
EVALUATE_ENTRYPOINT = PROJECT_ROOT / "evaluate_encoder.py"

PREDICTION_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.05, "005"),
    (0.10, "010"),
    (0.20, "020"),
    (0.30, "030"),
    (0.40, "040"),
    (0.50, "050"),
)


@dataclass(frozen=True)
class DatasetSplit:
    """One processed-H5 split used by a MetaWorld encoder."""

    name: str
    suffix: str
    expected_video_count: int
    is_prediction_query: bool = False


@dataclass(frozen=True)
class TaskSpec:
    """Pinned current MetaWorld checkpoint and its registry prefix."""

    slug: str
    registry_prefix: str
    run_name: str

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINT_ROOT / self.run_name / "encoder_epoch020000.pt"


SPLITS: tuple[DatasetSplit, ...] = (
    DatasetSplit("train", "36vid_train", 36),
    DatasetSplit("valid4", "4vid_valid", 4, is_prediction_query=True),
    DatasetSplit("valid20", "20vid_valid", 20),
    DatasetSplit("noise01", "50vid_noise01", 50),
    DatasetSplit("noise02", "50vid_noise02", 50, is_prediction_query=True),
)

# These are the newest complete checkpoints present when this script was added.
TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        slug="drawer-open",
        registry_prefix="metaworld_drawer_open_v2",
        run_name=(
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-drawer-open-v2-36vid_train-"
            "resnet50_conv4c-only_bn-20260903-155520"
        ),
    ),
    TaskSpec(
        slug="sweep-into",
        registry_prefix="metaworld_sweep_into_v2",
        run_name=(
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-sweep-into-v2-36vid_train-"
            "resnet50_conv4c-only_bn-20260905-022546"
        ),
    ),
    TaskSpec(
        slug="button-press-wall",
        registry_prefix="metaworld_button_press_wall_v2",
        run_name=(
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-button-press-wall-v2-36vid_train-"
            "resnet50_conv4c-only_bn-20260905-063427"
        ),
    ),
    TaskSpec(
        slug="door-lock",
        registry_prefix="metaworld_door_lock_v2",
        run_name=(
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-door-lock-v2-36vid_train-"
            "resnet50_conv4c-only_bn-20260905-104101"
        ),
    ),
    TaskSpec(
        slug="window-close",
        registry_prefix="metaworld_window_close_v2",
        run_name=(
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-window-close-v2-36vid_train-"
            "resnet50_conv4c-only_bn-20260905-144715"
        ),
    ),
    TaskSpec(
        slug="hammer",
        registry_prefix="metaworld_hammer_v2",
        run_name=(
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-hammer-v2-36vid_train-"
            "resnet50_conv4c-only_bn-20260905-185347"
        ),
    ),
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(loaded).__name__}")
    return loaded


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(value, file, sort_keys=False)


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _split_stem(task: TaskSpec, split: DatasetSplit) -> str:
    return f"{task.slug}-v2-{split.suffix}"


def _dataset_ref(task: TaskSpec, split: DatasetSplit) -> str:
    return f"{task.registry_prefix}_{split.suffix}"


def _processed_path(task: TaskSpec, split: DatasetSplit) -> Path:
    return PROCESSED_ROOT / f"{_split_stem(task, split)}.h5"


def _embedding_path(task: TaskSpec, split: DatasetSplit) -> Path:
    return EMBEDDING_ROOT / task.run_name / f"{_split_stem(task, split)}-embd.h5"


def _find_split(name: str) -> DatasetSplit:
    for split in SPLITS:
        if split.name == name:
            return split
    raise KeyError(f"Unknown split: {name}")


def _require_video_group(h5_file: h5py.File, path: Path) -> h5py.Group:
    videos = h5_file.get("videos")
    if not isinstance(videos, h5py.Group):
        raise RuntimeError(f"{path} must contain a /videos HDF5 group")
    return videos


def _validate_processed_h5(
    path: Path,
    expected_video_count: int,
    *,
    require_renderable_videos: bool,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Processed MetaWorld H5 not found: {path}")

    with h5py.File(path, "r") as h5_file:
        videos = _require_video_group(h5_file, path)
        if len(videos) != expected_video_count:
            raise RuntimeError(
                f"{path} has {len(videos)} videos; expected {expected_video_count}"
            )

        if not require_renderable_videos:
            return

        for video_id in sorted(videos):
            video = videos[video_id]
            if not isinstance(video, h5py.Group):
                raise RuntimeError(f"{path}:/videos/{video_id} must be a group")
            if "frames" not in video:
                raise RuntimeError(f"{path}:/videos/{video_id} is missing frames")
            source_path = Path(_as_text(video.attrs.get("path", "")))
            action_name = _as_text(video.attrs.get("action_name", ""))
            if not source_path.name or not action_name:
                raise RuntimeError(
                    f"{path}:/videos/{video_id} lacks renderable path/action_name metadata"
                )
            if source_path.stem != action_name:
                raise RuntimeError(
                    f"{path}:/videos/{video_id} action_name {action_name!r} does not "
                    f"match source path {source_path}"
                )
            fallback_path = PROJECT_ROOT / "datasets" / "raw" / source_path.parent.name / source_path.name
            if not source_path.is_file() and not fallback_path.is_file():
                raise FileNotFoundError(
                    f"No raw MP4 exists for {path}:/videos/{video_id}; checked "
                    f"{source_path} and {fallback_path}"
                )


def _validate_embedding_h5(path: Path, expected_video_count: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding H5 not found: {path}")

    with h5py.File(path, "r") as h5_file:
        if int(h5_file.attrs.get("embedding_dim", -1)) != 128:
            raise RuntimeError(f"{path} must have root embedding_dim=128")
        if _as_text(h5_file.attrs.get("embedding_normalization", "")) != "l2":
            raise RuntimeError(f"{path} must have embedding_normalization='l2'")
        videos = _require_video_group(h5_file, path)
        if len(videos) != expected_video_count:
            raise RuntimeError(
                f"{path} has {len(videos)} videos; expected {expected_video_count}"
            )
        for video_id in sorted(videos):
            video = videos[video_id]
            if not isinstance(video, h5py.Group):
                raise RuntimeError(f"{path}:/videos/{video_id} must be a group")
            if "embeddings" not in video or "target_steps" not in video:
                raise RuntimeError(
                    f"{path}:/videos/{video_id} must contain embeddings and target_steps"
                )
            embeddings = np.asarray(video["embeddings"][:], dtype=np.float32)
            target_steps = np.asarray(video["target_steps"][:])
            if embeddings.ndim != 2 or embeddings.shape[1] != 128 or embeddings.shape[0] < 1:
                raise RuntimeError(
                    f"{path}:/videos/{video_id}/embeddings must have shape [T, 128]"
                )
            if target_steps.shape != (embeddings.shape[0],):
                raise RuntimeError(
                    f"{path}:/videos/{video_id}/target_steps does not align with embeddings"
                )
            if not np.isfinite(embeddings).all():
                raise RuntimeError(f"{path}:/videos/{video_id}/embeddings contains NaN or Inf")


def _validate_fitted_model(path: Path, expert_h5_path: Path) -> None:
    with h5py.File(path, "r") as h5_file:
        attrs = h5_file.attrs
        if _as_text(attrs.get("task_name", "")) != "gaussian_progress_fitting":
            raise RuntimeError(f"{path} is not a Gaussian progress fitting artifact")
        if not _same_path(_as_text(attrs.get("expert_h5_path", "")), expert_h5_path):
            raise RuntimeError(f"{path} was not fitted from {expert_h5_path}")
        expected_attrs = {
            "num_bins": 20,
            "input_embedding_dim": 128,
            "embedding_dim": 32,
            "enable_pca": True,
            "covariance_mode": "independent",
            "embedding_normalization": "l2",
        }
        for key, expected in expected_attrs.items():
            value = attrs.get(key)
            if isinstance(expected, bool):
                matches = bool(value) is expected
            elif isinstance(expected, str):
                matches = _as_text(value) == expected
            else:
                matches = int(value) == expected
            if not matches:
                raise RuntimeError(f"{path} has unexpected {key}={value!r}")

        model = h5_file.get("model")
        if not isinstance(model, h5py.Group):
            raise RuntimeError(f"{path} is missing /model")
        if model["bin_means"].shape != (20, 32):
            raise RuntimeError(f"{path} has an unexpected /model/bin_means shape")
        if model["pca_components"].shape != (32, 128):
            raise RuntimeError(f"{path} has an unexpected /model/pca_components shape")


def _validate_prediction_h5(
    path: Path,
    *,
    model_path: Path,
    query_path: Path,
    calibration_path: Path,
    threshold: float,
    expected_video_count: int,
) -> None:
    with h5py.File(path, "r") as h5_file:
        attrs = h5_file.attrs
        if _as_text(attrs.get("task_name", "")) != "gaussian_progress_pred":
            raise RuntimeError(f"{path} is not a Gaussian progress prediction artifact")
        for key, expected in (
            ("gaussian_model_h5_path", model_path),
            ("nonexpert_h5_path", query_path),
            ("calibration_h5_path", calibration_path),
        ):
            if not _same_path(_as_text(attrs.get(key, "")), expected):
                raise RuntimeError(f"{path} has an unexpected {key}")
        if not np.isclose(float(attrs.get("ood_p_value_threshold", np.nan)), threshold):
            raise RuntimeError(f"{path} has an unexpected ood_p_value_threshold")
        if bool(attrs.get("enable_ood_filter", True)):
            raise RuntimeError(f"{path} unexpectedly enables the OOD filter")
        if not bool(attrs.get("save_posterior", False)):
            raise RuntimeError(f"{path} must save posterior values for video rendering")
        if _as_text(attrs.get("embedding_normalization", "")) != "l2":
            raise RuntimeError(f"{path} must preserve l2 embedding metadata")

        videos = h5_file.get("nonexperts")
        if not isinstance(videos, h5py.Group) or len(videos) != expected_video_count:
            raise RuntimeError(
                f"{path} must contain {expected_video_count} /nonexperts videos"
            )
        for video_id in sorted(videos):
            video = videos[video_id]
            required = {"posterior", "conformal_p_value", "is_ood", "progress_gated"}
            if not isinstance(video, h5py.Group) or not required.issubset(video.keys()):
                raise RuntimeError(f"{path}:/nonexperts/{video_id} lacks video-render inputs")

    figures_dir = path.parent / "figures"
    if figures_dir.exists():
        raise RuntimeError(f"{path.parent} unexpectedly contains figures")
    videos_dir = path.parent / "videos"
    video_files = sorted(videos_dir.glob("*.mp4")) if videos_dir.is_dir() else []
    if len(video_files) != expected_video_count:
        raise RuntimeError(
            f"{path.parent} has {len(video_files)} MP4s; expected {expected_video_count}"
        )


def _find_fitted_model(output_dir: Path, expert_h5_path: Path) -> Path | None:
    matches: list[Path] = []
    for candidate in output_dir.glob("*/gaussian_progress_model-covariance_mode-independent-*.h5"):
        try:
            _validate_fitted_model(candidate, expert_h5_path)
        except RuntimeError:
            continue
        matches.append(candidate)
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _prediction_identity_matches(
    path: Path,
    *,
    model_path: Path,
    query_path: Path,
    calibration_path: Path,
    threshold: float,
) -> bool:
    with h5py.File(path, "r") as h5_file:
        attrs = h5_file.attrs
        return (
            _as_text(attrs.get("task_name", "")) == "gaussian_progress_pred"
            and _same_path(_as_text(attrs.get("gaussian_model_h5_path", "")), model_path)
            and _same_path(_as_text(attrs.get("nonexpert_h5_path", "")), query_path)
            and _same_path(_as_text(attrs.get("calibration_h5_path", "")), calibration_path)
            and np.isclose(float(attrs.get("ood_p_value_threshold", np.nan)), threshold)
        )


def _find_existing_prediction(
    task_output_dir: Path,
    *,
    model_path: Path,
    query_path: Path,
    calibration_path: Path,
    threshold: float,
    expected_video_count: int,
) -> Path | None:
    matches: list[Path] = []
    for candidate in task_output_dir.glob("*/**/gaussian_progress_pred-*.h5"):
        if not _prediction_identity_matches(
            candidate,
            model_path=model_path,
            query_path=query_path,
            calibration_path=calibration_path,
            threshold=threshold,
        ):
            continue
        _validate_prediction_h5(
            candidate,
            model_path=model_path,
            query_path=query_path,
            calibration_path=calibration_path,
            threshold=threshold,
            expected_video_count=expected_video_count,
        )
        matches.append(candidate)
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple completed predictions match {query_path.name} at threshold {threshold}: "
            f"{matches}"
        )
    return matches[0] if matches else None


def _make_worker_env(config_root: Path, gpu: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env[CONFIGS_ENV_VAR] = str(config_root)
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def _run_entrypoint(config_root: Path, entrypoint: Path, args: Iterable[str], gpu: str | None) -> None:
    command = [sys.executable, str(entrypoint), *args]
    print(f"[pipeline] command: {shlex.join(command)}", flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_make_worker_env(config_root, gpu),
        check=True,
    )


def _prepare_work_directory(work_dir: Path, tasks: Iterable[TaskSpec]) -> dict[str, Path]:
    if work_dir.exists():
        raise FileExistsError(
            f"Work directory already exists: {work_dir}. Choose a new --work-dir."
        )
    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing configuration directory: {SOURCE_CONFIG_ROOT}")
    if not EXTRACT_ENTRYPOINT.is_file() or not EVALUATE_ENTRYPOINT.is_file():
        raise FileNotFoundError("Missing FineProg extraction or evaluation entrypoint")

    work_dir.mkdir(parents=True)
    source_snapshot = work_dir / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    roots: dict[str, Path] = {}
    for task in tasks:
        task_config_root = work_dir / f"configs_{task.slug}"
        shutil.copytree(source_snapshot, task_config_root)
        roots[task.slug] = task_config_root
    return roots


def _prepare_extract_config(
    config_root: Path,
    task: TaskSpec,
    split: DatasetSplit,
    embedding_path: Path,
    num_workers: int | None,
) -> None:
    path = config_root / "extract.yaml"
    config = _read_yaml(path)
    config.update(
        {
            "extract_dataset": _dataset_ref(task, split),
            "checkpoint_path": str(task.checkpoint_path),
            "embedding_dim": 128,
            "clip_len": 20,
            "context_size": 2,
            "context_stride": 8,
            "sample_all_stride": 1,
            "embedding_save_path": str(embedding_path),
        }
    )
    config.pop("checkpoint_ref", None)
    if num_workers is not None:
        config["num_workers"] = num_workers
    _write_yaml(path, config)


def _prepare_fitting_config(
    config_root: Path,
    expert_h5_path: Path,
    output_dir: Path,
) -> None:
    path = config_root / "eval" / "gaussian_progress_fitting.yaml"
    config = _read_yaml(path)
    config.update(
        {
            "expert_h5_path": str(expert_h5_path),
            "output_dir": str(output_dir),
            "num_bins": 20,
            "enable_pca": True,
            "pca_dim": 32,
            "covariance_mode": "independent",
            "enable_visualization": True,
        }
    )
    _write_yaml(path, config)


def _prepare_prediction_config(
    config_root: Path,
    *,
    model_path: Path,
    query_path: Path,
    calibration_path: Path,
    threshold: float,
) -> None:
    path = config_root / "eval" / "gaussian_progress_pred.yaml"
    config = _read_yaml(path)
    config.update(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(query_path),
            "enable_calibration": True,
            "calibration_h5_path": str(calibration_path),
            "ood_p_value_threshold": threshold,
            "enable_ood_filter": False,
            "save_posterior": True,
            "enable_visualization": False,
            "enable_video": True,
        }
    )
    _write_yaml(path, config)


def _extract_embedding(
    config_root: Path,
    task: TaskSpec,
    split: DatasetSplit,
    *,
    dry_run: bool,
    gpu: str | None,
    num_workers: int | None,
) -> Path:
    output_path = _embedding_path(task, split)
    if output_path.exists():
        _validate_embedding_h5(output_path, split.expected_video_count)
        print(f"[pipeline] embedding exists and is valid: {output_path}")
        return output_path

    _prepare_extract_config(config_root, task, split, output_path, num_workers)
    if dry_run:
        print(f"[pipeline] dry run: would extract {split.name} -> {output_path}")
        return output_path

    _run_entrypoint(config_root, EXTRACT_ENTRYPOINT, (), gpu)
    _validate_embedding_h5(output_path, split.expected_video_count)
    return output_path


def _fit_gaussian_model(
    config_root: Path,
    task: TaskSpec,
    expert_h5_path: Path,
    *,
    dry_run: bool,
    gpu: str | None,
) -> Path:
    output_dir = OUTPUT_ROOT / "gaussian_progress_fitting" / task.run_name / expert_h5_path.stem
    existing = _find_fitted_model(output_dir, expert_h5_path) if output_dir.is_dir() else None
    if existing is not None:
        print(f"[pipeline] fitted model exists and is valid: {existing}")
        return existing

    _prepare_fitting_config(config_root, expert_h5_path, output_dir)
    if dry_run:
        print(f"[pipeline] dry run: would fit Gaussian model in {output_dir}")
        return output_dir / "<timestamp>" / "gaussian_progress_model-<timestamp>.h5"

    _run_entrypoint(config_root, EVALUATE_ENTRYPOINT, ("--task", "gaussian_progress_fitting"), gpu)
    model_path = _find_fitted_model(output_dir, expert_h5_path)
    if model_path is None:
        raise RuntimeError(f"Gaussian fitting completed without a valid model in {output_dir}")
    return model_path


def _relocate_prediction_output(
    source_h5_path: Path,
    task_output_dir: Path,
    threshold_tag: str,
) -> Path:
    source_leaf_dir = source_h5_path.parent
    tagged_leaf_dir = source_leaf_dir.with_name(f"{source_leaf_dir.name}-ood{threshold_tag}")
    if tagged_leaf_dir.exists():
        raise FileExistsError(f"Prediction output directory already exists: {tagged_leaf_dir}")
    source_leaf_dir.rename(tagged_leaf_dir)

    target_query_dir = task_output_dir / tagged_leaf_dir.parent.name
    target_query_dir.mkdir(parents=True, exist_ok=True)
    target_leaf_dir = target_query_dir / tagged_leaf_dir.name
    if target_leaf_dir.exists():
        raise FileExistsError(f"Prediction destination already exists: {target_leaf_dir}")
    tagged_leaf_dir.rename(target_leaf_dir)

    for empty_dir in (source_leaf_dir.parent, source_leaf_dir.parent.parent):
        try:
            empty_dir.rmdir()
        except OSError:
            pass
    return target_leaf_dir / source_h5_path.name


def _predict_one(
    config_root: Path,
    task: TaskSpec,
    query_split: DatasetSplit,
    *,
    model_path: Path,
    calibration_path: Path,
    threshold: float,
    threshold_tag: str,
    dry_run: bool,
    gpu: str | None,
) -> Path | None:
    query_path = _embedding_path(task, query_split)
    task_output_dir = (
        OUTPUT_ROOT
        / "gaussian_progress_pred"
        / f"{model_path.stem}-{task.slug}"
    )
    existing = _find_existing_prediction(
        task_output_dir,
        model_path=model_path,
        query_path=query_path,
        calibration_path=calibration_path,
        threshold=threshold,
        expected_video_count=query_split.expected_video_count,
    ) if task_output_dir.is_dir() else None
    if existing is not None:
        print(
            f"[pipeline] prediction exists and is valid: {query_split.name} "
            f"ood{threshold_tag} -> {existing}"
        )
        return existing

    _prepare_prediction_config(
        config_root,
        model_path=model_path,
        query_path=query_path,
        calibration_path=calibration_path,
        threshold=threshold,
    )
    if dry_run:
        print(
            f"[pipeline] dry run: would predict {query_split.name} "
            f"at ood{threshold_tag}"
        )
        return None

    untagged_query_dir = (
        OUTPUT_ROOT / "gaussian_progress_pred" / model_path.stem / query_path.stem
    )
    before = {path.resolve() for path in untagged_query_dir.glob("*/gaussian_progress_pred-*.h5")}
    _run_entrypoint(config_root, EVALUATE_ENTRYPOINT, ("--task", "gaussian_progress_pred"), gpu)
    after = {path.resolve() for path in untagged_query_dir.glob("*/gaussian_progress_pred-*.h5")}
    created = sorted(after - before)
    if len(created) != 1:
        raise RuntimeError(
            f"Expected one new prediction under {untagged_query_dir}, found {created}"
        )
    source_h5_path = created[0]
    _validate_prediction_h5(
        source_h5_path,
        model_path=model_path,
        query_path=query_path,
        calibration_path=calibration_path,
        threshold=threshold,
        expected_video_count=query_split.expected_video_count,
    )
    final_h5_path = _relocate_prediction_output(
        source_h5_path,
        task_output_dir,
        threshold_tag,
    )
    _validate_prediction_h5(
        final_h5_path,
        model_path=model_path,
        query_path=query_path,
        calibration_path=calibration_path,
        threshold=threshold,
        expected_video_count=query_split.expected_video_count,
    )
    return final_h5_path


def _run_task(
    task: TaskSpec,
    config_root: Path,
    *,
    dry_run: bool,
    gpu: str | None,
    num_workers: int | None,
) -> None:
    print(f"\n[pipeline] ===== {task.slug} =====", flush=True)
    if not task.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {task.checkpoint_path}")

    for split in SPLITS:
        _validate_processed_h5(
            _processed_path(task, split),
            split.expected_video_count,
            require_renderable_videos=split.is_prediction_query,
        )

    embeddings = {
        split.name: _extract_embedding(
            config_root,
            task,
            split,
            dry_run=dry_run,
            gpu=gpu,
            num_workers=num_workers,
        )
        for split in SPLITS
    }
    if dry_run:
        print("[pipeline] dry run: fitting and prediction commands validated only")
        return

    model_path = _fit_gaussian_model(
        config_root,
        task,
        embeddings["train"],
        dry_run=False,
        gpu=gpu,
    )
    calibration_path = embeddings["valid20"]
    for split in SPLITS:
        if not split.is_prediction_query:
            continue
        for threshold, threshold_tag in PREDICTION_THRESHOLDS:
            _predict_one(
                config_root,
                task,
                split,
                model_path=model_path,
                calibration_path=calibration_path,
                threshold=threshold,
                threshold_tag=threshold_tag,
                dry_run=False,
                gpu=gpu,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the remaining six MetaWorld Gaussian-progress pipelines serially."
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=[task.slug for task in TASKS],
        help="Run only this task. Repeat to select multiple tasks (default: all).",
    )
    parser.add_argument(
        "--gpu",
        help="CUDA_VISIBLE_DEVICES value for every extraction subprocess (default: inherit).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override extract.yaml num_workers in the private config snapshots.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help=(
            "Directory for immutable private configuration snapshots (default: "
            "outputs/metaworld_gaussian_pipeline/<timestamp>)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all selected checkpoints and inputs without running extraction/evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    requested = set(args.task or [task.slug for task in TASKS])
    selected_tasks = tuple(task for task in TASKS if task.slug in requested)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else OUTPUT_ROOT / "metaworld_gaussian_pipeline" / timestamp
    )
    config_roots = _prepare_work_directory(work_dir, selected_tasks)
    print(f"[pipeline] work directory: {work_dir}", flush=True)
    print(f"[pipeline] tasks: {', '.join(task.slug for task in selected_tasks)}", flush=True)

    for task in selected_tasks:
        _run_task(
            task,
            config_roots[task.slug],
            dry_run=args.dry_run,
            gpu=args.gpu,
            num_workers=args.num_workers,
        )

    if args.dry_run:
        print("[pipeline] dry run complete; no embeddings, models, or predictions were written.")
    else:
        print("[pipeline] all selected MetaWorld Gaussian-progress jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
