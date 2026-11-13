#!/usr/bin/env python3
"""Run the paired Pick Mango Gaussian-progress pipelines serially.

Each checkpoint is evaluated only on the matching filtered or unfiltered
dataset family.  For each variant this script:

1. extracts embeddings for train, valid4, calibration18, and fail18;
2. fits a 20-bin PCA Gaussian progress model on train embeddings;
3. calibrates conformal p-values with calibration18 embeddings; and
4. predicts valid4 and fail18 at six OOD p-value thresholds.

The script snapshots ``configs_v2`` into a per-invocation work directory and
changes only those private YAML copies.  Prediction artifacts contain H5 files
and MP4s only; prediction PNG figures are disabled.

The fitting t-SNE visualization chooses ``gaussian_samples_per_bin`` from the
actual number of train embedding rows so that ``num_bins * samples_per_bin``
is as close as possible to the number of real points.

Examples:

    conda run --no-capture-output -n fineprog \
        python scripts/run_pick_mango_gaussian_pipeline.py --gpu 0

    conda run -n fineprog python scripts/run_pick_mango_gaussian_pipeline.py \
        --variant filtered --dry-run \
        --work-dir /tmp/fineprog_pick_mango_filtered_dryrun
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
PROCESSED_ROOT = PROJECT_ROOT / "datasets" / "processed"
EMBEDDING_ROOT = PROJECT_ROOT / "datasets" / "embeddings"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CONFIGS_ENV_VAR = "FINEPROG_CONFIGS_V2_DIR"

EXTRACT_ENTRYPOINT = PROJECT_ROOT / "extract_embeddings.py"
EVALUATE_ENTRYPOINT = PROJECT_ROOT / "evaluate_encoder.py"

INPUT_EMBEDDING_DIM = 128
NUM_PROGRESS_BINS = 20
MODEL_EMBEDDING_DIM = 32

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
    """One processed-H5 split used by a Pick Mango encoder."""

    name: str
    suffix: str
    expected_video_count: int
    is_prediction_query: bool = False


@dataclass(frozen=True)
class MangoVariant:
    """One paired Pick Mango checkpoint and dataset family."""

    slug: str
    dataset_prefix: str
    run_name: str

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINT_ROOT / self.run_name / "encoder_epoch020000.pt"

    @property
    def prediction_output_tag(self) -> str:
        return f"pick-mango-{self.slug}"


SPLITS: tuple[DatasetSplit, ...] = (
    DatasetSplit("train", "36vid_train", 36),
    DatasetSplit("valid4", "4vid_valid", 4, is_prediction_query=True),
    DatasetSplit("calibration18", "18vid_calibration", 18),
    DatasetSplit("fail18", "18vid_fail", 18, is_prediction_query=True),
)

VARIANTS: tuple[MangoVariant, ...] = (
    MangoVariant(
        slug="filtered",
        dataset_prefix="pick_mango_filtered_vid",
        run_name=(
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-pick_mango_filtered_vid-36vid_train-"
            "resnet50_conv4c-only_bn-20260910-022859"
        ),
    ),
    MangoVariant(
        slug="unfiltered",
        dataset_prefix="pick_mango_vid",
        run_name=(
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-pick_mango_vid-36vid_train-"
            "resnet50_conv4c-only_bn-20260910-063346"
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


def _split_stem(variant: MangoVariant, split: DatasetSplit) -> str:
    return f"{variant.dataset_prefix}-{split.suffix}"


def _dataset_ref(variant: MangoVariant, split: DatasetSplit) -> str:
    return f"{variant.dataset_prefix}_{split.suffix}"


def _processed_path(variant: MangoVariant, split: DatasetSplit) -> Path:
    return PROCESSED_ROOT / f"{_split_stem(variant, split)}.h5"


def _embedding_path(variant: MangoVariant, split: DatasetSplit) -> Path:
    return EMBEDDING_ROOT / variant.run_name / f"{_split_stem(variant, split)}-embd.h5"


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
        raise FileNotFoundError(f"Processed Pick Mango H5 not found: {path}")

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
            context = f"{path}:/videos/{video_id}"
            if not isinstance(video, h5py.Group):
                raise RuntimeError(f"{context} must be a group")
            frames = video.get("frames")
            if not isinstance(frames, h5py.Dataset):
                raise RuntimeError(f"{context} must contain a frames dataset")
            num_frames = video.attrs.get("num_frames")
            if (
                isinstance(num_frames, (bool, np.bool_))
                or not isinstance(num_frames, (int, np.integer))
                or int(num_frames) != int(frames.shape[0])
            ):
                raise RuntimeError(
                    f"{context} num_frames must equal frames.shape[0]={frames.shape[0]}"
                )

            source_path = Path(_as_text(video.attrs.get("path", "")))
            action_name = _as_text(video.attrs.get("action_name", ""))
            if not source_path.name or not action_name:
                raise RuntimeError(f"{context} lacks renderable path/action_name metadata")
            if source_path.stem != action_name:
                raise RuntimeError(
                    f"{context} action_name {action_name!r} does not match {source_path}"
                )
            fallback_path = (
                PROJECT_ROOT
                / "datasets"
                / "raw"
                / source_path.parent.name
                / source_path.name
            )
            if not source_path.is_file() and not fallback_path.is_file():
                raise FileNotFoundError(
                    f"No raw MP4 exists for {context}; checked {source_path} "
                    f"and {fallback_path}"
                )


def _validate_embedding_h5(path: Path, expected_video_count: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding H5 not found: {path}")

    with h5py.File(path, "r") as h5_file:
        if int(h5_file.attrs.get("embedding_dim", -1)) != INPUT_EMBEDDING_DIM:
            raise RuntimeError(f"{path} must have root embedding_dim={INPUT_EMBEDDING_DIM}")
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
            if (
                embeddings.ndim != 2
                or embeddings.shape[1] != INPUT_EMBEDDING_DIM
                or embeddings.shape[0] < 1
            ):
                raise RuntimeError(
                    f"{path}:/videos/{video_id}/embeddings must have shape "
                    f"[T, {INPUT_EMBEDDING_DIM}]"
                )
            if target_steps.shape != (embeddings.shape[0],):
                raise RuntimeError(
                    f"{path}:/videos/{video_id}/target_steps does not align with embeddings"
                )
            if not np.isfinite(embeddings).all():
                raise RuntimeError(
                    f"{path}:/videos/{video_id}/embeddings contains NaN or Inf"
                )


def _count_embedding_rows(path: Path) -> int:
    """Return the number of real points used by fitting and t-SNE."""
    with h5py.File(path, "r") as h5_file:
        videos = _require_video_group(h5_file, path)
        total = sum(int(videos[video_id]["embeddings"].shape[0]) for video_id in videos)
    if total < 1:
        raise RuntimeError(f"{path} contains no embedding rows")
    return total


def _gaussian_samples_per_bin(real_point_count: int, num_bins: int) -> int:
    """Choose an integer per-bin count whose total is nearest to real points."""
    if real_point_count < 1:
        raise ValueError("real_point_count must be >= 1")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    samples_per_bin = max(1, (real_point_count + num_bins // 2) // num_bins)
    difference = abs(num_bins * samples_per_bin - real_point_count)
    if difference > num_bins // 2:
        raise RuntimeError("Failed to choose a near-equal Gaussian visualization sample count")
    return samples_per_bin


def _validate_fitted_model(path: Path, expert_h5_path: Path) -> None:
    with h5py.File(path, "r") as h5_file:
        attrs = h5_file.attrs
        if _as_text(attrs.get("task_name", "")) != "gaussian_progress_fitting":
            raise RuntimeError(f"{path} is not a Gaussian progress fitting artifact")
        if not _same_path(_as_text(attrs.get("expert_h5_path", "")), expert_h5_path):
            raise RuntimeError(f"{path} was not fitted from {expert_h5_path}")
        expected_attrs = {
            "num_bins": NUM_PROGRESS_BINS,
            "input_embedding_dim": INPUT_EMBEDDING_DIM,
            "embedding_dim": MODEL_EMBEDDING_DIM,
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
        if model["bin_means"].shape != (NUM_PROGRESS_BINS, MODEL_EMBEDDING_DIM):
            raise RuntimeError(f"{path} has an unexpected /model/bin_means shape")
        if model["pca_components"].shape != (
            MODEL_EMBEDDING_DIM,
            INPUT_EMBEDDING_DIM,
        ):
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
        required = {"posterior", "conformal_p_value", "is_ood", "progress_gated"}
        for video_id in sorted(videos):
            video = videos[video_id]
            if not isinstance(video, h5py.Group) or not required.issubset(video.keys()):
                raise RuntimeError(f"{path}:/nonexperts/{video_id} lacks video-render inputs")

    if (path.parent / "figures").exists():
        raise RuntimeError(f"{path.parent} unexpectedly contains figures")
    videos_dir = path.parent / "videos"
    video_files = sorted(videos_dir.glob("*.mp4")) if videos_dir.is_dir() else []
    if len(video_files) != expected_video_count:
        raise RuntimeError(
            f"{path.parent} has {len(video_files)} MP4s; expected {expected_video_count}"
        )


def _find_fitted_model(output_dir: Path, expert_h5_path: Path) -> Path | None:
    matches: list[Path] = []
    pattern = "*/gaussian_progress_model-covariance_mode-independent-*.h5"
    for candidate in output_dir.glob(pattern):
        try:
            _validate_fitted_model(candidate, expert_h5_path)
        except (KeyError, OSError, RuntimeError, ValueError):
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
            and _same_path(
                _as_text(attrs.get("calibration_h5_path", "")), calibration_path
            )
            and np.isclose(float(attrs.get("ood_p_value_threshold", np.nan)), threshold)
        )


def _find_existing_prediction(
    variant_output_dir: Path,
    *,
    model_path: Path,
    query_path: Path,
    calibration_path: Path,
    threshold: float,
    expected_video_count: int,
) -> Path | None:
    matches: list[Path] = []
    for candidate in variant_output_dir.glob("*/**/gaussian_progress_pred-*.h5"):
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
            f"Multiple completed predictions match {query_path.name} at "
            f"threshold {threshold}: {matches}"
        )
    return matches[0] if matches else None


def _make_worker_env(config_root: Path, gpu: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env[CONFIGS_ENV_VAR] = str(config_root)
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def _run_entrypoint(
    config_root: Path,
    entrypoint: Path,
    args: Iterable[str],
    gpu: str | None,
) -> None:
    command = [sys.executable, str(entrypoint), *args]
    print(f"[pipeline] command: {shlex.join(command)}", flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_make_worker_env(config_root, gpu),
        check=True,
    )


def _prepare_work_directory(
    work_dir: Path,
    variants: Iterable[MangoVariant],
) -> dict[str, Path]:
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
    for variant in variants:
        config_root = work_dir / f"configs_{variant.slug}"
        shutil.copytree(source_snapshot, config_root)
        roots[variant.slug] = config_root
    return roots


def _prepare_extract_config(
    config_root: Path,
    variant: MangoVariant,
    split: DatasetSplit,
    embedding_path: Path,
    num_workers: int | None,
) -> None:
    path = config_root / "extract.yaml"
    config = _read_yaml(path)
    config.update(
        {
            "extract_dataset": _dataset_ref(variant, split),
            "checkpoint_path": str(variant.checkpoint_path),
            "embedding_dim": INPUT_EMBEDDING_DIM,
            "clip_len": 20,
            "context_size": 2,
            "context_stride": 16,
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
) -> tuple[int, int]:
    real_point_count = _count_embedding_rows(expert_h5_path)
    gaussian_samples_per_bin = _gaussian_samples_per_bin(
        real_point_count,
        NUM_PROGRESS_BINS,
    )

    path = config_root / "eval" / "gaussian_progress_fitting.yaml"
    config = _read_yaml(path)
    tsne_viz = config.get("tsne_viz") or {}
    if not isinstance(tsne_viz, dict):
        raise ValueError(f"{path}: tsne_viz must be a mapping")
    tsne_viz = dict(tsne_viz)
    tsne_viz["gaussian_samples_per_bin"] = gaussian_samples_per_bin
    config.update(
        {
            "expert_h5_path": str(expert_h5_path),
            "output_dir": str(output_dir),
            "num_bins": NUM_PROGRESS_BINS,
            "enable_pca": True,
            "pca_dim": MODEL_EMBEDDING_DIM,
            "covariance_mode": "independent",
            "enable_visualization": True,
            "tsne_viz": tsne_viz,
        }
    )
    _write_yaml(path, config)

    synthetic_point_count = NUM_PROGRESS_BINS * gaussian_samples_per_bin
    print(
        "[pipeline] fitting t-SNE samples: "
        f"real={real_point_count}, bins={NUM_PROGRESS_BINS}, "
        f"gaussian_samples_per_bin={gaussian_samples_per_bin}, "
        f"synthetic={synthetic_point_count}",
        flush=True,
    )
    return real_point_count, gaussian_samples_per_bin


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
    variant: MangoVariant,
    split: DatasetSplit,
    *,
    dry_run: bool,
    gpu: str | None,
    num_workers: int | None,
) -> Path:
    output_path = _embedding_path(variant, split)
    if output_path.exists():
        _validate_embedding_h5(output_path, split.expected_video_count)
        print(f"[pipeline] embedding exists and is valid: {output_path}")
        return output_path

    _prepare_extract_config(config_root, variant, split, output_path, num_workers)
    if dry_run:
        print(f"[pipeline] dry run: would extract {split.name} -> {output_path}")
        return output_path

    _run_entrypoint(config_root, EXTRACT_ENTRYPOINT, (), gpu)
    _validate_embedding_h5(output_path, split.expected_video_count)
    return output_path


def _fit_gaussian_model(
    config_root: Path,
    variant: MangoVariant,
    expert_h5_path: Path,
    *,
    gpu: str | None,
) -> Path:
    output_dir = (
        OUTPUT_ROOT
        / "gaussian_progress_fitting"
        / variant.run_name
        / expert_h5_path.stem
    )
    existing = _find_fitted_model(output_dir, expert_h5_path) if output_dir.is_dir() else None
    if existing is not None:
        print(f"[pipeline] fitted model exists and is valid: {existing}")
        return existing

    _prepare_fitting_config(config_root, expert_h5_path, output_dir)
    _run_entrypoint(
        config_root,
        EVALUATE_ENTRYPOINT,
        ("--task", "gaussian_progress_fitting"),
        gpu,
    )
    model_path = _find_fitted_model(output_dir, expert_h5_path)
    if model_path is None:
        raise RuntimeError(f"Gaussian fitting completed without a valid model in {output_dir}")
    return model_path


def _relocate_prediction_output(
    source_h5_path: Path,
    variant_output_dir: Path,
    threshold_tag: str,
) -> Path:
    source_leaf_dir = source_h5_path.parent
    tagged_leaf_dir = source_leaf_dir.with_name(f"{source_leaf_dir.name}-ood{threshold_tag}")
    if tagged_leaf_dir.exists():
        raise FileExistsError(f"Prediction output directory already exists: {tagged_leaf_dir}")
    source_leaf_dir.rename(tagged_leaf_dir)

    target_query_dir = variant_output_dir / tagged_leaf_dir.parent.name
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
    variant: MangoVariant,
    query_split: DatasetSplit,
    *,
    model_path: Path,
    calibration_path: Path,
    threshold: float,
    threshold_tag: str,
    gpu: str | None,
) -> Path:
    query_path = _embedding_path(variant, query_split)
    variant_output_dir = (
        OUTPUT_ROOT
        / "gaussian_progress_pred"
        / f"{model_path.stem}-{variant.prediction_output_tag}"
    )
    existing = (
        _find_existing_prediction(
            variant_output_dir,
            model_path=model_path,
            query_path=query_path,
            calibration_path=calibration_path,
            threshold=threshold,
            expected_video_count=query_split.expected_video_count,
        )
        if variant_output_dir.is_dir()
        else None
    )
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
    untagged_query_dir = (
        OUTPUT_ROOT / "gaussian_progress_pred" / model_path.stem / query_path.stem
    )
    before = {
        path.resolve()
        for path in untagged_query_dir.glob("*/gaussian_progress_pred-*.h5")
    }
    _run_entrypoint(
        config_root,
        EVALUATE_ENTRYPOINT,
        ("--task", "gaussian_progress_pred"),
        gpu,
    )
    after = {
        path.resolve()
        for path in untagged_query_dir.glob("*/gaussian_progress_pred-*.h5")
    }
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
        variant_output_dir,
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


def _run_variant(
    variant: MangoVariant,
    config_root: Path,
    *,
    dry_run: bool,
    gpu: str | None,
    num_workers: int | None,
) -> None:
    print(f"\n[pipeline] ===== Pick Mango {variant.slug} =====", flush=True)
    if not variant.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {variant.checkpoint_path}")

    for split in SPLITS:
        _validate_processed_h5(
            _processed_path(variant, split),
            split.expected_video_count,
            require_renderable_videos=split.is_prediction_query,
        )

    embeddings = {
        split.name: _extract_embedding(
            config_root,
            variant,
            split,
            dry_run=dry_run,
            gpu=gpu,
            num_workers=num_workers,
        )
        for split in SPLITS
    }
    if dry_run:
        print("[pipeline] dry run: fitting and prediction commands were not executed")
        return

    model_path = _fit_gaussian_model(
        config_root,
        variant,
        embeddings["train"],
        gpu=gpu,
    )
    calibration_path = embeddings["calibration18"]
    for split in SPLITS:
        if not split.is_prediction_query:
            continue
        for threshold, threshold_tag in PREDICTION_THRESHOLDS:
            _predict_one(
                config_root,
                variant,
                split,
                model_path=model_path,
                calibration_path=calibration_path,
                threshold=threshold,
                threshold_tag=threshold_tag,
                gpu=gpu,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired filtered/unfiltered Pick Mango Gaussian pipelines."
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=[variant.slug for variant in VARIANTS],
        help="Run only this variant. Repeat to select both (default: both).",
    )
    parser.add_argument(
        "--gpu",
        help="CUDA_VISIBLE_DEVICES value for every worker subprocess (default: inherit).",
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
            "Directory for private configuration snapshots (default: "
            "outputs/pick_mango_gaussian_pipeline/<timestamp>)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selected checkpoints and inputs without extraction/evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    requested = set(args.variant or [variant.slug for variant in VARIANTS])
    selected_variants = tuple(
        variant for variant in VARIANTS if variant.slug in requested
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else OUTPUT_ROOT / "pick_mango_gaussian_pipeline" / timestamp
    )
    config_roots = _prepare_work_directory(work_dir, selected_variants)
    print(f"[pipeline] work directory: {work_dir}", flush=True)
    print(
        f"[pipeline] variants: {', '.join(v.slug for v in selected_variants)}",
        flush=True,
    )

    for variant in selected_variants:
        _run_variant(
            variant,
            config_roots[variant.slug],
            dry_run=args.dry_run,
            gpu=args.gpu,
            num_workers=args.num_workers,
        )

    if args.dry_run:
        print("[pipeline] dry run complete; no embeddings, models, or predictions were written.")
    else:
        print("[pipeline] all selected Pick Mango Gaussian-progress jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
