#!/usr/bin/env python3
"""Fill missing Mango-checkpoint Gaussian progress predictions.

Only the missing Mango embedding for 260917 valid4 is extracted. Existing
valid predictions are reused when their H5 provenance and MP4 set validate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs_v2"
EMBED_ROOT = ROOT / "datasets" / "embeddings"
OUTPUT_ROOT = ROOT / "outputs" / "gaussian_progress_pred"
RUN_NAME = (
    "COMPOSITE_TCC_TEMPORAL_TRIPLET-pick_mango_vid-36vid_train-"
    "resnet50_conv4c-only_bn-20260910-063346"
)
CHECKPOINT = ROOT / "checkpoint" / RUN_NAME / "encoder_epoch020000.pt"
TRAIN_EMBEDDING = EMBED_ROOT / RUN_NAME / "pick_mango_vid-36vid_train-embd.h5"
CALIBRATION = EMBED_ROOT / RUN_NAME / "pick_mango_vid-18vid_calibration-embd.h5"
MODEL = (
    ROOT / "outputs" / "gaussian_progress_fitting" / RUN_NAME /
    "pick_mango_vid-36vid_train-embd" / "20260910-152003" /
    "gaussian_progress_model-covariance_mode-independent-20260910-152003.h5"
)
PREDICTION_ROOT = OUTPUT_ROOT / (
    "gaussian_progress_model-covariance_mode-independent-20260910-152003-"
    "pick-mango-unfiltered"
)

THRESHOLDS = ((0.05, "005"), (0.10, "010"), (0.20, "020"),
              (0.30, "030"), (0.40, "040"), (0.50, "050"))


@dataclass(frozen=True)
class Query:
    name: str
    h5_name: str
    expected_videos: int


QUERIES = (
    Query("fruit_valid4", "260917_fruit_expert_videos-4vid_valid.h5", 4),
    Query("mango_fail18", "pick_mango_vid-18vid_fail.h5", 18),
    Query("rollout48", "fruits_real_rollout_vid-48vid_all.h5", 48),
)


def _processed_path(query: Query) -> Path:
    return ROOT / "datasets" / "processed" / query.h5_name


def _embedding_path(query: Query) -> Path:
    return EMBED_ROOT / RUN_NAME / f"{Path(query.h5_name).stem}-embd.h5"


def _same_path(value: Any, expected: Path) -> bool:
    return Path(str(value)).expanduser().resolve() == expected.resolve()


def _validate_checkpoint_and_model() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("embedding_normalization") != "l2":
        raise ValueError(f"Unexpected checkpoint normalization: {CHECKPOINT}")
    projection = checkpoint["model_state_dict"]["temporal_embedder.proj.weight"]
    if tuple(projection.shape) != (128, 512):
        raise ValueError(f"Unexpected checkpoint embedding dimension: {tuple(projection.shape)}")

    for path in (TRAIN_EMBEDDING, CALIBRATION, MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)
    with h5py.File(TRAIN_EMBEDDING, "r") as train_file:
        if (int(train_file.attrs.get("embedding_dim", -1)) != 128
                or train_file.attrs.get("embedding_normalization") != "l2"
                or len(train_file["videos"]) != 36):
            raise ValueError(f"Invalid Mango train embedding: {TRAIN_EMBEDDING}")
    with h5py.File(CALIBRATION, "r") as cal_file:
        if (int(cal_file.attrs.get("embedding_dim", -1)) != 128
                or cal_file.attrs.get("embedding_normalization") != "l2"
                or len(cal_file["videos"]) != 18):
            raise ValueError(f"Invalid Mango calibration embedding: {CALIBRATION}")
    with h5py.File(MODEL, "r") as model_file:
        attrs = model_file.attrs
        if (attrs.get("task_name") != "gaussian_progress_fitting"
                or not _same_path(attrs.get("expert_h5_path", ""), TRAIN_EMBEDDING)
                or int(attrs.get("num_bins", -1)) != 20
                or int(attrs.get("input_embedding_dim", -1)) != 128
                or int(attrs.get("embedding_dim", -1)) != 32
                or attrs.get("covariance_mode") != "independent"
                or attrs.get("embedding_normalization") != "l2"):
            raise ValueError(f"Mango Gaussian model provenance/configuration mismatch: {MODEL}")


def _validate_processed(query: Query) -> None:
    path = _processed_path(query)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as file:
        videos = file.get("videos")
        if not isinstance(videos, h5py.Group) or len(videos) != query.expected_videos:
            raise ValueError(f"{path}: expected {query.expected_videos} videos")
        for video_id, video in videos.items():
            frames = video.get("frames")
            source = Path(str(video.attrs.get("path", "")))
            fallback = ROOT / "datasets" / "raw" / source.parent.name / source.name
            if (not isinstance(frames, h5py.Dataset)
                    or int(video.attrs.get("num_frames", -1)) != int(frames.shape[0])
                    or source.stem != str(video.attrs.get("action_name", ""))
                    or (not source.is_file() and not fallback.is_file())):
                raise ValueError(f"Invalid/renderless source video: {path}:/videos/{video_id}")


def _validate_embedding(path: Path, expected_count: int) -> bool:
    if not path.is_file():
        return False
    with h5py.File(path, "r") as file:
        videos = file.get("videos")
        if (int(file.attrs.get("embedding_dim", -1)) != 128
                or file.attrs.get("embedding_normalization") != "l2"
                or not isinstance(videos, h5py.Group)
                or len(videos) != expected_count):
            raise ValueError(f"Invalid embedding artifact: {path}")
        for video_id, video in videos.items():
            emb, steps = video.get("embeddings"), video.get("target_steps")
            if (not isinstance(emb, h5py.Dataset) or not isinstance(steps, h5py.Dataset)
                    or emb.ndim != 2 or emb.shape[1] != 128
                    or emb.shape[0] != steps.shape[0]):
                raise ValueError(f"Invalid embedding sequence: {path}:/videos/{video_id}")
    return True


def _check_prediction(path: Path, query_embedding: Path, threshold: float,
                      expected_count: int) -> bool:
    with h5py.File(path, "r") as file:
        attrs = file.attrs
        if (attrs.get("task_name") != "gaussian_progress_pred"
                or not _same_path(attrs.get("gaussian_model_h5_path", ""), MODEL)
                or not _same_path(attrs.get("calibration_h5_path", ""), CALIBRATION)
                or not _same_path(attrs.get("nonexpert_h5_path", ""), query_embedding)
                or not np.isclose(float(attrs.get("ood_p_value_threshold", -1)), threshold)
                or bool(attrs.get("enable_ood_filter", True))
                or not bool(attrs.get("enable_calibration", False))
                or not bool(attrs.get("save_posterior", False))):
            return False
        videos = file.get("nonexperts")
        if not isinstance(videos, h5py.Group) or len(videos) != expected_count:
            return False
        required = {"target_steps", "posterior", "conformal_p_value", "is_ood_raw", "is_ood", "progress_gated"}
        if any(not required.issubset(video.keys()) for video in videos.values()):
            return False
    mp4s = list((path.parent / "videos").glob("*.mp4"))
    return len(mp4s) == expected_count and all(video.stat().st_size > 0 for video in mp4s)


def _find_prediction(query_embedding: Path, threshold: float,
                     expected_count: int) -> Path | None:
    matches = []
    for path in OUTPUT_ROOT.rglob("gaussian_progress_pred-*.h5"):
        try:
            with h5py.File(path, "r") as file:
                attrs = file.attrs
                identity_match = (
                    attrs.get("task_name") == "gaussian_progress_pred"
                    and _same_path(attrs.get("gaussian_model_h5_path", ""), MODEL)
                    and _same_path(attrs.get("calibration_h5_path", ""), CALIBRATION)
                    and _same_path(attrs.get("nonexpert_h5_path", ""), query_embedding)
                    and np.isclose(float(attrs.get("ood_p_value_threshold", -1)), threshold)
                )
            if identity_match and _check_prediction(path, query_embedding, threshold, expected_count):
                matches.append(path)
        except (OSError, ValueError, KeyError):
            continue
    if len(matches) > 1:
        raise RuntimeError(f"Multiple complete predictions match {query_embedding} at {threshold}: {matches}")
    return matches[0] if matches else None


def _run_entrypoint(config_root: Path, script: str, task: str | None,
                    gpu: str | None) -> None:
    env = os.environ.copy()
    env["FINEPROG_CONFIGS_V2_DIR"] = str(config_root)
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [sys.executable, str(ROOT / script)]
    if task:
        command.extend(("--task", task))
    print("[mango-pred-fill] running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _write_config(path: Path, updates: dict) -> None:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config.update(updates)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def _extract_missing(config_root: Path, query: Query, output: Path,
                     gpu: str | None) -> None:
    _write_config(config_root / "extract.yaml", {
        "extract_dataset": "fruit_expert_videos_260917_4vid_valid",
        "checkpoint_path": str(CHECKPOINT),
        "embedding_save_path": str(output),
        "embedding_dim": 128, "clip_len": 20, "context_size": 2,
        "context_stride": 16, "sample_all_stride": 1, "num_workers": 0,
    })
    _run_entrypoint(config_root, "extract_embeddings.py", None, gpu)
    if not _validate_embedding(output, query.expected_videos):
        raise RuntimeError(f"Missing Mango embedding after extraction: {output}")


def _predict_missing(config_root: Path, query: Query, embedding: Path,
                     threshold: float, tag: str, gpu: str | None) -> Path:
    _write_config(config_root / "eval" / "gaussian_progress_pred.yaml", {
        "gaussian_model_h5_path": str(MODEL),
        "nonexpert_h5_path": str(embedding),
        "enable_calibration": True,
        "calibration_h5_path": str(CALIBRATION),
        "ood_p_value_threshold": threshold,
        "enable_ood_filter": False,
        "save_posterior": True,
        "enable_visualization": False,
        "enable_video": True,
    })
    source_parent = OUTPUT_ROOT / MODEL.stem / embedding.stem
    before = set(source_parent.glob("*/gaussian_progress_pred-*.h5"))
    _run_entrypoint(config_root, "evaluate_encoder.py", "gaussian_progress_pred", gpu)
    created = list(set(source_parent.glob("*/gaussian_progress_pred-*.h5")) - before)
    if len(created) != 1 or not _check_prediction(created[0], embedding, threshold, query.expected_videos):
        raise RuntimeError(f"Expected one valid new prediction for {query.name} at {threshold}")
    source_leaf = created[0].parent
    destination = PREDICTION_ROOT / f"{embedding.stem}" / f"{source_leaf.name}-ood{tag}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    source_leaf.rename(destination)
    final_h5 = destination / created[0].name
    if not _check_prediction(final_h5, embedding, threshold, query.expected_videos):
        raise RuntimeError(f"Prediction failed post-move validation: {final_h5}")
    print(f"[mango-pred-fill] saved {query.name} ood{tag}: {final_h5}", flush=True)
    return final_h5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES for FineProg subprocesses")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _validate_checkpoint_and_model()
    for query in QUERIES:
        _validate_processed(query)

    fruit_query = QUERIES[0]
    fruit_embedding = _embedding_path(fruit_query)
    has_fruit_embedding = _validate_embedding(fruit_embedding, fruit_query.expected_videos)
    embeddings = {
        query.name: fruit_embedding if query.name == fruit_query.name
        else _embedding_path(query)
        for query in QUERIES
    }
    for query in QUERIES[1:]:
        if not _validate_embedding(embeddings[query.name], query.expected_videos):
            raise FileNotFoundError(f"Missing reusable Mango embedding: {embeddings[query.name]}")

    existing: dict[tuple[str, str], Path] = {}
    missing = []
    for query in QUERIES:
        if query.name == fruit_query.name and not has_fruit_embedding:
            missing.append((query.name, "embedding"))
            continue
        for threshold, tag in THRESHOLDS:
            path = _find_prediction(embeddings[query.name], threshold, query.expected_videos)
            if path is None:
                missing.append((query.name, tag))
            else:
                existing[(query.name, tag)] = path

    print(f"[mango-pred-fill] checkpoint: {CHECKPOINT}", flush=True)
    print(f"[mango-pred-fill] context_stride: 16 (matched to the existing Mango extraction config)", flush=True)
    for query in QUERIES:
        print(f"[mango-pred-fill] query: {query.name} { _processed_path(query) } ({query.expected_videos})", flush=True)
    print(f"[mango-pred-fill] reusable predictions: {len(existing)}; missing: {missing}", flush=True)
    if args.dry_run:
        return

    work_dir = ROOT / "outputs" / "pick_mango_progress_pred_fill" / datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    if work_dir.exists():
        raise FileExistsError(work_dir)
    shutil.copytree(CONFIGS, work_dir)
    if not has_fruit_embedding:
        _extract_missing(work_dir, fruit_query, fruit_embedding, args.gpu)

    total = len(existing)
    for query in QUERIES:
        embedding = embeddings[query.name]
        if query.name == fruit_query.name:
            _validate_embedding(embedding, query.expected_videos)
        for threshold, tag in THRESHOLDS:
            if (query.name, tag) in existing:
                print(f"[mango-pred-fill] reuse {query.name} ood{tag}: {existing[(query.name, tag)]}", flush=True)
                continue
            _predict_missing(work_dir, query, embedding, threshold, tag, args.gpu)
            total += 1
    print(f"[mango-pred-fill] complete: {total} validated predictions", flush=True)


if __name__ == "__main__":
    main()
