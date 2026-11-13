#!/usr/bin/env python3
"""Serial CLI evaluation of the 260917 Fruit Expert ctx10/ctx20 encoders.

Run with: conda run --no-capture-output -n fineprog \
    python scripts/run_fruit_260917_gaussian_eval.py --gpu 0
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

import h5py
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config_v2 import ConfigV2  # noqa: E402


CONFIG_ROOT = ROOT / "configs_v2"
OUTPUT_ROOT = ROOT / "outputs"
THRESHOLDS = ((0.05, "005"), (0.10, "010"), (0.20, "020"),
              (0.30, "030"), (0.40, "040"), (0.50, "050"))


@dataclass(frozen=True)
class Run:
    slug: str
    stride: int
    ref: str


@dataclass(frozen=True)
class Split:
    slug: str
    ref: str
    count: int
    query: bool = False


RUNS = (Run("ctx10", 10, "fruit_260917_ctx10_ep20k"),
        Run("ctx20", 20, "fruit_260917_ctx20_ep20k"))
SPLITS = (
    Split("train36", "fruit_expert_videos_260917_36vid_train", 36),
    Split("calib14", "fruit_expert_videos_260917_14vid_valid", 14),
    Split("valid4", "fruit_expert_videos_260917_4vid_valid", 4, True),
    Split("fail18", "pick_mango_vid_18vid_fail", 18, True),
    Split("rollout48", "fruits_real_rollout_vid_48vid", 48, True),
)


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, changes: dict) -> None:
    config = _read_yaml(path)
    config.update(changes)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _same_path(actual: str, expected: Path) -> bool:
    return Path(actual).resolve() == expected.resolve()


def _datasets(registry: ConfigV2) -> dict[str, tuple[Path, list[str]]]:
    found = {}
    for split in SPLITS:
        path = Path(registry.resolve_dataset(split.ref)["processed_h5_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as file:
            videos = file.get("videos")
            if not isinstance(videos, h5py.Group) or len(videos) != split.count:
                raise ValueError(f"{path}: expected {split.count} videos")
            ids = sorted(videos)
            if split.query:
                for video_id in ids:
                    video = videos[video_id]
                    frames = video.get("frames")
                    source = Path(str(video.attrs.get("path", "")))
                    fallback = ROOT / "datasets" / "raw" / source.parent.name / source.name
                    if not isinstance(frames, h5py.Dataset) or int(video.attrs.get("num_frames", -1)) != len(frames):
                        raise ValueError(f"{path}:/videos/{video_id}: invalid frames")
                    if source.stem != str(video.attrs.get("action_name", "")):
                        raise ValueError(f"{path}:/videos/{video_id}: invalid action_name")
                    if not source.is_file() and not fallback.is_file():
                        raise FileNotFoundError(f"{path}:/videos/{video_id}: {source}")
        found[split.slug] = (path, ids)
    return found


def _checkpoint(registry: ConfigV2, run: Run) -> tuple[Path, str]:
    entry = registry.resolve_run(run.ref)
    path = Path(entry["checkpoint_path"])
    if not path.is_file() or path.name != "encoder_epoch020000.pt":
        raise FileNotFoundError(f"Expected epoch 20000 checkpoint: {path}")
    if not path.parent.name.endswith(f"-ctxstride{run.stride}"):
        raise ValueError(f"Checkpoint/context stride mismatch: {path}, {run.stride}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("embedding_normalization") != "l2":
        raise ValueError(f"{path}: expected L2 normalization")
    weights = checkpoint["model_state_dict"]
    if tuple(weights["temporal_embedder.proj.weight"].shape) != (128, 512):
        raise ValueError(f"{path}: expected 128-dimensional projection")
    return path, entry["run_name"]


def _worker(configs: Path, script: str, *, task: str | None, gpu: str | None) -> None:
    env = os.environ.copy()
    env["FINEPROG_CONFIGS_V2_DIR"] = str(configs)
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [sys.executable, str(ROOT / script)]
    if task:
        command += ["--task", task]
    print("[fruit-eval] running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _embedding_path(registry: ConfigV2, run: Run, split: Split) -> Path:
    return Path(registry.resolve_embedding(f"fruit_260917_{run.slug}_{split.slug}")["embedding_h5_path"])


def _check_embedding(path: Path, count: int) -> bool:
    if not path.is_file():
        return False
    with h5py.File(path, "r") as file:
        if int(file.attrs.get("embedding_dim", -1)) != 128 or file.attrs.get("embedding_normalization") != "l2":
            raise ValueError(f"{path}: unexpected embedding metadata")
        videos = file.get("videos")
        if not isinstance(videos, h5py.Group) or len(videos) != count:
            raise ValueError(f"{path}: expected {count} embedded videos")
        for video in videos.values():
            emb, steps = video["embeddings"], video["target_steps"]
            if emb.ndim != 2 or emb.shape[1] != 128 or emb.shape[0] != len(steps):
                raise ValueError(f"{path}: invalid embedding shape")
    return True


def _activation(configs: Path, run: Run, checkpoint: Path, split: Split,
                dataset: Path, ids: list[str], gpu: str | None) -> None:
    base = OUTPUT_ROOT / "activation_map"
    expected = [base / checkpoint.parent.name / dataset.stem / video_id /
                "videos" / "temporal_conv2_overlay.mp4" for video_id in ids]
    if all(path.is_file() and path.stat().st_size > 0 for path in expected):
        print(f"[fruit-eval] activation complete: {run.slug}/{split.slug}", flush=True)
        return
    _write_yaml(configs / "eval" / "activation_map.yaml", {
        "dataset_ref": split.ref, "checkpoint_ref": run.ref,
        "dataset_h5_path": str(dataset), "checkpoint_path": str(checkpoint),
        "train_config_path": str(configs / "train.yaml"),
        "output_dir": str(base), "device": "cuda", "embedding_dim": 128,
        "clip_len": 20, "context_size": 2, "context_stride": run.stride,
        "sample_all_stride": 1, "max_videos": "all", "num_workers": 0,
        "output_streams": ["temporal_conv2"], "save_mp4": True,
        "save_png": False, "save_h5": False,
    })
    assert int(_read_yaml(configs / "eval" / "activation_map.yaml")["context_stride"]) == run.stride
    _worker(configs, "evaluate_encoder.py", task="activation_map", gpu=gpu)
    if not all(path.is_file() and path.stat().st_size > 0 for path in expected):
        raise RuntimeError(f"Missing temporal activation videos: {run.slug}/{split.slug}")


def _extract(configs: Path, registry: ConfigV2, run: Run, checkpoint: Path,
             split: Split, gpu: str | None) -> Path:
    output = _embedding_path(registry, run, split)
    if _check_embedding(output, split.count):
        print(f"[fruit-eval] embedding complete: {output}", flush=True)
        return output
    _write_yaml(configs / "extract.yaml", {
        "extract_dataset": split.ref, "checkpoint_ref": run.ref,
        "checkpoint_path": str(checkpoint), "embedding_save_path": str(output),
        "embedding_dim": 128, "clip_len": 20, "context_size": 2,
        "context_stride": run.stride, "sample_all_stride": 1,
        "num_workers": 0,
    })
    assert int(_read_yaml(configs / "extract.yaml")["context_stride"]) == run.stride
    _worker(configs, "extract_embeddings.py", task=None, gpu=gpu)
    if not _check_embedding(output, split.count):
        raise RuntimeError(f"Missing embedding: {output}")
    return output


def _model_files(directory: Path, train_embedding: Path) -> list[Path]:
    matches = []
    for path in directory.glob("*/gaussian_progress_model-covariance_mode-independent-*.h5"):
        with h5py.File(path, "r") as file:
            attrs = file.attrs
            if (attrs.get("task_name") == "gaussian_progress_fitting"
                    and _same_path(str(attrs.get("expert_h5_path", "")), train_embedding)
                    and int(attrs.get("num_bins", -1)) == 20
                    and int(attrs.get("embedding_dim", -1)) == 32
                    and bool(attrs.get("enable_pca", False))):
                matches.append(path)
    return sorted(matches, key=lambda path: path.stat().st_mtime)


def _fit(configs: Path, run: Run, run_name: str, train_embedding: Path,
         gpu: str | None) -> Path:
    directory = OUTPUT_ROOT / "gaussian_progress_fitting" / run_name / train_embedding.stem
    existing = _model_files(directory, train_embedding)
    if existing:
        print(f"[fruit-eval] Gaussian model complete: {existing[-1]}", flush=True)
        return existing[-1]
    config = _read_yaml(configs / "eval" / "gaussian_progress_fitting.yaml")
    config["tsne_viz"]["gaussian_samples_per_bin"] = "match_real_bin_counts"
    config.update({"expert_embedding_ref": f"fruit_260917_{run.slug}_train36",
                   "expert_h5_path": str(train_embedding), "output_dir": str(directory),
                   "num_bins": 20, "enable_pca": True, "pca_dim": 32,
                   "covariance_mode": "independent", "enable_visualization": True})
    (configs / "eval" / "gaussian_progress_fitting.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _worker(configs, "evaluate_encoder.py", task="gaussian_progress_fitting", gpu=gpu)
    models = _model_files(directory, train_embedding)
    if not models:
        raise RuntimeError(f"Missing Gaussian model under {directory}")
    return models[-1]


def _check_prediction(path: Path, model: Path, query: Path, calibration: Path,
                      threshold: float, count: int) -> bool:
    if not path.is_file():
        return False
    with h5py.File(path, "r") as file:
        attrs = file.attrs
        if (attrs.get("task_name") != "gaussian_progress_pred"
                or not _same_path(str(attrs.get("gaussian_model_h5_path", "")), model)
                or not _same_path(str(attrs.get("nonexpert_h5_path", "")), query)
                or not _same_path(str(attrs.get("calibration_h5_path", "")), calibration)
                or not np.isclose(float(attrs.get("ood_p_value_threshold", -1)), threshold)
                or bool(attrs.get("enable_ood_filter", True))):
            raise ValueError(f"Prediction provenance mismatch: {path}")
        videos = file.get("nonexperts")
        if not isinstance(videos, h5py.Group) or len(videos) != count:
            raise ValueError(f"{path}: expected {count} predictions")
        for video in videos.values():
            if not {"target_steps", "posterior", "conformal_p_value", "is_ood_raw", "is_ood"}.issubset(video):
                raise ValueError(f"{path}: incomplete prediction arrays")
    mp4s = list((path.parent / "videos").glob("*.mp4"))
    if len(mp4s) != count or any(item.stat().st_size == 0 for item in mp4s):
        raise ValueError(f"{path}: expected {count} non-empty prediction MP4s")
    return True


def _predict(configs: Path, run: Run, split: Split, model: Path, query: Path,
             calibration: Path, threshold: float, tag: str, gpu: str | None) -> None:
    destination = OUTPUT_ROOT / "fruit_260917_gaussian_eval" / run.slug / split.slug / f"ood{tag}"
    if destination.exists():
        files = list(destination.glob("gaussian_progress_pred-*.h5"))
        if len(files) != 1 or not _check_prediction(files[0], model, query, calibration, threshold, split.count):
            raise RuntimeError(f"Incomplete existing prediction: {destination}")
        print(f"[fruit-eval] prediction complete: {run.slug}/{split.slug}/ood{tag}", flush=True)
        return
    _write_yaml(configs / "eval" / "gaussian_progress_pred.yaml", {
        "gaussian_model_h5_path": str(model), "nonexpert_h5_path": str(query),
        "enable_calibration": True, "calibration_h5_path": str(calibration),
        "ood_p_value_threshold": threshold, "enable_ood_filter": False,
        "save_posterior": True, "enable_visualization": False, "enable_video": True,
    })
    source_parent = OUTPUT_ROOT / "gaussian_progress_pred" / model.stem / query.stem
    before = set(source_parent.glob("*/gaussian_progress_pred-*.h5"))
    _worker(configs, "evaluate_encoder.py", task="gaussian_progress_pred", gpu=gpu)
    created = list(set(source_parent.glob("*/gaussian_progress_pred-*.h5")) - before)
    if len(created) != 1 or not _check_prediction(created[0], model, query, calibration, threshold, split.count):
        raise RuntimeError(f"Expected one complete prediction under {source_parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    created[0].parent.rename(destination)
    print(f"[fruit-eval] saved: {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES for CLI subprocesses")
    parser.add_argument("--stride", type=int, choices=(10, 20), action="append",
                        help="Run one stride; repeat to select both (default: both)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()

    registry = ConfigV2()
    selected = [run for run in RUNS if args.stride is None or run.stride in args.stride]
    datasets = _datasets(registry)
    checkpoints = {run.slug: _checkpoint(registry, run) for run in selected}
    for run in selected:
        checkpoint, _ = checkpoints[run.slug]
        print(f"[fruit-eval] preflight {run.slug}: {checkpoint}; context_stride={run.stride}", flush=True)
    for split in SPLITS:
        path, _ = datasets[split.slug]
        print(f"[fruit-eval] preflight {split.slug}: {path} ({split.count} videos)", flush=True)
    if args.dry_run:
        print("[fruit-eval] dry run complete; no evaluation artifacts written", flush=True)
        return

    work_dir = args.work_dir or OUTPUT_ROOT / "fruit_260917_gaussian_eval" / "configs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    if work_dir.exists():
        raise FileExistsError(work_dir)
    work_dir.mkdir(parents=True)
    for run in selected:
        checkpoint, run_name = checkpoints[run.slug]
        configs = work_dir / run.slug
        shutil.copytree(CONFIG_ROOT, configs)
        _write_yaml(configs / "train.yaml", {"context_stride": run.stride})
        print(f"[fruit-eval] start {run.slug}; config={configs}", flush=True)
        for split in SPLITS:
            if split.query:
                dataset, ids = datasets[split.slug]
                _activation(configs, run, checkpoint, split, dataset, ids, args.gpu)
        embeddings = {split.slug: _extract(configs, registry, run, checkpoint, split, args.gpu)
                      for split in SPLITS}
        model = _fit(configs, run, run_name, embeddings["train36"], args.gpu)
        for split in SPLITS:
            if split.query:
                for threshold, tag in THRESHOLDS:
                    _predict(configs, run, split, model, embeddings[split.slug],
                             embeddings["calib14"], threshold, tag, args.gpu)
    print("[fruit-eval] all selected runs complete", flush=True)


if __name__ == "__main__":
    main()
