#!/usr/bin/env python3
"""Train Pick Mango encoders with context strides 10 and 20, serially.

The script takes an isolated snapshot of ``configs_v2`` for each run and
overrides only ``train_dataset`` and ``context_stride``. The shared repository
configuration is left untouched.

Usage:
    conda run --no-capture-output -n fineprog \
        python scripts/train_pick_mango_context_stride_ablation.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
CONFIGS_ENV_VAR = "FINEPROG_CONFIGS_V2_DIR"
TRAIN_DATASET = "pick_mango_vid_36vid_train"
EXPECTED_H5 = PROJECT_ROOT / "datasets/processed/pick_mango_vid-36vid_train.h5"
EXPECTED_VIDEO_COUNT = 36
DEFAULT_CONTEXT_STRIDES = (10, 20)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def validate_training_dataset() -> Path:
    """Ensure the registry alias resolves to the checkpoint's original H5."""
    registry_path = SOURCE_CONFIG_ROOT / "registry" / "datasets.yaml"
    registry = _load_yaml(registry_path).get("datasets", {})
    if TRAIN_DATASET not in registry:
        raise KeyError(f"Missing dataset registry key: {TRAIN_DATASET}")
    processed_h5 = registry[TRAIN_DATASET].get("processed_h5")
    if processed_h5 != EXPECTED_H5.name:
        raise RuntimeError(
            f"{TRAIN_DATASET} resolves to {processed_h5!r}; "
            f"expected {EXPECTED_H5.name!r}"
        )
    if not EXPECTED_H5.is_file():
        raise FileNotFoundError(EXPECTED_H5)
    with h5py.File(EXPECTED_H5, "r") as h5_file:
        videos = h5_file.get("videos")
        if videos is None or len(videos) != EXPECTED_VIDEO_COUNT:
            actual = None if videos is None else len(videos)
            raise RuntimeError(
                f"Expected {EXPECTED_VIDEO_COUNT} videos in {EXPECTED_H5}; got {actual}"
            )
    return EXPECTED_H5


def make_config(source: Path, destination: Path, context_stride: int) -> Path:
    """Copy configs and apply only the training-dataset and stride overrides."""
    shutil.copytree(source, destination)
    train_path = destination / "train.yaml"
    train_config = _load_yaml(train_path)
    train_config["train_dataset"] = TRAIN_DATASET
    train_config["context_stride"] = int(context_stride)
    _dump_yaml(train_path, train_config)
    return train_path


def run_training_worker(
    config_root: Path,
    context_stride: int,
    sweep_group: str,
) -> int:
    """Run one encoder job and record its dataset/stride in W&B metadata."""
    os.environ[CONFIGS_ENV_VAR] = str(config_root)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import wandb
    import torch

    import train_encoder

    cuda_available = torch.cuda.is_available()
    print(
        "Worker CUDA check: "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
        f"available={cuda_available}, device_count={torch.cuda.device_count()}",
        flush=True,
    )
    if not cuda_available:
        raise RuntimeError(
            "CUDA is unavailable in the training worker; refusing to run this "
            "long ablation on CPU. Check GPU device permissions/visibility."
        )
    print(f"Training device: {torch.cuda.get_device_name(0)}", flush=True)

    train_config = train_encoder._TRAIN_V2
    if train_config.get("train_dataset") != TRAIN_DATASET:
        raise RuntimeError(
            f"Worker selected {train_config.get('train_dataset')!r}; "
            f"expected {TRAIN_DATASET!r}"
        )
    if int(train_config.get("context_stride", -1)) != context_stride:
        raise RuntimeError(
            f"Worker context_stride is {train_config.get('context_stride')!r}; "
            f"expected {context_stride}"
        )
    if Path(train_config.get("h5_path", "")).resolve() != EXPECTED_H5.resolve():
        raise RuntimeError(
            f"Worker H5 path is {train_config.get('h5_path')!r}; "
            f"expected {str(EXPECTED_H5)!r}"
        )

    original_init = wandb.init

    def init_with_ablation_metadata(*args, **kwargs):
        base_name = kwargs.get("name")
        if base_name:
            kwargs["name"] = f"{base_name}-ctxstride{context_stride}"
        kwargs.setdefault("group", sweep_group)
        kwargs.setdefault("job_type", "pick_mango_context_stride_ablation")
        run_config = dict(kwargs.get("config") or {})
        run_config.update(
            {
                "train_dataset": TRAIN_DATASET,
                "train_h5": str(EXPECTED_H5),
                "context_stride": context_stride,
                "sweep_group": sweep_group,
            }
        )
        kwargs["config"] = run_config
        return original_init(*args, **kwargs)

    wandb.init = init_with_ablation_metadata
    try:
        train_encoder.train(
            num_epochs=train_config.get("num_epochs", 20_000),
            batch_size=train_config.get("batch_size", 12),
            learning_rate=train_config.get("learning_rate", 4e-4),
            log_every=train_config.get("log_every", 100),
            num_workers=train_config.get("num_workers", 4),
            checkpoint_every=train_config.get("checkpoint_every", 2_000),
            checkpoint_dir=train_config.get("checkpoint_dir", "checkpoint"),
            h5_path=None,
            register=False,
            register_alias=None,
        )
    finally:
        wandb.init = original_init
    return 0


def _worker_env(config_root: Path, gpu: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env[CONFIGS_ENV_VAR] = str(config_root)
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run serial context-stride ablations on the same 36-video Pick "
            "Mango training H5 used by encoder_epoch020000.pt."
        )
    )
    parser.add_argument(
        "--gpu",
        help="CUDA_VISIBLE_DEVICES value for both runs (default: inherit).",
    )
    parser.add_argument(
        "--context-strides",
        type=int,
        nargs="+",
        default=DEFAULT_CONTEXT_STRIDES,
        help="Context strides to run serially (default: 10 20).",
    )
    parser.add_argument(
        "--sweep-group",
        default=None,
        help="W&B group for both runs (default: timestamped ablation name).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate configs without starting training.",
    )
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-context-stride", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sweep-group", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_config is not None:
        if args.worker_context_stride is None or args.worker_sweep_group is None:
            parser.error("worker mode requires stride and sweep group")
        return run_training_worker(
            args.worker_config,
            args.worker_context_stride,
            args.worker_sweep_group,
        )

    context_strides = tuple(args.context_strides)
    if not context_strides or any(stride <= 0 for stride in context_strides):
        parser.error("--context-strides must contain positive integers")
    if len(set(context_strides)) != len(context_strides):
        parser.error("--context-strides must not contain duplicates")
    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing config directory: {SOURCE_CONFIG_ROOT}")

    train_h5 = validate_training_dataset()
    sweep_group = args.sweep_group or (
        "pick_mango_vid_36vid_train_context_stride_"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    print(f"Training H5: {train_h5} ({EXPECTED_VIDEO_COUNT} videos)", flush=True)
    with tempfile.TemporaryDirectory(prefix="fineprog_pick_mango_ctxstride_") as temp_dir:
        jobs = []
        for context_stride in context_strides:
            config_root = Path(temp_dir) / f"ctxstride{context_stride}"
            train_path = make_config(SOURCE_CONFIG_ROOT, config_root, context_stride)
            jobs.append((context_stride, config_root, train_path))

        for index, (context_stride, config_root, train_path) in enumerate(jobs, start=1):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-config",
                str(config_root),
                "--worker-context-stride",
                str(context_stride),
                "--worker-sweep-group",
                sweep_group,
            ]
            print(
                f"[{index}/{len(jobs)}] {TRAIN_DATASET}, "
                f"context_stride={context_stride}",
                flush=True,
            )
            if args.dry_run:
                config = _load_yaml(train_path)
                print(f"  config: {train_path}", flush=True)
                print(f"  h5: {train_h5}", flush=True)
                print(f"  command: {' '.join(command)}", flush=True)
                continue

            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=_worker_env(config_root, args.gpu),
            )
            if result.returncode:
                print(
                    f"context_stride={context_stride} failed with exit code "
                    f"{result.returncode}; stopping before the next run.",
                    file=sys.stderr,
                    flush=True,
                )
                return result.returncode

    if args.dry_run:
        print("Dry run complete; no training started.", flush=True)
    else:
        print("Both Pick Mango context-stride runs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
