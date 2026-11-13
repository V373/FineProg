#!/usr/bin/env python3
"""Train Pick Mango encoders serially without changing configs_v2/train.yaml.

Usage:
    conda run --no-capture-output -n fineprog \
        python scripts/train_pick_mango_encoders_serial.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
DATASETS = (
    "pick_mango_filtered_vid_36vid_train",
    "pick_mango_vid_36vid_train",
)


def make_config(source: Path, destination: Path, dataset: str) -> None:
    """Copy configs and override only the training-dataset registry alias."""
    shutil.copytree(source, destination)
    train_path = destination / "train.yaml"
    config = yaml.safe_load(train_path.read_text(encoding="utf-8")) or {}
    config["train_dataset"] = dataset
    train_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train filtered and unfiltered Pick Mango encoders serially."
    )
    parser.add_argument(
        "--gpu",
        help="CUDA_VISIBLE_DEVICES value for both runs (default: inherit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the two isolated configs without starting training.",
    )
    args = parser.parse_args()

    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing config directory: {SOURCE_CONFIG_ROOT}")

    with tempfile.TemporaryDirectory(prefix="fineprog_pick_mango_") as temp_dir:
        config_roots = []
        for dataset in DATASETS:
            config_root = Path(temp_dir) / dataset
            make_config(SOURCE_CONFIG_ROOT, config_root, dataset)
            config_roots.append((dataset, config_root))

        for index, (dataset, config_root) in enumerate(config_roots, start=1):
            command = [sys.executable, str(PROJECT_ROOT / "train_encoder.py")]
            env = os.environ.copy()
            env["FINEPROG_CONFIGS_V2_DIR"] = str(config_root)
            if args.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = args.gpu

            print(f"[{index}/{len(DATASETS)}] {dataset}", flush=True)
            if args.dry_run:
                print(f"  config: {config_root}", flush=True)
                print(f"  command: {' '.join(command)}", flush=True)
                continue

            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
            if result.returncode:
                print(
                    f"{dataset} failed with exit code {result.returncode}; stopping.",
                    file=sys.stderr,
                )
                return result.returncode

        if args.dry_run:
            print("Dry run complete; no training started.")
            return 0

    print("Both Pick Mango encoder runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
