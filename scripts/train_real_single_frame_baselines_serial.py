#!/usr/bin/env python3
"""Prepare or train drawer_place/local -> HF1 -> towel/local -> HF1.

Use the fineprog conda environment. --dry-run validates the processed datasets
and writes four config snapshots without importing PyTorch or starting training.
Use a fresh --output-root for the actual launch; optimizer resume is not supported.

Example (from fineprog):
    python scripts/train_real_single_frame_baselines_serial.py \
        --dataset-manifest /path/to/data_manifest.json --gpu 0 --dry-run

Each stage uses the can/square single-frame 20k recipe. Frozen HF1 always starts
from this queue's completed local checkpoint for the same task. HF1 retains the
full causal episode history; history_frames=1 describes its input token only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / ".vibe/.codex/astra"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_pick_mango_single_frame_baselines_serial import (  # noqa: E402
    COMMON_CONFIG, EPOCHS, MEMORY_CONFIG, SOURCE_CONFIG_ROOT,
    _load_yaml, _require_cuda, _save_manifest, _write_yaml,
    sha256, validate_checkpoint,
)


TASKS = ("drawer_place", "towel")
STAGES = ("local", "hf1")


def validate_sources(manifest_path: Path) -> dict:
    """Validate the immutable data selection before preparing any run."""
    import h5py

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Expected dataset manifest schema_version=1")
    aliases = set()
    for task in TASKS:
        splits = data["tasks"][task]["splits"]
        seen_sources = set()
        for split in ("train", "valid", "calibration"):
            if split == "calibration" and split not in splits:
                continue
            entry = splits[split]
            count = entry["count"]
            expected = {"train": 36, "valid": 4}.get(split, count)
            if count != expected or count <= 0:
                raise ValueError(f"{task}/{split}: expected {expected} nonempty videos")
            path = Path(entry["path"])
            if not path.is_absolute():
                raise ValueError(f"Dataset path must be absolute: {path}")
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Dataset SHA256 mismatch: {path}")
            alias = entry["alias"]
            if alias in aliases:
                raise ValueError(f"Duplicate dataset alias: {alias}")
            aliases.add(alias)
            with h5py.File(path, "r") as handle:
                videos = handle["videos"]
                if len(videos) != count:
                    raise ValueError(f"{task}/{split}: expected {count} videos in {path}")
                for video in videos.values():
                    frames = video["frames"]
                    if (frames.ndim != 4 or frames.shape[0] < 1
                            or frames.shape[1:] != (224, 224, 3)
                            or frames.dtype.name != "uint8"):
                        raise ValueError(f"Expected uint8 RGB [T,224,224,3]: {frames.name}")
                    source = video.attrs.get("source_episode", video.attrs.get("action_name"))
                    if isinstance(source, bytes):
                        source = source.decode("utf-8")
                    if not source:
                        raise ValueError(f"Missing source episode: {path}:{video.name}")
                    if source in seen_sources:
                        raise ValueError(f"Duplicate/overlapping source episode: {task}/{source}")
                    seen_sources.add(source)
    composite = _load_yaml(SOURCE_CONFIG_ROOT / COMMON_CONFIG["loss_config"])
    if [(item["name"], item["weight"]) for item in composite["components"]] != [
        ("tcc", 1.0), ("temporal_triplet", 0.5),
    ]:
        raise ValueError("Expected reference TCC x 1.0 + temporal triplet x 0.5")
    for filename, expected in {
        "loss_tcc.yaml": {"stochastic_matching": False, "normalize_embeddings": False,
                          "loss_type": "regression_mse_var", "similarity_type": "l2",
                          "softmax_temperature": 0.001, "variance_lambda": 0.001,
                          "normalize_indices": True},
        "loss_temporal_triplet.yaml": {"margin": 0.2, "num_triplets_fraction": 1.0,
                                       "reduction": "mean", "normalize_embeddings": False,
                                       "squared_l2": True, "capped": False,
                                       "dynamic_margin_enabled": False},
    }.items():
        config = _load_yaml(SOURCE_CONFIG_ROOT / "loss" / filename)
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Reference loss changed: {filename}: {key}")
    return data


def prepare(root: Path, manifest_path: Path) -> dict:
    data = validate_sources(manifest_path)
    if not root.is_relative_to(OUTPUT_ROOT.resolve()) or root == OUTPUT_ROOT.resolve():
        raise ValueError(f"Output root must be a new directory inside {OUTPUT_ROOT}")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Use a new/empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "state": "prepared", "output_root": str(root),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256(manifest_path),
        "datasets": data["tasks"], "epochs_per_stage": EPOCHS,
        "stage_order": [f"{task}/{stage}" for task in TASKS for stage in STAGES],
        "stages": {},
    }
    for task in TASKS:
        splits = data["tasks"][task]["splits"]
        for stage in STAGES:
            stage_root = root / task / stage
            config_root = stage_root / "configs_v2"
            shutil.copytree(SOURCE_CONFIG_ROOT, config_root)
            registry = _load_yaml(config_root / "registry/datasets.yaml")
            for split, entry in splits.items():
                record = {"processed_h5": entry["path"],
                          "display_name": f"{task} {split} ({entry['count']} reviewed clips)"}
                if split == "train":
                    record["validation_dataset_ref"] = splits["valid"]["alias"]
                registry["datasets"][entry["alias"]] = record
            _write_yaml(config_root / "registry/datasets.yaml", registry)
            config = _load_yaml(config_root / "train.yaml")
            config.update(deepcopy(COMMON_CONFIG))
            config.pop("data_seed", None)
            config.update(
                train_dataset=splits["train"]["alias"],
                train_base="only_bn" if stage == "local" else "frozen",
                train_embedding=stage == "local", pretrained=stage == "local",
                checkpoint_dir=str(stage_root / "checkpoints"),
                run_name_prefix=f"single-frame-{task}-{stage}",
            )
            config["in_training_eval"] = {
                "enabled": True, "selected_eval_tasks_in_training": ["kendalls_tau"],
                "eval_freq_in_training": 1, "eval_dataset_ref": splits["valid"]["alias"],
                "log_images_to_wandb": False,
                "kendalls_tau": {"kendall_stride": 5, "kendall_distance": "sqeuclidean",
                                 "save_heatmap": False},
            }
            config["memory"] = {"enabled": False}
            if stage == "hf1":
                config["memory"] = {**deepcopy(MEMORY_CONFIG), "local_checkpoint_path": None}
            _write_yaml(config_root / "train.yaml", config)
            job = {
                "state": "pending", "task": task, "stage": stage,
                "config_root": str(config_root), "log_path": str(stage_root / "train.log"),
                "checkpoint_root": config["checkpoint_dir"],
                "train_config_sha256": sha256(config_root / "train.yaml"),
            }
            if stage == "hf1":
                job.update(source_stage=f"{task}/local", source_checkpoint=None,
                           source_checkpoint_sha256=None)
            manifest["stages"][f"{task}/{stage}"] = job
    _save_manifest(root, manifest)
    return manifest


def runtime_environment(root: Path) -> dict:
    """Keep generated caches and short multiprocessing sockets in this session."""
    session_root = OUTPUT_ROOT / root.relative_to(OUTPUT_ROOT).parts[0]
    tmp_root = session_root / "tmp"
    if len(os.fsencode(str(tmp_root / "pymp-xxxxxxxx/listener-xxxxxxxx"))) > 107:
        # Renamed experiment folders may retain a short compatibility symlink.
        # Use it for sockets while keeping the actual temporary files in-session.
        aliases = sorted(OUTPUT_ROOT.iterdir(), key=lambda p: len(str(p)))
        for alias in aliases:
            candidate = alias / "tmp"
            if (alias.is_symlink() and alias.resolve() == session_root.resolve()
                    and len(os.fsencode(str(candidate / "pymp-xxxxxxxx/listener-xxxxxxxx"))) <= 107):
                tmp_root = candidate
                break
        else:
            raise ValueError("Session name is too long for sockets; use a short output root or compatibility symlink")
    paths = {
        "TMPDIR": tmp_root, "TMP": tmp_root, "TEMP": tmp_root,
        "XDG_CACHE_HOME": root / "cache/xdg", "TORCH_HOME": root / "cache/torch",
        "WANDB_CACHE_DIR": root / "cache/wandb", "WANDB_DATA_DIR": root / "cache/wandb-data",
        "TRITON_CACHE_DIR": root / "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": root / "cache/torchinductor",
        "CUDA_CACHE_PATH": root / "cache/cuda", "MPLCONFIGDIR": root / "cache/matplotlib",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    # Reuse the already-installed ImageNet V1 file; never download into a global cache.
    weight_name = "resnet50-0676ba61.pth"
    installed = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache/torch")))
    weight_source = installed / "hub/checkpoints" / weight_name
    target = paths["TORCH_HOME"] / "hub/checkpoints" / weight_name
    if not target.exists():
        if not weight_source.is_file():
            raise FileNotFoundError(f"ImageNet V1 weights are missing: {weight_source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(weight_source.resolve())
    return {key: str(path) for key, path in paths.items()}


def run_live(command: list[str], environment: dict, log_path: Path) -> int:
    with log_path.open("xb") as log:
        with subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                while chunk := process.stdout.read1(65536):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    log.write(chunk)
                    log.flush()
                return process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise


def verify_frozen_local(local_path: Path, hf1_path: Path) -> None:
    import torch

    def local_state(path):
        state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
        return {key.removeprefix("module."): value for key, value in state.items()
                if not key.removeprefix("module.").startswith("memory.")}

    before, after = local_state(local_path), local_state(hf1_path)
    if before.keys() != after.keys() or any(
        not torch.equal(value, after[key]) for key, value in before.items()
    ):
        raise ValueError("Frozen HF1 changed local encoder parameters or buffers")


def run_queue(root: Path, manifest: dict, gpu: str | None) -> int:
    runtime = runtime_environment(root)
    os.environ.update(runtime)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    tempfile.tempdir = runtime["TMPDIR"]
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    _require_cuda()
    manifest["state"] = "running"
    for index, name in enumerate(manifest["stage_order"], 1):
        job = manifest["stages"][name]
        source = None
        try:
            config_path = Path(job["config_root"]) / "train.yaml"
            if job["stage"] == "hf1":
                source = manifest["stages"][job["source_stage"]]
                config = _load_yaml(config_path)
                config["memory"]["local_checkpoint_path"] = source["checkpoint_path"]
                _write_yaml(config_path, config)
                job.update(source_checkpoint=source["checkpoint_path"],
                           source_checkpoint_sha256=source["checkpoint_sha256"])
            job.update(state="running", train_config_sha256=sha256(config_path))
            manifest["active_stage"] = name
            _save_manifest(root, manifest)
            environment = os.environ.copy()
            for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_NAME"):
                environment.pop(key, None)
            environment.update(
                FINEPROG_CONFIGS_V2_DIR=job["config_root"], PYTHONUNBUFFERED="1",
                WANDB_RUN_GROUP=root.name, WANDB_JOB_TYPE=job["stage"],
                WANDB_DIR=str(Path(job["log_path"]).parent),
            )
            print(f"\n[{index}/4] {name}: {job['log_path']}", flush=True)
            command = [sys.executable, "-u", str(PROJECT_ROOT / "train_encoder.py")]
            code = run_live(command, environment, Path(job["log_path"]))
            job["exit_code"] = code
            if code:
                raise subprocess.CalledProcessError(code, command)
            finals = list(Path(job["checkpoint_root"]).glob(f"*/encoder_epoch{EPOCHS:06d}.pt"))
            if len(finals) != 1:
                raise RuntimeError(f"Expected one final 20k checkpoint, found {len(finals)}")
            job.update(validate_checkpoint(finals[0], job["stage"],
                                           job.get("source_checkpoint_sha256")))
            if source is not None:
                verify_frozen_local(Path(source["checkpoint_path"]), finals[0])
                job["local_state_unchanged"] = True
            job["state"] = "complete"
            _save_manifest(root, manifest)
            print(f"[{index}/4] Complete: {job['checkpoint_path']}", flush=True)
        except BaseException as error:
            job.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=str(error))
            manifest["state"] = job["state"]
            _save_manifest(root, manifest)
            raise
    manifest.update(state="complete", active_stage=None)
    _save_manifest(root, manifest)
    print(f"All four stages complete: {root / 'manifest.json'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        help="New/empty output directory inside fineprog/.vibe/.codex/astra")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES for all four stages (default: inherit)")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and prepare configs only")
    args = parser.parse_args(argv)
    root = (args.output_root or OUTPUT_ROOT /
            (datetime.now().strftime("%y%m%d%H%M%S") + "_r")).expanduser().resolve()
    manifest_path = args.dataset_manifest.expanduser().resolve()
    try:
        manifest = prepare(root, manifest_path)
        print(f"Experiment: {root}", flush=True)
        if args.dry_run:
            print("Four configs ready; no training started. Use a fresh output root to launch.", flush=True)
            return 0
        return run_queue(root, manifest, args.gpu)
    except KeyboardInterrupt:
        print("Training queue interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Training queue failed: {error}", file=sys.stderr)
        return error.returncode if isinstance(error, subprocess.CalledProcessError) else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
