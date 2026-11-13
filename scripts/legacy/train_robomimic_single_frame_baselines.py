#!/usr/bin/env python3
"""Can local20k -> frozen HF1 20k -> square local20k -> frozen HF1 20k.

From fineprog:
    conda run --no-capture-output -n fineprog python -u \
        scripts/legacy/train_robomimic_single_frame_baselines.py --gpu 0

--dry-run writes four config snapshots without starting training. Every launch
uses a fresh .vibe/.codex/sol experiment directory; optimizer-state resume is not
implemented. Logs stream to the terminal and each stage's train.log.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / ".vibe/.codex/sol"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
# Share the reference recipe and checkpoint contract, not its pick_mango queue.
from train_pick_mango_single_frame_baselines_serial import (  # noqa: E402
    COMMON_CONFIG, EPOCHS, MEMORY_CONFIG, SOURCE_CONFIG_ROOT,
    _load_yaml, _require_cuda, _save_manifest, _write_yaml,
    sha256, validate_checkpoint,
)


TASKS = ("can", "square")
STAGES = ("local", "hf1")


def validate_sources() -> None:
    import h5py

    registry = _load_yaml(SOURCE_CONFIG_ROOT / "registry/datasets.yaml")["datasets"]
    for task in TASKS:
        source_ids = []
        for count, split in ((36, "train"), (4, "valid")):
            alias = f"robomimic_{task}_ph_{count}vid_{split}"
            filename = f"robomimic_{task}_ph-{count}vid_{split}.h5"
            if registry[alias]["processed_h5"] != filename:
                raise ValueError(f"Unexpected dataset mapping: {alias}")
            with h5py.File(PROJECT_ROOT / "datasets/processed" / filename, "r") as handle:
                videos = handle["videos"]
                if len(videos) != count:
                    raise ValueError(f"Expected {count} videos: {filename}")
                source_ids.append({group.attrs["action_name"] for group in videos.values()})
        if source_ids[0] & source_ids[1]:
            raise ValueError(f"Train/valid overlap: {task}")
    composite = _load_yaml(SOURCE_CONFIG_ROOT / COMMON_CONFIG["loss_config"])
    if [(item["name"], item["weight"]) for item in composite["components"]] != [
        ("tcc", 1.0), ("temporal_triplet", 0.5),
    ]:
        raise ValueError("Expected reference TCC x 1.0 + temporal triplet x 0.5")


def prepare(root: Path) -> dict:
    validate_sources()
    if not root.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"Output root must be inside {OUTPUT_ROOT}")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Use a new/empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "state": "prepared", "epochs_per_stage": EPOCHS,
        "stage_order": [f"{task}/{stage}" for task in TASKS for stage in STAGES],
        "stages": {},
    }
    for task in TASKS:
        for stage in STAGES:
            stage_root = root / task / stage
            config_root = stage_root / "configs_v2"
            shutil.copytree(SOURCE_CONFIG_ROOT, config_root)
            config = _load_yaml(config_root / "train.yaml")
            config.update(deepcopy(COMMON_CONFIG))
            config.pop("data_seed", None)
            config.update(
                train_dataset=f"robomimic_{task}_ph_36vid_train",
                train_base="only_bn" if stage == "local" else "frozen",
                train_embedding=stage == "local", pretrained=stage == "local",
                checkpoint_dir=str(stage_root / "checkpoints"),
                run_name_prefix=f"single-frame-{task}-{stage}",
            )
            config["in_training_eval"]["eval_dataset_ref"] = f"robomimic_{task}_ph_4vid_valid"
            config["memory"] = {"enabled": False}
            if stage == "hf1":
                config["memory"] = {**deepcopy(MEMORY_CONFIG), "local_checkpoint_path": None}
            _write_yaml(config_root / "train.yaml", config)
            manifest["stages"][f"{task}/{stage}"] = {
                "state": "pending", "task": task, "stage": stage,
                "config_root": str(config_root), "log_path": str(stage_root / "train.log"),
                "checkpoint_root": config["checkpoint_dir"],
                "train_config_sha256": sha256(config_root / "train.yaml"),
            }
    _save_manifest(root, manifest)
    return manifest


def run_live(command: list[str], environment: dict, log_path: Path) -> int:
    """Tee raw bytes so tqdm carriage returns are visible immediately."""
    with log_path.open("xb") as log:
        with subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        ) as process:
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
    _require_cuda()
    manifest["state"] = "running"
    for index, name in enumerate(manifest["stage_order"], 1):
        job = manifest["stages"][name]
        source = None
        try:
            config_path = Path(job["config_root"]) / "train.yaml"
            if job["stage"] == "hf1":
                source = manifest["stages"][f"{job['task']}/local"]
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
            if gpu is not None:
                environment["CUDA_VISIBLE_DEVICES"] = gpu
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
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES; defaults to inherited selection")
    parser.add_argument("--output-root", type=Path, help="New/empty directory inside .vibe/.codex/sol")
    parser.add_argument("--dry-run", action="store_true", help="Prepare configs; do not train")
    args = parser.parse_args(argv)
    suffix = "_dry_run" if args.dry_run else ""
    root = (args.output_root or OUTPUT_ROOT /
            (datetime.now().strftime("%Y%m%d-%H%M%S-%f") +
             "_robomimic_single_frame_baselines" + suffix)).expanduser().resolve()
    try:
        manifest = prepare(root)
        print(f"Experiment: {root}", flush=True)
        if args.dry_run:
            print("Prepared can/local, can/hf1, square/local, square/hf1; no training started.", flush=True)
            return 0
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        return run_queue(root, manifest, args.gpu)
    except KeyboardInterrupt:
        print("Training queue interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Training queue failed: {error}", file=sys.stderr)
        return error.returncode if isinstance(error, subprocess.CalledProcessError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
