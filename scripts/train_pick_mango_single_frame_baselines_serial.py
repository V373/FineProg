#!/usr/bin/env python3
"""Train single-frame local, frozen HF1, then warm only-BN HF1 baselines.

Launch from fineprog with:
    conda run --no-capture-output -n fineprog python \
        scripts/train_pick_mango_single_frame_baselines_serial.py --gpu 0

Each stage runs 20,000 epochs with the original TCC + triplet loss and valid4
monitoring. Both HF1 stages start from the SAME newly trained local checkpoint;
the warm stage trains BN affine/statistics and the temporal embedder as well.
Persistent configs, train.log, checkpoints and manifest.json are written under
.vide/.codex/astra/<timestamp>_pick_mango_single_frame_baselines by default.

--dry-run prepares configs without CUDA or training. Its HF1 configs have an
unresolved local checkpoint dependency until the launcher completes stage one.
Use a fresh --output-root for each launch, including after a dry run. To reuse a
completed local stage, pass --local-checkpoint /path/encoder_epoch020000.pt;
this starts both HF1 stages anew. Optimizer-state resume is not implemented.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
TRAIN_DATASET = "pick_mango_vid_36vid_train"
EXPECTED_H5 = PROJECT_ROOT / "datasets/processed/pick_mango_vid-36vid_train.h5"
STAGES = ("local", "hf1", "hf1_warm_only_bn")
EPOCHS = 20_000
COMMON_CONFIG = {
    "train_dataset": TRAIN_DATASET, "backbone_name": "resnet50_conv4c",
    "pretrained": True, "embedding_dim": 128, "embedding_normalization": "l2",
    "single_frame_context": True, "seed": 42, "num_epochs": EPOCHS,
    "batch_size": 12, "num_workers": 4, "limit_cpu_threads": False,
    "transport_frames_as_uint8": True, "clip_len": 20, "context_size": 2,
    "context_stride": 20, "sampling_strategy": "offset_uniform", "random_offset": 1,
    "learning_rate": 0.0004, "log_every": 100, "checkpoint_every": 2000,
    "extract_backbone_cache": False, "cache_refresh_every": 500,
    "loss_name": "composite", "loss_config": "loss/loss_composite_tcc_triplet.yaml",
    "in_training_eval": {
        "enabled": True,
        "selected_eval_tasks_in_training": [
            "latent_distance_heatmap", "kendalls_tau", "activation_map",
        ],
        "eval_freq_in_training": 1, "eval_dataset_ref": None,
        "log_images_to_wandb": True,
        "latent_distance_heatmap": {
            "selected_video_index": 0, "normalize_embeddings": False,
            "convert_to_similarity": True, "similarity_tau": 1.0,
        },
        "kendalls_tau": {"kendall_stride": 5, "kendall_distance": "sqeuclidean"},
        "activation_map": {
            "max_videos": 1, "save_mp4": True, "save_png": False,
            "save_h5": False, "sample_all_stride": 1, "overlay_alpha": 0.45,
            "colormap": "jet", "resize_to": 224,
        },
    },
}
MEMORY_CONFIG = {
    "enabled": True, "mode": None, "dual_branch": False,
    "source_checkpoint_path": None, "local_config_path": None,
    "d_model": 128, "d_state": 64, "d_conv": 4, "expand": 2,
    "headdim": 64, "ngroups": 1, "chunk_size": 256, "alpha_init": 1.0,
    "use_residual": False, "normalize_input": True,
    "history_frames": 1, "history_stride": 1,
    "cache_local_embeddings": True, "freeze_bn_stats": True,
    "visual_chunk_size": 20, "checkpoint_visual": True,
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_manifest(root: Path, manifest: dict) -> None:
    manifest["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "manifest.json")


def validate_sources() -> None:
    """Refuse accidental data/loss changes before preparing the ablation."""
    import h5py

    registry = _load_yaml(SOURCE_CONFIG_ROOT / "registry/datasets.yaml")["datasets"]
    if registry[TRAIN_DATASET]["processed_h5"] != EXPECTED_H5.name:
        raise ValueError(f"{TRAIN_DATASET} no longer resolves to {EXPECTED_H5.name}")
    with h5py.File(EXPECTED_H5, "r") as handle:
        if "videos" not in handle or len(handle["videos"]) != 36:
            raise ValueError(f"Expected 36 training videos in {EXPECTED_H5}")
    composite = _load_yaml(SOURCE_CONFIG_ROOT / COMMON_CONFIG["loss_config"])
    expected = [("tcc", "tcc", 1.0, "loss_tcc.yaml"),
                ("temporal_triplet", "temporal_triplet", 0.5, "loss_temporal_triplet.yaml")]
    actual = [(item.get("alias"), item.get("name"), item.get("weight"), item.get("config_file"))
              for item in composite.get("components", [])]
    if actual != expected:
        raise ValueError("Expected the reference TCC x 1.0 + temporal triplet x 0.5 loss")
    expected_losses = {
        "loss_tcc.yaml": {
            "stochastic_matching": False, "normalize_embeddings": False,
            "loss_type": "regression_mse_var", "similarity_type": "l2",
            "softmax_temperature": 0.001, "variance_lambda": 0.001,
            "normalize_indices": True,
        },
        "loss_temporal_triplet.yaml": {
            "margin": 0.2, "num_triplets_fraction": 1.0, "reduction": "mean",
            "normalize_embeddings": False, "squared_l2": True, "capped": False,
            "dynamic_margin_enabled": False,
        },
    }
    for filename, fields in expected_losses.items():
        config = _load_yaml(SOURCE_CONFIG_ROOT / "loss" / filename)
        for key, expected_value in fields.items():
            if config.get(key) != expected_value:
                raise ValueError(f"Reference loss changed: {filename}: {key}")


def validate_checkpoint(path: Path, stage: str, source_sha256: str | None = None) -> dict:
    """Check semantic metadata before using or publishing a completed stage."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("encoder_config", {})
    expected = {
        "single_frame_context": True, "clip_len": 20, "context_size": 2,
        "context_stride": 20, "embedding_dim": 128, "embedding_normalization": "l2",
        "train_base": "frozen" if stage == "hf1" else "only_bn",
        "train_embedding": stage != "hf1",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Invalid {stage} checkpoint {path}: {key} must be {value!r}")
    if checkpoint.get("embedding_normalization") != "l2":
        raise ValueError(f"Checkpoint {path} must declare L2 normalization")
    if not checkpoint.get("model_state_dict"):
        raise ValueError(f"Checkpoint {path} has no encoder weights")
    provenance = checkpoint.get("training_provenance", {})
    if provenance.get("epoch") != EPOCHS:
        raise ValueError(f"Checkpoint {path} must have epoch {EPOCHS} provenance")
    memory = config.get("memory_config", {})
    if stage == "local":
        if memory.get("enabled", False) or any(
            key.removeprefix("module.").startswith("memory.")
            for key in checkpoint["model_state_dict"]
        ):
            raise ValueError(f"Local source checkpoint {path} contains a memory encoder")
    else:
        checks = {**MEMORY_CONFIG, "cache_local_embeddings": stage == "hf1",
                  "freeze_bn_stats": stage == "hf1"}
        for key, value in checks.items():
            if memory.get(key) != value:
                raise ValueError(f"Invalid {stage} checkpoint: memory.{key} must be {value!r}")
        if source_sha256 is None or memory.get("local_checkpoint_sha256") != source_sha256:
            raise ValueError(f"{stage} checkpoint does not reference the selected local source")
    return {"checkpoint_path": str(path.resolve()), "checkpoint_sha256": sha256(path),
            "training_provenance": provenance, "encoder_config": config}


def prepare(root: Path, local_checkpoint: Path | None) -> dict:
    validate_sources()
    source = validate_checkpoint(local_checkpoint, "local") if local_checkpoint else None
    # No directory merging or overwrites: every invocation owns a fresh experiment.
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Output root must be new or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "state": "prepared", "output_root": str(root),
        "train_dataset": TRAIN_DATASET, "train_h5": str(EXPECTED_H5),
        "stage_order": list(STAGES), "epochs_per_stage": EPOCHS, "stages": {},
    }
    for stage in STAGES:
        stage_root = root / stage
        config_root = stage_root / "configs_v2"
        shutil.copytree(SOURCE_CONFIG_ROOT, config_root)
        config = _load_yaml(config_root / "train.yaml")
        config.update(deepcopy(COMMON_CONFIG))
        config.pop("data_seed", None)  # Only valid in frozen raw-cache training.
        config.update(
            train_base="frozen" if stage == "hf1" else "only_bn",
            train_embedding=stage != "hf1", pretrained=stage == "local",
            checkpoint_dir=str(stage_root / "checkpoints"),
            run_name_prefix=f"single-frame-{stage.replace('_', '-')}",
        )
        config["memory"] = {"enabled": False}
        if stage != "local":
            config["memory"] = deepcopy(MEMORY_CONFIG)
            config["memory"].update(
                local_checkpoint_path=source["checkpoint_path"] if source else None,
                cache_local_embeddings=stage == "hf1", freeze_bn_stats=stage == "hf1",
            )
        train_path = config_root / "train.yaml"
        _write_yaml(train_path, config)
        job = {
            "state": "pending", "config_root": str(config_root),
            "train_config_sha256": sha256(train_path),
            "log_path": str(stage_root / "train.log"),
            "checkpoint_root": config["checkpoint_dir"],
        }
        if stage == "local" and source:
            job.update(state="reused", **source)
        elif stage != "local":
            job.update(source_stage="local",
                       source_checkpoint=source["checkpoint_path"] if source else None,
                       source_checkpoint_sha256=source["checkpoint_sha256"] if source else None)
        manifest["stages"][stage] = job
    _save_manifest(root, manifest)
    return manifest


def _require_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The single-frame baseline training queue requires CUDA")
    # Check both stages' native dependencies before spending time on the local run.
    from mamba_ssm import Mamba2
    import causal_conv1d


def run_queue(root: Path, manifest: dict, gpu: str | None) -> int:
    _require_cuda()
    manifest["state"] = "running"
    _save_manifest(root, manifest)
    for index, stage in enumerate(STAGES, 1):
        job = manifest["stages"][stage]
        if job["state"] == "reused":
            print(f"[{index}/3] Reusing local: {job['checkpoint_path']}", flush=True)
            continue
        train_path = Path(job["config_root"]) / "train.yaml"
        if stage != "local":
            source = manifest["stages"]["local"]
            config = _load_yaml(train_path)
            config["memory"]["local_checkpoint_path"] = source["checkpoint_path"]
            _write_yaml(train_path, config)
            job.update(source_checkpoint=source["checkpoint_path"],
                       source_checkpoint_sha256=source["checkpoint_sha256"],
                       train_config_sha256=sha256(train_path))
        environment = os.environ.copy()
        for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_NAME"):
            environment.pop(key, None)
        environment.update(FINEPROG_CONFIGS_V2_DIR=job["config_root"], PYTHONUNBUFFERED="1",
                           WANDB_RUN_GROUP=root.name, WANDB_JOB_TYPE=stage)
        if gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = gpu
        job["state"] = "running"
        manifest["active_stage"] = stage
        _save_manifest(root, manifest)
        print(f"[{index}/3] {stage}: {job['log_path']}", flush=True)
        try:
            with Path(job["log_path"]).open("x", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, str(PROJECT_ROOT / "train_encoder.py")],
                    cwd=PROJECT_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                )
            job["exit_code"] = result.returncode
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, result.args)
            finals = list(Path(job["checkpoint_root"]).glob(f"*/encoder_epoch{EPOCHS:06d}.pt"))
            if len(finals) != 1:
                raise RuntimeError(f"Expected exactly one final {stage} checkpoint; found {len(finals)}")
            job.update(validate_checkpoint(finals[0], stage, job.get("source_checkpoint_sha256")))
            job["state"] = "complete"
            _save_manifest(root, manifest)
            print(f"[{index}/3] Complete: {job['checkpoint_path']}", flush=True)
        except BaseException as error:
            job.update(state="failed", error=str(error))
            manifest["state"] = "failed"
            _save_manifest(root, manifest)
            raise
    manifest.update(state="complete", active_stage=None)
    _save_manifest(root, manifest)
    print(f"All baselines complete. Checkpoint paths: {root / 'manifest.json'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES for all stages (default: inherit)")
    parser.add_argument("--output-root", type=Path, help="New/empty persistent experiment directory")
    parser.add_argument("--dry-run", action="store_true", help="Prepare snapshots without training")
    parser.add_argument("--local-checkpoint", type=Path,
                        help="Reuse a completed single-frame local epoch-20000 checkpoint")
    args = parser.parse_args(argv)
    root = (args.output_root or PROJECT_ROOT / ".vide/.codex/astra" /
            (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_pick_mango_single_frame_baselines"))
    root = root.expanduser().resolve()
    local = args.local_checkpoint.expanduser().resolve() if args.local_checkpoint else None
    try:
        manifest = prepare(root, local)
        print(f"Experiment: {root}", flush=True)
        if args.dry_run:
            print("Dry run complete; no training started. Use a fresh output root to launch.", flush=True)
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
