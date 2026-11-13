#!/usr/bin/env python3
"""Run nine frozen, no-residual history ablations with normalized Mamba2 inputs.

conda run --no-capture-output -n fineprog python scripts/train_pick_mango_mamba2_serial.py
Add --dry-run to prepare persistent config snapshots without starting training.
"""

import argparse
from copy import deepcopy
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
LOCAL_CHECKPOINT = PROJECT_ROOT / (
    "checkpoint/COMPOSITE_TCC_TEMPORAL_TRIPLET-pick_mango_vid-36vid_train-"
    "resnet50_conv4c-only_bn-20260925-090936-ctxstride20/encoder_epoch020000.pt"
)

# The legacy local checkpoint has no encoder_config metadata. These are its
# original architecture settings, independent of this run's ablation flags.
LOCAL_ENCODER_CONFIG = {"context_size": 2, "context_stride": 20, "embedding_dim": 128}

# Shared recipe. Run-specific YAMLs are generated only under outputs/.
BASE_CONFIG = {
    "train_dataset": "pick_mango_vid_36vid_train",
    "backbone_name": "resnet50_conv4c",
    "pretrained": True,
    "embedding_dim": 128,
    "embedding_normalization": "l2",
    "seed": 42,
    "num_epochs": 20000,
    "batch_size": 12,
    "num_workers": 4,
    "limit_cpu_threads": False,
    "transport_frames_as_uint8": True,
    "clip_len": 20,
    "context_size": 2,
    "context_stride": 20,
    "sampling_strategy": "offset_uniform",
    "random_offset": 1,
    "learning_rate": 0.0004,
    "log_every": 100,
    "checkpoint_every": 2000,
    "extract_backbone_cache": False,
    "cache_refresh_every": 500,
    "loss_name": "composite",
    "loss_config": "loss/loss_composite_tcc_triplet.yaml",
    "in_training_eval": {
        "enabled": True,
        "selected_eval_tasks_in_training": [
            "latent_distance_heatmap", "kendalls_tau", "activation_map",
        ],
        "eval_freq_in_training": 1,
        "eval_dataset_ref": None,
        "log_images_to_wandb": True,
        "latent_distance_heatmap": {
            "selected_video_index": 0,
            "normalize_embeddings": False,
            "convert_to_similarity": True,
            "similarity_tau": 1.0,
        },
        "kendalls_tau": {"kendall_stride": 5, "kendall_distance": "sqeuclidean"},
        "activation_map": {
            "max_videos": 1,
            "save_mp4": True,
            "save_png": False,
            "save_h5": False,
            "sample_all_stride": 1,
            "overlay_alpha": 0.45,
            "colormap": "jet",
            "resize_to": 224,
        },
    },
    "memory": {
        "enabled": True,
        "local_checkpoint_path": str(LOCAL_CHECKPOINT),
        "local_config_path": None,  # Filled with the run's legacy architecture snapshot.
        "d_model": 128,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        "headdim": 64,
        "ngroups": 1,
        "chunk_size": 256,
        "alpha_init": 0.001,
        "use_residual": False,
        "history_frames": 1,
        "history_stride": 1,
        "cache_local_embeddings": True,
        "visual_chunk_size": 20,
        "checkpoint_visual": True,
        "freeze_bn_stats": True,
    },
}

# Edit ablation settings here; only_bn trains BN affine parameters/statistics
# alongside the temporal embedder and Mamba, while keeping convolution weights fixed.
MODE_OVERRIDES = {
    "frozen": {"train_base": "frozen", "train_embedding": False},
    "warm_only_bn": {
        "run_name_prefix": "normfix-v1-warm_only_bn",
        "train_base": "only_bn",
        "train_embedding": True,
        "num_epochs": 10000,
        "log_every": 50,
        "checkpoint_every": 500,
        "memory": {"cache_local_embeddings": False, "freeze_bn_stats": False,
                   "normalize_input": True},
    },
    "imagenet_only_bn": {
        "train_base": "only_bn",
        "train_embedding": True,
        "memory": {
            "local_checkpoint_path": None,
            "cache_local_embeddings": False,
            "freeze_bn_stats": False,
        },
    },
}

# The default queue contains only the nine frozen history retraining jobs.
MODES = []
for history_frames in (2, 4, 8):
    for history_stride in (5, 10, 20):
        mode = f"frozen_no_residual_hf{history_frames}_hs{history_stride}"
        MODE_OVERRIDES[mode] = {
            **MODE_OVERRIDES["frozen"],
            "run_name_prefix": f"normfix-v1-frozen-nores-hf{history_frames}-hs{history_stride}",
            "memory": {
                "use_residual": False,
                "normalize_input": True,
                "alpha_init": 1.0,
                "history_frames": history_frames,
                "history_stride": history_stride,
            },
        }
        MODES.append(mode)


def make_config(mode, config_root):
    config = deepcopy(BASE_CONFIG)
    overrides = deepcopy(MODE_OVERRIDES[mode])
    config["memory"].update(overrides.pop("memory", {}))
    config.update(overrides)
    memory = config["memory"]
    checkpoint_group = ("pick_mango_mamba2_normfix_v1" if memory.get("normalize_input", False)
                        else "pick_mango_mamba2")
    config["checkpoint_dir"] = f"checkpoint/{checkpoint_group}/{mode}"
    if memory["local_checkpoint_path"]:
        if not Path(memory["local_checkpoint_path"]).is_file():
            raise FileNotFoundError(memory["local_checkpoint_path"])
        memory["local_config_path"] = str(config_root / "local_encoder.yaml")

    shutil.copytree(SOURCE_CONFIG_ROOT, config_root)
    if memory["local_config_path"]:
        Path(memory["local_config_path"]).write_text(
            yaml.safe_dump(LOCAL_ENCODER_CONFIG, sort_keys=False), encoding="utf-8",
        )
    (config_root / "train.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Mamba2 training requires CUDA.")

    group = "pick_mango_mamba2_normfix_v1_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    snapshot_root = PROJECT_ROOT / "outputs" / group
    jobs = []
    for mode in MODES:
        config_root = snapshot_root / mode / "configs_v2"
        config = make_config(mode, config_root)
        train_path = config_root / "train.yaml"
        memory = config["memory"]

        env = os.environ.copy()
        env.update(FINEPROG_CONFIGS_V2_DIR=str(config_root), PYTHONUNBUFFERED="1",
                   WANDB_RUN_GROUP=group, WANDB_JOB_TYPE=mode)
        jobs.append((mode, env))
        print(f"{mode}: train_base={config['train_base']}, "
              f"train_embedding={config['train_embedding']}, "
              f"load_local={bool(memory['local_checkpoint_path'])}, "
              f"freeze_bn_stats={memory['freeze_bn_stats']}, "
              f"use_residual={memory['use_residual']}, "
              f"normalize_input={memory.get('normalize_input', False)}, "
              f"history_frames={memory['history_frames']}, "
              f"history_stride={memory['history_stride']}, "
              f"epochs={config['num_epochs']}, log_every={config['log_every']}, "
              f"checkpoint_every={config['checkpoint_every']}, "
              f"run_name_prefix={config['run_name_prefix']}\n  config: {train_path}", flush=True)

    for index, (mode, env) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {mode}", flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, str(PROJECT_ROOT / "train_encoder.py")],
                           cwd=PROJECT_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
