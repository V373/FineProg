#!/usr/bin/env python3
"""Run frozen dual Mamba2 hf4, long-hs20/short-hs5 branch-L2 ablations serially.

conda run --no-capture-output -n fineprog python scripts/train_pick_mango_mamba2_dual_l2_ablation.py
Add --dry-run to prepare config snapshots without starting training.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml

from train_pick_mango_mamba2_serial import PROJECT_ROOT, make_config

ABLATIONS = (
    ("dual-frozen-hf4-longhs20-shorths5-bl2-on", True),
    ("dual-frozen-hf4-longhs20-shorths5-bl2-off", False),
)


def make_dual_config(job, branch_output_l2, config_root):
    config = make_config("frozen", config_root)
    config["memory"].update(
        dual_branch=True,
        long_history_frames=4,
        short_history_frames=4,
        history_stride=20,
        long_history_stride=20,
        short_history_stride=5,
        normalize_input=True,
        alpha_init=1.0,
        branch_output_l2=branch_output_l2,
    )
    config["run_name_prefix"] = job
    config["checkpoint_dir"] = f"checkpoint/pick_mango_mamba2_dual_l2/{job}"
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

    group = "pick_mango_mamba2_dual_l2_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    jobs = []
    for job, branch_output_l2 in ABLATIONS:
        config_root = PROJECT_ROOT / "outputs" / group / job / "configs_v2"
        config = make_dual_config(job, branch_output_l2, config_root)
        env = os.environ.copy()
        env.update(FINEPROG_CONFIGS_V2_DIR=str(config_root), PYTHONUNBUFFERED="1",
                   WANDB_RUN_GROUP=group, WANDB_JOB_TYPE=job)
        jobs.append((job, env))
        memory = config["memory"]
        print(f"{job}: long_hf={memory['long_history_frames']}, "
              f"long_hs={memory['long_history_stride']}, "
              f"short_hf={memory['short_history_frames']}, "
              f"short_hs={memory['short_history_stride']}, "
              f"branch_output_l2={branch_output_l2}\n"
              f"  config: {config_root / 'train.yaml'}", flush=True)

    for index, (job, env) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {job}", flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, str(PROJECT_ROOT / "train_encoder.py")],
                           cwd=PROJECT_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
