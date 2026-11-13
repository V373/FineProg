#!/usr/bin/env python3
"""Run frozen Mamba2 alpha-init (1e-3, 1e-2, 0.1) and no-residual ablations serially.

conda run --no-capture-output -n fineprog python scripts/train_pick_mango_mamba2_alpha_ablation.py
Add --dry-run to prepare config snapshots without starting training.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml

from train_pick_mango_mamba2_serial import PROJECT_ROOT, make_config

# Without the residual, alpha is fixed at 1 and the output projection still trains.
ABLATIONS = (
    ("frozen_alpha_1e-3", 1e-3, True),
    ("frozen_alpha_1e-2", 1e-2, True),
    ("frozen_alpha_0.1", 0.1, True),
    ("frozen_no_residual", 1.0, False),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Mamba2 training requires CUDA.")

    group = "pick_mango_mamba2_alpha_ablation_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    jobs = []
    for job, alpha_init, use_residual in ABLATIONS:
        config_root = PROJECT_ROOT / "outputs" / group / job / "configs_v2"
        config = make_config("frozen", config_root)
        config["memory"].update(alpha_init=alpha_init, use_residual=use_residual)
        config["checkpoint_dir"] = f"checkpoint/pick_mango_mamba2_alpha_ablation/{job}"
        train_path = config_root / "train.yaml"
        train_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        env = os.environ.copy()
        env.update(FINEPROG_CONFIGS_V2_DIR=str(config_root), PYTHONUNBUFFERED="1",
                   WANDB_RUN_GROUP=group, WANDB_JOB_TYPE=job)
        jobs.append((job, env))
        print(f"{job}: alpha_init={alpha_init:g}, use_residual={use_residual}, "
              f"train_base={config['train_base']}, "
              f"train_embedding={config['train_embedding']}\n  config: {train_path}", flush=True)

    for index, (job, env) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {job}", flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, str(PROJECT_ROOT / "train_encoder.py")],
                           cwd=PROJECT_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
