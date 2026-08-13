#!/usr/bin/env python3
"""Serial sweep over ``context_size`` (8 → 3, descending) for TCC encoder training.

The script snapshots ``configs_v2`` once at startup, creates one isolated config
copy per ``context_size`` value, and overrides ONLY

    train.yaml   :: context_size
    extract.yaml :: context_size   (kept in sync; only if the key exists)

Every other training hyper-parameter is taken verbatim from the repository's
current ``configs_v2/train.yaml``.  The repository YAML files are never
modified — workers read the snapshot through the ``FINEPROG_CONFIGS_V2_DIR``
environment variable honoured by ``utils/config_v2.ConfigV2``.

Runs are strictly serial (one training process at a time).  Each run logs to
its own file and reports to wandb with:
  * run name suffix ``-ctx<N>``
  * ``group`` = sweep id, so all runs are grouped in the wandb UI
  * ``config.context_size`` / ``config.sweep_group`` injected

Usage (from the fineprog conda environment)
-------------------------------------------
    conda run -n fineprog python tests/sweep_context_size.py
    conda run -n fineprog python tests/sweep_context_size.py --dry-run
    conda run -n fineprog python tests/sweep_context_size.py --context-sizes 8 7 6
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


CONTEXT_SIZES = (4, 3)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
CONFIGS_ENV_VAR = "FINEPROG_CONFIGS_V2_DIR"


# ---------------------------------------------------------------------------
# Config snapshot preparation
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def set_context_size(config_root: Path, context_size: int) -> dict:
    """Override sweep-specific train.yaml values (and keep extract in sync)."""
    train_path = config_root / "train.yaml"
    train_cfg = _load_yaml(train_path)
    if "context_size" not in train_cfg:
        raise ValueError(f"'context_size' not found in {train_path}")
    train_cfg["context_size"] = int(context_size)
    train_cfg["batch_size"] = 6
    train_cfg["learning_rate"] = 2.0e-4
    _dump_yaml(train_path, train_cfg)

    # extract.yaml must use the same encoder input shape as training.
    extract_path = config_root / "extract.yaml"
    if extract_path.is_file():
        extract_cfg = _load_yaml(extract_path)
        if "context_size" in extract_cfg:
            extract_cfg["context_size"] = int(context_size)
            _dump_yaml(extract_path, extract_cfg)

    return {
        "train_yaml": str(train_path),
        "loss_name": train_cfg.get("loss_name"),
        "train_dataset": train_cfg.get("train_dataset"),
        "clip_len": train_cfg.get("clip_len"),
        "context_stride": train_cfg.get("context_stride"),
        "num_epochs": train_cfg.get("num_epochs"),
        "batch_size": train_cfg.get("batch_size"),
        "learning_rate": train_cfg.get("learning_rate"),
    }


def prepare_snapshots(sweep_root: Path, context_sizes: tuple[int, ...]) -> list[dict]:
    """Snapshot configs_v2 once, then derive one isolated config per value."""
    source_snapshot = sweep_root / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    jobs = []
    for context_size in context_sizes:
        config_root = sweep_root / f"configs_ctx{context_size}"
        shutil.copytree(source_snapshot, config_root)
        info = set_context_size(config_root, context_size)
        jobs.append(
            {
                "context_size": int(context_size),
                "config_root": config_root,
                "train_info": info,
                "status": "pending",
            }
        )
    return jobs


# ---------------------------------------------------------------------------
# Worker: run one training with an isolated ConfigV2 directory
# ---------------------------------------------------------------------------

def run_training_worker(config_root: Path, context_size: int, sweep_group: str) -> None:
    """Run train_encoder.train() against the supplied isolated config snapshot."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import wandb

    import train_encoder
    from utils.config_v2 import ConfigV2

    config_v2 = ConfigV2(config_root)
    train_config = config_v2.load_train()
    loss_config_rel = train_config.get("loss_config", "loss/loss_tcc.yaml")

    # train_encoder resolves these module globals on import. Point them at the
    # isolated snapshot before invoking the same train() call as its CLI.
    train_encoder._CFG_V2 = config_v2
    train_encoder._TRAIN_V2 = train_config
    train_encoder._V2_TRAIN_YAML = str(config_root / "train.yaml")
    train_encoder._loss_name_v2 = train_config.get("loss_name", "tcc")
    train_encoder._loss_cfg_file = loss_config_rel
    train_encoder._V2_LOSS_YAML = str(config_root / loss_config_rel)

    # Make the swept value visible in wandb (unique run name + group + config).
    suffix = f"ctx{context_size}"
    _original_init = wandb.init

    def _init_with_sweep_metadata(*args, **kwargs):
        base_name = kwargs.get("name")
        if base_name:
            kwargs["name"] = f"{base_name}-{suffix}"
        kwargs.setdefault("group", sweep_group)
        kwargs.setdefault("job_type", "context_size_sweep")
        run_config = dict(kwargs.get("config") or {})
        run_config["context_size"] = int(context_size)
        run_config["sweep_group"] = sweep_group
        kwargs["config"] = run_config
        return _original_init(*args, **kwargs)

    wandb.init = _init_with_sweep_metadata
    try:
        train_encoder.train(
            num_epochs=train_config.get("num_epochs", 5),
            batch_size=train_config.get("batch_size", 2),
            learning_rate=train_config.get("learning_rate", 1e-4),
            log_every=train_config.get("log_every", 10),
            num_workers=train_config.get("num_workers", 0),
            checkpoint_every=train_config.get("checkpoint_every", 1000),
            checkpoint_dir=train_config.get("checkpoint_dir", "checkpoints/encoder"),
            h5_path=None,
            register=False,
            register_alias=None,
        )
    finally:
        wandb.init = _original_init


# ---------------------------------------------------------------------------
# Launcher (strictly serial)
# ---------------------------------------------------------------------------

def _worker_env(config_root: Path, sweep_group: str, gpu: str | None) -> dict:
    env = os.environ.copy()
    env[CONFIGS_ENV_VAR] = str(config_root)
    env["WANDB_RUN_GROUP"] = sweep_group
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def run_one(job: dict, sweep_group: str, log_dir: Path, gpu: str | None) -> int:
    log_path = log_dir / f"ctx{job['context_size']}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config-root", str(job["config_root"]),
        "--worker-context-size", str(job["context_size"]),
        "--worker-sweep-group", sweep_group,
    ]
    print(f"[sweep] starting context_size={job['context_size']} → {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_worker_env(job["config_root"], sweep_group, gpu),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Serial sweep of context_size (8 down to 3) using the current "
            "configs_v2 TCC encoder training setup."
        )
    )
    parser.add_argument(
        "--context-sizes",
        type=int,
        nargs="+",
        default=list(CONTEXT_SIZES),
        help="context_size values to sweep, in run order (default: 8 7 6 5 4 3).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES of every worker (default: inherit).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-run logs (default: outputs/sweeps/<sweep_group>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate all config overrides without starting training.",
    )
    parser.add_argument("--worker-config-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-context-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sweep-group", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_config_root is not None:
        run_training_worker(
            args.worker_config_root.resolve(),
            int(args.worker_context_size),
            str(args.worker_sweep_group),
        )
        return 0

    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing config directory: {SOURCE_CONFIG_ROOT}")

    context_sizes = tuple(int(v) for v in args.context_sizes)
    if any(v < 1 for v in context_sizes):
        raise ValueError("context_size values must be >= 1")

    sweep_group = "context-size-sweep-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir or (PROJECT_ROOT / "outputs" / "sweeps" / sweep_group)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = prepare_snapshots(log_dir / "configs", context_sizes)
    print(f"[sweep] group   : {sweep_group}")
    print(f"[sweep] log dir : {log_dir}")
    for index, job in enumerate(jobs, start=1):
        info = job["train_info"]
        print(
            f"[sweep] prepared {index}/{len(jobs)}: context_size={job['context_size']} "
            f"(loss={info['loss_name']}, dataset={info['train_dataset']}, "
            f"clip_len={info['clip_len']}, context_stride={info['context_stride']}, "
            f"epochs={info['num_epochs']}, batch_size={info['batch_size']}, "
            f"learning_rate={info['learning_rate']}) "
            f"→ {info['train_yaml']}",
            flush=True,
        )

    if args.dry_run:
        print("[sweep] dry run complete; no training started.")
        return 0

    failures = 0
    for index, job in enumerate(jobs, start=1):
        print(f"\n[sweep] run {index}/{len(jobs)}", flush=True)
        code = run_one(job, sweep_group, log_dir, args.gpu)
        if code == 0:
            job["status"] = "ok"
            print(f"[sweep] DONE context_size={job['context_size']}", flush=True)
        else:
            job["status"] = f"failed(exit={code})"
            failures += 1
            log_path = log_dir / f"ctx{job['context_size']}.log"
            print(
                f"[sweep] FAILED context_size={job['context_size']} exit={code} "
                f"(log: {log_path})",
                flush=True,
            )

    print("\n[sweep] ── summary ──")
    for job in jobs:
        print(f"[sweep]   context_size={job['context_size']:<2d} {job['status']}")
    print(f"[sweep] wandb group: {sweep_group}")

    if failures:
        print(f"[sweep] {failures} run(s) failed; inspect logs in {log_dir}")
        return 1

    print("[sweep] all training runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
