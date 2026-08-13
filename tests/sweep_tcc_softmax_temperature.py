#!/usr/bin/env python3
"""Sweep ``softmax_temperature`` of the TCC loss on the current can-task config.

The script snapshots ``configs_v2`` once at startup, creates one isolated config
copy per temperature, and overrides ONLY
``configs_v2/loss/loss_tcc.yaml -> softmax_temperature``.
Every other training hyper-parameter is taken verbatim from the repository's
current ``configs_v2/train.yaml`` (dataset ``robomimic_can_ph_36vid_train``).
The repository YAML files are never modified.

Each run is a separate process, logs to its own file, and reports to wandb with:
  * run name suffix ``-softmaxT<value>``
  * ``group`` = sweep id, so the five runs are grouped in the wandb UI
  * ``config.softmax_temperature`` / ``config.sweep_group`` injected

Scheduling
----------
``--max-parallel N`` controls how many trainings run concurrently on the GPU.
Default is 1 (serial): measured on the RTX 5090 (32 GB), one run of the current
config (batch_size 12) reserves ~23 GB, so two concurrent runs OOM.  Raise it
only if you also shrink the per-run memory footprint.  Any run that fails is
retried once, serially, at the end so all five trainings are guaranteed to be
attempted to completion.

Usage (from the fineprog conda environment)
-------------------------------------------
    conda run -n fineprog python tests/sweep_tcc_softmax_temperature.py
    conda run -n fineprog python tests/sweep_tcc_softmax_temperature.py --dry-run
    conda run -n fineprog python tests/sweep_tcc_softmax_temperature.py --max-parallel 1
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


# Previous 5-run serial sweep (kept for reference):
# SOFTMAX_TEMPERATURES = (0.001, 0.0015, 0.002, 0.0025, 0.003)
SOFTMAX_TEMPERATURES = (0.0005,)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"

# Seconds to wait between launching two concurrent runs, so their wandb run
# names (second-resolution timestamp) and dataset loads do not collide.
LAUNCH_STAGGER_SEC = 20.0


def _label(temperature: float) -> str:
    """Filesystem/wandb friendly label, e.g. 0.0015 -> '0p0015'."""
    return f"{temperature:g}".replace(".", "p")


# ---------------------------------------------------------------------------
# Config snapshot preparation
# ---------------------------------------------------------------------------

def set_softmax_temperature(config_root: Path, temperature: float) -> Path:
    """Override softmax_temperature in the loss YAML referenced by train.yaml."""
    train_path = config_root / "train.yaml"
    train_config = yaml.safe_load(train_path.read_text(encoding="utf-8")) or {}

    if train_config.get("loss_name") != "tcc":
        raise ValueError(
            f"Expected loss_name='tcc' in {train_path}, "
            f"got {train_config.get('loss_name')!r}"
        )

    loss_config_rel = train_config.get("loss_config")
    if not loss_config_rel:
        raise ValueError(f"Missing loss_config in {train_path}")

    loss_path = config_root / str(loss_config_rel)
    loss_config = yaml.safe_load(loss_path.read_text(encoding="utf-8")) or {}
    if "softmax_temperature" not in loss_config:
        raise ValueError(f"'softmax_temperature' not found in {loss_path}")

    loss_config["softmax_temperature"] = float(temperature)
    loss_path.write_text(
        yaml.safe_dump(loss_config, sort_keys=False),
        encoding="utf-8",
    )
    return loss_path


def prepare_snapshots(sweep_root: Path) -> list[dict]:
    """Snapshot configs_v2 once, then derive one isolated config per temperature."""
    source_snapshot = sweep_root / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    prepared = []
    for temperature in SOFTMAX_TEMPERATURES:
        label = _label(temperature)
        config_root = sweep_root / f"configs_softmaxT_{label}"
        shutil.copytree(source_snapshot, config_root)
        loss_path = set_softmax_temperature(config_root, temperature)
        prepared.append(
            {
                "temperature": float(temperature),
                "label": label,
                "config_root": config_root,
                "loss_path": loss_path,
            }
        )
    return prepared


# ---------------------------------------------------------------------------
# Worker: run one training with an isolated ConfigV2 directory
# ---------------------------------------------------------------------------

def run_training_worker(config_root: Path, temperature: float, sweep_group: str) -> None:
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
    suffix = f"softmaxT{temperature:g}"
    _original_init = wandb.init

    def _init_with_sweep_metadata(*args, **kwargs):
        base_name = kwargs.get("name")
        if base_name:
            kwargs["name"] = f"{base_name}-{suffix}"
        kwargs.setdefault("group", sweep_group)
        kwargs.setdefault("job_type", "softmax_temperature_sweep")
        run_config = dict(kwargs.get("config") or {})
        run_config["softmax_temperature"] = float(temperature)
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
# Launcher
# ---------------------------------------------------------------------------

def _worker_command(job: dict, sweep_group: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config-root", str(job["config_root"]),
        "--worker-temperature", repr(job["temperature"]),
        "--worker-sweep-group", sweep_group,
    ]


def _worker_env(sweep_group: str, gpu: str | None) -> dict:
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = sweep_group
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def _launch(job: dict, sweep_group: str, log_dir: Path, gpu: str | None, attempt: int):
    log_path = log_dir / f"softmaxT_{job['label']}_attempt{attempt}.log"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        _worker_command(job, sweep_group),
        cwd=PROJECT_ROOT,
        env=_worker_env(sweep_group, gpu),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(
        f"[sweep] launched softmax_temperature={job['temperature']:g} "
        f"(pid={process.pid}) → {log_path}",
        flush=True,
    )
    return process, log_file, log_path


def run_jobs(
    jobs: list[dict],
    sweep_group: str,
    log_dir: Path,
    max_parallel: int,
    gpu: str | None,
    attempt: int,
) -> list[dict]:
    """Run jobs with a concurrency limit. Returns the list of failed jobs."""
    pending = list(jobs)
    running: list[tuple[dict, subprocess.Popen, object, Path]] = []
    failed: list[dict] = []

    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            process, log_file, log_path = _launch(job, sweep_group, log_dir, gpu, attempt)
            running.append((job, process, log_file, log_path))
            if pending and max_parallel > 1:
                time.sleep(LAUNCH_STAGGER_SEC)

        time.sleep(5.0)

        still_running = []
        for job, process, log_file, log_path in running:
            code = process.poll()
            if code is None:
                still_running.append((job, process, log_file, log_path))
                continue
            log_file.close()
            if code == 0:
                job["status"] = "ok"
                print(
                    f"[sweep] DONE softmax_temperature={job['temperature']:g} "
                    f"(log: {log_path})",
                    flush=True,
                )
            else:
                job["status"] = f"failed(exit={code})"
                failed.append(job)
                print(
                    f"[sweep] FAILED softmax_temperature={job['temperature']:g} "
                    f"exit={code} (log: {log_path})",
                    flush=True,
                )
        running = still_running

    return failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep TCC loss softmax_temperature over "
            "0.001 / 0.0015 / 0.002 / 0.0025 / 0.003 using the current "
            "configs_v2 can-task training setup."
        )
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help=(
            "Number of trainings to run concurrently (default: 1 = serial). "
            "One run of the current config reserves ~23 GB, so >1 OOMs on a "
            "32 GB card unless the per-run memory footprint is reduced."
        ),
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
    parser.add_argument("--worker-temperature", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sweep-group", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_config_root is not None:
        run_training_worker(
            args.worker_config_root.resolve(),
            float(args.worker_temperature),
            str(args.worker_sweep_group),
        )
        return 0

    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing config directory: {SOURCE_CONFIG_ROOT}")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    sweep_group = "tcc-softmaxT-sweep-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir or (PROJECT_ROOT / "outputs" / "sweeps" / sweep_group)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = prepare_snapshots(log_dir / "configs")
    print(f"[sweep] group        : {sweep_group}")
    print(f"[sweep] log dir      : {log_dir}")
    print(f"[sweep] max parallel : {args.max_parallel}")
    for index, job in enumerate(jobs, start=1):
        print(
            f"[sweep] prepared {index}/{len(jobs)}: "
            f"softmax_temperature={job['temperature']:g} → {job['loss_path']}",
            flush=True,
        )

    if args.dry_run:
        print("[sweep] dry run complete; no training started.")
        return 0

    failed = run_jobs(jobs, sweep_group, log_dir, args.max_parallel, args.gpu, attempt=1)

    if failed:
        print(
            f"\n[sweep] retrying {len(failed)} failed run(s) serially "
            f"(attempt 2) ...",
            flush=True,
        )
        failed = run_jobs(failed, sweep_group, log_dir, 1, args.gpu, attempt=2)

    print("\n[sweep] ── summary ──")
    for job in jobs:
        print(f"[sweep]   softmax_temperature={job['temperature']:<7g} {job.get('status', 'unknown')}")
    print(f"[sweep] wandb group: {sweep_group}")

    if failed:
        print(f"[sweep] {len(failed)} run(s) still failing; inspect logs in {log_dir}")
        return 1

    print("[sweep] all training runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
