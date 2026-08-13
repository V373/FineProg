#!/usr/bin/env python3
"""Serially sweep the TCC encoder ``embedding_dim`` over 64 / 32 / 16.

The script snapshots ``configs_v2`` once at startup and derives one isolated
config copy per embedding dimension.  Only two keys are overridden:

  * ``configs_v2/train.yaml -> embedding_dim``
  * ``configs_v2/loss/loss_tcc.yaml -> softmax_temperature`` (optional)

The temperature rescaling follows the relationship documented in
``configs_v2/train.yaml`` (``embedding_dim * softmax_temperature`` is kept
constant, i.e. 128→0.001, 64→0.002, 32→0.004, 16→0.008).  The constant is taken
from the *current* repository config, so changing ``train.yaml`` /
``loss_tcc.yaml`` automatically moves the whole sweep.  Disable it with
``--no-scale-softmax-temperature`` to hold the temperature fixed.

Every other training hyper-parameter is taken verbatim from the repository's
current ``configs_v2``.  The repository YAML files are never modified.

Runs are executed strictly one after another (a single training of the current
config already reserves ~23 GB on the RTX 5090).  Each run is a separate
process, logs to its own file, and reports to wandb with:
  * run name suffix ``-embdim<value>``
  * ``group`` = sweep id, so all runs are grouped in the wandb UI
  * ``config.embedding_dim`` / ``config.softmax_temperature`` /
    ``config.sweep_group`` injected

Any run that fails is retried once at the end, so every dimension is guaranteed
to be attempted to completion.

Usage (from the fineprog conda environment)
-------------------------------------------
    conda run -n fineprog python tests/sweep_embedding_dim.py
    conda run -n fineprog python tests/sweep_embedding_dim.py --dry-run
    conda run -n fineprog python tests/sweep_embedding_dim.py --dims 64 32
    conda run -n fineprog python tests/sweep_embedding_dim.py --no-scale-softmax-temperature
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


EMBEDDING_DIMS = (64, 32, 16)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"


# ---------------------------------------------------------------------------
# Config snapshot preparation
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def find_tcc_loss_yaml(config_root: Path) -> Path | None:
    """Locate the loss YAML holding ``softmax_temperature`` inside a snapshot.

    Prefers the file referenced by ``train.yaml -> loss_config`` (the plain TCC
    setup), then falls back to ``loss/loss_tcc.yaml`` (composite setups).
    """
    train_config = _read_yaml(config_root / "train.yaml")
    candidates: list[Path] = []
    loss_config_rel = train_config.get("loss_config")
    if loss_config_rel:
        candidates.append(config_root / str(loss_config_rel))
    candidates.append(config_root / "loss" / "loss_tcc.yaml")

    for candidate in candidates:
        if candidate.is_file() and "softmax_temperature" in _read_yaml(candidate):
            return candidate
    return None


def read_source_reference() -> tuple[int, float | None]:
    """Return (embedding_dim, softmax_temperature) of the repository config."""
    train_config = _read_yaml(SOURCE_CONFIG_ROOT / "train.yaml")
    if "embedding_dim" not in train_config:
        raise ValueError(
            f"'embedding_dim' not found in {SOURCE_CONFIG_ROOT / 'train.yaml'}"
        )
    base_dim = int(train_config["embedding_dim"])

    loss_path = find_tcc_loss_yaml(SOURCE_CONFIG_ROOT)
    base_temperature = (
        float(_read_yaml(loss_path)["softmax_temperature"]) if loss_path else None
    )
    return base_dim, base_temperature


def set_embedding_dim(config_root: Path, embedding_dim: int) -> Path:
    """Override embedding_dim in the snapshot's train.yaml."""
    train_path = config_root / "train.yaml"
    train_config = _read_yaml(train_path)
    if "embedding_dim" not in train_config:
        raise ValueError(f"'embedding_dim' not found in {train_path}")

    train_config["embedding_dim"] = int(embedding_dim)
    _write_yaml(train_path, train_config)
    return train_path


def set_softmax_temperature(config_root: Path, temperature: float) -> Path:
    """Override softmax_temperature in the snapshot's TCC loss YAML."""
    loss_path = find_tcc_loss_yaml(config_root)
    if loss_path is None:
        raise ValueError(
            f"No loss YAML with 'softmax_temperature' found under {config_root}"
        )

    loss_config = _read_yaml(loss_path)
    loss_config["softmax_temperature"] = float(temperature)
    _write_yaml(loss_path, loss_config)
    return loss_path


def prepare_snapshots(
    sweep_root: Path,
    embedding_dims: tuple[int, ...],
    scale_temperature: bool,
) -> list[dict]:
    """Snapshot configs_v2 once, then derive one isolated config per dimension."""
    base_dim, base_temperature = read_source_reference()

    source_snapshot = sweep_root / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    prepared: list[dict] = []
    for embedding_dim in embedding_dims:
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")

        config_root = sweep_root / f"configs_embdim_{embedding_dim}"
        shutil.copytree(source_snapshot, config_root)
        set_embedding_dim(config_root, embedding_dim)

        temperature: float | None = base_temperature
        loss_path: Path | None = None
        if scale_temperature:
            if base_temperature is None:
                raise ValueError(
                    "Cannot rescale softmax_temperature: no TCC loss YAML with "
                    f"'softmax_temperature' found under {SOURCE_CONFIG_ROOT}. "
                    "Re-run with --no-scale-softmax-temperature."
                )
            # Keep embedding_dim * softmax_temperature constant.
            temperature = base_temperature * base_dim / float(embedding_dim)
            loss_path = set_softmax_temperature(config_root, temperature)

        prepared.append(
            {
                "embedding_dim": int(embedding_dim),
                "softmax_temperature": temperature,
                "config_root": config_root,
                "loss_path": loss_path,
            }
        )
    return prepared


# ---------------------------------------------------------------------------
# Worker: run one training with an isolated ConfigV2 directory
# ---------------------------------------------------------------------------

def run_training_worker(
    config_root: Path,
    embedding_dim: int,
    sweep_group: str,
) -> None:
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

    loss_path = find_tcc_loss_yaml(config_root)
    softmax_temperature = (
        _read_yaml(loss_path).get("softmax_temperature") if loss_path else None
    )

    # Make the swept value visible in wandb (unique run name + group + config).
    suffix = f"embdim{embedding_dim}"
    _original_init = wandb.init

    def _init_with_sweep_metadata(*args, **kwargs):
        base_name = kwargs.get("name")
        if base_name:
            kwargs["name"] = f"{base_name}-{suffix}"
        kwargs.setdefault("group", sweep_group)
        kwargs.setdefault("job_type", "embedding_dim_sweep")
        run_config = dict(kwargs.get("config") or {})
        run_config["embedding_dim"] = int(embedding_dim)
        if softmax_temperature is not None:
            run_config["softmax_temperature"] = float(softmax_temperature)
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

def _worker_command(job: dict, sweep_group: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config-root", str(job["config_root"]),
        "--worker-embedding-dim", str(job["embedding_dim"]),
        "--worker-sweep-group", sweep_group,
    ]


def _worker_env(job: dict, sweep_group: str, gpu: str | None) -> dict:
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = sweep_group
    env["PYTHONUNBUFFERED"] = "1"
    # Entrypoints that build their own ConfigV2 (e.g. in-training eval helpers)
    # must see the isolated snapshot as well.
    env["FINEPROG_CONFIGS_V2_DIR"] = str(job["config_root"])
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def run_jobs(
    jobs: list[dict],
    sweep_group: str,
    log_dir: Path,
    gpu: str | None,
    attempt: int,
) -> list[dict]:
    """Run every job serially. Returns the list of failed jobs."""
    failed: list[dict] = []

    for index, job in enumerate(jobs, start=1):
        log_path = log_dir / f"embdim_{job['embedding_dim']}_attempt{attempt}.log"
        print(
            f"[sweep] ({index}/{len(jobs)}) starting embedding_dim="
            f"{job['embedding_dim']} → {log_path}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            code = subprocess.call(
                _worker_command(job, sweep_group),
                cwd=PROJECT_ROOT,
                env=_worker_env(job, sweep_group, gpu),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        if code == 0:
            job["status"] = "ok"
            print(
                f"[sweep] DONE embedding_dim={job['embedding_dim']} (log: {log_path})",
                flush=True,
            )
        else:
            job["status"] = f"failed(exit={code})"
            failed.append(job)
            print(
                f"[sweep] FAILED embedding_dim={job['embedding_dim']} "
                f"exit={code} (log: {log_path})",
                flush=True,
            )

    return failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Serially sweep the TCC encoder embedding_dim over 64 / 32 / 16 "
            "using the current configs_v2 training setup."
        )
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=list(EMBEDDING_DIMS),
        help="Embedding dimensions to sweep (default: 64 32 16).",
    )
    parser.add_argument(
        "--no-scale-softmax-temperature",
        dest="scale_softmax_temperature",
        action="store_false",
        help=(
            "Keep loss_tcc.yaml -> softmax_temperature at its current value. "
            "By default it is rescaled so embedding_dim * softmax_temperature "
            "stays constant (128→0.001, 64→0.002, 32→0.004, 16→0.008)."
        ),
    )
    parser.set_defaults(scale_softmax_temperature=True)
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
    parser.add_argument("--worker-embedding-dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sweep-group", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_config_root is not None:
        run_training_worker(
            args.worker_config_root.resolve(),
            int(args.worker_embedding_dim),
            str(args.worker_sweep_group),
        )
        return 0

    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing config directory: {SOURCE_CONFIG_ROOT}")
    if not args.dims:
        raise ValueError("--dims must list at least one embedding dimension")

    sweep_group = "tcc-embdim-sweep-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir or (PROJECT_ROOT / "outputs" / "sweeps" / sweep_group)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = prepare_snapshots(
        log_dir / "configs",
        tuple(args.dims),
        args.scale_softmax_temperature,
    )
    print(f"[sweep] group          : {sweep_group}")
    print(f"[sweep] log dir        : {log_dir}")
    print(f"[sweep] scheduling     : serial (1 run at a time)")
    print(f"[sweep] scale softmaxT : {args.scale_softmax_temperature}")
    for index, job in enumerate(jobs, start=1):
        temperature = job["softmax_temperature"]
        temperature_text = "unchanged" if temperature is None else f"{temperature:g}"
        print(
            f"[sweep] prepared {index}/{len(jobs)}: "
            f"embedding_dim={job['embedding_dim']:<4d} "
            f"softmax_temperature={temperature_text} → {job['config_root']}",
            flush=True,
        )

    if args.dry_run:
        print("[sweep] dry run complete; no training started.")
        return 0

    failed = run_jobs(jobs, sweep_group, log_dir, args.gpu, attempt=1)

    if failed:
        print(
            f"\n[sweep] retrying {len(failed)} failed run(s) serially (attempt 2) ...",
            flush=True,
        )
        failed = run_jobs(failed, sweep_group, log_dir, args.gpu, attempt=2)

    print("\n[sweep] ── summary ──")
    for job in jobs:
        print(
            f"[sweep]   embedding_dim={job['embedding_dim']:<4d} "
            f"{job.get('status', 'unknown')}"
        )
    print(f"[sweep] wandb group: {sweep_group}")

    if failed:
        print(f"[sweep] {len(failed)} run(s) still failing; inspect logs in {log_dir}")
        return 1

    print("[sweep] all training runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
