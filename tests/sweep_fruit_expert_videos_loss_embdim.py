#!/usr/bin/env python3
"""Serial loss / embedding-dimension sweep for fruit expert videos.

The following four variants are run strictly one after another on
``fruit_expert_videos_36vid_train``:

    #  loss                                  embedding_dim
    1  tcc                                   128
    2  tcc + 0.5 * temporal triplet          128
    3  tcc                                    32
    4  tcc + 0.5 * temporal triplet           32

Config isolation
----------------
``configs_v2`` is snapshotted once at startup and one isolated copy is derived
for each variant.  The repository's YAML files are never modified.  Apart from
the dataset, loss selection, embedding dimension, composite weights and the
dimension-scaled TCC temperature, all training settings come from the current
``configs_v2/train.yaml`` and its referenced loss configs.

By default, ``softmax_temperature`` is rescaled so that
``embedding_dim * softmax_temperature`` remains constant relative to the source
configuration (with the current config: 128 -> 0.001 and 32 -> 0.004).  Pass
``--no-scale-softmax-temperature`` to retain the source value for both dims.

Usage (from the fineprog conda environment)
-------------------------------------------
    conda run -n fineprog python tests/sweep_fruit_expert_videos_loss_embdim.py
    conda run -n fineprog python tests/sweep_fruit_expert_videos_loss_embdim.py --dry-run
    conda run -n fineprog python tests/sweep_fruit_expert_videos_loss_embdim.py --gpu 0
    conda run -n fineprog python tests/sweep_fruit_expert_videos_loss_embdim.py \
        --variants tcc_d128 tcc_triplet_w0p5_d128
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
CONFIGS_ENV_VAR = "FINEPROG_CONFIGS_V2_DIR"

TRAIN_DATASET = "fruit_expert_videos_36vid_train"
EVAL_DATASET = "fruit_expert_videos_4vid_valid"
TCC_WEIGHT = 1.0
TRIPLET_WEIGHT = 0.5

TCC_LOSS_CONFIG = "loss/loss_tcc.yaml"
TRIPLET_LOSS_CONFIG = "loss/loss_temporal_triplet.yaml"
COMPOSITE_LOSS_CONFIG = "loss/loss_composite_tcc_triplet.yaml"

# Insertion order is the default execution order requested above.
VARIANTS: dict[str, dict] = {
    "tcc_d128": {
        "loss": "tcc",
        "embedding_dim": 128,
        "triplet_weight": None,
    },
    "tcc_triplet_w0p5_d128": {
        "loss": "composite",
        "embedding_dim": 128,
        "triplet_weight": TRIPLET_WEIGHT,
    },
    "tcc_d32": {
        "loss": "tcc",
        "embedding_dim": 32,
        "triplet_weight": None,
    },
    "tcc_triplet_w0p5_d32": {
        "loss": "composite",
        "embedding_dim": 32,
        "triplet_weight": TRIPLET_WEIGHT,
    },
}


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def read_source_reference() -> tuple[int, float]:
    """Return the source config's (embedding_dim, softmax_temperature)."""
    train_path = SOURCE_CONFIG_ROOT / "train.yaml"
    train_config = _read_yaml(train_path)
    if "embedding_dim" not in train_config:
        raise ValueError(f"'embedding_dim' not found in {train_path}")

    tcc_path = SOURCE_CONFIG_ROOT / TCC_LOSS_CONFIG
    tcc_config = _read_yaml(tcc_path)
    if "softmax_temperature" not in tcc_config:
        raise ValueError(f"'softmax_temperature' not found in {tcc_path}")

    return int(train_config["embedding_dim"]), float(tcc_config["softmax_temperature"])


def apply_overrides(
    config_root: Path,
    variant: dict,
    softmax_temperature: float | None,
) -> dict:
    """Apply one sweep point to an isolated configs_v2 snapshot."""
    train_path = config_root / "train.yaml"
    train_config = _read_yaml(train_path)
    train_config["train_dataset"] = TRAIN_DATASET
    train_config["embedding_dim"] = int(variant["embedding_dim"])

    if variant["loss"] == "composite":
        train_config["loss_name"] = "composite"
        train_config["loss_config"] = COMPOSITE_LOSS_CONFIG
    else:
        train_config["loss_name"] = "tcc"
        train_config["loss_config"] = TCC_LOSS_CONFIG

    eval_block = train_config.get("in_training_eval")
    if isinstance(eval_block, dict):
        eval_block["eval_dataset_ref"] = EVAL_DATASET
    _write_yaml(train_path, train_config)

    # This child config is used by both plain TCC and the composite loss.
    tcc_path = config_root / TCC_LOSS_CONFIG
    if softmax_temperature is not None:
        tcc_config = _read_yaml(tcc_path)
        tcc_config["softmax_temperature"] = float(softmax_temperature)
        _write_yaml(tcc_path, tcc_config)
    effective_temperature = float(_read_yaml(tcc_path)["softmax_temperature"])

    if variant["loss"] == "composite":
        composite_path = config_root / COMPOSITE_LOSS_CONFIG
        composite_config = _read_yaml(composite_path)
        by_alias = {
            item.get("alias"): item
            for item in composite_config.get("components", [])
        }
        for alias in ("tcc", "temporal_triplet"):
            if alias not in by_alias:
                raise ValueError(f"Missing component alias '{alias}' in {composite_path}")
        by_alias["tcc"]["weight"] = TCC_WEIGHT
        by_alias["temporal_triplet"]["weight"] = float(variant["triplet_weight"])
        _write_yaml(composite_path, composite_config)

    triplet_path = config_root / TRIPLET_LOSS_CONFIG
    triplet_config = _read_yaml(triplet_path)
    if "margin" not in triplet_config:
        raise ValueError(f"'margin' not found in {triplet_path}")

    return {
        "softmax_temperature": effective_temperature,
        "triplet_margin": (
            float(triplet_config["margin"])
            if variant["loss"] == "composite"
            else None
        ),
        "num_epochs": int(train_config.get("num_epochs", 0)),
        "batch_size": int(train_config.get("batch_size", 0)),
        "learning_rate": float(train_config.get("learning_rate", 0.0)),
    }


def prepare_snapshots(
    sweep_root: Path,
    variant_names: list[str],
    scale_temperature: bool,
) -> list[dict]:
    """Snapshot configs_v2 and derive one isolated config per variant."""
    base_dim, base_temperature = read_source_reference()

    source_snapshot = sweep_root / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    jobs: list[dict] = []
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        config_root = sweep_root / f"configs_{variant_name}"
        shutil.copytree(source_snapshot, config_root)

        temperature = None
        if scale_temperature:
            temperature = base_temperature * base_dim / float(variant["embedding_dim"])

        resolved = apply_overrides(config_root, variant, temperature)
        jobs.append(
            {
                "label": variant_name,
                "variant": variant_name,
                "loss": variant["loss"],
                "embedding_dim": int(variant["embedding_dim"]),
                "triplet_weight": variant["triplet_weight"],
                "train_dataset": TRAIN_DATASET,
                "eval_dataset": EVAL_DATASET,
                "config_root": str(config_root),
                "status": "pending",
                **resolved,
            }
        )
    return jobs


def run_training_worker(config_root: Path, job_json: Path, sweep_group: str) -> None:
    """Run train_encoder.train() against one isolated config snapshot."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import wandb

    import train_encoder
    from utils.config_v2 import ConfigV2

    job = json.loads(job_json.read_text(encoding="utf-8"))
    config_v2 = ConfigV2(config_root)
    train_config = config_v2.load_train()
    loss_config_rel = train_config.get("loss_config", TCC_LOSS_CONFIG)

    # train_encoder resolves these globals at import time; redirect every one to
    # the isolated snapshot before invoking the same train() entrypoint as its CLI.
    train_encoder._CFG_V2 = config_v2
    train_encoder._TRAIN_V2 = train_config
    train_encoder._V2_TRAIN_YAML = str(config_root / "train.yaml")
    train_encoder._loss_name_v2 = train_config.get("loss_name", "tcc")
    train_encoder._loss_cfg_file = loss_config_rel
    train_encoder._V2_LOSS_YAML = str(config_root / loss_config_rel)

    original_init = wandb.init

    def _init_with_sweep_metadata(*args, **kwargs):
        base_name = kwargs.get("name")
        if base_name:
            kwargs["name"] = f"{base_name}-{job['variant']}"
        kwargs.setdefault("group", sweep_group)
        kwargs.setdefault("job_type", "fruit_expert_videos")
        run_config = dict(kwargs.get("config") or {})
        run_config.update(
            {
                "sweep_group": sweep_group,
                "sweep_variant": job["variant"],
                "train_dataset": job["train_dataset"],
                "eval_dataset": job["eval_dataset"],
                "loss_name": job["loss"],
                "embedding_dim": job["embedding_dim"],
                "softmax_temperature": job["softmax_temperature"],
                "triplet_weight": job["triplet_weight"],
                "triplet_margin": job["triplet_margin"],
                "tcc_weight": TCC_WEIGHT if job["loss"] == "composite" else None,
            }
        )
        kwargs["config"] = run_config
        return original_init(*args, **kwargs)

    wandb.init = _init_with_sweep_metadata
    try:
        train_encoder.train(
            num_epochs=train_config.get("num_epochs", 5),
            batch_size=train_config.get("batch_size", 2),
            learning_rate=train_config.get("learning_rate", 1e-4),
            log_every=train_config.get("log_every", 10),
            num_workers=train_config.get("num_workers", 0),
            checkpoint_every=train_config.get("checkpoint_every", 1000),
            checkpoint_dir=train_config.get("checkpoint_dir", "checkpoint"),
            h5_path=None,
            register=False,
            register_alias=None,
        )
    finally:
        wandb.init = original_init


def _worker_command(job: dict, job_json: Path, sweep_group: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config-root",
        job["config_root"],
        "--worker-job-json",
        str(job_json),
        "--worker-sweep-group",
        sweep_group,
    ]


def _worker_env(job: dict, sweep_group: str, gpu: str | None) -> dict:
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = sweep_group
    env["PYTHONUNBUFFERED"] = "1"
    env[CONFIGS_ENV_VAR] = job["config_root"]
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
    """Run every job serially and return the failed jobs."""
    failed: list[dict] = []
    for index, job in enumerate(jobs, start=1):
        job_json = log_dir / f"job_{job['label']}.json"
        job_json.write_text(json.dumps(job, indent=2), encoding="utf-8")
        log_path = log_dir / f"{job['label']}_attempt{attempt}.log"
        print(
            f"[sweep] ({index}/{len(jobs)}) starting {job['label']} -> {log_path}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            code = subprocess.call(
                _worker_command(job, job_json, sweep_group),
                cwd=PROJECT_ROOT,
                env=_worker_env(job, sweep_group, gpu),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        if code == 0:
            job["status"] = "ok"
            print(f"[sweep] DONE {job['label']} (log: {log_path})", flush=True)
        else:
            job["status"] = f"failed(exit={code})"
            failed.append(job)
            print(
                f"[sweep] FAILED {job['label']} exit={code} (log: {log_path})",
                flush=True,
            )
    return failed


def write_summary(jobs: list[dict], log_dir: Path, sweep_group: str) -> Path:
    summary_path = log_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "sweep_group": sweep_group,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "train_dataset": TRAIN_DATASET,
                "eval_dataset": EVAL_DATASET,
                "tcc_weight": TCC_WEIGHT,
                "runs": jobs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Serial four-run loss/embedding-dimension sweep on "
            "fruit_expert_videos_36vid_train."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=list(VARIANTS),
        help=f"Variants to sweep (default: {' '.join(VARIANTS)}).",
    )
    parser.add_argument(
        "--no-scale-softmax-temperature",
        dest="scale_softmax_temperature",
        action="store_false",
        help=(
            "Keep loss_tcc.yaml -> softmax_temperature at its source value. "
            "By default embedding_dim * softmax_temperature stays constant."
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
        help="Directory for logs/configs (default: outputs/sweeps/<sweep_group>).",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Do not retry failed runs once at the end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate config overrides without starting training.",
    )
    parser.add_argument("--worker-config-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-job-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sweep-group", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_config_root is not None:
        if args.worker_job_json is None or args.worker_sweep_group is None:
            parser.error("worker mode requires config root, job JSON and sweep group")
        run_training_worker(
            args.worker_config_root.resolve(),
            args.worker_job_json.resolve(),
            args.worker_sweep_group,
        )
        return 0

    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing config directory: {SOURCE_CONFIG_ROOT}")

    sweep_group = "fruit-loss-embdim-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir or (PROJECT_ROOT / "outputs" / "sweeps" / sweep_group)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = prepare_snapshots(
        log_dir / "configs",
        list(args.variants),
        args.scale_softmax_temperature,
    )

    print(f"[sweep] group          : {sweep_group}")
    print(f"[sweep] log dir        : {log_dir}")
    print("[sweep] scheduling     : serial (1 run at a time)")
    print(f"[sweep] train dataset  : {TRAIN_DATASET}")
    print(f"[sweep] eval dataset   : {EVAL_DATASET}")
    print(f"[sweep] scale softmaxT : {args.scale_softmax_temperature}")
    print(f"[sweep] total runs     : {len(jobs)}")
    header = (
        f"{'#':>3}  {'variant':<26}  {'loss':<10}  {'dim':>4}  "
        f"{'tripletW':>9}  {'softmaxT':>10}  {'tripletMargin':>13}"
    )
    print(header)
    print("-" * len(header))
    for index, job in enumerate(jobs, start=1):
        triplet_weight = job["triplet_weight"]
        triplet_cell = "-" if triplet_weight is None else f"{triplet_weight:g}"
        triplet_margin = job["triplet_margin"]
        margin_cell = "-" if triplet_margin is None else f"{triplet_margin:g}"
        print(
            f"{index:>3}  {job['variant']:<26}  {job['loss']:<10}  "
            f"{job['embedding_dim']:>4}  {triplet_cell:>9}  "
            f"{job['softmax_temperature']:>10g}  {margin_cell:>13}",
            flush=True,
        )

    if args.dry_run:
        summary_path = write_summary(jobs, log_dir, sweep_group)
        print(f"[sweep] dry run complete; summary: {summary_path}")
        return 0

    failed = run_jobs(jobs, sweep_group, log_dir, args.gpu, attempt=1)
    write_summary(jobs, log_dir, sweep_group)

    if failed and not args.no_retry:
        print(
            f"\n[sweep] retrying {len(failed)} failed run(s) serially (attempt 2) ...",
            flush=True,
        )
        failed = run_jobs(failed, sweep_group, log_dir, args.gpu, attempt=2)

    summary_path = write_summary(jobs, log_dir, sweep_group)
    print("\n[sweep] -- summary --")
    for job in jobs:
        print(f"[sweep]   {job['label']:<26}  {job.get('status', 'unknown')}")
    print(f"[sweep] wandb group : {sweep_group}")
    print(f"[sweep] summary json: {summary_path}")

    if failed:
        print(f"[sweep] {len(failed)} run(s) failed; inspect logs in {log_dir}")
        return 1
    print("[sweep] all runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
