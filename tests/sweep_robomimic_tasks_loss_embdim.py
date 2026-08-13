#!/usr/bin/env python3
"""Serial encoder-training sweep over robomimic lift / square / transport.

Per task, six training variants are run strictly one after another:

    #  loss                                      embedding_dim
    1  tcc                                       128
    2  composite tcc + triplet (margin 0.2, w=0.5)   128
    3  composite tcc + triplet (margin 0.2, w=1.0)   128
    4  tcc                                        64
    5  tcc                                        32
    6  composite tcc + triplet (margin 0.2, w=1.0)    32

3 tasks x 6 variants = 18 runs, executed variant-major: a variant is trained on
lift, square and transport before moving on to the next variant.

Config isolation
----------------
``configs_v2`` is snapshotted once at startup and one isolated copy is derived
per (task, variant).  Only these keys are overridden in the copy:

    train.yaml                          :: train_dataset, embedding_dim,
                                           loss_name, loss_config,
                                           in_training_eval.eval_dataset_ref
    loss/loss_tcc.yaml                  :: softmax_temperature (dim-scaled)
    loss/loss_temporal_triplet.yaml     :: margin
    loss/loss_composite_tcc_triplet.yaml:: component weights (tcc / triplet)

The repository's own YAML files are never modified.  Workers read the snapshot
through ``FINEPROG_CONFIGS_V2_DIR``, honoured by ``utils/config_v2.ConfigV2``.

softmax_temperature is rescaled so ``embedding_dim * softmax_temperature`` stays
constant w.r.t. the repository config (128 -> 0.001, 64 -> 0.002, 32 -> 0.004),
as documented in ``configs_v2/train.yaml``.  Disable with
``--no-scale-softmax-temperature``.

Every other hyper-parameter (num_epochs, batch_size, lr, clip_len, ...) is taken
verbatim from the current ``configs_v2``.

Usage (from the fineprog conda environment)
-------------------------------------------
    conda run -n fineprog python tests/sweep_robomimic_tasks_loss_embdim.py
    conda run -n fineprog python tests/sweep_robomimic_tasks_loss_embdim.py --dry-run
    conda run -n fineprog python tests/sweep_robomimic_tasks_loss_embdim.py --tasks lift square
    conda run -n fineprog python tests/sweep_robomimic_tasks_loss_embdim.py --variants tcc_d128 tcc_d32
    conda run -n fineprog python tests/sweep_robomimic_tasks_loss_embdim.py --gpu 0
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

TRIPLET_MARGIN = 0.2
TCC_WEIGHT = 1.0

TCC_LOSS_CONFIG = "loss/loss_tcc.yaml"
COMPOSITE_LOSS_CONFIG = "loss/loss_composite_tcc_triplet.yaml"

# Registry keys from configs_v2/registry/datasets.yaml
TASKS: dict[str, dict[str, str]] = {
    "lift": {
        "train_dataset": "robomimic_lift_ph_36vid_train",
        "eval_dataset": "robomimic_lift_ph_4vid_valid",
    },
    "square": {
        "train_dataset": "robomimic_square_ph_36vid_train",
        "eval_dataset": "robomimic_square_ph_4vid_valid",
    },
    "transport": {
        "train_dataset": "robomimic_transport_ph_36vid_train",
        "eval_dataset": "robomimic_transport_ph_4vid_valid",
    },
}

# loss: "tcc" | "composite"; triplet_weight only used when loss == "composite".
VARIANTS: dict[str, dict] = {
    "tcc_d128":              {"loss": "tcc",       "embedding_dim": 128, "triplet_weight": None},
    "tcc_triplet_w0p5_d128": {"loss": "composite", "embedding_dim": 128, "triplet_weight": 0.5},
    "tcc_triplet_w1p0_d128": {"loss": "composite", "embedding_dim": 128, "triplet_weight": 1.0},
    "tcc_d64":               {"loss": "tcc",       "embedding_dim": 64,  "triplet_weight": None},
    "tcc_d32":               {"loss": "tcc",       "embedding_dim": 32,  "triplet_weight": None},
    "tcc_triplet_w1p0_d32":  {"loss": "composite", "embedding_dim": 32,  "triplet_weight": 1.0},
}


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def read_source_reference() -> tuple[int, float]:
    """Return (embedding_dim, softmax_temperature) of the repository config."""
    train_config = _read_yaml(SOURCE_CONFIG_ROOT / "train.yaml")
    if "embedding_dim" not in train_config:
        raise ValueError(f"'embedding_dim' not found in {SOURCE_CONFIG_ROOT / 'train.yaml'}")

    tcc_path = SOURCE_CONFIG_ROOT / TCC_LOSS_CONFIG
    tcc_config = _read_yaml(tcc_path)
    if "softmax_temperature" not in tcc_config:
        raise ValueError(f"'softmax_temperature' not found in {tcc_path}")

    return int(train_config["embedding_dim"]), float(tcc_config["softmax_temperature"])


# ---------------------------------------------------------------------------
# Config snapshot preparation
# ---------------------------------------------------------------------------

def apply_overrides(
    config_root: Path,
    task: dict[str, str],
    variant: dict,
    softmax_temperature: float | None,
) -> dict:
    """Pin dataset / loss / embedding_dim for one sweep point."""
    train_path = config_root / "train.yaml"
    train_config = _read_yaml(train_path)

    train_config["train_dataset"] = task["train_dataset"]
    train_config["embedding_dim"] = int(variant["embedding_dim"])
    if variant["loss"] == "composite":
        train_config["loss_name"] = "composite"
        train_config["loss_config"] = COMPOSITE_LOSS_CONFIG
    else:
        train_config["loss_name"] = "tcc"
        train_config["loss_config"] = TCC_LOSS_CONFIG

    eval_block = train_config.get("in_training_eval")
    if isinstance(eval_block, dict):
        eval_block["eval_dataset_ref"] = task["eval_dataset"]
    _write_yaml(train_path, train_config)

    # TCC child config is shared by the plain and the composite setups.
    tcc_path = config_root / TCC_LOSS_CONFIG
    if softmax_temperature is not None:
        tcc_config = _read_yaml(tcc_path)
        tcc_config["softmax_temperature"] = float(softmax_temperature)
        _write_yaml(tcc_path, tcc_config)
    effective_temperature = float(_read_yaml(tcc_path)["softmax_temperature"])

    if variant["loss"] == "composite":
        composite_path = config_root / COMPOSITE_LOSS_CONFIG
        composite_config = _read_yaml(composite_path)
        by_alias = {item.get("alias"): item for item in composite_config.get("components", [])}
        for alias in ("tcc", "temporal_triplet"):
            if alias not in by_alias:
                raise ValueError(f"Missing component alias '{alias}' in {composite_path}")
        by_alias["tcc"]["weight"] = float(TCC_WEIGHT)
        by_alias["temporal_triplet"]["weight"] = float(variant["triplet_weight"])
        _write_yaml(composite_path, composite_config)

        triplet_path = composite_path.parent / str(by_alias["temporal_triplet"]["config_file"])
        triplet_config = _read_yaml(triplet_path)
        if "margin" not in triplet_config:
            raise ValueError(f"'margin' not found in {triplet_path}")
        triplet_config["margin"] = float(TRIPLET_MARGIN)
        _write_yaml(triplet_path, triplet_config)

    return {
        "softmax_temperature": effective_temperature,
        "num_epochs": int(train_config.get("num_epochs", 0)),
        "batch_size": int(train_config.get("batch_size", 0)),
        "learning_rate": float(train_config.get("learning_rate", 0.0)),
    }


def prepare_snapshots(
    sweep_root: Path,
    task_names: list[str],
    variant_names: list[str],
    scale_temperature: bool,
) -> list[dict]:
    """Snapshot configs_v2 once, then derive one isolated config per job."""
    base_dim, base_temperature = read_source_reference()

    source_snapshot = sweep_root / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    # Variant-major order: each variant is run across all tasks before the next.
    jobs: list[dict] = []
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        for task_name in task_names:
            task = TASKS[task_name]
            label = f"{task_name}__{variant_name}"
            config_root = sweep_root / f"configs_{label}"
            shutil.copytree(source_snapshot, config_root)

            temperature = None
            if scale_temperature:
                # Keep embedding_dim * softmax_temperature constant.
                temperature = base_temperature * base_dim / float(variant["embedding_dim"])

            resolved = apply_overrides(config_root, task, variant, temperature)
            jobs.append(
                {
                    "label": label,
                    "task": task_name,
                    "variant": variant_name,
                    "loss": variant["loss"],
                    "embedding_dim": int(variant["embedding_dim"]),
                    "triplet_weight": variant["triplet_weight"],
                    "triplet_margin": TRIPLET_MARGIN if variant["loss"] == "composite" else None,
                    "train_dataset": task["train_dataset"],
                    "eval_dataset": task["eval_dataset"],
                    "config_root": str(config_root),
                    "status": "pending",
                    **resolved,
                }
            )
    return jobs


# ---------------------------------------------------------------------------
# Worker: run one training with an isolated ConfigV2 directory
# ---------------------------------------------------------------------------

def run_training_worker(config_root: Path, job_json: Path, sweep_group: str) -> None:
    """Run train_encoder.train() against the supplied isolated config snapshot."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import wandb

    import train_encoder
    from utils.config_v2 import ConfigV2

    job = json.loads(job_json.read_text(encoding="utf-8"))

    config_v2 = ConfigV2(config_root)
    train_config = config_v2.load_train()
    loss_config_rel = train_config.get("loss_config", TCC_LOSS_CONFIG)

    # train_encoder resolves these module globals on import. Point them at the
    # isolated snapshot before invoking the same train() call as its CLI.
    train_encoder._CFG_V2 = config_v2
    train_encoder._TRAIN_V2 = train_config
    train_encoder._V2_TRAIN_YAML = str(config_root / "train.yaml")
    train_encoder._loss_name_v2 = train_config.get("loss_name", "tcc")
    train_encoder._loss_cfg_file = loss_config_rel
    train_encoder._V2_LOSS_YAML = str(config_root / loss_config_rel)

    suffix = job["variant"]
    _original_init = wandb.init

    def _init_with_sweep_metadata(*args, **kwargs):
        base_name = kwargs.get("name")
        if base_name:
            kwargs["name"] = f"{base_name}-{suffix}"
        kwargs.setdefault("group", sweep_group)
        kwargs.setdefault("job_type", f"task_{job['task']}")
        run_config = dict(kwargs.get("config") or {})
        run_config.update(
            {
                "sweep_group": sweep_group,
                "sweep_task": job["task"],
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
            checkpoint_dir=train_config.get("checkpoint_dir", "checkpoint"),
            h5_path=None,
            register=False,
            register_alias=None,
        )
    finally:
        wandb.init = _original_init


# ---------------------------------------------------------------------------
# Launcher (strictly serial)
# ---------------------------------------------------------------------------

def _worker_command(job: dict, job_json: Path, sweep_group: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config-root", job["config_root"],
        "--worker-job-json", str(job_json),
        "--worker-sweep-group", sweep_group,
    ]


def _worker_env(job: dict, sweep_group: str, gpu: str | None) -> dict:
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = sweep_group
    env["PYTHONUNBUFFERED"] = "1"
    # Entrypoints that build their own ConfigV2 (e.g. in-training eval helpers)
    # must see the isolated snapshot as well.
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
    """Run every job serially. Returns the list of failed jobs."""
    failed: list[dict] = []

    for index, job in enumerate(jobs, start=1):
        job_json = log_dir / f"job_{job['label']}.json"
        job_json.write_text(json.dumps(job, indent=2), encoding="utf-8")
        log_path = log_dir / f"{job['label']}_attempt{attempt}.log"
        print(
            f"[sweep] ({index}/{len(jobs)}) starting {job['label']} → {log_path}",
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
                "triplet_margin": TRIPLET_MARGIN,
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
            "Serial encoder-training sweep over robomimic lift/square/transport "
            "x {tcc, tcc+triplet(w=0.5/1.0)} x {embedding_dim 128/64/32}."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=list(TASKS),
        help=f"Tasks to sweep (default: {' '.join(TASKS)}).",
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
            "Keep loss_tcc.yaml -> softmax_temperature at its current value. "
            "By default it is rescaled so embedding_dim * softmax_temperature "
            "stays constant (128→0.001, 64→0.002, 32→0.004)."
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
        help="Directory for logs and configs (default: outputs/sweeps/<sweep_group>).",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Do not retry failed runs once at the end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate all config overrides without starting training.",
    )
    parser.add_argument("--worker-config-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-job-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sweep-group", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_config_root is not None:
        run_training_worker(
            args.worker_config_root.resolve(),
            args.worker_job_json.resolve(),
            str(args.worker_sweep_group),
        )
        return 0

    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing config directory: {SOURCE_CONFIG_ROOT}")

    sweep_group = "robomimic-task-loss-embdim-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir or (PROJECT_ROOT / "outputs" / "sweeps" / sweep_group)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = prepare_snapshots(
        log_dir / "configs",
        list(args.tasks),
        list(args.variants),
        args.scale_softmax_temperature,
    )

    print(f"[sweep] group          : {sweep_group}")
    print(f"[sweep] log dir        : {log_dir}")
    print(f"[sweep] scheduling     : serial (1 run at a time)")
    print(f"[sweep] scale softmaxT : {args.scale_softmax_temperature}")
    print(f"[sweep] total runs     : {len(jobs)}")
    header = (
        f"{'#':>3}  {'label':<32}  {'loss':<10}  {'dim':>4}  {'tripletW':>9}  "
        f"{'softmaxT':>10}  train_dataset"
    )
    print(header)
    print("-" * len(header))
    for index, job in enumerate(jobs, start=1):
        triplet_weight = job["triplet_weight"]
        triplet_cell = "-" if triplet_weight is None else f"{triplet_weight:g}"
        print(
            f"{index:>3}  {job['label']:<32}  {job['loss']:<10}  "
            f"{job['embedding_dim']:>4}  {triplet_cell:>9}  "
            f"{job['softmax_temperature']:>10g}  {job['train_dataset']}",
            flush=True,
        )

    if args.dry_run:
        write_summary(jobs, log_dir, sweep_group)
        print("[sweep] dry run complete; no training started.")
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

    print("\n[sweep] ── summary ──")
    for job in jobs:
        print(f"[sweep]   {job['label']:<32}  {job.get('status', 'unknown')}")
    print(f"[sweep] wandb group : {sweep_group}")
    print(f"[sweep] summary json: {summary_path}")

    if failed:
        print(f"[sweep] {len(failed)} run(s) failed; inspect logs in {log_dir}")
        return 1
    print("[sweep] all runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
