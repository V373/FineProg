#!/usr/bin/env python3
"""Serial sweep over the TemporalTriplet ``margin`` + full downstream pipeline.

For each margin in ``MARGINS`` the script runs, strictly serially:

    1. train      — composite TCC + TemporalTriplet encoder training
                    (TCC softmax_temperature fixed at 0.001,
                     component weights tcc=1.0 / temporal_triplet=0.5)
    2. extract    — CLI ``extract_embeddings.py`` on three datasets:
                    train-36, valid-4, worse-100
    3. fitting    — CLI ``evaluate_encoder.py --task gaussian_progress_fitting``
                    on the train-36 embeddings (one Gaussian model per run)
    4. pred x2    — CLI ``evaluate_encoder.py --task gaussian_progress_pred``
                    on valid-4 and worse-100, posterior_temperature = 1e6

Config isolation
----------------
``configs_v2`` is snapshotted once at startup and one isolated copy is created
per margin.  Only these keys are overridden in the copy:

    loss/loss_temporal_triplet.yaml     :: margin
    loss/loss_tcc.yaml                  :: softmax_temperature (0.001)
    loss/loss_composite_tcc_triplet.yaml:: component weights (1.0 / 0.5)
    extract.yaml                        :: extract_dataset / checkpoint_path /
                                           embedding_save_path / clip_len /
                                           context_size / context_stride
    eval/gaussian_progress_fitting.yaml :: expert_h5_path / output_dir
    eval/gaussian_progress_pred.yaml    :: gaussian_model_h5_path /
                                           nonexpert_h5_path /
                                           posterior_temperature

The repository's own YAML files are never modified.  CLI subprocesses read the
snapshot through the ``FINEPROG_CONFIGS_V2_DIR`` environment variable, which is
honoured by ``utils/config_v2.ConfigV2``.

Usage (from the fineprog conda environment)
-------------------------------------------
    conda run -n fineprog python tests/sweep_triplet_margin_pipeline.py
    conda run -n fineprog python tests/sweep_triplet_margin_pipeline.py --dry-run
    conda run -n fineprog python tests/sweep_triplet_margin_pipeline.py --margins 0.05 0.1
    conda run -n fineprog python tests/sweep_triplet_margin_pipeline.py --skip-training \
        --resume-summary outputs/sweeps/<group>/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


# ── Sweep definition ──────────────────────────────────────────────────────────
MARGINS = (0.05, 0.1, 0.2)

TCC_SOFTMAX_TEMPERATURE = 0.001
TCC_WEIGHT = 1.0
TRIPLET_WEIGHT = 0.5
POSTERIOR_TEMPERATURE = 1.0e6

# Registry keys from configs_v2/registry/datasets.yaml
EXPERT_DATASET_REF = "robomimic_can_ph_36vid_train"      # Gaussian fitting source
QUERY_DATASET_REFS = (
    "robomimic_can_ph_4vid_valid",                       # valid-4
    "robomimic_can_mh_100vid_worse",                     # worse-100
)
EXTRACT_DATASET_REFS = (EXPERT_DATASET_REF,) + QUERY_DATASET_REFS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
CONFIGS_ENV_VAR = "FINEPROG_CONFIGS_V2_DIR"


def _label(value: float) -> str:
    """Filesystem/wandb friendly label, e.g. 0.05 -> '0p05'."""
    return f"{value:g}".replace(".", "p")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Config snapshot preparation
# ---------------------------------------------------------------------------

def apply_loss_overrides(config_root: Path, margin: float) -> dict:
    """Pin the composite loss hyper-parameters for one sweep point."""
    train_path = config_root / "train.yaml"
    train_cfg = _load_yaml(train_path)

    if train_cfg.get("loss_name") != "composite":
        raise ValueError(
            f"Expected loss_name='composite' in {train_path}, "
            f"got {train_cfg.get('loss_name')!r}"
        )
    loss_config_rel = train_cfg.get("loss_config")
    if not loss_config_rel:
        raise ValueError(f"Missing loss_config in {train_path}")

    composite_path = config_root / str(loss_config_rel)
    composite_cfg = _load_yaml(composite_path)
    components = composite_cfg.get("components", [])
    by_alias = {item.get("alias"): item for item in components}
    for alias in ("tcc", "temporal_triplet"):
        if alias not in by_alias:
            raise ValueError(f"Missing component alias '{alias}' in {composite_path}")
    by_alias["tcc"]["weight"] = float(TCC_WEIGHT)
    by_alias["temporal_triplet"]["weight"] = float(TRIPLET_WEIGHT)
    _dump_yaml(composite_path, composite_cfg)

    loss_dir = composite_path.parent

    tcc_path = loss_dir / str(by_alias["tcc"]["config_file"])
    tcc_cfg = _load_yaml(tcc_path)
    if "softmax_temperature" not in tcc_cfg:
        raise ValueError(f"'softmax_temperature' not found in {tcc_path}")
    tcc_cfg["softmax_temperature"] = float(TCC_SOFTMAX_TEMPERATURE)
    _dump_yaml(tcc_path, tcc_cfg)

    triplet_path = loss_dir / str(by_alias["temporal_triplet"]["config_file"])
    triplet_cfg = _load_yaml(triplet_path)
    if "margin" not in triplet_cfg:
        raise ValueError(f"'margin' not found in {triplet_path}")
    triplet_cfg["margin"] = float(margin)
    _dump_yaml(triplet_path, triplet_cfg)

    return {
        "train_yaml": str(train_path),
        "composite_yaml": str(composite_path),
        "tcc_yaml": str(tcc_path),
        "triplet_yaml": str(triplet_path),
        "num_epochs": int(train_cfg.get("num_epochs", 0)),
        "checkpoint_every": int(train_cfg.get("checkpoint_every", 0)),
        "checkpoint_dir": str(train_cfg.get("checkpoint_dir", "checkpoint")),
        "train_dataset": train_cfg.get("train_dataset"),
        "clip_len": train_cfg.get("clip_len"),
        "context_size": train_cfg.get("context_size"),
        "context_stride": train_cfg.get("context_stride"),
    }


def prepare_snapshots(sweep_root: Path, margins: tuple[float, ...]) -> list[dict]:
    """Snapshot configs_v2 once, then derive one isolated config per margin."""
    source_snapshot = sweep_root / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    jobs = []
    for margin in margins:
        label = _label(margin)
        config_root = sweep_root / f"configs_margin_{label}"
        shutil.copytree(source_snapshot, config_root)
        train_info = apply_loss_overrides(config_root, margin)
        jobs.append(
            {
                "margin": float(margin),
                "label": label,
                "config_root": str(config_root),
                "train_config": train_info,
                "status": "pending",
            }
        )
    return jobs


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run_cli(
    command: list[str],
    config_root: Path,
    log_path: Path,
    step_name: str,
) -> str:
    """Run one CLI step with the isolated config snapshot; return its stdout."""
    env = os.environ.copy()
    env[CONFIGS_ENV_VAR] = str(config_root)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[sweep]   → {step_name}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured = []
        assert process.stdout is not None
        for line in process.stdout:
            captured.append(line)
            log_file.write(line)
        return_code = process.wait()

    output = "".join(captured)
    if return_code != 0:
        raise RuntimeError(
            f"[sweep] step '{step_name}' failed with exit code {return_code}; "
            f"see {log_path}"
        )
    return output


def _search_last(pattern: str, text: str, step_name: str) -> str:
    matches = re.findall(pattern, text)
    if not matches:
        raise RuntimeError(
            f"[sweep] could not parse '{pattern}' from the output of '{step_name}'."
        )
    return matches[-1].strip()


# ---------------------------------------------------------------------------
# Pipeline steps (per margin)
# ---------------------------------------------------------------------------

def step_train(job: dict, config_root: Path, log_dir: Path, sweep_group: str) -> dict:
    """Train one encoder and return {run_name, checkpoint_path}."""
    result_json = log_dir / f"train_result_{job['label']}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config-root", str(config_root),
        "--worker-margin", repr(job["margin"]),
        "--worker-sweep-group", sweep_group,
        "--worker-result-json", str(result_json),
    ]
    _run_cli(command, config_root, log_dir / f"01_train_{job['label']}.log", "train")

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    checkpoint_path = Path(payload["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"[sweep] expected final checkpoint not found: {checkpoint_path}"
        )
    return payload


def step_extract(
    job: dict,
    config_root: Path,
    log_dir: Path,
    run_name: str,
    checkpoint_path: str,
    epoch: int,
) -> dict:
    """Extract embeddings for all three datasets; return {dataset_ref: h5_path}."""
    from utils.config_v2 import ConfigV2

    cfg_v2 = ConfigV2(config_root)
    train_cfg = job["train_config"]
    embeddings_root = Path(cfg_v2._dirs["embeddings"]) / run_name

    extract_path = config_root / "extract.yaml"
    embedding_paths = {}

    for dataset_ref in EXTRACT_DATASET_REFS:
        stem = cfg_v2.resolve_dataset(dataset_ref)["h5_stem"]
        save_path = embeddings_root / f"{stem}-embd-ep{epoch:06d}.h5"

        extract_cfg = _load_yaml(extract_path)
        extract_cfg["extract_dataset"] = dataset_ref
        extract_cfg["checkpoint_path"] = str(checkpoint_path)
        extract_cfg.pop("checkpoint_ref", None)
        extract_cfg["embedding_save_path"] = str(save_path)
        # Sequence sampling must match the trained checkpoint.
        extract_cfg["clip_len"] = train_cfg["clip_len"]
        extract_cfg["context_size"] = train_cfg["context_size"]
        extract_cfg["context_stride"] = train_cfg["context_stride"]
        _dump_yaml(extract_path, extract_cfg)

        _run_cli(
            [sys.executable, "extract_embeddings.py"],
            config_root,
            log_dir / f"02_extract_{job['label']}_{dataset_ref}.log",
            f"extract[{dataset_ref}]",
        )
        if not save_path.is_file():
            raise FileNotFoundError(f"[sweep] embeddings not written: {save_path}")
        embedding_paths[dataset_ref] = str(save_path)

    return embedding_paths


def step_gaussian_fitting(
    job: dict,
    config_root: Path,
    log_dir: Path,
    expert_h5_path: str,
    run_output_dir: Path,
) -> dict:
    """Fit the progress-conditioned Gaussians on the expert embeddings."""
    fitting_path = config_root / "eval" / "gaussian_progress_fitting.yaml"
    fitting_cfg = _load_yaml(fitting_path)
    fitting_cfg["expert_h5_path"] = str(expert_h5_path)
    fitting_cfg["expert_embedding_ref"] = None
    fitting_cfg["output_dir"] = str(run_output_dir / "gaussian_progress_fitting")
    _dump_yaml(fitting_path, fitting_cfg)

    output = _run_cli(
        [sys.executable, "evaluate_encoder.py", "--task", "gaussian_progress_fitting"],
        config_root,
        log_dir / f"03_fitting_{job['label']}.log",
        "gaussian_progress_fitting",
    )
    model_h5_path = _search_last(
        r"\[gaussian_progress_fitting\] output_h5_path\s*:\s*(\S+)",
        output,
        "gaussian_progress_fitting",
    )
    return {
        "gaussian_model_h5_path": model_h5_path,
        "num_bins": int(fitting_cfg.get("num_bins", 0)),
        "covariance_mode": fitting_cfg.get("covariance_mode"),
    }


def step_gaussian_pred(
    job: dict,
    config_root: Path,
    log_dir: Path,
    gaussian_model_h5_path: str,
    dataset_ref: str,
    nonexpert_h5_path: str,
) -> dict:
    """Run one online prediction pass against the fitted Gaussian model."""
    pred_path = config_root / "eval" / "gaussian_progress_pred.yaml"
    pred_cfg = _load_yaml(pred_path)
    pred_cfg["gaussian_model_h5_path"] = str(gaussian_model_h5_path)
    pred_cfg["nonexpert_h5_path"] = str(nonexpert_h5_path)
    pred_cfg["posterior_temperature"] = float(POSTERIOR_TEMPERATURE)
    _dump_yaml(pred_path, pred_cfg)

    output = _run_cli(
        [sys.executable, "evaluate_encoder.py", "--task", "gaussian_progress_pred"],
        config_root,
        log_dir / f"04_pred_{job['label']}_{dataset_ref}.log",
        f"gaussian_progress_pred[{dataset_ref}]",
    )
    metric_value = float(
        _search_last(r"metric_value\s*:\s*([-\d.eE+]+)", output, "gaussian_progress_pred")
    )
    output_h5_path = _search_last(
        r"\[gaussian_progress_pred\] saved output H5:\s*(\S+)",
        output,
        "gaussian_progress_pred",
    )
    return {
        "dataset_ref": dataset_ref,
        "nonexpert_h5_path": str(nonexpert_h5_path),
        "metric_name": "global_mean_min_mahalanobis_sq",
        "metric_value": metric_value,
        "output_h5_path": output_h5_path,
    }


# ---------------------------------------------------------------------------
# Training worker (runs inside its own process)
# ---------------------------------------------------------------------------

def run_training_worker(
    config_root: Path,
    margin: float,
    sweep_group: str,
    result_json: Path,
) -> None:
    """Run train_encoder.train() against the supplied isolated config snapshot."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import wandb

    import train_encoder
    from utils.config_v2 import ConfigV2

    config_v2 = ConfigV2(config_root)
    train_config = config_v2.load_train()
    loss_config_rel = train_config.get("loss_config", "loss/loss_composite_tcc_triplet.yaml")

    # train_encoder resolves these module globals on import. Point them at the
    # isolated snapshot before invoking the same train() call as its CLI.
    train_encoder._CFG_V2 = config_v2
    train_encoder._TRAIN_V2 = train_config
    train_encoder._V2_TRAIN_YAML = str(config_root / "train.yaml")
    train_encoder._loss_name_v2 = train_config.get("loss_name", "composite")
    train_encoder._loss_cfg_file = loss_config_rel
    train_encoder._V2_LOSS_YAML = str(config_root / loss_config_rel)

    suffix = f"margin{margin:g}"
    captured = {}
    _original_init = wandb.init

    def _init_with_sweep_metadata(*args, **kwargs):
        base_name = kwargs.get("name")
        if base_name:
            kwargs["name"] = f"{base_name}-{suffix}"
        kwargs.setdefault("group", sweep_group)
        kwargs.setdefault("job_type", "triplet_margin_sweep")
        run_config = dict(kwargs.get("config") or {})
        run_config["triplet_margin"] = float(margin)
        run_config["triplet_weight"] = float(TRIPLET_WEIGHT)
        run_config["tcc_weight"] = float(TCC_WEIGHT)
        run_config["tcc_softmax_temperature"] = float(TCC_SOFTMAX_TEMPERATURE)
        run_config["sweep_group"] = sweep_group
        kwargs["config"] = run_config
        run = _original_init(*args, **kwargs)
        captured["run_name"] = run.name
        return run

    wandb.init = _init_with_sweep_metadata
    num_epochs = train_config.get("num_epochs", 5)
    checkpoint_dir = train_config.get("checkpoint_dir", "checkpoint")
    try:
        train_encoder.train(
            num_epochs=num_epochs,
            batch_size=train_config.get("batch_size", 2),
            learning_rate=train_config.get("learning_rate", 1e-4),
            log_every=train_config.get("log_every", 10),
            num_workers=train_config.get("num_workers", 0),
            checkpoint_every=train_config.get("checkpoint_every", 1000),
            checkpoint_dir=checkpoint_dir,
            h5_path=None,
            register=False,
            register_alias=None,
        )
    finally:
        wandb.init = _original_init

    run_name = captured.get("run_name")
    if not run_name:
        raise RuntimeError("[sweep] failed to capture the wandb run name.")

    checkpoint_root = Path(checkpoint_dir)
    if not checkpoint_root.is_absolute():
        checkpoint_root = PROJECT_ROOT / checkpoint_root
    checkpoint_path = checkpoint_root / run_name / f"encoder_epoch{num_epochs:06d}.pt"

    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "margin": float(margin),
                "num_epochs": int(num_epochs),
                "checkpoint_path": str(checkpoint_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def run_pipeline_for_job(
    job: dict,
    log_dir: Path,
    sweep_group: str,
    skip_training: bool,
) -> None:
    config_root = Path(job["config_root"])
    run_output_dir = log_dir / f"margin_{job['label']}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    job["output_dir"] = str(run_output_dir)

    # 1. train ---------------------------------------------------------------
    if skip_training:
        if not job.get("run_name") or not job.get("checkpoint_path"):
            raise ValueError(
                f"[sweep] --skip-training requires run_name/checkpoint_path for "
                f"margin={job['margin']:g} (supply them via --resume-summary)."
            )
        print(f"[sweep] skipping training; reusing run {job['run_name']}", flush=True)
    else:
        train_result = step_train(job, config_root, run_output_dir, sweep_group)
        job["run_name"] = train_result["run_name"]
        job["checkpoint_path"] = train_result["checkpoint_path"]
        job["num_epochs"] = train_result["num_epochs"]

    epoch = int(job.get("num_epochs") or job["train_config"]["num_epochs"])

    # 2. extract -------------------------------------------------------------
    job["embedding_paths"] = step_extract(
        job,
        config_root,
        run_output_dir,
        job["run_name"],
        job["checkpoint_path"],
        epoch,
    )

    # 3. gaussian fitting ----------------------------------------------------
    job["gaussian_fitting"] = step_gaussian_fitting(
        job,
        config_root,
        run_output_dir,
        job["embedding_paths"][EXPERT_DATASET_REF],
        run_output_dir,
    )

    # 4. gaussian prediction -------------------------------------------------
    job["gaussian_pred"] = [
        step_gaussian_pred(
            job,
            config_root,
            run_output_dir,
            job["gaussian_fitting"]["gaussian_model_h5_path"],
            dataset_ref,
            job["embedding_paths"][dataset_ref],
        )
        for dataset_ref in QUERY_DATASET_REFS
    ]

    job["status"] = "ok"


def write_summary(jobs: list[dict], log_dir: Path, sweep_group: str) -> Path:
    summary = {
        "sweep_group": sweep_group,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tcc_softmax_temperature": TCC_SOFTMAX_TEMPERATURE,
        "tcc_weight": TCC_WEIGHT,
        "triplet_weight": TRIPLET_WEIGHT,
        "posterior_temperature": POSTERIOR_TEMPERATURE,
        "expert_dataset_ref": EXPERT_DATASET_REF,
        "query_dataset_refs": list(QUERY_DATASET_REFS),
        "runs": jobs,
    }
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def print_summary(jobs: list[dict], sweep_group: str, summary_path: Path) -> None:
    print("\n[sweep] ── summary ──")
    header = f"{'margin':>7}  {'status':<22}  {'dataset':<32}  global_mean_min_mahalanobis_sq"
    print(header)
    print("-" * len(header))
    for job in jobs:
        preds = job.get("gaussian_pred") or []
        if not preds:
            print(f"{job['margin']:>7g}  {job['status']:<22}  {'-':<32}  -")
            continue
        for index, pred in enumerate(preds):
            margin_cell = f"{job['margin']:g}" if index == 0 else ""
            status_cell = job["status"] if index == 0 else ""
            print(
                f"{margin_cell:>7}  {status_cell:<22}  {pred['dataset_ref']:<32}  "
                f"{pred['metric_value']:.6f}"
            )
    print(f"\n[sweep] wandb group : {sweep_group}")
    print(f"[sweep] summary json: {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Serial sweep of TemporalTriplet margin (0.05 / 0.1 / 0.2) with "
            "TCC softmax_temperature fixed at 0.001, followed by embedding "
            "extraction and Gaussian progress fitting/prediction per run."
        )
    )
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=list(MARGINS),
        help=f"Margins to sweep (default: {' '.join(f'{m:g}' for m in MARGINS)}).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for logs and artifacts (default: outputs/sweeps/<sweep_group>).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse the runs recorded in --resume-summary and only redo the downstream steps.",
    )
    parser.add_argument(
        "--resume-summary",
        type=Path,
        default=None,
        help="summary.json of a previous sweep, used to recover run_name/checkpoint_path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate all config overrides without running anything.",
    )
    parser.add_argument("--worker-config-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-margin", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sweep-group", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_config_root is not None:
        run_training_worker(
            args.worker_config_root.resolve(),
            float(args.worker_margin),
            str(args.worker_sweep_group),
            args.worker_result_json.resolve(),
        )
        return 0

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing config directory: {SOURCE_CONFIG_ROOT}")

    sweep_group = "triplet-margin-sweep-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir or (PROJECT_ROOT / "outputs" / "sweeps" / sweep_group)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = prepare_snapshots(log_dir / "configs", tuple(args.margins))

    if args.resume_summary is not None:
        previous = json.loads(args.resume_summary.read_text(encoding="utf-8"))
        previous_by_label = {run["label"]: run for run in previous.get("runs", [])}
        for job in jobs:
            old = previous_by_label.get(job["label"], {})
            for key in ("run_name", "checkpoint_path", "num_epochs"):
                if old.get(key):
                    job[key] = old[key]

    print(f"[sweep] group   : {sweep_group}")
    print(f"[sweep] log dir : {log_dir}")
    print(f"[sweep] margins : {', '.join(f'{m:g}' for m in args.margins)}")
    for index, job in enumerate(jobs, start=1):
        print(
            f"[sweep] prepared {index}/{len(jobs)}: margin={job['margin']:g} "
            f"→ {job['config_root']}",
            flush=True,
        )

    if args.dry_run:
        print("[sweep] dry run complete; nothing executed.")
        return 0

    for index, job in enumerate(jobs, start=1):
        print(
            f"\n[sweep] ===== run {index}/{len(jobs)}: margin={job['margin']:g} =====",
            flush=True,
        )
        try:
            run_pipeline_for_job(job, log_dir, sweep_group, args.skip_training)
        except Exception as exc:  # keep the remaining margins running
            job["status"] = f"failed: {exc}"
            print(f"[sweep] FAILED margin={job['margin']:g}: {exc}", flush=True)
        write_summary(jobs, log_dir, sweep_group)

    summary_path = write_summary(jobs, log_dir, sweep_group)
    print_summary(jobs, sweep_group, summary_path)

    failed = [job for job in jobs if job["status"] != "ok"]
    if failed:
        print(f"[sweep] {len(failed)} run(s) failed; inspect logs in {log_dir}")
        return 1
    print("[sweep] all runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
