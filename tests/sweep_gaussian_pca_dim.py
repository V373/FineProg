#!/usr/bin/env python3
"""Serial sweep over the Gaussian model-space ``pca_dim`` for one trained run.

Fixed run under test
--------------------
    TCC-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260726-211221-softmaxT0.001
    (registry alias ``tcc_can_ph_36_softmaxt_0p001_ep10k_s20260726``)

For each ``pca_dim`` in ``PCA_DIMS`` the script runs, strictly serially:

    1. fitting  — CLI ``evaluate_encoder.py --task gaussian_progress_fitting``
                  on the train-36 expert embeddings, with
                  ``enable_pca=true`` / ``pca_dim=<dim>``
    2. pred xN  — CLI ``evaluate_encoder.py --task gaussian_progress_pred``
                  on every query set in ``QUERY_EMBEDDING_REFS``,
                  ``posterior_temperature = 1e4``

Where the artifacts land (default outputs/ tree, tagged by pca_dim)
-------------------------------------------------------------------
    outputs/gaussian_progress_fitting/<run_name>/<expert_stem>-pca<dim>/<ts>/
    outputs/gaussian_progress_pred/<model_stem>-pca<dim>/<query_stem>/<ts>/

The fitted model H5 is renamed to carry a ``-pca<dim>`` suffix right after
fitting, which is what propagates the tag into the prediction output tree
(``gaussian_progress_pred`` derives its directory from the model H5 stem).

Config isolation
----------------
``configs_v2`` is snapshotted once at startup and one isolated copy is created
per pca_dim.  Only these keys are overridden in the copy:

    eval/gaussian_progress_fitting.yaml :: expert_h5_path / expert_embedding_ref /
                                           enable_pca / pca_dim / output_dir
    eval/gaussian_progress_pred.yaml    :: gaussian_model_h5_path /
                                           nonexpert_h5_path /
                                           posterior_temperature

The repository's own YAML files are never modified.  CLI subprocesses read the
snapshot through the ``FINEPROG_CONFIGS_V2_DIR`` environment variable, which is
honoured by ``utils/config_v2.ConfigV2``.

Usage (from the fineprog conda environment)
-------------------------------------------
    conda run -n fineprog python tests/sweep_gaussian_pca_dim.py
    conda run -n fineprog python tests/sweep_gaussian_pca_dim.py --dry-run
    conda run -n fineprog python tests/sweep_gaussian_pca_dim.py --pca-dims 64 32
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
PCA_DIMS = (64, 32, 16, 8)
POSTERIOR_TEMPERATURE = 1.0e4

# Registry keys from configs_v2/registry/runs.yaml (embeddings section)
EXPERT_EMBEDDING_REF = "tcc_can_ph_36_softmaxt_0p001_ep10k_s20260726_train"
QUERY_EMBEDDING_REFS = (
    "tcc_can_ph_36_softmaxt_0p001_ep10k_s20260726_valid",       # valid-4
    "tcc_can_ph_36_softmaxt_0p001_ep10k_s20260726_mh_worse",    # worse-100
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
CONFIGS_ENV_VAR = "FINEPROG_CONFIGS_V2_DIR"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Config snapshot preparation
# ---------------------------------------------------------------------------

def resolve_inputs() -> dict:
    """Resolve the fixed expert / query embedding H5 paths from the registry."""
    from utils.config_v2 import ConfigV2

    cfg_v2 = ConfigV2()
    expert = cfg_v2.resolve_embedding(EXPERT_EMBEDDING_REF)
    expert_h5_path = Path(expert["embedding_h5_path"])
    if not expert_h5_path.is_file():
        raise FileNotFoundError(f"[sweep] expert embeddings not found: {expert_h5_path}")

    queries = []
    for ref in QUERY_EMBEDDING_REFS:
        entry = cfg_v2.resolve_embedding(ref)
        query_h5_path = Path(entry["embedding_h5_path"])
        if not query_h5_path.is_file():
            raise FileNotFoundError(
                f"[sweep] query embeddings not found for {ref}: {query_h5_path}"
            )
        queries.append({"embedding_ref": ref, "h5_path": str(query_h5_path)})

    return {
        "run_name": expert["run_name"],
        "expert_h5_path": str(expert_h5_path),
        "expert_h5_stem": expert_h5_path.stem,
        "fitting_root": str(
            Path(cfg_v2._dirs["outputs"])
            / "gaussian_progress_fitting"
            / expert["run_name"]
        ),
        "queries": queries,
    }


def prepare_snapshots(sweep_root: Path, pca_dims: tuple[int, ...]) -> list[dict]:
    """Snapshot configs_v2 once, then derive one isolated config per pca_dim."""
    source_snapshot = sweep_root / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    jobs = []
    for pca_dim in pca_dims:
        config_root = sweep_root / f"configs_pca{pca_dim}"
        shutil.copytree(source_snapshot, config_root)
        jobs.append(
            {
                "pca_dim": int(pca_dim),
                "label": f"pca{int(pca_dim)}",
                "config_root": config_root,
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
# Pipeline steps (per pca_dim)
# ---------------------------------------------------------------------------

def step_gaussian_fitting(
    job: dict,
    log_dir: Path,
    inputs: dict,
) -> dict:
    """Fit the progress-conditioned Gaussians at one model-space PCA dim."""
    config_root: Path = job["config_root"]
    fitting_path = config_root / "eval" / "gaussian_progress_fitting.yaml"
    fitting_cfg = _load_yaml(fitting_path)
    fitting_cfg["expert_h5_path"] = inputs["expert_h5_path"]
    fitting_cfg["expert_embedding_ref"] = None
    fitting_cfg["enable_pca"] = True
    fitting_cfg["pca_dim"] = int(job["pca_dim"])
    # Default outputs/ tree, with the pca_dim baked into the leaf directory.
    fitting_cfg["output_dir"] = str(
        Path(inputs["fitting_root"])
        / f"{inputs['expert_h5_stem']}-{job['label']}"
    )
    _dump_yaml(fitting_path, fitting_cfg)

    output = _run_cli(
        [sys.executable, "evaluate_encoder.py", "--task", "gaussian_progress_fitting"],
        config_root,
        log_dir / f"01_fitting_{job['label']}.log",
        f"gaussian_progress_fitting[{job['label']}]",
    )
    model_h5_path = Path(
        _search_last(
            r"\[gaussian_progress_fitting\] output_h5_path\s*:\s*(\S+)",
            output,
            "gaussian_progress_fitting",
        )
    )
    # The prediction task derives its output directory from the model H5 stem,
    # so tag the filename to keep the pred artifacts distinguishable too.
    tagged_model_h5_path = model_h5_path.with_name(
        f"{model_h5_path.stem}-{job['label']}{model_h5_path.suffix}"
    )
    model_h5_path.rename(tagged_model_h5_path)

    return {
        "gaussian_model_h5_path": str(tagged_model_h5_path),
        "fitting_output_dir": str(tagged_model_h5_path.parent),
        "num_bins": int(fitting_cfg.get("num_bins", 0)),
        "covariance_mode": fitting_cfg.get("covariance_mode"),
    }


def step_gaussian_pred(
    job: dict,
    log_dir: Path,
    gaussian_model_h5_path: str,
    query: dict,
) -> dict:
    """Run one online prediction pass against the fitted Gaussian model."""
    config_root: Path = job["config_root"]
    pred_path = config_root / "eval" / "gaussian_progress_pred.yaml"
    pred_cfg = _load_yaml(pred_path)
    pred_cfg["gaussian_model_h5_path"] = str(gaussian_model_h5_path)
    pred_cfg["nonexpert_h5_path"] = str(query["h5_path"])
    pred_cfg["posterior_temperature"] = float(POSTERIOR_TEMPERATURE)
    _dump_yaml(pred_path, pred_cfg)

    query_stem = Path(query["h5_path"]).stem
    output = _run_cli(
        [sys.executable, "evaluate_encoder.py", "--task", "gaussian_progress_pred"],
        config_root,
        log_dir / f"02_pred_{job['label']}_{query_stem}.log",
        f"gaussian_progress_pred[{job['label']}/{query_stem}]",
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
        "embedding_ref": query["embedding_ref"],
        "nonexpert_h5_path": str(query["h5_path"]),
        "posterior_temperature": float(POSTERIOR_TEMPERATURE),
        "metric_value": metric_value,
        "output_h5_path": output_h5_path,
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep gaussian_progress_fitting pca_dim + prediction at T=1e4."
    )
    parser.add_argument(
        "--pca-dims", type=int, nargs="+", default=list(PCA_DIMS),
        help=f"Model-space PCA dims to sweep (default: {list(PCA_DIMS)}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Prepare config snapshots and print the plan without running any CLI.",
    )
    args = parser.parse_args()

    pca_dims = tuple(int(d) for d in args.pca_dims)
    inputs = resolve_inputs()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_group = f"gaussian-pca-dim-sweep-{timestamp}"
    sweep_root = PROJECT_ROOT / "outputs" / "sweeps" / sweep_group
    log_dir = sweep_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)

    jobs = prepare_snapshots(sweep_root, pca_dims)

    print(f"[sweep] group                : {sweep_group}")
    print(f"[sweep] run_name             : {inputs['run_name']}")
    print(f"[sweep] expert_h5_path       : {inputs['expert_h5_path']}")
    print(f"[sweep] pca_dims             : {list(pca_dims)}")
    print(f"[sweep] posterior_temperature: {POSTERIOR_TEMPERATURE:g}")
    for query in inputs["queries"]:
        print(f"[sweep] query                : {query['embedding_ref']} → {query['h5_path']}")
    print(f"[sweep] sweep_root           : {sweep_root}")

    if args.dry_run:
        print("[sweep] dry-run: config snapshots prepared, no CLI executed.")
        return

    summary = {
        "sweep_group": sweep_group,
        "run_name": inputs["run_name"],
        "expert_h5_path": inputs["expert_h5_path"],
        "posterior_temperature": POSTERIOR_TEMPERATURE,
        "pca_dims": list(pca_dims),
        "results": [],
    }
    summary_path = sweep_root / "summary.json"

    for job in jobs:
        print(f"\n[sweep] ===== pca_dim = {job['pca_dim']} =====", flush=True)
        fitting = step_gaussian_fitting(job, log_dir, inputs)
        print(f"[sweep]   model H5: {fitting['gaussian_model_h5_path']}", flush=True)

        predictions = [
            step_gaussian_pred(
                job, log_dir, fitting["gaussian_model_h5_path"], query
            )
            for query in inputs["queries"]
        ]

        summary["results"].append(
            {
                "pca_dim": job["pca_dim"],
                "label": job["label"],
                "config_root": str(job["config_root"]),
                "fitting": fitting,
                "predictions": predictions,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[sweep] summary written to {summary_path}")
    for entry in summary["results"]:
        for prediction in entry["predictions"]:
            print(
                f"[sweep] pca_dim={entry['pca_dim']:>3d}  "
                f"{prediction['embedding_ref']:<55s}  "
                f"metric={prediction['metric_value']:.6f}"
            )


if __name__ == "__main__":
    main()
