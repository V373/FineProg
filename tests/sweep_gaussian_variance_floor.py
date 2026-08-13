#!/usr/bin/env python3
"""Sweep Gaussian ``variance_floor`` at a fixed model-space PCA dimension.

The sweep is fixed to the TCC softmax-temperature 0.001 run requested for the
Gaussian progress experiment.  Each job invokes the public evaluation CLI for
both fitting and prediction while reading an isolated ``configs_v2`` snapshot.
The repository's shared YAML files are never modified.

Outputs are tagged with both parameters:

    .../<expert_stem>-pca16-varfloor1e-6/<timestamp>/
    .../gaussian_progress_model-...-pca16-varfloor1e-6/<query_stem>/<timestamp>/

Usage:

    conda run -n fineprog python tests/sweep_gaussian_variance_floor.py
    conda run -n fineprog python tests/sweep_gaussian_variance_floor.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


PCA_DIM = 16
VARIANCE_FLOORS = (1.0e-6, 1.0e-5, 1.0e-4)
POSTERIOR_TEMPERATURE = 1.0e4

EXPERT_EMBEDDING_REF = "tcc_can_ph_36_softmaxt_0p001_ep10k_s20260726_train"
QUERY_EMBEDDING_REFS = (
    "tcc_can_ph_36_softmaxt_0p001_ep10k_s20260726_valid",
    "tcc_can_ph_36_softmaxt_0p001_ep10k_s20260726_mh_worse",
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


def _float_label(value: float) -> str:
    """Return a compact, path-safe scientific-notation label."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def resolve_inputs() -> dict:
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


def prepare_snapshots(
    sweep_root: Path,
    variance_floors: tuple[float, ...],
) -> list[dict]:
    source_snapshot = sweep_root / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    jobs = []
    for variance_floor in variance_floors:
        floor_label = _float_label(variance_floor)
        label = f"pca{PCA_DIM}-varfloor{floor_label}"
        config_root = sweep_root / f"configs_{label}"
        shutil.copytree(source_snapshot, config_root)
        jobs.append(
            {
                "variance_floor": float(variance_floor),
                "floor_label": floor_label,
                "label": label,
                "config_root": config_root,
            }
        )
    return jobs


def _run_cli(
    command: list[str],
    config_root: Path,
    log_path: Path,
    step_name: str,
) -> str:
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


def step_gaussian_fitting(job: dict, log_dir: Path, inputs: dict) -> dict:
    config_root: Path = job["config_root"]
    fitting_path = config_root / "eval" / "gaussian_progress_fitting.yaml"
    fitting_cfg = _load_yaml(fitting_path)
    fitting_cfg["expert_h5_path"] = inputs["expert_h5_path"]
    fitting_cfg["expert_embedding_ref"] = None
    fitting_cfg["enable_pca"] = True
    fitting_cfg["pca_dim"] = PCA_DIM
    fitting_cfg["variance_floor"] = float(job["variance_floor"])
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
    tagged_model_h5_path = model_h5_path.with_name(
        f"{model_h5_path.stem}-{job['label']}{model_h5_path.suffix}"
    )
    model_h5_path.rename(tagged_model_h5_path)

    return {
        "gaussian_model_h5_path": str(tagged_model_h5_path),
        "fitting_output_dir": str(tagged_model_h5_path.parent),
        "pca_dim": PCA_DIM,
        "variance_floor": float(job["variance_floor"]),
        "num_bins": int(fitting_cfg.get("num_bins", 0)),
        "covariance_mode": fitting_cfg.get("covariance_mode"),
    }


def step_gaussian_pred(
    job: dict,
    log_dir: Path,
    gaussian_model_h5_path: str,
    query: dict,
) -> dict:
    config_root: Path = job["config_root"]
    pred_path = config_root / "eval" / "gaussian_progress_pred.yaml"
    pred_cfg = _load_yaml(pred_path)
    pred_cfg["gaussian_model_h5_path"] = str(gaussian_model_h5_path)
    pred_cfg["nonexpert_h5_path"] = str(query["h5_path"])
    pred_cfg["posterior_temperature"] = POSTERIOR_TEMPERATURE
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
        "posterior_temperature": POSTERIOR_TEMPERATURE,
        "metric_value": metric_value,
        "output_h5_path": output_h5_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Gaussian variance_floor at pca_dim=16 and Tpost=1e4."
    )
    parser.add_argument(
        "--variance-floors",
        type=float,
        nargs="+",
        default=list(VARIANCE_FLOORS),
        help=f"Variance floors to sweep (default: {list(VARIANCE_FLOORS)}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare isolated configs and print the plan without invoking the CLI.",
    )
    args = parser.parse_args()

    variance_floors = tuple(float(value) for value in args.variance_floors)
    if len(set(variance_floors)) != len(variance_floors):
        raise ValueError("[sweep] variance floors must be unique.")
    if any(not math.isfinite(value) or value <= 0.0 for value in variance_floors):
        raise ValueError("[sweep] every variance floor must be finite and > 0.")

    inputs = resolve_inputs()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_group = f"gaussian-variance-floor-sweep-pca{PCA_DIM}-{timestamp}"
    sweep_root = PROJECT_ROOT / "outputs" / "sweeps" / sweep_group
    log_dir = sweep_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    jobs = prepare_snapshots(sweep_root, variance_floors)

    print(f"[sweep] group                : {sweep_group}")
    print(f"[sweep] run_name             : {inputs['run_name']}")
    print(f"[sweep] expert_h5_path       : {inputs['expert_h5_path']}")
    print(f"[sweep] pca_dim              : {PCA_DIM}")
    print(f"[sweep] variance_floors      : {list(variance_floors)}")
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
        "pca_dim": PCA_DIM,
        "variance_floors": list(variance_floors),
        "posterior_temperature": POSTERIOR_TEMPERATURE,
        "results": [],
    }
    summary_path = sweep_root / "summary.json"

    for job in jobs:
        print(
            f"\n[sweep] ===== variance_floor = {job['variance_floor']:g} "
            f"({job['label']}) =====",
            flush=True,
        )
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
                "variance_floor": job["variance_floor"],
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
                f"[sweep] variance_floor={entry['variance_floor']:.0e}  "
                f"{prediction['embedding_ref']:<55s}  "
                f"metric={prediction['metric_value']:.6f}"
            )


if __name__ == "__main__":
    main()
