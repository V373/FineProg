#!/usr/bin/env python3
"""Serial fruit Gaussian-progress ``num_bins`` sweep.

This runner snapshots the current ``configs_v2`` tree into a temporary
directory and runs the existing ``evaluate_encoder.py`` CLI once per job.  It
does not modify repository YAML files.  The t-SNE Gaussian sample budget is
fixed at 4000 synthetic points per run:

* 10 bins -> 400 samples per bin
* 5 bins  -> 800 samples per bin

The source configuration currently contains ``gaussian_samples_per_bin: 50``;
that value remains untouched in the repository snapshot source, but each
temporary job config receives the derived value above so that the total budget
is comparable across ``num_bins``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
CONFIGS_ENV_VAR = "FINEPROG_CONFIGS_V2_DIR"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gaussian_progress_fitting" / "fruit_expert_sweep"
TARGET_SYNTHETIC_SAMPLES = 4000
NUM_BINS = (10, 5)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VARIANTS: dict[str, dict] = {
    "tcc_d128": {
        "expert_embedding_ref": "fruit_tcc_d128_train36_ep20k",
        "enable_pca": True,
        "input_embedding_dim": 128,
    },
    "tcc_triplet_w0p5_d128": {
        "expert_embedding_ref": "fruit_tcc_triplet_w0p5_d128_train36_ep20k",
        "enable_pca": True,
        "input_embedding_dim": 128,
    },
    "tcc_d32": {
        "expert_embedding_ref": "fruit_tcc_d32_train36_ep20k",
        "enable_pca": False,
        "input_embedding_dim": 32,
    },
    "tcc_triplet_w0p5_d32": {
        "expert_embedding_ref": "fruit_tcc_triplet_w0p5_d32_train36_ep20k",
        "enable_pca": False,
        "input_embedding_dim": 32,
    },
}


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _samples_per_bin(num_bins: int) -> int:
    if num_bins < 2 or TARGET_SYNTHETIC_SAMPLES % num_bins != 0:
        raise ValueError(
            f"TARGET_SYNTHETIC_SAMPLES={TARGET_SYNTHETIC_SAMPLES} must be "
            f"divisible by num_bins={num_bins}."
        )
    samples_per_bin = TARGET_SYNTHETIC_SAMPLES // num_bins
    if samples_per_bin < 20:
        raise ValueError(
            "Derived gaussian_samples_per_bin must satisfy the task minimum "
            f"of 20; got {samples_per_bin}."
        )
    return samples_per_bin


def _resolve_inputs() -> dict[str, dict]:
    """Resolve registry refs and validate the expected fruit H5 inputs."""
    from utils.config_v2 import ConfigV2

    config = ConfigV2(SOURCE_CONFIG_ROOT)
    inputs: dict[str, dict] = {}
    for label, variant in VARIANTS.items():
        embedding_ref = variant["expert_embedding_ref"]
        entry = config.resolve_embedding(embedding_ref)
        h5_path = Path(entry["embedding_h5_path"])
        if not h5_path.is_file():
            raise FileNotFoundError(
                f"[{label}] resolved expert embedding H5 does not exist: {h5_path}"
            )

        with h5py.File(h5_path, "r") as h5_file:
            if "videos" not in h5_file or not isinstance(h5_file["videos"], h5py.Group):
                raise ValueError(f"[{label}] H5 is missing /videos: {h5_path}")
            videos = h5_file["videos"]
            video_ids = list(videos.keys())
            feature_count = 0
            embedding_dim = None
            for video_id in video_ids:
                if "embeddings" not in videos[video_id]:
                    raise ValueError(
                        f"[{label}] video {video_id!r} is missing embeddings: {h5_path}"
                    )
                dataset = videos[video_id]["embeddings"]
                if len(dataset.shape) != 2:
                    raise ValueError(
                        f"[{label}] video {video_id!r} embeddings are not 2D: "
                        f"{dataset.shape}"
                    )
                feature_count += int(dataset.shape[0])
                current_dim = int(dataset.shape[1])
                if embedding_dim is None:
                    embedding_dim = current_dim
                elif embedding_dim != current_dim:
                    raise ValueError(
                        f"[{label}] inconsistent embedding dimensions in {h5_path}"
                    )

        if len(video_ids) != 36:
            raise ValueError(
                f"[{label}] expected 36 expert videos, found {len(video_ids)}"
            )
        if feature_count != 7315:
            raise ValueError(
                f"[{label}] expected 7315 expert features, found {feature_count}"
            )
        expected_dim = int(variant["input_embedding_dim"])
        if embedding_dim != expected_dim:
            raise ValueError(
                f"[{label}] expected input dimension {expected_dim}, "
                f"found {embedding_dim}"
            )

        inputs[label] = {
            "expert_embedding_ref": embedding_ref,
            "expert_h5_path": str(h5_path),
            "num_expert_videos": len(video_ids),
            "num_expert_features": feature_count,
            "input_embedding_dim": embedding_dim,
        }
    return inputs


def _snapshot_files(root: Path) -> dict[str, str]:
    """Hash pre-existing files so later validation can detect overwrites."""
    if not root.exists():
        return {}
    snapshot: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        snapshot[str(path)] = digest.hexdigest()
    return snapshot


def _assert_snapshot_unchanged(snapshot: dict[str, str]) -> None:
    for path_string, expected_digest in snapshot.items():
        path = Path(path_string)
        if not path.is_file():
            raise AssertionError(f"Pre-existing output disappeared: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise AssertionError(f"Pre-existing output was modified: {path}")


def _prepare_job_config(
    source_snapshot: Path,
    job_root: Path,
    label: str,
    num_bins: int,
    variant: dict,
    inputs: dict,
) -> dict:
    config_root = job_root / "configs_v2"
    shutil.copytree(source_snapshot, config_root)

    fitting_path = config_root / "eval" / "gaussian_progress_fitting.yaml"
    fitting_config = _read_yaml(fitting_path)
    samples_per_bin = _samples_per_bin(num_bins)
    output_dir = OUTPUT_ROOT / label / f"num_bins_{num_bins}"

    fitting_config["expert_embedding_ref"] = inputs["expert_embedding_ref"]
    fitting_config["expert_h5_path"] = inputs["expert_h5_path"]
    fitting_config["num_bins"] = int(num_bins)
    fitting_config["enable_pca"] = bool(variant["enable_pca"])
    fitting_config["output_dir"] = str(output_dir)
    tsne_viz = dict(fitting_config.get("tsne_viz") or {})
    tsne_viz["gaussian_samples_per_bin"] = samples_per_bin
    fitting_config["tsne_viz"] = tsne_viz
    _write_yaml(fitting_path, fitting_config)

    return {
        "label": label,
        "num_bins": int(num_bins),
        "samples_per_bin": samples_per_bin,
        "expected_synthetic_samples": TARGET_SYNTHETIC_SAMPLES,
        "config_root": str(config_root),
        "output_dir": str(output_dir),
        **inputs,
        "enable_pca": bool(variant["enable_pca"]),
        "expected_input_embedding_dim": int(variant["input_embedding_dim"]),
        "expected_embedding_dim": 32,
        "status": "pending",
    }


def _run_cli(job: dict, log_path: Path) -> str:
    env = os.environ.copy()
    env[CONFIGS_ENV_VAR] = job["config_root"]
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "evaluate_encoder.py",
        "--task",
        "gaussian_progress_fitting",
    ]
    print(f"[sweep] running {job['label']} bins={job['num_bins']}", flush=True)
    print(f"[sweep]   command: {' '.join(command)}", flush=True)
    print(f"[sweep]   log: {log_path}", flush=True)

    captured: list[str] = []
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
        assert process.stdout is not None
        for line in process.stdout:
            captured.append(line)
            log_file.write(line)
            log_file.flush()
            print(line, end="", flush=True)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"CLI failed for {job['label']} bins={job['num_bins']} with exit "
            f"code {return_code}; see {log_path}"
        )
    return "".join(captured)


def _last_match(pattern: str, output: str, description: str) -> str:
    matches = re.findall(pattern, output)
    if not matches:
        raise AssertionError(f"Could not find {description} in CLI output")
    match = matches[-1]
    return match if isinstance(match, str) else match[0]


def _validate_job_output(job: dict, output: str, log_path: Path) -> dict:
    expected_counts = [job["samples_per_bin"]] * job["num_bins"]
    counts_text = _last_match(
        r"Gaussian samples per bin:\s*(\[[^\n]+\])",
        output,
        "Gaussian samples per bin",
    )
    try:
        actual_counts = [int(value) for value in ast.literal_eval(counts_text)]
    except (SyntaxError, ValueError, TypeError) as exc:
        raise AssertionError(f"Invalid Gaussian sample-count list: {counts_text}") from exc
    if actual_counts != expected_counts:
        raise AssertionError(
            f"Expected Gaussian sample counts {expected_counts}, got {actual_counts}"
        )
    if sum(actual_counts) != TARGET_SYNTHETIC_SAMPLES:
        raise AssertionError(
            f"Expected {TARGET_SYNTHETIC_SAMPLES} synthetic samples, "
            f"got {sum(actual_counts)}"
        )

    synthetic_text = _last_match(
        r"synthetic=(\d+)", output, "synthetic t-SNE point count"
    )
    synthetic_count = int(synthetic_text)
    if synthetic_count != TARGET_SYNTHETIC_SAMPLES:
        raise AssertionError(
            f"Expected runtime synthetic count {TARGET_SYNTHETIC_SAMPLES}, "
            f"got {synthetic_count}"
        )

    model_path = Path(
        _last_match(
            r"\[gaussian_progress_fitting\] output_h5_path\s*:\s*(\S+)",
            output,
            "output_h5_path",
        )
    )
    visualization_path = Path(
        _last_match(
            r"\[gaussian_progress_fitting\] output_visualization_path\s*:\s*(\S+)",
            output,
            "output_visualization_path",
        )
    )
    if not model_path.is_file():
        raise AssertionError(f"Output H5 does not exist: {model_path}")
    if not visualization_path.is_file() or visualization_path.stat().st_size <= 0:
        raise AssertionError(f"Output visualization is missing or empty: {visualization_path}")
    expected_output_root = Path(job["output_dir"]).resolve()
    if expected_output_root not in model_path.resolve().parents:
        raise AssertionError(
            f"Output H5 {model_path} is outside expected directory {expected_output_root}"
        )

    with h5py.File(model_path, "r") as h5_file:
        attrs = h5_file.attrs
        checks = {
            "task_name": "gaussian_progress_fitting",
            "num_bins": job["num_bins"],
            "num_expert_videos": 36,
            "num_expert_features": 7315,
            "input_embedding_dim": job["expected_input_embedding_dim"],
            "embedding_dim": job["expected_embedding_dim"],
            "covariance_mode": "independent",
        }
        for name, expected in checks.items():
            actual = attrs.get(name)
            if actual != expected:
                raise AssertionError(
                    f"{model_path}: attr {name} expected {expected!r}, got {actual!r}"
                )
        if bool(attrs.get("enable_pca")) != job["enable_pca"]:
            raise AssertionError(
                f"{model_path}: enable_pca expected {job['enable_pca']}, "
                f"got {attrs.get('enable_pca')!r}"
            )
        if "model" not in h5_file or "bin_counts" not in h5_file["model"]:
            raise AssertionError(f"{model_path}: missing /model/bin_counts")
        bin_counts = [int(value) for value in h5_file["model"]["bin_counts"][:]]
        if len(bin_counts) != job["num_bins"] or sum(bin_counts) != 7315:
            raise AssertionError(
                f"{model_path}: invalid fitted bin counts {bin_counts}"
            )

    return {
        "log_path": str(log_path),
        "model_h5_path": str(model_path),
        "visualization_path": str(visualization_path),
        "gaussian_sample_counts": actual_counts,
        "num_synthetic_points": synthetic_count,
        "model_attrs": {
            "num_bins": job["num_bins"],
            "enable_pca": job["enable_pca"],
            "input_embedding_dim": job["expected_input_embedding_dim"],
            "embedding_dim": job["expected_embedding_dim"],
            "num_expert_videos": 36,
            "num_expert_features": 7315,
        },
    }


def _write_summary(path: Path, sweep_id: str, jobs: list[dict], status: str) -> None:
    path.write_text(
        json.dumps(
            {
                "sweep_id": sweep_id,
                "status": status,
                "target_synthetic_samples": TARGET_SYNTHETIC_SAMPLES,
                "num_bins": list(NUM_BINS),
                "jobs": jobs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Serial fruit Gaussian-progress num_bins sweep with a fixed "
            "4000-point t-SNE Gaussian sample budget."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare isolated configs and preflight inputs without running the CLI.",
    )
    args = parser.parse_args()

    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing configs_v2 directory: {SOURCE_CONFIG_ROOT}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_id = f"num_bins_{timestamp}"
    log_dir = OUTPUT_ROOT / f"logs_num_bins_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=False)
    summary_path = log_dir / "summary.json"
    preexisting_outputs = _snapshot_files(OUTPUT_ROOT)
    jobs: list[dict] = []

    print(f"[sweep] sweep id                 : {sweep_id}")
    print(f"[sweep] output root              : {OUTPUT_ROOT}")
    print(f"[sweep] log directory             : {log_dir}")
    print(f"[sweep] target synthetic samples : {TARGET_SYNTHETIC_SAMPLES}")
    print("[sweep] scheduling                : serial")

    try:
        inputs = _resolve_inputs()
        with tempfile.TemporaryDirectory(prefix="fineprog_fruit_gaussian_") as temp_dir:
            temp_root = Path(temp_dir)
            source_snapshot = temp_root / "configs_source_snapshot"
            shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

            for label, variant in VARIANTS.items():
                for num_bins in NUM_BINS:
                    job_root = temp_root / f"{label}_bins{num_bins}"
                    job = _prepare_job_config(
                        source_snapshot=source_snapshot,
                        job_root=job_root,
                        label=label,
                        num_bins=num_bins,
                        variant=variant,
                        inputs=inputs[label],
                    )
                    jobs.append(job)

            print("[sweep] jobs:")
            for index, job in enumerate(jobs, start=1):
                print(
                    f"[sweep]   {index:02d}. {job['label']:<28} "
                    f"num_bins={job['num_bins']} "
                    f"gaussian_samples_per_bin={job['samples_per_bin']} "
                    f"total={job['expected_synthetic_samples']}"
                )

            _write_summary(summary_path, sweep_id, jobs, "dry_run" if args.dry_run else "prepared")
            if args.dry_run:
                print(f"[sweep] dry-run complete; summary: {summary_path}")
                return 0

            for index, job in enumerate(jobs, start=1):
                log_path = log_dir / f"{index:02d}_{job['label']}_bins{job['num_bins']}.log"
                job["status"] = "running"
                _write_summary(summary_path, sweep_id, jobs, "running")
                output = _run_cli(job, log_path)
                job["validation"] = _validate_job_output(job, output, log_path)
                job["status"] = "complete"
                _write_summary(summary_path, sweep_id, jobs, "running")
                print(
                    f"[sweep] complete {job['label']} bins={job['num_bins']} "
                    f"synthetic={TARGET_SYNTHETIC_SAMPLES}",
                    flush=True,
                )

        _assert_snapshot_unchanged(preexisting_outputs)
        _write_summary(summary_path, sweep_id, jobs, "complete")
        print(f"[sweep] all {len(jobs)} jobs completed")
        print(f"[sweep] summary: {summary_path}")
        return 0
    except Exception as exc:
        for job in jobs:
            if job.get("status") == "running":
                job["status"] = "failed"
                job["error"] = str(exc)
        _write_summary(summary_path, sweep_id, jobs, "failed")
        print(f"[sweep] FAILED: {exc}", file=sys.stderr)
        print(f"[sweep] partial summary: {summary_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
