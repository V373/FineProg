#!/usr/bin/env python3
"""Scale IQL (SB3-style) offline training via policy_training/train_policy.py.

Execution order mirrors the existing robomimic scale launcher:
- Outer loop over masks.
- For each mask, launch all dataset x seed jobs with bounded parallelism.

Each job writes a temporary YAML config derived from a base config and patches:
- iql.features_extractor_type = resnet18conv
- dataset.h5_path (always from reward_labeled/resnet18feats)
- dataset.filter_key
- dataset.obs_keys / normalize_obs (resnet feature setup)
- seed
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


POLICY_ROOT = Path(__file__).resolve().parent
TRAIN_ENTRY = POLICY_ROOT / "train_policy.py"
BASE_CONFIG = POLICY_ROOT / "configs" / "iql.yaml"
DATASET_DIR = POLICY_ROOT / "datasets" / "robomimic" / "can" / "mh" / "reward_labeled"
RESNET_FEAT_DIR = DATASET_DIR / "resnet18feats"

RESNET_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "agentview_image",
    "robot0_eye_in_hand_image",
]

DEFAULT_DATASET_STEMS = [
    "image_2view_v15_reward_labeled_original",
    "image_2view_v15_reward_labeled_PBRS",
]

DATASET_LABELS = {
    "image_2view_v15_reward_labeled_original": "original",
    "image_2view_v15_reward_labeled_PBRS": "pbrs",
}

DEFAULT_MASKS = [
    # "IQL_expert_half",
    "IQL_expert",
    "IQL_expert_worse",
]

DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_MAX_PARALLEL = 2
STATUS_INTERVAL = 15


_lock = threading.Lock()
_active: dict[str, float] = {}
_procs: dict[str, subprocess.Popen] = {}


def fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale launcher for policy_training IQL runs")
    parser.add_argument(
        "--base-config",
        type=str,
        default=str(BASE_CONFIG),
        help="Base YAML config path used as template for all runs",
    )
    parser.add_argument(
        "--dataset-stems",
        nargs="+",
        default=DEFAULT_DATASET_STEMS,
        help="Dataset stems under reward_labeled (without .hdf5 suffix)",
    )
    parser.add_argument(
        "--masks",
        nargs="+",
        default=DEFAULT_MASKS,
        help="Mask names to assign to dataset.filter_key",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seed list",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help="Maximum concurrent training processes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override passed to train_policy.py",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Pass --smoke to all train_policy.py runs",
    )
    parser.add_argument(
        "--strict-dataset-check",
        action="store_true",
        help="Fail if any requested dataset file is missing",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_dataset_paths(stems: list[str], strict: bool) -> list[tuple[str, str, Path]]:
    resolved: list[tuple[str, str, Path]] = []
    missing: list[Path] = []

    for stem in stems:
        ds_path = RESNET_FEAT_DIR / f"{stem}_resnet18feats.hdf5"

        if not ds_path.exists():
            missing.append(ds_path)
            log(f"[WARN] dataset not found, skip: {ds_path}")
            continue

        label = DATASET_LABELS.get(stem, stem)
        resolved.append((stem, label, ds_path))

    if strict and missing:
        first = missing[0]
        raise FileNotFoundError(
            f"Missing {len(missing)} dataset files, first missing: {first}"
        )

    if not resolved:
        raise RuntimeError("No valid dataset files found after resolution")

    return resolved


def make_temp_config(
    base_cfg: dict,
    mask: str,
    dataset_path: Path,
    seed: int,
    run_name: str,
    run_timestamp: str,
) -> Path:
    cfg = dict(base_cfg)
    cfg["iql"] = dict(base_cfg.get("iql", {}))
    cfg["dataset"] = dict(base_cfg.get("dataset", {}))
    cfg["train"] = dict(base_cfg.get("train", {}))

    # Force the feature extractor and dataset fields to match resnet18feats HDF5.
    cfg["iql"]["features_extractor_type"] = "resnet18conv"
    cfg["dataset"]["h5_path"] = str(dataset_path)
    cfg["dataset"]["filter_key"] = str(mask)
    cfg["dataset"]["obs_keys"] = list(RESNET_OBS_KEYS)
    cfg["dataset"]["normalize_obs"] = False
    cfg["seed"] = int(seed)
    cfg["scale_run_name"] = str(run_name)
    cfg["scale_run_dir_name"] = "scale_run"
    cfg["run_timestamp"] = str(run_timestamp)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f"iql_sb3_tmp_{run_name}_",
        delete=False,
        encoding="utf-8",
    )
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    return Path(tmp.name)


def _status_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(timeout=STATUS_INTERVAL):
        with _lock:
            snapshot = list(_active.keys())
        if snapshot:
            joined = ", ".join(snapshot)
            log(f"[STATUS] running ({len(snapshot)}): {joined}")


def run_one(
    base_cfg: dict,
    mask: str,
    dataset_stem: str,
    dataset_label: str,
    dataset_path: Path,
    seed: int,
    job_idx: int,
    total: int,
    device: str | None,
    smoke: bool,
    run_log_dir: Path,
    run_timestamp: str,
) -> dict:
    group_name = f"IQL__{mask}__{dataset_label}"
    run_name = f"{group_name}__seed{seed}"

    tmp_cfg = make_temp_config(base_cfg, mask, dataset_path, seed, run_name, run_timestamp)
    log_file = run_log_dir / f"{run_name}.log"

    cmd = [
        sys.executable,
        str(TRAIN_ENTRY),
        "--config",
        str(tmp_cfg),
    ]
    if device:
        cmd.extend(["--device", str(device)])
    if smoke:
        cmd.append("--smoke")

    start = time.time()
    with _lock:
        _active[run_name] = start

    log(
        f"[{job_idx}/{total}] START  {run_name}\n"
        f"          mask:    {mask}\n"
        f"          dataset: {dataset_path.name}\n"
        f"          seed:    {seed}\n"
        f"          log:     {log_file}"
    )

    try:
        with open(log_file, "w", encoding="utf-8") as out_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(POLICY_ROOT),
                env=os.environ.copy(),
                stdout=out_f,
                stderr=subprocess.STDOUT,
            )
            with _lock:
                _procs[run_name] = proc
            rc = proc.wait()
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    except Exception as exc:
        status = f"ERROR: {exc}"
    finally:
        elapsed = time.time() - start
        with _lock:
            _active.pop(run_name, None)
            _procs.pop(run_name, None)
        try:
            tmp_cfg.unlink()
        except OSError:
            pass

    log(
        f"[{job_idx}/{total}] {status}  {run_name}  seed={seed} "
        f"(elapsed {fmt_duration(elapsed)})"
    )
    return {
        "run_name": run_name,
        "status": status,
        "elapsed": elapsed,
        "log_file": str(log_file),
    }


def _sigint_handler(signum, frame) -> None:
    print("\n[INTERRUPTED] Ctrl+C received, terminating active runs...", flush=True)
    with _lock:
        procs = dict(_procs)
    for name, proc in procs.items():
        print(f"  -> terminating: {name}", flush=True)
        try:
            proc.terminate()
        except Exception:
            pass
    sys.exit(130)


def main() -> None:
    signal.signal(signal.SIGINT, _sigint_handler)
    args = parse_args()

    base_config_path = Path(args.base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    base_cfg = load_yaml(base_config_path)
    datasets = resolve_dataset_paths(list(args.dataset_stems), bool(args.strict_dataset_check))

    masks = list(args.masks)
    seeds = list(args.seeds)
    max_parallel = int(args.max_parallel)

    if max_parallel <= 0:
        raise ValueError("--max-parallel must be >= 1")

    total_runs = len(masks) * len(datasets) * len(seeds)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_log_dir = POLICY_ROOT / "outputs" / "scale_train_logs" / f"iql_scale_train_sb3_{ts}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  IQL Scale Training (SB3) - {total_runs} runs, max {max_parallel} parallel")
    print("=" * 72)
    print(f"  Base config: {base_config_path}")
    print(f"  Masks    ({len(masks)}): {', '.join(masks)}")
    print(f"  Datasets ({len(datasets)}): {', '.join(label for _, label, _ in datasets)}")
    print(f"  Seeds    ({len(seeds)}): {seeds}")
    print(f"  Run logs: {run_log_dir}")
    print()
    print("  Execution plan (mask -> dataset -> seeds, bounded parallel):")
    idx = 1
    for mask in masks:
        for _, label, ds_path in datasets:
            for seed in seeds:
                print(f"    {idx:3d}. {mask} + {label} ({ds_path.name}) seed={seed}")
                idx += 1
    print("=" * 72)
    print()

    stop_event = threading.Event()
    status_thread = threading.Thread(target=_status_loop, args=(stop_event,), daemon=True)
    status_thread.start()

    overall_start = time.time()
    results: list[dict] = []
    job_counter = 0

    for mask_idx, mask in enumerate(masks, start=1):
        log(
            f"-- Mask {mask_idx}/{len(masks)}: {mask} "
            f"({len(datasets)} datasets x {len(seeds)} seeds) --"
        )

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            future_map = {}
            for dataset_stem, dataset_label, dataset_path in datasets:
                for seed in seeds:
                    job_counter += 1
                    fut = pool.submit(
                        run_one,
                        base_cfg,
                        mask,
                        dataset_stem,
                        dataset_label,
                        dataset_path,
                        seed,
                        job_counter,
                        total_runs,
                        args.device,
                        bool(args.smoke),
                        run_log_dir,
                        ts,
                    )
                    future_map[fut] = (mask, dataset_stem, seed)

            for fut in as_completed(future_map):
                _ = future_map[fut]
                results.append(fut.result())

        log(f"-- Mask {mask_idx}/{len(masks)} complete: {mask} --")
        print()

    stop_event.set()

    total_elapsed = time.time() - overall_start
    ok = [r for r in results if r["status"] == "OK"]
    bad = [r for r in results if r["status"] != "OK"]

    print()
    print("=" * 72)
    print(
        f"  DONE - {len(ok)}/{total_runs} succeeded, "
        f"{len(bad)} failed, total elapsed {fmt_duration(total_elapsed)}"
    )
    if bad:
        print("\n  Failed runs:")
        for r in bad:
            print(f"    - {r['run_name']} [{r['status']}] log={r['log_file']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
