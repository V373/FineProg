#!/usr/bin/env python3
"""
iql_scale_train.py
──────────────────
Scales IQL training across 3 datasets × 5 masks = 15 training runs.

Datasets  (in .../datasets/can/mh/):
  - image_2view_v15_reward_labeled_original
  - image_2view_v15_reward_labeled_dense
  - image_2view_v15_reward_labeled_PBRS

Masks (train.hdf5_filter_key):
  - IQL_expert_half
  - IQL_expert
  - IQL_epxert_okay_halfmix          (typo preserved to match actual filter key name)
  - IQL_expert_worse_halfmix
  - IQL_expert_okay_worse_halfmix

Order: for each mask, all 3 datasets are trained to completion (≤2 parallel),
then the script advances to the next mask.

Max parallelism: 2 concurrent training processes.

Usage (from any directory):
    conda run -n fineprog python \\
        /home/user/zhangzk/projects/fineprog/scripts/robomimic/iql_scale_train.py
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

ROBOMIMIC_ROOT = Path("/home/user/zhangzk/projects/fineprog/third_party/robomimic")
BASE_CONFIG    = ROBOMIMIC_ROOT / "robomimic/exps/templates/iql.json"
DATASET_DIR    = ROBOMIMIC_ROOT / "datasets/can/mh"

DATASETS = [
    "image_2view_v15_reward_labeled_original",
    "image_2view_v15_reward_labeled_dense",
    "image_2view_v15_reward_labeled_PBRS",
]

# Short labels used in run names and log output
DATASET_LABELS = {
    "image_2view_v15_reward_labeled_original": "original",
    "image_2view_v15_reward_labeled_dense":    "dense",
    "image_2view_v15_reward_labeled_PBRS":     "PBRS",
}

MASKS = [
    # "IQL_expert_half",
    "IQL_expert",
    # "IQL_epxert_okay_halfmix",
    "IQL_expert_worse_halfmix",
    "IQL_expert_worse",
    "IQL_okay_worse",
    # "IQL_expert_okay_worse_halfmix",
]

# SEEDS = [1, 2, 3, 4, 5]
SEEDS = [1, 2, 3]
# Smoke test: temporarily set MASKS = MASKS[:1], DATASETS = DATASETS[:1], SEEDS = [1, 2]
# to verify that 2 seed runs land in the same W&B group before launching the full experiment.

MAX_PARALLEL  = 1 # Set to 1 if only have single GPU
STATUS_INTERVAL = 15  # seconds between status prints

# ── Global state (protected by _lock) ────────────────────────────────────────

_lock   = threading.Lock()
_active: dict[str, float] = {}          # run_name -> start_time (wall clock)
_procs:  dict[str, subprocess.Popen] = {}  # run_name -> Popen (for Ctrl+C cleanup)

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def make_temp_config(mask: str, dataset_stem: str, seed: int, run_name: str) -> Path:
    """Return a temp JSON with mask, dataset path, seed, and run name patched."""
    with open(BASE_CONFIG) as f:
        cfg = json.load(f)

    cfg["experiment"]["name"]       = run_name
    cfg["train"]["hdf5_filter_key"] = mask
    cfg["train"]["data"]            = str(DATASET_DIR / f"{dataset_stem}.hdf5")
    cfg["train"]["seed"]            = seed

    # Dense reward requires more training epochs to converge.
    if DATASET_LABELS.get(dataset_stem) == "dense":
        cfg["train"]["num_epochs"] = 800

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"iql_tmp_{run_name}_",
        delete=False,
    )
    json.dump(cfg, tmp, indent=4)
    tmp.close()
    return Path(tmp.name)


# ── Status printer (background thread) ───────────────────────────────────────

def _status_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(timeout=STATUS_INTERVAL):
        with _lock:
            snapshot = list(_active.keys())
        if snapshot:
            lines = ["[STATUS] Currently running:"]
            for name in snapshot:
                lines.append(f"         • {name}")
            # print("\n".join(lines), flush=True)


# ── Single training run ───────────────────────────────────────────────────────

def run_one(mask: str, dataset_stem: str, seed: int, job_idx: int, total: int) -> dict:
    """Launch one training process, block until it finishes, return result info."""
    label      = DATASET_LABELS[dataset_stem]
    group_name = f"IQL__{mask}__{label}"
    run_name   = f"{group_name}__seed{seed}"
    dataset_path = str(DATASET_DIR / f"{dataset_stem}.hdf5")

    tmp_cfg = make_temp_config(mask, dataset_stem, seed, run_name)

    cmd = [
        sys.executable,
        str(ROBOMIMIC_ROOT / "robomimic/scripts/train_speedup.py"),
        "--config", str(tmp_cfg),
        "--dataset", dataset_path,
    ]

    # Pass W&B group/name via env vars so robomimic doesn't need modification.
    # WANDB_RUN_GROUP is not set by robomimic's wandb.init(), so it's picked up
    # cleanly from the environment.  WANDB_NAME is set for completeness, though
    # log_utils.py already derives a name from config.experiment.name (which
    # already includes the seed via run_name).
    env = os.environ.copy()
    env["WANDB_RUN_GROUP"] = group_name
    env["WANDB_NAME"]      = run_name

    start = time.time()
    with _lock:
        _active[run_name] = start

    log(
        f"[{job_idx}/{total}] START  {run_name}\n"
        f"          group:   {group_name}\n"
        f"          seed:    {seed}\n"
        f"          mask:    {mask}\n"
        f"          dataset: {dataset_stem}.hdf5"
    )

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROBOMIMIC_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with _lock:
            _procs[run_name] = proc
        rc     = proc.wait()
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
        f"[{job_idx}/{total}] {status}  {run_name}  seed={seed}  "
        f"(elapsed {fmt_duration(elapsed)})"
    )

    return {"run_name": run_name, "status": status, "elapsed": elapsed}


# ── Signal handler ───────────────────────────────────────────────────────────

def _sigint_handler(signum, frame) -> None:
    print("\n[INTERRUPTED] Ctrl+C received — terminating all active training processes...", flush=True)
    with _lock:
        procs = dict(_procs)
    for name, proc in procs.items():
        print(f"  → terminating: {name}", flush=True)
        try:
            proc.terminate()
        except Exception:
            pass
    sys.exit(130)  # conventional exit code for SIGINT


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    signal.signal(signal.SIGINT, _sigint_handler)
    total_runs = len(MASKS) * len(DATASETS) * len(SEEDS)

    print("=" * 72)
    print(f"  IQL Scale Training — {total_runs} runs total, max {MAX_PARALLEL} parallel")
    print("=" * 72)
    print(f"  Masks    ({len(MASKS)}): {', '.join(MASKS)}")
    print(f"  Datasets ({len(DATASETS)}): {', '.join(DATASET_LABELS.values())}")
    print(f"  Seeds    ({len(SEEDS)}): {SEEDS}")
    print()
    print("  Execution plan (mask → dataset → seeds, ≤2 in parallel):")
    idx = 1
    for mask in MASKS:
        for ds in DATASETS:
            for seed in SEEDS:
                print(f"    {idx:3d}. {mask}  +  {DATASET_LABELS[ds]}  seed={seed}")
                idx += 1
    print("=" * 72)
    print()

    # Start background status printer
    stop_event   = threading.Event()
    status_thread = threading.Thread(
        target=_status_loop, args=(stop_event,), daemon=True
    )
    status_thread.start()

    overall_start = time.time()
    results       = []
    job_counter   = 0

    # Outer loop: one mask at a time; within each mask, all dataset×seed combos
    # are submitted together (still capped at MAX_PARALLEL concurrent processes).
    for mask_idx, mask in enumerate(MASKS, start=1):
        log(
            f"── Mask {mask_idx}/{len(MASKS)}: {mask} "
            f"({len(DATASETS)} datasets × {len(SEEDS)} seeds) ──────────────────"
        )

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {}
            for ds in DATASETS:
                for seed in SEEDS:
                    job_counter += 1
                    f = pool.submit(run_one, mask, ds, seed, job_counter, total_runs)
                    futures[f] = (mask, ds, seed)

            # collect results in completion order
            for f in futures:
                result = f.result()   # blocks until done
                results.append(result)

        log(f"── Mask {mask_idx}/{len(MASKS)} complete: {mask} ──────────────────")
        print()

    stop_event.set()

    # ── Final summary ──────────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    ok  = [r for r in results if r["status"] == "OK"]
    bad = [r for r in results if r["status"] != "OK"]

    print()
    print("=" * 72)
    print(f"  DONE — {len(ok)}/{total_runs} succeeded, "
          f"{len(bad)} failed, "
          f"total elapsed {fmt_duration(total_elapsed)}")
    if bad:
        print("\n  Failed runs:")
        for r in bad:
            print(f"    • {r['run_name']}  [{r['status']}]")
    print("=" * 72)


if __name__ == "__main__":
    main()
