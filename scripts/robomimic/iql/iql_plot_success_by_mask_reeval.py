#!/usr/bin/env python3
"""
iql_plot_success_by_mask_reeval.py
──────────────────────────────────
Re-evaluate historical IQL checkpoints via run_trained_agent.py and plot
success-rate curves by mask.

Compared with iql_plot_success_by_mask.py:
- does NOT read success from wandb / log files
- evaluates each checkpoint by invoking run_trained_agent.py
- uses evaluated Success_Rate to build historical SR curves

For each mask:
- one figure is generated
- each figure contains 3 reward labels (original / dense / PBRS)
- each label curve is mean across seeds
- confidence interval shading is computed across seeds

Run:
    conda run -n fineprog python \
      /home/user/zhangzk/projects/fineprog/scripts/robomimic/iql/iql_plot_success_by_mask_reeval.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# -- Configuration -------------------------------------------------------------

ROBOMIMIC_ROOT = Path("/home/user/zhangzk/projects/fineprog/third_party/robomimic")
RUN_AGENT_SCRIPT = ROBOMIMIC_ROOT / "robomimic/scripts/run_trained_agent.py"
IQL_MODELS_ROOT = Path("/home/user/zhangzk/projects/fineprog/third_party/robomimic/iql_trained_models")
OUTPUT_ROOT = Path("/home/user/zhangzk/projects/fineprog/outputs/robomimic/iql")

TASK_NAME = "can"
TASK_DIR_NAME = "PickPlaceCan"

MASKS = [
    # "IQL_expert_half",
    "IQL_expert",
    # "IQL_expert_worse_halfmix",
    # "IQL_expert_worse",
    # "IQL_okay_worse",
]

REWARD_LABELS = ["original", "dense", "PBRS"]
SEEDS = [1, 2, 3]

# Re-eval settings
EVAL_N_ROLLOUTS = 100
EVAL_HORIZON = 400
EVAL_ENV_NAME = None  # e.g. "PickPlaceCan"
EVAL_SEED = 0
MAX_PARALLEL_EVALS = 5

# Checkpoint epoch schedule (hard-coded)
CKPT_EPOCH_INTERVAL = 100
CKPT_TOTAL_EPOCHS = 500

# Confidence interval setup: mean ± Z * SEM
CI_Z = 1.96
FIG_SIZE = (9, 5)
DPI = 300

LABEL_COLORS = {
    "original": "#1f77b4",
    "dense": "#d62728",
    "PBRS": "#2ca02c",
}

# -- Helpers ------------------------------------------------------------------

# Match `model_epoch_<N>[_...].pth`. Note: the optional suffix group must consume
# the underscore, otherwise the leading `(\d+)` is forced to backtrack and a
# bare `model_epoch_500.pth` is mis-parsed as epoch=50 (it would only match
# `model_epoch_50` and leave `0` for the trailing `.+`).
CKPT_EPOCH_RE = re.compile(r"^model_epoch_(\d+)(?:_.*)?\.pth$")
JSON_BLOCK_RE = re.compile(r"Average Rollout Stats\s*(\{.*?\})", re.DOTALL)


def log(msg: str) -> None:
    print(msg, flush=True)


def _latest_timestamp_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    cands = [p for p in root.iterdir() if p.is_dir()]
    if not cands:
        return None
    return sorted(cands)[-1]


def _find_latest_run_dir(mask: str, label: str, seed: int) -> Path | None:
    run_root = IQL_MODELS_ROOT / f"IQL__{mask}__{label}__seed{seed}"
    if not run_root.exists():
        return None

    task_root = run_root / TASK_DIR_NAME / label
    if not task_root.exists():
        return None

    return _latest_timestamp_dir(task_root)


def _extract_epoch_from_ckpt_name(name: str) -> int | None:
    m = CKPT_EPOCH_RE.match(name)
    if not m:
        return None
    ep = m.group(1)
    return int(ep) if ep is not None else None


def _required_epochs() -> List[int]:
    if CKPT_EPOCH_INTERVAL <= 0:
        raise ValueError("CKPT_EPOCH_INTERVAL must be > 0")
    if CKPT_TOTAL_EPOCHS <= 0:
        raise ValueError("CKPT_TOTAL_EPOCHS must be > 0")
    return list(range(CKPT_EPOCH_INTERVAL, CKPT_TOTAL_EPOCHS + 1, CKPT_EPOCH_INTERVAL))


def _ckpt_priority(name: str, epoch: int) -> Tuple[int, int, str]:
    # Prefer exact model_epoch_<N>.pth; otherwise choose deterministic fallback.
    exact_name = f"model_epoch_{epoch}.pth"
    exact_rank = 0 if name == exact_name else 1
    return (exact_rank, len(name), name)


def _collect_checkpoints(
    run_dir: Path,
    required_epochs: List[int],
) -> Tuple[List[Tuple[int, Path]], List[str], List[int], Dict[int, List[str]]]:
    models_dir = run_dir / "models"
    if not models_dir.exists():
        return [], [], required_epochs, {}

    required_set = set(required_epochs)
    epoch_to_candidates: Dict[int, List[Path]] = {}
    ignored_non_exact: List[str] = []

    for p in models_dir.glob("model_epoch_*.pth"):
        ep = _extract_epoch_from_ckpt_name(p.name)
        if ep is None:
            ignored_non_exact.append(p.name)
            continue
        if ep not in required_set:
            continue
        epoch_to_candidates.setdefault(ep, []).append(p)

    duplicate_choices: Dict[int, List[str]] = {}
    selected: List[Tuple[int, Path]] = []
    for ep in required_epochs:
        cands = epoch_to_candidates.get(ep, [])
        if not cands:
            continue
        cands_sorted = sorted(cands, key=lambda x: _ckpt_priority(x.name, ep))
        selected.append((ep, cands_sorted[0]))
        if len(cands_sorted) > 1:
            duplicate_choices[ep] = [c.name for c in cands_sorted]

    missing_epochs = [ep for ep in required_epochs if ep not in epoch_to_candidates]
    return selected, ignored_non_exact, missing_epochs, duplicate_choices


def _run_agent_eval(ckpt_path: Path) -> Tuple[float | None, str | None]:
    cmd = [
        sys.executable,
        str(RUN_AGENT_SCRIPT),
        "--agent",
        str(ckpt_path),
        "--n_rollouts",
        str(EVAL_N_ROLLOUTS),
        "--horizon",
        str(EVAL_HORIZON),
        "--seed",
        str(EVAL_SEED),
    ]
    if EVAL_ENV_NAME is not None:
        cmd.extend(["--env", str(EVAL_ENV_NAME)])

    proc = subprocess.run(
        cmd,
        cwd=str(ROBOMIMIC_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        return None, f"eval failed (rc={proc.returncode})"

    m = JSON_BLOCK_RE.search(proc.stdout)
    if m is None:
        return None, "cannot parse rollout stats"

    try:
        stats = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None, "malformed rollout stats json"

    success = stats.get("Success_Rate", None)
    if success is None:
        return None, "Success_Rate missing in stats"

    try:
        return float(success), None
    except (TypeError, ValueError):
        return None, "invalid Success_Rate value"


def _stack_seed_series(seed_series: List[Dict[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (epochs, matrix[seed, epoch_idx]) with NaN for missing points."""
    all_epochs = sorted({e for d in seed_series for e in d.keys()})
    if not all_epochs:
        return np.array([], dtype=np.int32), np.empty((0, 0), dtype=np.float64)

    mat = np.full((len(seed_series), len(all_epochs)), np.nan, dtype=np.float64)
    epoch_to_idx = {e: i for i, e in enumerate(all_epochs)}

    for row, series in enumerate(seed_series):
        for ep, val in series.items():
            mat[row, epoch_to_idx[ep]] = val

    return np.array(all_epochs, dtype=np.int32), mat


def _mean_ci(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and 95% CI per column, handling NaN."""
    if mat.size == 0:
        return np.array([]), np.array([]), np.array([])

    mean = np.nanmean(mat, axis=0)
    count = np.sum(~np.isnan(mat), axis=0)

    std = np.full_like(mean, np.nan, dtype=np.float64)
    for i in range(mat.shape[1]):
        col = mat[:, i]
        col = col[~np.isnan(col)]
        if col.size >= 2:
            std[i] = np.std(col, ddof=1)
        elif col.size == 1:
            std[i] = 0.0

    sem = np.divide(std, np.sqrt(np.maximum(count, 1)), where=~np.isnan(std))
    ci = CI_Z * sem

    lower = np.clip(mean - ci, 0.0, 1.0)
    upper = np.clip(mean + ci, 0.0, 1.0)
    return mean, lower, upper


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


# -- Main plotting flow --------------------------------------------------------

def main() -> None:
    out_dir = OUTPUT_ROOT / TASK_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log(f"IQL Success Re-eval Plotter | task={TASK_NAME}")
    log(f"masks={MASKS}")
    log(f"labels={REWARD_LABELS}, seeds={SEEDS}")
    log(f"n_rollouts={EVAL_N_ROLLOUTS}, horizon={EVAL_HORIZON}, eval_env={EVAL_ENV_NAME}")
    log(f"ckpt_epoch_interval={CKPT_EPOCH_INTERVAL}, ckpt_total_epochs={CKPT_TOTAL_EPOCHS}")
    log(f"max_parallel_evals={MAX_PARALLEL_EVALS}")
    log(f"output_dir={out_dir}")
    log("=" * 80)

    required_epochs = _required_epochs()
    log(f"required_epochs={required_epochs}")

    for mask in MASKS:
        log(f"\n[mask] {mask}")

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        plotted_any = False

        for label in REWARD_LABELS:
            per_seed_series: List[Dict[int, float]] = []
            missing = []
            jobs: List[Tuple[int, int, Path]] = []
            seed_to_ep2sr: Dict[int, Dict[int, float]] = {s: {} for s in SEEDS}
            seed_to_expected_epochs: Dict[int, List[int]] = {s: [] for s in SEEDS}

            for seed in SEEDS:
                run_dir = _find_latest_run_dir(mask, label, seed)
                if run_dir is None:
                    missing.append(seed)
                    continue

                ckpts, ignored_non_exact, missing_model_epochs, duplicate_choices = _collect_checkpoints(
                    run_dir=run_dir,
                    required_epochs=required_epochs,
                )
                if ignored_non_exact:
                    log(
                        f"  [warning] label={label}, seed={seed}: ignored unparsable ckpts "
                        f"(example: {ignored_non_exact[:3]})"
                    )
                if duplicate_choices:
                    for ep in sorted(duplicate_choices.keys()):
                        log(
                            f"  [warning] label={label}, seed={seed}: epoch={ep} has multiple ckpts; "
                            f"picked {duplicate_choices[ep][0]}"
                        )
                if missing_model_epochs:
                    log(
                        f"  [warning] label={label}, seed={seed}: missing model ckpts "
                        f"for epochs={missing_model_epochs} (will be ignored in mean/var)"
                    )
                if not ckpts:
                    log(
                        f"  [warning] label={label}, seed={seed}: no usable ckpts found on required epochs; "
                        f"skip this seed"
                    )
                    missing.append(seed)
                    continue

                seed_to_expected_epochs[seed] = [ep for ep, _ in ckpts]
                log(f"  - label={label}, seed={seed}: queued {len(ckpts)} checkpoints")
                for ep, ckpt_path in ckpts:
                    jobs.append((seed, ep, ckpt_path))

            label_expected_epochs = required_epochs

            if label_expected_epochs:
                for seed in SEEDS:
                    seed_epochs = set(seed_to_expected_epochs.get(seed, []))
                    if not seed_epochs:
                        continue
                    missing_ckpts = [ep for ep in label_expected_epochs if ep not in seed_epochs]
                    if missing_ckpts:
                        log(
                            f"  [warning] label={label}, seed={seed}: missing model ckpts "
                            f"for epochs={missing_ckpts} (will be ignored in mean/var)"
                        )

            if jobs:
                total_ckpts = len(jobs)
                total_rollouts = total_ckpts * EVAL_N_ROLLOUTS
                done_ckpts = 0
                done_rollouts = 0

                log(
                    f"  - label={label}: start parallel eval "
                    f"({total_ckpts} ckpts, {total_rollouts} rollouts expected)"
                )

                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_EVALS) as pool:
                    futures = {
                        pool.submit(_run_agent_eval, ckpt_path): (seed, ep, ckpt_path)
                        for seed, ep, ckpt_path in jobs
                    }

                    for fut in as_completed(futures):
                        seed, ep, ckpt_path = futures[fut]
                        sr, err = fut.result()

                        done_ckpts += 1
                        done_rollouts += EVAL_N_ROLLOUTS

                        if sr is not None:
                            seed_to_ep2sr[seed][ep] = sr
                            log(
                                f"      [progress] {done_ckpts}/{total_ckpts} ckpts | "
                                f"{done_rollouts}/{total_rollouts} rollouts | "
                                f"seed={seed} epoch={ep:<4d} SR={sr:.4f}"
                            )
                        else:
                            log(
                                f"      [progress] {done_ckpts}/{total_ckpts} ckpts | "
                                f"{done_rollouts}/{total_rollouts} rollouts | "
                                f"seed={seed} epoch={ep:<4d} FAILED ({err})"
                            )

            for seed in SEEDS:
                ep2sr = seed_to_ep2sr.get(seed, {})
                expected_epochs = seed_to_expected_epochs.get(seed, [])
                if expected_epochs:
                    missing_epochs = sorted(set(expected_epochs) - set(ep2sr.keys()))
                    if missing_epochs:
                        log(
                            f"  [warning] label={label}, seed={seed}: missing eval data for "
                            f"epochs={missing_epochs} (will be ignored in mean/var)"
                        )
                if ep2sr:
                    per_seed_series.append(dict(sorted(ep2sr.items(), key=lambda kv: kv[0])))
                elif seed not in missing:
                    missing.append(seed)

            if not per_seed_series:
                log(f"  - label={label}: no valid eval series found (missing seeds: {missing})")
                continue

            epochs, mat = _stack_seed_series(per_seed_series)
            mean, low, high = _mean_ci(mat)

            color = LABEL_COLORS.get(label, None)
            n_seed = len(per_seed_series)

            ax.plot(epochs, mean, label=f"{label} (n={n_seed})", color=color, linewidth=2.0)
            ax.fill_between(epochs, low, high, color=color, alpha=0.20)
            plotted_any = True

            if missing:
                log(f"  - label={label}: used {n_seed} seeds, missing/invalid seeds={missing}")
            else:
                log(f"  - label={label}: used all {n_seed} seeds")

        if plotted_any:
            ax.set_title(f"IQL Re-eval Success Rate | task={TASK_NAME} | mask={mask}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")

            fig.tight_layout()
            save_path = out_dir / f"{_safe_name(mask)}_reeval.png"
            fig.savefig(save_path, dpi=DPI)
            log(f"  -> saved: {save_path}")
        else:
            log("  -> skipped (no plottable data)")

        plt.close(fig)

    log("\nDone.")


if __name__ == "__main__":
    main()
