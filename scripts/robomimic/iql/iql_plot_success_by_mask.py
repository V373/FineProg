#!/usr/bin/env python3
"""
iql_plot_success_by_mask.py
───────────────────────────
Plot IQL success-rate curves by mask for can-style robomimic tasks.

For each mask:
- one figure is generated
- each figure contains 3 reward labels (original / dense / PBRS)
- each label curve is mean across seeds
- confidence interval shading is computed across seeds

Data source:
- latest run folder under each training output directory
- preferred log file: logs/wandb/latest-run/files/output.log
- fallback log file:  logs/log.txt

Run:
    conda run -n fineprog python \
      /home/user/zhangzk/projects/fineprog/scripts/robomimic/iql/iql_plot_success_by_mask.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ── Configuration (hard-coded, similar style to iql_scale_train.py) ──────────

IQL_MODELS_ROOT = Path("/home/user/zhangzk/projects/fineprog/third_party/robomimic/iql_trained_models")
OUTPUT_ROOT = Path("/home/user/zhangzk/projects/fineprog/outputs/robomimic/iql")

TASK_NAME = "can"
TASK_DIR_NAME = "PickPlaceCan"

MASKS = [
    "IQL_expert_half",
    # "IQL_expert",
    # "IQL_expert_worse_halfmix",
    # "IQL_expert_worse",
    # "IQL_okay_worse",
]

REWARD_LABELS = ["original", "dense", "PBRS"]
SEEDS = [1, 2, 3]

# Confidence interval setup: mean ± Z * SEM
CI_Z = 1.96
FIG_SIZE = (9, 5)
DPI = 300

LABEL_COLORS = {
    "original": "#1f77b4",
    "dense": "#d62728",
    "PBRS": "#2ca02c",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

EPOCH_ROLLOUT_RE = re.compile(r"Epoch\s+(\d+)\s+Rollouts\s+took")
SUCCESS_RE = re.compile(r'"Success_Rate"\s*:\s*([0-9]*\.?[0-9]+)')


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


def _pick_log_file(run_dir: Path) -> Path | None:
    preferred = run_dir / "logs/wandb/latest-run/files/output.log"
    fallback = run_dir / "logs/log.txt"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    return None


def _parse_epoch_success_from_log(log_path: Path) -> Dict[int, float]:
    """Parse {epoch -> success_rate} from output log.

    The robust anchor in these logs is:
      Epoch <N> Rollouts took ...
      ...
      "Success_Rate": <v>

    Repeated duplicate lines are naturally de-duplicated via dict overwrite.
    """
    result: Dict[int, float] = {}
    pending_epoch: int | None = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_epoch = EPOCH_ROLLOUT_RE.search(line)
            if m_epoch:
                pending_epoch = int(m_epoch.group(1))
                continue

            if pending_epoch is not None:
                m_succ = SUCCESS_RE.search(line)
                if m_succ:
                    result[pending_epoch] = float(m_succ.group(1))
                    pending_epoch = None

    return dict(sorted(result.items(), key=lambda kv: kv[0]))


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


# ── Main plotting flow ────────────────────────────────────────────────────────

def main() -> None:
    out_dir = OUTPUT_ROOT / TASK_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log(f"IQL Success Plotter | task={TASK_NAME}")
    log(f"masks={MASKS}")
    log(f"labels={REWARD_LABELS}, seeds={SEEDS}")
    log(f"output_dir={out_dir}")
    log("=" * 80)

    for mask in MASKS:
        log(f"\n[mask] {mask}")

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        plotted_any = False

        for label in REWARD_LABELS:
            per_seed_series: List[Dict[int, float]] = []
            missing = []

            for seed in SEEDS:
                run_dir = _find_latest_run_dir(mask, label, seed)
                if run_dir is None:
                    missing.append(seed)
                    continue

                log_file = _pick_log_file(run_dir)
                if log_file is None:
                    missing.append(seed)
                    continue

                series = _parse_epoch_success_from_log(log_file)
                if not series:
                    missing.append(seed)
                    continue

                per_seed_series.append(series)

            if not per_seed_series:
                log(f"  - label={label}: no valid series found (missing seeds: {missing})")
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
            ax.set_title(f"IQL Success Rate | task={TASK_NAME} | mask={mask}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")

            fig.tight_layout()
            save_path = out_dir / f"{_safe_name(mask)}.png"
            fig.savefig(save_path, dpi=DPI)
            log(f"  -> saved: {save_path}")
        else:
            log("  -> skipped (no plottable data)")

        plt.close(fig)

    log("\nDone.")


if __name__ == "__main__":
    main()
