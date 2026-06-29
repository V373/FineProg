#!/usr/bin/env python3
"""Re-evaluate a checkpoint sweep and plot success-rate curves with CI.

For each `step_*.pt` checkpoint in a directory, this script reuses
`policy_training/evaluate_policy.py` to run evaluation with the seed list
defined in the YAML config, collects the per-seed success rate from the
generated JSON outputs, stores a success-rate matrix, and plots the mean curve
with confidence interval shading.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ckpt_success_reeval"
EVALUATE_POLICY_SCRIPT = PROJECT_ROOT / "evaluate_policy.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.eval_utils import (  # noqa: E402
    _apply_cli_overrides,
    _load_eval_config,
    _resolve_auto_eval_paths,
    load_checkpoint_for_eval,
)

CKPT_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate a checkpoint sweep and plot success rates")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/evaluate_policy.yaml"))
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing step_*.pt checkpoints")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--evaluate_script", type=str, default=str(EVALUATE_POLICY_SCRIPT))
    parser.add_argument("--plot_title", type=str, default=None)
    parser.add_argument("--plot_xlabel", type=str, default="Checkpoint Step")
    parser.add_argument("--plot_ylabel", type=str, default="Success Rate")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ci_z", type=float, default=1.96)
    parser.add_argument("--n_rollouts", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--worker_device", type=str, default=None)
    parser.add_argument("--no_video", action="store_true", default=True)
    return parser.parse_args()


def _discover_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint directory does not exist: {ckpt_dir}")
    if not ckpt_dir.is_dir():
        raise NotADirectoryError(f"checkpoint path is not a directory: {ckpt_dir}")

    ckpts: list[tuple[int, Path]] = []
    for path in ckpt_dir.iterdir():
        if not path.is_file() or path.name == "final.pt":
            continue
        match = CKPT_STEP_RE.match(path.name)
        if match is None:
            continue
        ckpts.append((int(match.group(1)), path))

    ckpts.sort(key=lambda item: item[0])
    if not ckpts:
        raise ValueError(f"no step_*.pt checkpoints found in {ckpt_dir}")
    return ckpts


def _plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {k: _plain(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _normalize_seeds(seed_value: Any) -> list[int]:
    if isinstance(seed_value, (list, tuple)):
        seeds = [int(item) for item in seed_value]
    else:
        seeds = [int(seed_value)]
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return seeds


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, steps: list[int], seeds: list[int], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step"] + [f"seed_{seed}" for seed in seeds])
        for row_idx, step in enumerate(steps):
            row = [step]
            for col_idx in range(len(seeds)):
                row.append(matrix[col_idx, row_idx])
            writer.writerow(row)


def _stack_seed_series(seed_series: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    all_steps = sorted({step for series in seed_series for step in series.keys()})
    if not all_steps:
        return np.array([], dtype=np.int32), np.empty((0, 0), dtype=np.float64)

    matrix = np.full((len(seed_series), len(all_steps)), np.nan, dtype=np.float64)
    step_to_idx = {step: idx for idx, step in enumerate(all_steps)}
    for row_idx, series in enumerate(seed_series):
        for step, value in series.items():
            matrix[row_idx, step_to_idx[step]] = value
    return np.asarray(all_steps, dtype=np.int32), matrix


def _mean_ci(matrix: np.ndarray, ci_z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    mean = np.nanmean(matrix, axis=0)
    count = np.sum(~np.isnan(matrix), axis=0)
    std = np.full_like(mean, np.nan, dtype=np.float64)

    for idx in range(matrix.shape[1]):
        column = matrix[:, idx]
        column = column[~np.isnan(column)]
        if column.size >= 2:
            std[idx] = np.std(column, ddof=1)
        elif column.size == 1:
            std[idx] = 0.0

    sem = np.divide(std, np.sqrt(np.maximum(count, 1)), where=~np.isnan(std))
    ci = ci_z * sem
    lower = np.clip(mean - ci, 0.0, 1.0)
    upper = np.clip(mean + ci, 0.0, 1.0)
    return mean, lower, upper


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _build_eval_cmd(args: argparse.Namespace, ckpt_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(args.evaluate_script).resolve()),
        "--config",
        str(Path(args.config).resolve()),
        "--agent",
        str(ckpt_path.resolve()),
    ]
    if args.no_video:
        cmd.append("--no_video")
    if args.n_rollouts is not None:
        cmd.extend(["--n_rollouts", str(args.n_rollouts)])
    if args.horizon is not None:
        cmd.extend(["--horizon", str(args.horizon)])
    if args.num_workers is not None:
        cmd.extend(["--num_workers", str(args.num_workers)])
    if args.worker_device is not None:
        cmd.extend(["--worker_device", str(args.worker_device)])
    return cmd


def _read_success_rate(json_path: Path) -> float | None:
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary", {})
    value = summary.get("Success_Rate")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_output_root(args: argparse.Namespace, ckpt_dir: Path) -> Path:
    root = Path(args.output_root).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    try:
        rel = ckpt_dir.resolve().relative_to(PROJECT_ROOT)
        return root / rel
    except ValueError:
        return root / _safe_name(ckpt_dir.name)


def _make_override_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        agent=None,
        device=None,
        seed=None,
        seeds=None,
        n_rollouts=args.n_rollouts,
        horizon=args.horizon,
        num_workers=args.num_workers,
        worker_device=args.worker_device,
        env=None,
        video_path=None,
        no_video=args.no_video,
        video_skip=None,
        video_fps=None,
        frame_height=None,
        frame_width=None,
        camera_names=None,
        json_path=None,
        output_dir=None,
        stochastic=None,
    )


def main() -> None:
    args = _parse_args()
    eval_script = Path(args.evaluate_script).resolve()
    eval_config = Path(args.config).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    runtime_cfg = _load_eval_config(str(eval_config))
    eval_cfg = _apply_cli_overrides(runtime_cfg, _make_override_args(args))

    seeds = _normalize_seeds(eval_cfg["seed"])
    if args.no_video and bool(eval_cfg["video"]["enabled"]):
        eval_cfg["video"]["enabled"] = False

    ckpts = _discover_checkpoints(ckpt_dir)
    output_root = _resolve_output_root(args, ckpt_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / _safe_name(str(ckpt_dir.name))
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("Checkpoint success-rate re-eval", flush=True)
    print(f"config={eval_config}", flush=True)
    print(f"ckpt_dir={ckpt_dir}", flush=True)
    print(f"num_checkpoints={len(ckpts)}", flush=True)
    print(f"seeds={seeds}", flush=True)
    print(f"output_dir={run_dir}", flush=True)
    print("=" * 80, flush=True)

    step_values: list[int] = []
    seed_series: list[dict[int, float]] = [dict() for _ in seeds]
    failures: list[dict[str, Any]] = []

    # Sequential processing is used to keep evaluation paths deterministic and
    # avoid over-saturating the GPU / filesystem when each checkpoint itself
    # already uses multiple rollout workers.
    for step, ckpt_path in ckpts:
        print(f"[ckpt] step={step} path={ckpt_path.name}", flush=True)
        runtime = load_checkpoint_for_eval(str(ckpt_path), str(eval_cfg["device"]))
        cmd = _build_eval_cmd(args, ckpt_path)
        cmd.extend(["--seeds", *[str(seed) for seed in seeds]])
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, check=False, env=env)
        if proc.returncode != 0:
            failures.append({"step": step, "ckpt_path": str(ckpt_path), "error": f"eval failed rc={proc.returncode}"})
            continue

        step_values.append(step)
        for seed_idx, seed in enumerate(seeds):
            per_seed_cfg = deepcopy(eval_cfg)
            per_seed_cfg["seed"] = int(seed)
            per_seed_cfg["video"]["enabled"] = False
            output_plan = _resolve_auto_eval_paths(
                eval_cfg=per_seed_cfg,
                loaded=runtime,
                config_dir=runtime_cfg.config_dir,
                n_rollouts=int(per_seed_cfg["rollout"]["n_rollouts"]),
                horizon=int(per_seed_cfg["rollout"]["horizon"]),
            )
            json_path = Path(output_plan["json"]["path"])
            sr = _read_success_rate(json_path)
            if sr is None:
                failures.append({"step": step, "seed": int(seed), "ckpt_path": str(ckpt_path), "error": f"missing success rate in {json_path}"})
            else:
                seed_series[seed_idx][step] = sr
        print(f"  -> success_rates={[seed_series[idx].get(step, np.nan) for idx in range(len(seeds))]}", flush=True)

    epochs, matrix = _stack_seed_series(seed_series)
    mean, lower, upper = _mean_ci(matrix, float(args.ci_z))

    _write_csv(run_dir / "success_rate_matrix.csv", step_values, seeds, matrix)
    _write_json(
        run_dir / "success_rate_matrix.json",
        {
            "config": str(eval_config),
            "ckpt_dir": str(ckpt_dir),
            "steps": step_values,
            "seeds": seeds,
            "success_rate_matrix": matrix.tolist(),
            "mean": mean.tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "failures": failures,
        },
    )
    np.save(run_dir / "success_rate_matrix.npy", matrix)
    np.save(run_dir / "steps.npy", np.asarray(step_values, dtype=np.int32))
    np.save(run_dir / "seeds.npy", np.asarray(seeds, dtype=np.int32))

    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(10, 5))
    if epochs.size > 0:
        ax.plot(epochs, mean, color="#1f77b4", linewidth=2.0, label=f"mean (n={len(seeds)})")
        ax.fill_between(epochs, lower, upper, color="#1f77b4", alpha=0.20, label="95% CI")
    ax.set_xlabel(args.plot_xlabel)
    ax.set_ylabel(args.plot_ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(args.plot_title or f"Success Rate vs Checkpoint Step | {ckpt_dir.name}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    plot_path = run_dir / f"{_safe_name(ckpt_dir.name)}_success_curve.png"
    fig.savefig(plot_path, dpi=args.dpi)
    plt.close(fig)

    print(f"saved_matrix={run_dir / 'success_rate_matrix.csv'}", flush=True)
    print(f"saved_plot={plot_path}", flush=True)
    if failures:
        print(f"failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
