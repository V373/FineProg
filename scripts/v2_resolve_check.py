#!/usr/bin/env python3
"""V2 config smoke-test: resolve all stage/eval configs and check paths.

Loads every configs_v2/ YAML, resolves all refs to absolute paths,
pretty-prints the results, and reports which paths exist on disk.
No data is read or processed — safe to run at any time.

Usage
-----
    # Full check (all stages + all eval tasks):
    conda run -n fineprog python scripts/v2_resolve_check.py

    # Single stage/task:
    conda run -n fineprog python scripts/v2_resolve_check.py --task train
    conda run -n fineprog python scripts/v2_resolve_check.py --task extract
    conda run -n fineprog python scripts/v2_resolve_check.py --task data_process
    conda run -n fineprog python scripts/v2_resolve_check.py --task eval_kendalls_tau
    conda run -n fineprog python scripts/v2_resolve_check.py --task eval_expert_projection
    conda run -n fineprog python scripts/v2_resolve_check.py --task eval_classification

    # Manual registry lookup:
    conda run -n fineprog python scripts/v2_resolve_check.py --dataset robomimic_can_mh_100vid_okay
    conda run -n fineprog python scripts/v2_resolve_check.py --run can_ph_180_ep50k
    conda run -n fineprog python scripts/v2_resolve_check.py --embedding can_mh_okay_ep50k
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root or from scripts/ directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJ_ROOT  = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJ_ROOT))

from utils.config_v2 import ConfigV2   # noqa: E402


# ── Formatting helpers ─────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


def _check_and_print(cfg: ConfigV2, resolved: dict, title: str) -> bool:
    """Print resolved config and path checks.  Returns True if all paths exist."""
    cfg.print_config(resolved, title)
    path_checks = cfg.check_paths(resolved)
    if not path_checks:
        print("  (no file paths to check)\n")
        return True
    all_ok = True
    print("  Path existence check:")
    for key, path, exists in path_checks:
        status = "✓" if exists else "✗ MISSING"
        print(f"    [{status}]  {key}")
        print(f"              {path}")
        if not exists:
            all_ok = False
    if all_ok:
        print("  → All paths exist.\n")
    else:
        print("  → Some paths are missing (expected if not yet generated).\n")
    return all_ok


# ── Per-stage runners ──────────────────────────────────────────────────────

def run_data_process(cfg: ConfigV2) -> bool:
    return _check_and_print(cfg, cfg.load_data_process(), "Stage: data_process")


def run_train(cfg: ConfigV2) -> bool:
    return _check_and_print(cfg, cfg.load_train(), "Stage: train")


def run_extract(cfg: ConfigV2) -> bool:
    return _check_and_print(cfg, cfg.load_extract(), "Stage: extract")


def run_eval(cfg: ConfigV2, task_name: str) -> bool:
    return _check_and_print(cfg, cfg.load_eval(task_name), f"Eval task: {task_name}")


# ── Registry lookup runners ────────────────────────────────────────────────

def run_dataset_lookup(cfg: ConfigV2, dataset_ref: str) -> None:
    _section(f"Dataset registry lookup: {dataset_ref}")
    resolved = cfg.resolve_dataset(dataset_ref)
    cfg.print_config(resolved, f"dataset: {dataset_ref}")
    path_checks = cfg.check_paths({"dataset": resolved})
    for key, path, exists in path_checks:
        status = "✓" if exists else "✗ MISSING"
        print(f"  [{status}]  {path}")


def run_run_lookup(cfg: ConfigV2, run_ref: str) -> None:
    _section(f"Run registry lookup: {run_ref}")
    resolved = cfg.resolve_run(run_ref)
    cfg.print_config(resolved, f"run: {run_ref}")
    path_checks = cfg.check_paths({"run": resolved})
    for key, path, exists in path_checks:
        status = "✓" if exists else "✗ MISSING"
        print(f"  [{status}]  {path}")


def run_embedding_lookup(cfg: ConfigV2, emb_ref: str) -> None:
    _section(f"Embedding registry lookup: {emb_ref}")
    resolved = cfg.resolve_embedding(emb_ref)
    cfg.print_config(resolved, f"embedding: {emb_ref}")
    path_checks = cfg.check_paths({"embedding": resolved})
    for key, path, exists in path_checks:
        status = "✓" if exists else "✗ MISSING"
        print(f"  [{status}]  {path}")


# ── Main ───────────────────────────────────────────────────────────────────

_ALL_TASKS = [
    "data_process",
    "train",
    "extract",
    "eval_kendalls_tau",
    "eval_expert_projection",
    "eval_classification",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2 config smoke-test — resolve paths and check existence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["all"] + _ALL_TASKS,
        default="all",
        help="Which stage/eval config to check (default: all)",
    )
    parser.add_argument(
        "--dataset",
        metavar="DATASET_REF",
        help="Look up a single dataset key from datasets.yaml",
    )
    parser.add_argument(
        "--run",
        metavar="RUN_REF",
        help="Look up a single run alias from runs.yaml",
    )
    parser.add_argument(
        "--embedding",
        metavar="EMBEDDING_REF",
        help="Look up a single embedding alias from runs.yaml",
    )
    parser.add_argument(
        "--configs-dir",
        metavar="DIR",
        default=None,
        help="Override path to configs_v2/ directory",
    )
    args = parser.parse_args()

    cfg = ConfigV2(args.configs_dir)
    print(f"[v2_resolve_check] configs_v2 root : {cfg._root}")
    print(f"[v2_resolve_check] project root    : {cfg._proj_root}")

    # ── Registry lookups ────────────────────────────────────────────────
    if args.dataset:
        run_dataset_lookup(cfg, args.dataset)
    if args.run:
        run_run_lookup(cfg, args.run)
    if args.embedding:
        run_embedding_lookup(cfg, args.embedding)
    if args.dataset or args.run or args.embedding:
        return  # lookup mode — don't also run all stages

    # ── Stage/eval checks ───────────────────────────────────────────────
    task = args.task
    results: dict[str, bool] = {}

    _runners = {
        "data_process":          run_data_process,
        "train":                 run_train,
        "extract":               run_extract,
        "eval_kendalls_tau":     lambda c: run_eval(c, "kendalls_tau"),
        "eval_expert_projection":lambda c: run_eval(c, "expert_projection"),
        "eval_classification":   lambda c: run_eval(c, "classification"),
    }

    to_run = _ALL_TASKS if task == "all" else [task]
    for name in to_run:
        try:
            results[name] = _runners[name](cfg)
        except Exception as exc:          # noqa: BLE001
            print(f"\n  [ERROR] {name}: {exc}\n")
            results[name] = False

    # ── Summary ─────────────────────────────────────────────────────────
    if task == "all":
        print("\n" + "=" * 64)
        print("  Summary")
        print("=" * 64)
        for name, ok in results.items():
            status = "OK" if ok else "ISSUES"
            print(f"  [{status:6s}]  {name}")
        print()

    print("[v2_resolve_check] Done.")


if __name__ == "__main__":
    main()
