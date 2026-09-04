#!/usr/bin/env python3
"""Train the ten MetaWorld encoders serially from the current V2 config.

Only ``train.yaml::train_dataset`` differs between jobs.  At startup the
script snapshots the repository's current ``configs_v2`` directory, derives
one isolated config directory per task, and points each training subprocess at
its copy through ``FINEPROG_CONFIGS_V2_DIR``.  The repository configuration is
never modified.

Training order:

    assembly -> soccer -> disassemble -> coffee push -> drawer open -> sweep into
    -> button press wall -> door lock -> window close -> hammer

Each subprocess uses ``train_encoder.py`` unchanged, including its W&B
project, metrics, in-training evaluation, run naming, and checkpoint behavior.

Usage::

    conda run --no-capture-output -n fineprog \
        python scripts/train_metaworld_encoders_serial.py
    conda run -n fineprog \
        python scripts/train_metaworld_encoders_serial.py --dry-run
    conda run --no-capture-output -n fineprog \
        python scripts/train_metaworld_encoders_serial.py --gpu 0
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"
TRAIN_ENTRYPOINT = PROJECT_ROOT / "train_encoder.py"
CONFIGS_ENV_VAR = "FINEPROG_CONFIGS_V2_DIR"

# Registry aliases from configs_v2/registry/datasets.yaml, in required order.
TASKS: tuple[tuple[str, str], ...] = (
    # ("assembly", "metaworld_assembly_v2_36vid_train"),
    # ("soccer", "metaworld_soccer_v2_36vid_train"),
    # ("disassemble", "metaworld_disassemble_v2_36vid_train"),
    # ("coffee_push", "metaworld_coffee_push_v2_36vid_train"),
    # ("drawer_open", "metaworld_drawer_open_v2_36vid_train"),
    ("sweep_into", "metaworld_sweep_into_v2_36vid_train"),
    ("button_press_wall", "metaworld_button_press_wall_v2_36vid_train"),
    ("door_lock", "metaworld_door_lock_v2_36vid_train"),
    ("window_close", "metaworld_window_close_v2_36vid_train"),
    ("hammer", "metaworld_hammer_v2_36vid_train"),
)

_TRAIN_DATASET_LINE = re.compile(
    r"^(?P<prefix>train_dataset:\s*)(?P<value>\S+)(?P<suffix>\s*(?:#.*)?)$",
    re.MULTILINE,
)
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _set_train_dataset(train_path: Path, dataset_alias: str) -> None:
    """Replace exactly the train_dataset value while preserving the YAML text."""
    source = train_path.read_text(encoding="utf-8")
    matches = list(_TRAIN_DATASET_LINE.finditer(source))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one train_dataset line in {train_path}, "
            f"found {len(matches)}"
        )
    updated = _TRAIN_DATASET_LINE.sub(
        lambda match: (
            f"{match.group('prefix')}{dataset_alias}{match.group('suffix')}"
        ),
        source,
        count=1,
    )
    train_path.write_text(updated, encoding="utf-8")


def _assert_only_dataset_changed(base_train: Path, task_train: Path) -> None:
    base = _load_yaml(base_train)
    task = _load_yaml(task_train)
    base.pop("train_dataset", None)
    task.pop("train_dataset", None)
    if task != base:
        raise ValueError(
            f"Unexpected non-dataset config change between {base_train} and "
            f"{task_train}"
        )


def _validate_dataset(config_root: Path, expected_alias: str) -> Path:
    """Resolve the isolated config and verify its 36-video training H5."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils.config_v2 import ConfigV2

    resolved = ConfigV2(config_root).load_train()
    if resolved.get("train_dataset") != expected_alias:
        raise ValueError(
            f"Expected train_dataset={expected_alias!r}, got "
            f"{resolved.get('train_dataset')!r}"
        )

    h5_path = Path(resolved["h5_path"])
    if not h5_path.is_file():
        raise FileNotFoundError(f"Training dataset does not exist: {h5_path}")
    with h5py.File(h5_path, "r") as h5_file:
        videos = h5_file.get("videos")
        if not isinstance(videos, h5py.Group) or len(videos) != 36:
            count = len(videos) if isinstance(videos, h5py.Group) else 0
            raise ValueError(f"Expected 36 videos in {h5_path}, found {count}")
    return h5_path


def _prepare_jobs(run_dir: Path) -> list[dict]:
    """Snapshot current configs_v2 and build one dataset-only override per task."""
    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing config directory: {SOURCE_CONFIG_ROOT}")
    if not TRAIN_ENTRYPOINT.is_file():
        raise FileNotFoundError(f"Missing training entrypoint: {TRAIN_ENTRYPOINT}")

    source_snapshot = run_dir / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    jobs = []
    for task_name, dataset_alias in TASKS:
        config_root = run_dir / f"configs_{task_name}"
        shutil.copytree(source_snapshot, config_root)
        task_train = config_root / "train.yaml"
        _set_train_dataset(task_train, dataset_alias)
        _assert_only_dataset_changed(source_snapshot / "train.yaml", task_train)
        h5_path = _validate_dataset(config_root, dataset_alias)
        jobs.append(
            {
                "task": task_name,
                "dataset_alias": dataset_alias,
                "config_root": config_root,
                "h5_path": h5_path,
            }
        )
    return jobs


def _worker_env(config_root: Path, wandb_group: str, gpu: str | None) -> dict:
    env = os.environ.copy()
    env[CONFIGS_ENV_VAR] = str(config_root)
    env["WANDB_RUN_GROUP"] = wandb_group
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate the active training process and its DataLoader workers."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def _handle_stop_signal(signum: int, _frame) -> None:
    """Abort the serial loop; _run_one's finally block cleans up workers."""
    for stop_signal in _STOP_SIGNALS:
        signal.signal(stop_signal, signal.SIG_DFL)
    print(
        f"[serial] received {signal.Signals(signum).name}; stopping.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(128 + signum)


def _install_signal_handlers() -> None:
    for stop_signal in _STOP_SIGNALS:
        signal.signal(stop_signal, _handle_stop_signal)


def _run_one(job: dict, wandb_group: str, gpu: str | None) -> int:
    command = [sys.executable, str(TRAIN_ENTRYPOINT)]
    print(f"[serial] command: {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=_worker_env(job["config_root"], wandb_group, gpu),
        start_new_session=True,
    )
    try:
        return process.wait()
    finally:
        _terminate_process_group(process)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ten MetaWorld encoders serially using current train.yaml."
    )
    parser.add_argument(
        "--gpu",
        help="CUDA_VISIBLE_DEVICES value for every run (default: inherit).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Config snapshot directory (default: "
            "outputs/serial/<timestamped-group>)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate all configs and datasets without training.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    wandb_group = "metaworld-encoder-serial-" + datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    run_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "serial" / wandb_group
    )
    if run_dir.exists():
        raise FileExistsError(f"Output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    jobs = _prepare_jobs(run_dir)
    print(f"[serial] W&B group : {wandb_group}")
    print(f"[serial] run dir   : {run_dir}")
    for index, job in enumerate(jobs, start=1):
        print(
            f"[serial] {index}/{len(jobs)} {job['task']}: "
            f"{job['dataset_alias']} -> {job['h5_path']}",
            flush=True,
        )

    if args.dry_run:
        print("[serial] dry run complete; no training started.")
        return 0

    _install_signal_handlers()
    failures = []
    for index, job in enumerate(jobs, start=1):
        print(
            f"[serial] starting {index}/{len(jobs)}: {job['task']} "
            f"({job['dataset_alias']})",
            flush=True,
        )
        status = _run_one(job, wandb_group, args.gpu)
        if status != 0:
            failures.append(f"{job['task']} (exit={status})")
            print(
                f"[serial] failed: {job['task']} exited with status {status}; "
                "continuing with the next task.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"[serial] finished: {job['task']}", flush=True)

    if failures:
        print(
            "[serial] completed with failures: " + ", ".join(failures),
            file=sys.stderr,
        )
        return 1

    print(f"[serial] all {len(TASKS)} MetaWorld encoder runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
