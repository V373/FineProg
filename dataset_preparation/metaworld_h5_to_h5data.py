#!/usr/bin/env python3
"""Convert MetaWorld trajectory HDF5 files into FineProg video HDF5 files.

MetaWorld stores one trajectory per ``/traj_N`` group and keeps rendered BGR
frames in ``observations``. FineProg expects RGB frames under continuous
six-digit ``/videos`` groups. This converter performs that structural and
colour-channel conversion directly, without creating intermediate MP4 files.

Examples:
    # One output containing every trajectory (automatic filename).
    python dataset_preparation/metaworld_h5_to_h5data.py data.hdf5

    # Put the last four numerically ordered trajectories in validation.
    python dataset_preparation/metaworld_h5_to_h5data.py data.hdf5 \
        --valid-count 4

    # Explicit split filenames (``.h5`` is appended when omitted).
    python dataset_preparation/metaworld_h5_to_h5data.py data.hdf5 \
        --valid-count 4 \
        --train-output-name assembly_train \
        --valid-output-name assembly_valid
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np


IMAGE_SIZE = 224
CHUNK_LEN = 8
DEFAULT_FPS = 80
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed"
TRAJECTORY_PATTERN = re.compile(r"^traj_(\d+)$")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def trajectory_sort_key(name: str) -> int:
    """Return the numeric suffix of a canonical ``traj_N`` name."""
    match = TRAJECTORY_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid trajectory group name: {name!r}; expected traj_N")
    return int(match.group(1))


def _trajectory_names(source: h5py.File) -> list[str]:
    names = []
    for name in source.keys():
        if not name.startswith("traj_"):
            continue
        if not isinstance(source[name], h5py.Group):
            raise ValueError(f"/{name} must be an HDF5 group")
        trajectory_sort_key(name)
        names.append(name)
    if not names:
        raise ValueError(f"No traj_N groups found in {source.filename}")
    return sorted(names, key=trajectory_sort_key)


def _read_env_metadata(source: h5py.File) -> dict:
    raw = source.attrs.get("env_metadata")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        metadata = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Could not parse env_metadata; using filename and default FPS")
        return {}
    if not isinstance(metadata, dict):
        logger.warning("env_metadata is not a JSON object; using fallback values")
        return {}
    return metadata


def _read_fps(metadata: dict) -> int:
    raw_fps = metadata.get("fps", DEFAULT_FPS)
    if isinstance(raw_fps, (bool, np.bool_)):
        return DEFAULT_FPS
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError):
        return DEFAULT_FPS
    if not np.isfinite(fps) or fps <= 0:
        return DEFAULT_FPS
    return int(round(fps))


def _safe_task_name(metadata: dict, source_path: Path) -> str:
    raw_name = str(metadata.get("task_name") or source_path.stem)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._-")
    return safe_name or "metaworld"


def _normalise_output_name(name: str) -> str:
    if not name or name in {".", ".."}:
        raise ValueError("Output name must be a non-empty filename")
    path = Path(name)
    if path.is_absolute() or path.name != name:
        raise ValueError(
            f"Output name must be a filename without directories, got {name!r}"
        )
    if path.suffix == "":
        return f"{name}.h5"
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"Output filename must end in .h5 or .hdf5, got {name!r}")
    return name


def _build_output_plan(
    trajectory_names: Sequence[str],
    task_name: str,
    output_dir: Path,
    output_name: str | None,
    valid_count: int | None,
    train_output_name: str | None,
    valid_output_name: str | None,
) -> list[tuple[Path, list[str]]]:
    total = len(trajectory_names)
    if valid_count is None:
        if train_output_name is not None or valid_output_name is not None:
            raise ValueError(
                "--train-output-name and --valid-output-name require --valid-count"
            )
        final_name = _normalise_output_name(
            output_name or f"{task_name}-{total}vid.h5"
        )
        return [(output_dir / final_name, list(trajectory_names))]

    if output_name is not None:
        raise ValueError("--output-name cannot be combined with --valid-count")
    if isinstance(valid_count, bool) or not 1 <= valid_count < total:
        raise ValueError(
            f"valid_count must satisfy 1 <= valid_count < {total}, got {valid_count}"
        )

    train_count = total - valid_count
    train_name = _normalise_output_name(
        train_output_name or f"{task_name}-{train_count}vid_train.h5"
    )
    valid_name = _normalise_output_name(
        valid_output_name or f"{task_name}-{valid_count}vid_valid.h5"
    )
    if train_name == valid_name:
        raise ValueError("Train and validation output filenames must be different")

    return [
        (output_dir / train_name, list(trajectory_names[:train_count])),
        (output_dir / valid_name, list(trajectory_names[train_count:])),
    ]


def _validate_observations(
    trajectory_name: str,
    trajectory_group: h5py.Group,
) -> h5py.Dataset:
    if "observations" not in trajectory_group:
        raise KeyError(f"/{trajectory_name} is missing observations")
    observations = trajectory_group["observations"]
    if not isinstance(observations, h5py.Dataset):
        raise ValueError(f"/{trajectory_name}/observations must be a dataset")
    if observations.ndim != 4 or observations.shape[-1] != 3:
        raise ValueError(
            f"/{trajectory_name}/observations must have shape [T,H,W,3], "
            f"got {observations.shape}"
        )
    if observations.shape[0] == 0:
        raise ValueError(f"/{trajectory_name}/observations is empty")
    if observations.dtype != np.dtype(np.uint8):
        raise ValueError(
            f"/{trajectory_name}/observations must have dtype uint8, "
            f"got {observations.dtype}"
        )
    return observations


def _write_frames(
    source: h5py.Dataset, destination: h5py.Dataset, source_start: int = 0
) -> None:
    """Copy BGR source frames to an RGB 224x224 destination in small chunks."""
    total = source.shape[0] - source_start
    _, source_height, source_width, _ = source.shape
    needs_resize = (source_height, source_width) != (IMAGE_SIZE, IMAGE_SIZE)

    for destination_start in range(0, total, CHUNK_LEN):
        destination_end = min(destination_start + CHUNK_LEN, total)
        source_slice_start = source_start + destination_start
        source_slice_end = source_start + destination_end
        rgb = source[source_slice_start:source_slice_end][..., ::-1]
        if needs_resize:
            rgb = np.stack(
                [cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE)) for frame in rgb],
                axis=0,
            )
        destination[destination_start:destination_end] = np.ascontiguousarray(
            rgb, dtype=np.uint8
        )


def _write_output(
    source: h5py.File,
    source_path: Path,
    trajectory_names: Sequence[str],
    output_path: Path,
    fps: int,
) -> None:
    with h5py.File(output_path, "w", libver="earliest") as destination:
        videos = destination.create_group("videos")
        for output_index, trajectory_name in enumerate(trajectory_names, start=1):
            observations = _validate_observations(
                trajectory_name, source[trajectory_name]
            )
            frame_count = int(observations.shape[0]) - 1
            video = videos.create_group(f"{output_index:06d}")
            frames = video.create_dataset(
                "frames",
                shape=(frame_count, IMAGE_SIZE, IMAGE_SIZE, 3),
                dtype=np.uint8,
                compression=None,
                chunks=(min(CHUNK_LEN, frame_count), IMAGE_SIZE, IMAGE_SIZE, 3),
            )
            _write_frames(observations, frames, source_start=1)

            # Match the metadata produced by the existing MP4 converter. The
            # path is the location where hdf5_obs_to_mp4.py would have written
            # this trajectory; it is compatibility metadata and need not exist.
            hypothetical_mp4 = (
                source_path.parent
                / "video"
                / source_path.stem
                / f"{trajectory_name}.mp4"
            )
            video.attrs["action_name"] = trajectory_name
            video.attrs["action_id"] = 0
            video.attrs["fps"] = fps
            video.attrs["num_frames"] = frame_count
            video.attrs["path"] = str(hypothetical_mp4)
        destination.flush()


def _validate_output(output_path: Path, expected_trajectories: Sequence[str]) -> None:
    with h5py.File(output_path, "r") as output:
        if list(output.keys()) != ["videos"]:
            raise RuntimeError(f"Invalid output root structure in {output_path}")
        videos = output["videos"]
        expected_ids = [
            f"{index:06d}" for index in range(1, len(expected_trajectories) + 1)
        ]
        if sorted(videos.keys()) != expected_ids:
            raise RuntimeError(f"Invalid video IDs in {output_path}")
        for video_id, trajectory_name in zip(expected_ids, expected_trajectories):
            video = videos[video_id]
            frames = video.get("frames")
            if not isinstance(frames, h5py.Dataset):
                raise RuntimeError(f"Missing /videos/{video_id}/frames in {output_path}")
            if frames.ndim != 4 or frames.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise RuntimeError(
                    f"Invalid frame shape at /videos/{video_id}/frames: {frames.shape}"
                )
            if frames.dtype != np.dtype(np.uint8):
                raise RuntimeError(
                    f"Invalid frame dtype at /videos/{video_id}/frames: {frames.dtype}"
                )
            if video.attrs.get("action_name") != trajectory_name:
                raise RuntimeError(
                    f"Trajectory mapping mismatch at /videos/{video_id}"
                )
            if int(video.attrs.get("num_frames", -1)) != frames.shape[0]:
                raise RuntimeError(f"Frame-count mismatch at /videos/{video_id}")


def convert_metaworld_h5(
    input_hdf5: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_name: str | None = None,
    valid_count: int | None = None,
    train_output_name: str | None = None,
    valid_output_name: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Convert one MetaWorld HDF5 and return the finalized output path(s)."""
    source_path = Path(input_hdf5).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Input HDF5 does not exist: {source_path}")
    if source_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"Input must be an .h5 or .hdf5 file: {source_path}")

    destination_dir = Path(output_dir).expanduser().resolve()
    with h5py.File(source_path, "r") as source:
        trajectory_names = _trajectory_names(source)
        metadata = _read_env_metadata(source)
        fps = _read_fps(metadata)
        task_name = _safe_task_name(metadata, source_path)
        output_plan = _build_output_plan(
            trajectory_names=trajectory_names,
            task_name=task_name,
            output_dir=destination_dir,
            output_name=output_name,
            valid_count=valid_count,
            train_output_name=train_output_name,
            valid_output_name=valid_output_name,
        )

        targets = [target for target, _ in output_plan]
        if len(set(targets)) != len(targets):
            raise ValueError("Output paths must be unique")
        for target in targets:
            if target == source_path:
                raise ValueError("Output path must not overwrite the input HDF5")
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"Output already exists: {target}; pass --overwrite to replace it"
                )

        destination_dir.mkdir(parents=True, exist_ok=True)
        temporary_outputs: list[tuple[Path, Path]] = []
        current_temporary: Path | None = None
        try:
            # Finish and validate every split before replacing any final file.
            for target, selected_trajectories in output_plan:
                current_temporary = target.with_name(
                    f".{target.name}.{uuid.uuid4().hex}.tmp"
                )
                logger.info(
                    "Writing %s trajectories to temporary file %s",
                    len(selected_trajectories),
                    current_temporary,
                )
                _write_output(
                    source=source,
                    source_path=source_path,
                    trajectory_names=selected_trajectories,
                    output_path=current_temporary,
                    fps=fps,
                )
                _validate_output(current_temporary, selected_trajectories)
                temporary_outputs.append((current_temporary, target))
                current_temporary = None

            for temporary, target in temporary_outputs:
                os.replace(temporary, target)
                logger.info("Saved %s", target)
        except Exception:
            for temporary, _ in temporary_outputs:
                temporary.unlink(missing_ok=True)
            # Also remove the current temporary file if writing failed before
            # it was appended to temporary_outputs.
            if current_temporary is not None:
                current_temporary.unlink(missing_ok=True)
            raise

    return [target for target, _ in output_plan]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_hdf5", type=Path, help="MetaWorld .h5/.hdf5 file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        help="Single-file output name; incompatible with --valid-count",
    )
    parser.add_argument(
        "--valid-count",
        type=int,
        help="Put the last N numerically ordered trajectories in validation",
    )
    parser.add_argument(
        "--train-output-name",
        help="Train filename in split mode (automatic when omitted)",
    )
    parser.add_argument(
        "--valid-output-name",
        help="Validation filename in split mode (automatic when omitted)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files that already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = convert_metaworld_h5(
        args.input_hdf5,
        output_dir=args.output_dir,
        output_name=args.output_name,
        valid_count=args.valid_count,
        train_output_name=args.train_output_name,
        valid_output_name=args.valid_output_name,
        overwrite=args.overwrite,
    )
    print("Conversion complete:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
