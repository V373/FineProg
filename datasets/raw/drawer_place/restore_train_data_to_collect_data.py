#!/usr/bin/env python3
"""Losslessly restore drawer_place HDF5 episodes into collect_data layout.

The source HDF5 files already contain JPEG bitstreams. This tool writes those
bytes directly to the four camera folders and serializes the six observation
arrays in the same per-frame pickle layout used by the fruits collection data.
Each restored episode also includes a hard link to its original HDF5 file, so
root attributes and drawer-specific annotation datasets remain byte-for-byte
available without duplicating the HDF5 payload on this filesystem.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
from typing import Any
import uuid

import h5py
import numpy as np


DATASET_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = DATASET_ROOT / "train_data"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "collect_data"
FORMAT_VERSION = "drawer_place_collect_data_v1"
SOURCE_LINK_NAME = "source.hdf5"
MANIFEST_NAME = "manifest.json"
HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CameraSpec:
    source_name: str
    output_dir: str
    compress_len_row: int


CAMERAS = (
    CameraSpec("top", "topImg", 0),
    CameraSpec("left_wrist", "leftImg", 1),
    CameraSpec("right_wrist", "rightImg", 2),
    CameraSpec("top_full", "topFullImg", 3),
)

# Keep source dtypes exactly as read from HDF5. The fruits reference files use
# float64 for some fields, but promoting the drawer_place float32 arrays would
# no longer be a lossless restoration.
OBSERVATION_FIELDS = (
    ("joint_positions", "observations/qpos"),
    ("joint_velocities", "observations/qvel"),
    ("ee_pos_quat", "observations/ee_pos_quat"),
    ("gripper_position", "observations/gripper_position"),
    ("eef_pose", "observations/eef_pose"),
    ("control", "action"),
)

_EPISODE_NAME_RE = re.compile(r"episode_init_(\d+)\.hdf5$")


class DatasetFormatError(RuntimeError):
    """Raised when an input episode cannot be restored losslessly."""


@dataclass(frozen=True)
class EpisodeRecord:
    source_file: str
    episode_dir: str
    frame_count: int
    source_sha256: str


@dataclass(frozen=True)
class RestorationSummary:
    output_dir: Path
    episode_count: int
    frame_count: int
    jpeg_count: int


def _episode_sort_key(path: Path) -> tuple[int, int | str]:
    match = _EPISODE_NAME_RE.fullmatch(path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def find_source_episodes(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    episodes = sorted(
        (path for path in input_dir.glob("episode_init_*.hdf5") if path.is_file()),
        key=_episode_sort_key,
    )
    if not episodes:
        raise FileNotFoundError(
            f"No episode_init_*.hdf5 files found in input directory: {input_dir}"
        )

    invalid_names = [path.name for path in episodes if not _EPISODE_NAME_RE.fullmatch(path.name)]
    if invalid_names:
        raise DatasetFormatError(
            "Source HDF5 files must be named episode_init_<integer>.hdf5, got: "
            + ", ".join(invalid_names)
        )
    return episodes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dataset(root: h5py.Group | h5py.File, path: str) -> h5py.Dataset:
    try:
        item = root[path]
    except KeyError as exc:
        raise DatasetFormatError(f"Missing required dataset /{path}") from exc
    if not isinstance(item, h5py.Dataset):
        raise DatasetFormatError(f"Expected /{path} to be an HDF5 dataset")
    return item


def _validate_source_episode(
    source: h5py.File,
    source_path: Path,
) -> tuple[int, h5py.Dataset, dict[str, h5py.Dataset], dict[str, h5py.Dataset]]:
    action = _require_dataset(source, "action")
    if action.ndim < 1 or action.shape[0] <= 0:
        raise DatasetFormatError(f"{source_path}: /action must contain at least one frame")
    frame_count = int(action.shape[0])

    observation_datasets: dict[str, h5py.Dataset] = {}
    for output_key, source_key in OBSERVATION_FIELDS:
        dataset = _require_dataset(source, source_key)
        if dataset.ndim < 1 or dataset.shape[0] != frame_count:
            raise DatasetFormatError(
                f"{source_path}: /{source_key} has shape {dataset.shape}, "
                f"expected time axis {frame_count}"
            )
        observation_datasets[output_key] = dataset

    compress_len = _require_dataset(source, "compress_len")
    if compress_len.shape != (len(CAMERAS), frame_count):
        raise DatasetFormatError(
            f"{source_path}: /compress_len has shape {compress_len.shape}, expected "
            f"({len(CAMERAS)}, {frame_count})"
        )
    if not np.issubdtype(compress_len.dtype, np.integer):
        raise DatasetFormatError(
            f"{source_path}: /compress_len must have integer dtype, got {compress_len.dtype}"
        )

    image_datasets: dict[str, h5py.Dataset] = {}
    for spec in CAMERAS:
        source_key = f"observations/images/{spec.source_name}"
        dataset = _require_dataset(source, source_key)
        if dataset.ndim != 2 or dataset.shape[0] != frame_count:
            raise DatasetFormatError(
                f"{source_path}: /{source_key} has shape {dataset.shape}, expected "
                f"({frame_count}, encoded_jpeg_bytes)"
            )
        if dataset.dtype != np.dtype(np.uint8):
            raise DatasetFormatError(
                f"{source_path}: /{source_key} must have uint8 dtype, got {dataset.dtype}"
            )
        image_datasets[spec.source_name] = dataset

    return frame_count, compress_len, observation_datasets, image_datasets


def _jpeg_payload(
    encoded_dataset: h5py.Dataset,
    encoded_length: int,
    frame_index: int,
    source_path: Path,
    camera_name: str,
) -> bytes:
    if encoded_length <= 4 or encoded_length > encoded_dataset.shape[1]:
        raise DatasetFormatError(
            f"{source_path}: invalid JPEG length {encoded_length} for "
            f"{camera_name} frame {frame_index}"
        )
    payload = encoded_dataset[frame_index, :encoded_length].tobytes()
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise DatasetFormatError(
            f"{source_path}: {camera_name} frame {frame_index} is not a complete JPEG"
        )
    return payload


def _write_episode(source_path: Path, episode_dir: Path) -> EpisodeRecord:
    episode_dir.mkdir()
    observation_dir = episode_dir / "observation"
    observation_dir.mkdir()
    image_dirs = {spec.source_name: episode_dir / spec.output_dir for spec in CAMERAS}
    for image_dir in image_dirs.values():
        image_dir.mkdir()

    with h5py.File(source_path, "r") as source:
        frame_count, compress_len, observation_datasets, image_datasets = _validate_source_episode(
            source, source_path
        )
        lengths = compress_len[:]

        for frame_index in range(frame_count):
            for spec in CAMERAS:
                payload = _jpeg_payload(
                    image_datasets[spec.source_name],
                    int(lengths[spec.compress_len_row, frame_index]),
                    frame_index,
                    source_path,
                    spec.source_name,
                )
                (image_dirs[spec.source_name] / f"{frame_index}.jpg").write_bytes(payload)

            observation = {
                output_key: np.asarray(dataset[frame_index]).copy()
                for output_key, dataset in observation_datasets.items()
            }
            with (observation_dir / f"{frame_index}.pkl").open("wb") as output_file:
                pickle.dump(observation, output_file, protocol=4)

    # The source and staging directory are preflighted onto the same device.
    # A hard link is an exact archive of datasets and attributes not flattened
    # into the collect_data layout.
    os.link(source_path, episode_dir / SOURCE_LINK_NAME)
    return EpisodeRecord(
        source_file=source_path.name,
        episode_dir=episode_dir.name,
        frame_count=frame_count,
        source_sha256=sha256_file(source_path),
    )


def _arrays_match(expected: np.ndarray, actual: Any) -> bool:
    if not isinstance(actual, np.ndarray):
        return False
    return (
        expected.dtype == actual.dtype
        and expected.shape == actual.shape
        and expected.tobytes() == actual.tobytes()
    )


def _validate_restoration(
    staging_dir: Path,
    source_paths: list[Path],
    records: list[EpisodeRecord],
) -> None:
    if len(source_paths) != len(records):
        raise RuntimeError("Internal error: source and output record counts differ")

    expected_observation_keys = [output_key for output_key, _ in OBSERVATION_FIELDS]
    for source_path, record in zip(source_paths, records, strict=True):
        episode_dir = staging_dir / record.episode_dir
        archive_path = episode_dir / SOURCE_LINK_NAME
        if not archive_path.is_file():
            raise RuntimeError(f"Missing source archive: {archive_path}")
        source_stat = source_path.stat()
        archive_stat = archive_path.stat()
        if (source_stat.st_dev, source_stat.st_ino) != (archive_stat.st_dev, archive_stat.st_ino):
            raise RuntimeError(f"Source archive is not a hard link: {archive_path}")
        if sha256_file(archive_path) != record.source_sha256:
            raise RuntimeError(f"Source archive checksum mismatch: {archive_path}")

        with h5py.File(source_path, "r") as source:
            frame_count, compress_len, observation_datasets, image_datasets = _validate_source_episode(
                source, source_path
            )
            if frame_count != record.frame_count:
                raise RuntimeError(
                    f"Frame count changed while restoring {source_path}: "
                    f"{record.frame_count} -> {frame_count}"
                )
            lengths = compress_len[:]

            for frame_index in range(frame_count):
                for spec in CAMERAS:
                    expected_jpeg = _jpeg_payload(
                        image_datasets[spec.source_name],
                        int(lengths[spec.compress_len_row, frame_index]),
                        frame_index,
                        source_path,
                        spec.source_name,
                    )
                    output_path = episode_dir / spec.output_dir / f"{frame_index}.jpg"
                    if not output_path.is_file() or output_path.read_bytes() != expected_jpeg:
                        raise RuntimeError(f"JPEG payload mismatch: {output_path}")

                observation_path = episode_dir / "observation" / f"{frame_index}.pkl"
                with observation_path.open("rb") as observation_file:
                    restored_observation = pickle.load(observation_file)
                if list(restored_observation.keys()) != expected_observation_keys:
                    raise RuntimeError(f"Observation schema mismatch: {observation_path}")
                for output_key, dataset in observation_datasets.items():
                    expected_value = np.asarray(dataset[frame_index])
                    if not _arrays_match(expected_value, restored_observation[output_key]):
                        raise RuntimeError(
                            f"Observation payload mismatch: {observation_path} ({output_key})"
                        )


def _write_manifest(
    staging_dir: Path,
    input_dir: Path,
    records: list[EpisodeRecord],
) -> None:
    manifest = {
        "format_version": FORMAT_VERSION,
        "source_root": str(input_dir),
        "source_archive": {"filename": SOURCE_LINK_NAME, "mode": "hardlink"},
        "camera_mapping": [
            {
                "source_hdf5_path": f"observations/images/{spec.source_name}",
                "compress_len_row": spec.compress_len_row,
                "output_dir": spec.output_dir,
            }
            for spec in CAMERAS
        ],
        "observation_mapping": [
            {"output_key": output_key, "source_hdf5_path": source_key}
            for output_key, source_key in OBSERVATION_FIELDS
        ],
        "episodes": [asdict(record) for record in records],
    }
    with (staging_dir / MANIFEST_NAME).open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2, sort_keys=True)
        manifest_file.write("\n")


def _preflight(input_dir: Path, output_dir: Path) -> None:
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output directory must not be the input directory or a child of it")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {output_dir.parent}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty and will not be overwritten: {output_dir}"
            )
    if os.stat(input_dir).st_dev != os.stat(output_dir.parent).st_dev:
        raise OSError(
            "Input and output must be on the same filesystem so source.hdf5 can "
            "be preserved as a hard link"
        )


def _publish_staging_dir(staging_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory changed while restoring and will not be overwritten: "
                f"{output_dir}"
            )
        output_dir.rmdir()
    os.replace(staging_dir, output_dir)


def restore_collect_data(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> RestorationSummary:
    """Restore all drawer_place source episodes and publish a validated output tree."""
    source_root = Path(input_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    source_paths = find_source_episodes(source_root)
    _preflight(source_root, destination)

    staging_dir = destination.parent / f".{destination.name}.restore-{uuid.uuid4().hex}.tmp"
    staging_dir.mkdir()
    try:
        records = [_write_episode(source_path, staging_dir / source_path.stem) for source_path in source_paths]
        _validate_restoration(staging_dir, source_paths, records)
        _write_manifest(staging_dir, source_root, records)
        _publish_staging_dir(staging_dir, destination)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    frame_count = sum(record.frame_count for record in records)
    return RestorationSummary(
        output_dir=destination,
        episode_count=len(records),
        frame_count=frame_count,
        jpeg_count=frame_count * len(CAMERAS),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore drawer_place train_data HDF5 files into collect_data layout"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = restore_collect_data(args.input_dir, args.output_dir)
    print(
        f"Restored {summary.episode_count} episodes, {summary.frame_count} frames, "
        f"and {summary.jpeg_count} JPEGs to {summary.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
