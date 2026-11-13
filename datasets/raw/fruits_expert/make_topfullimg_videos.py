#!/usr/bin/env python3
"""Convert timestamped topFullImg frames into MP4 and optional FineProg H5 files."""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np


# ---------------------------------------------------------------------------
# Hard-coded configuration. Edit these values before running if needed.
# ---------------------------------------------------------------------------
INPUT_ROOT = Path(
    "/home/user/zhangzk/projects/fineprog/datasets/raw/fruits_expert/fruits_new/collect_data"
)
OUTPUT_ROOT = Path(
    "/home/user/zhangzk/projects/fineprog/datasets/raw/fruit_expert_videos"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
H5_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed"
FRAME_DIR_NAME = "topFullImg"
FPS = 20.0
VIDEO_EXTENSION = ".mp4"
VIDEO_NAME_PREFIX = "demo"
VIDEO_CODEC = "libx264"
VIDEO_QUALITY = 8
OUTPUT_PIXEL_FORMAT = "yuv420p"
OVERWRITE_EXISTING = True
ENABLE_CROP = True
ENABLE_DUPLICATE_FRAME_FILTER = False
ENABLE_FRAME_INDEX_OVERLAY = False
CROP_SIZE = (224, 224)  # (width, height), used when ENABLE_CROP is True
# Keep this identical to the configuration used to produce pick_mango_vid:
# crop a 224x224 box from the 640x480 topFullImg source frames. H is downward
# from the top; W is rightward from the left.
CROP_OFFSET_H = 168
CROP_OFFSET_W = 256
CROP_PREVIEW_FILENAME = "crop_preview.jpg"
CROP_BOX_COLOR = (0, 255, 0)  # BGR: green
CROP_BOX_THICKNESS = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
H5_IMAGE_SIZE = 224
H5_CHUNK_LEN = 8
TRAIN_COUNT = 40
VALID_COUNT = 4
CALIBRATION_COUNT = 18
FAIL_COUNT = 18
SPLIT_SEED = 42


@dataclass(frozen=True)
class ClipConversion:
    """One MP4 conversion selected from the original, timestamp-sorted demos."""

    source_demo_id: int
    output_demo_id: int
    timestamp_dir: Path
    start_idx: int
    end_idx: int


@dataclass(frozen=True)
class ClipH5Split:
    """One deterministic H5 split over consecutively numbered clipped demos."""

    label: str
    split: str
    conversions: tuple[ClipConversion, ...]


def natural_sort_key(path: Path) -> list[tuple[int, object]]:
    """Sort names such as 2.jpg before 10.jpg."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [(0, int(part)) if part.isdigit() else (1, part) for part in parts]


def find_frames(frame_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )


def warn_about_missing_numeric_frames(frames: list[Path], timestamp: str) -> None:
    numeric_indices = [int(frame.stem) for frame in frames if frame.stem.isdigit()]
    if len(numeric_indices) != len(frames) or not numeric_indices:
        return

    missing = sorted(
        set(range(numeric_indices[0], numeric_indices[-1] + 1))
        - set(numeric_indices)
    )
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        suffix = " ..." if len(missing) > 10 else ""
        print(
            f"  Warning: {timestamp} is missing {len(missing)} numeric frame(s): "
            f"{preview}{suffix}",
            file=sys.stderr,
        )


def crop_bounds(width: int, height: int) -> tuple[int, int, int, int]:
    """Return left, top, right, bottom for the configured crop box."""
    crop_width, crop_height = CROP_SIZE
    left = CROP_OFFSET_W
    top = CROP_OFFSET_H
    right = left + crop_width
    bottom = top + crop_height
    if left < 0 or top < 0 or right > width or bottom > height:
        raise RuntimeError(
            f"Crop box is outside the image: image={width}x{height}, "
            f"crop={crop_width}x{crop_height}, "
            f"offset(H,W)=({CROP_OFFSET_H},{CROP_OFFSET_W})"
        )
    return left, top, right, bottom


def prepare_frame(
    frame,
    frame_path: Path,
    expected_source_size: tuple[int, int],
):
    """Validate a frame and optionally return its configured crop."""
    source_width, source_height = expected_source_size
    height, width = frame.shape[:2]
    if (width, height) != expected_source_size:
        raise RuntimeError(
            f"Frame size mismatch in {frame_path}: "
            f"expected {source_width}x{source_height}, got {width}x{height}"
        )

    if not ENABLE_CROP:
        return frame

    try:
        left, top, right, bottom = crop_bounds(width, height)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} ({frame_path})") from exc
    return frame[top:bottom, left:right]


def load_processed_frames(
    timestamp_dir: Path,
    *,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Load one episode, optionally select one-based frames, then crop and filter."""
    frame_dir = timestamp_dir / FRAME_DIR_NAME
    if not frame_dir.is_dir():
        raise RuntimeError(f"Missing {FRAME_DIR_NAME}: {frame_dir}")

    frame_paths = find_frames(frame_dir)
    if not frame_paths:
        raise RuntimeError(f"No supported images found in {frame_dir}")
    warn_about_missing_numeric_frames(frame_paths, timestamp_dir.name)
    source_frame_count = len(frame_paths)

    if start_idx is not None or end_idx is not None:
        if start_idx is None or end_idx is None:
            raise ValueError("Both start_idx and end_idx must be provided together")
        if start_idx < 1 or start_idx > end_idx or end_idx > source_frame_count:
            raise ValueError(
                f"Invalid one-based frame range {start_idx}..{end_idx} for "
                f"{timestamp_dir.name}; source has {source_frame_count} frames"
            )
        # The source script's optional overlay displays frame_index + 1, so the
        # CSV's one-based inclusive range maps directly to this slice.
        frame_paths = frame_paths[start_idx - 1 : end_idx]

    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    source_height, source_width = first_frame.shape[:2]
    expected_source_size = (source_width, source_height)

    output_frames: list[np.ndarray] = []
    duplicates_removed = 0
    seen_frame_bytes: set[bytes] = set()
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")
        frame = prepare_frame(frame, frame_path, expected_source_size)
        frame_rgb = np.ascontiguousarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )

        if ENABLE_DUPLICATE_FRAME_FILTER:
            frame_bytes = frame_rgb.tobytes()
            if frame_bytes in seen_frame_bytes:
                duplicates_removed += 1
                continue
            seen_frame_bytes.add(frame_bytes)
        output_frames.append(frame_rgb)

    if not output_frames:
        raise RuntimeError(f"No frames remain after filtering: {frame_dir}")
    return np.stack(output_frames, axis=0), source_frame_count, duplicates_removed


def build_clip_conversion_plan(
    timestamp_dirs: list[Path], clip_csv: Path
) -> list[ClipConversion]:
    """Validate CSV clipping metadata and assign consecutive output demo IDs."""
    if not clip_csv.is_file():
        raise FileNotFoundError(f"Clip CSV does not exist: {clip_csv}")

    with clip_csv.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        expected_headers = ["demo_id", "start_idx", "end_idx"]
        if reader.fieldnames != expected_headers:
            raise ValueError(
                f"Clip CSV must have headers {expected_headers}, got {reader.fieldnames}"
            )
        rows = list(reader)

    if len(rows) != len(timestamp_dirs):
        raise ValueError(
            f"Clip CSV has {len(rows)} rows but input has {len(timestamp_dirs)} demos"
        )

    conversions: list[ClipConversion] = []
    for expected_demo_id, row in enumerate(rows):
        demo_id_text = (row["demo_id"] or "").strip()
        try:
            source_demo_id = int(demo_id_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid demo_id {demo_id_text!r} in CSV row {expected_demo_id + 2}"
            ) from exc
        if source_demo_id != expected_demo_id:
            raise ValueError(
                "Clip CSV demo_id values must be the consecutive original IDs "
                f"0..{len(timestamp_dirs) - 1}; row {expected_demo_id + 2} has "
                f"{source_demo_id}"
            )

        start_text = (row["start_idx"] or "").strip()
        end_text = (row["end_idx"] or "").strip()
        if end_text.lower() == "fault":
            continue

        try:
            start_idx = int(start_text)
            end_idx = int(end_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid frame range {start_text!r}..{end_text!r} for demo "
                f"{source_demo_id}"
            ) from exc

        timestamp_dir = timestamp_dirs[source_demo_id]
        frame_dir = timestamp_dir / FRAME_DIR_NAME
        frame_count = len(find_frames(frame_dir)) if frame_dir.is_dir() else 0
        if not frame_count:
            raise RuntimeError(f"Missing or empty {FRAME_DIR_NAME}: {frame_dir}")
        if start_idx < 1 or start_idx > end_idx or end_idx > frame_count:
            raise ValueError(
                f"Invalid one-based frame range {start_idx}..{end_idx} for original "
                f"demo {source_demo_id} ({timestamp_dir.name}); source has "
                f"{frame_count} frames"
            )

        conversions.append(
            ClipConversion(
                source_demo_id=source_demo_id,
                output_demo_id=len(conversions),
                timestamp_dir=timestamp_dir,
                start_idx=start_idx,
                end_idx=end_idx,
            )
        )

    if not conversions:
        raise ValueError(f"Clip CSV contains no non-fault demos: {clip_csv}")
    return conversions


def build_clip_h5_splits(
    conversions: list[ClipConversion],
) -> tuple[ClipH5Split, ...]:
    """Build the documented 36-train, 4-valid and 14-valid H5 splits."""
    expected_ids = list(range(54))
    output_ids = [conversion.output_demo_id for conversion in conversions]
    if output_ids != expected_ids:
        raise ValueError(
            "CSV H5 generation requires clipped output demo IDs 0..53; got "
            f"{output_ids}"
        )

    def select(
        label: str, split: str, start_id: int, end_id: int
    ) -> ClipH5Split:
        selected = tuple(conversions[start_id : end_id + 1])
        selected_ids = [conversion.output_demo_id for conversion in selected]
        expected = list(range(start_id, end_id + 1))
        if selected_ids != expected:
            raise ValueError(
                f"Invalid {label} H5 split: expected demo IDs {expected}, "
                f"got {selected_ids}"
            )
        return ClipH5Split(label=label, split=split, conversions=selected)

    return (
        select("train", "train", 0, 35),
        select("small_valid", "valid", 36, 39),
        select("large_valid", "valid", 40, 53),
    )


def save_crop_preview(timestamp_dirs: list[Path]) -> Path | None:
    """Draw the crop box on the first frame of the first valid episode."""
    if not ENABLE_CROP:
        print("[skip] Crop preview disabled because ENABLE_CROP is False")
        return None

    for timestamp_dir in timestamp_dirs:
        frame_dir = timestamp_dir / FRAME_DIR_NAME
        if not frame_dir.is_dir():
            continue
        frames = find_frames(frame_dir)
        if not frames:
            continue

        first_frame_path = frames[0]
        frame = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read crop preview frame: {first_frame_path}")

        height, width = frame.shape[:2]
        left, top, right, bottom = crop_bounds(width, height)
        preview = frame.copy()
        cv2.rectangle(
            preview,
            (left, top),
            (right - 1, bottom - 1),
            CROP_BOX_COLOR,
            CROP_BOX_THICKNESS,
        )

        label = (
            f"crop {CROP_SIZE[0]}x{CROP_SIZE[1]} "
            f"offset H={CROP_OFFSET_H}, W={CROP_OFFSET_W}"
        )
        label_origin = (left, max(24, top - 10))
        cv2.putText(
            preview,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            CROP_BOX_COLOR,
            2,
            cv2.LINE_AA,
        )

        output_path = OUTPUT_ROOT / CROP_PREVIEW_FILENAME
        if not cv2.imwrite(str(output_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write crop preview: {output_path}")
        print(f"[done] Crop preview from {first_frame_path} -> {output_path}")
        return output_path

    raise RuntimeError(f"No frames found for crop preview under {INPUT_ROOT}")


def add_frame_index_overlay(frame_rgb: np.ndarray, frame_index: int) -> np.ndarray:
    """Return an RGB video frame with its one-based index at the top left."""
    annotated = frame_rgb.copy()
    text = str(frame_index + 1)
    origin = (8, 30)
    cv2.putText(
        annotated,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def convert_timestamp(
    timestamp_dir: Path,
    video_index: int,
    *,
    source_demo_id: int | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> tuple[str, int, int]:
    timestamp = timestamp_dir.name
    frame_dir = timestamp_dir / FRAME_DIR_NAME
    output_name = f"{VIDEO_NAME_PREFIX}_{video_index}{VIDEO_EXTENSION}"
    output_path = OUTPUT_ROOT / output_name
    temporary_path = OUTPUT_ROOT / (
        f".{VIDEO_NAME_PREFIX}_{video_index}.partial{VIDEO_EXTENSION}"
    )

    if not frame_dir.is_dir():
        return "missing", 0, 0

    if output_path.exists() and not OVERWRITE_EXISTING:
        print(f"[skip] {output_path} already exists")
        if start_idx is not None and end_idx is not None:
            return "skipped", end_idx - start_idx + 1, 0
        return "skipped", len(find_frames(frame_dir)), 0

    frames, source_frame_count, duplicates_removed = load_processed_frames(
        timestamp_dir, start_idx=start_idx, end_idx=end_idx
    )
    output_height, output_width = frames.shape[1:3]

    # A temporary file prevents a failed conversion from looking complete.
    if temporary_path.exists():
        temporary_path.unlink()

    # Match the project's existing video generation path: FFmpeg/libx264
    # produces H.264 video that VS Code's Chromium-based player can decode.
    writer = imageio.get_writer(
        str(temporary_path),
        fps=FPS,
        format="ffmpeg",
        codec=VIDEO_CODEC,
        quality=VIDEO_QUALITY,
        pixelformat=OUTPUT_PIXEL_FORMAT,
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )

    try:
        for frame_index, frame_rgb in enumerate(frames):
            # In clip mode retain the source-video labels: start_idx is the
            # one-based label shown for the first selected frame.
            overlay_index = (
                start_idx - 1 + frame_index
                if start_idx is not None
                else frame_index
            )
            output_frame = (
                add_frame_index_overlay(frame_rgb, overlay_index)
                if ENABLE_FRAME_INDEX_OVERLAY
                else frame_rgb
            )
            writer.append_data(output_frame)
    except Exception:
        writer.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    else:
        writer.close()

    written = len(frames)
    if not temporary_path.is_file():
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(
            f"Incomplete output for {timestamp}: expected {written} output frames"
        )

    os.replace(temporary_path, output_path)
    duration = written / FPS
    if ENABLE_DUPLICATE_FRAME_FILTER and duplicates_removed:
        print(
            f"[filter] {timestamp}: duplicate RGB frames detected="
            f"{duplicates_removed}, removed={duplicates_removed}, "
            f"kept={written}/{source_frame_count}"
        )
    source_description = timestamp
    if source_demo_id is not None:
        source_description = (
            f"demo_{video_index} <- original demo_{source_demo_id} "
            f"({timestamp}, frames {start_idx}..{end_idx})"
        )
    print(
        f"[done] {source_description}: {written} output frames from "
        f"{source_frame_count} source frames, {output_width}x{output_height}, "
        f"{duration:.2f}s -> {output_path}"
    )
    return "converted", written, duplicates_removed


def remove_stale_clip_videos(output_count: int, source_count: int) -> list[Path]:
    """Remove old tail demos after a successful clip-mode renumbering."""
    removed: list[Path] = []
    for video_index in range(output_count, source_count):
        output_path = OUTPUT_ROOT / f"{VIDEO_NAME_PREFIX}_{video_index}{VIDEO_EXTENSION}"
        if output_path.is_file():
            output_path.unlink()
            removed.append(output_path)
    return removed


def build_split_plan(
    timestamp_dirs: list[Path],
    *,
    train_count: int,
    valid_count: int,
    calibration_count: int,
    fail_count: int,
    split_seed: int,
) -> dict[str, list[Path]]:
    """Build reproducible train/valid/calibration/fail episode lists."""
    total = len(timestamp_dirs)
    counts = {
        "train_count": train_count,
        "valid_count": valid_count,
        "calibration_count": calibration_count,
        "fail_count": fail_count,
    }
    if any(isinstance(value, bool) or value <= 0 for value in counts.values()):
        raise ValueError(f"All split counts must be positive integers: {counts}")
    if valid_count >= train_count:
        raise ValueError(
            f"valid_count must be smaller than train_count, got "
            f"{valid_count} >= {train_count}"
        )
    if train_count + calibration_count + fail_count != total:
        raise ValueError(
            "train_count + calibration_count + fail_count must equal the "
            f"number of episodes ({total}), got "
            f"{train_count} + {calibration_count} + {fail_count}"
        )

    candidate_count = train_count + calibration_count
    shuffled_candidates = list(timestamp_dirs[:candidate_count])
    random.Random(split_seed).shuffle(shuffled_candidates)
    train_candidates = shuffled_candidates[:train_count]
    return {
        "train": train_candidates[: train_count - valid_count],
        "valid": train_candidates[train_count - valid_count :],
        "calibration": shuffled_candidates[train_count:],
        "fail": list(timestamp_dirs[candidate_count:]),
    }


def _normalise_h5_prefix(prefix: str) -> str:
    path = Path(prefix)
    if not prefix or path.name != prefix or prefix in {".", ".."}:
        raise ValueError(f"H5 prefix must be a filename prefix, got {prefix!r}")
    return prefix


def build_h5_output_paths(
    split_plan: dict[str, list[Path]], h5_output_dir: Path, h5_prefix: str
) -> dict[str, Path]:
    prefix = _normalise_h5_prefix(h5_prefix)
    return {
        split_name: h5_output_dir
        / f"{prefix}-{len(episodes)}vid_{split_name}.h5"
        for split_name, episodes in split_plan.items()
    }


def _write_h5_file(
    output_path: Path,
    split_name: str,
    episodes: list[Path],
    episode_indices: dict[Path, int],
    split_seed: int,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with h5py.File(output_path, "w", libver="earliest") as h5file:
        h5file.attrs["split"] = split_name
        h5file.attrs["split_seed"] = split_seed
        h5file.attrs["source_root"] = str(INPUT_ROOT)
        h5file.attrs["duplicate_filter"] = ENABLE_DUPLICATE_FRAME_FILTER
        h5file.attrs["crop_offset_h"] = CROP_OFFSET_H
        h5file.attrs["crop_offset_w"] = CROP_OFFSET_W
        videos = h5file.create_group("videos")

        for local_index, episode in enumerate(episodes, start=1):
            demo_index = episode_indices[episode]
            action_name = f"{VIDEO_NAME_PREFIX}_{demo_index}"
            video_path = OUTPUT_ROOT / f"{action_name}{VIDEO_EXTENSION}"
            source_frame_dir = episode / FRAME_DIR_NAME
            frames, source_frame_count, duplicates_removed = load_processed_frames(
                episode
            )

            video_id = f"{local_index:06d}"
            video = videos.create_group(video_id)
            video.create_dataset(
                "frames",
                data=frames,
                dtype=np.uint8,
                compression=None,
                chunks=(
                    min(H5_CHUNK_LEN, len(frames)),
                    H5_IMAGE_SIZE,
                    H5_IMAGE_SIZE,
                    3,
                ),
            )
            video.attrs["action_name"] = action_name
            video.attrs["action_id"] = 0
            video.attrs["fps"] = FPS
            video.attrs["num_frames"] = len(frames)
            video.attrs["path"] = str(video_path)
            video.attrs["source_episode"] = episode.name
            video.attrs["source_frame_dir"] = str(source_frame_dir)

            records.append(
                {
                    "processed_h5_name": "",
                    "video_idx": video_id,
                    "source_video_name": video_path.name,
                    "source_video_path": str(video_path),
                    "source_episode": episode.name,
                    "source_frame_dir": str(source_frame_dir),
                }
            )
            print(
                f"[h5:{split_name}] {video_id} <- {action_name} "
                f"({len(frames)}/{source_frame_count} frames, "
                f"duplicates_removed={duplicates_removed})"
            )
        h5file.flush()
    return records


def _validate_h5(output_path: Path, episodes: list[Path]) -> None:
    with h5py.File(output_path, "r") as h5file:
        videos = h5file.get("videos")
        if not isinstance(videos, h5py.Group) or len(videos) != len(episodes):
            raise RuntimeError(f"Invalid videos group in {output_path}")
        expected_ids = [f"{index:06d}" for index in range(1, len(episodes) + 1)]
        if sorted(videos.keys()) != expected_ids:
            raise RuntimeError(f"Invalid video IDs in {output_path}")
        for video_id in expected_ids:
            video = videos[video_id]
            frames = video.get("frames")
            if not isinstance(frames, h5py.Dataset):
                raise RuntimeError(f"Missing /videos/{video_id}/frames in {output_path}")
            if frames.dtype != np.uint8 or frames.shape[1:] != (
                H5_IMAGE_SIZE,
                H5_IMAGE_SIZE,
                3,
            ):
                raise RuntimeError(
                    f"Invalid frames at {output_path}:/videos/{video_id}: "
                    f"shape={frames.shape}, dtype={frames.dtype}"
                )
            if int(video.attrs.get("num_frames", -1)) != len(frames):
                raise RuntimeError(f"Frame-count mismatch in {output_path}:{video_id}")


def _write_mapping_csv(
    csv_path: Path, h5_path: Path, records: list[dict[str, str]]
) -> None:
    fieldnames = [
        "processed_h5_name",
        "video_idx",
        "source_video_name",
        "source_video_path",
        "source_episode",
        "source_frame_dir",
    ]
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "processed_h5_name": h5_path.name})


def write_h5_splits(
    timestamp_dirs: list[Path],
    split_plan: dict[str, list[Path]],
    *,
    h5_output_dir: Path,
    h5_prefix: str,
    split_seed: int,
    overwrite: bool,
) -> list[Path]:
    """Write and validate every split before publishing the final H5 files."""
    output_paths = build_h5_output_paths(split_plan, h5_output_dir, h5_prefix)
    mapping_dir = h5_output_dir / "idx_mapping"
    mapping_paths = {
        name: mapping_dir / f"{path.stem}_idx_mapping.csv"
        for name, path in output_paths.items()
    }
    existing = [
        path
        for path in [*output_paths.values(), *mapping_paths.values()]
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "H5 output already exists; pass --overwrite-h5 to replace it: "
            + ", ".join(map(str, existing))
        )

    h5_output_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir.mkdir(parents=True, exist_ok=True)
    episode_indices = {episode: index for index, episode in enumerate(timestamp_dirs)}
    temporary_h5: dict[str, Path] = {}
    temporary_csv: dict[str, Path] = {}
    try:
        for split_name, episodes in split_plan.items():
            final_h5 = output_paths[split_name]
            temp_h5 = final_h5.with_name(
                f".{final_h5.name}.{uuid.uuid4().hex}.tmp"
            )
            temporary_h5[split_name] = temp_h5
            records = _write_h5_file(
                temp_h5, split_name, episodes, episode_indices, split_seed
            )
            _validate_h5(temp_h5, episodes)

            final_csv = mapping_paths[split_name]
            temp_csv = final_csv.with_name(
                f".{final_csv.name}.{uuid.uuid4().hex}.tmp"
            )
            temporary_csv[split_name] = temp_csv
            _write_mapping_csv(temp_csv, final_h5, records)

        for split_name in split_plan:
            os.replace(temporary_h5[split_name], output_paths[split_name])
            os.replace(temporary_csv[split_name], mapping_paths[split_name])
            print(f"[h5:done] {output_paths[split_name]}")
    except Exception:
        for temporary_path in [*temporary_h5.values(), *temporary_csv.values()]:
            temporary_path.unlink(missing_ok=True)
        raise
    return [output_paths[name] for name in split_plan]


def build_clip_h5_output_paths(
    h5_splits: tuple[ClipH5Split, ...],
    h5_output_dir: Path,
    h5_prefix: str,
) -> dict[str, Path]:
    """Return the date-prefixed H5 path for each documented clip split."""
    prefix = _normalise_h5_prefix(h5_prefix)
    output_paths = {
        h5_split.label: h5_output_dir
        / f"{prefix}-{len(h5_split.conversions)}vid_{h5_split.split}.h5"
        for h5_split in h5_splits
    }
    if len(set(output_paths.values())) != len(output_paths):
        raise ValueError(f"Clip H5 output paths are not unique: {output_paths}")
    return output_paths


def validate_corresponding_clip_mp4s(
    conversions: list[ClipConversion],
) -> None:
    """Require the already-generated MP4 names used by H5 metadata to exist."""
    missing = [
        OUTPUT_ROOT / f"{VIDEO_NAME_PREFIX}_{conversion.output_demo_id}{VIDEO_EXTENSION}"
        for conversion in conversions
        if not (
            OUTPUT_ROOT
            / f"{VIDEO_NAME_PREFIX}_{conversion.output_demo_id}{VIDEO_EXTENSION}"
        ).is_file()
    ]
    if missing:
        preview = ", ".join(path.name for path in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} corresponding clipped MP4 file(s): "
            f"{preview}{suffix}"
        )


def _write_clip_h5_file(
    output_path: Path,
    h5_split: ClipH5Split,
    clip_csv: Path,
) -> list[dict[str, str | int]]:
    """Write one H5 directly from the selected, cropped source image frames."""
    records: list[dict[str, str | int]] = []
    with h5py.File(output_path, "w", libver="earliest") as h5file:
        h5file.attrs["split"] = h5_split.split
        h5file.attrs["source_root"] = str(INPUT_ROOT.resolve())
        h5file.attrs["clip_csv"] = str(clip_csv.resolve())
        h5file.attrs["frame_index_base"] = 1
        h5file.attrs["duplicate_filter"] = ENABLE_DUPLICATE_FRAME_FILTER
        h5file.attrs["crop_offset_h"] = CROP_OFFSET_H
        h5file.attrs["crop_offset_w"] = CROP_OFFSET_W
        videos = h5file.create_group("videos")

        for local_index, conversion in enumerate(h5_split.conversions, start=1):
            action_name = f"{VIDEO_NAME_PREFIX}_{conversion.output_demo_id}"
            video_path = OUTPUT_ROOT / f"{action_name}{VIDEO_EXTENSION}"
            source_frame_dir = conversion.timestamp_dir / FRAME_DIR_NAME
            frames, source_frame_count, duplicates_removed = load_processed_frames(
                conversion.timestamp_dir,
                start_idx=conversion.start_idx,
                end_idx=conversion.end_idx,
            )
            expected_frame_count = conversion.end_idx - conversion.start_idx + 1
            if len(frames) != expected_frame_count:
                raise RuntimeError(
                    f"Clip frame-count mismatch for {action_name}: expected "
                    f"{expected_frame_count}, got {len(frames)}"
                )

            video_id = f"{local_index:06d}"
            video = videos.create_group(video_id)
            video.create_dataset(
                "frames",
                data=frames,
                dtype=np.uint8,
                compression=None,
                chunks=(
                    min(H5_CHUNK_LEN, len(frames)),
                    H5_IMAGE_SIZE,
                    H5_IMAGE_SIZE,
                    3,
                ),
            )
            video.attrs["action_name"] = action_name
            video.attrs["action_id"] = 0
            video.attrs["fps"] = FPS
            video.attrs["num_frames"] = len(frames)
            video.attrs["path"] = str(video_path.resolve())
            video.attrs["source_episode"] = conversion.timestamp_dir.name
            video.attrs["source_frame_dir"] = str(source_frame_dir.resolve())
            video.attrs["source_demo_id"] = conversion.source_demo_id
            video.attrs["source_frame_count"] = source_frame_count
            video.attrs["source_start_idx"] = conversion.start_idx
            video.attrs["source_end_idx"] = conversion.end_idx

            records.append(
                {
                    "processed_h5_name": "",
                    "video_idx": video_id,
                    "output_demo_id": conversion.output_demo_id,
                    "source_video_name": video_path.name,
                    "source_video_path": str(video_path.resolve()),
                    "source_episode": conversion.timestamp_dir.name,
                    "source_frame_dir": str(source_frame_dir.resolve()),
                    "source_demo_id": conversion.source_demo_id,
                    "source_start_idx": conversion.start_idx,
                    "source_end_idx": conversion.end_idx,
                }
            )
            print(
                f"[h5:{h5_split.label}] {video_id} <- {action_name} "
                f"(original demo {conversion.source_demo_id}, frames "
                f"{conversion.start_idx}..{conversion.end_idx}, "
                f"kept={len(frames)}/{source_frame_count}, "
                f"duplicates_removed={duplicates_removed})"
            )
        h5file.flush()
    return records


def _validate_clip_h5(output_path: Path, h5_split: ClipH5Split) -> None:
    """Validate H5 structure and metadata against its direct-image clip plan."""
    with h5py.File(output_path, "r") as h5file:
        if h5file.attrs.get("split") != h5_split.split:
            raise RuntimeError(f"Invalid split attribute in {output_path}")
        videos = h5file.get("videos")
        if not isinstance(videos, h5py.Group) or len(videos) != len(
            h5_split.conversions
        ):
            raise RuntimeError(f"Invalid videos group in {output_path}")

        expected_ids = [
            f"{index:06d}" for index in range(1, len(h5_split.conversions) + 1)
        ]
        if sorted(videos.keys()) != expected_ids:
            raise RuntimeError(f"Invalid video IDs in {output_path}")

        for video_id, conversion in zip(expected_ids, h5_split.conversions):
            video = videos[video_id]
            frames = video.get("frames")
            expected_frame_count = conversion.end_idx - conversion.start_idx + 1
            if not isinstance(frames, h5py.Dataset):
                raise RuntimeError(f"Missing frames at {output_path}:{video_id}")
            if frames.dtype != np.uint8 or frames.shape != (
                expected_frame_count,
                H5_IMAGE_SIZE,
                H5_IMAGE_SIZE,
                3,
            ):
                raise RuntimeError(
                    f"Invalid frames at {output_path}:{video_id}: "
                    f"shape={frames.shape}, dtype={frames.dtype}"
                )

            action_name = f"{VIDEO_NAME_PREFIX}_{conversion.output_demo_id}"
            expected_video_path = str(
                (OUTPUT_ROOT / f"{action_name}{VIDEO_EXTENSION}").resolve()
            )
            expected_attrs = {
                "action_name": action_name,
                "num_frames": expected_frame_count,
                "path": expected_video_path,
                "source_episode": conversion.timestamp_dir.name,
                "source_demo_id": conversion.source_demo_id,
                "source_start_idx": conversion.start_idx,
                "source_end_idx": conversion.end_idx,
            }
            for name, expected_value in expected_attrs.items():
                actual_value = video.attrs.get(name)
                if actual_value != expected_value:
                    raise RuntimeError(
                        f"Invalid {name} at {output_path}:{video_id}: "
                        f"expected {expected_value!r}, got {actual_value!r}"
                    )


def _write_clip_mapping_csv(
    csv_path: Path,
    h5_path: Path,
    records: list[dict[str, str | int]],
) -> None:
    fieldnames = [
        "processed_h5_name",
        "video_idx",
        "output_demo_id",
        "source_video_name",
        "source_video_path",
        "source_episode",
        "source_frame_dir",
        "source_demo_id",
        "source_start_idx",
        "source_end_idx",
    ]
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "processed_h5_name": h5_path.name})


def write_clip_h5_splits(
    h5_splits: tuple[ClipH5Split, ...],
    clip_csv: Path,
    *,
    h5_output_dir: Path,
    h5_prefix: str,
    overwrite: bool,
) -> list[Path]:
    """Write, validate and publish the documented clipped-image H5 datasets."""
    output_paths = build_clip_h5_output_paths(h5_splits, h5_output_dir, h5_prefix)
    mapping_dir = h5_output_dir / "idx_mapping"
    mapping_paths = {
        h5_split.label: mapping_dir / f"{output_paths[h5_split.label].stem}_idx_mapping.csv"
        for h5_split in h5_splits
    }
    existing = [
        path
        for path in [*output_paths.values(), *mapping_paths.values()]
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Clipped H5 output already exists; pass --overwrite-h5 to replace it: "
            + ", ".join(map(str, existing))
        )

    h5_output_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir.mkdir(parents=True, exist_ok=True)
    temporary_h5: dict[str, Path] = {}
    temporary_csv: dict[str, Path] = {}
    try:
        for h5_split in h5_splits:
            label = h5_split.label
            final_h5 = output_paths[label]
            temp_h5 = final_h5.with_name(
                f".{final_h5.name}.{uuid.uuid4().hex}.tmp"
            )
            temporary_h5[label] = temp_h5
            records = _write_clip_h5_file(temp_h5, h5_split, clip_csv)
            _validate_clip_h5(temp_h5, h5_split)

            final_csv = mapping_paths[label]
            temp_csv = final_csv.with_name(
                f".{final_csv.name}.{uuid.uuid4().hex}.tmp"
            )
            temporary_csv[label] = temp_csv
            _write_clip_mapping_csv(temp_csv, final_h5, records)

        for h5_split in h5_splits:
            label = h5_split.label
            os.replace(temporary_h5[label], output_paths[label])
            os.replace(temporary_csv[label], mapping_paths[label])
            print(f"[h5:done] {output_paths[label]}")
    except Exception:
        for temporary_path in [*temporary_h5.values(), *temporary_csv.values()]:
            temporary_path.unlink(missing_ok=True)
        raise
    return [output_paths[h5_split.label] for h5_split in h5_splits]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert topFullImg frames from each timestamp directory into MP4 videos."
    )
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--clip-csv",
        type=Path,
        help=(
            "CSV with demo_id,start_idx,end_idx; fault rows are skipped and "
            "valid rows are renumbered consecutively"
        ),
    )
    parser.add_argument(
        "--skip-crop-preview",
        action="store_true",
        help="do not write or update crop_preview.jpg",
    )
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument(
        "--overlay-frame-index",
        action="store_true",
        help="draw the one-based source-frame index at the top left of each MP4 frame",
    )
    parser.add_argument(
        "--filter-duplicates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "remove bitwise-identical RGB frames; overrides "
            "ENABLE_DUPLICATE_FRAME_FILTER"
        ),
    )
    parser.add_argument(
        "--write-h5",
        action="store_true",
        help="also write train, valid, calibration and fail FineProg H5 files",
    )
    parser.add_argument(
        "--h5-only",
        action="store_true",
        help=(
            "with --clip-csv --write-h5, write direct-image H5 files without "
            "rewriting MP4 files or crop_preview.jpg"
        ),
    )
    parser.add_argument("--train-count", type=int, default=TRAIN_COUNT)
    parser.add_argument("--valid-count", type=int, default=VALID_COUNT)
    parser.add_argument(
        "--calibration-count", type=int, default=CALIBRATION_COUNT
    )
    parser.add_argument("--fail-count", type=int, default=FAIL_COUNT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--h5-output-dir",
        type=Path,
        default=H5_OUTPUT_DIR,
        help=f"H5 output directory (default: {H5_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--h5-prefix",
        help="H5 filename prefix (default: the --output-root directory name)",
    )
    parser.add_argument(
        "--overwrite-h5",
        action="store_true",
        help="replace existing H5 and index-mapping outputs",
    )
    return parser.parse_args()


def main() -> int:
    global INPUT_ROOT, OUTPUT_ROOT, ENABLE_CROP, ENABLE_DUPLICATE_FRAME_FILTER
    global ENABLE_FRAME_INDEX_OVERLAY
    args = parse_args()
    INPUT_ROOT = args.input_root
    OUTPUT_ROOT = args.output_root
    ENABLE_CROP = not args.no_crop
    ENABLE_FRAME_INDEX_OVERLAY = args.overlay_frame_index
    if args.filter_duplicates is not None:
        ENABLE_DUPLICATE_FRAME_FILTER = args.filter_duplicates

    if args.h5_only and not args.write_h5:
        print("--h5-only requires --write-h5", file=sys.stderr)
        return 2
    if args.h5_only and args.clip_csv is None:
        print("--h5-only requires --clip-csv", file=sys.stderr)
        return 2

    if not INPUT_ROOT.is_dir():
        print(f"Input root does not exist: {INPUT_ROOT}", file=sys.stderr)
        return 2
    if not VIDEO_CODEC:
        print("VIDEO_CODEC must not be empty.", file=sys.stderr)
        return 2
    if FPS <= 0:
        print("FPS must be greater than zero.", file=sys.stderr)
        return 2
    if ENABLE_CROP and (len(CROP_SIZE) != 2 or any(size <= 0 for size in CROP_SIZE)):
        print("CROP_SIZE must contain two positive integers.", file=sys.stderr)
        return 2
    if ENABLE_CROP and (
        not isinstance(CROP_OFFSET_H, int)
        or not isinstance(CROP_OFFSET_W, int)
        or CROP_OFFSET_H < 0
        or CROP_OFFSET_W < 0
    ):
        print("CROP_OFFSET_H and CROP_OFFSET_W must be non-negative integers.", file=sys.stderr)
        return 2

    timestamp_dirs = sorted(
        (path for path in INPUT_ROOT.iterdir() if path.is_dir()),
        key=natural_sort_key,
    )
    if not timestamp_dirs:
        print(f"No timestamp directories found in {INPUT_ROOT}", file=sys.stderr)
        return 2

    clip_conversions: list[ClipConversion] | None = None
    if args.clip_csv is not None:
        try:
            clip_conversions = build_clip_conversion_plan(
                timestamp_dirs, args.clip_csv
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"Invalid clip CSV: {exc}", file=sys.stderr)
            return 2
        if ENABLE_DUPLICATE_FRAME_FILTER:
            print(
                "--clip-csv requires duplicate-frame filtering to be disabled so "
                "each closed frame range is preserved exactly",
                file=sys.stderr,
            )
            return 2

    split_plan: dict[str, list[Path]] | None = None
    clip_h5_splits: tuple[ClipH5Split, ...] | None = None
    h5_prefix = args.h5_prefix or OUTPUT_ROOT.name
    if args.write_h5:
        if not ENABLE_CROP or CROP_SIZE != (H5_IMAGE_SIZE, H5_IMAGE_SIZE):
            print(
                "--write-h5 requires an enabled 224x224 crop; "
                "it cannot be combined with --no-crop",
                file=sys.stderr,
            )
            return 2
        try:
            if clip_conversions is not None:
                clip_h5_splits = build_clip_h5_splits(clip_conversions)
                output_paths = build_clip_h5_output_paths(
                    clip_h5_splits, args.h5_output_dir, h5_prefix
                )
                mapping_dir = args.h5_output_dir / "idx_mapping"
                planned_paths = [
                    *output_paths.values(),
                    *(
                        mapping_dir / f"{path.stem}_idx_mapping.csv"
                        for path in output_paths.values()
                    ),
                ]
                existing = [path for path in planned_paths if path.exists()]
                if existing and not args.overwrite_h5:
                    raise FileExistsError(
                        "Clipped H5 output already exists; pass --overwrite-h5 "
                        "to replace it: "
                        + ", ".join(map(str, existing))
                    )
                if args.h5_only:
                    validate_corresponding_clip_mp4s(clip_conversions)
            else:
                split_plan = build_split_plan(
                    timestamp_dirs,
                    train_count=args.train_count,
                    valid_count=args.valid_count,
                    calibration_count=args.calibration_count,
                    fail_count=args.fail_count,
                    split_seed=args.split_seed,
                )
                output_paths = build_h5_output_paths(
                    split_plan, args.h5_output_dir, h5_prefix
                )
                mapping_dir = args.h5_output_dir / "idx_mapping"
                planned_paths = [
                    *output_paths.values(),
                    *(
                        mapping_dir / f"{path.stem}_idx_mapping.csv"
                        for path in output_paths.values()
                    ),
                ]
                existing = [path for path in planned_paths if path.exists()]
                if existing and not args.overwrite_h5:
                    raise FileExistsError(
                        "H5 output already exists; pass --overwrite-h5 to replace it: "
                        + ", ".join(map(str, existing))
                    )
                for timestamp_dir in timestamp_dirs:
                    frame_dir = timestamp_dir / FRAME_DIR_NAME
                    if not frame_dir.is_dir() or not find_frames(frame_dir):
                        raise RuntimeError(
                            f"Missing or empty {FRAME_DIR_NAME}: {frame_dir}"
                        )
        except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
            print(f"Invalid H5 plan: {exc}", file=sys.stderr)
            return 2

    if not args.h5_only:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    converted = skipped = missing = failed = total_frames = 0
    total_duplicates_removed = 0

    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"FPS:    {FPS:g}")
    print(f"Codec:  {VIDEO_CODEC} ({OUTPUT_PIXEL_FORMAT})")
    crop_status = (
        f"{CROP_SIZE[0]}x{CROP_SIZE[1]} at "
        f"offset(H,W)=({CROP_OFFSET_H},{CROP_OFFSET_W})"
        if ENABLE_CROP
        else "disabled"
    )
    print(f"Crop:   {crop_status}")
    duplicate_filter_status = (
        "enabled (exact, bitwise-identical RGB)"
        if ENABLE_DUPLICATE_FRAME_FILTER
        else "disabled"
    )
    print(f"Duplicate filter: {duplicate_filter_status}")
    print(f"Frame-index overlay: {'enabled' if ENABLE_FRAME_INDEX_OVERLAY else 'disabled'}")
    print(f"Found:  {len(timestamp_dirs)} timestamp directories\n")
    if clip_conversions is not None:
        skipped_faults = len(timestamp_dirs) - len(clip_conversions)
        print(
            f"Clip CSV: {args.clip_csv} (valid={len(clip_conversions)}, "
            f"fault={skipped_faults}, one-based inclusive ranges)\n"
        )
    if clip_h5_splits is not None:
        print(
            "Clipped H5 splits: "
            + ", ".join(
                f"{h5_split.label}={len(h5_split.conversions)}"
                for h5_split in clip_h5_splits
            )
            + "\n"
        )
    elif split_plan is not None:
        print(
            "H5 splits: "
            + ", ".join(f"{name}={len(items)}" for name, items in split_plan.items())
            + f" (seed={args.split_seed})\n"
        )

    if not args.skip_crop_preview and not args.h5_only:
        try:
            save_crop_preview(timestamp_dirs)
        except Exception as exc:
            print(f"[fail] Crop preview: {exc}", file=sys.stderr)
            return 1

    if args.h5_only:
        print("[skip] MP4 conversion and crop preview skipped by --h5-only")
    elif clip_conversions is None:
        for video_index, timestamp_dir in enumerate(timestamp_dirs):
            try:
                status, frame_count, duplicates_removed = convert_timestamp(
                    timestamp_dir, video_index
                )
                total_frames += frame_count
                total_duplicates_removed += duplicates_removed
                if status == "converted":
                    converted += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    missing += 1
                    print(f"[skip] Missing {FRAME_DIR_NAME}: {timestamp_dir}")
            except Exception as exc:
                failed += 1
                print(f"[fail] {timestamp_dir.name}: {exc}", file=sys.stderr)
    else:
        for conversion in clip_conversions:
            try:
                status, frame_count, duplicates_removed = convert_timestamp(
                    conversion.timestamp_dir,
                    conversion.output_demo_id,
                    source_demo_id=conversion.source_demo_id,
                    start_idx=conversion.start_idx,
                    end_idx=conversion.end_idx,
                )
                total_frames += frame_count
                total_duplicates_removed += duplicates_removed
                if status == "converted":
                    converted += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    missing += 1
                    print(f"[skip] Missing {FRAME_DIR_NAME}: {conversion.timestamp_dir}")
            except Exception as exc:
                failed += 1
                print(
                    f"[fail] original demo {conversion.source_demo_id} "
                    f"({conversion.timestamp_dir.name}): {exc}",
                    file=sys.stderr,
                )

    print(
        "\nSummary: "
        f"converted={converted}, skipped={skipped}, missing={missing}, "
        f"failed={failed}, frames={total_frames}, "
        f"duplicates_removed={total_duplicates_removed}"
    )
    if failed or missing:
        return 1

    if clip_conversions is not None and not args.h5_only:
        removed_videos = remove_stale_clip_videos(
            len(clip_conversions), len(timestamp_dirs)
        )
        if removed_videos:
            print(
                "Removed stale MP4s: "
                + ", ".join(path.name for path in removed_videos)
            )

    if clip_h5_splits is not None:
        try:
            write_clip_h5_splits(
                clip_h5_splits,
                args.clip_csv,
                h5_output_dir=args.h5_output_dir,
                h5_prefix=h5_prefix,
                overwrite=args.overwrite_h5,
            )
        except Exception as exc:
            print(f"[fail] clipped H5 conversion: {exc}", file=sys.stderr)
            return 1
    elif split_plan is not None:
        try:
            write_h5_splits(
                timestamp_dirs,
                split_plan,
                h5_output_dir=args.h5_output_dir,
                h5_prefix=h5_prefix,
                split_seed=args.split_seed,
                overwrite=args.overwrite_h5,
            )
        except Exception as exc:
            print(f"[fail] H5 conversion: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
