"""
Convert specified MP4 videos to individual PNG image files.

For each selected video, extracts frames at TARGET_FPS and saves them as:
  datasets/raw_img/{video_stem}/frame_{idx:06d}.png

Usage:
    python mp4vid_to_png.py --task pouring_train56 [--fps 10] [--size 224]

To process only specific videos, edit VIDEO_NAMES below.
"""

import os
import re
import cv2
import numpy as np
from pathlib import Path
import logging
import argparse

# ---------------------------------------------------------------------------
# Hardcoded video names to process (stem without extension).
# Set to None or [] to process ALL videos in the task folder.
# ---------------------------------------------------------------------------
# VIDEO_NAMES = [
#     # Example: only extract these two clips.
#     "clearsoda_to_white_real_view_0",
#     "clearsoda_to_white_real_view_1",
#     "milk_to_white_real_view_0",
#     "milk_to_white_real_view_1",
#     "pom_to_clear_real_view_0",
#     "pom_to_clear_real_view_1",
# ]
VIDEO_NAMES = None  # Set to None or [] to process ALL videos in the task folder.
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_frames_to_png(
    video_path: Path,
    output_dir: Path,
    target_fps: int,
    frame_size: tuple,
) -> int:
    """
    Extract frames from a video file and save as PNG images.

    Args:
        video_path:  Path to the mp4/mov file.
        output_dir:  Directory to save frame PNGs.
        target_fps:  Target sampling rate (frames per second).
        frame_size:  Output (width, height) for each frame.

    Returns:
        Number of frames saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return 0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute stride: 1 = no downsampling; otherwise downsample to target_fps
    stride = (
        max(1, round(original_fps / target_fps))
        if (target_fps is not None and original_fps > 0)
        else 1
    )

    logger.info(
        f"  Source : {video_path.name} | "
        f"{original_fps:.1f} fps | {total_frames} frames | {width}x{height}"
    )
    logger.info(
        "  Stride : " + (
            "1 (no downsampling, all frames)"
            if target_fps is None
            else f"every {stride} frame(s) → ~{target_fps} fps output"
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                # BGR → RGB, then resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, frame_size)

                png_path = output_dir / f"frame_{saved:06d}.png"
                # cv2.imwrite expects BGR
                cv2.imwrite(
                    str(png_path),
                    cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR),
                )
                saved += 1

            frame_idx += 1
    finally:
        cap.release()

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from MP4 videos and save as PNG files."
    )
    parser.add_argument(
        "task",
        type=str,
        help="Task folder name under datasets/raw/, e.g. pouring_train56",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Downsample to this FPS (default: None = no downsampling, extract every frame)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output frame size in pixels, square (default: 224)",
    )
    args = parser.parse_args()

    base_dir = Path("/home/user/zhangzk/projects/fineprog/datasets")
    raw_root = base_dir / "raw" / args.task
    output_root = base_dir / "raw_img"

    if not raw_root.exists():
        logger.error(f"Task folder not found: {raw_root}")
        return

    # Collect video files
    all_videos = sorted(
        list(raw_root.glob("*.mp4")) + list(raw_root.glob("*.mov")),
        key=lambda p: p.name,
    )

    # Filter to VIDEO_NAMES if specified
    if VIDEO_NAMES:
        selected = [v for v in all_videos if v.stem in VIDEO_NAMES]
        skipped = [n for n in VIDEO_NAMES if not any(v.stem == n for v in all_videos)]
        if skipped:
            logger.warning(f"Videos not found in {raw_root}: {skipped}")
    else:
        selected = all_videos

    if not selected:
        logger.error("No videos to process.")
        return

    logger.info(
        f"Task     : {args.task}\n"
        f"Raw root : {raw_root}\n"
        f"Output   : {output_root}\n"
        f"Videos   : {len(selected)} selected / {len(all_videos)} total\n"
        f"Target   : {'all frames (no downsample)' if args.fps is None else f'{args.fps} fps'}, {args.size}x{args.size} px"
    )
    print("-" * 60)

    summary = []
    for video_path in selected:
        video_stem = video_path.stem
        out_dir = output_root / args.task / video_stem

        logger.info(f"Processing: {video_stem}")
        n_saved = extract_frames_to_png(
            video_path=video_path,
            output_dir=out_dir,
            target_fps=args.fps,
            frame_size=(args.size, args.size),
        )
        summary.append((video_stem, n_saved, out_dir))
        logger.info(f"  Saved  : {n_saved} frames → {out_dir}")
        print()

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"{'VIDEO NAME':<45} {'FRAMES':>7}  OUTPUT PATH")
    print("-" * 60)
    total_frames = 0
    for stem, n, out_dir in summary:
        rel = out_dir.relative_to(base_dir)
        print(f"{stem:<45} {n:>7}  {rel}")
        total_frames += n
    print("-" * 60)
    print(f"{'TOTAL':<45} {total_frames:>7}  frames across {len(summary)} video(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
