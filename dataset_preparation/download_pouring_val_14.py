"""
Download 14 selected validation videos from the Hugging Face dataset
`sermanet/multiview-pouring` (7 real trials × 2 camera views).

Usage:
    python download_pouring_val_14.py \
        --output_dir /home/user/zhangzk/projects/fineprog/datasets/raw/pouring_all_val

Dependencies:
    pip install huggingface_hub
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
    sys.exit(1)


REPO_ID = "sermanet/multiview-pouring"
REPO_TYPE = "dataset"

# 14 selected validation videos: 7 real trials × 2 camera views
SELECTED_VAL_VIDEOS = [
    "videos/val/clearodwalla_to_clear0_real_view0.mov",
    "videos/val/clearodwalla_to_clear0_real_view1.mov",
    "videos/val/clearsoda_to_white0_real_view0.mov",
    "videos/val/clearsoda_to_white0_real_view1.mov",
    "videos/val/clearwater_to_white0_real_view0.mov",
    "videos/val/clearwater_to_white0_real_view1.mov",
    "videos/val/creamsoda_to_clear0_real_view0.mov",
    "videos/val/creamsoda_to_clear0_real_view1.mov",
    "videos/val/green_to_clear0_real_view0.mov",
    "videos/val/green_to_clear0_real_view1.mov",
    "videos/val/milk_to_clear0_real_view0.mov",
    "videos/val/milk_to_clear0_real_view1.mov",
    "videos/val/pom_to_clear0_real_view0.mov",
    "videos/val/pom_to_clear0_real_view1.mov",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download 14 validation videos from sermanet/multiview-pouring on Hugging Face."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user/zhangzk/projects/fineprog/datasets/raw/pouring_all_val",
        help="Local directory to save downloaded .mov files.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token (optional, for private repos).",
    )
    return parser.parse_args()


def file_is_valid(path: Path) -> bool:
    """Return True if the file exists and has non-zero size."""
    return path.exists() and path.stat().st_size > 0


def download_videos(output_dir: Path, token: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = []

    print(f"Target directory : {output_dir}")
    print(f"Files to download: {len(SELECTED_VAL_VIDEOS)}\n")

    for repo_path in SELECTED_VAL_VIDEOS:
        filename = Path(repo_path).name
        dest = output_dir / filename

        if file_is_valid(dest):
            print(f"  [SKIP] {filename} (already exists, {dest.stat().st_size} bytes)")
            skipped += 1
            continue

        print(f"  [DOWN] {filename} ...", end="", flush=True)
        try:
            tmp_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=repo_path,
                local_dir=str(output_dir),
                token=token,
            )
            # hf_hub_download with local_dir places the file at
            # <local_dir>/<filename_in_repo>; symlink it to the flat dest if needed.
            resolved = Path(tmp_path).resolve()
            if resolved != dest.resolve():
                # Move file to flat output dir (drop the videos/val/ sub-path)
                resolved.rename(dest)
            print(f" done ({dest.stat().st_size} bytes)")
            downloaded += 1
        except Exception as exc:
            print(f" FAILED: {exc}")
            failed.append(filename)

    # --- Verification ---
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    all_valid = True
    for repo_path in SELECTED_VAL_VIDEOS:
        filename = Path(repo_path).name
        dest = output_dir / filename
        if file_is_valid(dest):
            print(f"  [OK ] {filename} ({dest.stat().st_size} bytes)")
        else:
            print(f"  [ERR] {filename} — missing or empty!")
            all_valid = False

    # Count total files present in output_dir
    total_in_dir = len(list(output_dir.glob("*.mov")))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {len(failed)}")
    if failed:
        for f in failed:
            print(f"    - {f}")
    print(f"  Total .mov files in output_dir: {total_in_dir}")
    print(f"  Expected   : 14")
    print(f"  Verification {'PASSED' if all_valid and total_in_dir >= 14 else 'FAILED'}")

    if not all_valid or len(failed) > 0:
        sys.exit(1)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    download_videos(output_dir, token=args.token)


if __name__ == "__main__":
    main()
