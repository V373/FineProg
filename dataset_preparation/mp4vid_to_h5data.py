"""
Convert MP4 videos to HDF5 format with frame extraction and resizing.

This script:
1. Scans all MP4 files in raw_root
2. Maps action names to action IDs
3. Extracts frames from each video at sampled intervals
4. Resizes frames to 224x224
5. Saves as HDF5 with proper structure

Robomimic mode (--robomimic):
  - Also reads a *.hdf5 file in the same task folder that contains split masks
    under /mask/{percent}_train and /mask/{percent}_valid (e.g. 20_percent_train).
  - Uses those 1-based video indices to split the MP4 list and writes two
    separate H5 files:  {task}-{n}vid_train.h5  and  {task}-{n}vid_valid.h5
"""

import os
import re
import glob
import csv
import h5py
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

# Fixed parameters
IMAGE_SIZE = 224
TARGET_FPS = None  # None = no downsampling; set to an int (e.g. 10) to downsample
COMPRESSION = None  # Can be None, "lzf", or "gzip"
CHUNK_LEN = 8
IDX_MAPPING_DIR = Path("datasets/processed/idx_mapping")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobomimicMaskReader:
    """
    Read train/valid split masks from a robomimic *.hdf5 file.

    The file is expected to contain a group /mask with datasets named like:
        20_percent_train, 20_percent_valid,
        50_percent_train, 50_percent_valid,
        train, valid
    Each dataset holds 1-based demo indices (integers), e.g. [1, 3, 7, ...].
    """

    def __init__(self, hdf5_path: str):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Robomimic HDF5 not found: {self.hdf5_path}")

    def available_keys(self) -> List[str]:
        with h5py.File(str(self.hdf5_path), "r") as f:
            if "mask" not in f:
                raise KeyError("No /mask group found in the robomimic HDF5 file.")
            return list(f["mask"].keys())

    def read_split(self, percent: str, split: str) -> List[int]:
        """
        Read 1-based video indices for a given percent and split.

        Args:
            percent: e.g. "20_percent", "50_percent", or "train"/"valid" for full split
            split:   "train" or "valid"

        Returns:
            Sorted list of 1-based video indices.
        """
        if percent in ("train", "valid"):
            # Full split — key is just "train" or "valid"
            key = split
        else:
            key = f"{percent}_{split}"

        with h5py.File(str(self.hdf5_path), "r") as f:
            if "mask" not in f:
                raise KeyError("No /mask group found in the robomimic HDF5 file.")
            if key not in f["mask"]:
                available = list(f["mask"].keys())
                raise KeyError(
                    f"Mask key '{key}' not found. Available: {available}"
                )
            raw = f["mask"][key][()]
            indices = []
            for x in raw:
                if isinstance(x, (bytes, np.bytes_)):
                    x = x.decode("utf-8")
                s = str(x)
                # Accept bare integers ("10") or demo_N style names ("demo_10")
                m = re.search(r"\d+$", s)
                if m:
                    indices.append(int(m.group()))
                else:
                    raise ValueError(f"Cannot extract index from mask entry: {x!r}")
            indices = sorted(indices)
        logger.info(f"Loaded mask '{key}': {len(indices)} videos")
        return indices

    def read_mask(self, key: str) -> List[int]:
        """
        Read 1-based video indices for a direct mask key (single-split mode).

        Args:
            key: Exact mask dataset name, e.g. "train", "valid",
                 "20_percent_train", "20_percent_valid".

        Returns:
            Sorted list of 1-based video indices.
        """
        with h5py.File(str(self.hdf5_path), "r") as f:
            if "mask" not in f:
                raise KeyError("No /mask group found in the robomimic HDF5 file.")
            if key not in f["mask"]:
                available = list(f["mask"].keys())
                raise KeyError(
                    f"Mask key '{key}' not found. Available: {available}"
                )
            raw = f["mask"][key][()]
            indices = []
            for x in raw:
                if isinstance(x, (bytes, np.bytes_)):
                    x = x.decode("utf-8")
                s = str(x)
                m = re.search(r"\d+$", s)
                if m:
                    indices.append(int(m.group()))
                else:
                    raise ValueError(f"Cannot extract index from mask entry: {x!r}")
            indices = sorted(indices)
        logger.info(f"Loaded mask '{key}': {len(indices)} videos")
        return indices

    @staticmethod
    def find_robomimic_hdf5(raw_root: Path) -> Path:
        """Find the single *.hdf5 file in raw_root (raises if 0 or >1 found)."""
        matches = list(raw_root.glob("*.hdf5"))
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No *.hdf5 file found in {raw_root}. "
                "Robomimic mode requires a mask HDF5 file in the task folder."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple *.hdf5 files found in {raw_root}: {matches}. "
                "Please keep only one robomimic mask file."
            )
        return matches[0]


class MP4ToH5Converter:
    """Convert MP4 videos to HDF5 format."""
    
    def __init__(
        self,
        raw_root: str,
        output_dir: str,
        target_fps=None,
        robomimic: bool = False,
        robomimic_percent: str = "20_percent",
        two_split: bool = False,
    ):
        """
        Initialize converter.
        
        Args:
            raw_root: Directory containing mp4 files
            output_dir: Directory to save h5 files
            target_fps: Target FPS for downsampling; None = no downsampling (extract every frame)
            robomimic: If True, read split masks from a *.hdf5 in raw_root.
            robomimic_percent:
                two_split=True  — percent prefix, e.g. "20_percent"; produces
                                  {percent}_train and {percent}_valid H5 files.
                two_split=False — direct mask key, e.g. "train", "valid",
                                  "20_percent_train"; produces one H5 file.
            two_split: When True (and robomimic=True), generate separate train
                       and valid H5 files from {robomimic_percent}_train/_valid
                       masks.  When False (default), generate a single H5 file
                       from the mask key given by robomimic_percent directly.
        """
        self.raw_root = Path(raw_root)
        self.output_dir = Path(output_dir)
        self.frame_size = (IMAGE_SIZE, IMAGE_SIZE)
        self.target_fps = target_fps
        self.robomimic = robomimic
        self.robomimic_percent = robomimic_percent
        self.two_split = two_split
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract folder name for output h5 filename
        self.folder_name = self.raw_root.name
        # output_h5_path is determined after processing (depends on video count)
        self.output_h5_path: Optional[Path] = None
        # two_split=True outputs
        self.output_h5_train_path: Optional[Path] = None
        self.output_h5_valid_path: Optional[Path] = None
        # two_split=False (single mask) output
        self.output_h5_single_path: Optional[Path] = None

        # Store action name to ID mapping
        self.action_to_id: Dict[str, int] = {}
        self.mp4_files: List[Path] = []
    
    def scan_mp4_files(self) -> List[Path]:
        """
        Scan for all mp4 and mov files in raw_root.

        Returns:
            List of Path objects for video files
        """
        mp4_files = sorted(self.raw_root.glob("*.mp4"))
        mov_files = sorted(self.raw_root.glob("*.mov"))
        all_files = sorted(mp4_files + mov_files, key=lambda p: p.name)
        logger.info(
            f"Found {len(mp4_files)} mp4 and {len(mov_files)} mov files "
            f"({len(all_files)} total) in {self.raw_root}"
        )
        self.mp4_files = all_files
        return all_files
    
    def extract_action_name(self, filename: str) -> str:
        """
        Extract action name from a video filename.

        Handles both formats:
          - MP4: {action_name}_real_view_{view_id}.mp4  (underscore before digit)
          - MOV: {action_name}_real_view{view_id}.mov   (no underscore before digit)

        Args:
            filename: Video filename (mp4 or mov)

        Returns:
            Action name string
        """
        # Remove .mp4 or .mov extension (case-insensitive)
        name = re.sub(r"\.(mp4|mov)$", "", filename, flags=re.IGNORECASE)
        # Match both _real_view_N (mp4) and _real_viewN (mov) suffixes
        match = re.match(r"(.+?)_real_view_?\d+$", name)
        if match:
            return match.group(1)
        else:
            # Fallback: strip whichever suffix is present
            return re.sub(r"_real_view_?\d+$", "", name)
    
    def build_action_mapping(self) -> Dict[str, int]:
        """
        Build mapping from action name to action ID.
        
        Returns:
            Dictionary mapping action names to IDs
        """
        if not self.mp4_files:
            self.scan_mp4_files()
        
        action_names = set()
        for mp4_path in self.mp4_files:
            action_name = self.extract_action_name(mp4_path.name)
            action_names.add(action_name)
        
        # Sort for consistent ordering
        action_names = sorted(action_names)
        self.action_to_id = {name: i for i, name in enumerate(action_names)}
        
        logger.info(f"Built action mapping with {len(self.action_to_id)} unique actions")
        for name, idx in sorted(self.action_to_id.items()):
            logger.info(f"  {idx:04d}: {name}")
        
        return self.action_to_id
    
    def get_video_info(self, video_path: Path) -> Tuple[int, int, int]:
        """
        Get video information.
        
        Args:
            video_path: Path to mp4 file
            
        Returns:
            Tuple of (fps, num_frames, width, height)
        """
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return fps, num_frames, width, height
        finally:
            cap.release()
    
    def calculate_sample_stride(self, original_fps: int) -> int:
        """
        Calculate frame sampling stride based on original FPS.
        
        Args:
            original_fps: Original video FPS
            
        Returns:
            Sampling stride (every n-th frame)
        """
        if self.target_fps is None:
            return 1  # no downsampling: extract every frame
        if original_fps == 0:
            return 1
        stride = max(1, original_fps // self.target_fps)
        return stride
    
    def extract_frames(self, video_path: Path, stride: int) -> np.ndarray:
        """
        Extract frames from video at specified stride and resize to target size.
        
        Args:
            video_path: Path to mp4 file
            stride: Frame sampling stride
            
        Returns:
            Array of shape [T, H, W, 3] with uint8 values (RGB)
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at specified stride
                if frame_idx % stride == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize to target size
                    frame_resized = cv2.resize(frame_rgb, self.frame_size)
                    frames.append(frame_resized)
                
                frame_idx += 1
        finally:
            cap.release()
        
        if frames:
            frames_array = np.stack(frames, axis=0).astype(np.uint8)
            logger.info(f"Extracted {frames_array.shape[0]} frames from {video_path.name}")
            return frames_array
        else:
            logger.warning(f"No frames extracted from {video_path.name}")
            return np.array([], dtype=np.uint8).reshape(0, *self.frame_size, 3)
    
    def _write_videos_to_h5(
        self,
        h5file: h5py.File,
        video_paths: List[Path],
        base_idx_offset: int = 0,
    ) -> Tuple[int, List[dict]]:
        """
        Write a list of video paths into an already-open H5 file.

        Group names are 1-based and continuous starting from 000001.

        Args:
            h5file:          Open, writable H5 file.
            video_paths:     Ordered list of video Paths to write.
            base_idx_offset: Unused; kept for API clarity (groups always start at 1).

        Returns:
            Tuple of (number of videos successfully written, mapping_records).
            mapping_records is a list of dicts with keys: video_idx,
            source_video_name, source_video_path.
        """
        videos_group = h5file.require_group("videos")
        written = 0
        mapping_records: List[dict] = []

        for local_idx, video_path in enumerate(video_paths, 1):
            try:
                logger.info(f"  [{local_idx}/{len(video_paths)}] {video_path.name}")

                original_fps, num_frames, width, height = self.get_video_info(video_path)
                logger.info(f"    Original: {original_fps}fps, {num_frames} frames, {width}x{height}")

                stride = self.calculate_sample_stride(original_fps)
                logger.info(f"    Sampling stride: {stride}")

                frames = self.extract_frames(video_path, stride)

                if frames.size == 0:
                    logger.warning(f"    Skipping {video_path.name} - no frames extracted")
                    continue

                action_name = self.extract_action_name(video_path.name)
                action_id = 0

                video_group_name = f"{local_idx:06d}"
                video_group = videos_group.create_group(video_group_name)

                chunk_size = min(CHUNK_LEN, len(frames))
                video_group.create_dataset(
                    "frames",
                    data=frames,
                    dtype="uint8",
                    compression=None,
                    chunks=(chunk_size, IMAGE_SIZE, IMAGE_SIZE, 3),
                )

                video_group.attrs["action_name"] = action_name
                video_group.attrs["action_id"] = action_id
                video_group.attrs["fps"] = self.target_fps if self.target_fps is not None else original_fps
                video_group.attrs["num_frames"] = frames.shape[0]
                video_group.attrs["path"] = str(video_path)

                logger.info(f"    Saved: {video_group_name} - {frames.shape[0]} frames")
                written += 1
                mapping_records.append({
                    "video_idx": video_group_name,
                    "source_video_name": video_path.name,
                    "source_video_path": str(video_path),
                })

            except Exception as e:
                logger.error(f"    Error processing {video_path.name}: {e}", exc_info=True)
                continue

        return written, mapping_records

    def _save_idx_mapping(self, mapping_records: List[dict], final_h5_path: Path) -> None:
        """Save a CSV mapping of video_idx to source video info for a given H5 file."""
        mapping_dir = IDX_MAPPING_DIR
        mapping_dir.mkdir(parents=True, exist_ok=True)
        csv_path = mapping_dir / f"{final_h5_path.stem}_idx_mapping.csv"
        fieldnames = ["processed_h5_name", "video_idx", "source_video_name", "source_video_path"]
        with open(str(csv_path), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for rec in mapping_records:
                writer.writerow({
                    "processed_h5_name": final_h5_path.name,
                    "video_idx": rec["video_idx"],
                    "source_video_name": rec["source_video_name"],
                    "source_video_path": rec["source_video_path"],
                })
        logger.info(f"Saved idx mapping: {csv_path} ({len(mapping_records)} rows)")

    def _finalize_h5(self, tmp_path: Path, suffix: str, n_videos: int) -> Path:
        """
        Rename tmp H5 to final name and validate.

        Args:
            tmp_path:  Temporary H5 path.
            suffix:    Extra suffix appended before '.h5', e.g. "_train" or "".
            n_videos:  Number of videos written (used in filename).

        Returns:
            Final H5 path.
        """
        final_name = f"{self.folder_name}-{n_videos}vid{suffix}.h5"
        final_path = self.output_dir / final_name

        with h5py.File(str(tmp_path), "r") as f:
            actual = len(f["videos"])
            print(f"VALID H5 ({suffix or 'full'}): keys={list(f.keys())}, videos={actual}")
            if actual > 0:
                first_key = list(f["videos"].keys())[0]
                print(f"  FIRST VIDEO: {first_key}, FRAMES SHAPE: {f['videos'][first_key]['frames'].shape}")

        os.replace(tmp_path, final_path)
        logger.info(f"Saved: {final_path}")
        return final_path

    def process_videos(self) -> None:
        if not self.mp4_files:
            self.scan_mp4_files()
        if not self.action_to_id:
            self.build_action_mapping()

        if self.robomimic:
            self._process_videos_robomimic()
        else:
            self._process_videos_standard()

    def _process_videos_standard(self) -> None:
        """Original behaviour: write all videos into one H5 file."""
        tmp_h5_path = self.output_dir / f"{self.folder_name}_tmp.h5"
        if tmp_h5_path.exists():
            tmp_h5_path.unlink()

        with h5py.File(str(tmp_h5_path), "w", libver="earliest") as h5file:
            written, mapping_records = self._write_videos_to_h5(h5file, self.mp4_files)
            h5file.flush()

        self.output_h5_path = self._finalize_h5(tmp_h5_path, "", written)
        self._save_idx_mapping(mapping_records, self.output_h5_path)

    def _build_demo_id_map(self) -> Dict[int, Path]:
        """
        Build a mapping from demo integer ID to Path for robomimic-style filenames.

        Expects files named demo_N.{mp4,mov} where N is the demo index.
        Returns {N: path} so mask entries like 'demo_10' → 10 → demo_10.mp4.
        """
        demo_map: Dict[int, Path] = {}
        for path in self.mp4_files:
            m = re.match(r"demo_(\d+)\.(mp4|mov)$", path.name, re.IGNORECASE)
            if m:
                demo_map[int(m.group(1))] = path
            else:
                logger.warning(f"  File '{path.name}' does not match demo_N.mp4 pattern; skipping in robomimic mode.")
        return demo_map

    def _process_videos_robomimic(self) -> None:
        """Dispatch to two-split, single-mask, or all-videos processing."""
        if self.robomimic_percent == "all":
            self._process_videos_robomimic_all()
        elif self.two_split:
            self._process_videos_robomimic_two_split()
        else:
            self._process_videos_robomimic_single()

    def _process_videos_robomimic_all(self) -> None:
        """
        Robomimic 'all' mode: process every video in the directory, sorted by
        the numeric ID extracted from the filename (demo_N → N) in ascending
        order.  No mask file is required.

        Output filename: {folder_name}-{n}vid_all.h5
        """
        logger.info("Robomimic 'all' mode: processing all videos sorted by numeric demo ID.")
        demo_id_map = self._build_demo_id_map()
        sorted_paths = [demo_id_map[k] for k in sorted(demo_id_map.keys())]
        logger.info(f"  {len(sorted_paths)} videos found, sorted by numeric ID.")

        tmp_path = self.output_dir / f"{self.folder_name}_tmp_single.h5"
        if tmp_path.exists():
            tmp_path.unlink()
        with h5py.File(str(tmp_path), "w", libver="earliest") as h5file:
            written, mapping_records = self._write_videos_to_h5(h5file, sorted_paths)
            h5file.flush()
        self.output_h5_single_path = self._finalize_h5(tmp_path, "_all", written)
        self._save_idx_mapping(mapping_records, self.output_h5_single_path)

    def _process_videos_robomimic_two_split(self) -> None:
        """
        Robomimic two-split mode: read {robomimic_percent}_train and
        {robomimic_percent}_valid masks and write two separate H5 files.

        The mask HDF5 stores demo names like b'demo_10'.  We extract the integer
        suffix and look up the corresponding demo_N.mp4 file by ID (not by
        position in the sorted file list).
        """
        mask_hdf5_path = RobomimicMaskReader.find_robomimic_hdf5(self.raw_root)
        logger.info(f"Using robomimic mask file: {mask_hdf5_path}")
        reader = RobomimicMaskReader(str(mask_hdf5_path))

        logger.info(f"Available mask keys: {reader.available_keys()}")

        train_indices = reader.read_split(self.robomimic_percent, "train")
        valid_indices = reader.read_split(self.robomimic_percent, "valid")

        # Build demo_id → path mapping (demo_10.mp4 → {10: Path(...)})
        demo_id_map = self._build_demo_id_map()
        logger.info(f"Built demo_id map with {len(demo_id_map)} entries")

        def indices_to_paths(indices: List[int]) -> List[Path]:
            paths = []
            for idx in indices:
                if idx not in demo_id_map:
                    logger.warning(f"  demo_{idx} not found in mp4 files, skipping.")
                    continue
                paths.append(demo_id_map[idx])
            return paths

        train_paths = indices_to_paths(train_indices)
        valid_paths = indices_to_paths(valid_indices)

        logger.info(
            f"Robomimic split '{self.robomimic_percent}': "
            f"{len(train_paths)} train, {len(valid_paths)} valid"
        )

        # --- Write train H5 ---
        tmp_train = self.output_dir / f"{self.folder_name}_tmp_train.h5"
        if tmp_train.exists():
            tmp_train.unlink()
        with h5py.File(str(tmp_train), "w", libver="earliest") as h5file:
            written_train, mapping_records_train = self._write_videos_to_h5(h5file, train_paths)
            h5file.flush()
        self.output_h5_train_path = self._finalize_h5(tmp_train, "_train", written_train)
        self._save_idx_mapping(mapping_records_train, self.output_h5_train_path)

        # --- Write valid H5 ---
        tmp_valid = self.output_dir / f"{self.folder_name}_tmp_valid.h5"
        if tmp_valid.exists():
            tmp_valid.unlink()
        with h5py.File(str(tmp_valid), "w", libver="earliest") as h5file:
            written_valid, mapping_records_valid = self._write_videos_to_h5(h5file, valid_paths)
            h5file.flush()
        self.output_h5_valid_path = self._finalize_h5(tmp_valid, "_valid", written_valid)
        self._save_idx_mapping(mapping_records_valid, self.output_h5_valid_path)

    def _process_videos_robomimic_single(self) -> None:
        """
        Robomimic single-mask mode: read the mask whose key equals
        robomimic_percent directly (e.g. "train", "valid",
        "20_percent_train") and write one H5 file.

        Output filename: {folder_name}-{n}vid_{mask_key}.h5
        """
        mask_hdf5_path = RobomimicMaskReader.find_robomimic_hdf5(self.raw_root)
        logger.info(f"Using robomimic mask file: {mask_hdf5_path}")
        reader = RobomimicMaskReader(str(mask_hdf5_path))

        logger.info(f"Available mask keys: {reader.available_keys()}")

        mask_key = self.robomimic_percent
        indices = reader.read_mask(mask_key)

        demo_id_map = self._build_demo_id_map()
        logger.info(f"Built demo_id map with {len(demo_id_map)} entries")

        paths: List[Path] = []
        for idx in indices:
            if idx not in demo_id_map:
                logger.warning(f"  demo_{idx} not found in mp4 files, skipping.")
                continue
            paths.append(demo_id_map[idx])

        logger.info(
            f"Robomimic single mask '{mask_key}': {len(paths)} videos to process"
        )

        # Sanitise mask_key for use in a filename (replace spaces with _)
        suffix = f"_{mask_key.replace(' ', '_')}"
        tmp_path = self.output_dir / f"{self.folder_name}_tmp_single.h5"
        if tmp_path.exists():
            tmp_path.unlink()
        with h5py.File(str(tmp_path), "w", libver="earliest") as h5file:
            written, mapping_records = self._write_videos_to_h5(h5file, paths)
            h5file.flush()
        self.output_h5_single_path = self._finalize_h5(tmp_path, suffix, written)
        self._save_idx_mapping(mapping_records, self.output_h5_single_path)

    def run(self) -> None:
        logger.info(f"Starting conversion: {self.raw_root} -> {self.output_dir}")
        self.scan_mp4_files()
        self.build_action_mapping()
        self.process_videos()

        if self.robomimic:
            if self.two_split:
                for label, path in [("TRAIN", self.output_h5_train_path), ("VALID", self.output_h5_valid_path)]:
                    print(f"FINAL FILE [{label}]:", path)
                    if path is not None:
                        print(f"EXISTS [{label}]:", path.exists())
                        if path.exists():
                            print(f"SIZE [{label}]:", path.stat().st_size, "bytes")
            else:
                path = self.output_h5_single_path
                print(f"FINAL FILE [SINGLE mask='{self.robomimic_percent}']:", path)
                if path is not None:
                    print("EXISTS:", path.exists())
                    if path.exists():
                        print("SIZE:", path.stat().st_size, "bytes")
        else:
            print("FINAL FILE:", self.output_h5_path)
            print("EXISTS:", self.output_h5_path.exists())
            if self.output_h5_path.exists():
                print("SIZE:", self.output_h5_path.stat().st_size, "bytes")

        logger.info("Conversion complete!")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Convert MP4 videos to HDF5 format")
    parser.add_argument("task", type=str, help="Task name, e.g. pouring")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Downsample to this FPS (default: None = no downsampling, extract every frame)",
    )
    parser.add_argument(
        "--robomimic",
        action="store_true",
        default=False,
        help=(
            "Robomimic mode: read a *.hdf5 mask file from the task folder "
            "and generate H5 file(s) from mask indices."
        ),
    )
    parser.add_argument(
        "--two_split",
        action="store_true",
        default=False,
        help=(
            "(Requires --robomimic) Generate two separate H5 files from "
            "{robomimic_percent}_train and {robomimic_percent}_valid masks. "
            "Default (False): generate a single H5 file from the mask key "
            "given directly by robomimic_percent (e.g. 'train', 'valid', "
            "'20_percent_train')."
        ),
    )
    parser.add_argument(
        "--robomimic_percent",
        type=str,
        default=None,
        help=(
            "Mask percent key to use in robomimic mode, e.g. '20_percent', '50_percent'. "
            "Overrides the value resolved from configs_v2/data_process.yaml (dataset mask_key). "
            "Default when not set: value from config, or '20_percent' as fallback."
        ),
    )
    parser.add_argument(
        "--register",
        action="store_true",
        default=False,
        help="[v2] After conversion, register the dataset into configs_v2/registry/datasets.yaml.",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        dest="register_alias",
        help="[v2] Registry alias (auto-suggested if not set). Requires --register.",
    )
    args = parser.parse_args()

    base_dir = "/home/user/zhangzk/projects/fineprog/datasets"
    raw_root = os.path.join(base_dir, "raw", args.task)
    output_dir = os.path.join(base_dir, "processed")

    # Resolve robomimic_percent: CLI arg > configs_v2/data_process.yaml (mask_key) > default
    robomimic_percent = args.robomimic_percent
    if robomimic_percent is None:
        try:
            # [v2] Read mask_key from V2 data_process config via dataset registry
            import sys as _sys
            _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _proj not in _sys.path:
                _sys.path.insert(0, _proj)
            from utils.config_v2 import ConfigV2
            _dp_cfg = ConfigV2().load_data_process()
            _ds_info = _dp_cfg.get("dataset_info", {})
            robomimic_percent = _ds_info.get("mask_key")   # [v2] mask_key = robomimic_percent
            if robomimic_percent:
                print(f"[main] [v2] robomimic_percent (from mask_key): {robomimic_percent}")
        except Exception:
            pass
        if robomimic_percent is None:
            robomimic_percent = "20_percent"

    converter = MP4ToH5Converter(
        raw_root=raw_root,
        output_dir=output_dir,
        target_fps=args.fps,
        robomimic=args.robomimic,
        robomimic_percent=robomimic_percent,
        two_split=args.two_split,
    )
    converter.run()

    # [v2] Optional: register dataset(s) into configs_v2/registry/datasets.yaml
    if args.register:
        import sys as _sys
        _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _proj not in _sys.path:
            _sys.path.insert(0, _proj)
        from utils.registry_v2 import RegistryV2
        _reg = RegistryV2()

        # Resolve the robomimic mask HDF5 path (relative to project root) if present.
        _raw_root_p = Path(raw_root)
        _robomimic_hdf5_rel: Optional[str] = None
        if args.robomimic:
            try:
                _hdf5_abs = RobomimicMaskReader.find_robomimic_hdf5(_raw_root_p)
                # Store as path relative to project root (matches datasets.yaml convention).
                _robomimic_hdf5_rel = str(_hdf5_abs.relative_to(Path(_proj)))
            except Exception as _exc:
                print(f"[main] [v2] WARNING: could not locate robomimic HDF5 for registration: {_exc}")

        def _register_one(h5_path: Optional[Path], mask_key_str: Optional[str], alias_override: Optional[str]):
            if h5_path is None or not h5_path.exists():
                return
            _alias = alias_override or _reg.suggest_dataset_alias(h5_path.name, mask_key=mask_key_str)
            _reg.register_dataset(
                alias          = _alias,
                processed_h5   = h5_path.name,
                display_name   = f"{args.task} ({h5_path.name})",
                raw_dir        = args.task,          # relative to dirs.raw (datasets/raw)
                robomimic_hdf5 = _robomimic_hdf5_rel,  # relative to project root, or None
                mask_key       = mask_key_str,
            )
            print(f"[main] [v2] Dataset registered as '{_alias}' in configs_v2/registry/datasets.yaml")

        if args.robomimic and args.two_split:
            # Two files — auto-generate aliases; --alias applies to train split
            _register_one(converter.output_h5_train_path, f"{robomimic_percent}_train", args.register_alias)
            _register_one(converter.output_h5_valid_path, f"{robomimic_percent}_valid", None)
        elif args.robomimic and not args.two_split:
            _register_one(converter.output_h5_single_path, robomimic_percent, args.register_alias)
        else:
            _register_one(converter.output_h5_path, None, args.register_alias)


if __name__ == "__main__":
    main()
