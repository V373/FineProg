"""
Add phase_labels and keyframe_labels to an existing TCC embeddings H5 file.

For each video found in both the H5 file and the CSV, two new datasets are
written (or overwritten) in-place under /videos/{video_id}/:

    phase_labels       [T_out]  int64
    keyframe_labels    [T_out]  int64

Additionally, ALL video groups in the H5 receive an attr:

    labeled  bool   True  if phase/keyframe labels were written
                    False if the video has no entry in the CSV

CSV format (new — columns: name, id, key_frame_idx):
    name                            human-readable video name
    id                              zero-padded numeric ID (e.g. 000003)
    key_frame_idx                   key event frame index

Legacy CSV format (columns: video_id, key_frame_idx) is also accepted.

H5 lookup order: id first, then name.  This allows the H5 to store videos
under either their numeric ID keys or their name keys.

Phase assignment (K = number of key frames):
    target_steps[t] in [e0, e1)       -> phase 0
    target_steps[t] in [e1, e2)       -> phase 1
    ...
    target_steps[t] in [e(K-2), eK-1] -> phase K-2   (last interval is closed)
    outside [e0, eK-1]                -> -1

Keyframe assignment:
    target_steps[t] == e_i  -> keyframe_labels[t] = i
    otherwise               -> -1

Usage:
    python dataset_preparation/add_phase_labels.py \\
        --embd_h5  datasets/embeddings/.../pouring-2vid-embd.h5 \\
        --keyframes_csv  "datasets/phase labels/pouring_phase_labels.csv"

The script writes labels to a NEW file with '-labeled' appended to the stem,
saved in the same directory as the input H5.  It does NOT modify the original.
embeddings, target_steps, and attrs are copied unchanged.
"""

import argparse
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add phase_labels and keyframe_labels to a TCC embeddings H5 file."
    )
    parser.add_argument(
        "--embd_h5",
        type=str,
        required=True,
        help="Path to the embeddings H5 file (modified in-place).",
    )
    parser.add_argument(
        "--keyframes_csv",
        type=str,
        required=True,
        help="Path to the CSV file with columns: video_id, key_frame_idx.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_keyframes_csv(
    csv_path: str,
) -> tuple[dict[str, list[int]], set[str]]:
    """
    Read the CSV and return:
        lookup_map  : dict mapping video name AND numeric id -> sorted key_frame_idx list.
                      Both keys point to the same list, enabling H5 lookup by either.
        unique_names: set of unique video names (for warning / reporting).

    Supported CSV formats:
        New    : columns  name, id, key_frame_idx
        Legacy : columns  video_id, key_frame_idx

    Videos with fewer than 2 key frames are skipped with a warning.
    """
    df = pd.read_csv(csv_path, dtype=str)

    if "name" in df.columns and "id" in df.columns:
        name_col, id_col = "name", "id"
    elif "video_id" in df.columns:
        name_col, id_col = "video_id", None
    else:
        raise ValueError(
            "CSV must have columns (name, id, key_frame_idx) "
            "or legacy (video_id, key_frame_idx); "
            f"found {set(df.columns)}"
        )

    if "key_frame_idx" not in df.columns:
        raise ValueError("CSV must contain a 'key_frame_idx' column.")

    df["key_frame_idx"] = df["key_frame_idx"].astype(int)

    lookup_map: dict[str, list[int]] = {}
    unique_names: set[str] = set()

    for vid_name, grp_df in df.groupby(name_col, sort=False):
        vid_name = str(vid_name)
        frames = sorted(grp_df["key_frame_idx"].tolist())

        if len(frames) < 2:
            warnings.warn(
                f"[CSV] '{vid_name}' has only {len(frames)} key frame(s) — "
                "need at least 2 to define a phase. Skipping."
            )
            continue

        unique_names.add(vid_name)
        lookup_map[vid_name] = frames

        if id_col is not None:
            vid_id_val = str(grp_df[id_col].iloc[0]).strip()
            if vid_id_val and vid_id_val != vid_name:
                lookup_map[vid_id_val] = frames  # same list, second key

    return lookup_map, unique_names


# ---------------------------------------------------------------------------
# Label builders
# ---------------------------------------------------------------------------

def build_keyframe_labels(
    target_steps: np.ndarray,
    key_frames: list[int],
) -> np.ndarray:
    """
    Return keyframe_labels of shape [T_out], dtype int64.

    keyframe_labels[t] = index of key frame if target_steps[t] matches one,
    otherwise -1.
    """
    labels = np.full(len(target_steps), -1, dtype=np.int64)
    kf_set = {kf: idx for idx, kf in enumerate(key_frames)}
    for t, step in enumerate(target_steps):
        if int(step) in kf_set:
            labels[t] = kf_set[int(step)]
    return labels


def build_phase_labels(
    target_steps: np.ndarray,
    key_frames: list[int],
) -> np.ndarray:
    """
    Return phase_labels of shape [T_out], dtype int64.

    Phases are half-open intervals [e_i, e_{i+1}) except the last which is
    closed [e_{K-2}, e_{K-1}].  Frames outside [e0, eK-1] get label -1.
    """
    labels = np.full(len(target_steps), -1, dtype=np.int64)
    e = key_frames  # already sorted
    K = len(e)      # number of key frames; number of phases = K - 1

    for t, step in enumerate(target_steps):
        s = int(step)
        if s < e[0] or s > e[-1]:
            continue  # out of range -> -1
        # Find phase: largest i such that e[i] <= s
        # Use linear scan (T_out and K are both small)
        phase = -1
        for i in range(K - 1):
            lo = e[i]
            hi = e[i + 1]
            # Last interval is closed on the right
            if i == K - 2:
                if lo <= s <= hi:
                    phase = i
                    break
            else:
                if lo <= s < hi:
                    phase = i
                    break
        labels[t] = phase

    return labels


# ---------------------------------------------------------------------------
# Keyframe scaling
# ---------------------------------------------------------------------------

def scale_keyframes_to_embedding(
    target_steps: np.ndarray,
    key_frames: list[int],
    vid_id: str,
) -> list[int]:
    """
    Check whether the last key frame matches max(target_steps).
    If they differ, scale ALL key frame indices proportionally:

        scaled_kf[i] = round(key_frames[i] * max_step / last_kf)

    The first key frame (index 0) is always preserved as-is so that the
    phase sequence starts at the same relative position.
    Values are clamped to [0, max_step].

    Returns the (possibly scaled) key frame list.
    """
    max_step = int(target_steps.max())
    last_kf = key_frames[-1]

    if last_kf == max_step:
        return key_frames  # exact match — no scaling needed

    scale = max_step / last_kf
    print(
        f"  [scale] '{vid_id}': last_kf={last_kf} != max_target_step={max_step} "
        f"-> scale factor = {scale:.4f}"
    )

    scaled: list[int] = []
    for i, kf in enumerate(key_frames):
        s = int(round(kf * scale))
        s = max(0, min(s, max_step))
        scaled.append(s)

    # Warn if rounding collapsed two consecutive key frames to the same index
    for i in range(len(scaled) - 1):
        if scaled[i] >= scaled[i + 1]:
            warnings.warn(
                f"[{vid_id}] After scaling, key frames {i} and {i+1} are "
                f"non-increasing: {scaled[i]} >= {scaled[i+1]}. "
                "Phase labels may be incorrect."
            )

    print(f"  [scale] original : {key_frames}")
    print(f"  [scale] scaled   : {scaled}")
    return scaled


# ---------------------------------------------------------------------------
# H5 helpers
# ---------------------------------------------------------------------------

def _write_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Delete existing dataset if present, then write new one."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    embd_h5_path = Path(args.embd_h5)
    csv_path = Path(args.keyframes_csv)

    if not embd_h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {embd_h5_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------
    keyframes_map, csv_names = load_keyframes_csv(str(csv_path))
    print(f"[CSV]  Loaded {len(csv_names)} video(s) with valid key frames.")

    # ------------------------------------------------------------------
    # Derive output path: same dir, stem + '-labeled'
    # ------------------------------------------------------------------
    out_h5_path = embd_h5_path.with_name(
        embd_h5_path.stem + "-labeled" + embd_h5_path.suffix
    )
    print(f"[OUT]  Output file : {out_h5_path}")

    # Copy the original H5 to the output path so we start with all existing
    # groups/datasets intact and only add/overwrite the label datasets.
    shutil.copy2(str(embd_h5_path), str(out_h5_path))
    print(f"[OUT]  Copied source H5 to output path.")

    # ------------------------------------------------------------------
    # Open OUTPUT H5 and process
    # ------------------------------------------------------------------
    with h5py.File(str(out_h5_path), "r+") as hf:
        videos_group = hf.get("videos")
        if videos_group is None:
            raise ValueError("H5 file has no '/videos' group.")

        h5_video_ids = list(videos_group.keys())
        h5_keys_set = set(h5_video_ids)
        print(f"[H5]   Found {len(h5_video_ids)} video(s) in H5.")

        # Warn about CSV videos that cannot be found in H5 by either name or id
        for name in csv_names:
            if name not in h5_keys_set and name not in keyframes_map:
                warnings.warn(
                    f"[CSV -> H5] '{name}' is in CSV but not found in H5. Skipping."
                )
        # Also check id-keyed entries
        for key in keyframes_map:
            if key not in h5_keys_set:
                # Suppress duplicate warning if the name was already warned above
                pass  # mismatch resolved at H5 loop below

        processed = 0
        unlabeled = 0

        for vid_id in h5_video_ids:
            grp = videos_group[vid_id]

            if vid_id not in keyframes_map:
                # No CSV entry for this video — mark as unlabeled and skip
                warnings.warn(
                    f"[H5 -> CSV] '{vid_id}' is in H5 but has no key frames in CSV. "
                    "Setting labeled=False."
                )
                grp.attrs["labeled"] = False
                unlabeled += 1
                continue

            key_frames: list[int] = keyframes_map[vid_id]
            target_steps: np.ndarray = grp["target_steps"][:]  # [T_out]

            # Scale key frames to match the embedding's frame range if needed
            key_frames = scale_keyframes_to_embedding(target_steps, key_frames, vid_id)

            T_out = len(target_steps)
            n_phases = len(key_frames) - 1

            # Build labels
            kf_labels = build_keyframe_labels(target_steps, key_frames)
            ph_labels = build_phase_labels(target_steps, key_frames)

            # Count hits and warn about missing key frames
            n_kf_hits = int((kf_labels >= 0).sum())
            missing_kfs = [
                kf for kf in key_frames if int(kf) not in set(target_steps.tolist())
            ]
            if missing_kfs:
                warnings.warn(
                    f"[{vid_id}] {len(missing_kfs)} key frame(s) not found in "
                    f"target_steps: {missing_kfs}"
                )

            phase_uniq = sorted(np.unique(ph_labels).tolist())

            print(
                f"\n[{vid_id}]\n"
                f"  T_out        : {T_out}\n"
                f"  Key frames   : {len(key_frames)}  -> {key_frames}\n"
                f"  Phases       : {n_phases}\n"
                f"  KF hits      : {n_kf_hits} / {len(key_frames)}\n"
                f"  phase uniq   : {phase_uniq}"
            )

            # Write labels and mark as labeled
            _write_dataset(grp, "phase_labels", ph_labels)
            _write_dataset(grp, "keyframe_labels", kf_labels)
            grp.attrs["labeled"] = True

            processed += 1

    print(
        f"\n{'='*60}\n"
        f"Done.\n"
        f"  Labeled   : {processed} video(s) — phase_labels + keyframe_labels written.\n"
        f"  Unlabeled : {unlabeled} video(s) — labeled=False set, no label datasets written.\n"
        f"  Output H5 : {out_h5_path}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
