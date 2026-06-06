"""
make_robomimic_video.py

Single-file script to:
  Step A: Generate an image-observation HDF5 from a raw robomimic dataset
          by calling dataset_states_to_obs.py.
  Step B: Read the image HDF5 directly and write one MP4 per demo into
          a videos/ subdirectory (no playback_dataset.py, no long concat video).
  Sanity Check: Validate the raw HDF5, image HDF5, and per-demo MP4s.

All parameters are hard-coded in CONFIG below – no argparse.
"""

import sys
import subprocess
import pathlib

import h5py
import imageio.v2 as imageio
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG – edit these values to control everything
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Paths ────────────────────────────────────────────────────────────────
    "ROBOMIMIC_ROOT": pathlib.Path(__file__).parent.resolve(),
    "DATASET_ROOT":   pathlib.Path(__file__).parent.resolve() / "datasets",

    # ── Dataset selection ────────────────────────────────────────────────────
    "TASK":         "lift",   # e.g. "square", "lift", "can", "transport"
    "DATASET_TYPE": "mh",       # e.g. "ph", "mh", "mg", "paired"

    # ── File names ──────────────────────────────────────────────────────────
    "RAW_HDF5_NAME":   "demo_v15.hdf5",
    "IMAGE_HDF5_NAME": "image_224_v15.hdf5",

    # ── Camera / rendering ───────────────────────────────────────────────────
    "CAMERA_NAMES":  ["agentview"],  # one camera → strict 224×224 frames
    "CAMERA_HEIGHT": 224,
    "CAMERA_WIDTH":  224,

    # ── dataset_states_to_obs.py options ────────────────────────────────────
    "DONE_MODE":        2,      # 0=success states, 1=end of traj, 2=both
    "SHAPED_REWARD":    False,  # --shaped
    "USE_DEPTH":        False,  # --depth
    "COPY_REWARDS":     False,  # --copy_rewards
    "COPY_DONES":       False,  # --copy_dones
    "COMPRESS":         True,   # --compress
    "EXCLUDE_NEXT_OBS": True,   # --exclude-next-obs
    # Limit trajectories for obs generation (None = all)
    "N_TRAJECTORIES_FOR_OBS": None,

    # ── Per-demo video options ───────────────────────────────────────────────
    "FPS":        20,   # output video frame rate
    "VIDEO_SKIP": 1,    # write one frame every VIDEO_SKIP timesteps
    # Stop after this many demos (None = all demos in HDF5)
    "MAX_VIDEOS": None,

    # ── Sanity-check settings ────────────────────────────────────────────────
    "EXPECTED_CHANNELS": 3,
    "EXPECTED_DTYPE":    "uint8",

    # ── Overwrite behaviour ──────────────────────────────────────────────────
    "OVERWRITE_IMAGE_HDF5": False,  # True → delete & regenerate image HDF5
    "OVERWRITE_VIDEOS":     True,   # True → overwrite existing per-demo MP4s
}


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def dataset_dir() -> pathlib.Path:
    return pathlib.Path(CONFIG["DATASET_ROOT"]) / CONFIG["TASK"] / CONFIG["DATASET_TYPE"]


def raw_hdf5_path() -> pathlib.Path:
    return dataset_dir() / CONFIG["RAW_HDF5_NAME"]


def image_hdf5_path() -> pathlib.Path:
    return dataset_dir() / CONFIG["IMAGE_HDF5_NAME"]


def videos_dir() -> pathlib.Path:
    return image_hdf5_path().parent / "videos"


def script_path(name: str) -> pathlib.Path:
    return pathlib.Path(CONFIG["ROBOMIMIC_ROOT"]) / "robomimic" / "scripts" / name


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def run_cmd(cmd: list) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n[CMD] {printable}\n")
    subprocess.run([str(c) for c in cmd], check=True)


def obs_key_from_camera(camera_name: str) -> str:
    """agentview → agentview_image,  robot0_eye_in_hand → robot0_eye_in_hand_image"""
    return f"{camera_name}_image"


# ─────────────────────────────────────────────────────────────────────────────
# Step A – generate image HDF5 from raw dataset states
# ─────────────────────────────────────────────────────────────────────────────

def generate_image_hdf5() -> None:
    out_path = image_hdf5_path()

    if out_path.exists():
        if CONFIG["OVERWRITE_IMAGE_HDF5"]:
            print(f"[Step A] OVERWRITE_IMAGE_HDF5=True – removing: {out_path}")
            out_path.unlink()
        else:
            print(f"[Step A] Image HDF5 exists, skipping (OVERWRITE_IMAGE_HDF5=False): {out_path}")
            return

    print(f"[Step A] Generating image HDF5 → {out_path}")

    cmd = [
        sys.executable,
        script_path("dataset_states_to_obs.py"),
        "--dataset",       raw_hdf5_path(),
        "--output_name",   CONFIG["IMAGE_HDF5_NAME"],  # script writes beside --dataset
        "--done_mode",     str(CONFIG["DONE_MODE"]),
        "--camera_names",  *CONFIG["CAMERA_NAMES"],
        "--camera_height", str(CONFIG["CAMERA_HEIGHT"]),
        "--camera_width",  str(CONFIG["CAMERA_WIDTH"]),
    ]

    if CONFIG["SHAPED_REWARD"]:
        cmd.append("--shaped")
    if CONFIG["USE_DEPTH"]:
        cmd.append("--depth")
    if CONFIG["COPY_REWARDS"]:
        cmd.append("--copy_rewards")
    if CONFIG["COPY_DONES"]:
        cmd.append("--copy_dones")
    if CONFIG["COMPRESS"]:
        cmd.append("--compress")
    if CONFIG["EXCLUDE_NEXT_OBS"]:
        cmd.append("--exclude-next-obs")
    if CONFIG["N_TRAJECTORIES_FOR_OBS"] is not None:
        cmd.extend(["--n", str(CONFIG["N_TRAJECTORIES_FOR_OBS"])])

    run_cmd(cmd)
    print(f"[Step A] Done → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step B – write one MP4 per demo directly from the image HDF5
# ─────────────────────────────────────────────────────────────────────────────

def generate_individual_videos() -> None:
    hdf5_path = image_hdf5_path()
    out_dir = videos_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    fps        = CONFIG["FPS"]
    video_skip = CONFIG["VIDEO_SKIP"]
    max_videos = CONFIG["MAX_VIDEOS"]
    overwrite  = CONFIG["OVERWRITE_VIDEOS"]
    cam_names  = CONFIG["CAMERA_NAMES"]

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

    if max_videos is not None:
        demos = demos[:max_videos]

    print(f"\n[Step B] Writing {len(demos)} individual MP4s → {out_dir}")

    for i, demo_name in enumerate(demos):
        out_path = out_dir / f"{demo_name}.mp4"

        if out_path.exists():
            if overwrite:
                out_path.unlink()
            else:
                print(f"  [{i+1}/{len(demos)}] skip (exists): {out_path.name}")
                continue

        # Read image arrays for every camera  [T, H, W, C]
        with h5py.File(hdf5_path, "r") as f:
            cam_arrays = [
                f[f"data/{demo_name}/obs/{obs_key_from_camera(cam)}"][:]
                for cam in cam_names
            ]

        T = cam_arrays[0].shape[0]

        writer = imageio.get_writer(str(out_path), fps=fps)
        try:
            for t in range(0, T, video_skip):
                if len(cam_arrays) == 1:
                    frame = cam_arrays[0][t]              # (H, W, 3)
                else:
                    frame = np.concatenate(               # (H, W*N, 3)
                        [arr[t] for arr in cam_arrays], axis=1
                    )
                writer.append_data(frame)
        finally:
            writer.close()

        print(f"  [{i+1}/{len(demos)}] {out_path.name}  T={T}  frames={len(range(0, T, video_skip))}")

    print(f"[Step B] Done – videos in {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check_raw_hdf5() -> None:
    p = raw_hdf5_path()
    print(f"\n[Sanity] Raw HDF5: {p}")
    if not p.exists():
        raise FileNotFoundError(
            f"Raw HDF5 not found: {p}\n"
            "Please ensure the dataset is downloaded before running this script."
        )
    print(f"  exists ✓  ({p.stat().st_size / 1e6:.1f} MB)")


def sanity_check_image_hdf5() -> None:
    p = image_hdf5_path()
    print(f"\n[Sanity] Image HDF5: {p}")

    if not p.exists():
        raise FileNotFoundError(f"Image HDF5 not found: {p}")

    H = CONFIG["CAMERA_HEIGHT"]
    W = CONFIG["CAMERA_WIDTH"]
    C = CONFIG["EXPECTED_CHANNELS"]
    expected_dtype = CONFIG["EXPECTED_DTYPE"]

    with h5py.File(p, "r") as f:
        if "data" not in f:
            raise KeyError(f"Missing root 'data' group in: {p}")
        data_grp = f["data"]

        demos = sorted(data_grp.keys(), key=lambda x: int(x.split("_")[1]))
        if len(demos) == 0:
            raise ValueError(f"'data' group has no demos in: {p}")
        print(f"  demos : {len(demos)}  (first: {demos[0]})")

        first_demo = demos[0]
        if "obs" not in data_grp[first_demo]:
            raise KeyError(f"Demo '{first_demo}' has no 'obs' group in: {p}")
        obs_grp = data_grp[first_demo]["obs"]

        for cam in CONFIG["CAMERA_NAMES"]:
            obs_key = obs_key_from_camera(cam)
            if obs_key not in obs_grp:
                raise KeyError(
                    f"Key '{obs_key}' missing from {first_demo}/obs.\n"
                    f"  Available: {list(obs_grp.keys())}\n  File: {p}"
                )

            tensor = obs_grp[obs_key]
            shape  = tensor.shape
            dtype  = str(tensor.dtype)

            print(f"  {first_demo}/obs/{obs_key} : shape={shape}  dtype={dtype}")

            if tensor.ndim != 4:
                raise ValueError(
                    f"Expected 4-D tensor for '{obs_key}', got {tensor.ndim}-D: {shape}"
                )

            T_a, H_a, W_a, C_a = shape
            if H_a != H or W_a != W or C_a != C:
                raise ValueError(
                    f"Shape mismatch for '{obs_key}'.\n"
                    f"  Expected [T, {H}, {W}, {C}]  Got {list(shape)}"
                )

            if dtype != expected_dtype:
                raise ValueError(
                    f"dtype mismatch for '{obs_key}': expected {expected_dtype}, got {dtype}"
                )

    print(f"[Sanity] Image HDF5 passed ✓")


def sanity_check_videos() -> None:
    out_dir = videos_dir()
    print(f"\n[Sanity] Videos dir: {out_dir}")

    if not out_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {out_dir}")

    # Determine which demos we expected to generate
    with h5py.File(image_hdf5_path(), "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

    if CONFIG["MAX_VIDEOS"] is not None:
        demos = demos[: CONFIG["MAX_VIDEOS"]]

    H  = CONFIG["CAMERA_HEIGHT"]
    W  = CONFIG["CAMERA_WIDTH"]
    N  = len(CONFIG["CAMERA_NAMES"])
    expected_shape = (H, W * N, 3)

    print(f"  expected videos : {len(demos)}  expected first-frame shape : {expected_shape}")

    for demo_name in demos:
        mp4 = out_dir / f"{demo_name}.mp4"
        if not mp4.exists():
            raise FileNotFoundError(f"Video missing: {mp4}")
        if mp4.stat().st_size == 0:
            raise ValueError(f"Video is empty (0 bytes): {mp4}")

    # Spot-check first video's first frame
    first_mp4 = out_dir / f"{demos[0]}.mp4"
    reader = imageio.get_reader(str(first_mp4))
    try:
        first_frame = reader.get_data(0)
    finally:
        reader.close()

    print(f"  {first_mp4.name} first frame : shape={first_frame.shape}  dtype={first_frame.dtype}")

    if first_frame.shape != expected_shape:
        raise ValueError(
            f"First-frame shape mismatch.\n"
            f"  Expected {expected_shape}  Got {first_frame.shape}"
        )

    print(f"[Sanity] Videos passed ✓  ({len(demos)} MP4s in {out_dir})")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("make_robomimic_video.py")
    print("=" * 70)
    print(f"  ROBOMIMIC_ROOT       : {CONFIG['ROBOMIMIC_ROOT']}")
    print(f"  DATASET_ROOT         : {CONFIG['DATASET_ROOT']}")
    print(f"  Task / Type          : {CONFIG['TASK']} / {CONFIG['DATASET_TYPE']}")
    print(f"  Raw HDF5             : {raw_hdf5_path()}")
    print(f"  Image HDF5           : {image_hdf5_path()}")
    print(f"  Videos dir           : {videos_dir()}")
    print(f"  Camera(s)            : {CONFIG['CAMERA_NAMES']}  {CONFIG['CAMERA_HEIGHT']}×{CONFIG['CAMERA_WIDTH']}")
    print(f"  FPS / VIDEO_SKIP     : {CONFIG['FPS']} / {CONFIG['VIDEO_SKIP']}")
    print(f"  MAX_VIDEOS           : {CONFIG['MAX_VIDEOS']}")
    print(f"  OVERWRITE_IMAGE_HDF5 : {CONFIG['OVERWRITE_IMAGE_HDF5']}")
    print(f"  OVERWRITE_VIDEOS     : {CONFIG['OVERWRITE_VIDEOS']}")
    print("=" * 70)

    sanity_check_raw_hdf5()
    generate_image_hdf5()
    sanity_check_image_hdf5()
    generate_individual_videos()
    sanity_check_videos()

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Image HDF5 : {image_hdf5_path()}")
    print(f"  Videos     : {videos_dir()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
