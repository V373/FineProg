"""
make_robomimic_dense_reward.py

Compute dense rewards and PBRS rewards from progress labels and write them
to three sibling HDF5 files derived from a single robomimic dataset.

For each demo of length T in the input file, this script produces:
    r_dense[t] = sparse_scale * rewards[t] + progress[t]
    r_pbrs[t]  = sparse_scale * rewards[t] + PBRS_GAMMA * progress[t+1] - progress[t]
with the boundary convention progress_next[T-1] = progress[T-1].

Three output files are written, all sharing a common base name:
    <base>_original.hdf5  : keeps the original `rewards` field unchanged and
                            adds new datasets `dense_rewards` and
                            `PBRS_rewards` (same layout as the previous
                            single-file output).
    <base>_dense.hdf5     : overwrites `rewards` with r_dense, and renames
                            the original rewards field to `original_rewards`.
    <base>_PBRS.hdf5      : overwrites `rewards` with r_pbrs, and renames
                            the original rewards field to `original_rewards`.

In addition, the script reads the source `mask/better`, `mask/okay`,
`mask/worse`, `mask/okay_better`, `mask/worse_better`, and `mask/worse_okay`
arrays (each a 1-D array of demo names) from the input robomimic file and
writes 8 new IQL-style masks into the `mask/` group of every output file:
    IQL_expert_half            = better[: B//2]
    IQL_expert                 = better
    IQL_epxert_okay_halfmix    = concat(better[: B//2], okay[: O//2])
    IQL_expert_worse_halfmix   = concat(better[: B//2], worse[: W//2])
    IQL_expert_okay_worse_halfmix
                               = concat(better[: B//2], okay[: O//2], worse[: W//2])
    IQL_expert_okay            = okay_better
    IQL_expert_worse           = worse_better
    IQL_okay_worse             = worse_okay
The lengths of the first five masks follow the 1 : 2 : 2 : 2 : 3 ratio.

Usage:
    python make_robomimic_dense_reward.py \
        --robomimic_h5 /path/to/low_dim.hdf5 \
        --progress_h5  /path/to/expert_projection_progress.h5 \
        [--output_h5   /path/to/low_dim_v15_reward_labeled.hdf5] \
        [--sparse_scale 1.0]

If --output_h5 is omitted, the base name is derived from --robomimic_h5 by
appending `_reward_labeled`. The three outputs are then:
    <base>_original.hdf5, <base>_dense.hdf5, <base>_PBRS.hdf5.
"""

import argparse
import os
import shutil

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------

PBRS_GAMMA: float = 0.99  # discount factor used in PBRS shaping term


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _demo_sort_key(demo_key: str) -> int:
    """Return the integer suffix of 'demo_<N>'."""
    return int(demo_key.split("_")[-1])


def _video_sort_key(video_id: str) -> int:
    """Return int value of a zero-padded video id string."""
    return int(video_id)


# Names of the new IQL-style masks written into the `mask/` group of every
# output file. Kept as a module-level constant so the order is stable when
# logging dimensions.
IQL_MASK_NAMES = (
    "IQL_expert_half",
    "IQL_expert",
    "IQL_epxert_okay_halfmix",
    "IQL_expert_worse_halfmix",
    "IQL_expert_okay_worse_halfmix",
    "IQL_expert_okay",
    "IQL_expert_worse",
    "IQL_okay_worse",
)


def _build_iql_masks(
    better: np.ndarray,
    okay: np.ndarray,
    worse: np.ndarray,
    okay_better: np.ndarray,
    worse_better: np.ndarray,
    worse_okay: np.ndarray,
) -> dict:
    """Build the IQL-style masks from the source masks.

    `better` / `okay` / `worse` are each a 1-D array of demo-name strings, with
    identical length (B == O == W). If lengths differ, "first half" is taken
    per-source and the halfmix masks have length B/2, B, B/2+O/2, B/2+W/2,
    and B/2+O/2+W/2 — i.e. the 1 : 2 : 2 : 2 : 3 ratio holds when B == O == W.

    `okay_better` / `worse_better` / `worse_okay` are the source dataset's
    combined masks, copied verbatim into `IQL_expert_okay`,
    `IQL_expert_worse`, and `IQL_okay_worse` respectively.
    """
    def _first_half(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr[: len(arr) // 2])

    better_half = _first_half(better)
    okay_half = _first_half(okay)
    worse_half = _first_half(worse)

    return {
        "IQL_expert_half": np.asarray(better_half),
        "IQL_expert": np.asarray(better),
        "IQL_epxert_okay_halfmix": np.concatenate(
            [np.asarray(better_half), np.asarray(okay_half)]
        ),
        "IQL_expert_worse_halfmix": np.concatenate(
            [np.asarray(better_half), np.asarray(worse_half)]
        ),
        "IQL_expert_okay_worse_halfmix": np.concatenate(
            [np.asarray(better_half), np.asarray(okay_half), np.asarray(worse_half)]
        ),
        "IQL_expert_okay": np.asarray(okay_better),
        "IQL_expert_worse": np.asarray(worse_better),
        "IQL_okay_worse": np.asarray(worse_okay),
    }


# ---------------------------------------------------------------------------
# main logic
# ---------------------------------------------------------------------------

def _write_or_replace_dataset(
    grp: "h5py.Group",
    name: str,
    data: np.ndarray,
    dtype: "np.dtype | None" = None,
    compression: str = "gzip",
) -> None:
    """Create (or overwrite) a dataset under `grp` with the given name/data."""
    if name in grp:
        del grp[name]
    # HDF5 has no native object dtype; convert string object arrays to bytes.
    if data.dtype == object:
        data = data.astype("S")
        dtype = None  # let h5py infer the fixed-length byte-string dtype
    grp.create_dataset(
        name,
        data=data,
        dtype=dtype if dtype is not None else data.dtype,
        compression=compression,
    )


def make_labeled_rewards(
    robomimic_h5_path: str,
    progress_h5_path: str,
    output_h5_path: str,
    sparse_scale: float = 1.0,
) -> None:
    """Compute dense + PBRS rewards and write 3 sibling h5 files.

    The three output files are derived from `output_h5_path` by inserting
    `_original`, `_dense`, `_PBRS` before the file extension:
        <stem>_original<ext>  : original `rewards` plus added
                                `dense_rewards` and `PBRS_rewards` datasets.
        <stem>_dense<ext>     : `rewards` overwritten with r_dense;
                                the original `rewards` data is preserved
                                under `original_rewards`.
        <stem>_PBRS<ext>      : `rewards` overwritten with r_pbrs;
                                the original `rewards` data is preserved
                                under `original_rewards`.
    """
    # ------------------------------------------------------------------
    # 1. Derive 3 output paths from the base output path
    # ------------------------------------------------------------------
    base, ext = os.path.splitext(output_h5_path)
    if ext == "":
        ext = ".hdf5"
    out_original = f"{base}_original{ext}"
    out_dense = f"{base}_dense{ext}"
    out_pbrs = f"{base}_PBRS{ext}"

    # ------------------------------------------------------------------
    # 2. Read demo keys + source masks (better/okay/worse) from robomimic file
    # ------------------------------------------------------------------
    with h5py.File(robomimic_h5_path, "r") as f_rob:
        data_grp = f_rob["data"]
        demo_keys = sorted(data_grp.keys(), key=_demo_sort_key)

        # Read source masks (each is a 1-D array of demo-name strings) used
        # to construct the 5 new IQL-style masks. They live under the `mask/`
        # group of the robomimic file.
        if "mask" not in f_rob:
            raise ValueError(
                f"Robomimic HDF5 has no `mask` group: {robomimic_h5_path}"
            )
        mask_grp = f_rob["mask"]
        for src_name in (
            "better",
            "okay",
            "worse",
            "okay_better",
            "worse_better",
            "worse_okay",
        ):
            if src_name not in mask_grp:
                raise ValueError(
                    f"Robomimic HDF5 is missing required mask '{src_name}' "
                    f"under /mask: {robomimic_h5_path}"
                )
        better_mask = mask_grp["better"][:]
        okay_mask = mask_grp["okay"][:]
        worse_mask = mask_grp["worse"][:]
        okay_better_mask = mask_grp["okay_better"][:]
        worse_better_mask = mask_grp["worse_better"][:]
        worse_okay_mask = mask_grp["worse_okay"][:]
        # Decode bytes → str so concatenations stay consistent regardless
        # of how h5py surfaced the underlying fixed-width |S* dtype.
        def _decode(arr: np.ndarray) -> np.ndarray:
            out = np.empty(arr.shape, dtype=object)
            flat_in = arr.ravel()
            flat_out = out.ravel()
            for i, v in enumerate(flat_in):
                flat_out[i] = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
            return out
        better_mask = _decode(better_mask)
        okay_mask = _decode(okay_mask)
        worse_mask = _decode(worse_mask)
        okay_better_mask = _decode(okay_better_mask)
        worse_better_mask = _decode(worse_better_mask)
        worse_okay_mask = _decode(worse_okay_mask)

    n_demos = len(demo_keys)
    if n_demos == 0:
        raise ValueError("No demo groups found under /data in robomimic HDF5.")

    # Build the new IQL-style masks up front (input-independent of rewards)
    iql_masks = _build_iql_masks(
        better_mask,
        okay_mask,
        worse_mask,
        okay_better_mask,
        worse_better_mask,
        worse_okay_mask,
    )
    print("[INFO] Built IQL masks (name: length):")
    for name in IQL_MASK_NAMES:
        print(f"        {name:32s}: {len(iql_masks[name])}")

    # ------------------------------------------------------------------
    # 3. Read video ids from progress file (sorted by numeric value)
    # ------------------------------------------------------------------
    with h5py.File(progress_h5_path, "r") as f_prog:
        nonexperts_grp = f_prog["nonexperts"]
        video_ids = sorted(nonexperts_grp.keys(), key=_video_sort_key)

    n_videos = len(video_ids)
    if n_demos != n_videos:
        raise ValueError(
            f"Demo count mismatch: robomimic has {n_demos} demos, "
            f"progress file has {n_videos} videos."
        )

    print(f"[INFO] {n_demos} demos / {n_videos} videos – counts match.")

    # ------------------------------------------------------------------
    # 4. Copy robomimic HDF5 to each of the 3 output paths
    # ------------------------------------------------------------------
    for out_path in (out_original, out_dense, out_pbrs):
        if os.path.exists(out_path):
            print(
                f"[WARNING] Output file already exists and will be overwritten: "
                f"{out_path}"
            )
        print(f"[INFO] Copying {robomimic_h5_path}  →  {out_path}")
        shutil.copy2(robomimic_h5_path, out_path)

    # ------------------------------------------------------------------
    # 5. Open all 3 output files (r+) and progress file, then write rewards
    # ------------------------------------------------------------------
    formula_dense = f"dense_rewards = {sparse_scale:g} * rewards + progress_label"
    formula_pbrs = (
        f"PBRS_rewards = {sparse_scale:g} * rewards "
        f"+ {PBRS_GAMMA:g} * progress_next - progress_label"
    )

    with h5py.File(out_original, "r+") as f_orig, \
         h5py.File(out_dense, "r+") as f_dense, \
         h5py.File(out_pbrs, "r+") as f_pbrs, \
         h5py.File(progress_h5_path, "r") as f_prog:

        data_orig = f_orig["data"]
        data_dense = f_dense["data"]
        data_pbrs = f_pbrs["data"]

        for demo_key, video_id in zip(demo_keys, video_ids):
            # -- load original sparse rewards (all 3 files are identical at this point) --
            sparse_ds = data_orig[demo_key]["rewards"]
            orig_rewards = sparse_ds[:]
            orig_dtype = sparse_ds.dtype
            if orig_rewards.ndim != 1:
                raise ValueError(
                    f"{demo_key}/rewards must be 1-D, got shape {orig_rewards.shape}."
                )
            T = orig_rewards.shape[0]

            # -- load progress label --
            progress_label = f_prog[f"nonexperts/{video_id}/progress_label"][:]
            if progress_label.ndim != 1:
                raise ValueError(
                    f"nonexperts/{video_id}/progress_label must be 1-D, "
                    f"got shape {progress_label.shape}."
                )
            if progress_label.shape[0] != T:
                raise ValueError(
                    f"Length mismatch for {demo_key} ↔ {video_id}: "
                    f"rewards has {T} steps, progress_label has "
                    f"{progress_label.shape[0]} steps."
                )

            # -- compute dense reward --
            dense_reward = (
                sparse_scale * orig_rewards + progress_label
            ).astype(np.float32)
            if not np.all(np.isfinite(dense_reward)):
                raise ValueError(
                    f"dense_rewards for {demo_key} contains nan or inf values."
                )

            # -- compute PBRS reward (boundary: progress_next[T-1] = progress[T-1]) --
            progress_next = np.empty_like(progress_label)
            progress_next[:-1] = progress_label[1:]
            progress_next[-1] = progress_label[-1]
            pbrs_reward = (
                sparse_scale * orig_rewards
                + PBRS_GAMMA * progress_next
                - progress_label
            ).astype(np.float32)
            if not np.all(np.isfinite(pbrs_reward)):
                raise ValueError(
                    f"PBRS_rewards for {demo_key} contains nan or inf values."
                )

            # -- write to _original: keep `rewards` unchanged, add dense + PBRS fields --
            dgr = data_orig[demo_key]
            _write_or_replace_dataset(dgr, "dense_rewards", dense_reward, np.float32)
            _write_or_replace_dataset(dgr, "PBRS_rewards", pbrs_reward, np.float32)

            # -- write to _dense: replace `rewards` with dense, save original as `original_rewards` --
            dgr = data_dense[demo_key]
            del dgr["rewards"]
            _write_or_replace_dataset(dgr, "original_rewards", orig_rewards, orig_dtype)
            _write_or_replace_dataset(dgr, "rewards", dense_reward, np.float32)

            # -- write to _PBRS: replace `rewards` with PBRS, save original as `original_rewards` --
            dgr = data_pbrs[demo_key]
            del dgr["rewards"]
            _write_or_replace_dataset(dgr, "original_rewards", orig_rewards, orig_dtype)
            _write_or_replace_dataset(dgr, "rewards", pbrs_reward, np.float32)

            print(
                f"  {demo_key} | video {video_id} | T={T:4d} | "
                f"sparse [{orig_rewards.min():.4f}, {orig_rewards.max():.4f}] | "
                f"progress [{progress_label.min():.4f}, {progress_label.max():.4f}] | "
                f"dense [{dense_reward.min():.4f}, {dense_reward.max():.4f}] | "
                f"pbrs  [{pbrs_reward.min():.4f}, {pbrs_reward.max():.4f}]"
            )

        # -- write metadata attrs on all 3 output files --
        for f_out, variant in (
            (f_orig, "original"),
            (f_dense, "dense"),
            (f_pbrs, "PBRS"),
        ):
            data_grp_out = f_out["data"]
            common_attrs = {
                "progress_h5_path": progress_h5_path,
                "sparse_scale": float(sparse_scale),
                "pbrs_gamma": float(PBRS_GAMMA),
                "dense_reward_formula": formula_dense,
                "pbrs_formula": formula_pbrs,
                "output_variant": variant,
                "dense_reward_source": "sparse_reward_plus_progress_label",
            }
            for k, v in common_attrs.items():
                f_out.attrs[k] = v
                data_grp_out.attrs[k] = v

            # -- write the 5 IQL-style masks into the file's `mask/` group --
            if "mask" not in f_out:
                f_out.create_group("mask")
            out_mask_grp = f_out["mask"]
            for iql_name in IQL_MASK_NAMES:
                _write_or_replace_dataset(
                    out_mask_grp,
                    iql_name,
                    iql_masks[iql_name],
                    compression="gzip",
                )

    print(f"[DONE] Original (rewards + dense_rewards + PBRS_rewards) → {out_original}")
    print(f"[DONE] Dense   (rewards=dense,  original_rewards=original) → {out_dense}")
    print(f"[DONE] PBRS    (rewards=PBRS,   original_rewards=original) → {out_pbrs}")
    print(
        "[DONE] Added 5 IQL masks "
        f"({', '.join(f'{n}={len(iql_masks[n])}' for n in IQL_MASK_NAMES)}) "
        "to the `mask/` group of all 3 output files."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dense and PBRS rewards from progress labels and write them "
            "to 3 sibling HDF5 files (suffixed _original, _dense, _PBRS)."
        ),
    )
    parser.add_argument(
        "--robomimic_h5",
        required=True,
        help="Path to the original robomimic HDF5 file (e.g. low_dim_v15.hdf5).",
    )
    parser.add_argument(
        "--progress_h5",
        required=True,
        help="Path to the progress-labeled HDF5 file (expert_projection output).",
    )
    parser.add_argument(
        "--output_h5",
        default=None,
        help=(
            "Base path for the 3 output HDF5 files. The 3 files are written as "
            "<stem>_original<ext>, <stem>_dense<ext>, <stem>_PBRS<ext>. "
            "Defaults to <robomimic_h5_stem>_reward_labeled.hdf5 in the same directory."
        ),
    )
    parser.add_argument(
        "--sparse_scale",
        type=float,
        default=1.0,
        help=(
            "Scaling factor applied to the original sparse reward before "
            "computing dense / PBRS rewards. Defaults to 1."
        ),
    )
    args = parser.parse_args()

    if args.output_h5 is None:
        base, ext = os.path.splitext(args.robomimic_h5)
        args.output_h5 = base + "_reward_labeled" + ext

    make_labeled_rewards(
        robomimic_h5_path=args.robomimic_h5,
        progress_h5_path=args.progress_h5,
        output_h5_path=args.output_h5,
        sparse_scale=args.sparse_scale,
    )


if __name__ == "__main__":
    main()
