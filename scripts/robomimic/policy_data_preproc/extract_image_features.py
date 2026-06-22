"""
Extract frozen ResNet18 image features from a robomimic-format HDF5 dataset.

Processing pipeline (mirrors robomimic IQL training dataflow exactly):
  1. Read uint8 HWC images from h5 (obs/<key> and next_obs/<key>)
  2. Cast to float32, divide by 255  -> [0.0, 1.0]
  3. HWC -> CHW
  4. ImageNet normalization: (x - mean) / std
  5. Pass through frozen pretrained ResNet18 (conv backbone only, no avgpool/fc)
     -> [512, H/32, W/32] per frame
  6. Write everything else as-is, replace image obs with feature tensors [T, 512, H/32, W/32]

Output h5 has identical group/attribute structure to the input; only the
specified image obs keys are replaced with extracted feature arrays.

Usage:
  python extract_image_features.py --input <in.hdf5> --output <out.hdf5>

Optional:
  --batch-size   Number of frames to process at once on GPU (default: 64)
  --device       'cuda' or 'cpu' (default: cuda if available)
"""

import argparse
import math
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision import models as vision_models

# ---------------------------------------------------------------------------
# CONFIGURATION – hardcode which observation keys contain images to process
# ---------------------------------------------------------------------------
IMAGE_OBS_KEYS = [
    "agentview_image",
    "robot0_eye_in_hand_image",
]

# ImageNet normalization constants (matches ResNet18Conv imagenet_norm=True)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------


def build_resnet18_backbone(device: torch.device) -> nn.Module:
    """
    Load pretrained ResNet18, strip avgpool + fc (last 2 children),
    freeze all parameters, set to eval mode.
    Returns a module that maps [B, 3, H, W] -> [B, 512, H/32, W/32].
    """
    net = vision_models.resnet18(weights=vision_models.ResNet18_Weights.DEFAULT)
    # Remove avgpool and fc  (identical to robomimic ResNet18Conv: children()[:-2])
    backbone = nn.Sequential(*list(net.children())[:-2])
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    backbone.to(device)
    return backbone


def process_frames_numpy(frames_uint8_hwc: np.ndarray) -> np.ndarray:
    """
    Apply steps 2-4 of the pipeline in numpy, returning float32 CHW array.

    Args:
        frames_uint8_hwc: [T, H, W, C] uint8

    Returns:
        [T, C, H, W] float32, ImageNet-normalized
    """
    frames = frames_uint8_hwc.astype(np.float32) / 255.0          # [0,1]
    frames = frames.clip(0.0, 1.0)
    # HWC -> CHW:  (T, H, W, C) -> (T, C, H, W)
    frames = frames.transpose(0, 3, 1, 2)
    # ImageNet normalization broadcast over spatial dims
    mean = IMAGENET_MEAN[None, :, None, None]  # [1, 3, 1, 1]
    std  = IMAGENET_STD[None, :, None, None]
    frames = (frames - mean) / std
    return frames


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    frames_chw: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run backbone on float32 CHW frames in batches.

    Args:
        frames_chw: [T, 3, H, W] float32, ImageNet-normalized
        batch_size: frames per GPU batch

    Returns:
        [T, 512, H//32_ceil, W//32_ceil] float32 numpy array
    """
    T = frames_chw.shape[0]
    all_feats = []
    for start in range(0, T, batch_size):
        chunk = frames_chw[start: start + batch_size]
        tensor = torch.from_numpy(chunk).to(device)
        feats = backbone(tensor)           # [B, 512, h, w]
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)  # [T, 512, h, w]


def copy_attrs(src, dst):
    """Copy all HDF5 attributes from src to dst."""
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def copy_group_structure(src_file: h5py.File, dst_file: h5py.File,
                         image_keys_set: set,
                         backbone: nn.Module,
                         batch_size: int,
                         device: torch.device):
    """
    Recursively copy src_file -> dst_file.
    For obs/<image_key> and next_obs/<image_key> datasets, replace with features.
    Everything else is copied verbatim.
    """
    def _copy_item(src_item, dst_parent, name, obs_prefix: str):
        """
        src_item : h5py.Group or h5py.Dataset
        dst_parent : h5py.Group (destination parent)
        name : str  (key under dst_parent)
        obs_prefix : 'obs' | 'next_obs' | '' ('' means not inside an obs group)
        """
        if isinstance(src_item, h5py.Group):
            dst_grp = dst_parent.create_group(name)
            copy_attrs(src_item, dst_grp)
            for child_name, child_item in src_item.items():
                # Determine if we are stepping into an obs/next_obs group
                if name in ("obs", "next_obs") and obs_prefix == "":
                    child_prefix = name
                else:
                    child_prefix = obs_prefix
                _copy_item(child_item, dst_grp, child_name, child_prefix)

        elif isinstance(src_item, h5py.Dataset):
            # Check if this is an image key that should be replaced
            if obs_prefix in ("obs", "next_obs") and name in image_keys_set:
                _process_and_write_image_dataset(src_item, dst_parent, name,
                                                 backbone, batch_size, device)
            else:
                # Verbatim copy
                data = src_item[()]
                ds = dst_parent.create_dataset(name, data=data,
                                               compression=src_item.compression,
                                               compression_opts=src_item.compression_opts)
                copy_attrs(src_item, ds)
        else:
            # Named type or other – skip
            pass

    for top_name, top_item in src_file.items():
        _copy_item(top_item, dst_file, top_name, "")

    copy_attrs(src_file, dst_file)


def _process_and_write_image_dataset(src_ds: h5py.Dataset,
                                     dst_parent: h5py.Group,
                                     name: str,
                                     backbone: nn.Module,
                                     batch_size: int,
                                     device: torch.device):
    """
    Read a [T, H, W, 3] uint8 dataset, run full pipeline, write [T, 512, h, w].
    """
    raw = src_ds[()]   # numpy, uint8, [T, H, W, 3]
    if raw.ndim != 4 or raw.shape[-1] != 3:
        # Not a standard RGB image tensor – copy verbatim and warn
        print(f"  WARNING: {name} has unexpected shape {raw.shape}, copying as-is.")
        ds = dst_parent.create_dataset(name, data=raw,
                                       compression=src_ds.compression,
                                       compression_opts=src_ds.compression_opts)
        copy_attrs(src_ds, ds)
        return

    T, H, W, C = raw.shape
    processed = process_frames_numpy(raw)          # [T, 3, H, W] float32 ImageNet-norm
    feats = extract_features(backbone, processed, batch_size, device)  # [T, 512, h, w]

    ds = dst_parent.create_dataset(name, data=feats.astype(np.float32))
    copy_attrs(src_ds, ds)
    # Record provenance
    ds.attrs["_preproc_source_shape"] = list(raw.shape)
    ds.attrs["_preproc_pipeline"] = "uint8/255 -> HWC2CHW -> imagenet_norm -> ResNet18_conv_backbone"
    print(f"    {name}: {list(raw.shape)} -> {list(feats.shape)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ResNet18 image features from robomimic HDF5 dataset."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to input robomimic HDF5 file.")
    parser.add_argument("--output", "-o", required=True,
                        help="Path for output HDF5 file (will be created/overwritten).")
    parser.add_argument("--batch-size", "-b", type=int, default=64,
                        help="Number of frames per GPU batch (default: 64).")
    parser.add_argument("--device", "-d", default=None,
                        help="Torch device: 'cuda', 'cuda:0', 'cpu', etc. "
                             "Defaults to 'cuda' if available.")
    args = parser.parse_args()

    # ---- device setup ----
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---- validate input ----
    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # ---- build backbone ----
    print("Loading pretrained ResNet18 backbone (frozen)...")
    backbone = build_resnet18_backbone(device)
    print("Backbone ready.")

    image_keys_set = set(IMAGE_OBS_KEYS)
    print(f"Image keys to process: {sorted(image_keys_set)}")

    # ---- process ----
    print(f"\nInput : {args.input}")
    print(f"Output: {args.output}\n")

    with h5py.File(args.input, "r") as src_f, \
         h5py.File(args.output, "w") as dst_f:

        # Count demos for progress reporting
        data_grp = src_f.get("data")
        if data_grp is None:
            print("WARNING: no 'data' group found – copying everything verbatim.")
            demos = []
        else:
            demos = sorted(data_grp.keys(),
                           key=lambda k: int(k[5:]) if k.startswith("demo_") else 0)
        n_demos = len(demos)
        print(f"Found {n_demos} demonstrations.\n")

        # Copy the whole file, replacing image datasets along the way
        copy_group_structure(src_f, dst_f, image_keys_set,
                             backbone, args.batch_size, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
