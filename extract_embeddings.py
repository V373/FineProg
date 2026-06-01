"""
Extract embeddings from a trained TCCEncoder for all videos in an H5 dataset.

Pipeline:
    1. Load configs_v2/extract.yaml via ConfigV2
    2. Build DataLoader with sample_all=True  (batch_size forced to 1)
    3. Load TCCEncoder weights from checkpoint
    4. Run encoder in chunks of encoder.clip_len across each full video
    5. Save results to an H5 file

Output H5 format:
    /videos/<video_id>/
        embeddings      [T_out, D]  float32
        target_steps    [T_out]     int64
        attrs:
            seq_len     int
            action_id   int

Usage:
    python extract_embeddings.py
    python extract_embeddings.py --extract_h5_path <dataset.h5>
"""


import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Make sure project root is on the path so relative imports work when the
# script is invoked from any working directory.
_PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJ_ROOT))

from dataset_preparation.h5vid_dataset import build_dataloader
from models.encoder import TCCEncoder
# [v2] V2 config resolver (independent of old config system)
from utils.config_v2 import ConfigV2


# ---------------------------------------------------------------------------
# [v2] V2 config constants
# ---------------------------------------------------------------------------
_V2_EXTRACT_YAML = str((Path(__file__).parent / "configs_v2" / "extract.yaml"))


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load model weights from a PyTorch checkpoint.

    Supports:
    - bare state_dict saved via torch.save(model.state_dict(), path)
    - wrapped dict with key 'model_state_dict' or 'state_dict'
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt  # assume it is already a state_dict
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    print(f"[load_checkpoint] Loaded weights from {ckpt_path}")


# ---------------------------------------------------------------------------
# Chunked forward pass
# ---------------------------------------------------------------------------

def _run_encoder_chunked(
    encoder: TCCEncoder,
    frames: torch.Tensor,
    frames_per_batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Run encoder over a single video's frames in chunks of frames_per_batch.

    TCCEncoder.forward asserts input clip_len == encoder.clip_len.
    We therefore split T_out into chunks; the last chunk is right-padded with
    its final frame when it is shorter than frames_per_batch, then trimmed.

    Args:
        frames:           [1, T_out, context_size, 3, H, W]
        frames_per_batch: chunk size (must equal encoder.clip_len)
        device:           torch device

    Returns:
        embeddings: [1, T_out, D]
    """
    frames = frames.to(device)
    T_out = frames.shape[1]
    embs_chunks = []

    for start in range(0, T_out, frames_per_batch):
        end = min(start + frames_per_batch, T_out)
        chunk = frames[:, start:end, ...]          # [1, actual_len, ctx, 3, H, W]
        actual_len = end - start

        # Pad the last (possibly short) chunk to exactly frames_per_batch
        if actual_len < frames_per_batch:
            pad_len = frames_per_batch - actual_len
            # Repeat last frame pad_len times
            pad = chunk[:, -1:, ...].expand(-1, pad_len, -1, -1, -1, -1)
            chunk = torch.cat([chunk, pad], dim=1)  # [1, frames_per_batch, ...]

        embs = encoder(chunk)                      # [1, frames_per_batch, D]
        embs_chunks.append(embs[:, :actual_len, :])  # trim padding back off

    return torch.cat(embs_chunks, dim=1)           # [1, T_out, D]


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_embeddings(
    encoder: TCCEncoder,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    frames_per_batch: int | None = None,
) -> list[dict]:
    """Extract embeddings for every video in dataloader.

    Requires the dataloader to have been built with sample_all=True
    (batch_size is therefore 1).

    Args:
        encoder:          TCCEncoder in eval() mode
        dataloader:       DataLoader with sample_all=True, batch_size=1
        device:           torch device
        frames_per_batch: frames per encoder forward pass;
                          defaults to encoder.clip_len

    Returns:
        List of dicts, one per video:
            video_id      str
            embeddings    np.ndarray  [T_out, D]  float32
            target_steps  np.ndarray  [T_out]     int64
            seq_len       int
            action_id     int
    """
    if frames_per_batch is None:
        frames_per_batch = encoder.clip_len

    results = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # frames: [1, T_out, context_size, 3, H, W]
            frames = batch["frames"]
            T_out = frames.shape[1]

            # [1, T_out, D]
            embs = _run_encoder_chunked(encoder, frames, frames_per_batch, device)

            embs_np = embs[0].cpu().numpy()                     # [T_out, D]
            target_steps_np = batch["target_steps"][0].numpy()  # [T_out]

            entry = {
                "video_id":     batch["video_id"][0],
                "embeddings":   embs_np,
                "target_steps": target_steps_np,
                "seq_len":      int(batch["seq_len"][0].item()),
                "action_id":    int(batch["action_id"][0].item()),
            }
            results.append(entry)

            print(
                f"  [{i + 1}/{len(dataloader)}]  video_id={entry['video_id']}"
                f"  T_out={T_out}  embeddings={embs_np.shape}"
            )

    return results


# ---------------------------------------------------------------------------
# Save to H5
# ---------------------------------------------------------------------------

def save_embeddings_h5(results: list[dict], save_path: str) -> None:
    """Save extracted embeddings to an H5 file.

    Structure:
        /videos/<video_id>/
            embeddings      [T_out, D]  float32  (gzip compressed)
            target_steps    [T_out]     int64
            attrs:
                seq_len     int
                action_id   int

    Args:
        results:   list of dicts from extract_embeddings()
        save_path: destination H5 file path
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(save_path, "w") as f:
        videos_grp = f.create_group("videos")
        for entry in results:
            vid = videos_grp.create_group(entry["video_id"])
            vid.create_dataset(
                "embeddings",
                data=entry["embeddings"].astype(np.float32),
                compression="gzip",
            )
            vid.create_dataset(
                "target_steps",
                data=entry["target_steps"].astype(np.int64),
            )
            vid.attrs["seq_len"]   = entry["seq_len"]
            vid.attrs["action_id"] = entry["action_id"]
    print(f"[save_embeddings_h5] Saved {len(results)} videos to {save_path}")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(extract_h5_path: str | None = None,
         register: bool = False, register_alias: str | None = None) -> None:
    """Full embedding extraction pipeline."""
    # [v2] Load extract stage config via V2 resolver
    _ext_v2 = ConfigV2()
    cfg = _ext_v2.load_extract()                     # [v2] resolves checkpoint_path, extract_h5_path, embedding_save_path
    print(f"[main] [v2] extract_h5_path : {cfg.get('extract_h5_path')}")
    print(f"[main] [v2] checkpoint_path : {cfg.get('checkpoint_path')}")
    print(f"[main] [v2] embedding_save_path: {cfg.get('embedding_save_path')}")

    # CLI --extract_h5_path still supported as override
    if extract_h5_path is not None:
        cfg["extract_h5_path"] = extract_h5_path
        print(f"[main] extract_h5_path overridden by CLI: {extract_h5_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Device: {device}")

    # --- DataLoader ---
    # sample_all is explicitly overridden to True regardless of config value.
    dataloader = build_dataloader(                       # [v2]
        config_path=_V2_EXTRACT_YAML,                   # [v2] configs_v2/extract.yaml has clip_len/context_size/etc.
        sample_all=True,                                # explicit override
        sample_all_stride=cfg.get("sample_all_stride", 1),
        shuffle=False,                                  # no shuffling for export
        num_workers=cfg.get("num_workers", 0),
        split="extract",
        h5_path_override=cfg["extract_h5_path"],        # [v2] resolved absolute path
    )

    # --- Encoder ---
    encoder = TCCEncoder(config_path=_V2_EXTRACT_YAML)  # [v2] has clip_len, context_size, context_stride
    encoder.to(device)

    ckpt_path = cfg.get("checkpoint_path")
    if ckpt_path:
        load_checkpoint(encoder, ckpt_path, device)
    else:
        print("[main] WARNING: no checkpoint_path in config — using random weights")

    encoder.eval()
    frames_per_batch = encoder.clip_len
    print(f"[main] frames_per_batch: {frames_per_batch}")
    print(f"[main] Extracting embeddings for {len(dataloader.dataset)} videos ...")

    # --- Extract ---
    results = extract_embeddings(encoder, dataloader, device, frames_per_batch)

    # --- Save ---
    # Use embedding_save_path from config if set; otherwise derive from extract_h5_path stem.
    if cfg.get("embedding_save_path"):
        save_path = cfg["embedding_save_path"]
    else:
        extract_h5 = cfg.get("extract_h5_path", cfg.get("h5_path", ""))
        stem = Path(extract_h5).stem if extract_h5 else "embeddings"
        save_path = str(_PROJ_ROOT / "datasets" / "embeddings" / f"{stem}-embd.h5")
    save_embeddings_h5(results, save_path)
    print(f"[main] Done. Embeddings saved to {save_path}")

    # [v2] Optional: register embedding into configs_v2/registry/runs.yaml
    if register:
        from utils.registry_v2 import RegistryV2
        _run_ref     = cfg.get("checkpoint_ref", "")
        _dataset_ref = cfg.get("extract_dataset", "")
        _variant     = "standard"
        _reg = RegistryV2()
        _alias = register_alias or _reg.suggest_embedding_alias(_run_ref, _dataset_ref, _variant)
        _reg.register_embedding(
            alias       = _alias,
            run_ref     = _run_ref,
            dataset_ref = _dataset_ref,
            variant     = _variant,
            description = f"{_dataset_ref} embeddings (standard)",
        )
        print(f"[main] [v2] Embedding registered as '{_alias}' in configs_v2/registry/runs.yaml")


# ---------------------------------------------------------------------------
# __main__ entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract TCC embeddings from H5 dataset")
    parser.add_argument(
        "--extract_h5_path", type=str, default=None,
        help="Path to the H5 dataset to extract embeddings from. "
             "Overrides extract_dataset in configs_v2/extract.yaml.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        default=False,
        help="[v2] After extraction, register the embedding into configs_v2/registry/runs.yaml.",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        dest="register_alias",
        help="[v2] Registry alias for the embedding (auto-suggested if not set). Requires --register.",
    )
    args = parser.parse_args()

    # [v2] Delegate to main() — handles the full pipeline including optional registration
    main(
        extract_h5_path=args.extract_h5_path,
        register=args.register,
        register_alias=args.register_alias,
    )
