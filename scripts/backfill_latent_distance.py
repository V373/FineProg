"""Backfill latent-distance-heatmap images to W&B for 3 source training runs.

Each source run was trained with `plot_mode: anchor_distance_curves`, which
exposed a logging bug: the upload layer only read `output_heatmap_path(s)` keys
and silently dropped the curve images.  This script regenerates all 8 checkpoints
(5k–40k) per run, posting images to *new* dedicated W&B runs (one per source run).

Usage (from the mytcc/ project root):
    # Smoke test – first run, first checkpoint only:
    conda run -n fineprog python scripts/backfill_latent_distance.py --smoke-test

    # Full backfill of all 3 runs × 8 checkpoints:
    conda run -n fineprog python scripts/backfill_latent_distance.py

    # Single source run only (0-based index):
    conda run -n fineprog python scripts/backfill_latent_distance.py --run-index 0

Options:
    --smoke-test   Only process run 0, epoch 5000 (quick sanity check)
    --run-index N  Only process the Nth source run (0, 1, or 2)
    --dry-run      Skip wandb.init; print what would be logged without uploading
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).resolve().parent.parent
_PROJECTS_ROOT   = _PROJ_ROOT.parent  # …/projects/
for _p in (str(_PROJ_ROOT), str(_PROJECTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── mytcc imports ─────────────────────────────────────────────────────────────
from fineprog.evaluate_encoder import run_eval_task                        # noqa: E402
from extract_embeddings import load_checkpoint            # noqa: E402
from models.encoder import TCCEncoder                     # noqa: E402
from utils.config_v2 import ConfigV2                      # noqa: E402
from utils.in_training_eval import (                      # noqa: E402
    _collect_image_payload,
    _extract_embeddings_for_eval,
)

# ── Source run definitions ────────────────────────────────────────────────────
# Each entry describes one training run whose latent-distance images were
# never uploaded due to the logging bug.
#
# Fields:
#   run_name      – checkpoint directory name under checkpoint/
#   wandb_run_id  – W&B run id of the original training run (for the tag)
#   eval_dataset  – evaluation dataset ref (auto-paired validation split)
_SOURCE_RUNS: list[dict] = [
    {
        "run_name": (
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train"
            "-resnet50_conv4c-only_bn-20260521-193510"
        ),
        "wandb_run_id": "7gs4inaf",
        "eval_dataset": "robomimic_can_ph_4vid_valid",
    },
    {
        "run_name": (
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train"
            "-resnet50_conv4c-only_bn-20260521-193537"
        ),
        "wandb_run_id": "3xtzxwlx",
        "eval_dataset": "robomimic_can_ph_4vid_valid",
    },
    {
        "run_name": (
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train"
            "-resnet50_conv4c-only_bn-20260521-193550"
        ),
        "wandb_run_id": "4c4voni4",
        "eval_dataset": "robomimic_can_ph_4vid_valid",
    },
]

# Checkpoints to process (epoch numbers = filenames after "encoder_epoch").
_CHECKPOINT_EPOCHS: list[int] = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]

# V2 extract YAML path (used to build TCCEncoder with correct shape params).
_V2_EXTRACT_YAML = str(_PROJ_ROOT / "configs_v2" / "extract.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_timestamp(run_name: str) -> str:
    """Extract YYYYMMDD-HHMMSS from a checkpoint run_name.

    Checkpoint dirs end with ...-<YYYYMMDD>-<HHMMSS>.  Split from the right
    and rejoin the last two tokens.

    Example:
        'COMPOSITE_TCC_TEMPORAL_TRIPLET-...-only_bn-20260521-193510'
        → '20260521-193510'
    """
    parts = run_name.rsplit("-", 2)
    if len(parts) == 3:
        return f"{parts[1]}-{parts[2]}"
    return run_name  # fallback: use full name


# ─────────────────────────────────────────────────────────────────────────────
# Core per-run backfill logic
# ─────────────────────────────────────────────────────────────────────────────

def backfill_one_run(
    run_info:     dict,
    epochs:       list[int],
    device:       torch.device,
    dry_run:      bool = False,
) -> None:
    """Backfill latent-distance images for all *epochs* of one source run.

    Creates one new W&B run (or dry-runs without wandb).  Each checkpoint's
    images and scalar metric are logged at step = epoch.

    Args:
        run_info: dict from _SOURCE_RUNS
        epochs:   list of epoch numbers to process
        device:   torch device
        dry_run:  if True, skip wandb.init and all wandb.log calls
    """
    run_name:    str = run_info["run_name"]
    source_wid:  str = run_info["wandb_run_id"]
    eval_ds_ref: str = run_info["eval_dataset"]
    timestamp:   str = _extract_timestamp(run_name)

    ckpt_dir = _PROJ_ROOT / "checkpoint" / run_name
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Output root for regenerated artifacts (separate from original training outputs)
    backfill_root = _PROJ_ROOT / "outputs" / "latent_distance_heatmap_backfill" / run_name
    backfill_root.mkdir(parents=True, exist_ok=True)

    # ── W&B run ──────────────────────────────────────────────────────────────
    backfill_run_name = f"backfill-latent-dist-{timestamp}"
    if not dry_run:
        import wandb
        wb_run = wandb.init(
            project="mytcc",
            name=backfill_run_name,
            tags=["backfill", "latent_distance_heatmap", source_wid],
            config={
                "source_run_id":  source_wid,
                "source_run_name": run_name,
                "eval_dataset":   eval_ds_ref,
                "backfill_epochs": epochs,
                "script":         "scripts/backfill_latent_distance.py",
            },
        )
        print(f"[backfill] W&B run: {wb_run.url}")
    else:
        print(f"[backfill] DRY RUN — would create wandb run: {backfill_run_name}")

    # ── Build encoder (shape params from V2 extract YAML) ────────────────────
    encoder = TCCEncoder(config_path=_V2_EXTRACT_YAML)
    encoder.to(device)

    try:
        for epoch in epochs:
            print(f"\n[backfill] ── epoch {epoch:06d} / run {source_wid} ──")

            # 1. Load checkpoint weights
            ckpt_path = ckpt_dir / f"encoder_epoch{epoch:06d}.pt"
            if not ckpt_path.is_file():
                print(f"[backfill] WARNING: checkpoint not found, skipping: {ckpt_path}")
                continue
            load_checkpoint(encoder, str(ckpt_path), device)
            encoder.eval()

            # 2. Extract embeddings to a temp H5
            with tempfile.NamedTemporaryFile(
                suffix=".h5",
                prefix=f"backfill_{source_wid}_ep{epoch:06d}_",
                dir=str(backfill_root),
                delete=False,
            ) as tmp_f:
                tmp_h5 = tmp_f.name

            try:
                _extract_embeddings_for_eval(
                    encoder=encoder,
                    eval_dataset_ref=eval_ds_ref,
                    save_path=tmp_h5,
                    device=device,
                    num_workers=0,
                )

                # 3. Run latent_distance_heatmap with current YAML settings
                epoch_output_dir = str(
                    backfill_root / f"epoch_{epoch:06d}" / "latent_distance_heatmap"
                )
                overrides = {
                    "embedding_h5_path": tmp_h5,
                    "output_dir":        epoch_output_dir,
                }
                cfg_v2 = ConfigV2()
                resolved = cfg_v2.load_eval("latent_distance_heatmap", overrides=overrides)
                result = run_eval_task("latent_distance_heatmap", resolved)

            finally:
                # Always clean up the temp H5 even if eval fails
                if os.path.isfile(tmp_h5):
                    os.remove(tmp_h5)
                    print(f"[backfill] Removed temp H5: {tmp_h5}")

            # 4. Build log payload: scalar metric + images (both families)
            metric_key = f"eval/train/latent_distance_heatmap/{result['metric_name']}"
            log_payload: dict = {metric_key: result["metric_value"]}

            image_payload = _collect_image_payload("latent_distance_heatmap", result)
            log_payload.update(image_payload)

            n_imgs = len(image_payload)
            print(
                f"[backfill] epoch={epoch:6d}  "
                f"{metric_key}={result['metric_value']:.6f}  "
                f"images={n_imgs}"
            )
            # Print each uploaded key so missing videos are immediately visible.
            for k in sorted(image_payload):
                print(f"[backfill]   ✓ {k.rsplit('/', 1)[-1]}")

            if not dry_run:
                import wandb
                wandb.log(log_payload, step=epoch)
            else:
                img_paths = [
                    str(v) for v in image_payload.values()
                ] if image_payload else ["(none)"]
                print(f"[backfill] DRY RUN — would log: step={epoch}, "
                      f"metric={result['metric_value']:.6f}, "
                      f"images={img_paths}")

    finally:
        if not dry_run:
            import wandb
            wandb.finish()
            print(f"[backfill] W&B run finished for source run {source_wid}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill latent-distance W&B images for 3 source training runs."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Only run 0, epoch 5000 (quick sanity check).",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        metavar="N",
        help="Only process the Nth source run (0-based). Default: process all 3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be logged without uploading to W&B.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[backfill] Device: {device}")

    # Select source runs
    if args.run_index is not None:
        if not (0 <= args.run_index < len(_SOURCE_RUNS)):
            parser.error(f"--run-index must be 0..{len(_SOURCE_RUNS)-1}")
        source_runs = [_SOURCE_RUNS[args.run_index]]
    else:
        source_runs = _SOURCE_RUNS

    # Select epochs
    if args.smoke_test:
        epochs = [5000]
        if args.run_index is None:
            source_runs = [_SOURCE_RUNS[0]]
        print("[backfill] Smoke-test mode: run 0, epoch 5000 only.")
    else:
        epochs = _CHECKPOINT_EPOCHS

    # Run
    for run_info in source_runs:
        print(f"\n[backfill] ══════════ Source run: {run_info['wandb_run_id']} ══════════")
        backfill_one_run(
            run_info=run_info,
            epochs=epochs,
            device=device,
            dry_run=args.dry_run,
        )

    print("\n[backfill] All done.")


if __name__ == "__main__":
    main()
