"""Evaluate latent-distance-heatmap for 4 target runs at their latest checkpoint.

Runs:
  1. COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-...-20260521-193510
  2. COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-...-20260521-193537
  3. COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-...-20260521-193550
  4. TCC-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-164424

For each run:
  - Loads epoch-40000 encoder checkpoint
  - Extracts embeddings from robomimic_can_ph_4vid_valid (36-vid model's paired split)
  - Runs latent_distance_heatmap evaluation using YAML settings from
    configs_v2/eval/latent_distance_heatmap.yaml (unmodified)
  - Saves output PNGs to outputs/latent_distance_heatmap/<run_name>/epoch_040000/

No YAML files are modified.  No W&B logging.

Usage (from mytcc/ project root):
    conda run -n fineprog python scripts/eval_latent_distance_4runs.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).resolve().parent.parent
_PROJECTS_ROOT   = _PROJ_ROOT.parent  # …/projects/
for _p in (str(_PROJ_ROOT), str(_PROJECTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── mytcc imports ─────────────────────────────────────────────────────────────
from evaluate import run_eval_task                            # noqa: E402
from extract_embeddings import load_checkpoint                # noqa: E402
from models.encoder import TCCEncoder                         # noqa: E402
from utils.config_v2 import ConfigV2                          # noqa: E402
from utils.in_training_eval import _extract_embeddings_for_eval  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
_V2_EXTRACT_YAML = str(_PROJ_ROOT / "configs_v2" / "extract.yaml")

_EPOCH = 40000

_EVAL_DATASET_REF = "robomimic_can_ph_4vid_valid"  # paired validation for 36-vid models

_TARGET_RUNS: list[str] = [
    "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-193510",
    "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-193537",
    "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-193550",
    "TCC-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-164424",
]


def eval_one_run(run_name: str, device: torch.device) -> None:
    """Extract embeddings and run latent_distance_heatmap for *run_name* at epoch 40000."""
    print(f"\n{'=' * 72}")
    print(f"[eval_4runs] run: {run_name}")
    print(f"[eval_4runs] epoch: {_EPOCH}")
    print(f"{'=' * 72}")

    ckpt_path = _PROJ_ROOT / "checkpoint" / run_name / f"encoder_epoch{_EPOCH:06d}.pt"
    if not ckpt_path.is_file():
        print(f"[eval_4runs] ERROR: checkpoint not found: {ckpt_path} — skipping")
        return

    # Output directory for this run's eval artifacts
    output_dir = str(
        _PROJ_ROOT / "outputs" / "latent_distance_heatmap" / run_name / f"epoch_{_EPOCH:06d}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Persistent embedding H5 saved alongside the output (not a temp file)
    embd_h5_path = os.path.join(output_dir, "embeddings.h5")

    # ── 1. Build encoder and load checkpoint ──────────────────────────────────
    encoder = TCCEncoder(config_path=_V2_EXTRACT_YAML)
    encoder.to(device)
    load_checkpoint(encoder, str(ckpt_path), device)
    encoder.eval()

    # ── 2. Extract embeddings ─────────────────────────────────────────────────
    _extract_embeddings_for_eval(
        encoder=encoder,
        eval_dataset_ref=_EVAL_DATASET_REF,
        save_path=embd_h5_path,
        device=device,
        num_workers=0,
    )

    # ── 3. Run latent_distance_heatmap evaluation ─────────────────────────────
    # Inject embedding_h5_path and output_dir as overrides so no YAML is modified.
    cfg_v2 = ConfigV2()
    resolved = cfg_v2.load_eval(
        "latent_distance_heatmap",
        overrides={
            "embedding_h5_path": embd_h5_path,
            "output_dir":        output_dir,
        },
    )
    result = run_eval_task("latent_distance_heatmap", resolved)

    print(
        f"\n[eval_4runs] DONE  {result['metric_name']} = {result['metric_value']:.6f}"
    )
    print(f"[eval_4runs] output_dir: {output_dir}")

    # Print any saved file paths from the result
    for key in ("output_heatmap_paths", "output_curve_paths",
                "output_heatmap_path", "output_curve_path"):
        val = result.get(key)
        if val:
            paths = val if isinstance(val, list) else [val]
            for p in paths:
                print(f"[eval_4runs]   saved: {p}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_4runs] device: {device}")
    print(f"[eval_4runs] eval_dataset: {_EVAL_DATASET_REF}")
    print(f"[eval_4runs] epoch: {_EPOCH}")
    print(f"[eval_4runs] runs ({len(_TARGET_RUNS)}):")
    for r in _TARGET_RUNS:
        print(f"  {r}")

    for run_name in _TARGET_RUNS:
        eval_one_run(run_name, device)

    print("\n[eval_4runs] All done.")


if __name__ == "__main__":
    main()
