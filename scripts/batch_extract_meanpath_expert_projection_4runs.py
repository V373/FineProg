"""Batch orchestrator: extract embeddings, compute mean path, and run expert projection
for 4 target runs at their latest checkpoint.

Runs:
  1. COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-...-20260521-193510
  2. COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-...-20260521-193537
  3. COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-...-20260521-193550
  4. TCC-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-164424

For each run:
  Phase 1 – Extract embeddings (3 datasets, encoder loaded ONCE per run):
    robomimic_can_ph_36vid_train   → datasets/embeddings/{run}/robomimic_can_ph-36vid_train-embd.h5
    robomimic_can_ph_4vid_valid    → datasets/embeddings/{run}/robomimic_can_ph-4vid_valid-embd.h5
    robomimic_can_mh_100vid_worse  → datasets/embeddings/{run}/robomimic_can_mh-100vid_worse-embd.h5

  Phase 2 – Compute mean latent path (train dataset only):
    Input:  datasets/embeddings/{run}/robomimic_can_ph-36vid_train-embd.h5
    Output: datasets/embeddings/{run}/robomimic_can_ph-36vid_train-embd-mean_path.h5
    Plots:  outputs/mean_path/{run}/...

  Phase 3 – Expert projection (worse dataset projected onto train mean path):
    Expert:     robomimic_can_ph-36vid_train-embd-mean_path.h5
    Non-expert: robomimic_can_mh-100vid_worse-embd.h5
    Output dir: outputs/expert_projection/robomimic_can_ph-36vid_train-embd-mean_path/
                                          robomimic_can_mh-100vid_worse-embd/
    Products:   expert_projection-*.h5
                alignment_curve_*.png
                tsne_projection_*.png
                alignment_anim_*.mp4  (requires raw MP4 files in datasets/raw/robomimic_can_mh/)

No YAML files are modified.  No W&B logging.  Encoder loaded once per run.

Usage (from mytcc/ project root):
    conda run -n fineprog python scripts/batch_extract_meanpath_expert_projection_4runs.py --dry-run
    conda run -n fineprog python scripts/batch_extract_meanpath_expert_projection_4runs.py --skip-existing
    conda run -n fineprog python scripts/batch_extract_meanpath_expert_projection_4runs.py
    conda run -n fineprog python scripts/batch_extract_meanpath_expert_projection_4runs.py --summary-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).resolve().parent.parent
_PROJECTS_ROOT   = _PROJ_ROOT.parent  # …/projects/
for _p in (str(_PROJ_ROOT), str(_PROJECTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── scripts/ on sys.path (for compute_mean_embedding_path) ────────────────────
_SCRIPTS_DIR = _PROJ_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ── mytcc imports ─────────────────────────────────────────────────────────────
from fineprog.algos.eval_task.base_task import build_task                # noqa: E402
from extract_embeddings import load_checkpoint                         # noqa: E402
from models.encoder import TCCEncoder                                  # noqa: E402
from utils.config_v2 import ConfigV2                                   # noqa: E402
from utils.in_training_eval import _extract_embeddings_for_eval        # noqa: E402

# ── mean-path helpers ─────────────────────────────────────────────────────────
from compute_mean_embedding_path import (                              # noqa: E402
    compute_all_cumdist_progress,
    compute_basic_dispersion,
    compute_mean_path,
    load_all_embeddings,
    plot_cumdist_progress,
    plot_cumdist_progress_dual,
    resolve_paths as resolve_meanpath_paths,
    save_results as save_meanpath_results,
)

# ── Constants ─────────────────────────────────────────────────────────────────
_V2_EXTRACT_YAML = str(_PROJ_ROOT / "configs_v2" / "extract.yaml")

_TARGET_RUNS: list[str] = [
    "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-193510",
    "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-193537",
    "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-193550",
    "TCC-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-164424",
]

# 3 datasets to extract embeddings for (train first so mean-path input is ready)
_EXTRACT_DATASETS: list[str] = [
    "robomimic_can_ph_36vid_train",
    "robomimic_can_ph_4vid_valid",
    "robomimic_can_mh_100vid_worse",
]

_MEANPATH_DATASET_REF  = "robomimic_can_ph_36vid_train"
_NONEXPERT_DATASET_REF = "robomimic_can_mh_100vid_worse"


# ── Path helpers ──────────────────────────────────────────────────────────────

def find_latest_checkpoint(run_name: str) -> tuple[Path, int]:
    """Return (checkpoint_path, epoch) for the highest-epoch .pt file."""
    ckpt_dir = _PROJ_ROOT / "checkpoint" / run_name
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    pts = list(ckpt_dir.glob("encoder_epoch*.pt"))
    if not pts:
        raise FileNotFoundError(f"No encoder_epoch*.pt found in {ckpt_dir}")

    def _epoch(p: Path) -> int:
        try:
            return int(p.stem.replace("encoder_epoch", ""))
        except ValueError:
            return -1

    pts.sort(key=_epoch)
    latest = pts[-1]
    return latest, _epoch(latest)


def _embd_path(run_name: str, dataset_ref: str) -> Path:
    ds = ConfigV2().resolve_dataset(dataset_ref)
    return _PROJ_ROOT / "datasets" / "embeddings" / run_name / f"{ds['h5_stem']}-embd.h5"


def _meanpath_h5(run_name: str) -> Path:
    ds = ConfigV2().resolve_dataset(_MEANPATH_DATASET_REF)
    return _PROJ_ROOT / "datasets" / "embeddings" / run_name / f"{ds['h5_stem']}-embd-mean_path.h5"


# ── Phase 1: Extraction ───────────────────────────────────────────────────────

def extract_all_embeddings(
    run_name: str,
    ckpt_path: Path,
    device: torch.device,
    skip_existing: bool,
) -> dict[str, Path]:
    """Extract embeddings for all 3 datasets for one run.

    Encoder is loaded once and reused across all 3 datasets.
    Returns {dataset_ref: embd_h5_path}.
    """
    embd_paths: dict[str, Path] = {
        ds_ref: _embd_path(run_name, ds_ref) for ds_ref in _EXTRACT_DATASETS
    }

    to_extract = [
        ds_ref for ds_ref in _EXTRACT_DATASETS
        if not (skip_existing and embd_paths[ds_ref].exists())
    ]

    if not to_extract:
        print(f"[batch] all 3 embds already exist for this run — skipping extraction")
        return embd_paths

    print(f"\n[batch] Phase 1 – Extracting embeddings ({len(to_extract)} datasets) …")
    encoder = TCCEncoder(config_path=_V2_EXTRACT_YAML)
    encoder.to(device)
    load_checkpoint(encoder, str(ckpt_path), device)
    encoder.eval()

    try:
        for ds_ref in to_extract:
            out_path = embd_paths[ds_ref]
            if skip_existing and out_path.exists():
                print(f"[batch]   skip (exists): {out_path.name}")
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _extract_embeddings_for_eval(
                encoder=encoder,
                eval_dataset_ref=ds_ref,
                save_path=str(out_path),
                device=device,
                num_workers=0,
            )
            print(f"[batch]   ✓ saved: {out_path}")
    finally:
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return embd_paths


# ── Phase 2: Mean path ────────────────────────────────────────────────────────

def compute_and_save_mean_path(
    run_name: str,
    train_embd_h5: Path,
    skip_existing: bool,
) -> Path:
    """Compute mean latent path from train embeddings. Returns path to mean_path H5."""
    out_h5 = _meanpath_h5(run_name)

    if skip_existing and out_h5.exists():
        print(f"[batch] Phase 2 – skip mean_path (exists): {out_h5.name}")
        return out_h5

    print(f"\n[batch] Phase 2 – Computing mean path from {train_embd_h5.name} …")
    # resolve_meanpath_paths derives: output path (same dir as input) + plot path
    _, resolved_out, resolved_plot = resolve_meanpath_paths(
        str(train_embd_h5), str(out_h5), None
    )

    records = load_all_embeddings(str(train_embd_h5))
    print(f"[batch]   loaded {len(records)} videos for mean path")

    mean_emb, resampled, metadata = compute_mean_path(records)
    dispersion = compute_basic_dispersion(resampled, mean_emb)
    cumdist_data = compute_all_cumdist_progress(resampled, mean_emb)

    save_meanpath_results(
        resolved_out, mean_emb, dispersion, cumdist_data, metadata, resolved_plot
    )
    plot_cumdist_progress(cumdist_data, metadata["K"], resolved_plot)
    plot_cumdist_progress_dual(cumdist_data, metadata["K"], resolved_plot)

    print(f"[batch]   ✓ saved mean_path H5: {resolved_out}")
    return Path(resolved_out)


# ── Phase 3: Expert projection ────────────────────────────────────────────────

def run_expert_projection(
    expert_h5: Path,
    nonexpert_h5: Path,
) -> dict:
    """Project non-expert embeddings onto expert mean path. Returns result dict."""
    cfg_v2 = ConfigV2()
    resolved = cfg_v2.load_eval(
        "expert_projection",
        overrides={
            # Clear ref keys so the resolver does not try to look them up in registry
            "expert_embedding_ref":    None,
            "nonexpert_embedding_ref": None,
            # Inject absolute H5 paths directly
            "expert_h5_path":          str(expert_h5),
            "nonexpert_h5_path":       str(nonexpert_h5),
            # dataset ref still needed so resolver can fill raw_hdf5_path / mask_key / raw_dir
            "nonexpert_dataset_ref":   _NONEXPERT_DATASET_REF,
            # Task-specific settings — visualization_video_ids read from YAML
            "save_alpha":              True,
            "save_entropy":            True,
            "save_visualization":      True,
            "save_tsne_visualization": True,
            # nonexpert_video_raw_dir stays ~ → auto-derived from dataset raw_dir_path
        },
    )

    task = build_task("expert_projection")
    task.configure(resolved)
    return task.evaluate(None)


# ── Per-run orchestration ─────────────────────────────────────────────────────

def process_run(
    run_name: str,
    device: torch.device,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    """Full pipeline for one run: 3 embds → mean path → expert projection."""
    print(f"\n{'=' * 72}")
    print(f"[batch] RUN: {run_name}")
    print(f"{'=' * 72}")

    ckpt_path, epoch = find_latest_checkpoint(run_name)
    print(f"[batch] Latest checkpoint: epoch={epoch}  ({ckpt_path.name})")

    # Pre-compute expected paths (used for both dry-run reporting and execution)
    embd_paths_exp = {ds: _embd_path(run_name, ds) for ds in _EXTRACT_DATASETS}
    meanpath_exp = _meanpath_h5(run_name)
    ne_stem = embd_paths_exp[_NONEXPERT_DATASET_REF].stem
    proj_dir = (
        _PROJ_ROOT / "outputs" / "expert_projection"
        / meanpath_exp.stem / ne_stem
    )

    if dry_run:
        print(f"\n[dry-run] Phase 1 – Extraction  (ckpt epoch {epoch}):")
        for ds_ref, path in embd_paths_exp.items():
            status = "EXISTS" if path.exists() else "will create"
            print(f"  [{status:<11s}]  {path}")
        print(f"\n[dry-run] Phase 2 – Mean path:")
        status = "EXISTS" if meanpath_exp.exists() else "will create"
        print(f"  [{status:<11s}]  {meanpath_exp}")
        print(f"\n[dry-run] Phase 3 – Expert projection:")
        print(f"  expert      : {meanpath_exp.name}")
        print(f"  non-expert  : {embd_paths_exp[_NONEXPERT_DATASET_REF].name}")
        print(f"  video_ids   : [51]")
        print(f"  output dir  : {proj_dir}/")
        return

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    embd_paths = extract_all_embeddings(run_name, ckpt_path, device, skip_existing)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    mean_h5 = compute_and_save_mean_path(
        run_name, embd_paths[_MEANPATH_DATASET_REF], skip_existing
    )

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    print(f"\n[batch] Phase 3 – Expert projection …")
    result = run_expert_projection(
        expert_h5=mean_h5,
        nonexpert_h5=embd_paths[_NONEXPERT_DATASET_REF],
    )
    print(f"[batch]   ✓ {result['metric_name']} = {result['metric_value']:.6f}")
    print(f"[batch]   ✓ output H5: {result.get('output_h5_path')}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    """Report existence of all expected output artifacts across 4 runs."""
    print(f"\n{'=' * 72}")
    print("  SUMMARY – expected artifacts check")
    print(f"{'=' * 72}")

    all_ok = True
    for run_name in _TARGET_RUNS:
        short = run_name[-20:]  # last 20 chars as identifier
        print(f"\n  Run: …{short}")

        # 3 standard embds
        for ds_ref in _EXTRACT_DATASETS:
            p = _embd_path(run_name, ds_ref)
            ok = p.exists()
            if not ok:
                all_ok = False
            tag = "✓" if ok else "✗ MISSING"
            print(f"    [{tag}] embd  {p.name}")

        # mean_path H5
        mp = _meanpath_h5(run_name)
        ok = mp.exists()
        if not ok:
            all_ok = False
        tag = "✓" if ok else "✗ MISSING"
        print(f"    [{tag}] mean  {mp.name}")

        # expert_projection output dir (files may be in a per-run subdir)
        ne_stem = _embd_path(run_name, _NONEXPERT_DATASET_REF).stem
        proj_dir = _PROJ_ROOT / "outputs" / "expert_projection" / mp.stem / ne_stem
        run_subdir = proj_dir / run_name
        # prefer the per-run subdir if it exists, fall back to the flat dir
        search_dir = run_subdir if run_subdir.is_dir() else proj_dir
        has_h5  = search_dir.is_dir() and any(search_dir.glob("expert_projection-*.h5"))
        has_png = search_dir.is_dir() and any(search_dir.rglob("*.png"))
        has_mp4 = search_dir.is_dir() and any(search_dir.rglob("*.mp4"))
        ok = has_h5 and has_png and has_mp4
        if not ok:
            all_ok = False
        detail = (
            f"h5={'✓' if has_h5 else '✗'} "
            f"png={'✓' if has_png else '✗'} "
            f"mp4={'✓' if has_mp4 else '✗'}"
        )
        print(f"    [{'✓' if ok else '✗ INCOMPLETE'}] proj  {search_dir.relative_to(_PROJ_ROOT)}  ({detail})")

    print()
    if all_ok:
        print("  Overall: ALL ARTIFACTS PRESENT ✓")
    else:
        print("  Overall: SOME ARTIFACTS MISSING ✗")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch extract/mean-path/expert-projection for 4 robomimic_can_ph 36vid runs."
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned actions without executing them.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip outputs that already exist on disk.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite all existing outputs (overrides --skip-existing).",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Only print the artifact-existence summary; do not run anything.",
    )
    args = parser.parse_args()

    if args.summary_only:
        print_summary()
        return

    skip_existing = args.skip_existing and not args.force

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[batch] device        : {device}")
    print(f"[batch] mode          : {'DRY-RUN' if args.dry_run else 'EXECUTE'}")
    print(f"[batch] skip_existing : {skip_existing}")
    print(f"[batch] runs ({len(_TARGET_RUNS)}):")
    for r in _TARGET_RUNS:
        print(f"  {r}")

    for run_name in _TARGET_RUNS:
        process_run(run_name, device, skip_existing, dry_run=args.dry_run)

    if not args.dry_run:
        print_summary()

    print("\n[batch] Done.")


if __name__ == "__main__":
    main()
