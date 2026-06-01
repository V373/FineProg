"""One-shot script: VOC eval on 4 checkpoints, val 4-vid, raw L2 (normalize_embeddings=True)."""
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from utils.config_v2 import ConfigV2
from evaluate import run_eval_task

_EMBD_BASE = os.path.join(os.path.dirname(__file__), "..", "datasets", "embeddings")

checkpoints = [
    (
        "TCC-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-164424",
        os.path.join(_EMBD_BASE,
            "TCC-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-164424",
            "robomimic_can_ph-4vid_valid-embd.h5"),
    ),
    (
        "COMPOSITE_TCC_TEMPORAL_TRIPLET-...193510",
        os.path.join(_EMBD_BASE,
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-193510",
            "robomimic_can_ph-4vid_valid-embd.h5"),
    ),
    (
        "COMPOSITE_TCC_TEMPORAL_TRIPLET-...193537",
        os.path.join(_EMBD_BASE,
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-193537",
            "robomimic_can_ph-4vid_valid-embd.h5"),
    ),
    (
        "COMPOSITE_TCC_TEMPORAL_TRIPLET-...193550",
        os.path.join(_EMBD_BASE,
            "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-193550",
            "robomimic_can_ph-4vid_valid-embd.h5"),
    ),
]

cfg_v2 = ConfigV2()
results = []

for label, h5_rel in checkpoints:
    h5_abs = os.path.abspath(h5_rel)
    print(f"\n{'='*70}")
    print(f"Run:  {label}")
    print(f"H5:   {h5_abs}")
    resolved = cfg_v2.load_eval(
        "latent_distance_heatmap",
        overrides={
            "embedding_h5_path": h5_abs,
            "selected_video_index": "all",
            "normalize_embeddings": True,
            "convert_to_similarity": False,
            "output_dir": None,
        },
    )
    r = run_eval_task("latent_distance_heatmap", resolved)
    voc      = r.get("metric_value")
    n_valid  = r.get("voc_n_valid_videos", "?")
    skipped  = r.get("voc_skipped_video_ids", [])
    print(f"  --> VOC (Spearman): {voc:.6f}  [{n_valid}/4 valid videos]")
    if skipped:
        print(f"  Skipped videos: {skipped}")
    results.append((label, voc, n_valid))

print(f"\n{'='*70}")
print("SUMMARY  —  VOC (Spearman) | val 4-vid | L2-normalized embeddings | raw L2")
print(f"{'='*70}")
for label, voc, n_valid in results:
    print(f"  {label:<55s}  VOC={voc:.4f}  ({n_valid}/4 valid)")
print()
