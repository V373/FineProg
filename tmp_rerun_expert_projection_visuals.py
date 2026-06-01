import datetime
from pathlib import Path

import h5py

from utils.config_v2 import ConfigV2
from fineprog.algos.eval_task.base_task import build_task
from fineprog.algos.eval_task.tcc_eval_tasks.task_expert_projection import _build_demo_name_map


cfg = ConfigV2().load_eval("expert_projection")
source_h5 = Path(
    "outputs/expert_projection/robomimic_can_ph-36vid_train-embd-mean_path/"
    "robomimic_can_mh-100vid_worse-embd/"
    "COMPOSITE_TCC_TEMPORAL_TRIPLET-robomimic_can_ph-36vid_train-resnet50_conv4c-only_bn-20260521-193510/"
    "expert_projection-20260525-200454.h5"
)
if not source_h5.exists():
    raise FileNotFoundError(source_h5)

with h5py.File(cfg["nonexpert_h5_path"], "r") as f:
    all_video_ids = sorted(f["videos"].keys())

viz_cfg = cfg.get("visualization_video_ids", None)
if viz_cfg == "all":
    viz_video_ids = all_video_ids
elif not viz_cfg:
    viz_video_ids = list(all_video_ids[: min(4, len(all_video_ids))])
else:
    viz_video_ids = [
        all_video_ids[i]
        for i in viz_cfg
        if isinstance(i, int) and 0 <= i < len(all_video_ids)
    ]

if cfg.get("nonexpert_raw_hdf5_path") and cfg.get("nonexpert_mask_key"):
    demo_name_map = _build_demo_name_map(
        cfg["nonexpert_raw_hdf5_path"],
        cfg["nonexpert_mask_key"],
    )
else:
    demo_name_map = {}

print("source_h5=", source_h5)
print("viz_video_count=", len(viz_video_ids))

video_raw_dir = cfg.get("nonexpert_video_raw_dir", "") or None
save_tsne = bool(cfg.get("save_tsne_visualization", False))

task = build_task("expert_projection")
task.configure(cfg)
shared_bundle = task._build_shared_tsne_bundle(
    output_h5_path=source_h5,
    viz_video_ids=viz_video_ids,
    demo_name_map=demo_name_map,
)
if shared_bundle is None:
    raise RuntimeError("Shared t-SNE bundle build failed")

print("shared_n_total=", shared_bundle["n_total"])
print("shared_perplexity=", shared_bundle["perplexity_used"])
print("shared_video_count=", len(shared_bundle["video_ids"]))

rerun_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-vizrerun-clamped"
output_root = source_h5.parent
video_index_map = {vid: idx for idx, vid in enumerate(viz_video_ids)}

for video_id in viz_video_ids:
    video_idx = video_index_map[video_id]
    vis_output_dir = output_root / f"{video_idx:03d}_{rerun_tag}"
    print(f"rendering video_id={video_id} -> {vis_output_dir}")
    task._save_visualizations(
        output_h5_path=source_h5,
        vis_output_dir=str(vis_output_dir),
        viz_video_ids=[video_id],
        demo_name_map=demo_name_map,
        video_raw_dir=video_raw_dir,
        save_tsne=save_tsne,
        shared_tsne_bundle=shared_bundle,
    )

print("rerun_done=", rerun_tag)