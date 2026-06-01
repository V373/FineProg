"""Smoke test for in-training eval pipeline (dual-task: latent_distance_heatmap + kendalls_tau).

Run with:
    cd /home/user/zhangzk/projects
    conda run -n fineprog python fineprog/_smoke_test_in_training_eval.py
"""
import sys
import os
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # projects/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # fineprog/

import torch
from utils.config_v2 import ConfigV2
from models.encoder import TCCEncoder
from utils.in_training_eval import run_in_training_eval, _resolve_task_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[smoke] device={device}")

# ── Helper checks ─────────────────────────────────────────────────────────────
# Verify _resolve_task_list handles both config shapes.
assert _resolve_task_list({"selected_eval_tasks_in_training": ["latent_distance_heatmap", "kendalls_tau"]}) \
    == ["latent_distance_heatmap", "kendalls_tau"], "_resolve_task_list: new list field failed"
assert _resolve_task_list({"selected_eval_task_in_training": "latent_distance_heatmap"}) \
    == ["latent_distance_heatmap"], "_resolve_task_list: old single field failed"
assert _resolve_task_list({}) \
    == ["latent_distance_heatmap"], "_resolve_task_list: default failed"
print("[smoke] _resolve_task_list: OK")

# 1. Build encoder from the V2 train config (pass YAML path, not dict)
_V2_TRAIN_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs_v2", "train.yaml")
cfg_v2 = ConfigV2()
train_cfg = cfg_v2.load_train()
encoder = TCCEncoder(
    config_path=_V2_TRAIN_YAML,
    train_config_path=_V2_TRAIN_YAML,
).to(device)
encoder.train()
encoder.configure_trainability()
print(f"[smoke] Encoder built OK: {type(encoder).__name__}")

# 2. Build eval_cfg: dual-task, no wandb
ite_cfg = dict(train_cfg["in_training_eval"])
ite_cfg["enabled"] = True
ite_cfg["eval_freq_in_training"] = 1
ite_cfg["log_images_to_wandb"] = False  # no wandb run in this test
# Ensure dual-task list is set explicitly regardless of what train.yaml says
ite_cfg["selected_eval_tasks_in_training"] = ["latent_distance_heatmap", "kendalls_tau"]

tmp_ckpt_dir = tempfile.mkdtemp(prefix="smoke_ckpt_")
print(f"[smoke] tmp_ckpt_dir={tmp_ckpt_dir}")

import time
t0 = time.perf_counter()

try:
    # 3. Run the eval hook (epoch=1, checkpoint_count=1 → should execute)
    run_in_training_eval(
        encoder=encoder,
        eval_cfg=ite_cfg,
        train_cfg=train_cfg,
        run_checkpoint_dir=tmp_ckpt_dir,
        epoch=1,
        checkpoint_count=1,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    print(f"[smoke] Wall-clock: {elapsed:.1f}s for dual-task eval")

    eval_out = os.path.join(tmp_ckpt_dir, "eval_epoch000002")

    # 4. Verify temp H5 is deleted
    tmp_h5 = os.path.join(eval_out, "tmp_eval_embd.h5")
    assert not os.path.isfile(tmp_h5), f"Temp H5 was NOT cleaned up: {tmp_h5}"
    print("[smoke] Temp H5 cleanup: OK")

    # 5. Verify encoder is back in train mode
    assert encoder.training, "Encoder was NOT restored to training mode"
    print("[smoke] Encoder train mode restored: OK")

    # 6. Verify per-task subdirectories exist
    for task in ("latent_distance_heatmap", "kendalls_tau"):
        task_dir = os.path.join(eval_out, task)
        assert os.path.isdir(task_dir), f"Task output dir missing: {task_dir}"
        # Recurse into sub-dirs (latent_distance_heatmap writes PNGs into a
        # timestamp sub-directory; kendalls_tau writes directly into task_dir).
        pngs = []
        for dirpath, _, filenames in os.walk(task_dir):
            pngs.extend(f for f in filenames if f.endswith(".png"))
        assert pngs, f"No PNG heatmap found (recursively) in {task_dir}"
        print(f"[smoke] {task}: dir OK, PNG={pngs}")

    print("\n[smoke] *** Smoke test PASSED ***")

finally:
    shutil.rmtree(tmp_ckpt_dir, ignore_errors=True)
