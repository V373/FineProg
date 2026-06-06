"""In-training evaluation hook for mytcc.

Runs all selected eval tasks after every ``eval_freq_in_training`` checkpoint
saves, using the live encoder state.  Embeddings are extracted exactly once per
checkpoint and shared by all tasks.

Pipeline for each eval call:
  1. Switch encoder to eval mode.
  2. Resolve the ordered task list from eval config.
  3. Resolve the eval dataset and extract embeddings into a temporary H5.
  4. For each task in the list:
       a. Merge runtime overrides with task YAML via ConfigV2.load_eval().
       b. Run the task (via evaluate.run_eval_task) in its own output subdir.
       c. Log scalar metric + optional heatmap images to the active wandb run.
          Task-level failures are isolated — one task failing does not abort others.
  5. Delete the temporary H5 (always — in the finally block).
  6. Restore encoder training mode.

Supported tasks: latent_distance_heatmap, kendalls_tau
Excluded:        classification, expert_projection
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

# Project root (mytcc/)
_PROJ_ROOT = Path(__file__).resolve().parent.parent
# V2 train config path — used to carry clip_len / context_size / context_stride
# into the extraction DataLoader so it matches the encoder's expected input shape.
_V2_TRAIN_YAML = str(_PROJ_ROOT / "configs_v2" / "train.yaml")

# Tasks fully implemented for in-training use.
_IMPLEMENTED_TASKS = {"latent_distance_heatmap", "kendalls_tau"}


# ---------------------------------------------------------------------------
# Helper: normalize task selection to an ordered list
# ---------------------------------------------------------------------------

def _resolve_task_list(eval_cfg: dict) -> list:
    """Return the ordered list of eval tasks from the in_training_eval config block.

    Accepts two config shapes (new field takes precedence):

    - New:  ``selected_eval_tasks_in_training: [latent_distance_heatmap, kendalls_tau]``
    - Old:  ``selected_eval_task_in_training: latent_distance_heatmap``  (single string)

    Falls back to ``["latent_distance_heatmap"]`` when neither field is present.
    """
    new_field = eval_cfg.get("selected_eval_tasks_in_training")
    if new_field is not None:
        if isinstance(new_field, list):
            return [str(t) for t in new_field]
        return [str(new_field)]
    old_field = eval_cfg.get("selected_eval_task_in_training")
    if old_field is not None:
        return [str(old_field)]
    return ["latent_distance_heatmap"]


# ---------------------------------------------------------------------------
# Public entry point — called from train.py after each checkpoint save
# ---------------------------------------------------------------------------

def run_in_training_eval(
    encoder,
    eval_cfg: dict,
    train_cfg: dict,
    run_checkpoint_dir: str,
    epoch: int,
    checkpoint_count: int,
    device: "torch.device",
) -> None:
    """Run one in-training evaluation pass and log results to wandb.

    Temporarily switches encoder to eval mode, extracts embeddings from the
    configured dataset, executes the eval task, logs scalar + image results to
    the active wandb run, removes the temporary H5, and restores encoder state.

    Args:
        encoder:            Live TCCEncoder instance (expected in train mode).
        eval_cfg:           ``in_training_eval`` sub-dict from resolved train config.
        train_cfg:          Full resolved train config dict (``_TRAIN_V2``).
        run_checkpoint_dir: Per-run checkpoint directory.
        epoch:              Current epoch index (0-based).
        checkpoint_count:   How many checkpoints have been saved so far (1-indexed).
        device:             Torch device.
    """
    # Frequency gate: skip unless this checkpoint count is a multiple of eval_freq.
    freq = int(eval_cfg.get("eval_freq_in_training", 1))
    if freq < 1 or (checkpoint_count % freq) != 0:
        return

    # Resolve task list and filter to implemented tasks.
    requested_tasks = _resolve_task_list(eval_cfg)
    valid_tasks = [t for t in requested_tasks if t in _IMPLEMENTED_TASKS]
    for t in requested_tasks:
        if t not in _IMPLEMENTED_TASKS:
            print(
                f"[in_training_eval] Task '{t}' is not supported for in-training eval; skipping. "
                f"Implemented: {sorted(_IMPLEMENTED_TASKS)}"
            )
    if not valid_tasks:
        return

    print(
        f"\n[in_training_eval] ── Eval at epoch {epoch + 1} "
        f"(checkpoint #{checkpoint_count}, tasks={valid_tasks}) ──"
    )

    # Decide which dataset to extract from.
    # Resolution order:
    #   1. Explicit eval_dataset_ref in the in_training_eval config block (full override)
    #   2. Registry-curated validation partner of the current train_dataset (auto)
    #   3. train_dataset itself as the final fallback
    explicit_ref = eval_cfg.get("eval_dataset_ref")
    if explicit_ref:
        eval_dataset_ref = explicit_ref
        print(f"[in_training_eval] eval dataset: {eval_dataset_ref} (explicit config)")
    else:
        train_ds_ref = train_cfg.get("train_dataset", "")
        from utils.config_v2 import ConfigV2  # lazy import (already loaded by training)
        paired_valid = ConfigV2().resolve_validation_dataset(train_ds_ref) if train_ds_ref else None
        if paired_valid:
            eval_dataset_ref = paired_valid
            print(f"[in_training_eval] eval dataset: {paired_valid} "
                  f"(auto-paired validation for '{train_ds_ref}')")
        elif train_ds_ref:
            eval_dataset_ref = train_ds_ref
            print(f"[in_training_eval] eval dataset: {train_ds_ref} "
                  f"(no validation pair registered; using train dataset)")
        else:
            print("[in_training_eval] ERROR: cannot determine eval dataset ref; skipping.")
            return
    if not eval_dataset_ref:
        print("[in_training_eval] ERROR: cannot determine eval dataset ref; skipping.")
        return

    # Build per-eval output directory under run_checkpoint_dir.
    eval_output_dir = os.path.join(run_checkpoint_dir, f"eval_epoch{epoch + 1:06d}")
    os.makedirs(eval_output_dir, exist_ok=True)
    tmp_embd_h5 = os.path.join(eval_output_dir, "tmp_eval_embd.h5")

    # Switch encoder to eval mode (BN uses running statistics).
    was_training = encoder.training
    encoder.eval()

    try:
        # ── 1. Extract embeddings once — shared by all tasks ───────────────
        _extract_embeddings_for_eval(
            encoder=encoder,
            eval_dataset_ref=eval_dataset_ref,
            save_path=tmp_embd_h5,
            device=device,
            num_workers=int(train_cfg.get("num_workers", 0)),
        )

        # ── 2. Run each task against the same embedding H5 ─────────────────
        from utils.config_v2 import ConfigV2  # noqa: PLC0415
        from evaluate_encoder import run_eval_task  # noqa: PLC0415
        for task_name in valid_tasks:
            # Per-task output sub-directory keeps artifacts organised.
            task_output_dir = os.path.join(eval_output_dir, task_name)
            os.makedirs(task_output_dir, exist_ok=True)
            try:
                task_override_keys = dict(eval_cfg.get(task_name) or {})
                runtime_overrides = {
                    **task_override_keys,
                    "embedding_h5_path": tmp_embd_h5,
                    "output_dir": task_output_dir,
                }
                resolved = ConfigV2().load_eval(task_name, overrides=runtime_overrides)
                result = run_eval_task(task_name, resolved)
                print(
                    f"[in_training_eval] {task_name}: "
                    f"{result['metric_name']} = {result['metric_value']:.6f}"
                )
                _log_eval_to_wandb(
                    task_name=task_name,
                    result=result,
                    epoch=epoch,
                    log_images=bool(eval_cfg.get("log_images_to_wandb", True)),
                )
            except Exception:
                print(f"[in_training_eval] ERROR in task '{task_name}' (continuing to next task):")
                traceback.print_exc()

    except Exception:
        print("[in_training_eval] ERROR during embedding extraction (aborting eval):")
        traceback.print_exc()

    finally:
        # Always delete the temporary embedding H5 (success or failure).
        if os.path.isfile(tmp_embd_h5):
            try:
                os.remove(tmp_embd_h5)
                print(f"[in_training_eval] Removed temp embedding H5: {tmp_embd_h5}")
            except OSError as exc:
                print(f"[in_training_eval] Warning: could not remove temp H5: {exc}")

        # Always restore encoder training state.
        if was_training:
            encoder.train()
            encoder.configure_trainability()

        print(f"[in_training_eval] ── Eval done ──\n")


# ---------------------------------------------------------------------------
# Internal: embedding extraction
# ---------------------------------------------------------------------------

def _extract_embeddings_for_eval(
    encoder,
    eval_dataset_ref: str,
    save_path: str,
    device: "torch.device",
    num_workers: int = 0,
) -> None:
    """Extract embeddings from *eval_dataset_ref* and write to *save_path*.

    Uses the training-shape config (clip_len / context_size / context_stride
    from configs_v2/train.yaml) and ``sample_all=True`` so every frame is covered.
    The encoder should already be in eval mode when this is called.
    """
    from dataset_preparation.h5vid_dataset import build_dataloader  # noqa: PLC0415
    from extract_embeddings import extract_embeddings, save_embeddings_h5  # noqa: PLC0415
    from utils.config_v2 import ConfigV2  # noqa: PLC0415

    # Resolve H5 path for the eval dataset from the V2 registry.
    ds_info = ConfigV2().resolve_dataset(eval_dataset_ref)
    h5_path = ds_info["processed_h5_path"]

    # Build extraction DataLoader.
    # config_path carries clip_len / context_size / context_stride;
    # h5_path_override pins the actual video H5 file.
    dataloader = build_dataloader(
        config_path=_V2_TRAIN_YAML,
        sample_all=True,
        sample_all_stride=1,
        shuffle=False,
        num_workers=num_workers,
        split="extract",
        h5_path_override=h5_path,
    )
    n_videos = len(dataloader.dataset)
    print(f"[in_training_eval] Extracting embeddings: dataset={eval_dataset_ref} "
          f"({n_videos} videos) → {save_path}")

    with torch.no_grad():
        results = extract_embeddings(encoder, dataloader, device)

    save_embeddings_h5(results, save_path)
    print(f"[in_training_eval] Extraction done ({len(results)} videos).")


# ---------------------------------------------------------------------------
# Internal: wandb logging
# ---------------------------------------------------------------------------

def _collect_image_payload(task_name: str, result: dict) -> dict:
    """Build a wandb-key → wandb.Image dict from a task result dict.

    Key naming convention (video-aware when possible)
    ─────────────────────────────────────────────────
    When ``result`` contains ``per_video_results`` (all-video mode of
    latent_distance_heatmap), the per-image W&B key suffix is
    ``vid<video_id>`` (e.g. ``heatmap_vid000001``), making every panel
    immediately identifiable in the W&B UI.

    When ``result`` contains ``selected_video_id`` (single-video mode),
    the same ``vid<video_id>`` suffix is used for i=0.

    For tasks that provide no video-id information (e.g. kendalls_tau)
    the suffix falls back to the 0-based integer index.

    Image families handled
    ──────────────────────
    - output_heatmap_path / output_heatmap_paths  → keys  heatmap_<suffix>
    - output_curve_path  / output_curve_paths     → keys  curve_<suffix>

    For tasks that never emit curve paths (e.g. kendalls_tau) the curve
    branch is silently skipped.

    Returns an empty dict when no valid image paths are found.
    """
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        return {}

    payload: dict = {}

    # Build a video-id label for each 0-based index.
    # "all" mode: per_video_results[i]["video_id"]  (list of per-video dicts)
    # single-video mode: selected_video_id  (applies only at i=0)
    per_video: list[dict] = result.get("per_video_results") or []

    def _key_suffix(i: int) -> str:
        """Return 'vid<video_id>' when available, else str(i)."""
        if i < len(per_video):
            vid_id = per_video[i].get("video_id")
            if vid_id:
                return f"vid{vid_id}"
        if i == 0 and result.get("selected_video_id"):
            return f"vid{result['selected_video_id']}"
        return str(i)

    # ── heatmap family ────────────────────────────────────────────────
    heatmap_paths: list[str] = list(result.get("output_heatmap_paths") or [])
    single_hm = result.get("output_heatmap_path")
    if not heatmap_paths and single_hm:
        heatmap_paths = [single_hm]
    for i, p in enumerate(heatmap_paths):
        if p and os.path.isfile(p):
            payload[f"eval/train/{task_name}/heatmap_{_key_suffix(i)}"] = wandb.Image(p)

    # ── anchor distance curve family ─────────────────────────────────
    curve_paths: list[str] = list(result.get("output_curve_paths") or [])
    single_cv = result.get("output_curve_path")
    if not curve_paths and single_cv:
        curve_paths = [single_cv]
    for i, p in enumerate(curve_paths):
        if p and os.path.isfile(p):
            payload[f"eval/train/{task_name}/curve_{_key_suffix(i)}"] = wandb.Image(p)

    return payload


def _log_eval_to_wandb(
    task_name: str,
    result: dict,
    epoch: int,
    log_images: bool,
) -> None:
    """Log scalar metric and optional images to the active wandb run.

    Uploads both heatmap and anchor-distance-curve images when the task
    result dict carries the corresponding path keys.  Either family may be
    absent (returns None paths) without causing an error.
    """
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        print("[in_training_eval] wandb not available; skipping wandb logging.")
        return

    if wandb.run is None:
        print("[in_training_eval] No active wandb run; skipping logging.")
        return

    metric_key = f"eval/train/{task_name}/{result['metric_name']}"
    log_payload: dict = {metric_key: result["metric_value"]}

    if log_images:
        log_payload.update(_collect_image_payload(task_name, result))

    wandb.log(log_payload, step=epoch + 1)
    img_keys = [k for k in log_payload if "/heatmap_" in k or "/curve_" in k]
    n_imgs = len(img_keys)
    if img_keys:
        print(f"[in_training_eval] Uploading {n_imgs} image(s): " +
              ", ".join(k.rsplit("/", 1)[-1] for k in sorted(img_keys)))
    print(
        f"[in_training_eval] Logged to wandb: {metric_key}={result['metric_value']:.6f}"
        + (f" + {n_imgs} image(s)" if n_imgs else "")
    )
