#!/usr/bin/env python3
"""Replay one configured intervention offline, with the original fixed Gaussian.

Uses the existing gaussian_progress_pred YAML plus temporal_diagnostics and
temporal_intervention. This entrypoint never trains or fits a Gaussian.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_v2 import ConfigV2
from utils.encoder_checkpoint import build_encoder_from_checkpoint, _checkpoint_sha256
from utils.temporal_diagnostics import (
    parse_temporal_config, select_video_ids, upload_offline_summary,
    validate_prediction_sources, read_prediction_metadata,
)
from utils.temporal_intervention import (
    make_history_mask, replay_intervention, render_intervention_comparison,
)
from algos.eval_task.tcc_eval_tasks.task_gaussian_progress_pred import (
    _apply_model_pca, _infer_one_trajectory, _parse_pred_config,
    _read_calibration_distance_bins, _read_gaussian_model,
)


def run_intervention(config, output_dir):
    intervention = dict(config.get("temporal_intervention", {}))
    if intervention.get("enabled") is not True:
        raise ValueError("Set temporal_intervention.enabled=true for this explicit offline entrypoint.")
    cfg = _parse_pred_config(config)
    display = parse_temporal_config({**config.get("temporal_diagnostics", {}), "enabled": True})
    query_path = cfg["nonexpert_h5_path"]
    calibration_path = cfg["calibration_h5_path"] if cfg["enable_calibration"] else None
    expected_hash = validate_prediction_sources(query_path, cfg["gaussian_model_h5_path"], calibration_path)
    with h5py.File(query_path, "r") as file:
        provenance = read_prediction_metadata(file)
        selected = select_video_ids(display, list(file["videos"]))
        expected = {vid: (file[f"videos/{vid}/embeddings"][:],
                          file[f"videos/{vid}/target_steps"][:]) for vid in selected}
    checkpoint_path = provenance["checkpoint_path"]
    if _checkpoint_sha256(checkpoint_path) != expected_hash:
        raise ValueError("Actual checkpoint file differs from the query/Gaussian checkpoint hash.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder_from_checkpoint(checkpoint_path, device).eval()
    if encoder.memory_config.get("mode") != "cross_attention":
        raise ValueError("This replay entrypoint requires a cross_attention checkpoint.")
    model = _read_gaussian_model(cfg["gaussian_model_h5_path"])
    calibration = (_read_calibration_distance_bins(calibration_path, model)
                   if calibration_path else None)

    def infer(result):
        result.update(_infer_one_trajectory(
            _apply_model_pca(result["embeddings"], model), model,
            cfg["posterior_temperature"], cfg["entropy_epsilon"], calibration,
            cfg["ood_p_value_threshold"], cfg["enable_ood_filter"],
            cfg["ood_filter_max_gap"], cfg["ood_filter_min_ood_run"]))

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"kind": "fixed_gaussian_intervention", "config": display,
                "intervention_config": intervention, "provenance": provenance,
                "gaussian_model_h5_path": cfg["gaussian_model_h5_path"],
                "calibration_h5_path": calibration_path, "local_status": "running",
                "upload_status": "not_requested", "per_video_results": []}
    manifest_path = output_dir / "manifest.json"
    h5_path = output_dir / "interventions.h5"
    try:
        with h5py.File(provenance["processed_h5_path"], "r") as videos, h5py.File(h5_path, "w-") as output:
            output.attrs["kind"] = "fixed_gaussian_intervention"
            output.attrs["checkpoint_sha256"] = expected_hash
            output.attrs["config"] = json.dumps(config)
            for video_id in selected:
                frames = np.asarray(videos[f"videos/{video_id}/frames"][:])
                if not np.array_equal(expected[video_id][1], np.arange(len(frames))):
                    raise ValueError("Interventions require complete original frame indices.")
                with torch.no_grad():
                    raw = encoder.encode_local_episode(torch.from_numpy(frames).permute(0, 3, 1, 2))
                reference = replay_intervention(encoder, raw)
                if not np.allclose(reference["embeddings"], expected[video_id][0], rtol=0, atol=1e-5):
                    raise ValueError("Replayed original embeddings differ from the saved query artifact.")
                infer(reference)
                mask, metadata = make_history_mask(
                    reference["diagnostics"], intervention.get("mask_mode", "none"),
                    query_step=intervention.get("query_step", -1),
                    interval_frames=display["interval_frames"], seed=intervention.get("seed", 42))
                result = {"video_id": video_id, "png_path": None, "video_path": None, **metadata}
                group = output.create_group(f"videos/{video_id}")
                group.attrs["intervention"] = json.dumps(metadata)
                runs = {"original": reference}
                if mask is not None:
                    changed = replay_intervention(
                        encoder, raw, attention_mode=intervention.get("attention_mode"),
                        fusion_mode=intervention.get("fusion_mode"), history_mask=mask)
                    infer(changed)
                    runs["intervention"] = changed
                    group.create_dataset("history_key_mask", data=mask)
                    group.create_dataset("embedding_change_l2", data=np.linalg.norm(
                        changed["embeddings"] - reference["embeddings"], axis=-1))
                    group.create_dataset("progress_change", data=changed["progress_mean"] - reference["progress_mean"])
                    png_path = output_dir / (video_id.replace("/", "_").replace("\\", "_") + ".png")
                    render_intervention_comparison(reference, changed, png_path)
                    result["png_path"] = str(png_path)
                for name, data in runs.items():
                    run = group.create_group(name)
                    run.attrs["config"] = json.dumps(data["config"])
                    for key, value in data.items():
                        if isinstance(value, np.ndarray):
                            run.create_dataset(key, data=value, compression="gzip")
                    diagnostic = run.create_group("temporal_diagnostics")
                    diagnostic.attrs["attention_mode"] = data["config"]["attention_mode"]
                    for key, value in data["diagnostics"].items():
                        diagnostic.create_dataset(key, data=value, compression="gzip")
                manifest["per_video_results"].append(result)
        manifest.update(local_status="complete", output_h5_path=str(h5_path))
    except Exception as exc:
        manifest.update(local_status="failed", error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if display["log_to_wandb"]:
        upload_offline_summary(manifest_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Gaussian prediction YAML with intervention settings")
    parser.add_argument("--output-dir", help="New output directory; must not exist")
    args = parser.parse_args()
    config = ConfigV2().load_eval("gaussian_progress_pred", config_path=args.config)
    output = args.output_dir or PROJECT_ROOT / "outputs" / "temporal_interventions" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_intervention(config, output)
