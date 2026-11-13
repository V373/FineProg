"""Offline temporal retrieval diagnostics. Rendering never advances an encoder.

The embedding / prediction H5 is the source of truth. W&B receives only a
bounded summary, after local artifacts have been finalized. Retry an upload
without recomputing anything with::

    python -m utils.temporal_diagnostics --upload-summary-only manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DIAGNOSTIC_KEYS = (
    "attention", "attention_valid", "history_lengths", "frame_indices",
    "branch_weights", "branch_difference", "current_component_norm",
    "history_component_norm", "history_injection_ratio",
)
PROVENANCE_KEYS = (
    "checkpoint_path", "checkpoint_sha256", "encoder_config",
    "processed_h5_path", "training_provenance", "source_provenance",
    "attention_mode", "ablation_config",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        if value.startswith(("{", "[")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_temporal_config(value: dict | None) -> dict:
    cfg = dict(value or {})
    defaults = {
        "enabled": False, "video_ids": [], "baseline_prediction_h5": None,
        "save_png": True, "save_mp4": True, "interval_frames": 16,
        "num_intervals": 3, "enrichment_vmax": 4.0,
        "log_to_wandb": False, "wandb_log_video": False, "wandb_max_videos": 1,
    }
    cfg = {**defaults, **cfg}
    for key in ("enabled", "save_png", "save_mp4", "log_to_wandb", "wandb_log_video"):
        if not isinstance(cfg[key], (bool, np.bool_)):
            raise ValueError(f"temporal_diagnostics.{key} must be boolean.")
        cfg[key] = bool(cfg[key])
    for key in ("interval_frames", "num_intervals", "wandb_max_videos"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"temporal_diagnostics.{key} must be a positive integer.")
    cfg["enrichment_vmax"] = float(cfg["enrichment_vmax"])
    if not np.isfinite(cfg["enrichment_vmax"]) or cfg["enrichment_vmax"] <= 1:
        raise ValueError("temporal_diagnostics.enrichment_vmax must be finite and > 1.")
    ids = cfg["video_ids"]
    if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids):
        raise ValueError("temporal_diagnostics.video_ids must be an explicit list of IDs.")
    if cfg["enabled"] and not ids:
        raise ValueError("Enabled temporal_diagnostics requires nonempty video_ids.")
    if len(set(ids)) != len(ids):
        raise ValueError("temporal_diagnostics.video_ids must not contain duplicates.")
    if cfg["baseline_prediction_h5"] is not None:
        cfg["baseline_prediction_h5"] = str(Path(cfg["baseline_prediction_h5"]).expanduser().resolve())
    return cfg


def select_video_ids(config: dict, available: list[str]) -> list[str]:
    if not config["enabled"]:
        return []
    selected = list(config["video_ids"])
    missing = set(selected).difference(available)
    if missing:
        raise ValueError(f"Temporal diagnostic videos are missing: {sorted(missing)}")
    return selected


def validate_temporal_diagnostics(data: dict, *, attention_mode: str = "learned") -> dict:
    """Validate one unpadded trajectory and return NumPy views of its arrays."""
    missing = set(DIAGNOSTIC_KEYS).difference(data)
    if missing:
        raise ValueError(f"Temporal diagnostics missing arrays: {sorted(missing)}")
    arrays = {key: np.asarray(data[key]) for key in DIAGNOSTIC_KEYS}
    indices = arrays["frame_indices"]
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(indices) < 1:
        raise ValueError("frame_indices must be a nonempty integer vector.")
    size = len(indices)
    if not np.array_equal(indices, np.arange(size)):
        raise ValueError("Temporal diagnostics require frame_indices=arange(T), sample_all_stride=1.")
    for key, array in arrays.items():
        shape = (size, size) if key in ("attention", "attention_valid") else (
            (size, 2) if key == "branch_weights" else (size,)
        )
        if array.shape != shape:
            raise ValueError(f"{key} must have shape {shape}; got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or Inf.")
    valid = arrays["attention_valid"]
    lengths = arrays["history_lengths"]
    attention = arrays["attention"]
    if valid.dtype != np.dtype(bool):
        raise ValueError("attention_valid must be boolean.")
    if not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("history_lengths must be integer.")
    causal = np.arange(size)[None, :] < np.arange(size)[:, None]
    if np.any(valid & ~causal):
        raise ValueError("attention_valid includes current or future positions.")
    if not np.array_equal(lengths, valid.sum(axis=1)):
        raise ValueError("history_lengths does not match the attention_valid mask.")
    if np.any(attention < 0) or np.any(attention[~valid] != 0):
        raise ValueError("attention must be nonnegative and zero at masked positions.")
    if attention_mode not in ("learned", "uniform", "none"):
        raise ValueError(f"Unsupported attention_mode: {attention_mode}")
    row_sum = (lengths > 0).astype(float) if attention_mode != "none" else np.zeros(size)
    if not np.allclose(attention.sum(axis=1), row_sum, atol=2e-6, rtol=0):
        raise ValueError("attention row sums disagree with history and attention_mode.")
    if attention_mode == "uniform":
        expected = valid / np.maximum(lengths[:, None], 1)
        if not np.allclose(attention, expected, atol=2e-6, rtol=0):
            raise ValueError("Uniform attention is not uniform over valid history.")
    weights = arrays["branch_weights"]
    if np.any((weights < 0) | (weights > 1)) or not np.allclose(weights.sum(-1), 1, atol=2e-6, rtol=0):
        raise ValueError("branch_weights must be [long,short] probabilities summing to one.")
    if np.any((arrays["branch_difference"] < -2e-6) | (arrays["branch_difference"] > 2 + 2e-6)):
        raise ValueError("branch_difference must equal a cosine distance in [0,2].")
    for key in ("current_component_norm", "history_component_norm", "history_injection_ratio"):
        if np.any(arrays[key] < 0):
            raise ValueError(f"{key} must be nonnegative.")
    return arrays


def read_temporal_diagnostics(group: h5py.Group) -> dict:
    metadata = {key: _json_value(value) for key, value in group.attrs.items()}
    mode = metadata.get("attention_mode", "learned")
    arrays = validate_temporal_diagnostics(
        {key: group[key][:] for key in DIAGNOSTIC_KEYS if key in group}, attention_mode=mode,
    )
    return {**arrays, "metadata": metadata, "attention_mode": mode}


def copy_prediction_provenance(source: h5py.File, destination: h5py.File) -> None:
    for key in PROVENANCE_KEYS:
        if key in source.attrs:
            destination.attrs[key] = source.attrs[key]


def validate_prediction_sources(query_h5_path: str, model_h5_path: str,
                                calibration_h5_path: str | None = None) -> str:
    """Require matching frozen encoder artifacts for the new diagnostic workflow."""
    hashes = []
    paths = [query_h5_path, model_h5_path]
    if calibration_h5_path is not None:
        paths.append(calibration_h5_path)
    for path in paths:
        with h5py.File(path, "r") as file:
            digest = _json_value(file.attrs.get("checkpoint_sha256", ""))
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Temporal diagnostics requires checkpoint_sha256 provenance in {path}; "
                             "prepare matching extraction / fitting artifacts explicitly.")
        hashes.append(digest)
    if len(set(hashes)) != 1:
        raise ValueError("Temporal diagnostics checkpoint_sha256 mismatch across query, Gaussian and calibration artifacts.")
    return hashes[0]


def copy_trajectory_diagnostics(source: h5py.Group, destination: h5py.Group,
                                target_steps: np.ndarray) -> None:
    if "temporal_diagnostics" not in source:
        raise ValueError(f"{source.name} has no temporal_diagnostics; re-extract with diagnostics enabled.")
    data = read_temporal_diagnostics(source["temporal_diagnostics"])
    diagnostic_hash = data["metadata"].get("checkpoint_sha256")
    if diagnostic_hash is not None and diagnostic_hash != _json_value(source.file.attrs.get("checkpoint_sha256")):
        raise ValueError("Temporal diagnostic subgroup checkpoint_sha256 does not match the query H5 root.")
    if not np.array_equal(data["frame_indices"], target_steps):
        raise ValueError("Temporal diagnostics / prediction target_steps mismatch (sample_all_stride=1 required).")
    source.copy(source["temporal_diagnostics"], destination, name="temporal_diagnostics")


def _hold_progress(progress: np.ndarray, is_ood: np.ndarray) -> np.ndarray:
    result = np.empty_like(progress)
    last = 0.0
    for index in range(len(progress)):
        if not is_ood[index]:
            last = progress[index]
        result[index] = last
    return result


def read_prediction_metadata(file: h5py.File) -> dict:
    metadata = {key: _json_value(value) for key, value in file.attrs.items()}
    embedding_path = metadata.get("nonexpert_h5_path")
    # Referenced paths may have been overwritten after this prediction was made.
    # Never infer a checkpoint / configuration / training run identity from them.
    metadata["checkpoint_provenance_status"] = "pinned" if metadata.get("checkpoint_sha256") else "unverified"
    if "processed_h5_path" not in metadata and embedding_path and Path(embedding_path).is_file():
        with h5py.File(embedding_path, "r") as embedding_file:
            if "processed_h5_path" in embedding_file.attrs:
                metadata["processed_h5_path"] = _json_value(embedding_file.attrs["processed_h5_path"])
                metadata["processed_h5_path_source"] = "inferred_from_nonexpert_h5"
    return metadata


def read_prediction_trajectory(file: h5py.File, video_id: str, *, diagnostics: bool = True) -> dict:
    group = file[f"nonexperts/{video_id}"]
    keys = ("target_steps", "progress_mean", "progress_gated", "is_ood_raw", "is_ood", "posterior")
    data = {key: np.asarray(group[key][:]) for key in keys}
    size = len(data["target_steps"])
    if size < 1 or not np.array_equal(data["target_steps"], np.arange(size)):
        raise ValueError("Temporal rendering requires target_steps=arange(T), sample_all_stride=1.")
    for key in keys[1:]:
        array = data[key]
        if array.shape[0] != size or not np.isfinite(array).all():
            raise ValueError(f"Invalid prediction {key} for {video_id}.")
    data["progress_gated_raw"] = (group["progress_gated_raw"][:] if "progress_gated_raw" in group
                                  else _hold_progress(data["progress_mean"], data["is_ood_raw"]))
    for key in ("keyframe_labels", "phase_labels"):
        if key in group:
            data[key] = group[key][:]
            if data[key].shape != (size,):
                raise ValueError(f"{key} must match target_steps.")
    if diagnostics:
        data["diagnostics"] = read_temporal_diagnostics(group["temporal_diagnostics"])
        if not np.array_equal(data["diagnostics"]["frame_indices"], data["target_steps"]):
            raise ValueError("Temporal frame_indices / prediction target_steps mismatch.")
    data["video_id"] = video_id
    return data


def validate_baseline_alignment(current: dict, baseline: dict,
                                current_metadata: dict, baseline_metadata: dict) -> None:
    if current["video_id"] != baseline["video_id"] or not np.array_equal(
        current["target_steps"], baseline["target_steps"]
    ):
        raise ValueError("Baseline video_id / target_steps mismatch.")
    left = current_metadata.get("processed_h5_path")
    right = baseline_metadata.get("processed_h5_path")
    if not left or not right or Path(left).expanduser().resolve() != Path(right).expanduser().resolve():
        raise ValueError("Baseline and new prediction must identify the same processed_h5_path.")
    encoder_config = _json_value(current_metadata.get("encoder_config", {}))
    memory_config = encoder_config.get("memory_config", {}) if isinstance(encoder_config, dict) else {}
    expected_hash = memory_config.get("source_checkpoint_sha256")
    baseline_hash = baseline_metadata.get("checkpoint_sha256")
    if expected_hash and baseline_hash and expected_hash != baseline_hash:
        raise ValueError("Baseline checkpoint_sha256 does not match the new encoder's source_checkpoint_sha256.")
    status = "verified" if expected_hash and baseline_hash else "unverified"
    baseline_metadata["source_checkpoint_verification"] = status
    baseline["source_checkpoint_verification"] = status


def summarize_history_intervals(attention_row: np.ndarray, valid_row: np.ndarray,
                                *, interval_frames: int = 16, num_intervals: int = 3) -> list[dict]:
    """Rank fixed disjoint historical frame blocks; ties choose the earlier block."""
    row = np.asarray(attention_row)
    valid = np.asarray(valid_row, dtype=bool)
    if row.ndim != 1 or valid.shape != row.shape or interval_frames < 1 or num_intervals < 1:
        raise ValueError("Invalid history interval inputs.")
    intervals = []
    for start in range(0, len(row), interval_frames):
        indices = np.flatnonzero(valid[start:start + interval_frames]) + start
        if not len(indices):
            continue
        weights = row[indices]
        mass = float(weights.sum())
        if mass <= 0:
            continue
        representative = int(indices[min(np.searchsorted(np.cumsum(weights), mass / 2), len(indices) - 1)])
        intervals.append({"start": int(indices[0]), "end": int(indices[-1]) + 1,
                          "mass": mass, "representative": representative})
    return sorted(intervals, key=lambda entry: (-entry["mass"], entry["start"]))[:num_intervals]


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _event_markers(data: dict) -> list[tuple[int, str]]:
    keyframes = np.asarray(data.get("keyframe_labels", []))
    indices = np.flatnonzero(keyframes >= 0)
    if len(indices):
        return [(int(index), f"event {keyframes[index]}") for index in indices]
    phases = np.asarray(data.get("phase_labels", []))
    if len(phases) < 2:
        return []
    # A change adjacent to an unknown (-1) label is not an annotated boundary.
    transitions = (phases[:-1] >= 0) & (phases[1:] >= 0) & (phases[:-1] != phases[1:])
    return [(int(index), f"phase {phases[index - 1]} → {phases[index]}")
            for index in np.flatnonzero(transitions) + 1]


def _query_axes(axis, size: int, enrichment_vmax: float) -> dict:
    """Create a fixed-scale query panel shared by static and animated media."""
    enrichment_axis = axis.twinx()
    line, = axis.plot(np.arange(size), np.zeros(size), color="C0", label="A query row")
    enriched_line, = enrichment_axis.plot(np.arange(size), np.zeros(size), color="C1", label="n×A")
    axis.set_xlim(-0.5, size - 0.5)
    axis.set_ylim(0, 1)
    enrichment_axis.set_ylim(0, enrichment_vmax)
    axis.set_ylabel("A [0,1]", fontsize=7, color="C0")
    enrichment_axis.set_ylabel(f"n×A [0,{enrichment_vmax:g}]", fontsize=7, color="C1")
    axis.set_xlabel("Historical frame index j", fontsize=7)
    axis.tick_params(labelsize=6)
    enrichment_axis.tick_params(labelsize=6)
    axis.legend(handles=[line, enriched_line], loc="upper left", fontsize=6, ncol=2)
    return {"axis": axis, "line": line, "enriched_line": enriched_line, "spans": []}


def _update_query_panel(panel: dict, diag: dict, step: int, *, interval_frames: int,
                        num_intervals: int, position: str, timing_note: str = "",
                        history_cards: bool = False) -> list[dict]:
    """Use one interval selection for both the shaded query and history cards."""
    valid = diag["attention_valid"][step]
    row = diag["attention"][step]
    count = int(diag["history_lengths"][step])
    panel["line"].set_ydata(np.where(valid, row, np.nan))
    panel["enriched_line"].set_ydata(np.where(valid, count * row, np.nan))
    intervals = summarize_history_intervals(row, valid, interval_frames=interval_frames,
                                            num_intervals=num_intervals)
    # Each video query has its own intervals; retaining patches would leave stale reads.
    for span in panel["spans"]:
        span.remove()
    panel["spans"] = [panel["axis"].axvspan(item["start"], item["end"],
                                           color="C0", alpha=0.18, lw=0, zorder=0.1)
                      for item in intervals]
    status = ("empty history: no read" if count == 0 else
              "no historical read (zero attention)" if not intervals else
              "one history token: no selection" if count == 1 else f"{count} valid history tokens")
    shading_note = f"Blue shading: top {num_intervals} positive-mass historical intervals"
    if history_cards:
        shading_note += " (same as history cards)"
    panel["axis"].set_title(
        f"Query {position}: {status}; mode={diag['attention_mode']}{timing_note}\n{shading_note}",
        fontsize=8,
    )
    return intervals


def _overview_figure(data: dict, baseline: dict | None, enrichment_vmax: float, *,
                     query_step: int | None = None, interval_frames: int = 16,
                     num_intervals: int = 3):
    plt = _pyplot()
    diag = data["diagnostics"]
    steps = data["target_steps"]
    size = len(steps)
    has_query = query_step is not None
    fig = plt.figure(figsize=(10, 13 if has_query else 11), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.018, 1, 0.982))
    ratios = (1.25, 1, 1.2, 1.2, 1.2) + ((1.0,) if has_query else ())
    grid = fig.add_gridspec(len(ratios), 1, height_ratios=ratios)
    progress_ax = fig.add_subplot(grid[0])
    for values, label, style, color in (
        (data["progress_mean"], "New raw", "-", "C0"),
        (data["progress_gated_raw"], "New online gated", "--", "C1"),
        (data["progress_gated"], "New final gated", ":", "C3"),
    ):
        progress_ax.plot(steps, values, style, color=color, label=label, lw=1)
    if baseline is not None:
        prefix = ("Baseline" if baseline.get("source_checkpoint_verification") == "verified"
                  else "Baseline (source unverified)")
        for key, name, style in (("progress_mean", "raw", "-"),
                                 ("progress_gated_raw", "online gated", "--"),
                                 ("progress_gated", "final gated", ":")):
            progress_ax.plot(steps, baseline[key], style, color="0.35", label=f"{prefix} {name}", lw=1)
    ood_rows = [(data["is_ood_raw"], "New raw OOD", "C1"), (data["is_ood"], "New final OOD", "C3")]
    if baseline is not None:
        ood_rows += [(baseline["is_ood_raw"], "Base raw OOD", "0.5"),
                     (baseline["is_ood"], "Base final OOD", "0.15")]
    for row_index, (mask, name, color) in enumerate(ood_rows):
        y = -0.10 - 0.09 * row_index
        progress_ax.scatter(steps[mask.astype(bool)], np.full(int(mask.sum()), y),
                            s=6, marker="s", color=color)
        progress_ax.text(0.99, y, name, transform=progress_ax.get_yaxis_transform(),
                         ha="right", va="center", fontsize=6, color=color)
    progress_ax.set_ylim(-0.10 - 0.09 * len(ood_rows), 1.08)
    progress_ax.set_ylabel("Progress / OOD")
    progress_ax.legend(loc="upper left", ncol=3, fontsize=6)
    query_note = f"; selected query frame {query_step}" if has_query else ""
    progress_ax.set_title(
        f"{data['video_id']} — temporal retrieval{query_note}\n(an explicit read, not causal attribution)", fontsize=9)
    subgrid = grid[1].subgridspec(1, 3)
    gate_ax, norm_ax, ratio_ax = [fig.add_subplot(subgrid[index]) for index in range(3)]
    gate_ax.plot(steps, diag["branch_weights"][:, 0], label="w long")
    gate_ax.plot(steps, diag["branch_weights"][:, 1], label="w short")
    gate_ax.plot(steps, diag["branch_difference"], label="1−cos(S,L)", lw=1)
    gate_ax.set_ylim(0, 2.05)
    gate_ax.legend(fontsize=6)
    gate_ax.set_title("Effective fusion / branch difference", fontsize=8)
    norm_ax.plot(steps, diag["current_component_norm"], label="||Wc x||")
    norm_ax.plot(steps, diag["history_component_norm"], label="||Wh r||")
    norm_ax.legend(fontsize=6)
    norm_ax.set_title("Current / history component norms", fontsize=8)
    ratio_ax.plot(steps, diag["history_injection_ratio"], color="C4")
    ratio_ax.set_title("History injection ratio ρ", fontsize=8)
    att_ax, enriched_ax, post_ax = [fig.add_subplot(grid[index]) for index in (2, 3, 4)]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    for axis, array, vmax, title in (
        (att_ax, diag["attention"], 1.0, "A: past-only attention; gray=masked/empty"),
        (enriched_ax, diag["attention"] * diag["history_lengths"][:, None], enrichment_vmax,
         f"History enrichment n×A (fixed scale; values >{enrichment_vmax:g} saturate)"),
    ):
        heatmap = axis.imshow(np.ma.array(array.T, mask=~diag["attention_valid"].T),
                              aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
        fig.colorbar(heatmap, ax=axis, pad=0.01, fraction=0.025)
        axis.set_ylabel("Historical frame j")
        axis.set_title(title, fontsize=8)
    post_image = post_ax.imshow(data["posterior"].T, origin="lower", aspect="auto",
                                interpolation="nearest", vmin=0, vmax=1, cmap="viridis",
                                extent=(-0.5, size - 0.5, 0, 1))
    fig.colorbar(post_image, ax=post_ax, pad=0.01, fraction=0.025)
    post_ax.set_ylabel("Progress bin")
    post_ax.set_title("Gaussian posterior (fixed scale [0,1])", fontsize=8)
    time_axes = (progress_ax, gate_ax, norm_ax, ratio_ax, post_ax)
    all_time_axes = (*time_axes, att_ax, enriched_ax)
    for axis in all_time_axes:
        axis.set_xlim(-0.5, size - 0.5)
        axis.set_xlabel("Current frame t", fontsize=7)
        axis.tick_params(labelsize=6)
    for index, label in _event_markers(data):
        for axis in all_time_axes:
            axis.axvline(index, color="C2", lw=0.7, alpha=0.7)
        progress_ax.text(index, 1.03, label, fontsize=5)
    if has_query:
        query_panel = _query_axes(fig.add_subplot(grid[5]), size, enrichment_vmax)
        _update_query_panel(query_panel, diag, query_step, interval_frames=interval_frames,
                            num_intervals=num_intervals, position=f"frame {query_step}")
    fig.text(0.5, 0.001, f"Attention mode: {diag['attention_mode']}; empty rows carry no read; n=1 has no selection.",
             ha="center", fontsize=7)
    cursors = [axis.axvline(0, color="magenta", lw=0.8, animated=True) for axis in time_axes]
    row_cursors = [axis.axvline(0, color="magenta", lw=0.8, animated=True) for axis in (att_ax, enriched_ax)]
    return fig, cursors, row_cursors


def render_temporal_png(data: dict, output_path: str | Path, *, baseline: dict | None = None,
                        enrichment_vmax: float = 4.0, query_step: int | None = None,
                        interval_frames: int = 16, num_intervals: int = 3) -> None:
    plt = _pyplot()
    size = len(data["target_steps"])
    query_step = size - 1 if query_step is None else query_step
    if isinstance(query_step, bool) or not isinstance(query_step, (int, np.integer)) or not 0 <= query_step < size:
        raise ValueError("query_step must be an integer within the trajectory.")
    fig, cursors, row_cursors = _overview_figure(
        data, baseline, enrichment_vmax, query_step=query_step,
        interval_frames=interval_frames, num_intervals=num_intervals,
    )
    try:
        for cursor in cursors + row_cursors:
            cursor.set_xdata([query_step, query_step])
            cursor.set_animated(False)
        fig.savefig(output_path, dpi=120)
    finally:
        plt.close(fig)


def render_temporal_video(data: dict, frames, fps: float | None, output_path: str | Path, *,
                          baseline: dict | None = None, interval_frames: int = 16,
                          num_intervals: int = 3, enrichment_vmax: float = 4.0) -> None:
    """Compose cached overview + query row + current and interval representative frames."""
    import cv2
    import imageio.v2 as imageio
    plt = _pyplot()
    size = len(data["target_steps"])
    if len(frames) != size or (fps is not None and (not np.isfinite(fps) or fps <= 0)):
        raise ValueError("Video frames must match the prediction; source fps must be positive or None.")
    playback_fps = float(fps) if fps is not None else 10.0
    if len(frames.shape) != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Temporal video requires uint8 RGB frames [T,H,W,3].")
    diag = data["diagnostics"]
    fig, cursors, row_cursors = _overview_figure(data, baseline, enrichment_vmax)
    query_fig, query_ax = plt.subplots(figsize=(10, 1.7), layout="constrained")
    query_panel_artists = _query_axes(query_ax, size, enrichment_vmax)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    width = int(fig.canvas.get_width_height()[0])
    # Disable layout recomputation when using the cached background for cursors.
    fig.set_layout_engine(None)
    try:
        with imageio.get_writer(str(output_path), fps=playback_fps, codec="libx264",
                               macro_block_size=2, pixelformat="yuv420p") as writer:
            for step in range(size):
                fig.canvas.restore_region(background)
                for cursor in cursors:
                    cursor.set_xdata([step, step])
                    cursor.axes.draw_artist(cursor)
                for cursor in row_cursors:
                    cursor.set_xdata([step, step])
                    cursor.axes.draw_artist(cursor)
                overview = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                position = f"frame {step}" + (f" ({step / fps:.2f}s)" if fps is not None else "")
                timing_note = "" if fps is not None else "; source FPS unknown, playback only: 10 fps"
                intervals = _update_query_panel(
                    query_panel_artists, diag, step, interval_frames=interval_frames,
                    num_intervals=num_intervals, position=position, timing_note=timing_note,
                    history_cards=True,
                )
                query_fig.canvas.draw()
                query_panel = np.asarray(query_fig.canvas.buffer_rgba())[..., :3].copy()
                if query_panel.shape[1] != width:
                    query_panel = cv2.resize(query_panel, (width, query_panel.shape[0]))
                montage = np.full((190, width, 3), 255, dtype=np.uint8)
                slots = 1 + num_intervals
                slot_width = width // slots
                cards = [(step, f"Current {position}", "")]
                cards += [(item["representative"], f"Past [{item['start']},{item['end']}) mass={item['mass']:.3f}",
                           (f"{item['start'] / fps:.2f}-{item['end'] / fps:.2f}s; " if fps is not None else "")
                           + f"frame {item['representative']}")
                          for item in intervals]
                for slot, (index, title, subtitle) in enumerate(cards):
                    frame = np.asarray(frames[index])
                    scale = min((slot_width - 8) / frame.shape[1], 142 / frame.shape[0])
                    thumbnail = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)),
                                                    max(1, round(frame.shape[0] * scale))))
                    x = slot * slot_width + (slot_width - thumbnail.shape[1]) // 2
                    montage[4:4 + thumbnail.shape[0], x:x + thumbnail.shape[1]] = thumbnail
                    cv2.putText(montage, title, (slot * slot_width + 3, 162), cv2.FONT_HERSHEY_SIMPLEX,
                                0.32, (25, 25, 25), 1, cv2.LINE_AA)
                    cv2.putText(montage, subtitle, (slot * slot_width + 3, 179), cv2.FONT_HERSHEY_SIMPLEX,
                                0.30, (25, 25, 25), 1, cv2.LINE_AA)
                if not intervals:
                    cv2.putText(montage, "No retrieved historical interval", (slot_width + 10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)
                writer.append_data(np.concatenate((montage, query_panel, overview), axis=0))
    finally:
        plt.close(fig)
        plt.close(query_fig)


def _source_fps(group: h5py.Group, video_id: str) -> float | None:
    try:
        fps = float(group.attrs.get("fps", 0))
    except (TypeError, ValueError):
        fps = 0.0
    if not np.isfinite(fps) or fps <= 0:
        import cv2
        from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_pred import _resolve_raw_video_path
        try:
            path, _ = _resolve_raw_video_path(group.parent, video_id)
            capture = cv2.VideoCapture(str(path))
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
            finally:
                capture.release()
        except (OSError, ValueError, KeyError):
            return None
    return fps if np.isfinite(fps) and fps > 0 else None


def _write_manifest(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_value) + "\n")
    temporary.replace(path)


def render_prediction_diagnostics(prediction_h5_path: str | Path, config: dict) -> dict:
    """Finalize all selected local artifacts, then optionally upload a small summary."""
    from contextlib import ExitStack
    cfg = parse_temporal_config(config)
    if not cfg["enabled"]:
        return {}
    prediction_path = Path(prediction_h5_path).expanduser().resolve()
    output_dir = prediction_path.parent / "temporal_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {"prediction_h5_path": str(prediction_path), "config": cfg,
                "local_status": "running", "upload_status": "not_requested", "per_video_results": []}
    _write_manifest(manifest_path, manifest)
    try:
        with ExitStack() as resources:
            prediction_file = resources.enter_context(h5py.File(prediction_path, "r"))
            metadata = read_prediction_metadata(prediction_file)
            manifest["provenance"] = metadata
            selected = select_video_ids(cfg, list(prediction_file["nonexperts"]))
            baseline_file = (resources.enter_context(h5py.File(cfg["baseline_prediction_h5"], "r"))
                             if cfg["baseline_prediction_h5"] else None)
            baseline_metadata = read_prediction_metadata(baseline_file) if baseline_file is not None else None
            if baseline_metadata is not None:
                manifest["baseline_provenance"] = baseline_metadata
            processed_file = None
            if cfg["save_mp4"]:
                if not metadata.get("processed_h5_path"):
                    raise ValueError("Temporal video requires processed_h5_path provenance.")
                processed_file = resources.enter_context(h5py.File(metadata["processed_h5_path"], "r"))
            sums = {"real_frame_count": 0, "nonempty_history_count": 0,
                    "long_weight_sum": 0.0, "short_weight_sum": 0.0, "branch_difference_sum": 0.0,
                    "history_injection_ratio_sum": 0.0, "raw_ood_count": 0, "final_ood_count": 0}
            for video_id in selected:
                data = read_prediction_trajectory(prediction_file, video_id)
                baseline = (read_prediction_trajectory(baseline_file, video_id, diagnostics=False)
                            if baseline_file is not None else None)
                if baseline is not None:
                    validate_baseline_alignment(data, baseline, metadata, baseline_metadata)
                # video_id is an H5 group key, not an output path.
                name = video_id.replace("/", "_").replace("\\", "_")
                result = {"video_id": video_id, "png_path": None, "video_path": None}
                if cfg["save_png"]:
                    png_path = output_dir / f"{name}.png"
                    render_temporal_png(data, png_path, baseline=baseline, enrichment_vmax=cfg["enrichment_vmax"],
                                        interval_frames=cfg["interval_frames"], num_intervals=cfg["num_intervals"])
                    result["png_path"] = str(png_path)
                if cfg["save_mp4"]:
                    group = processed_file[f"videos/{video_id}"]
                    video_path = output_dir / f"{name}.mp4"
                    source_fps = _source_fps(group, video_id)
                    render_temporal_video(data, group["frames"], source_fps, video_path,
                                          baseline=baseline, interval_frames=cfg["interval_frames"],
                                          num_intervals=cfg["num_intervals"], enrichment_vmax=cfg["enrichment_vmax"])
                    result["video_path"] = str(video_path)
                    result["source_fps"] = source_fps
                    result["playback_fps"] = source_fps if source_fps is not None else 10.0
                diag = data["diagnostics"]
                nonempty = diag["history_lengths"] > 0
                sums["real_frame_count"] += len(data["target_steps"])
                sums["nonempty_history_count"] += int(nonempty.sum())
                for key in ("long", "short"):
                    sums[f"{key}_weight_sum"] += float(diag["branch_weights"][:, 0 if key == "long" else 1].sum())
                sums["branch_difference_sum"] += float(diag["branch_difference"].sum())
                sums["history_injection_ratio_sum"] += float(diag["history_injection_ratio"][nonempty].sum())
                sums["raw_ood_count"] += int(data["is_ood_raw"].sum())
                sums["final_ood_count"] += int(data["is_ood"].sum())
                manifest["per_video_results"].append(result)
                _write_manifest(manifest_path, manifest)
            count = sums["real_frame_count"]
            metrics = {"real_frame_count": count, "nonempty_history_count": sums["nonempty_history_count"]}
            if count:
                for key in ("long_weight", "short_weight", "branch_difference"):
                    metrics[f"{key}_mean"] = sums[f"{key}_sum"] / count
                metrics["raw_ood_fraction"] = sums["raw_ood_count"] / count
                metrics["final_ood_fraction"] = sums["final_ood_count"] / count
            if sums["nonempty_history_count"]:
                metrics["history_injection_ratio_mean"] = sums["history_injection_ratio_sum"] / sums["nonempty_history_count"]
            manifest["metrics"] = metrics
        manifest["local_status"] = "complete"
        _write_manifest(manifest_path, manifest)
    except Exception as exc:
        manifest["local_status"] = "failed"
        manifest["local_error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(manifest_path, manifest)
        raise
    if cfg["log_to_wandb"]:
        upload_offline_summary(manifest_path)
    return {"temporal_manifest_path": str(manifest_path), "per_video_results": manifest["per_video_results"]}


def upload_offline_summary(manifest_path: str | Path, *, wandb_module=None) -> dict:
    """Upload only saved summary media; failures never remove or invalidate local work."""
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("local_status") != "complete":
        raise ValueError("Only a completed local diagnostic manifest can be uploaded.")
    run = None
    try:
        from uuid import uuid4
        if wandb_module is None:
            import wandb as wandb_module
        cfg = parse_temporal_config(manifest["config"])
        provenance = manifest.get("provenance", {})
        source = provenance.get("training_provenance", {})
        if not isinstance(source, dict):
            source = {}
        project = source.get("project", source.get("wandb_project", "mytcc"))
        entity = source.get("entity", source.get("wandb_entity"))
        previous_run = getattr(wandb_module, "run", None)
        run = wandb_module.init(project=project, entity=entity, job_type="eval", reinit="create_new",
                                id=uuid4().hex, resume="never",
                                group=source.get("run_id", source.get("wandb_run_id")),
                                name=f"temporal-eval-{str(provenance.get('checkpoint_sha256', 'unknown'))[:12]}",
                                config={"source": provenance, "temporal_diagnostics": cfg})
        forbidden_ids = {value for value in (getattr(previous_run, "id", None),
                                             source.get("run_id", source.get("wandb_run_id"))) if value}
        if run is None or run is previous_run or getattr(run, "id", None) in forbidden_ids:
            run = None
            raise RuntimeError("W&B did not create a separate evaluation run.")
        payload = {f"temporal/{key}": value for key, value in manifest.get("metrics", {}).items()}
        selected = manifest["per_video_results"][:cfg["wandb_max_videos"]]
        for entry in selected:
            prefix = f"temporal/{entry['video_id']}"
            if entry.get("png_path"):
                payload[f"{prefix}/summary"] = wandb_module.Image(entry["png_path"])
            if entry.get("intervention_png_path"):
                payload[f"{prefix}/intervention"] = wandb_module.Image(entry["intervention_png_path"])
            if cfg["wandb_log_video"] and entry.get("video_path"):
                payload[f"{prefix}/video"] = wandb_module.Video(entry["video_path"], format="mp4")
        run.log(payload)
        run.summary.update({"local_manifest_path": str(path), "prediction_h5_path": manifest.get("prediction_h5_path"),
                            "result_h5_path": manifest.get("prediction_h5_path", manifest.get("output_h5_path")),
                            "local_trajectory_count": len(manifest["per_video_results"]),
                            "source_checkpoint_sha256": provenance.get("checkpoint_sha256", "unknown")})
        manifest["wandb_run_id"] = run.id
        manifest["wandb_run_url"] = getattr(run, "url", None)
        run.finish()
        run = None
        manifest["upload_status"] = "complete"
        manifest.pop("upload_error", None)
    except Exception as exc:
        manifest["upload_status"] = "failed"
        manifest["upload_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[temporal_diagnostics] W&B upload failed; local artifacts remain at {path.parent}: {exc}")
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
        _write_manifest(path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-summary-only", type=Path, required=True)
    args = parser.parse_args()
    upload_offline_summary(args.upload_summary_only)
