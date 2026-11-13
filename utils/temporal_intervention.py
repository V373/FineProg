"""Small offline counterfactuals: keep observations/short fixed and replay long."""

from contextlib import contextmanager

import numpy as np
import torch

from .temporal_diagnostics import summarize_history_intervals


@contextmanager
def _intervention_modes(memory, attention_mode, fusion_mode):
    attention = memory.attention_mode if attention_mode is None else attention_mode
    fusion = memory.fusion_mode if fusion_mode is None else fusion_mode
    if attention not in ("learned", "uniform", "none"):
        raise ValueError("Invalid intervention attention_mode.")
    if fusion not in ("learned", "fixed", "short_only", "long_only"):
        raise ValueError("Invalid intervention fusion_mode.")
    previous = memory.attention_mode, memory.fusion_mode
    try:
        memory.attention_mode, memory.fusion_mode = attention, fusion
        yield
    finally:
        memory.attention_mode, memory.fusion_mode = previous


@torch.no_grad()
def replay_intervention(encoder, raw, *, attention_mode=None, fusion_mode=None,
                        history_mask=None):
    """Replay a complete local episode without touching any online stream state.

    Each counterfactual starts at t=0; changes to a historical key therefore
    affect every subsequent eligible long read and recurrent update. The short
    path receives exactly the same local sequence and frozen parameters.
    """
    if encoder.training:
        raise ValueError("Intervention replay requires encoder.eval().")
    if encoder.memory_config.get("mode") != "cross_attention":
        raise ValueError("Intervention replay requires cross_attention memory.")
    if raw.ndim != 2 or len(raw) < 1:
        raise ValueError("Intervention raw embeddings must be nonempty [T,D].")
    diagnostic = {}
    if history_mask is not None:
        history_mask = torch.as_tensor(history_mask, device=raw.device)
        if history_mask.dtype != torch.bool or history_mask.shape != (len(raw),):
            raise ValueError("Intervention history_mask must be boolean [T].")
        history_mask = history_mask[None]
    with _intervention_modes(encoder.memory, attention_mode, fusion_mode):
        embeddings = encoder.forward_episode(
            [raw], [len(raw)], diagnostics=diagnostic, history_mask=history_mask)[0]
        config = {"attention_mode": encoder.memory.attention_mode,
                  "fusion_mode": encoder.memory.fusion_mode}
    arrays = {key: value[0].cpu().numpy() for key, value in diagnostic.items()}
    arrays["frame_indices"] = np.arange(len(raw), dtype=np.int64)
    return {"embeddings": embeddings.cpu().numpy(), "diagnostics": arrays, "config": config}


def make_history_mask(diagnostics, mode, *, query_step=-1, interval_frames=16, seed=42):
    """Choose a top-mass or equally long random interval from the original read.

    Return (allowed_keys, metadata). None means the requested comparison is
    unavailable, never an unlabelled no-op. Random intervals may overlap the
    selected top interval but cannot equal it, and must contain eligible keys.
    """
    attention = np.asarray(diagnostics["attention"])
    valid = np.asarray(diagnostics["attention_valid"], dtype=bool)
    if attention.ndim != 2 or attention.shape != valid.shape or attention.shape[0] != attention.shape[1]:
        raise ValueError("Expected a square trajectory attention matrix and matching mask.")
    size = len(attention)
    if isinstance(query_step, bool) or not isinstance(query_step, (int, np.integer)):
        raise ValueError("query_step must be an integer.")
    query = size - 1 if query_step == -1 else int(query_step)
    if not 0 <= query < size:
        raise ValueError("query_step is outside the trajectory.")
    if mode not in ("none", "top_mass", "random_matched"):
        raise ValueError("mask_mode must be none, top_mass or random_matched.")
    metadata = {"mask_mode": mode, "query_step": query, "interval_frames": interval_frames,
                "seed": int(seed), "available": True}
    allowed = np.ones(size, dtype=bool)
    if mode == "none":
        return allowed, metadata
    intervals = summarize_history_intervals(attention[query], valid[query],
                                            interval_frames=interval_frames, num_intervals=1)
    if not intervals:
        return None, {**metadata, "available": False, "reason": "No nonzero historical attention mass."}
    top = intervals[0]
    start, end = top["start"], top["end"]
    metadata["top_interval"] = top
    if mode == "random_matched":
        length = end - start
        candidates = [i for i in range(query - length + 1)
                      if i != start and valid[query, i:i + length].all()]
        if not candidates:
            return None, {**metadata, "available": False, "reason": "No distinct equal-length random interval."}
        start = int(np.random.default_rng(seed).choice(candidates))
        end = start + length
    allowed[start:end] = False
    return allowed, {**metadata, "start": start, "end": end}


def render_intervention_comparison(reference, intervention, output_path, *, title="Fixed-Gaussian intervention"):
    """Plot already computed results; no model or Gaussian calls while rendering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = np.arange(len(reference["embeddings"]))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(steps, np.linalg.norm(intervention["embeddings"] - reference["embeddings"], axis=-1))
    axes[0].set_ylabel("Embedding L2 change")
    for result, label, style in ((reference, "Original", "-"), (intervention, "Intervention", "--")):
        axes[1].plot(steps, result["diagnostics"]["branch_weights"][:, 0], style, label=label)
        if "progress_mean" in result:
            axes[2].plot(steps, result["progress_mean"], style, label=label + " raw")
        if "progress_gated_raw" in result:
            axes[2].plot(steps, result["progress_gated_raw"], ":", label=label + " online gated")
    axes[1].set_ylabel("Actual long weight")
    axes[1].set_ylim(0, 1)
    axes[2].set_ylabel("Fixed-Gaussian progress")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Original frame index")
    axes[1].legend()
    if axes[2].lines:
        axes[2].legend()
    fig.suptitle(title)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
