"""
Compute the arithmetic mean embedding path across all videos in an H5 embedding file.

Algorithm
---------
1. Load embeddings [T_i, D] for every video.
2. Compute target length  K = round(mean(T_i)).
3. For each video, normalise time to [0, 1] and linearly interpolate all D
   dimensions to K equally-spaced steps.
4. Average the resampled embeddings across all videos -> [K, D].
5. Write a new H5 file (same schema as extract_embeddings.py output) containing
   one "video" entry (video_id = "mean") with the averaged path plus dispersion
   and cumulative-latent-distance progress diagnostics.

Usage
-----
    python scripts/compute_mean_embedding_path.py --input <path-to-embd.h5>
    python scripts/compute_mean_embedding_path.py --input <path-to-embd.h5> --output <out.h5>
"""

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_run_name_v2(embedding_ref: str | None = None) -> str | None:
    """[v2] Get run_name from configs_v2/runs.yaml via ConfigV2."""
    if not embedding_ref:
        return None
    try:
        import sys as _sys
        if str(_PROJECT_ROOT) not in _sys.path:
            _sys.path.insert(0, str(_PROJECT_ROOT))
        from utils.config_v2 import ConfigV2
        return ConfigV2().resolve_embedding(embedding_ref).get("run_name")
    except Exception:
        return None


def _infer_refs_from_input(
    input_h5: str,
    embedding_ref: str | None = None,
) -> tuple[str | None, str | None]:
    """[v2] Auto-detect (run_ref, dataset_ref) for --register from the registry.

    Strategy
    --------
    1. If *embedding_ref* is provided, resolve it directly via ConfigV2 and
       return the (run_ref, dataset_ref) stored in its registry entry.
    2. Otherwise, derive both refs by scanning the registry:
       a. run_ref   — find the runs.yaml entry whose ``run_name`` matches the
                      parent directory of *input_h5* (the per-run embedding dir).
       b. dataset_ref — find the datasets.yaml entry whose ``processed_h5``
                        stem matches the stem of *input_h5* after stripping
                        known embedding suffixes (-embd, -embd-mean_path, …).

    Returns (run_ref, dataset_ref); either may be None if not found.
    """
    import sys as _sys
    if str(_PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_PROJECT_ROOT))
    try:
        from utils.config_v2 import ConfigV2
        cfg = ConfigV2()
    except Exception as exc:
        print(f"[_infer_refs_from_input] WARNING: could not load ConfigV2: {exc}")
        return None, None

    # ── Strategy 1: resolve via embedding_ref ────────────────────────────────
    if embedding_ref:
        try:
            entry = cfg._embeddings.get(embedding_ref, {})
            run_ref    = entry.get("run_ref")
            dataset_ref = entry.get("dataset_ref")
            if run_ref and dataset_ref:
                print(
                    f"[_infer_refs_from_input] Resolved from embedding_ref '{embedding_ref}': "
                    f"run_ref='{run_ref}'  dataset_ref='{dataset_ref}'"
                )
                return run_ref, dataset_ref
        except Exception as exc:
            print(f"[_infer_refs_from_input] WARNING: embedding_ref lookup failed: {exc}")

    # ── Strategy 2: scan registry by path ────────────────────────────────────
    input_p   = Path(input_h5)
    run_name  = input_p.parent.name   # parent dir = per-run embedding folder

    # Strip known embedding suffixes to recover the dataset processed_h5 stem
    stem = input_p.stem
    for sfx in ("-embd-mean_path", "-embd-labeled", "-embd"):
        if stem.endswith(sfx):
            stem = stem[: -len(sfx)]
            break
    # stem is now e.g. "robomimic_can_ph-180vid_train"

    # run_ref: find the runs entry whose run_name matches the directory name
    run_ref = None
    for alias, run_entry in cfg._runs.items():
        if run_entry.get("run_name") == run_name:
            run_ref = alias
            break

    # dataset_ref: find the dataset whose processed_h5 stem matches
    dataset_ref = None
    for alias, ds_entry in cfg._datasets.items():
        if Path(ds_entry.get("processed_h5", "")).stem == stem:
            dataset_ref = alias
            break

    if run_ref or dataset_ref:
        print(
            f"[_infer_refs_from_input] Auto-detected from path "
            f"(run_name='{run_name}', stem='{stem}'): "
            f"run_ref={run_ref!r}  dataset_ref={dataset_ref!r}"
        )
    else:
        print(
            f"[_infer_refs_from_input] WARNING: could not find registry entries for "
            f"run_name='{run_name}' / stem='{stem}' — "
            "pass --run_ref and --dataset_ref explicitly."
        )
    return run_ref, dataset_ref


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_paths(
    input_path: str, output_path: str | None, plot_output: str | None
):
    """Return (input_h5, output_h5, plot_png).

    output_h5 defaults to <input_stem>-mean_path.h5 in the same directory as input.
    plot_png  defaults to outputs/mean_path/<run_name>/<output_h5_stem>-cumdist_progress.png
              where run_name is derived from the parent directory name of input_h5.
    """
    if not input_path:
        raise ValueError("--input is required")

    if not output_path:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}-mean_path.h5")
        print(f"[resolve_paths] output  (default): {output_path}")

    if not plot_output:
        # Derive run_name from the parent folder of the input H5
        run_name = Path(input_path).parent.name
        out_stem = Path(output_path).stem
        plot_output = str(
            _PROJECT_ROOT / "outputs" / "mean_path" / run_name
            / f"{out_stem}-cumdist_progress.png"
        )
        print(f"[resolve_paths] plot    (default): {plot_output}")

    return input_path, output_path, plot_output


# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------

def load_all_embeddings(h5_path: str, video_root: str = "videos", embd_key: str = "embeddings"):
    """Return list of (video_id, embeddings [T_i, D]) pairs, with full input validation."""
    # Check file exists
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"Input H5 file not found: {h5_path}")

    records = []
    first_D = None
    dim_errors = []

    with h5py.File(h5_path, "r") as f:
        # Check video_root group exists
        if video_root not in f:
            raise ValueError(
                f"Group '{video_root}' not found in H5 file. "
                f"Top-level keys: {list(f.keys())}"
            )

        grp = f[video_root]
        for vid_id in grp.keys():
            vid_grp = grp[vid_id]

            # Check embd_key dataset exists
            if embd_key not in vid_grp:
                print(f"[load_all_embeddings] WARNING: '{embd_key}' not found in "
                      f"video '{vid_id}' — skipping. Keys: {list(vid_grp.keys())}")
                continue

            emb = vid_grp[embd_key][:]  # [T_i, D]

            # Check 2D
            if emb.ndim != 2:
                print(f"[load_all_embeddings] WARNING: embeddings for '{vid_id}' "
                      f"have unexpected shape {emb.shape} (expected 2D) — skipping.")
                continue

            T, D = emb.shape

            # Check T >= 1
            if T < 1:
                print(f"[load_all_embeddings] WARNING: embeddings for '{vid_id}' "
                      f"have T={T} < 1 — skipping.")
                continue

            # Check latent dim consistency
            if first_D is None:
                first_D = D
            elif D != first_D:
                dim_errors.append((vid_id, emb.shape))
                continue

            records.append((vid_id, emb))

    if dim_errors:
        details = ", ".join(f"{vid}: {sh}" for vid, sh in dim_errors)
        raise ValueError(
            f"Inconsistent latent dimension (expected D={first_D}): {details}"
        )

    if not records:
        raise ValueError("No valid video embeddings found in the H5 file.")

    return records


def interpolate_to_length(emb: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a [T, D] embedding array to [target_len, D] via linear interpolation.

    Time is normalised to [0, 1] for both source and target grids so videos of
    different lengths are compared on a common relative-progress axis.
    """
    if target_len <= 0:
        raise ValueError(f"target_len must be >= 1, got {target_len}")

    T, D = emb.shape

    if T == target_len:
        return emb.astype(np.float32)

    # Single-frame video: broadcast the one frame
    if T == 1:
        return np.repeat(emb.astype(np.float32), target_len, axis=0)

    src_t = np.linspace(0.0, 1.0, T)
    dst_t = np.linspace(0.0, 1.0, target_len)

    out = np.empty((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(dst_t, src_t, emb[:, d])
    return out


def compute_mean_path(
    records: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (mean_emb [K, D], resampled [num_videos, K, D], metadata dict).

    metadata keys: num_videos, K, D, length_min, length_max, length_mean.
    """
    lengths = [emb.shape[0] for _, emb in records]
    K = int(round(np.mean(lengths)))
    D = records[0][1].shape[1]

    print(
        f"[compute_mean_path] {len(records)} videos  |  "
        f"length range [{min(lengths)}, {max(lengths)}]  |  "
        f"mean={np.mean(lengths):.2f}  ->  K={K}  |  D={D}"
    )

    resampled = np.stack(
        [interpolate_to_length(emb, K) for _, emb in records],
        axis=0,
    )  # [num_videos, K, D]

    mean_emb = resampled.mean(axis=0)  # [K, D]

    metadata = {
        "num_videos": len(records),
        "K": K,
        "D": D,
        "length_min": int(min(lengths)),
        "length_max": int(max(lengths)),
        "length_mean": float(np.mean(lengths)),
    }
    return mean_emb, resampled, metadata


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------

def compute_basic_dispersion(
    resampled: np.ndarray, mean_emb: np.ndarray
) -> np.ndarray:
    """Return dispersion_rms_l2 shape [K].

    dispersion_rms_l2[k] = RMS of per-video L2 distances to mean_emb at step k.

    Steps:
        diff    = resampled - mean_emb[None, :, :]   [num_videos, K, D]
        l2_dist = ||diff||_2 along D axis             [num_videos, K]
        rms_l2  = sqrt(mean(l2_dist**2, axis=0))      [K]
    """
    diff = resampled - mean_emb[None, :, :]           # [num_videos, K, D]
    l2_dist = np.linalg.norm(diff, axis=2)            # [num_videos, K]
    dispersion_rms_l2 = np.sqrt(np.mean(l2_dist ** 2, axis=0))  # [K]

    K = dispersion_rms_l2.shape[0]
    mean_d   = float(np.mean(dispersion_rms_l2))
    max_d    = float(np.max(dispersion_rms_l2))
    argmax_k = int(np.argmax(dispersion_rms_l2))
    argmax_progress = argmax_k / (K - 1) if K > 1 else 0.0

    print(
        f"[compute_basic_dispersion] RMS L2 dispersion: "
        f"mean={mean_d:.4f}  max={max_d:.4f}  "
        f"argmax_step={argmax_k}  argmax_progress={argmax_progress:.4f}"
    )
    return dispersion_rms_l2


# ---------------------------------------------------------------------------
# Cumulative latent distance progress proxy
# ---------------------------------------------------------------------------

def compute_cumulative_latent_distance_progress(
    path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cum_dist [K], norm_cum_dist [K]) for a single latent trajectory.

    step_dist[0]  = 0
    step_dist[k]  = ||path[k] - path[k-1]||_2  for k >= 1
    cum_dist      = cumsum(step_dist)
    norm_cum_dist = cum_dist / cum_dist[-1]  (or zeros if total distance == 0)
    """
    K, _ = path.shape
    step_dist = np.zeros(K, dtype=np.float64)
    if K > 1:
        diffs = path[1:] - path[:-1]                      # [K-1, D]
        step_dist[1:] = np.linalg.norm(diffs, axis=1)     # [K-1]

    cum_dist = np.cumsum(step_dist)

    if cum_dist[-1] > 0:
        norm_cum_dist = cum_dist / cum_dist[-1]
    else:
        norm_cum_dist = np.zeros_like(cum_dist)

    return cum_dist.astype(np.float32), norm_cum_dist.astype(np.float32)


def compute_all_cumdist_progress(
    resampled: np.ndarray, mean_emb: np.ndarray
) -> dict:
    """Compute cumulative latent distance progress proxies for all demos + mean path.

    Returns dict with keys:
        all_cum_dist            [num_videos, K]  raw cumulative L2 distance
        all_cum_dist_mean       [K]              mean across videos
        all_norm_cum_dist       [num_videos, K]  normalized to [0, 1]
        all_norm_cum_dist_mean  [K]
        all_norm_cum_dist_std   [K]
        mean_path_cum_dist      [K]              raw cumulative L2 of mean path
        mean_path_norm_cum_dist [K]
    """
    num_videos, K, _ = resampled.shape

    all_cum  = np.empty((num_videos, K), dtype=np.float32)
    all_norm = np.empty((num_videos, K), dtype=np.float32)
    for i in range(num_videos):
        cum, norm = compute_cumulative_latent_distance_progress(resampled[i])
        all_cum[i]  = cum
        all_norm[i] = norm

    mean_path_cum, mean_path_norm = compute_cumulative_latent_distance_progress(mean_emb)

    return {
        "all_cum_dist":            all_cum,
        "all_cum_dist_mean":       all_cum.mean(axis=0).astype(np.float32),
        "all_norm_cum_dist":       all_norm,
        "all_norm_cum_dist_mean":  all_norm.mean(axis=0).astype(np.float32),
        "all_norm_cum_dist_std":   all_norm.std(axis=0).astype(np.float32),
        "mean_path_cum_dist":      mean_path_cum,
        "mean_path_norm_cum_dist": mean_path_norm,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_cumdist_progress(cumdist_data: dict, K: int, plot_path: str) -> None:
    """Save a PNG: time progress vs raw cumulative latent distance."""
    time_progress = np.linspace(0.0, 1.0, K)

    all_cum      = cumdist_data["all_cum_dist"]         # [num_videos, K]
    demo_avg_cum = cumdist_data["all_cum_dist_mean"]    # [K]
    mean_path_cd = cumdist_data["mean_path_cum_dist"]   # [K]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Individual demo curves
    for i in range(all_cum.shape[0]):
        ax.plot(time_progress, all_cum[i], color="steelblue",
                alpha=0.15, linewidth=0.7)

    # Demo average
    ax.plot(time_progress, demo_avg_cum, color="steelblue", linewidth=2.0,
            label="demo average")

    # Mean latent path's own cumulative distance
    ax.plot(time_progress, mean_path_cd, color="tomato", linewidth=2.0,
            linestyle="--", label="mean path")

    # Phantom line for individual-demo legend entry
    ax.plot([], [], color="steelblue", alpha=0.5, linewidth=0.7,
            label="individual demos")

    ax.set_xlabel("Normalized time progress")
    ax.set_ylabel("Cumulative latent L2 distance")
    ax.set_title("Cumulative Latent Distance vs Time Progress")
    ax.legend(loc="upper left")
    ax.set_xlim(0.0, 1.0)

    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[plot_cumdist_progress] Saved plot -> {plot_path}")


def plot_cumdist_progress_dual(cumdist_data: dict, K: int, plot_path: str) -> None:
    """Save a side-by-side PNG: absolute cumulative latent dist (left) and
    normalised cumulative latent dist 0→1 (right), both vs time progress.

    Each panel shows individual demo curves (faint), the per-demo mean
    (solid blue), and the mean latent path (dashed red).
    The normalised panel also shows a ±1σ band around the demo mean.
    """
    time_progress = np.linspace(0.0, 1.0, K)

    all_cum       = cumdist_data["all_cum_dist"]            # [num_videos, K]
    demo_avg_cum  = cumdist_data["all_cum_dist_mean"]       # [K]
    mean_path_cd  = cumdist_data["mean_path_cum_dist"]      # [K]

    all_norm      = cumdist_data["all_norm_cum_dist"]        # [num_videos, K]
    demo_avg_norm = cumdist_data["all_norm_cum_dist_mean"]   # [K]
    demo_std_norm = cumdist_data["all_norm_cum_dist_std"]    # [K]
    mean_path_norm= cumdist_data["mean_path_norm_cum_dist"]  # [K]

    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: absolute cumulative distance ───────────────────────────────────
    for i in range(all_cum.shape[0]):
        ax_abs.plot(time_progress, all_cum[i], color="steelblue",
                    alpha=0.12, linewidth=0.6)
    ax_abs.plot([], [], color="steelblue", alpha=0.5, linewidth=0.7,
                label="individual demos")
    ax_abs.plot(time_progress, demo_avg_cum, color="steelblue", linewidth=2.0,
                label="demo average")
    ax_abs.plot(time_progress, mean_path_cd, color="tomato", linewidth=2.0,
                linestyle="--", label="mean path")
    ax_abs.set_xlabel("Normalized time progress")
    ax_abs.set_ylabel("Cumulative latent L2 distance")
    ax_abs.set_title("Absolute Cumulative Latent Distance")
    ax_abs.legend(loc="upper left", fontsize=8)
    ax_abs.set_xlim(0.0, 1.0)

    # ── right: normalised (0→1) cumulative distance ───────────────────────────
    for i in range(all_norm.shape[0]):
        ax_norm.plot(time_progress, all_norm[i], color="steelblue",
                     alpha=0.12, linewidth=0.6)
    ax_norm.fill_between(
        time_progress,
        demo_avg_norm - demo_std_norm,
        demo_avg_norm + demo_std_norm,
        color="steelblue", alpha=0.18, label="demo mean ±σ",
    )
    ax_norm.plot([], [], color="steelblue", alpha=0.5, linewidth=0.7,
                 label="individual demos")
    ax_norm.plot(time_progress, demo_avg_norm, color="steelblue", linewidth=2.0,
                 label="demo average")
    ax_norm.plot(time_progress, mean_path_norm, color="tomato", linewidth=2.0,
                 linestyle="--", label="mean path")
    # diagonal reference (uniform progress proxy)
    ax_norm.plot([0, 1], [0, 1], color="gray", linewidth=1.0,
                 linestyle=":", label="uniform (diagonal)")
    ax_norm.set_xlabel("Normalized time progress")
    ax_norm.set_ylabel("Normalized cumulative latent L2 distance")
    ax_norm.set_title("Normalised Cumulative Latent Distance (0→1)")
    ax_norm.legend(loc="upper left", fontsize=8)
    ax_norm.set_xlim(0.0, 1.0)
    ax_norm.set_ylim(0.0, 1.0)

    fig.suptitle(
        "Cumulative Latent Distance vs Time Progress",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    dual_path = Path(plot_path).with_stem(Path(plot_path).stem + "-dual")
    dual_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(dual_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_cumdist_progress_dual] Saved plot -> {dual_path}")

# Hardcoded overrides for the mean-path t-SNE plot. These values are specific
# to this visualization and intentionally do not live in visualize YAML.
_TSNE_MEAN_MARKER       = "D"    # diamond to distinguish mean path from training
_TSNE_MEAN_MARKER_SCALE = 4      # mean path point_size = plot.point_size * this
_TSNE_MEAN_ALPHA        = 1.0
_TSNE_MEAN_ZORDER       = 10
_TSNE_MEAN_EDGE_COLOR   = "darkred"
_TSNE_MEAN_EDGE_LW      = 0.4
_TSNE_TRAIN_CMAP        = "Blues"   # override cmap_progress_group1 if absent
_TSNE_MEAN_CMAP         = "Reds"    # mean-path progress colormap


# ---------------------------------------------------------------------------
# t-SNE: training frames + mean path
# ---------------------------------------------------------------------------


def plot_tsne_with_mean_path(
    records: list,
    mean_emb: np.ndarray,
    metadata: dict,
    output_h5: str,
    viz_cfg: dict | None = None,
) -> str | None:
    """Run t-SNE on (sampled) training frames + mean path, then save a PNG.

    Training frames are coloured blue (Blues cmap, by temporal progress).
    The mean path is coloured red (Reds cmap, larger diamond markers).
    Both are projected into a **shared** 2-D space.

    Parameters
    ----------
    records    : list of (video_id, emb [T_i, D]) as returned by load_all_embeddings
    mean_emb   : [K, D] mean latent path
    metadata   : dict from compute_mean_path (used for logging)
    output_h5  : path to the output H5 file (used to derive the output directory)

    Returns
    -------
    Path to the saved PNG, or None if skipped.
    """
    if viz_cfg is not None:
        cfg = viz_cfg
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("[plot_tsne_with_mean_path] No V2 visualize config provided — skipping t-SNE.")
        return None, None

    # ── build training embeddings array (all frames, then sample) ─────────────
    train_emb_list  = []
    train_prog_list = []
    train_vid_list  = []
    for vid_id, emb in records:
        T = len(emb)
        progress = np.linspace(0.0, 1.0, T)
        train_emb_list.append(emb)
        train_prog_list.append(progress)
        train_vid_list.extend([vid_id] * T)

    train_emb  = np.concatenate(train_emb_list,  axis=0).astype(np.float32)
    train_prog = np.concatenate(train_prog_list, axis=0)
    train_vids = np.array(train_vid_list)

    # ── sample training frames ─────────────────────────────────────────────────
    rng           = np.random.default_rng(cfg.get("random_seed", 42))
    max_per_video = cfg.get("max_frames_per_video", 300)
    max_total     = cfg.get("max_total_frames", 10000)

    sel_idx = []
    for vid in np.unique(train_vids):
        idx = np.where(train_vids == vid)[0]
        if len(idx) > max_per_video:
            idx = rng.choice(idx, size=max_per_video, replace=False)
        sel_idx.append(idx)
    sel_idx = np.concatenate(sel_idx)
    if len(sel_idx) > max_total:
        sel_idx = rng.choice(sel_idx, size=max_total, replace=False)

    train_emb_s  = train_emb[sel_idx]
    train_prog_s = train_prog[sel_idx]
    train_vids_s = train_vids[sel_idx]
    n_train      = len(train_emb_s)

    # ── mean path progress ─────────────────────────────────────────────────────
    K         = len(mean_emb)
    mean_prog = np.linspace(0.0, 1.0, K)

    # ── joint preprocessing ───────────────────────────────────────────────────
    emb_all = np.concatenate([train_emb_s, mean_emb.astype(np.float32)], axis=0)
    n_total = len(emb_all)
    print(
        f"[plot_tsne_with_mean_path] Total frames for t-SNE: {n_total} "
        f"(train={n_train}, mean_path={K})"
    )

    if cfg.get("standardize", True):
        from sklearn.preprocessing import StandardScaler
        scaler  = StandardScaler()
        emb_all = scaler.fit_transform(emb_all)

    if cfg.get("use_pca_before_tsne", True):
        from sklearn.decomposition import PCA
        pca_dim     = cfg.get("pca_dim", 50)
        n_comp      = min(pca_dim, emb_all.shape[1], n_total - 1)
        pca         = PCA(n_components=n_comp)
        emb_all     = pca.fit_transform(emb_all)
        print(
            f"[plot_tsne_with_mean_path] PCA → {n_comp} dims "
            f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})"
        )

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    from sklearn.manifold import TSNE

    tcfg           = cfg.get("tsne", {})
    config_perp    = tcfg.get("perplexity", 30)
    perplexity     = min(config_perp, max(5, (n_total - 1) // 3))
    if perplexity != config_perp:
        print(
            f"[plot_tsne_with_mean_path] Adjusted perplexity "
            f"{config_perp} → {perplexity} (N={n_total})"
        )

    tsne_kwargs = dict(
        n_components  = tcfg.get("n_components", 2),
        perplexity    = perplexity,
        learning_rate = tcfg.get("learning_rate", "auto"),
        init          = tcfg.get("init", "pca"),
        random_state  = cfg.get("random_seed", 42),
    )
    max_iter = tcfg.get("max_iter", 1000)
    print("[plot_tsne_with_mean_path] Running t-SNE ...")
    try:
        coords = TSNE(**tsne_kwargs, max_iter=max_iter).fit_transform(emb_all)
    except TypeError:
        coords = TSNE(**tsne_kwargs, n_iter=max_iter).fit_transform(emb_all)
    print(f"[plot_tsne_with_mean_path] t-SNE done. Perplexity used: {perplexity}")

    coords_train = coords[:n_train]
    coords_mean  = coords[n_train:]

    # ── plot ──────────────────────────────────────────────────────────────────
    pcfg    = cfg.get("plot", {})
    figsize = pcfg.get("figsize", [9, 6])
    dpi     = pcfg.get("dpi", 300)
    s       = pcfg.get("point_size", 5)
    alpha   = pcfg.get("alpha", 0.75)
    # Hardcoded overrides: always use Blues for training, Reds for mean path
    cmap_train = _TSNE_TRAIN_CMAP
    cmap_mean  = _TSNE_MEAN_CMAP

    label1 = cfg.get("group1_label", "Training")

    fig, ax = plt.subplots(figsize=figsize)

    sc_train = ax.scatter(
        coords_train[:, 0], coords_train[:, 1],
        c=train_prog_s, cmap=cmap_train, vmin=0.0, vmax=1.0,
        s=s, alpha=alpha, zorder=1,
    )
    sc_mean = ax.scatter(
        coords_mean[:, 0], coords_mean[:, 1],
        c=mean_prog, cmap=cmap_mean, vmin=0.0, vmax=1.0,
        s=s * _TSNE_MEAN_MARKER_SCALE, alpha=_TSNE_MEAN_ALPHA,
        marker=_TSNE_MEAN_MARKER, zorder=_TSNE_MEAN_ZORDER,
        edgecolors=_TSNE_MEAN_EDGE_COLOR, linewidths=_TSNE_MEAN_EDGE_LW,
    )

    cb_train = plt.colorbar(sc_train, ax=ax, fraction=0.040, pad=0.02)
    cb_train.set_label(f"{label1} Progress")
    cb_mean = plt.colorbar(sc_mean, ax=ax, fraction=0.040, pad=0.10)
    cb_mean.set_label("Mean Path Progress")

    ax.set_title(
        f"t-SNE: {label1} (blue) + Mean Path (red ◆) — Temporal Progress\n"
        f"{metadata['num_videos']} training videos · K={K} mean-path steps"
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=plt.get_cmap(cmap_train)(0.65), markersize=6,
            label=f"{label1} ({n_train} frames sampled from {metadata['num_videos']} videos)",
        ),
        Line2D(
            [0], [0], marker=_TSNE_MEAN_MARKER, color="w",
            markerfacecolor=plt.get_cmap(cmap_mean)(0.7),
            markeredgecolor=_TSNE_MEAN_EDGE_COLOR,
            markeredgewidth=_TSNE_MEAN_EDGE_LW,
            markersize=8,
            label=f"Mean Path ({K} steps)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = output_dir / f"tsne_mean_path_{timestamp}.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_tsne_with_mean_path] Saved → {out_path}")

    tsne_data = dict(
        coords           = coords,
        n_train          = n_train,
        train_vids_s     = train_vids_s,
        train_prog_s     = train_prog_s,
        mean_prog        = mean_prog,
        output_dir       = output_dir,
        cfg              = cfg,
        metadata         = metadata,
        timestamp        = timestamp,
        vid_frame_counts = {vid_id: len(emb) for vid_id, emb in records},
    )
    return str(out_path), tsne_data


def plot_latent_path_curves_on_tsne(
    coords,
    n_train,
    train_vids_s,
    train_prog_s,
    mean_prog,
    output_dir,
    cfg,
    metadata,
    timestamp,
    vid_frame_counts,
) -> str:
    """Plot three randomly-selected demo trajectories + mean path as gradient curves.

    Only demos whose original frame count exceeds the mean frame count across
    all training videos are eligible for selection.  Each demo is drawn with a
    distinct blue-family colormap; the mean path uses the Reds colormap.
    All paths are projected in the shared t-SNE space from
    plot_tsne_with_mean_path.

    Parameters are the items returned inside ``tsne_data`` by
    plot_tsne_with_mean_path.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    # Three distinct blue-family cmaps, one per selected demo
    _DEMO_CMAPS = ["Blues", "PuBu", "GnBu"]
    N_DEMOS     = 3

    coords_train = coords[:n_train]   # [n_train, 2]
    coords_mean  = coords[n_train:]   # [K, 2]
    K = len(coords_mean)

    # ── select demos with frame count > mean frame count ──────────────────────
    mean_len    = metadata["length_mean"]
    eligible    = sorted(
        vid for vid in np.unique(train_vids_s)
        if vid_frame_counts.get(vid, 0) > mean_len
    )
    if not eligible:
        print(
            "[plot_latent_path_curves_on_tsne] No eligible demos "
            f"(frame count > mean {mean_len:.1f}) — using all demos."
        )
        eligible = sorted(np.unique(train_vids_s).tolist())

    rng = np.random.default_rng(cfg.get("random_seed", 42))
    n_pick = min(N_DEMOS, len(eligible))
    selected_vids = rng.choice(eligible, size=n_pick, replace=False).tolist()
    print(
        f"[plot_latent_path_curves_on_tsne] Eligible demos (>{mean_len:.1f} frames): "
        f"{len(eligible)}  |  Selected: {selected_vids}"
    )

    # ── helper: build a gradient LineCollection ───────────────────────────────
    def _make_lc(xy, progress, cmap_name, lw=2.0, alpha=0.9, zorder=2):
        pts  = xy.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)   # [N-1, 2, 2]
        mid  = (progress[:-1] + progress[1:]) / 2.0
        lc   = LineCollection(
            segs, cmap=cmap_name,
            norm=plt.Normalize(0.0, 1.0),
            linewidth=lw, alpha=alpha, zorder=zorder,
        )
        lc.set_array(mid)
        return lc

    # ── figure ────────────────────────────────────────────────────────────────
    pcfg    = cfg.get("plot", {})
    figsize = pcfg.get("figsize", [9, 6])
    dpi     = pcfg.get("dpi", 300)
    s       = pcfg.get("point_size", 5)
    label1  = cfg.get("group1_label", "Training")

    fig, ax = plt.subplots(figsize=figsize)

    legend_elements = []
    last_sc_demo = None
    for i, vid in enumerate(selected_vids):
        cmap_demo   = _DEMO_CMAPS[i % len(_DEMO_CMAPS)]
        demo_mask   = train_vids_s == vid
        demo_coords = coords_train[demo_mask]
        demo_prog   = train_prog_s[demo_mask]

        # Sort by progress so line follows temporal order
        order       = np.argsort(demo_prog)
        demo_coords = demo_coords[order]
        demo_prog   = demo_prog[order]

        zorder_lc = 2 + i * 2
        zorder_sc = zorder_lc + 1

        if len(demo_coords) >= 2:
            lc = _make_lc(demo_coords, demo_prog, cmap_demo,
                          lw=2.0, alpha=0.88, zorder=zorder_lc)
            ax.add_collection(lc)

        sc = ax.scatter(
            demo_coords[:, 0], demo_coords[:, 1],
            c=demo_prog, cmap=cmap_demo, vmin=0.0, vmax=1.0,
            s=s, alpha=0.9, zorder=zorder_sc,
        )
        last_sc_demo = sc
        n_frames = vid_frame_counts.get(vid, len(demo_coords))
        legend_elements.append(
            Line2D(
                [0], [0], color=plt.get_cmap(cmap_demo)(0.7), linewidth=2.0,
                label=f"{label1} demo: {vid} ({n_frames} frames)",
            )
        )

    # Mean path
    if len(coords_mean) >= 2:
        lc_mean = _make_lc(coords_mean, mean_prog, _TSNE_MEAN_CMAP,
                            lw=2.5, alpha=1.0, zorder=2 + n_pick * 2)
        ax.add_collection(lc_mean)
    s_mean  = s * _TSNE_MEAN_MARKER_SCALE
    sc_mean = ax.scatter(
        coords_mean[:, 0], coords_mean[:, 1],
        c=mean_prog, cmap=_TSNE_MEAN_CMAP, vmin=0.0, vmax=1.0,
        s=s_mean, alpha=_TSNE_MEAN_ALPHA,
        marker=_TSNE_MEAN_MARKER, zorder=2 + n_pick * 2 + 1,
        edgecolors=_TSNE_MEAN_EDGE_COLOR, linewidths=_TSNE_MEAN_EDGE_LW,
    )

    ax.autoscale()

    if last_sc_demo is not None:
        cb_demo = plt.colorbar(last_sc_demo, ax=ax, fraction=0.040, pad=0.02)
        cb_demo.set_label(f"{label1} Demo Progress")
    cb_mean = plt.colorbar(sc_mean, ax=ax, fraction=0.040, pad=0.10)
    cb_mean.set_label("Mean Path Progress")

    legend_elements.append(
        Line2D(
            [0], [0], color=plt.get_cmap(_TSNE_MEAN_CMAP)(0.7), linewidth=2.5,
            label=f"Mean Path ({K} steps)",
        )
    )
    ax.set_title(
        f"Latent Paths in t-SNE Space  ({n_pick} demos, frame count > mean {mean_len:.0f})\n"
        f"Blue-family: selected demos  |  Red: Mean Path (◆)  ·  "
        f"{metadata['num_videos']} training videos, K={K}"
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

    out_path = output_dir / f"tsne_latent_paths_{timestamp}.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_latent_path_curves_on_tsne] Saved → {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# H5 output
# ---------------------------------------------------------------------------

def save_results(
    output_path: str,
    mean_emb: np.ndarray,
    dispersion_rms_l2: np.ndarray,
    cumdist_data: dict,
    metadata: dict,
    plot_path: str,
    video_root: str = "videos",
) -> None:
    """Write all results to H5 using the same schema as extract_embeddings.py."""
    K, D = mean_emb.shape
    target_steps = np.arange(K, dtype=np.int64)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        grp = f.require_group(f"{video_root}/mean")

        # Core mean path
        grp.create_dataset("embeddings",   data=mean_emb.astype(np.float32))
        grp.create_dataset("target_steps", data=target_steps)

        # Dispersion
        grp.create_dataset("dispersion_rms_l2",
                           data=dispersion_rms_l2.astype(np.float32))

        # Cumulative-distance diagnostics
        grp.create_dataset("mean_path_norm_cumdist_progress",
                           data=cumdist_data["mean_path_norm_cum_dist"])
        grp.create_dataset("demo_avg_norm_cumdist_progress",
                           data=cumdist_data["all_norm_cum_dist_mean"])
        grp.create_dataset("demo_std_norm_cumdist_progress",
                           data=cumdist_data["all_norm_cum_dist_std"])

        # Attrs
        grp.attrs["seq_len"]            = K
        grp.attrs["action_id"]          = 0
        grp.attrs["num_source_videos"]  = metadata["num_videos"]
        grp.attrs["source_length_min"]  = metadata["length_min"]
        grp.attrs["source_length_max"]  = metadata["length_max"]
        grp.attrs["source_length_mean"] = metadata["length_mean"]
        grp.attrs["latent_dim"]         = D
        grp.attrs["normalization"]      = "time"
        grp.attrs["interpolation"]      = "linear"
        grp.attrs["aggregation"]        = "arithmetic_mean"

        grp.attrs["dispersion_metric"]        = "rms_l2_to_mean_path"
        grp.attrs["dispersion_rms_l2_mean"]   = float(np.mean(dispersion_rms_l2))
        grp.attrs["dispersion_rms_l2_max"]    = float(np.max(dispersion_rms_l2))
        grp.attrs["dispersion_rms_l2_argmax"] = int(np.argmax(dispersion_rms_l2))

        grp.attrs["cumdist_progress_proxy"]     = "normalized_cumulative_l2_distance"
        grp.attrs["plot_cumdist_progress_path"] = str(plot_path)

    print(f"[save_results] Saved H5 ({K} steps, {D}-dim) -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute mean embedding path across all videos."
    )
    p.add_argument("--input", required=True,
                   help="Path to source embd.h5.")
    p.add_argument("--output", default=None,
                   help="Output .h5 path. Defaults to <input-stem>-mean_path.h5 in same dir.")
    p.add_argument("--video-root", default="videos",
                   help="Top-level group name inside the H5 file (default: 'videos').")
    p.add_argument("--embd-key", default="embeddings",
                   help="Dataset key for embeddings inside each video group "
                        "(default: 'embeddings').")
    p.add_argument("--plot-output", default=None,
                   help="Path to save cumulative latent distance progress plot. "
                        "Defaults to <output_h5_stem>-cumdist_progress.png.")
    p.add_argument("--no-tsne", action="store_true",
                   help="Skip t-SNE visualisation of training frames + mean path.")
    p.add_argument("--embedding_ref", default=None,
                   help="[v2] Embedding alias from configs_v2/runs.yaml (used to derive run_name for t-SNE output dir).")
    p.add_argument("--register", action="store_true", default=False,
                   help="[v2] Register the output mean_path embedding into configs_v2/registry/runs.yaml.")
    p.add_argument("--alias", default=None, dest="register_alias",
                   help="[v2] Registry alias for the embedding (auto-suggested if not set). Requires --register.")
    p.add_argument("--run_ref", default=None,
                   help="[v2] Run alias in configs_v2/registry/runs.yaml. "
                        "Auto-inferred from the input H5 path / --embedding_ref when --register is set.")
    p.add_argument("--dataset_ref", default=None,
                   help="[v2] Dataset alias in configs_v2/registry/datasets.yaml. "
                        "Auto-inferred from the input H5 path / --embedding_ref when --register is set.")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Resolve paths
    input_h5, output_h5, plot_png = resolve_paths(
        args.input, args.output, args.plot_output
    )

    # 2. Load embeddings
    print(f"[main] Loading embeddings from: {input_h5}")
    records = load_all_embeddings(
        input_h5, video_root=args.video_root, embd_key=args.embd_key
    )
    print(f"[main] Loaded {len(records)} videos")

    # 3. Compute mean path
    mean_emb, resampled, metadata = compute_mean_path(records)

    # 4. Dispersion
    dispersion_rms_l2 = compute_basic_dispersion(resampled, mean_emb)

    # 5. Cumulative latent distance progress proxy
    cumdist_data = compute_all_cumdist_progress(resampled, mean_emb)

    # 6. Save H5
    save_results(
        output_h5, mean_emb, dispersion_rms_l2, cumdist_data,
        metadata, plot_png, video_root=args.video_root,
    )

    # 7. Save plots
    plot_cumdist_progress(cumdist_data, metadata["K"], plot_png)
    plot_cumdist_progress_dual(cumdist_data, metadata["K"], plot_png)

    # 8. t-SNE: training frames + mean path  +  latent-path curves plot
    if not args.no_tsne:
        # [v2] Load mean_path visualize config
        _viz_cfg_v2 = None
        try:
            import sys as _sys
            if str(_PROJECT_ROOT) not in _sys.path:
                _sys.path.insert(0, str(_PROJECT_ROOT))
            from utils.config_v2 import ConfigV2
            _v2_overrides = {}
            _emb_ref_v2 = getattr(args, "embedding_ref", None)
            if _emb_ref_v2:
                _v2_overrides["embedding_ref"] = _emb_ref_v2
            _viz_cfg_v2 = ConfigV2().load_visualize(
                "mean_path", overrides=_v2_overrides or None
            )
            # Point output_dir to the mean_path sub-folder derived from H5 stem
            if _viz_cfg_v2.get("output_dir"):
                _viz_cfg_v2["output_dir"] = str(
                    Path(_viz_cfg_v2["output_dir"]) / Path(output_h5).stem
                )
        except Exception as _e:
            print(f"[main] V2 visualize config unavailable ({_e}); skipping t-SNE visualization.")
        _, tsne_data = plot_tsne_with_mean_path(
            records, mean_emb, metadata, output_h5, viz_cfg=_viz_cfg_v2
        )
        if tsne_data is not None:
            plot_latent_path_curves_on_tsne(**tsne_data)

    # [v2] Optional: register mean_path embedding into configs_v2/registry/runs.yaml
    if args.register:
        import sys as _sys
        if str(_PROJECT_ROOT) not in _sys.path:
            _sys.path.insert(0, str(_PROJECT_ROOT))
        from utils.registry_v2 import RegistryV2
        _reg = RegistryV2()

        # Auto-infer run_ref / dataset_ref from the input embedding path;
        # explicit CLI args take priority if provided.
        _auto_run_ref, _auto_dataset_ref = _infer_refs_from_input(
            input_h5, getattr(args, "embedding_ref", None)
        )
        _run_ref     = args.run_ref     or _auto_run_ref     or ""
        _dataset_ref = args.dataset_ref or _auto_dataset_ref or ""

        if not _run_ref or not _dataset_ref:
            print(
                f"[main] [v2] WARNING: --register skipped — could not determine "
                f"run_ref={_run_ref!r} or dataset_ref={_dataset_ref!r}.  "
                "Pass --run_ref / --dataset_ref explicitly."
            )
        else:
            _variant = "mean_path"
            _alias = args.register_alias or _reg.suggest_embedding_alias(
                _run_ref, _dataset_ref, _variant
            )
            _reg.register_embedding(
                alias       = _alias,
                run_ref     = _run_ref,
                dataset_ref = _dataset_ref,
                variant     = _variant,
                description = f"{_dataset_ref} mean_path embedding",
            )
            print(f"[main] [v2] Embedding registered as '{_alias}' in configs_v2/registry/runs.yaml")


if __name__ == "__main__":
    main()
