"""
Gap Analysis t-SNE visualization.

Loads two groups of TCC embeddings (train + val), runs t-SNE on the joint data,
then produces a combined MP4 video composed of:

  Left panel  — t-SNE scatter with only 4 selected videos highlighted (progress sweep)
  Right panel — 2×2 grid of the corresponding raw MP4 footage, frame-locked to progress

The 4 videos can be specified by raw filename in the config, or selected randomly
from the combined train/val pool.

Usage:
    cd /home/user/zhangzk/projects/fineprog
    python scripts/visualize_embeddings_tsne_gap_analysis.py \\
    --viz_config configs_v2/visualize/tsne_gap_analysis.yaml
"""

import argparse
import os
from datetime import datetime

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 4 visually distinct colours (tab10 indices 0-3): blue, orange, green, red
_SEL_COLORS = [
    (0.122, 0.467, 0.706),   # tab-blue
    (1.000, 0.498, 0.055),   # tab-orange
    (0.173, 0.627, 0.173),   # tab-green
    (0.839, 0.153, 0.157),   # tab-red
]

def _resolve_run_name_v2(embedding_ref: str | None = None) -> str | None:
    """[v2] Get run_name from configs_v2/runs.yaml via ConfigV2."""
    if not embedding_ref:
        return None
    try:
        import sys as _sys
        if _PROJECT_ROOT not in _sys.path:
            _sys.path.insert(0, _PROJECT_ROOT)
        from utils.config_v2 import ConfigV2
        return ConfigV2().resolve_embedding(embedding_ref).get("run_name")
    except Exception:
        return None


def _resolve_h5_v2(embedding_ref: str) -> str:
    """[v2] Resolve embedding H5 absolute path from runs.yaml registry."""
    import sys as _sys
    if _PROJECT_ROOT not in _sys.path:
        _sys.path.insert(0, _PROJECT_ROOT)
    from utils.config_v2 import ConfigV2
    return ConfigV2().resolve_embedding(embedding_ref)["embedding_h5_path"]


def resolve_h5_path(cfg: dict, stem_key: str, embedding_ref: str | None = None) -> str:
    """Return the embedding HDF5 path for *stem_key*.

    Priority:
    1. Explicit override ``embedding_h5_path_<stem_key>`` in cfg.
    2. [v2] *embedding_ref* argument: resolved via ConfigV2.
    """
    override_key = f"embedding_h5_path_{stem_key}"
    if cfg.get(override_key):
        return cfg[override_key]

    # [v2] Resolve via embedding registry
    if embedding_ref:
        h5_path = _resolve_h5_v2(embedding_ref)
        print(f"[resolve] [v2] {stem_key} → {h5_path}")
        return h5_path

    dataset_stem = cfg.get(stem_key)
    if not dataset_stem:
        raise ValueError(f"'{stem_key}' must be set in the config.")
    raise ValueError(
        f"Cannot resolve '{stem_key}': pass --embedding_ref / --embedding_ref_group2 "
        "or set embedding_h5_path override in config."
    )


# ── embedding loading ─────────────────────────────────────────────────────────

def load_embeddings_from_h5(h5_path: str, cfg: dict):
    """Load all embeddings from an HDF5 file.

    Returns
    -------
    embeddings   : ndarray [N, D]
    video_names  : ndarray [N]  str  (H5 video-group key, e.g. '000001')
    target_steps : ndarray [N]  int
    progress     : ndarray [N]  float 0-1
    mean_seq_len : float
    """
    video_root = cfg.get("h5_video_root", "videos")
    embd_key   = cfg.get("embedding_key", "embeddings")
    steps_key  = cfg.get("target_steps_key", "target_steps")

    all_embeddings, all_video_names, all_target_steps, all_progress, seq_lens = \
        [], [], [], [], []

    with h5py.File(h5_path, "r") as f:
        videos_grp = f[video_root]
        for video_id in videos_grp.keys():
            grp = videos_grp[video_id]
            embeddings   = grp[embd_key][:]
            target_steps = grp[steps_key][:]
            seq_len      = int(grp.attrs.get("seq_len", len(embeddings)))
            T            = len(embeddings)
            progress     = target_steps / max(seq_len - 1, 1)

            all_embeddings.append(embeddings)
            all_video_names.extend([video_id] * T)
            all_target_steps.append(target_steps)
            all_progress.append(progress)
            seq_lens.append(seq_len)

    all_embeddings   = np.concatenate(all_embeddings,   axis=0)
    all_target_steps = np.concatenate(all_target_steps, axis=0)
    all_progress     = np.concatenate(all_progress,     axis=0)
    all_video_names  = np.array(all_video_names)
    mean_seq_len     = float(np.mean(seq_lens)) if seq_lens else 100.0

    return all_embeddings, all_video_names, all_target_steps, all_progress, mean_seq_len


def sample_embeddings(embeddings, video_names, target_steps, progress, cfg):
    rng           = np.random.default_rng(cfg.get("random_seed", 42))
    max_per_video = cfg.get("max_frames_per_video", 300)
    max_total     = cfg.get("max_total_frames", 10000)

    unique_videos = np.unique(video_names)
    sel_indices   = []
    for vid in unique_videos:
        idx = np.where(video_names == vid)[0]
        if len(idx) > max_per_video:
            idx = rng.choice(idx, size=max_per_video, replace=False)
        sel_indices.append(idx)

    sel_indices = np.concatenate(sel_indices, axis=0)
    if len(sel_indices) > max_total:
        sel_indices = rng.choice(sel_indices, size=max_total, replace=False)

    return (
        embeddings[sel_indices],
        video_names[sel_indices],
        target_steps[sel_indices],
        progress[sel_indices],
    )


def load_selected_video_embeddings(h5_path: str, vid_ids: list, cfg: dict) -> dict:
    """Load full embeddings for specific video IDs from an HDF5 file.

    Returns
    -------
    dict  vid_id → {'embeddings': ndarray [T, D], 'target_steps': ndarray [T]}
    Sorted by target_steps within each video.
    """
    video_root = cfg.get("h5_video_root", "videos")
    embd_key   = cfg.get("embedding_key", "embeddings")
    steps_key  = cfg.get("target_steps_key", "target_steps")

    result = {}
    with h5py.File(h5_path, "r") as f:
        vg = f[video_root]
        for vid_id in vid_ids:
            if vid_id not in vg:
                print(f"[WARNING] vid_id '{vid_id}' not found in {h5_path}")
                continue
            grp   = vg[vid_id]
            embs  = grp[embd_key][:]
            steps = grp[steps_key][:]
            order = np.argsort(steps)
            result[vid_id] = {
                "embeddings":   embs[order],
                "target_steps": steps[order],
            }
    return result


# ── preprocessing + t-SNE ────────────────────────────────────────────────────

def preprocess_embeddings(embeddings: np.ndarray, cfg: dict) -> np.ndarray:
    """Standardise + optional PCA on the concatenated embedding array."""
    if cfg.get("standardize", True):
        embeddings = StandardScaler().fit_transform(embeddings)

    if cfg.get("use_pca_before_tsne", True):
        pca_dim    = cfg.get("pca_dim", 50)
        N, D       = embeddings.shape
        n_comp     = min(pca_dim, D, N - 1)
        pca        = PCA(n_components=n_comp)
        embeddings = pca.fit_transform(embeddings)
        print(
            f"PCA: {n_comp} dims "
            f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})"
        )
    return embeddings


def run_tsne(embeddings: np.ndarray, cfg: dict):
    N = len(embeddings)
    if N < 10:
        raise ValueError(f"Too few samples for t-SNE: {N}.")

    tcfg             = cfg.get("tsne", {})
    cfg_perplexity   = tcfg.get("perplexity", 30)
    perplexity       = min(cfg_perplexity, max(5, (N - 1) // 3))
    if perplexity != cfg_perplexity:
        print(f"Adjusted perplexity {cfg_perplexity} → {perplexity} (N={N})")

    tsne_kwargs = dict(
        n_components=tcfg.get("n_components", 2),
        perplexity=perplexity,
        learning_rate=tcfg.get("learning_rate", "auto"),
        init=tcfg.get("init", "pca"),
        random_state=cfg.get("random_seed", 42),
    )
    max_iter = tcfg.get("max_iter", 1000)
    try:
        coords = TSNE(**tsne_kwargs, max_iter=max_iter).fit_transform(embeddings)
    except TypeError:
        coords = TSNE(**tsne_kwargs, n_iter=max_iter).fit_transform(embeddings)

    return coords, perplexity


# ── H5 video-ID → raw filename mapping ───────────────────────────────────────

def build_vid_id_to_rawname(dataset_stem: str) -> dict:
    """Return a dict mapping H5 video-group keys (e.g. '000001') to raw video
    filenames without extension (e.g. 'demo_0').

    Reads the *processed* H5 file which stores ``attrs['path']`` on every
    video group.  Falls back to an empty dict if the file is missing.
    """
    processed_h5 = os.path.join(
        _PROJECT_ROOT, "datasets", "processed", f"{dataset_stem}.h5"
    )
    mapping: dict[str, str] = {}
    if not os.path.exists(processed_h5):
        print(f"[WARNING] Processed H5 not found: {processed_h5}. "
              "Cannot map video IDs to filenames.")
        return mapping

    with h5py.File(processed_h5, "r") as f:
        if "videos" not in f:
            print(f"[WARNING] No 'videos' group in {processed_h5}.")
            return mapping
        for vid_id in f["videos"].keys():
            raw_path = f["videos"][vid_id].attrs.get("path", "")
            if raw_path:
                fname = os.path.splitext(os.path.basename(str(raw_path)))[0]
                mapping[vid_id] = fname

    return mapping


# ── 4-video selection ─────────────────────────────────────────────────────────

def select_four_videos(
    cfg: dict,
    id_to_name_g1: dict,
    id_to_name_g2: dict,
) -> list[dict]:
    """Return info for the 4 selected videos.

    Each element is a dict:
        name     : raw filename stem (e.g. 'demo_0')
        group    : 'g1' or 'g2'
        vid_id   : H5 video-group key (e.g. '000001')
        color    : (R, G, B) float tuple
        marker   : 'o' (group1) or '^' (group2)
    """
    # reverse maps: raw_name → (group, vid_id)
    name_map: dict[str, tuple[str, str]] = {}
    for vid_id, name in id_to_name_g1.items():
        name_map[name] = ("g1", vid_id)
    for vid_id, name in id_to_name_g2.items():
        if name not in name_map:   # g1 takes precedence for duplicates
            name_map[name] = ("g2", vid_id)

    sel_names_cfg = cfg.get("selected_videos")
    if sel_names_cfg:
        sel_names = [str(n) for n in sel_names_cfg][:4]
        # Warn about any names not found in either group
        for n in sel_names:
            if n not in name_map:
                print(f"[WARNING] selected_video '{n}' not found in either group.")
        sel_names = [n for n in sel_names if n in name_map]
    else:
        rng       = np.random.default_rng(cfg.get("random_seed", 42))
        all_names = sorted(name_map.keys())
        sel_names = rng.choice(
            all_names, size=min(4, len(all_names)), replace=False
        ).tolist()
        print(f"Randomly selected videos: {sel_names}")

    result = []
    for i, name in enumerate(sel_names):
        group, vid_id = name_map[name]
        result.append(
            dict(
                name=name,
                group=group,
                vid_id=vid_id,
                color=_SEL_COLORS[i % len(_SEL_COLORS)],
                marker="o" if group == "g1" else "^",
            )
        )
    return result


# ── raw video loading ─────────────────────────────────────────────────────────

def load_raw_video_frames(task_name: str, video_name: str) -> np.ndarray:
    """Load all frames of a raw MP4 as uint8 [T, H, W, 3] (RGB).

    Tries ``{video_name}.mp4`` first, then ``.mov``.
    Returns an empty array of shape (0, 1, 1, 3) on failure.
    """
    raw_dir = os.path.join(_PROJECT_ROOT, "datasets", "raw", task_name)
    for ext in (".mp4", ".mov"):
        path = os.path.join(raw_dir, f"{video_name}{ext}")
        if os.path.exists(path):
            break
    else:
        print(f"[WARNING] Raw video not found: {raw_dir}/{video_name}.mp4 (or .mov)")
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)

    cap    = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        print(f"[WARNING] No frames extracted from {path}")
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)

    return np.stack(frames, axis=0)


def save_latent_dist_plot(
    sel_videos: list,
    emb_data_by_sel_idx: dict,
    cfg: dict,
    output_dir: str,
    timestamp: str = None,
) -> str:
    """Save a static PNG of cumulative latent distance (L2) vs normalised
    progress (0-1) for the 4 selected videos."""
    vcfg    = cfg.get("gap_video", {})
    figsize = vcfg.get("figsize", [8, 6])
    dpi     = int(vcfg.get("dpi", 300))

    normalize_y = bool(cfg.get("normalize_latent_dist", False))

    fig, ax = plt.subplots(figsize=figsize)
    for i, v in enumerate(sel_videos):
        data = emb_data_by_sel_idx.get(i)
        if data is None or len(data["embeddings"]) < 2:
            continue
        embs  = data["embeddings"]
        steps = data["target_steps"]
        diffs      = np.linalg.norm(np.diff(embs, axis=0), axis=-1)   # [T-1]
        cumul      = np.concatenate([[0.0], np.cumsum(diffs)])         # [T]
        if normalize_y:
            cumul = cumul / max(float(cumul[-1]), 1e-8)                # normalise to [0,1]
        max_step   = max(int(steps[-1]), 1)
        prog       = steps / max_step                                   # [T] in [0,1]
        ax.plot(prog, cumul, color=v["color"], linewidth=1.5,
                label=f"{v['name']} ({v['group'].upper()})")

    ax.set_xlabel("Progress (0-1)")
    ax.set_ylabel("Cumulative L2 Distance (normalized)" if normalize_y else "Cumulative L2 Distance")
    ax.set_title("Cumulative Latent Distance vs Progress")
    ax.set_xlim(0, 1)
    if normalize_y:
        ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    fname    = f"latent_dist_{timestamp}.png" if timestamp else "latent_dist.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── gap-analysis video ────────────────────────────────────────────────────────

def make_gap_analysis_video(
    coords: np.ndarray,
    vnames1: np.ndarray,
    vnames2: np.ndarray,
    prog1: np.ndarray,
    prog2: np.ndarray,
    n1: int,
    sel_videos: list[dict],
    raw_frames_list: list[np.ndarray],
    emb_data_by_sel_idx: dict,
    n_frames: int,
    cfg: dict,
    output_dir: str,
    timestamp: str = None,
) -> str:
    """Render the combined t-SNE + raw-footage video.

    Parameters
    ----------
    coords          [N, 2]  shared t-SNE coordinates (group1 first, then group2)
    vnames1/2       video-name arrays for group1 / group2
    prog1/2         progress arrays (0-1) for group1 / group2
    n1              number of group1 points
    sel_videos      list of 4 dicts from select_four_videos()
    raw_frames_list list of 4 uint8 ndarray [T, H, W, 3] from load_raw_video_frames()
    n_frames        total output frames (progress sweeps 0 → 1)
    """
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "imageio is required.  Install with:  pip install imageio[ffmpeg]"
        ) from exc

    vcfg      = cfg.get("gap_video", {})
    fps       = int(vcfg.get("fps", 10))
    dpi       = int(vcfg.get("dpi", 300))
    bg_alpha  = float(vcfg.get("background_alpha", 0.06))
    hl_alpha  = float(vcfg.get("highlight_alpha_max", 0.95))
    figsize   = vcfg.get("figsize", [8, 6])
    s         = float(vcfg.get("point_size", 5))
    sel_marker_scale = float(vcfg.get("sel_marker_size_scale", 5))
    draw_border   = bool(vcfg.get("draw_cell_border", True))
    border_px     = int(vcfg.get("cell_border_px", 6))
    lbl_fontsize  = int(vcfg.get("cell_label_fontsize", 14))

    label1 = cfg.get("group1_label", "Group 1")
    label2 = cfg.get("group2_label", "Group 2")

    n2    = len(vnames2)
    n_all = n1 + n2
    prog_all = np.concatenate([prog1, prog2]).astype(np.float32)

    # ── per-point base colour + highlight membership ──────────────────────────
    # Build a fast lookup: (group, vid_id) → selected-video index (0-3) or -1
    sel_g1_ids = {v["vid_id"]: i for i, v in enumerate(sel_videos) if v["group"] == "g1"}
    sel_g2_ids = {v["vid_id"]: i for i, v in enumerate(sel_videos) if v["group"] == "g2"}

    base_rgb    = np.full((n_all, 3), 0.5, dtype=np.float32)   # grey for non-selected
    sel_idx_arr = np.full(n_all, -1, dtype=np.int32)            # which of the 4 selected

    for i in range(n1):
        vid = vnames1[i]
        si  = sel_g1_ids.get(vid, -1)
        if si >= 0:
            base_rgb[i]    = np.array(sel_videos[si]["color"], dtype=np.float32)
            sel_idx_arr[i] = si

    for j in range(n2):
        vid = vnames2[j]
        si  = sel_g2_ids.get(vid, -1)
        if si >= 0:
            base_rgb[n1 + j]    = np.array(sel_videos[si]["color"], dtype=np.float32)
            sel_idx_arr[n1 + j] = si

    is_highlighted = sel_idx_arr >= 0   # boolean mask

    # Per-point scatter sizes: selected videos are rendered larger
    sizes_all = np.where(is_highlighted, s * sel_marker_scale, s).astype(np.float32)

    # Precompute per-selected-video point indices for fast hard highlight
    sel_vid_point_indices = [
        np.where(sel_idx_arr == si)[0] for si in range(len(sel_videos))
    ]

    def _compute_rgba(p_current: float) -> np.ndarray:
        """Hard highlight: only the single closest point per selected video is lit."""
        alpha = np.full(n_all, bg_alpha, dtype=np.float32)
        for indices in sel_vid_point_indices:
            if len(indices) == 0:
                continue
            closest = indices[np.argmin(np.abs(prog_all[indices] - p_current))]
            alpha[closest] = hl_alpha
        return np.column_stack([base_rgb, alpha])

    # ── matplotlib t-SNE panel ────────────────────────────────────────────────
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax  = fig.add_subplot(111)

    coords1 = coords[:n1]
    coords2 = coords[n1:]
    xpad = (coords[:, 0].max() - coords[:, 0].min()) * 0.03
    ypad = (coords[:, 1].max() - coords[:, 1].min()) * 0.03
    ax.set_xlim(coords[:, 0].min() - xpad, coords[:, 0].max() + xpad)
    ax.set_ylim(coords[:, 1].min() - ypad, coords[:, 1].max() + ypad)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_aspect("equal")

    rgba_init = _compute_rgba(0.0)
    sc1 = ax.scatter(coords1[:, 0], coords1[:, 1],
                     c=rgba_init[:n1],  s=sizes_all[:n1], marker="o", linewidths=0)
    sc2 = ax.scatter(coords2[:, 0], coords2[:, 1],
                     c=rgba_init[n1:],  s=sizes_all[n1:], marker="^", linewidths=0)

    # Legend: 4 selected videos
    from matplotlib.lines import Line2D
    legend_elems = []
    for v in sel_videos:
        legend_elems.append(
            Line2D([0], [0], marker=v["marker"], color="w",
                   markerfacecolor=v["color"], markersize=6,
                   label=f"{v['name']} ({v['group'].upper()})")
        )
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7)

    title_obj = ax.set_title(
        f"{label1} (○) vs {label2} (△)  ·  progress = 0.000", fontsize=9
    )
    fig.tight_layout()

    # ── determine t-SNE panel pixel dimensions ────────────────────────────────
    fig.canvas.draw()
    buf_h = int(fig.get_figheight() * dpi)
    buf_w = int(fig.get_figwidth()  * dpi)
    # Align to even pixels (required by libx264)
    buf_h = buf_h - buf_h % 2
    buf_w = buf_w - buf_w % 2

    # ── 2×2 grid cell dimensions ──────────────────────────────────────────────
    cell_w = buf_w // 2
    cell_h = buf_h // 2

    # ── latent distance figure (right-most panel) ─────────────────────────────
    fig_ld = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig_ld)
    ax_ld  = fig_ld.add_subplot(111)

    normalize_y = bool(cfg.get("normalize_latent_dist", False))

    for _i, _v in enumerate(sel_videos):
        _data = emb_data_by_sel_idx.get(_i)
        if _data is None or len(_data["embeddings"]) < 2:
            continue
        _embs  = _data["embeddings"]
        _steps = _data["target_steps"]
        _diffs  = np.linalg.norm(np.diff(_embs, axis=0), axis=-1)   # [T-1]
        _cumul  = np.concatenate([[0.0], np.cumsum(_diffs)])         # [T]
        if normalize_y:
            _cumul = _cumul / max(float(_cumul[-1]), 1e-8)           # normalise to [0,1]
        _max_st = max(int(_steps[-1]), 1)
        _prog   = _steps / _max_st                                    # [T] in [0,1]
        ax_ld.plot(_prog, _cumul, color=_v["color"], linewidth=1.5,
                   label=f"{_v['name']} ({_v['group'].upper()})")

    ax_ld.set_xlabel("Progress (0-1)")
    ax_ld.set_ylabel("Cumulative L2 Distance (normalized)" if normalize_y else "Cumulative L2 Distance")
    ax_ld.set_title("Cumulative Latent Distance vs Progress")
    ax_ld.set_xlim(0, 1)
    if normalize_y:
        ax_ld.set_ylim(0, 1)
    ax_ld.legend(fontsize=7, loc="upper left")
    vline_ld = ax_ld.axvline(x=0.0, color="black", linewidth=1.5, linestyle="--")
    fig_ld.tight_layout()

    # ── output path ───────────────────────────────────────────────────────────
    fname    = f"tsne_gap_analysis_{timestamp}.mp4" if timestamp else "tsne_gap_analysis.mp4"
    out_path = os.path.join(output_dir, fname)
    print(f"Rendering {n_frames} frames at {fps} fps → {out_path}")
    print(f"  t-SNE panel: {buf_w}×{buf_h} px  |  grid cell: {cell_w}×{cell_h} px")
    print(f"  bg_alpha={bg_alpha}  hl_alpha={hl_alpha}  (hard single-point highlight)")

    try:
        writer = imageio.get_writer(
            out_path,
            fps=fps,
            format="ffmpeg",
            codec="libx264",
            macro_block_size=1,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open video writer for {out_path}.\n"
            "Ensure ffmpeg and imageio[ffmpeg] are installed.\n"
            f"Original error: {exc}"
        ) from exc

    log_every = max(1, n_frames // 20)

    def _render_grid(p_current: float) -> np.ndarray:
        """Build the 2×2 raw footage grid for the given progress."""
        grid = np.zeros((buf_h, buf_w, 3), dtype=np.uint8)
        for i, (vid_info, raw_frames) in enumerate(zip(sel_videos, raw_frames_list)):
            r = i // 2
            c = i % 2
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w

            if len(raw_frames) == 0:
                # No frames: grey placeholder
                cell = np.full((cell_h, cell_w, 3), 80, dtype=np.uint8)
            else:
                t          = min(int(round(p_current * (len(raw_frames) - 1))),
                                 len(raw_frames) - 1)
                frame_bgr  = cv2.resize(
                    raw_frames[t],          # already RGB from load_raw_video_frames
                    (cell_w, cell_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                cell = frame_bgr            # uint8 RGB

            # Coloured border
            if draw_border:
                color_u8 = tuple(int(x * 255) for x in vid_info["color"])
                bp = border_px
                cell[:bp,  :] = color_u8
                cell[-bp:, :] = color_u8
                cell[:,  :bp] = color_u8
                cell[:, -bp:] = color_u8

            # Video name label
            label_text = vid_info["name"]
            font_scale = lbl_fontsize / 20.0
            cv2.putText(
                cell, label_text,
                (border_px + 4, border_px + int(lbl_fontsize * 1.2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            grid[y0:y1, x0:x1] = cell

        return grid

    try:
        for f in range(n_frames):
            p_current = f / max(n_frames - 1, 1)

            # ── t-SNE panel ──────────────────────────────────────────────────
            rgba = _compute_rgba(p_current)
            sc1.set_facecolor(rgba[:n1])
            sc2.set_facecolor(rgba[n1:])
            title_obj.set_text(
                f"{label1} (○) vs {label2} (△)  ·  progress = {p_current:.3f}"
            )
            fig.canvas.draw()
            tsne_panel = np.asarray(fig.canvas.buffer_rgba()).copy()[:, :, :3]
            tsne_panel = tsne_panel[:buf_h, :buf_w]          # enforce even dims

            # ── 2×2 raw grid ─────────────────────────────────────────────────
            raw_grid = _render_grid(p_current)

            # ── latent distance panel ─────────────────────────────────────────
            vline_ld.set_xdata([p_current, p_current])
            fig_ld.canvas.draw()
            ld_panel = np.asarray(fig_ld.canvas.buffer_rgba()).copy()[:, :, :3]
            ld_panel = ld_panel[:buf_h, :buf_w]

            # ── combine left | centre | right ────────────────────────────────
            combined = np.concatenate([tsne_panel, raw_grid, ld_panel], axis=1)
            writer.append_data(combined)

            if f % log_every == 0:
                print(f"  [{f + 1:5d}/{n_frames}]  progress = {p_current:.3f}")
    finally:
        writer.close()

    plt.close("all")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gap analysis: t-SNE + raw footage for 4 selected videos."
    )
    parser.add_argument(
        "--viz_config",
        default=None,
        help="[v2] Path to a per-flow visualize YAML override "
             "(default: configs_v2/visualize/tsne_gap_analysis.yaml).",
    )
    parser.add_argument(
        "--normalize-latent-dist",
        dest="normalize_latent_dist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize cumulative latent dist y-axis to [0,1] (default: value from config, "
             "or False if not set). Use --normalize-latent-dist to normalise.",
    )
    parser.add_argument(
        "--embedding_ref",
        default=None,
        help="[v2] Embedding alias for group 1 (overrides the YAML value).",
    )
    parser.add_argument(
        "--embedding_ref_group2",
        default=None,
        help="[v2] Embedding alias for group 2 (overrides the YAML value).",
    )
    args = parser.parse_args()

    # [v2] Load and resolve via ConfigV2
    from utils.config_v2 import ConfigV2
    _overrides = {}
    if args.embedding_ref:
        _overrides["embedding_ref"] = args.embedding_ref
    if args.embedding_ref_group2:
        _overrides["embedding_ref_group2"] = args.embedding_ref_group2
    cfg = ConfigV2().load_visualize(
        "tsne_gap_analysis",
        config_path=args.viz_config,
        overrides=_overrides or None,
    )

    # CLI flag overrides config value
    if args.normalize_latent_dist is not None:
        cfg["normalize_latent_dist"] = args.normalize_latent_dist

    # ── resolved paths ─────────────────────────────────────────────────────────
    h5_path1   = cfg["embedding_h5_path"]
    h5_path2   = cfg["embedding_h5_path_group2"]
    label1     = cfg.get("group1_label", "Group1")
    label2     = cfg.get("group2_label", "Group2")
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Group 1 ({label1}): {h5_path1}")
    print(f"Group 2 ({label2}): {h5_path2}")
    print(f"Output dir        : {output_dir}")

    # ── load embeddings ───────────────────────────────────────────────────────
    emb1, vnames1, tsteps1, prog1, mean_sl1 = load_embeddings_from_h5(h5_path1, cfg)
    emb2, vnames2, tsteps2, prog2, mean_sl2 = load_embeddings_from_h5(h5_path2, cfg)
    print(
        f"Loaded {len(np.unique(vnames1))} videos from group1 ({len(emb1)} frames, "
        f"mean seq_len={mean_sl1:.1f})"
    )
    print(
        f"Loaded {len(np.unique(vnames2))} videos from group2 ({len(emb2)} frames, "
        f"mean seq_len={mean_sl2:.1f})"
    )

    # ── sample ────────────────────────────────────────────────────────────────
    emb1, vnames1, tsteps1, prog1 = sample_embeddings(emb1, vnames1, tsteps1, prog1, cfg)
    emb2, vnames2, tsteps2, prog2 = sample_embeddings(emb2, vnames2, tsteps2, prog2, cfg)
    n1, n2 = len(emb1), len(emb2)
    print(f"After sampling: group1={n1}, group2={n2}")

    # ── build H5-ID → raw filename mappings ──────────────────────────────────
    stem1 = cfg.get("h5_stem", "")
    stem2 = cfg.get("h5_stem_group2", "")
    id_to_name_g1 = build_vid_id_to_rawname(stem1)
    id_to_name_g2 = build_vid_id_to_rawname(stem2)
    print(
        f"Processed H5 mappings: group1={len(id_to_name_g1)} videos, "
        f"group2={len(id_to_name_g2)} videos"
    )

    # ── select 4 videos ───────────────────────────────────────────────────────
    sel_videos = select_four_videos(cfg, id_to_name_g1, id_to_name_g2)
    if len(sel_videos) == 0:
        raise RuntimeError(
            "No valid videos could be selected. "
            "Check 'selected_videos' in your config or the processed H5 mappings."
        )
    print("Selected videos:")
    for v in sel_videos:
        print(f"  {v['name']}  group={v['group']}  vid_id={v['vid_id']}")

    # ── joint preprocessing + t-SNE ───────────────────────────────────────────
    emb_all  = np.concatenate([emb1, emb2], axis=0)
    print(f"Total frames for t-SNE: {len(emb_all)}, dim={emb_all.shape[1]}")
    emb_proc = preprocess_embeddings(emb_all, cfg)

    print("Running t-SNE on joint data …")
    coords, perplexity_used = run_tsne(emb_proc, cfg)
    print(f"Perplexity used: {perplexity_used}")

    # ── determine n_frames ────────────────────────────────────────────────────
    vcfg        = cfg.get("gap_video", {})
    n_frames_cfg = vcfg.get("n_frames")
    max_n_frames = vcfg.get("max_n_frames", 500)

    if n_frames_cfg is not None:
        n_frames = int(n_frames_cfg)
    else:
        n_frames = n1 + n2   # total loaded embedding frames
        if max_n_frames is not None:
            n_frames = min(n_frames, int(max_n_frames))

    n_frames = max(n_frames, 2)
    print(f"Output video: n_frames={n_frames}, fps={vcfg.get('fps', 10)}")

    # ── load raw video frames ─────────────────────────────────────────────────
    task_name = cfg.get("task_name")
    if not task_name:
        raise ValueError("'task_name' must be set in the config (subfolder of datasets/raw/).")

    print(f"Loading raw video frames from datasets/raw/{task_name}/…")
    raw_frames_list = []
    for v in sel_videos:
        frames = load_raw_video_frames(task_name, v["name"])
        print(
            f"  {v['name']}: {len(frames)} frames"
            + ("" if len(frames) > 0 else "  ← WARNING: no frames loaded")
        )
        raw_frames_list.append(frames)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── load embeddings for latent distance plot ──────────────────────────────
    g1_sel_ids = [v["vid_id"] for v in sel_videos if v["group"] == "g1"]
    g2_sel_ids = [v["vid_id"] for v in sel_videos if v["group"] == "g2"]
    emb_g1_raw = load_selected_video_embeddings(h5_path1, g1_sel_ids, cfg)
    emb_g2_raw = load_selected_video_embeddings(h5_path2, g2_sel_ids, cfg)

    emb_data_by_sel_idx: dict = {}
    for i, v in enumerate(sel_videos):
        emb_data_by_sel_idx[i] = (
            emb_g1_raw.get(v["vid_id"]) if v["group"] == "g1"
            else emb_g2_raw.get(v["vid_id"])
        )

    path_ld = save_latent_dist_plot(
        sel_videos, emb_data_by_sel_idx, cfg, output_dir, timestamp
    )
    print(f"Saved latent dist plot  : {path_ld}")

    # ── render combined video ─────────────────────────────────────────────────
    out_path  = make_gap_analysis_video(
        coords=coords,
        vnames1=vnames1,
        vnames2=vnames2,
        prog1=prog1,
        prog2=prog2,
        n1=n1,
        sel_videos=sel_videos,
        raw_frames_list=raw_frames_list,
        emb_data_by_sel_idx=emb_data_by_sel_idx,
        n_frames=n_frames,
        cfg=cfg,
        output_dir=output_dir,
        timestamp=timestamp,
    )
    print(f"\nSaved gap analysis video: {out_path}")


if __name__ == "__main__":
    main()
