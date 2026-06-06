"""
t-SNE visualization comparing two groups of TCC embeddings (e.g. training vs val)
projected into a **shared** 2D space by running t-SNE on the concatenated data.

Both groups are plotted in the same axes with distinct sequential colormaps so that
temporal progress within each group is still visible while the two groups remain
visually distinguishable.

Usage:
    cd /home/user/zhangzk/projects/fineprog
    python scripts/visualize_embeddings_tsne_2groups.py
    python scripts/visualize_embeddings_tsne_2groups.py --viz_config configs_v2/visualize/tsne_2groups.yaml
"""

import argparse
import os
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, _PROJECT_ROOT)


def _resolve_run_name_v2(embedding_ref: str | None = None) -> str | None:
    """[v2] Get run_name from configs_v2/runs.yaml via ConfigV2."""
    if not embedding_ref:
        return None
    try:
        from utils.config_v2 import ConfigV2
        return ConfigV2().resolve_embedding(embedding_ref).get("run_name")
    except Exception:
        return None


def _resolve_h5_v2(embedding_ref: str) -> str:
    """[v2] Resolve embedding H5 absolute path from runs.yaml registry."""
    from utils.config_v2 import ConfigV2
    return ConfigV2().resolve_embedding(embedding_ref)["embedding_h5_path"]


def resolve_h5_path(cfg: dict, stem_key: str, embedding_ref: str | None = None) -> str:
    """Return the HDF5 path for *stem_key*.

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

    raise ValueError(
        f"Cannot resolve '{stem_key}': pass --embedding_ref / --embedding_ref_group2 "
        f"or set '{override_key}' in the visualize config."
    )


def load_embeddings_from_h5(h5_path: str, cfg: dict):
    """Load embeddings from HDF5.

    Returns
    -------
    embeddings   : ndarray [N, D]
    video_names  : ndarray [N]  (str)
    target_steps : ndarray [N]  (int)
    progress     : ndarray [N]  (float, 0-1)
    mean_seq_len : float        (average sequence length across all videos)
    """
    video_root = cfg.get("h5_video_root", "videos")
    embd_key = cfg.get("embedding_key", "embeddings")
    steps_key = cfg.get("target_steps_key", "target_steps")

    all_embeddings = []
    all_video_names = []
    all_target_steps = []
    all_progress = []
    seq_lens = []

    with h5py.File(h5_path, "r") as f:
        videos_grp = f[video_root]
        for video_id in videos_grp.keys():
            grp = videos_grp[video_id]
            embeddings = grp[embd_key][:]       # [T, D]
            target_steps = grp[steps_key][:]    # [T]

            seq_len = int(grp.attrs.get("seq_len", len(embeddings)))
            T = len(embeddings)
            progress = target_steps / max(seq_len - 1, 1)

            all_embeddings.append(embeddings)
            all_video_names.extend([video_id] * T)
            all_target_steps.append(target_steps)
            all_progress.append(progress)
            seq_lens.append(seq_len)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_target_steps = np.concatenate(all_target_steps, axis=0)
    all_progress = np.concatenate(all_progress, axis=0)
    all_video_names = np.array(all_video_names)
    mean_seq_len = float(np.mean(seq_lens)) if seq_lens else 100.0

    return all_embeddings, all_video_names, all_target_steps, all_progress, mean_seq_len


def sample_embeddings(embeddings, video_names, target_steps, progress, cfg):
    rng = np.random.default_rng(cfg.get("random_seed", 42))
    max_per_video = cfg.get("max_frames_per_video", 300)
    max_total = cfg.get("max_total_frames", 10000)

    unique_videos = np.unique(video_names)
    sel_indices = []
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


def filter_top_n_videos(embeddings, video_names, target_steps, progress, top_n, seed=42):
    rng = np.random.default_rng(seed)
    unique_all = np.unique(video_names)
    keep = set(rng.choice(unique_all, size=min(top_n, len(unique_all)), replace=False))
    mask = np.array([v in keep for v in video_names])
    return embeddings[mask], video_names[mask], target_steps[mask], progress[mask]


def preprocess_embeddings(embeddings, cfg):
    """Joint standardisation + optional PCA applied to the already-concatenated array."""
    if cfg.get("standardize", True):
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    if cfg.get("use_pca_before_tsne", True):
        pca_dim = cfg.get("pca_dim", 50)
        N, D = embeddings.shape
        n_components = min(pca_dim, D, N - 1)
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(embeddings)
        print(
            f"PCA: reduced to {n_components} dims "
            f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})"
        )

    return embeddings


def run_tsne(embeddings, cfg):
    N = len(embeddings)
    if N < 10:
        raise ValueError(f"Too few samples for t-SNE: {N}. Need at least 10.")

    tcfg = cfg.get("tsne", {})
    config_perplexity = tcfg.get("perplexity", 30)
    perplexity = min(config_perplexity, max(5, (N - 1) // 3))
    if perplexity != config_perplexity:
        print(f"Adjusted perplexity from {config_perplexity} to {perplexity} (N={N})")

    tsne_kwargs = dict(
        n_components=tcfg.get("n_components", 2),
        perplexity=perplexity,
        learning_rate=tcfg.get("learning_rate", "auto"),
        init=tcfg.get("init", "pca"),
        random_state=cfg.get("random_seed", 42),
    )
    max_iter = tcfg.get("max_iter", 1000)
    try:
        tsne = TSNE(**tsne_kwargs, max_iter=max_iter)
        coords = tsne.fit_transform(embeddings)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, n_iter=max_iter)
        coords = tsne.fit_transform(embeddings)

    return coords, perplexity


# ── animation ────────────────────────────────────────────────────────────────


def make_tsne_animation(
    coords, vnames1, vnames2, prog1, prog2, n1, avg_seq_len, cfg, output_dir, timestamp=None
):
    """Render an mp4 animation of the shared t-SNE space.

    The animation sweeps temporal progress 0→1.  Selected videos show a smooth
    Gaussian highlight centred at the current progress; all other points remain
    faint throughout.  No legend is drawn (see static by-video plot instead).

    Parameters
    ----------
    coords       : [N, 2]  t-SNE coordinates (group1 first, then group2)
    vnames1/2    : video-name arrays for group1 / group2
    prog1/2      : progress arrays (0-1) for group1 / group2
    n1           : number of points belonging to group1
    avg_seq_len  : determines n_frames = 2 * avg_seq_len

    Config (tsne_video section)
    --------------------------
    fps                      int   output frame rate (default 30)
    dpi                      int   render DPI for each frame (default 150)
    progress_sigma           float Gaussian σ for the highlight sweep (default 0.05)
    background_alpha         float alpha of all non-highlighted points (default 0.08)
    highlight_alpha_max      float peak alpha at the exact current progress (default 0.95)
    highlight_videos_group1  list  video IDs to animate in group1; null = all
    highlight_videos_group2  list  video IDs to animate in group2; null = all
    """
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for video export. "
            "Install it with:  pip install imageio[ffmpeg]"
        ) from exc

    pcfg = cfg.get("plot", {})
    vcfg = cfg.get("tsne_video", {})

    fps       = int(vcfg.get("fps", 30))
    sigma     = float(vcfg.get("progress_sigma", 0.05))
    bg_alpha  = float(vcfg.get("background_alpha", 0.08))
    hl_alpha  = float(vcfg.get("highlight_alpha_max", 0.95))
    video_dpi = int(vcfg.get("dpi", 150))
    figsize   = pcfg.get("figsize", [8, 6])
    s         = pcfg.get("point_size", 5)

    hl_vids1_cfg = vcfg.get("highlight_videos_group1", "all")
    hl_vids2_cfg = vcfg.get("highlight_videos_group2", "all")

    # null  → no videos highlighted (all stay faint)
    # "all" → all videos in the group participate in the sweep
    # list  → only the listed video IDs participate in the sweep
    def _resolve_hl_set(val):
        if val is None:
            return set()   # empty set → nothing highlighted
        if val == "all":
            return None    # None internally → all highlighted
        return {str(v) for v in val}

    hl_set1 = _resolve_hl_set(hl_vids1_cfg)
    hl_set2 = _resolve_hl_set(hl_vids2_cfg)

    label1 = cfg.get("group1_label", "Group 1")
    label2 = cfg.get("group2_label", "Group 2")

    n_frames = max(2, int(round(2.0 * avg_seq_len)))
    n2   = len(vnames2)
    n_all = n1 + n2

    # ── per-point base RGB colour ──────────────────────────────────────────────
    unique_vids1 = sorted(set(vnames1))
    unique_vids2 = sorted(set(vnames2))
    cmap1_fn = plt.get_cmap("tab20",  max(len(unique_vids1), 1))
    cmap2_fn = plt.get_cmap("tab20b", max(len(unique_vids2), 1))
    vid1_to_idx = {v: i for i, v in enumerate(unique_vids1)}
    vid2_to_idx = {v: i for i, v in enumerate(unique_vids2)}

    base_rgb      = np.zeros((n_all, 3), dtype=np.float32)
    is_highlighted = np.zeros(n_all, dtype=bool)

    for i in range(n1):
        vid = vnames1[i]
        base_rgb[i] = np.array(cmap1_fn(vid1_to_idx[vid]))[:3]
        if hl_set1 is None or vid in hl_set1:
            is_highlighted[i] = True

    for j in range(n2):
        vid = vnames2[j]
        base_rgb[n1 + j] = np.array(cmap2_fn(vid2_to_idx[vid]))[:3]
        if hl_set2 is None or vid in hl_set2:
            is_highlighted[n1 + j] = True

    prog_all = np.concatenate([prog1, prog2]).astype(np.float32)  # [N]

    # ── RGBA helper ───────────────────────────────────────────────────────────
    def _compute_rgba(p_current: float) -> np.ndarray:
        """Gaussian highlight centred at p_current; non-highlighted → bg_alpha."""
        diff = prog_all - p_current
        w    = np.exp(-0.5 * (diff / sigma) ** 2)           # [N], 0..1
        alpha = np.where(
            is_highlighted,
            bg_alpha + (hl_alpha - bg_alpha) * w,
            bg_alpha,
        ).astype(np.float32)
        return np.column_stack([base_rgb, alpha])            # [N, 4]

    # ── figure setup (Agg canvas, no display) ─────────────────────────────────
    fig = Figure(figsize=figsize, dpi=video_dpi)
    FigureCanvasAgg(fig)   # attaches Agg canvas to fig
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
                     c=rgba_init[:n1], s=s, marker="o", linewidths=0)
    sc2 = ax.scatter(coords2[:, 0], coords2[:, 1],
                     c=rgba_init[n1:], s=s, marker="^", linewidths=0)

    title_obj = ax.set_title(
        f"{label1} (○) vs {label2} (△)  ·  progress = 0.000", fontsize=9
    )
    fig.tight_layout()

    # ── output path ───────────────────────────────────────────────────────────
    fname = (
        f"tsne_2groups_video_{timestamp}.mp4"
        if timestamp
        else "tsne_2groups_video.mp4"
    )
    out_path = os.path.join(output_dir, fname)
    print(f"Rendering {n_frames} frames at {fps} fps → {out_path}")
    print(f"  avg_seq_len={avg_seq_len:.1f}  sigma={sigma}  "
          f"bg_alpha={bg_alpha}  hl_alpha={hl_alpha}")

    # ── render frames ─────────────────────────────────────────────────────────
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
            "Make sure ffmpeg is installed and imageio[ffmpeg] is available.\n"
            f"Original error: {exc}"
        ) from exc

    log_every = max(1, n_frames // 20)
    try:
        for f in range(n_frames):
            p_current = f / max(n_frames - 1, 1)
            rgba = _compute_rgba(p_current)
            sc1.set_facecolor(rgba[:n1])
            sc2.set_facecolor(rgba[n1:])
            title_obj.set_text(
                f"{label1} (○) vs {label2} (△)  ·  progress = {p_current:.3f}"
            )

            fig.canvas.draw()
            # buffer_rgba returns a memoryview; .copy() gives a writable ndarray
            frame = np.asarray(fig.canvas.buffer_rgba()).copy()[:, :, :3]
            # libx264 requires even pixel dimensions
            h, w = frame.shape[:2]
            frame = frame[: h - h % 2, : w - w % 2]
            writer.append_data(frame)

            if f % log_every == 0:
                print(f"  [{f + 1:4d}/{n_frames}]  progress = {p_current:.3f}")
    finally:
        writer.close()

    plt.close("all")
    return out_path


# ── plotting ──────────────────────────────────────────────────────────────────


def plot_2groups_by_progress(coords, prog1, prog2, n1, cfg, output_dir, timestamp=None):
    """Shared axes, group1 colored by one sequential cmap, group2 by another."""
    pcfg = cfg.get("plot", {})
    figsize = pcfg.get("figsize", [9, 6])
    dpi = pcfg.get("dpi", 300)
    s = pcfg.get("point_size", 5)
    alpha = pcfg.get("alpha", 0.75)
    cmap1 = pcfg.get("cmap_progress_group1", "Blues")
    cmap2 = pcfg.get("cmap_progress_group2", "Reds")

    label1 = cfg.get("group1_label", "Group 1")
    label2 = cfg.get("group2_label", "Group 2")

    coords1 = coords[:n1]
    coords2 = coords[n1:]

    fig, ax = plt.subplots(figsize=figsize)

    sc1 = ax.scatter(
        coords1[:, 0], coords1[:, 1],
        c=prog1, cmap=cmap1, vmin=0.0, vmax=1.0,
        s=s, alpha=alpha,
    )
    sc2 = ax.scatter(
        coords2[:, 0], coords2[:, 1],
        c=prog2, cmap=cmap2, vmin=0.0, vmax=1.0,
        s=s, alpha=alpha,
    )

    cb1 = plt.colorbar(sc1, ax=ax, fraction=0.040, pad=0.02)
    cb1.set_label(f"{label1} Progress")
    cb2 = plt.colorbar(sc2, ax=ax, fraction=0.040, pad=0.10)
    cb2.set_label(f"{label2} Progress")

    ax.set_title(f"t-SNE: {label1} vs {label2} — Temporal Progress")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    # Legend proxies for the two groups
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=plt.get_cmap(cmap1)(0.7), markersize=6, label=label1),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=plt.get_cmap(cmap2)(0.7), markersize=6, label=label2),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

    fname = (
        f"tsne_2groups_by_progress_{timestamp}.png"
        if timestamp
        else "tsne_2groups_by_progress.png"
    )
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_2groups_by_video(coords, vnames1, vnames2, n1, cfg, output_dir, timestamp=None):
    """Shared axes; group1 videos use tab20 (circles), group2 videos use tab20b (triangles)."""
    pcfg = cfg.get("plot", {})
    figsize = pcfg.get("figsize", [8, 6])
    dpi = pcfg.get("dpi", 300)
    s = pcfg.get("point_size", 5)
    alpha = pcfg.get("alpha", 0.75)

    label1 = cfg.get("group1_label", "Group 1")
    label2 = cfg.get("group2_label", "Group 2")

    coords1 = coords[:n1]
    coords2 = coords[n1:]

    unique_vids1 = sorted(set(vnames1))
    unique_vids2 = sorted(set(vnames2))
    cmap1 = plt.get_cmap("tab20", max(len(unique_vids1), 1))
    cmap2 = plt.get_cmap("tab20b", max(len(unique_vids2), 1))

    fig, ax = plt.subplots(figsize=figsize)

    for i, vid in enumerate(unique_vids1):
        mask = vnames1 == vid
        ax.scatter(
            coords1[mask, 0], coords1[mask, 1],
            s=s, alpha=alpha,
            color=cmap1(i),
            marker="o",
            label=f"[{label1}] {vid}",
        )

    for i, vid in enumerate(unique_vids2):
        mask = vnames2 == vid
        ax.scatter(
            coords2[mask, 0], coords2[mask, 1],
            s=s, alpha=alpha,
            color=cmap2(i),
            marker="^",
            label=f"[{label2}] {vid}",
        )

    ax.set_title(
        f"t-SNE: {label1} vs {label2} — By Video\n"
        f"({label1}: circles  |  {label2}: triangles)"
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_aspect("equal")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=5,
        markerscale=2,
    )

    fname = (
        f"tsne_2groups_by_video_{timestamp}.png"
        if timestamp
        else "tsne_2groups_by_video.png"
    )
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE of two embedding groups in a shared 2D space."
    )
    parser.add_argument(
        "--viz_config",
        default=None,
        help="[v2] Path to a per-flow visualize YAML override "
             "(default: configs_v2/visualize/tsne_2groups.yaml).",
    )
    parser.add_argument(
        "--top_n_videos",
        type=int,
        default=None,
        help="Randomly keep at most N videos per group before sampling.",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Skip mp4 animation rendering (only produce static PNG plots).",
    )
    parser.add_argument(
        "--video_only",
        action="store_true",
        help="Skip static PNG plots and only render the mp4 animation.",
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
        "tsne_2groups",
        config_path=args.viz_config,
        overrides=_overrides or None,
    )

    # ── resolved paths ─────────────────────────────────────────────────────────
    h5_path1   = cfg["embedding_h5_path"]
    h5_path2   = cfg["embedding_h5_path_group2"]
    label1     = cfg.get("group1_label", "Group1")
    label2     = cfg.get("group2_label", "Group2")
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Group 1 ({label1}): {h5_path1}")
    print(f"Group 2 ({label2}): {h5_path2}")
    print(f"output_dir        : {output_dir}")

    # ── load ───────────────────────────────────────────────────────────────────
    emb1, vnames1, tsteps1, prog1, mean_sl1 = load_embeddings_from_h5(h5_path1, cfg)
    emb2, vnames2, tsteps2, prog2, mean_sl2 = load_embeddings_from_h5(h5_path2, cfg)
    print(f"Loaded {len(np.unique(vnames1))} videos from group 1 ({len(emb1)} frames), "
          f"mean seq_len={mean_sl1:.1f}")
    print(f"Loaded {len(np.unique(vnames2))} videos from group 2 ({len(emb2)} frames), "
          f"mean seq_len={mean_sl2:.1f}")
    avg_seq_len = (mean_sl1 + mean_sl2) / 2.0

    # ── optional video filter ─────────────────────────────────────────────────
    if args.top_n_videos is not None:
        seed = cfg.get("random_seed", 42)
        emb1, vnames1, tsteps1, prog1 = filter_top_n_videos(
            emb1, vnames1, tsteps1, prog1, args.top_n_videos, seed
        )
        emb2, vnames2, tsteps2, prog2 = filter_top_n_videos(
            emb2, vnames2, tsteps2, prog2, args.top_n_videos, seed
        )
        print(
            f"After video filter: group1={len(emb1)} frames, group2={len(emb2)} frames"
        )

    # ── sample ────────────────────────────────────────────────────────────────
    emb1, vnames1, tsteps1, prog1 = sample_embeddings(emb1, vnames1, tsteps1, prog1, cfg)
    emb2, vnames2, tsteps2, prog2 = sample_embeddings(emb2, vnames2, tsteps2, prog2, cfg)
    n1, n2 = len(emb1), len(emb2)
    print(f"Sampled: group1={n1}, group2={n2}")

    # ── joint preprocessing + t-SNE ───────────────────────────────────────────
    emb_all = np.concatenate([emb1, emb2], axis=0)
    print(f"Total frames for t-SNE: {len(emb_all)}, embedding dim: {emb_all.shape[1]}")

    emb_proc = preprocess_embeddings(emb_all, cfg)

    print("Running t-SNE on joint data ...")
    coords, perplexity_used = run_tsne(emb_proc, cfg)
    print(f"Perplexity used: {perplexity_used}")

    # ── plot ──────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    video_only = args.video_only or cfg.get("tsne_video", {}).get("video_only", False)

    if not video_only:
        path_prog = plot_2groups_by_progress(
            coords, prog1, prog2, n1, cfg, output_dir, timestamp
        )
        path_vid = plot_2groups_by_video(
            coords, vnames1, vnames2, n1, cfg, output_dir, timestamp
        )
        print(f"Saved t-SNE by progress : {path_prog}")
        print(f"Saved t-SNE by video    : {path_vid}")

    # ── animated video ────────────────────────────────────────────────────────
    if not args.no_video:
        path_video = make_tsne_animation(
            coords, vnames1, vnames2, prog1, prog2, n1, avg_seq_len,
            cfg, output_dir, timestamp,
        )
        print(f"Saved animation         : {path_video}")


if __name__ == "__main__":
    main()
