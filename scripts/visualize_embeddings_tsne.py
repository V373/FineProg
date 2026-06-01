"""
Minimal t-SNE visualization for TCC embeddings.

Usage:
    cd /home/user/zhangzk/projects/fineprog
    python scripts/visualize_embeddings_tsne.py
    python scripts/visualize_embeddings_tsne.py --embedding_ref can_ph_valid_ep50k
    python scripts/visualize_embeddings_tsne.py --viz_config configs_v2/visualize/tsne.yaml
"""

import argparse
import os
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, _PROJECT_ROOT)


# Path resolution is handled entirely by ConfigV2.load_visualize('tsne') via
# the embedding_ref in configs_v2/visualize/tsne.yaml.


def load_embeddings_from_h5(cfg):
    h5_path = cfg["embedding_h5_path"]
    video_root = cfg.get("h5_video_root", "videos")
    embd_key = cfg.get("embedding_key", "embeddings")
    steps_key = cfg.get("target_steps_key", "target_steps")

    all_embeddings = []
    all_video_names = []
    all_target_steps = []
    all_progress = []

    with h5py.File(h5_path, "r") as f:
        videos_grp = f[video_root]
        for video_id in videos_grp.keys():
            grp = videos_grp[video_id]
            embeddings = grp[embd_key][:]          # [T_out, D]
            target_steps = grp[steps_key][:]       # [T_out]

            seq_len = grp.attrs.get("seq_len", len(embeddings))
            seq_len = int(seq_len)

            T = len(embeddings)
            progress = target_steps / max(seq_len - 1, 1)

            all_embeddings.append(embeddings)
            all_video_names.extend([video_id] * T)
            all_target_steps.append(target_steps)
            all_progress.append(progress)

    all_embeddings = np.concatenate(all_embeddings, axis=0)   # [N, D]
    all_target_steps = np.concatenate(all_target_steps, axis=0)
    all_progress = np.concatenate(all_progress, axis=0)
    all_video_names = np.array(all_video_names)

    return all_embeddings, all_video_names, all_target_steps, all_progress


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


def preprocess_embeddings(embeddings, cfg):
    if cfg.get("standardize", True):
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    if cfg.get("use_pca_before_tsne", True):
        pca_dim = cfg.get("pca_dim", 50)
        N, D = embeddings.shape
        n_components = min(pca_dim, D, N - 1)
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(embeddings)
        print(f"PCA: reduced to {n_components} dims "
              f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

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


def plot_by_video(coords, video_names, cfg, output_dir, timestamp=None):
    pcfg = cfg.get("plot", {})
    figsize = pcfg.get("figsize", [8, 6])
    dpi = pcfg.get("dpi", 300)
    s = pcfg.get("point_size", 5)
    alpha = pcfg.get("alpha", 0.75)

    unique_videos = sorted(set(video_names))
    cmap = plt.get_cmap("tab20", len(unique_videos))
    vid_to_idx = {v: i for i, v in enumerate(unique_videos)}

    fig, ax = plt.subplots(figsize=figsize)
    for vid in unique_videos:
        mask = video_names == vid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=s, alpha=alpha,
            color=cmap(vid_to_idx[vid]),
            label=vid,
        )

    ax.set_title("t-SNE of TCC Embeddings Colored by Video")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=6,
        markerscale=2,
    )

    fname = f"tsne_by_video_{timestamp}.png" if timestamp else "tsne_by_video.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_progress(coords, progress, cfg, output_dir, timestamp=None):
    pcfg = cfg.get("plot", {})
    figsize = pcfg.get("figsize", [8, 6])
    dpi = pcfg.get("dpi", 300)
    s = pcfg.get("point_size", 5)
    alpha = pcfg.get("alpha", 0.75)
    cmap_name = pcfg.get("cmap_progress", "viridis")

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=progress, cmap=cmap_name,
        vmin=0.0, vmax=1.0,
        s=s, alpha=alpha,
    )
    plt.colorbar(sc, ax=ax, label="Temporal Progress")
    ax.set_title("t-SNE of TCC Embeddings Colored by Temporal Progress")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    fname = f"tsne_by_progress_{timestamp}.png" if timestamp else "tsne_by_progress.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--viz_config",
        default=None,
        help="[v2] Path to a per-flow visualize YAML override "
             "(default: configs_v2/visualize/tsne.yaml).",
    )
    parser.add_argument(
        "--embedding_ref",
        default=None,
        help="[v2] Embedding alias from configs_v2/registry/runs.yaml. "
             "Overrides the embedding_ref set in the visualize YAML.",
    )
    parser.add_argument(
        "--top_n_videos",
        type=int,
        default=None,
        help="Only visualize embeddings from N randomly selected videos. "
             "If not set, all videos are used.",
    )
    args = parser.parse_args()

    # [v2] Load and resolve via ConfigV2
    from utils.config_v2 import ConfigV2
    _overrides = {}
    if args.embedding_ref:
        _overrides["embedding_ref"] = args.embedding_ref
    cfg = ConfigV2().load_visualize("tsne", config_path=args.viz_config, overrides=_overrides or None)
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"embedding_h5_path : {cfg['embedding_h5_path']}")
    print(f"output_dir        : {output_dir}")

    # Load
    embeddings, video_names, target_steps, progress = load_embeddings_from_h5(cfg)
    n_videos = len(np.unique(video_names))
    print(f"Loaded videos: {n_videos}")

    # Filter to a random subset of N videos if requested
    if args.top_n_videos is not None:
        top_n = args.top_n_videos
        rng_filter = np.random.default_rng(cfg.get("random_seed", 42))
        unique_all = np.unique(video_names)
        keep_videos = set(
            rng_filter.choice(unique_all, size=min(top_n, len(unique_all)), replace=False)
        )
        mask = np.array([v in keep_videos for v in video_names])
        embeddings   = embeddings[mask]
        video_names  = video_names[mask]
        target_steps = target_steps[mask]
        progress     = progress[mask]
        print(f"Randomly selected {len(keep_videos)} videos ({mask.sum()} frames)")

    # Sample
    embeddings, video_names, target_steps, progress = sample_embeddings(
        embeddings, video_names, target_steps, progress, cfg
    )
    N, D = embeddings.shape
    print(f"Sampled frames: {N}")
    print(f"Embedding dim : {D}")

    # Preprocess
    embeddings_proc = preprocess_embeddings(embeddings, cfg)

    # t-SNE
    print("Running t-SNE ...")
    coords, perplexity_used = run_tsne(embeddings_proc, cfg)
    print(f"Perplexity used: {perplexity_used}")

    # Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_video = plot_by_video(coords, video_names, cfg, output_dir, timestamp)
    path_progress = plot_by_progress(coords, progress, cfg, output_dir, timestamp)

    print(f"Saved t-SNE by video to    : {path_video}")
    print(f"Saved t-SNE by progress to : {path_progress}")


if __name__ == "__main__":
    main()
