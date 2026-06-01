"""Latent Distance Heatmap evaluation task.

Given a single embedding trajectory selected from a pre-extracted H5 file,
this task computes the symmetric T×T pairwise Euclidean L2 distance matrix
M where M[i, j] = ||z_i - z_j||_2, prints compact summary statistics to
the terminal, and saves one or both of:

* a PNG **heatmap** of the full T×T matrix, and / or
* an **anchor distance curve** plot where each frame is used as an anchor and
  its distance (or similarity) to every other frame in the trajectory is drawn
  as a separate curve, with all curves overlaid in a single figure.

Math
----
For the selected trajectory z_0, ..., z_{T-1} in R^D:

    S[i, j] = ||z_i||_2^2 + ||z_j||_2^2 - 2 * z_i @ z_j   (vectorised)
    S        = clip(S, 0)                                    (numerical safety)
    M[i, j]  = sqrt(S[i, j])                                (Euclidean L2)

M is symmetric, M[i, i] == 0, all entries are non-negative.

Optional pre-processing
-----------------------
If ``normalize_embeddings=true`` in the eval config, each frame embedding is
L2-normalized before pairwise distances are computed:

    z_hat_t = z_t / max(||z_t||_2, eps)

This changes the geometry from raw latent-space Euclidean distance to
Euclidean distance on the unit sphere, which is often easier to compare across
different runs or checkpoints.

Plot mode
---------
Controlled by ``plot_mode`` in the eval config:

``heatmap``               (default) — only the T×T heatmap PNG.
``anchor_distance_curves``          — only the overlaid anchor curve PNG.
``both``                            — both PNGs saved to the same session dir.

Usage (via evaluate.py):
    Set embedding_ref, selected_video_index, and optionally plot_mode in
    configs_v2/eval/latent_distance_heatmap.yaml, then run:
        python evaluate.py --task latent_distance_heatmap
"""

import datetime
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

# Ensure projects root is on sys.path when run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PROJECTS_ROOT = os.path.abspath(os.path.join(_PROJ_ROOT, ".."))
if _PROJECTS_ROOT not in sys.path:
    sys.path.insert(0, _PROJECTS_ROOT)

from fineprog.algos.eval_task.base_task import BaseTask  # noqa: E402


# ---------------------------------------------------------------------------
# H5 helper: select one trajectory by sorted index
# ---------------------------------------------------------------------------

def _read_trajectory_by_index(h5_path: str, index: int):
    """Return (video_id, embeddings, target_steps, seq_len) for a sorted index.

    The embedding H5 is expected to have the structure produced by
    extract_embeddings.py::

        /videos/<video_id>/
            embeddings    [T_out, D]  float32
            target_steps  [T_out]     int64
            attrs:
                seq_len   int

    Parameters
    ----------
    h5_path : str
        Absolute path to the embedding H5 file.
    index : int
        0-based position in the sorted video_id list.

    Returns
    -------
    tuple of (video_id, embeddings, target_steps, seq_len)
    """
    with h5py.File(h5_path, "r") as f:
        if "videos" not in f:
            raise ValueError(
                f"[latent_distance_heatmap] H5 file has no /videos group: {h5_path}"
            )
        video_ids = sorted(f["videos"].keys())
        n_videos = len(video_ids)
        if n_videos == 0:
            raise ValueError(
                f"[latent_distance_heatmap] /videos group is empty: {h5_path}"
            )
        if not (0 <= index < n_videos):
            raise ValueError(
                f"[latent_distance_heatmap] selected_video_index={index} is out of range "
                f"[0, {n_videos - 1}] for {h5_path}"
            )
        video_id = video_ids[index]
        grp = f["videos"][video_id]
        embeddings   = np.array(grp["embeddings"],   dtype=np.float32)    # [T, D]
        target_steps = np.array(grp["target_steps"], dtype=np.float64)    # [T]
        seq_len      = int(grp.attrs.get("seq_len", len(embeddings)))
    return video_id, embeddings, target_steps, seq_len


# ---------------------------------------------------------------------------
# H5 helper: read all trajectories in sorted order
# ---------------------------------------------------------------------------

def _read_all_trajectories(h5_path: str) -> list:
    """Return list of (video_id, embeddings, target_steps, seq_len) for all videos.

    Videos are yielded in lexicographic sort order (same order as
    ``_read_trajectory_by_index``).

    Parameters
    ----------
    h5_path : str
        Absolute path to the embedding H5 file.

    Returns
    -------
    list of tuple  [(video_id, embeddings, target_steps, seq_len), ...]
    """
    records = []
    with h5py.File(h5_path, "r") as f:
        if "videos" not in f:
            raise ValueError(
                f"[latent_distance_heatmap] H5 file has no /videos group: {h5_path}"
            )
        video_ids = sorted(f["videos"].keys())
        if not video_ids:
            raise ValueError(
                f"[latent_distance_heatmap] /videos group is empty: {h5_path}"
            )
        for video_id in video_ids:
            grp = f["videos"][video_id]
            embeddings   = np.array(grp["embeddings"],   dtype=np.float32)
            target_steps = np.array(grp["target_steps"], dtype=np.float64)
            seq_len      = int(grp.attrs.get("seq_len", len(embeddings)))
            records.append((video_id, embeddings, target_steps, seq_len))
    return records


# ---------------------------------------------------------------------------
# Core computation: pairwise Euclidean L2 distance matrix
# ---------------------------------------------------------------------------

def _pairwise_l2(embs: np.ndarray) -> np.ndarray:
    """Compute the symmetric T×T pairwise Euclidean L2 distance matrix.

    Uses the numerically stable squared-expansion formula::

        S[i,j] = ||z_i||^2 + ||z_j||^2 - 2 * z_i @ z_j.T
        S       = clip(S, 0)          # remove floating-point negatives
        M       = sqrt(S)

    Parameters
    ----------
    embs : np.ndarray, shape [T, D]
        Embedding matrix for one trajectory.

    Returns
    -------
    M : np.ndarray, shape [T, T], float32
        Symmetric distance matrix with zero diagonal.
    """
    embs = embs.astype(np.float32)
    sq_norms = (embs ** 2).sum(axis=1, keepdims=True)  # [T, 1]
    # S[i, j] = ||z_i||^2 + ||z_j||^2 - 2 * z_i @ z_j
    S = sq_norms + sq_norms.T - 2.0 * (embs @ embs.T)
    np.clip(S, 0.0, None, out=S)
    return np.sqrt(S)


def _l2_normalize_rows(embs: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """L2-normalize each row of an embedding matrix.

    Parameters
    ----------
    embs : np.ndarray, shape [T, D]
        Embedding matrix for one trajectory.
    eps : float
        Minimum norm used to avoid division by zero.

    Returns
    -------
    np.ndarray, shape [T, D], float32
        Row-wise L2-normalized embedding matrix.
    """
    embs = embs.astype(np.float32, copy=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, np.float32(eps))
    return embs / norms


def _l2_to_similarity(M: np.ndarray, tau: float) -> np.ndarray:
    """Convert an L2 distance matrix to a similarity matrix via exp(-M/tau).

    Parameters
    ----------
    M   : np.ndarray, shape [T, T]  —  non-negative L2 distances.
    tau : float  —  bandwidth; at M[i,j]=tau, similarity=e^{-1}≈0.368.

    Returns
    -------
    S : np.ndarray, shape [T, T], float32
        Similarity values in (0, 1].  S[i,i]=1 (diagonal), decreasing with
        distance.
    """
    return np.exp(-M / tau).astype(np.float32)


# ---------------------------------------------------------------------------
# VOC (Velocity Ordering Consistency) helpers
# ---------------------------------------------------------------------------

def _safe_spearman_corr(values: np.ndarray, frame_indices: np.ndarray):
    """Compute Spearman rank correlation between *values* and *frame_indices*.

    Returns the correlation as a float, or ``None`` when the input is too
    small (< 2 samples), constant (zero variance), or numerically invalid.

    Parameters
    ----------
    values : np.ndarray, shape [N]
        Per-frame quantity to correlate (distances or negated distances).
    frame_indices : np.ndarray, shape [N]
        Corresponding frame indices (monotonically increasing integers).

    Returns
    -------
    float | None
    """
    n = len(values)
    if n < 2:
        return None
    # Constant inputs produce zero-variance rank sequences → NaN correlation.
    if np.all(values == values[0]):
        return None
    corr = float(spearmanr(values, frame_indices).correlation)
    if not np.isfinite(corr):
        return None
    return corr


def _compute_anchor_voc_from_row(
    anchor_row: np.ndarray,
    anchor_index: int,
):
    """Compute per-anchor VOC from one row of the pairwise distance matrix.

    For **future frames** (t > anchor_index)::

        VOC_future = Spearman( {d[anchor, t]},   {t} )

    Distances are expected to grow as t moves further into the future, so
    positive correlation indicates good temporal structure.

    For **past frames** (t < anchor_index), distances are *negated* before
    computing the correlation::

        VOC_past = Spearman( {-d[anchor, t]},  {t} )

    Negation ensures that closer-in-time past frames (small d) get a higher
    negated value, yielding positive correlation with t when the latent space
    has consistent temporal ordering.

    The per-anchor VOC is the mean of available (non-None) sides.  Returns
    ``None`` when both sides are invalid (< 2 frames on each side, or NaN).

    Parameters
    ----------
    anchor_row : np.ndarray, shape [T]
        Row ``a`` from the T×T distance matrix: d[a, 0], …, d[a, T-1].
    anchor_index : int
        The index ``a`` of this anchor frame.

    Returns
    -------
    float | None
    """
    T       = len(anchor_row)
    indices = np.arange(T, dtype=np.float64)

    # ── Future side: t > anchor_index ─────────────────────────────────────
    future_mask    = indices > anchor_index
    future_indices = indices[future_mask]
    future_dists   = anchor_row[future_mask].astype(np.float64)
    voc_future     = _safe_spearman_corr(future_dists, future_indices)

    # ── Past side: t < anchor_index (negate distances) ────────────────────
    past_mask    = indices < anchor_index
    past_indices = indices[past_mask]
    past_dists   = anchor_row[past_mask].astype(np.float64)
    voc_past     = _safe_spearman_corr(-past_dists, past_indices)

    # ── mean_valid: average non-None sides ────────────────────────────────
    valid = [v for v in (voc_past, voc_future) if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _compute_video_voc(distance_matrix: np.ndarray) -> dict:
    """Compute the VOC score for a single video trajectory.

    Iterates over all T frames as anchors, computes per-anchor VOC via
    :func:`_compute_anchor_voc_from_row`, skips anchors whose VOC is ``None``,
    and returns the mean over valid anchors.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape [T, T]
        Raw (non-similarity-converted) pairwise L2 distance matrix.

    Returns
    -------
    dict with keys:
        voc       float | None  — video-level VOC, or None if no valid anchor.
        n_anchors int           — total number of anchors = T.
        n_valid   int           — anchors that contributed a valid VOC value.
    """
    T = distance_matrix.shape[0]
    anchor_vocs: list = []
    for a in range(T):
        v = _compute_anchor_voc_from_row(distance_matrix[a, :], anchor_index=a)
        if v is not None:
            anchor_vocs.append(v)
    n_valid = len(anchor_vocs)
    voc     = float(np.mean(anchor_vocs)) if anchor_vocs else None
    return {"voc": voc, "n_anchors": T, "n_valid": n_valid}


# ---------------------------------------------------------------------------
# Compact terminal log
# ---------------------------------------------------------------------------

def _log_matrix_stats(
    M: np.ndarray,
    video_id: str,
    percentiles: list,
    mode: str = "distance",
) -> None:
    """Print shape and off-diagonal summary statistics; never print the matrix."""
    T = M.shape[0]
    # Mask out diagonal for off-diagonal statistics
    mask = ~np.eye(T, dtype=bool)
    off_diag = M[mask]

    label = "similarity" if mode == "similarity" else "distance"
    print(f"[latent_distance_heatmap] video_id             : {video_id}")
    print(f"[latent_distance_heatmap] {label}_matrix shape : {M.shape}")
    print(f"[latent_distance_heatmap] off-diagonal {label} stats  :")
    print(f"  min    = {float(off_diag.min()):.6f}")
    print(f"  max    = {float(off_diag.max()):.6f}")
    print(f"  mean   = {float(off_diag.mean()):.6f}")
    print(f"  median = {float(np.median(off_diag)):.6f}")
    if percentiles:
        for p in percentiles:
            print(f"  p{p:02d}    = {float(np.percentile(off_diag, p)):.6f}")
    diag = np.diag(M)
    if mode == "similarity":
        print(f"[latent_distance_heatmap] diagonal mean (expect ~1): {float(diag.mean()):.6f}")
    else:
        print(f"[latent_distance_heatmap] diagonal max (expect ~0): {float(diag.max()):.2e}")


# ---------------------------------------------------------------------------
# Heatmap saver
# ---------------------------------------------------------------------------

def _save_heatmap(
    M: np.ndarray,
    video_id: str,
    target_steps: np.ndarray,
    output_dir: str,
    colormap: str = "viridis",
    figsize: tuple = (8, 7),
    dpi: int = 150,
    show_colorbar: bool = True,
    mode: str = "distance",
    tau: float = 0.1,
    normalized_embeddings: bool = False,
) -> str:
    """Save a T×T heatmap PNG of the distance matrix.

    Axis ticks show target_steps values if they are non-trivially monotonic;
    otherwise frame indices are used.

    Parameters
    ----------
    M : np.ndarray, shape [T, T]
    video_id : str
    target_steps : np.ndarray, shape [T]
    output_dir : str
    colormap : str
    figsize : tuple
    dpi : int
    show_colorbar : bool

    Returns
    -------
    str  Absolute path to the saved PNG.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    T = M.shape[0]

    # Decide axis labels: use target_steps if they are distinct and monotonic
    steps_monotone = (
        len(target_steps) == T
        and bool(np.all(np.diff(target_steps) > 0))
    )
    tick_labels = target_steps.astype(int).tolist() if steps_monotone else list(range(T))

    # Thin out ticks if trajectory is long to avoid label clutter
    max_ticks = 20
    if T > max_ticks:
        step_every = max(1, T // max_ticks)
        tick_positions = list(range(0, T, step_every))
        tick_labels_shown = [tick_labels[i] for i in tick_positions]
    else:
        tick_positions = list(range(T))
        tick_labels_shown = tick_labels

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap=colormap, aspect="equal", origin="upper")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_shown, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels_shown, fontsize=7)

    axis_label = "frame step" if steps_monotone else "frame index"
    ax.set_xlabel(axis_label, fontsize=9)
    ax.set_ylabel(axis_label, fontsize=9)
    normalization_tag = " | L2-normalized emb" if normalized_embeddings else " | raw emb"
    if mode == "similarity":
        title_prefix   = f"RBF Similarity (τ={tau})"
        colorbar_label = f"similarity  exp(-d/τ={tau})"
    else:
        title_prefix   = "Pairwise L2 distance"
        colorbar_label = "L2 distance"
    ax.set_title(
        f"{title_prefix}{normalization_tag}  |  video: {video_id}  |  T={T}",
        fontsize=10,
    )

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(colorbar_label, fontsize=8)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"latent_distance_heatmap_{video_id}.png")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Anchor distance curve saver
# ---------------------------------------------------------------------------

def _save_anchor_distance_curves(
    M: np.ndarray,
    video_id: str,
    target_steps: np.ndarray,
    output_dir: str,
    curve_colormap: str = "viridis",
    curve_alpha: float = 0.35,
    curve_linewidth: float = 0.9,
    figsize: tuple = (10, 5),
    dpi: int = 150,
    mode: str = "distance",
    tau: float = 0.1,
    normalized_embeddings: bool = False,
    anchor_fraction: float = 1.0,
) -> str:
    """Save a multi-curve PNG where each (selected) anchor frame produces one curve.

    For anchor frame *a*, the plotted curve is ``M[a, :]``, i.e. the distance
    (or similarity) from frame *a* to every other frame in the trajectory.
    Selected anchor curves are overlaid in a single axes, coloured by each
    anchor's position in the full trajectory using *curve_colormap*.  No
    per-curve legend is added to keep the figure legible for long trajectories.

    Parameters
    ----------
    M : np.ndarray, shape [T, T]
        Distance (or similarity) matrix produced by ``_pairwise_l2`` /
        ``_l2_to_similarity``.
    video_id : str
    target_steps : np.ndarray, shape [T]
    output_dir : str
    curve_colormap : str
        Matplotlib colormap used to colour curves by anchor position.
    curve_alpha : float
        Opacity of each individual curve (0–1).  Lower values reduce
        over-plotting clutter for long trajectories.
    curve_linewidth : float
    figsize : tuple
    dpi : int
    mode : str
        ``"distance"`` or ``"similarity"`` — controls axis label and title.
    tau : float
        Bandwidth shown in title when mode=="similarity".
    normalized_embeddings : bool
    anchor_fraction : float
        Fraction of anchor frames to plot, in (0, 1].  Frames are sampled
        evenly across the trajectory so coverage is uniform.  1.0 (default)
        plots all T frames; 0.1 plots ~10% of frames.

    Returns
    -------
    str  Absolute path to the saved PNG.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.cm as mcm      # noqa: PLC0415

    T = M.shape[0]

    # ── x-axis: prefer target_steps if monotonically increasing ──────────
    steps_monotone = (
        len(target_steps) == T
        and bool(np.all(np.diff(target_steps) > 0))
    )
    xs = target_steps.astype(np.float32) if steps_monotone else np.arange(T, dtype=np.float32)
    x_label = "frame step" if steps_monotone else "frame index"

    # ── x-axis tick thinning (mirrors _save_heatmap logic) ───────────────
    max_ticks = 20
    if T > max_ticks:
        step_every = max(1, T // max_ticks)
        tick_xs = xs[::step_every]
    else:
        tick_xs = xs

    # ── select anchor indices (evenly spaced subsample) ──────────────────
    anchor_fraction = float(np.clip(anchor_fraction, 1e-6, 1.0))
    n_anchors = max(1, round(T * anchor_fraction))
    anchor_indices = np.round(np.linspace(0, T - 1, n_anchors)).astype(int)
    anchor_indices = np.unique(anchor_indices)  # deduplicate after rounding
    print(
        f"[latent_distance_heatmap] anchor_fraction={anchor_fraction:.3g}  →  "
        f"plotting {len(anchor_indices)}/{T} anchor curves"
    )

    # ── colour each anchor by its normalised position along the trajectory ─
    cmap = mcm.get_cmap(curve_colormap)

    fig, ax = plt.subplots(figsize=figsize)
    for a in anchor_indices:
        color = cmap(int(a) / max(T - 1, 1))
        ax.plot(xs, M[a, :], color=color, alpha=curve_alpha, linewidth=curve_linewidth)

    ax.set_xticks(tick_xs)
    ax.set_xticklabels([str(int(v)) for v in tick_xs], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel(x_label, fontsize=9)

    normalization_tag = " | L2-normalized emb" if normalized_embeddings else " | raw emb"
    n_plotted = len(anchor_indices)
    anchor_tag = f"anchors: {n_plotted}/{T}" if n_plotted < T else f"anchors: {T}"
    if mode == "similarity":
        y_label      = f"similarity  exp(-d/τ={tau})"
        title_prefix = f"Anchor distance curves — RBF similarity (τ={tau})"
    else:
        y_label      = "L2 distance"
        title_prefix = "Anchor distance curves — pairwise L2"
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(
        f"{title_prefix}{normalization_tag}  |  video: {video_id}  |  T={T}  |  {anchor_tag}",
        fontsize=10,
    )

    # ── colourbar as anchor-position legend ───────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("anchor frame index", fontsize=8)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"anchor_distance_curves_{video_id}.png")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class LatentDistanceHeatmapTask(BaseTask):
    """Pairwise Euclidean L2 distance heatmap for one embedding trajectory.

    Reads the target H5 directly from the resolved V2 config; the
    ``embeddings_dataset`` argument to ``evaluate()`` is ignored (pass None).
    """

    def __init__(self):
        super().__init__(task_name="latent_distance_heatmap", downstream_task=False)
        self.config: dict = {}

    def configure(self, config: dict) -> None:
        """Store the resolved V2 config dict produced by ConfigV2.load_eval()."""
        self.config = config

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, embeddings_dataset=None) -> dict:  # noqa: ARG002
        cfg = self.config

        # ── required config fields ────────────────────────────────────
        h5_path   = cfg["embedding_h5_path"]
        selector  = cfg.get("selected_video_index", 0)
        output_dir = cfg.get("output_dir") or os.path.join(
            _PROJ_ROOT, "outputs", "latent_distance_heatmap"
        )

        # ── optional visualization / logging fields ───────────────────
        plot_mode             = str(cfg.get("plot_mode",               "heatmap"))
        colormap              = str(cfg.get("colormap",              "viridis"))
        figsize               = tuple(cfg.get("figsize",              [8, 7]))
        dpi                   = int(cfg.get("dpi",                    150))
        show_colorbar         = bool(cfg.get("show_colorbar",          True))
        percentiles           = list(cfg.get("log_percentiles",        [25, 75, 95]))
        convert_to_similarity = bool(cfg.get("convert_to_similarity",  False))
        similarity_tau        = float(cfg.get("similarity_tau",         0.1))
        normalize_embeddings  = bool(cfg.get("normalize_embeddings",   False))
        normalization_eps     = float(cfg.get("normalization_eps",      1.0e-12))
        curve_colormap        = str(cfg.get("curve_colormap",          "viridis"))
        curve_alpha           = float(cfg.get("curve_alpha",            0.35))
        curve_linewidth       = float(cfg.get("curve_linewidth",        0.9))
        curve_figsize         = tuple(cfg.get("curve_figsize",         [10, 5]))
        curve_anchor_fraction = float(cfg.get("curve_anchor_fraction",  1.0))
        eval_timestamp        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_output_dir    = os.path.join(
            output_dir,
            f"tau{similarity_tau:g}_{eval_timestamp}",
        )

        # "all" (string, case-insensitive) → process every video in the H5
        run_all = isinstance(selector, str) and selector.strip().lower() == "all"

        print()
        print(f"[latent_distance_heatmap] embedding_h5_path   : {h5_path}")
        print(f"[latent_distance_heatmap] selected_video_index: {selector}")
        print(f"[latent_distance_heatmap] output_dir          : {output_dir}")
        print(f"[latent_distance_heatmap] session_output_dir  : {session_output_dir}")
        print(f"[latent_distance_heatmap] normalize_embeddings: {normalize_embeddings}")
        print(f"[latent_distance_heatmap] plot_mode           : {plot_mode}")

        # ── inner helper: process one (video_id, embs, steps, seq_len) ──
        def _process_one(video_id, embs, target_steps, seq_len):
            embs         = embs[:seq_len]
            target_steps = target_steps[:seq_len]
            T, D = embs.shape
            print(f"[latent_distance_heatmap] --- {video_id}  shape: ({T}, {D}) ---")
            if normalize_embeddings:
                raw_norms = np.linalg.norm(embs, axis=1)
                embs = _l2_normalize_rows(embs, eps=normalization_eps)
                normed_norms = np.linalg.norm(embs, axis=1)
                print(
                    "[latent_distance_heatmap] embedding norm stats  : "
                    f"raw_mean={float(raw_norms.mean()):.6f}, "
                    f"raw_min={float(raw_norms.min()):.6f}, "
                    f"raw_max={float(raw_norms.max()):.6f}"
                )
                print(
                    "[latent_distance_heatmap] after L2-normalize    : "
                    f"mean={float(normed_norms.mean()):.6f}, "
                    f"min={float(normed_norms.min()):.6f}, "
                    f"max={float(normed_norms.max()):.6f}"
                )
            M = _pairwise_l2(embs)
            # ── VOC: always computed on raw L2 distances ───────────────────
            voc_result      = _compute_video_voc(M)
            voc_val         = voc_result["voc"]
            n_valid_anchors = voc_result["n_valid"]
            n_total_anchors = voc_result["n_anchors"]
            print(
                "[latent_distance_heatmap] VOC (Spearman)        : "
                + (f"{voc_val:.6f}" if voc_val is not None else "N/A")
                + f"  ({n_valid_anchors}/{n_total_anchors} valid anchors)"
            )
            if convert_to_similarity:
                M_vis = _l2_to_similarity(M, similarity_tau)
                mode  = "similarity"
            else:
                M_vis = M
                mode  = "distance"
            _log_matrix_stats(M_vis, video_id, percentiles, mode=mode)
            mask       = ~np.eye(T, dtype=bool)
            mean_val   = float(M_vis[mask].mean()) if T > 1 else 0.0
            legacy_key = "mean_offdiag_similarity" if convert_to_similarity else "mean_offdiag_l2_distance"

            heatmap_path = None
            curve_path   = None

            if plot_mode in ("heatmap", "both"):
                heatmap_path = _save_heatmap(
                    M=M_vis,
                    video_id=video_id,
                    target_steps=target_steps,
                    output_dir=session_output_dir,
                    colormap=colormap,
                    figsize=figsize,
                    dpi=dpi,
                    show_colorbar=show_colorbar,
                    mode=mode,
                    tau=similarity_tau,
                    normalized_embeddings=normalize_embeddings,
                )
                print(f"[latent_distance_heatmap] heatmap saved      : {heatmap_path}")

            if plot_mode in ("anchor_distance_curves", "both"):
                curve_path = _save_anchor_distance_curves(
                    M=M_vis,
                    video_id=video_id,
                    target_steps=target_steps,
                    output_dir=session_output_dir,
                    curve_colormap=curve_colormap,
                    curve_alpha=curve_alpha,
                    curve_linewidth=curve_linewidth,
                    figsize=curve_figsize,
                    dpi=dpi,
                    mode=mode,
                    tau=similarity_tau,
                    normalized_embeddings=normalize_embeddings,
                    anchor_fraction=curve_anchor_fraction,
                )
                print(f"[latent_distance_heatmap] curve plot saved   : {curve_path}")

            return {
                "video_id":              video_id,
                "voc_spearman":          voc_val,
                "voc_n_valid_anchors":   n_valid_anchors,
                "voc_n_total_anchors":   n_total_anchors,
                legacy_key:              mean_val,
                "output_heatmap_path":   heatmap_path,
                "output_curve_path":     curve_path,
                "distance_matrix_shape": list(M.shape),
                "normalize_embeddings":  normalize_embeddings,
            }

        # ── mode: all videos ──────────────────────────────────────────
        if run_all:
            records  = _read_all_trajectories(h5_path)
            n_videos = len(records)
            print(f"[latent_distance_heatmap] mode: all ({n_videos} videos)")

            per_video = []
            for vid_id, embs, steps, seq_len in records:
                per_video.append(_process_one(vid_id, embs, steps, seq_len))

            legacy_key   = "mean_offdiag_similarity" if convert_to_similarity else "mean_offdiag_l2_distance"
            mean_vals    = [r[legacy_key] for r in per_video]
            global_mean  = float(np.mean(mean_vals)) if mean_vals else 0.0
            valid_vocs     = [r["voc_spearman"] for r in per_video if r["voc_spearman"] is not None]
            skipped_ids    = [r["video_id"] for r in per_video if r["voc_spearman"] is None]
            n_valid_videos = len(valid_vocs)
            if not valid_vocs:
                raise ValueError(
                    f"[latent_distance_heatmap] No valid video VOC values: "
                    f"all {n_videos} videos were skipped (trajectories too short "
                    f"for rank correlation)."
                )
            dataset_voc = float(np.mean(valid_vocs))
            print()
            print(
                f"[latent_distance_heatmap] Dataset VOC (Spearman): {dataset_voc:.6f}  "
                f"({n_valid_videos}/{n_videos} valid videos)"
            )
            print(f"[latent_distance_heatmap] global {legacy_key}: {global_mean:.6f}")
            if skipped_ids:
                print(f"[latent_distance_heatmap] skipped video_ids   : {skipped_ids}")
            print(f"[latent_distance_heatmap] {n_videos} heatmaps saved to: {session_output_dir}")

            return {
                "task_name":              "latent_distance_heatmap",
                "metric_name":            "voc_spearman",
                "metric_value":           dataset_voc,
                "n_videos":               n_videos,
                "voc_n_valid_videos":     n_valid_videos,
                "voc_skipped_video_ids":  skipped_ids,
                legacy_key:               global_mean,
                "output_heatmap_dir":     session_output_dir,
                "output_heatmap_paths":   [r["output_heatmap_path"] for r in per_video],
                "output_curve_paths":     [r["output_curve_path"]   for r in per_video],
                "per_video_results":      per_video,
            }

        # ── mode: single video by index ───────────────────────────────
        video_index = int(selector)
        video_id, embs, target_steps, seq_len = _read_trajectory_by_index(
            h5_path, video_index
        )
        res = _process_one(video_id, embs, target_steps, seq_len)

        legacy_key = "mean_offdiag_similarity" if convert_to_similarity else "mean_offdiag_l2_distance"
        voc_val    = res["voc_spearman"]
        if voc_val is None:
            raise ValueError(
                f"[latent_distance_heatmap] No valid anchors for video '{res['video_id']}': "
                f"T={res['distance_matrix_shape'][0]}; VOC cannot be computed."
            )
        return {
            "task_name":              "latent_distance_heatmap",
            "metric_name":            "voc_spearman",
            "metric_value":           voc_val,
            "voc_n_valid_anchors":    res["voc_n_valid_anchors"],
            "voc_n_total_anchors":    res["voc_n_total_anchors"],
            legacy_key:               res[legacy_key],
            "output_heatmap_dir":     session_output_dir,
            "output_heatmap_path":    res["output_heatmap_path"],
            "output_heatmap_paths":   [res["output_heatmap_path"]],  # normalized list (mirrors all-mode)
            "output_curve_path":      res["output_curve_path"],
            "output_curve_paths":     [res["output_curve_path"]],    # normalized list (mirrors all-mode)
            "selected_video_index":   video_index,
            "selected_video_id":      res["video_id"],
            "distance_matrix_shape":  res["distance_matrix_shape"],
            "normalize_embeddings":   normalize_embeddings,
        }
