"""Kendall's Tau evaluation task.

Measures how well the temporal ordering of one video's embeddings is preserved
when mapped to another video via nearest-neighbour lookup in embedding space.

Reference: google-research/tcc/evaluation/kendalls_tau.py
"""

import os
import sys

# Ensure the projects root is on sys.path so that `fineprog` is importable
# both when this file is run directly and when imported as part of the package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECTS_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
if _PROJECTS_ROOT not in sys.path:
    sys.path.insert(0, _PROJECTS_ROOT)

import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

from fineprog.algos.eval_task.base_task import BaseTask

# outputs/ lives under the fineprog project root (4 levels up from this file)
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_OUTPUTS_DIR = os.path.join(_PROJ_ROOT, "outputs", "kendall_heatmap")


def _compute_kendalls_tau(embs_list, stride=1, distance="sqeuclidean"):
    """Compute mean Kendall's Tau across all ordered pairs of sequences.

    Args:
        embs_list: List of arrays, each shape [T_i, D].
        stride:    Down-sampling stride applied to both query and candidate.
        distance:  Distance metric passed to scipy.spatial.distance.cdist.

    Returns:
        Tuple (mean_tau, tau_matrix):
            mean_tau   – Mean Kendall's Tau (float). Returns 0.0 if all NaN.
            tau_matrix – np.ndarray [N, N], NaN on diagonal; tau_matrix[i,j]
                         is the tau for query=i, candidate=j.
    """
    num_seqs = len(embs_list)
    tau_matrix = np.full((num_seqs, num_seqs), np.nan)
    taus = []

    for i in range(num_seqs):
        query_feats = embs_list[i][::stride]          # [T_q, D]
        for j in range(num_seqs):
            if i == j:
                continue
            candidate_feats = embs_list[j][::stride]  # [T_c, D]

            # Pairwise distance matrix: [T_q, T_c]
            dists = cdist(query_feats, candidate_feats, metric=distance)

            # Nearest-neighbour index in candidate for each query frame
            nns = np.argmin(dists, axis=1)            # [T_q]

            tau_val = kendalltau(np.arange(len(nns)), nns).correlation
            if not np.isnan(tau_val):
                tau_matrix[i, j] = tau_val
                taus.append(tau_val)

    mean_tau = float(np.mean(taus)) if taus else 0.0
    return mean_tau, tau_matrix


def _print_kendall_report(tau_matrix: np.ndarray, video_ids: list) -> None:
    """Print a pairwise Kendall's Tau table (row=query, col=candidate)."""
    n = len(video_ids)
    # Truncate video_id labels to at most 8 chars for readability
    labels = [str(vid)[-8:] for vid in video_ids]
    col_w = max(8, max(len(l) for l in labels) + 1)

    header_sep = "-" * (col_w + 1 + n * (col_w + 1))
    print()
    print("Pairwise Kendall's Tau  (row = query → col = candidate)")
    print(header_sep)
    # Column headers
    print(" " * (col_w + 1) + "".join(f"{l:>{col_w}} " for l in labels))
    print(header_sep)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>{col_w}} "
        for j in range(n):
            if i == j:
                row += f"{'---':>{col_w}} "
            elif np.isnan(tau_matrix[i, j]):
                row += f"{'nan':>{col_w}} "
            else:
                row += f"{tau_matrix[i, j]:>{col_w}.4f} "
        print(row)
    print(header_sep)
    # Per-video mean (excluding self)
    print("Row means (query avg tau):")
    for i, label in enumerate(labels):
        vals = [tau_matrix[i, j] for j in range(n) if i != j and not np.isnan(tau_matrix[i, j])]
        mean_val = float(np.mean(vals)) if vals else float("nan")
        print(f"  {label:>{col_w}}: {mean_val:.4f}")
    print(header_sep)
    print()


def _save_kendall_heatmap(
    tau_matrix: np.ndarray,
    video_ids: list,
    mean_tau: float,
    output_dir: str = None,
) -> str:
    """Save the pairwise Kendall's Tau matrix as a discrete heatmap PNG.

    Args:
        tau_matrix: [N, N] float array, NaN on diagonal.
        video_ids:  List of video identifier strings.
        mean_tau:   Overall mean tau (shown in title).
        output_dir: Directory to save the PNG.  Falls back to ``_OUTPUTS_DIR``.

    Returns:
        Absolute path to the saved PNG file.
    """
    if output_dir is None:
        output_dir = _OUTPUTS_DIR
    n = len(video_ids)
    labels = [str(vid)[-12:] for vid in video_ids]

    # Build display matrix: NaN on diagonal stays NaN for masking
    disp = tau_matrix.copy()

    fig_size = max(5, n * 0.7 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Mask diagonal so it renders as grey
    masked = np.ma.array(disp, mask=np.eye(n, dtype=bool))

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#cccccc")  # diagonal → grey

    im = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#666666")
            elif np.isnan(tau_matrix[i, j]):
                ax.text(j, i, "nan", ha="center", va="center", fontsize=7, color="black")
            else:
                val = tau_matrix[i, j]
                text_color = "black" if abs(val) < 0.6 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=max(6, min(9, int(72 / n))), color=text_color)

    # Axes labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Candidate video", fontsize=9)
    ax.set_ylabel("Query video", fontsize=9)
    ax.set_title(f"Pairwise Kendall's Tau  (mean = {mean_tau:.4f})", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"kendall_heatmap_{ts}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


class KendallsTauTask(BaseTask):
    """Kendall's Tau alignment evaluation on pre-extracted embeddings."""

    def __init__(
        self,
        distance: str = "sqeuclidean",
        stride: int = 1,
        visualize: bool = False,
        config_path: str = None,
        output_dir: str = None,
    ):
        super().__init__(task_name="kendalls_tau", downstream_task=True)
        self.distance = distance
        self.stride = stride
        self.visualize = visualize  # reserved; not used in current implementation
        self.output_dir = output_dir  # None → falls back to _OUTPUTS_DIR

    def configure(self, config: dict) -> None:
        """Apply resolved V2 config dict to this task."""
        self.stride   = config.get("kendall_stride",   self.stride)
        self.distance = config.get("kendall_distance", self.distance)
        if config.get("output_dir"):
            self.output_dir = config["output_dir"]

    def evaluate(self, embeddings_dataset: dict) -> dict:
        """Compute Kendall's Tau on a pre-extracted embeddings dataset.

        Args:
            embeddings_dataset: dict with at least:
                "embeddings"   – list of np.ndarray, each [T_i, D]
                "target_steps" – list of np.ndarray, each [T_i]  (not used here)
                "seq_len"      – list of int
                "video_id"     – list of identifiers
                "action_id"    – list of identifiers

        Returns:
            {
                "task_name":    "kendalls_tau",
                "metric_name":  "kendalls_tau",
                "metric_value": float
            }
        """
        embs_list = embeddings_dataset["embeddings"]

        # Truncate each sequence to its declared length if provided
        seq_lens = embeddings_dataset.get("seq_len")
        if seq_lens is not None:
            embs_list = [
                np.asarray(e)[: int(l)]
                for e, l in zip(embs_list, seq_lens)
            ]
        else:
            embs_list = [np.asarray(e) for e in embs_list]

        video_ids = embeddings_dataset.get("video_id", [str(i) for i in range(len(embs_list))])

        tau, tau_matrix = _compute_kendalls_tau(embs_list, stride=self.stride, distance=self.distance)

        heatmap_path = None
        if len(embs_list) > 1:
            _print_kendall_report(tau_matrix, video_ids)
            heatmap_path = _save_kendall_heatmap(tau_matrix, video_ids, tau, output_dir=self.output_dir)
            print(f"[kendalls_tau] heatmap saved → {heatmap_path}")

        result = {
            "task_name": self.task_name,
            "metric_name": "kendalls_tau",
            "metric_value": tau,
        }
        if heatmap_path is not None:
            result["output_heatmap_path"]  = heatmap_path
            result["output_heatmap_paths"] = [heatmap_path]
        return result


# ---------------------------------------------------------------------------
# Minimal sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Dummy test
    rng = np.random.default_rng(42)
    T, D = 20, 16

    # Video 0 and 1: nearly monotone embeddings → high tau expected
    base = rng.standard_normal((T, D)).cumsum(axis=0)
    noise = rng.standard_normal((T, D)) * 0.05

    emb0 = base + noise
    emb1 = base + rng.standard_normal((T, D)) * 0.05  # similar ordering

    # Video 2: reversed order → should drag mean down
    emb2 = emb0[::-1].copy()

    dataset = {
        "video_id": ["vid0", "vid1", "vid2"],
        "embeddings": [emb0, emb1, emb2],
        "target_steps": [np.arange(T)] * 3,
        "seq_len": [T, T, T],
        "action_id": [0, 0, 0],
    }

    task = KendallsTauTask(distance="sqeuclidean", stride=1)
    result = task.evaluate(dataset)

    print(f"task_name      : {result['task_name']}")
    print(f"metric_name    : {result['metric_name']}")
    print(f"metric_value   : {result['metric_value']:.4f}")
