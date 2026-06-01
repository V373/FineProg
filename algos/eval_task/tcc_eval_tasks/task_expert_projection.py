"""Expert Projection evaluation task for TCC embeddings.

Given a single expert embedding trajectory (H5) and a set of non-expert video
embeddings (H5), this task projects each non-expert frame embedding onto the
expert trajectory via TCC-style soft nearest-neighbour matching.

No model training, no backprop, no encoder re-run.  Operates purely on
pre-extracted H5 embeddings.

Usage (via evaluate.py):
    Set the task config in configs_v2/eval/expert_projection.yaml and run:
        python evaluate.py --task expert_projection
"""

import datetime
import os
import re
import time
from pathlib import Path

import h5py
import numpy as np

from fineprog.algos.eval_task.base_task import BaseTask

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_VIZ_BASE_DIR = os.path.join(_PROJ_ROOT, "outputs", "expert_projection")


# ---------------------------------------------------------------------------
# Helper: numerically stable softmax along an axis
# ---------------------------------------------------------------------------

def _build_demo_name_map(raw_hdf5_path: str, mask_key: str) -> dict:
    """Return {video_id_str: demo_name_str} for a Robomimic mask.

    Reads the mask from *raw_hdf5_path* under /mask/<mask_key>, extracts the
    numeric suffix from each entry (accepting both 'demo_N' and bare 'N'),
    sorts numerically, and maps the 1-based sequential embedding video_id
    (e.g. '000001') to the original demo name (e.g. 'demo_142').
    Returns an empty dict on any failure so callers degrade gracefully.
    """
    try:
        with h5py.File(raw_hdf5_path, "r") as f:
            if "mask" not in f or mask_key not in f["mask"]:
                print(
                    f"[expert_projection] demo_name_map: mask key '{mask_key}' "
                    f"not found in {raw_hdf5_path} – using video_ids as labels"
                )
                return {}
            raw = f["mask"][mask_key][()]
        indices = []
        names = []
        for x in raw:
            if isinstance(x, (bytes, np.bytes_)):
                x = x.decode("utf-8")
            s = str(x)
            m = re.search(r"\d+$", s)
            if m:
                num = int(m.group())
                indices.append(num)
                # Normalise to demo_N format regardless of input style
                names.append(f"demo_{num}" if not s.startswith("demo_") else s)
        paired = sorted(zip(indices, names), key=lambda t: t[0])
        return {
            str(rank).zfill(6): demo_name
            for rank, (_, demo_name) in enumerate(paired, 1)
        }
    except Exception as exc:
        print(f"[expert_projection] demo_name_map: failed ({exc}) – using video_ids as labels")
        return {}


def _stable_softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    """Return softmax(logits) computed stably along *axis*."""
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Helper: read a single expert embedding trajectory from an H5 file
# ---------------------------------------------------------------------------

def _read_expert_h5(path: str):
    """Return (embeddings, target_steps) for the expert H5 file.

    Supports two layouts:
        Format A: /embeddings  and  /target_steps  (flat)
        Format B: /videos/<one_video_id>/embeddings  and  /target_steps
    """
    with h5py.File(path, "r") as f:
        if "embeddings" in f:
            # Format A
            embs = np.array(f["embeddings"])
            if "target_steps" in f:
                steps = np.array(f["target_steps"])
            else:
                steps = np.arange(embs.shape[0])
        elif "videos" in f:
            # Format B – take the first video group
            video_ids = sorted(f["videos"].keys())
            if not video_ids:
                raise ValueError(f"Expert H5 has an empty /videos group: {path}")
            grp = f["videos"][video_ids[0]]
            embs = np.array(grp["embeddings"])
            if "target_steps" in grp:
                steps = np.array(grp["target_steps"])
            else:
                steps = np.arange(embs.shape[0])
        else:
            raise ValueError(
                f"Expert H5 file has neither /embeddings nor /videos: {path}"
            )
    return embs.astype(np.float32), steps.astype(np.float64)


# ---------------------------------------------------------------------------
# Helper: read all non-expert video embeddings from an H5 file
# ---------------------------------------------------------------------------

def _read_nonexpert_h5(path: str):
    """Return list of (video_id, embeddings, target_steps) tuples.

    Expects standard format:
        /videos/<video_id>/embeddings
        /videos/<video_id>/target_steps  (optional)
    """
    records = []
    with h5py.File(path, "r") as f:
        if "videos" not in f:
            raise ValueError(f"Non-expert H5 has no /videos group: {path}")
        for video_id in sorted(f["videos"].keys()):
            grp = f["videos"][video_id]
            embs = np.array(grp["embeddings"]).astype(np.float32)
            if "target_steps" in grp:
                steps = np.array(grp["target_steps"]).astype(np.float64)
            else:
                steps = np.arange(embs.shape[0], dtype=np.float64)
            records.append((video_id, embs, steps))
    return records


# ---------------------------------------------------------------------------
# Core projection logic (pure numpy, no torch)
# ---------------------------------------------------------------------------

def _project_one_video(
    nonexpert_embs: np.ndarray,   # [T_q, D]
    nonexpert_steps: np.ndarray,  # [T_q]
    expert_embs: np.ndarray,      # [T_e, D]
    expert_steps: np.ndarray,     # [T_e]
    temperature: float,
    save_entropy: bool,
    save_alpha: bool,
) -> dict:
    """Run SNN projection for one non-expert video.

    Returns a dict with all per-frame outputs and per-video summary metrics.
    """
    T_q, D = nonexpert_embs.shape
    T_e = expert_embs.shape[0]

    # 1. Pairwise squared Euclidean distance  [T_q, T_e]
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    q_sq = (nonexpert_embs ** 2).sum(axis=1, keepdims=True)   # [T_q, 1]
    e_sq = (expert_embs ** 2).sum(axis=1, keepdims=True).T    # [1, T_e]
    dists = q_sq + e_sq - 2.0 * nonexpert_embs @ expert_embs.T  # [T_q, T_e]
    # Numerical noise can produce tiny negatives; clip to 0
    np.clip(dists, 0.0, None, out=dists)

    # 2. Soft nearest-neighbour weights  [T_q, T_e]
    logits = -dists / temperature
    alpha = _stable_softmax(logits, axis=1)

    # 3. Projected embeddings  [T_q, D]
    projected_embs = alpha @ expert_embs

    # 4. Hard nearest neighbour
    nn_indices = np.argmin(dists, axis=1)                       # [T_q]
    nn_expert_steps = expert_steps[nn_indices]                  # [T_q]
    nn_distances = dists[np.arange(T_q), nn_indices]            # [T_q]

    # 5. Soft expected expert index / step
    expert_indices_float = np.arange(T_e, dtype=np.float64)
    soft_expert_index = alpha @ expert_indices_float             # [T_q]
    soft_expert_step = alpha @ expert_steps.astype(np.float64)  # [T_q]

    result = {
        "projected_embs":    projected_embs,          # [T_q, D]
        "nn_indices":        nn_indices,               # [T_q]  int
        "nn_expert_steps":   nn_expert_steps,          # [T_q]
        "nn_distances":      nn_distances,             # [T_q]
        "soft_expert_index": soft_expert_index,        # [T_q]
        "soft_expert_step":  soft_expert_step,         # [T_q]
        "mean_hard_nn_distance": float(nn_distances.mean()),
        "mean_soft_expert_step": float(soft_expert_step.mean()),
    }

    # 6. Optional entropy
    if save_entropy:
        eps = 1e-12
        entropy = -(alpha * np.log(alpha + eps)).sum(axis=1)    # [T_q]
        norm_entropy = entropy / np.log(T_e + eps)
        result["entropy"] = entropy
        result["normalized_entropy"] = norm_entropy
        result["mean_entropy"] = float(entropy.mean())
        result["mean_normalized_entropy"] = float(norm_entropy.mean())

    # 7. Optional TCC cycle-back diagnostic
    # project -> back-match -> measure cycle consistency
    back_dists = (
        (projected_embs ** 2).sum(axis=1, keepdims=True)
        + (nonexpert_embs ** 2).sum(axis=1, keepdims=True).T
        - 2.0 * projected_embs @ nonexpert_embs.T
    )                                                           # [T_q, T_q]
    np.clip(back_dists, 0.0, None, out=back_dists)
    beta = _stable_softmax(-back_dists / temperature, axis=1)  # [T_q, T_q]
    q_idx = np.arange(T_q, dtype=np.float64)
    cycle_mu = beta @ q_idx                                     # [T_q]
    cycle_var = beta @ ((q_idx[None, :] - cycle_mu[:, None]) ** 2)  # [T_q]
    cycle_abs_err = np.abs(cycle_mu - q_idx)                   # [T_q]
    result["mean_cycle_abs_error"] = float(cycle_abs_err.mean())
    result["mean_cycle_var"] = float(cycle_var.mean())

    # 8. Optional alpha matrix
    if save_alpha:
        result["alpha"] = alpha                                 # [T_q, T_e]

    return result


# ---------------------------------------------------------------------------
# Main task class
# ---------------------------------------------------------------------------

class ExpertProjectionTask(BaseTask):
    """Projects non-expert frame embeddings onto an expert trajectory via SNN."""

    def __init__(self):
        super().__init__(task_name="expert_projection", downstream_task=False)
        self.config: dict = {}

    def configure(self, config: dict) -> None:
        """Store the resolved eval config dict produced by ConfigV2.load_eval()."""
        self.config = config

    def _build_shared_tsne_bundle(
        self,
        output_h5_path: Path,
        viz_video_ids: list,
        demo_name_map: dict = None,
    ) -> dict | None:
        """Compute one shared PCA -> t-SNE fit for all requested videos."""
        build_start = time.perf_counter()
        if demo_name_map is None:
            demo_name_map = {}
        try:
            from sklearn.decomposition import PCA          # noqa: PLC0415
            from sklearn.manifold import TSNE              # noqa: PLC0415
            from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
        except ImportError as exc:
            print(f"[expert_projection] tsne: scikit-learn not available ({exc}) – skipping")
            return None

        import matplotlib                                   # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt                    # noqa: PLC0415

        _tsne_cfg_all = self.config.get("tsne_viz", {})
        _plot_cfg = _tsne_cfg_all.get("plot", {})
        _tsne_sub = _tsne_cfg_all.get("tsne", {})

        with h5py.File(output_h5_path, "r") as f:
            expert_embs = np.array(f["expert"]["embeddings"], dtype=np.float32)
            expert_steps = np.array(f["expert"]["target_steps"], dtype=np.float64)
            nonexpert_h5_str = str(f.attrs.get("nonexpert_h5_path", ""))
            ne_grp = f["nonexperts"]
            selected_ids = [vid_id for vid_id in viz_video_ids if vid_id in ne_grp]
            proj_embs_by_vid = {
                vid_id: np.array(ne_grp[vid_id]["projected_embeddings"], dtype=np.float32)
                for vid_id in selected_ids
            }

        if not selected_ids:
            print("[expert_projection] tsne: no videos found for shared fit – skipping")
            return None

        ne_h5 = Path(nonexpert_h5_str)
        if not ne_h5.exists():
            print(f"[expert_projection] tsne: nonexpert H5 not found ({ne_h5}) – skipping")
            return None
        with h5py.File(str(ne_h5), "r") as f:
            ne_embs_by_vid = {
                vid_id: np.array(f["videos"][vid_id]["embeddings"], dtype=np.float32)
                for vid_id in selected_ids
            }

        max_expert = int(_tsne_cfg_all.get("max_frames_expert", 200))
        max_per_vid = int(_tsne_cfg_all.get("max_frames_nonexpert", 80))
        random_seed = int(_tsne_cfg_all.get("random_seed", 42))
        rng = np.random.default_rng(random_seed)
        print(
            "[expert_projection] tsne: shared fit config "
            f"videos={len(selected_ids)} max_frames_expert={max_expert} "
            f"max_frames_nonexpert={max_per_vid} random_seed={random_seed}"
        )

        if len(expert_embs) > max_expert:
            idx_e = np.sort(rng.choice(len(expert_embs), max_expert, replace=False))
        else:
            idx_e = np.arange(len(expert_embs))
        expert_embs_s = expert_embs[idx_e]
        expert_steps_s = expert_steps[idx_e]

        ne_s_list = []
        proj_s_list = []
        ne_prog_by_vid = {}
        sampled_counts = {}
        for vid_id in selected_ids:
            ne_e = ne_embs_by_vid[vid_id]
            proj_e = proj_embs_by_vid[vid_id]
            t_q = len(ne_e)
            idx_q = (
                np.sort(rng.choice(t_q, max_per_vid, replace=False))
                if t_q > max_per_vid else np.arange(t_q)
            )
            ne_s_list.append(ne_e[idx_q])
            proj_s_list.append(proj_e[idx_q])
            ne_prog_by_vid[vid_id] = idx_q.astype(np.float64) / max(t_q - 1, 1)
            sampled_counts[vid_id] = len(idx_q)

        n_expert = len(expert_embs_s)
        sampled_nonexpert = int(sum(sampled_counts.values()))
        orig_counts = np.array([len(ne_embs_by_vid[vid_id]) for vid_id in selected_ids], dtype=np.int64)
        sampled_count_arr = np.array([sampled_counts[vid_id] for vid_id in selected_ids], dtype=np.int64)
        print(
            "[expert_projection] tsne: sampled frames "
            f"expert={n_expert}/{len(expert_embs)} "
            f"nonexpert_total={sampled_nonexpert}/{int(orig_counts.sum())} "
            f"per_video[min/median/max]={sampled_count_arr.min()}/{int(np.median(sampled_count_arr))}/{sampled_count_arr.max()} "
            f"orig_len[min/median/max]={orig_counts.min()}/{int(np.median(orig_counts))}/{orig_counts.max()}"
        )
        preview_pairs = ", ".join(
            f"{vid_id}:{sampled_counts[vid_id]}/{len(ne_embs_by_vid[vid_id])}"
            for vid_id in selected_ids[: min(5, len(selected_ids))]
        )
        if preview_pairs:
            print(f"[expert_projection] tsne: sample preview {preview_pairs}")
        all_embs = np.concatenate([expert_embs_s] + ne_s_list + proj_s_list, axis=0)
        n_total, dim = all_embs.shape

        standardize = bool(_tsne_cfg_all.get("preprocessing", {}).get("standardize", True))
        use_pca = bool(_tsne_cfg_all.get("preprocessing", {}).get("use_pca_before_tsne", True))
        pca_dim_cfg = int(_tsne_cfg_all.get("preprocessing", {}).get("pca_dim", 50))
        print(
            "[expert_projection] tsne: preprocessing "
            f"standardize={standardize} use_pca={use_pca} "
            f"requested_pca_dim={pca_dim_cfg} total_points={n_total} embedding_dim={dim}"
        )

        all_proc = StandardScaler().fit_transform(all_embs) if standardize else all_embs.copy()
        if use_pca:
            pca_dim = min(pca_dim_cfg, dim, n_total - 1)
            pca = PCA(n_components=pca_dim)
            all_pca = pca.fit_transform(all_proc)
            print(
                f"[expert_projection] tsne: shared fit n={n_total}  PCA {dim}→{pca_dim}  "
                f"var={pca.explained_variance_ratio_.sum():.3f}"
            )
        else:
            all_pca = all_proc
            print(f"[expert_projection] tsne: shared fit n={n_total}  (no PCA, D={dim})")

        perplexity_cfg = float(_tsne_sub.get("perplexity", 30))
        learning_rate = _tsne_sub.get("learning_rate", "auto")
        init = _tsne_sub.get("init", "pca")
        max_iter = int(_tsne_sub.get("max_iter", 1000))
        n_components = int(_tsne_sub.get("n_components", 2))
        verbose = int(_tsne_sub.get("verbose", 2))
        method = _tsne_sub.get("method", "barnes_hut")
        angle = float(_tsne_sub.get("angle", 0.5))
        n_jobs = _tsne_sub.get("n_jobs", None)
        perplexity_mode = str(_tsne_sub.get("perplexity_mode", "max_safe")).strip().lower()
        perplexity_cap = max(5.0, (n_total - 1) / 3.0)
        if perplexity_mode == "config":
            perplexity = perplexity_cfg
        elif perplexity_mode in {"config_clamped", "clamped_config", "clamp"}:
            perplexity = min(perplexity_cfg, perplexity_cap)
        else:
            perplexity = perplexity_cap
        print(
            "[expert_projection] tsne: shared fit "
            f"videos={len(selected_ids)} expert={n_expert} "
            f"nonexpert={sampled_nonexpert} projected={sampled_nonexpert}"
        )
        print(
            "[expert_projection] tsne: running shared t-SNE "
            f"(mode={perplexity_mode}, config={perplexity_cfg:.1f}, "
            f"cap={perplexity_cap:.1f}, used={perplexity:.1f}) …"
        )
        print(
            "[expert_projection] tsne: runtime params "
            f"learning_rate={learning_rate} init={init} max_iter={max_iter} "
            f"method={method} angle={angle} n_jobs={n_jobs} verbose={verbose}"
        )
        fit_start = time.perf_counter()
        coords = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            init=init,
            random_state=random_seed,
            max_iter=max_iter,
            verbose=verbose,
            method=method,
            angle=angle,
            n_jobs=n_jobs,
        ).fit_transform(all_pca)
        fit_elapsed = time.perf_counter() - fit_start
        print(
            "[expert_projection] tsne: shared fit finished "
            f"in {fit_elapsed:.1f}s ({fit_elapsed / 60.0:.2f} min)"
        )

        n_ne_per = [sampled_counts[vid_id] for vid_id in selected_ids]
        cuts = np.cumsum([0, n_expert] + n_ne_per + n_ne_per)
        coords_expert = coords[cuts[0]: cuts[1]]
        coords_ne_by_vid = {}
        coords_proj_by_vid = {}
        n_videos = len(selected_ids)
        for idx, vid_id in enumerate(selected_ids):
            coords_ne_by_vid[vid_id] = coords[cuts[1 + idx]: cuts[2 + idx]]
            coords_proj_by_vid[vid_id] = coords[
                cuts[1 + n_videos + idx]: cuts[2 + n_videos + idx]
            ]

        expert_prog = (
            (expert_steps_s - expert_steps_s.min())
            / max(expert_steps_s.max() - expert_steps_s.min(), 1e-6)
        )
        cmap = plt.get_cmap("tab10" if len(selected_ids) <= 10 else "tab20")
        vid_colors_by_vid = {
            vid_id: cmap(idx % cmap.N)
            for idx, vid_id in enumerate(selected_ids)
        }

        default_ne_cmaps = [
            "Reds", "Greens", "Oranges", "Purples",
            "YlOrBr", "PuBu", "RdPu", "YlGn",
        ]
        cmap_ne_names = _plot_cfg.get("cmap_ne_list", default_ne_cmaps)
        ne_cmaps_by_vid = {
            vid_id: plt.get_cmap(cmap_ne_names[idx % len(cmap_ne_names)])
            for idx, vid_id in enumerate(selected_ids)
        }

        x_min = float(coords[:, 0].min())
        x_max = float(coords[:, 0].max())
        y_min = float(coords[:, 1].min())
        y_max = float(coords[:, 1].max())
        pad_x = 0.05 * max(x_max - x_min, 1.0)
        pad_y = 0.05 * max(y_max - y_min, 1.0)
        total_elapsed = time.perf_counter() - build_start
        print(
            "[expert_projection] tsne: shared bundle ready "
            f"in {total_elapsed:.1f}s ({total_elapsed / 60.0:.2f} min) "
            f"xlim=({x_min - pad_x:.3f}, {x_max + pad_x:.3f}) "
            f"ylim=({y_min - pad_y:.3f}, {y_max + pad_y:.3f})"
        )

        return {
            "video_ids": selected_ids,
            "coords_expert": coords_expert,
            "coords_ne_by_vid": coords_ne_by_vid,
            "coords_proj_by_vid": coords_proj_by_vid,
            "expert_prog": expert_prog,
            "ne_prog_by_vid": ne_prog_by_vid,
            "ne_cmaps_by_vid": ne_cmaps_by_vid,
            "vid_colors_by_vid": vid_colors_by_vid,
            "plot_pt_size": float(_plot_cfg.get("point_size", 20)),
            "plot_figsize": tuple(_plot_cfg.get("figsize", [10, 8])),
            "cmap_expert_name": _plot_cfg.get("cmap_progress_group1", "Blues"),
            "axis_xlim": (x_min - pad_x, x_max + pad_x),
            "axis_ylim": (y_min - pad_y, y_max + pad_y),
            "perplexity_used": float(perplexity),
            "n_total": int(n_total),
            "labels_by_vid": {
                vid_id: demo_name_map.get(vid_id, vid_id)
                for vid_id in selected_ids
            },
        }

    def _slice_shared_tsne_bundle(
        self,
        shared_tsne_bundle: dict,
        viz_video_ids: list,
    ) -> dict | None:
        """Return the per-video slice for rendering from a shared t-SNE bundle."""
        video_ids = [
            vid_id for vid_id in viz_video_ids
            if vid_id in shared_tsne_bundle["coords_ne_by_vid"]
        ]
        if not video_ids:
            return None

        return {
            "video_ids": video_ids,
            "coords_expert": shared_tsne_bundle["coords_expert"],
            "coords_ne": [shared_tsne_bundle["coords_ne_by_vid"][vid_id] for vid_id in video_ids],
            "coords_proj": [shared_tsne_bundle["coords_proj_by_vid"][vid_id] for vid_id in video_ids],
            "expert_prog": shared_tsne_bundle["expert_prog"],
            "ne_prog_list": [shared_tsne_bundle["ne_prog_by_vid"][vid_id] for vid_id in video_ids],
            "ne_cmaps": [shared_tsne_bundle["ne_cmaps_by_vid"][vid_id] for vid_id in video_ids],
            "vid_colors": [shared_tsne_bundle["vid_colors_by_vid"][vid_id] for vid_id in video_ids],
            "plot_pt_size": shared_tsne_bundle["plot_pt_size"],
            "plot_figsize": shared_tsne_bundle["plot_figsize"],
            "cmap_expert_name": shared_tsne_bundle["cmap_expert_name"],
            "axis_xlim": shared_tsne_bundle["axis_xlim"],
            "axis_ylim": shared_tsne_bundle["axis_ylim"],
            "perplexity_used": shared_tsne_bundle["perplexity_used"],
            "n_total": shared_tsne_bundle["n_total"],
            "labels": [shared_tsne_bundle["labels_by_vid"][vid_id] for vid_id in video_ids],
        }

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, embeddings_dataset=None) -> dict:  # noqa: ARG002
        cfg = self.config

        # --- paths ---
        expert_h5_path = cfg["expert_h5_path"]
        nonexpert_h5_path = cfg["nonexpert_h5_path"]
        temperature = float(cfg.get("projection_temperature", 0.1))
        save_alpha = bool(cfg.get("save_alpha", False))
        save_entropy = bool(cfg.get("save_entropy", True))
        save_visualization = bool(cfg.get("save_visualization", False))

        print()
        print("[expert_projection] expert_h5_path   :", expert_h5_path)
        print("[expert_projection] nonexpert_h5_path:", nonexpert_h5_path)

        # --- read expert ---
        expert_embs, expert_steps = _read_expert_h5(expert_h5_path)
        T_e, D = expert_embs.shape
        print(f"[expert_projection] expert_embs shape: {expert_embs.shape}")

        # --- read non-expert videos ---
        nonexpert_records = _read_nonexpert_h5(nonexpert_h5_path)
        n_videos = len(nonexpert_records)
        print(f"[expert_projection] number of non-expert videos: {n_videos}")

        # --- build output H5 path ---
        # Structured under outputs/expert_projection/<expert_stem>/<nonexpert_stem>/[<run_name>/]
        # If the expert H5 lives inside a per-run subdirectory of the embeddings root
        # (i.e. .../datasets/embeddings/{run_name}/{file}.h5), an extra run-level
        # subdirectory is automatically appended so results from different runs stay
        # separate.  When the H5 is directly under the embeddings root (legacy / registry
        # based paths), no extra layer is added.  Per-video visualisation artifacts are
        # stored under additional child directories inside this run-level directory.
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        expert_stem = Path(expert_h5_path).stem
        nonexpert_stem = Path(nonexpert_h5_path).stem
        _expert_parent = Path(expert_h5_path).parent
        _run_subdir = (
            _expert_parent.name
            if _expert_parent.parent.name == "embeddings"
            else ""
        )
        output_h5_dir = (
            Path(_PROJ_ROOT) / "outputs" / "expert_projection"
            / expert_stem / nonexpert_stem
        )
        if _run_subdir:
            output_h5_dir = output_h5_dir / _run_subdir
        output_h5_dir.mkdir(parents=True, exist_ok=True)
        output_h5_path = output_h5_dir / f"expert_projection-{timestamp}.h5"
        print(f"[expert_projection] output_h5_path   : {output_h5_path}")

        # --- per-video projection + write H5 ---
        per_video_mean_hard_nn = []

        with h5py.File(output_h5_path, "w") as f_out:

            # root attrs
            f_out.attrs["task_name"] = "expert_projection"
            f_out.attrs["expert_h5_path"] = str(expert_h5_path)
            f_out.attrs["nonexpert_h5_path"] = str(nonexpert_h5_path)
            f_out.attrs["projection_temperature"] = temperature
            f_out.attrs["num_nonexpert_videos"] = n_videos

            # /expert group
            expert_grp = f_out.create_group("expert")
            expert_grp.create_dataset("embeddings", data=expert_embs, compression="gzip")
            expert_grp.create_dataset("target_steps", data=expert_steps)
            expert_grp.attrs["T"] = T_e
            expert_grp.attrs["D"] = D

            # /nonexperts group
            ne_grp = f_out.create_group("nonexperts")

            for video_id, ne_embs, ne_steps in nonexpert_records:
                T_q = ne_embs.shape[0]
                print(f"  [video {video_id}] embeddings shape: {ne_embs.shape}")

                res = _project_one_video(
                    nonexpert_embs=ne_embs,
                    nonexpert_steps=ne_steps,
                    expert_embs=expert_embs,
                    expert_steps=expert_steps,
                    temperature=temperature,
                    save_entropy=save_entropy,
                    save_alpha=save_alpha,
                )

                # print per-video summary
                print(
                    f"  [video {video_id}]"
                    f"  mean_hard_nn_distance={res['mean_hard_nn_distance']:.6f}"
                    f"  mean_soft_expert_step={res['mean_soft_expert_step']:.4f}"
                )
                if save_entropy:
                    print(
                        f"  [video {video_id}]"
                        f"  mean_entropy={res['mean_entropy']:.6f}"
                        f"  mean_normalized_entropy={res['mean_normalized_entropy']:.6f}"
                    )
                print(
                    f"  [video {video_id}]"
                    f"  mean_cycle_abs_error={res['mean_cycle_abs_error']:.6f}"
                    f"  mean_cycle_var={res['mean_cycle_var']:.6f}"
                )
                if save_alpha:
                    _alpha_peak = float(res["alpha"].max(axis=1).mean())
                    print(
                        f"  [video {video_id}]"
                        f"  alpha_mean_peak_weight={_alpha_peak:.6f}"
                    )

                per_video_mean_hard_nn.append(res["mean_hard_nn_distance"])

                # write video group
                vid_grp = ne_grp.create_group(video_id)
                vid_grp.create_dataset("projected_embeddings", data=res["projected_embs"].astype(np.float32), compression="gzip")
                vid_grp.create_dataset("target_steps", data=ne_steps)
                vid_grp.create_dataset("hard_nn_expert_index", data=res["nn_indices"].astype(np.int64))
                vid_grp.create_dataset("hard_nn_expert_step", data=res["nn_expert_steps"])
                vid_grp.create_dataset("hard_nn_distance", data=res["nn_distances"].astype(np.float32))
                vid_grp.create_dataset("soft_expert_index", data=res["soft_expert_index"].astype(np.float32))
                vid_grp.create_dataset("soft_expert_step", data=res["soft_expert_step"].astype(np.float32))

                if save_entropy:
                    vid_grp.create_dataset("entropy", data=res["entropy"].astype(np.float32))
                    vid_grp.create_dataset("normalized_entropy", data=res["normalized_entropy"].astype(np.float32))

                if save_alpha:
                    vid_grp.create_dataset("alpha", data=res["alpha"].astype(np.float32), compression="gzip")

                # video group attrs
                vid_grp.attrs["mean_hard_nn_distance"] = res["mean_hard_nn_distance"]
                vid_grp.attrs["mean_soft_expert_step"] = res["mean_soft_expert_step"]
                if save_entropy:
                    vid_grp.attrs["mean_entropy"] = res["mean_entropy"]
                    vid_grp.attrs["mean_normalized_entropy"] = res["mean_normalized_entropy"]

        # --- global summary ---
        global_mean_hard_nn = float(np.mean(per_video_mean_hard_nn)) if per_video_mean_hard_nn else float("nan")
        print()
        print(f"[expert_projection] global_mean_hard_nn_distance: {global_mean_hard_nn:.6f}")

        # --- optional visualisation ---
        if save_visualization:
            _all_video_ids = [vid_id for vid_id, _, _ in nonexpert_records]
            _video_index_map = {
                vid_id: idx for idx, vid_id in enumerate(_all_video_ids)
            }
            viz_cfg = cfg.get("visualization_video_ids", None)
            if viz_cfg == "all":
                viz_video_ids = _all_video_ids
            elif not viz_cfg:  # None, empty string, or empty list
                k = min(4, len(_all_video_ids))
                viz_video_ids = list(
                    np.random.default_rng(42).choice(_all_video_ids, size=k, replace=False).tolist()
                )
            else:
                viz_video_ids = [
                    _all_video_ids[i]
                    for i in viz_cfg
                    if isinstance(i, int) and 0 <= i < len(_all_video_ids)
                ]
            print(
                f"[expert_projection] visualizing {len(viz_video_ids)} videos: {viz_video_ids}"
            )

            # Build optional video_id -> demo_name mapping from raw HDF5
            _raw_h5 = cfg.get("nonexpert_raw_hdf5_path", "") or ""
            _mask_key = cfg.get("nonexpert_mask_key", "") or ""
            if _raw_h5 and _mask_key:
                demo_name_map = _build_demo_name_map(_raw_h5, _mask_key)
                print(
                    f"[expert_projection] demo_name_map: {len(demo_name_map)} entries loaded "
                    f"(mask='{_mask_key}', raw_hdf5='{_raw_h5}')"
                )
            else:
                demo_name_map = {}

            _video_raw_dir = cfg.get("nonexpert_video_raw_dir", "") or ""
            _save_tsne = bool(cfg.get("save_tsne_visualization", False))
            shared_tsne_bundle = None
            if _save_tsne and viz_video_ids:
                shared_tsne_bundle = self._build_shared_tsne_bundle(
                    output_h5_path=output_h5_path,
                    viz_video_ids=viz_video_ids,
                    demo_name_map=demo_name_map,
                )
                if shared_tsne_bundle is None:
                    print("[expert_projection] tsne: shared fit unavailable – skipping t-SNE visualisations")
                    _save_tsne = False
            for video_id in viz_video_ids:
                video_idx = _video_index_map.get(video_id)
                if video_idx is None:
                    print(
                        f"[expert_projection] skipping visualisation for unknown video_id={video_id}"
                    )
                    continue
                video_output_dir = output_h5_dir / f"{video_idx:03d}_{timestamp}"
                print(
                    "[expert_projection] saving per-video outputs for "
                    f"video_id={video_id} idx={video_idx} -> {video_output_dir}"
                )
                self._save_visualizations(
                    output_h5_path=output_h5_path,
                    vis_output_dir=str(video_output_dir),
                    viz_video_ids=[video_id],
                    demo_name_map=demo_name_map,
                    video_raw_dir=_video_raw_dir or None,
                    save_tsne=_save_tsne,
                    shared_tsne_bundle=shared_tsne_bundle,
                )

        return {
            "task_name":    "expert_projection",
            "metric_name":  "global_mean_hard_nn_distance",
            "metric_value": global_mean_hard_nn,
            "output_h5_path": str(output_h5_path),
        }

    # ------------------------------------------------------------------
    # Visualisation helper (kept separate to keep evaluate() readable)
    # ------------------------------------------------------------------

    def _save_visualizations(
        self,
        output_h5_path: Path,
        vis_output_dir: str,
        viz_video_ids: list,
        demo_name_map: dict = None,
        video_raw_dir: str = None,
        save_tsne: bool = False,
        shared_tsne_bundle: dict | None = None,
    ) -> None:
        """Save alignment-curve visualisations for the selected non-expert videos.

        Always writes a static soft-NN alignment PNG.  When *video_raw_dir* is
        provided, also writes an animated MP4 (GIF fallback) showing the
        soft-NN plot with a sweeping time cursor alongside synchronised demo
        video frames read directly from <video_raw_dir>/<demo_name>.mp4.
        All videos are normalised to the same length on the time axis.
        """
        if demo_name_map is None:
            demo_name_map = {}
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h5_stem = Path(output_h5_path).stem
        save_dir = Path(vis_output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── read soft-NN curves ──────────────────────────────────────────────
        soft_curves = []
        with h5py.File(output_h5_path, "r") as f:
            ne_grp = f["nonexperts"]
            video_ids = [vid_id for vid_id in viz_video_ids if vid_id in ne_grp]
            for video_id in video_ids:
                soft_curves.append(
                    np.array(ne_grp[video_id]["soft_expert_step"], dtype=np.float64)
                )

        if not soft_curves:
            print("[expert_projection] no videos found for visualisation – skipping")
            return

        N_PTS = 300
        xs = np.linspace(0.0, 1.0, N_PTS)

        def _interp(curve: np.ndarray) -> np.ndarray:
            return np.interp(xs, np.linspace(0.0, 1.0, len(curve)), curve)

        soft_mat = np.stack([_interp(c) for c in soft_curves])   # [N, N_PTS]
        cmap = plt.get_cmap("tab10" if len(video_ids) <= 10 else "tab20")
        colors = [cmap(i % cmap.N) for i in range(len(video_ids))]
        labels = [demo_name_map.get(vid_id, vid_id) for vid_id in video_ids]

        # ── static PNG (soft-NN only) ────────────────────────────────────────
        fig_s, ax_s = plt.subplots(1, 1, figsize=(8, 5))
        fig_s.suptitle(
            f"Expert projection alignment — {h5_stem}\n({len(video_ids)} non-expert videos)",
            fontsize=11,
        )
        for row, label, color in zip(soft_mat, labels, colors):
            ax_s.plot(xs, row, color=color, linewidth=1.2, label=label)
        ax_s.set_xlabel("non-expert frame (normalised)")
        ax_s.set_ylabel("expert step")
        ax_s.set_title("Soft expected expert step")
        ax_s.legend(fontsize=7, ncol=max(1, len(video_ids) // 10),
                    loc="upper left", framealpha=0.6)
        fig_s.tight_layout()
        png_path = save_dir / f"alignment_curve_{h5_stem}.png"
        fig_s.savefig(png_path, dpi=120)
        plt.close(fig_s)
        print(f"[expert_projection] saved alignment curve: {png_path}")

        # ── animated MP4 (soft-NN + video frames) ───────────────────────────
        if not video_raw_dir:
            # No animation: produce static t-SNE (if requested) then return early.
            if save_tsne:
                self._save_tsne_visualization(
                    output_h5_path=output_h5_path,
                    vis_output_dir=vis_output_dir,
                    viz_video_ids=video_ids,
                    demo_name_map=demo_name_map,
                    shared_tsne_bundle=shared_tsne_bundle,
                )
            return

        import cv2  # noqa: PLC0415

        def _read_mp4_frames(mp4_path: str):
            """Return uint8 RGB frames array [T, H, W, 3] or None on failure."""
            cap = cv2.VideoCapture(mp4_path)
            if not cap.isOpened():
                return None
            frames = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap.release()
            return np.stack(frames) if frames else None

        video_frames: dict = {}   # video_id -> uint8 [T, H, W, 3]
        raw_dir = Path(video_raw_dir)
        for vid_id in video_ids:
            demo_name = demo_name_map.get(vid_id, vid_id)   # e.g. "demo_142"
            mp4_path = raw_dir / f"{demo_name}.mp4"
            if not mp4_path.exists():
                print(f"[expert_projection] MP4 not found: {mp4_path} – skipping")
                continue
            arr = _read_mp4_frames(str(mp4_path))
            if arr is None or len(arr) == 0:
                print(f"[expert_projection] failed to read frames from {mp4_path} – skipping")
                continue
            video_frames[vid_id] = arr
            print(f"[expert_projection] loaded {len(arr)} frames from {mp4_path.name}")

        anim_ids = [v for v in video_ids if v in video_frames]
        if not anim_ids:
            print("[expert_projection] no frames loaded – skipping animation")
            return

        # ── t-SNE: compute ONCE for anim_ids; reuse for static PNG + animation ──
        # This guarantees both plots share identical coordinates (same data, same
        # seed, single run) and avoids the previous bug where the static plot used
        # `video_ids` while the animation used the narrower `anim_ids`.
        _tsne_precomputed = None
        if save_tsne:
            _tsne_precomputed = self._save_tsne_visualization(
                output_h5_path=output_h5_path,
                vis_output_dir=vis_output_dir,
                viz_video_ids=anim_ids,
                demo_name_map=demo_name_map,
                shared_tsne_bundle=shared_tsne_bundle,
            )

        # ── t-SNE animation figure (built from precomputed coords; no second run) ──
        _tsne_fig_anim = None
        _tsne_sc_expert = None
        _tsne_sc_ne: list = []
        _tsne_sc_proj: list = []
        _tsne_expert_prog_a = None
        _tsne_ne_prog_a: list = []
        _tsne_vcols_a: list = []
        _tsne_ne_cmaps_a: list = []
        _tsne_sigma   = 0.01
        _tsne_bg_al   = 0.08
        _tsne_hl_al   = 0.95
        _tsne_cmap_e_fn = None

        if save_tsne and _tsne_precomputed is not None:
            try:
                _tc_all = self.config.get("tsne_viz", {})
                _pc2 = _tc_all.get("plot", {})
                _vc  = _tc_all.get("tsne_video", {})

                _tsne_sigma   = float(_vc.get("progress_sigma",      0.01))
                _tsne_bg_al   = float(_vc.get("background_alpha",    0.08))
                _tsne_hl_al   = float(_vc.get("highlight_alpha_max", 0.95))
                _tsne_cmap_e_name = _pc2.get("cmap_progress_group1", "Blues")
                _tsne_cmap_e_fn   = plt.get_cmap(_tsne_cmap_e_name)

                # Unpack precomputed t-SNE coords (identical to those in static PNG)
                _tids      = _tsne_precomputed["video_ids"]
                _ce_a      = _tsne_precomputed["coords_expert"]
                _cne_a     = _tsne_precomputed["coords_ne"]
                _cpr_a     = _tsne_precomputed["coords_proj"]
                _ep_a      = _tsne_precomputed["expert_prog"]
                _ne_prog_l  = _tsne_precomputed["ne_prog_list"]
                _ne_cmaps_a = _tsne_precomputed["ne_cmaps"]
                _vcols_a    = _tsne_precomputed["vid_colors"]
                _pt_a       = _tsne_precomputed["plot_pt_size"]
                _fs_a       = _tsne_precomputed["plot_figsize"]
                _ax_xlim    = _tsne_precomputed["axis_xlim"]
                _ax_ylim    = _tsne_precomputed["axis_ylim"]

                _tsne_fig_anim, _tax = plt.subplots(figsize=_fs_a)

                # Expert scatter – solid mid-tone color for animation (gradient only in static PNG)
                _base_e_rgb = np.array(_tsne_cmap_e_fn(0.55)[:3], dtype=np.float32)
                _init_e = np.tile(
                    np.array([*_base_e_rgb, _tsne_bg_al], dtype=np.float32), (len(_ep_a), 1)
                )
                _tsne_sc_expert = _tax.scatter(
                    _ce_a[:, 0], _ce_a[:, 1],
                    c=_init_e, s=_pt_a, zorder=2,
                )

                # NE + proj scatters + thin connecting lines (gradient colormaps)
                for _i2, (_cn, _cp, _ne_cm2) in enumerate(zip(_cne_a, _cpr_a, _ne_cmaps_a)):
                    _ne_prog2 = _ne_prog_l[_i2]
                    _mid_col2 = _ne_cm2(0.5)
                    for _j2 in range(len(_cn)):
                        _tax.plot(
                            [_cn[_j2, 0], _cp[_j2, 0]],
                            [_cn[_j2, 1], _cp[_j2, 1]],
                            color=_mid_col2, linewidth=0.5, alpha=0.15, zorder=1,
                        )
                    # NE/proj: solid tab10 color per video for animation (gradient only in static PNG)
                    _vcol2 = np.array(_vcols_a[_i2], dtype=np.float32)  # [4] RGBA
                    _init_ne = np.tile(_vcol2, (len(_cn), 1)).copy()
                    _init_ne[:, 3] = _tsne_bg_al
                    _sne2 = _tax.scatter(
                        _cn[:, 0], _cn[:, 1],
                        c=_init_ne, s=_pt_a * 1.4, marker="o", zorder=3,
                        label=demo_name_map.get(_tids[_i2], _tids[_i2]),
                    )
                    _spr2 = _tax.scatter(
                        _cp[:, 0], _cp[:, 1],
                        c=_init_ne.copy(), s=_pt_a * 1.4, marker="^", zorder=3,
                    )
                    _tsne_sc_ne.append(_sne2)
                    _tsne_sc_proj.append(_spr2)

                _tax.set_xlabel("t-SNE dim 1", fontsize=8)
                _tax.set_ylabel("t-SNE dim 2", fontsize=8)
                _tax.set_title("t-SNE  Expert(\u25cf) / NE(\u25cb) / Proj(\u25b3)", fontsize=9)
                _tax.set_xlim(*_ax_xlim)
                _tax.set_ylim(*_ax_ylim)
                _tax.legend(
                    fontsize=6, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.6,
                )
                _tsne_fig_anim.tight_layout()

                _tsne_expert_prog_a = _ep_a
                _tsne_ne_prog_a     = _ne_prog_l
                _tsne_vcols_a       = _vcols_a
                _tsne_ne_cmaps_a    = _ne_cmaps_a
                print("[expert_projection] tsne animation figure ready (coords reused from static PNG)")

            except Exception as _texc:
                import traceback as _tb; _tb.print_exc()  # noqa: E702
                print(f"[expert_projection] tsne anim setup failed ({_texc}) \u2013 skipping t-SNE in animation")
                if _tsne_fig_anim is not None:
                    plt.close(_tsne_fig_anim)
                    _tsne_fig_anim = None

        n_vids = len(anim_ids)
        n_cols_v = min(n_vids, 2)
        n_rows_v = (n_vids + n_cols_v - 1) // n_cols_v

        fig_w = 7.0 + 3.5 * n_cols_v
        fig_h = max(5.0, 3.5 * n_rows_v)
        fig_a = plt.figure(figsize=(fig_w, fig_h))
        gs = fig_a.add_gridspec(
            n_rows_v, 1 + n_cols_v,
            width_ratios=[2.0] + [1.0] * n_cols_v,
            hspace=0.35, wspace=0.15,
        )

        # Left panel: soft-NN alignment plot (spans all rows)
        ax_soft = fig_a.add_subplot(gs[:, 0])
        for row, label, color in zip(soft_mat, labels, colors):
            ax_soft.plot(xs, row, color=color, linewidth=1.2, label=label)
        ax_soft.set_xlabel("non-expert frame (normalised)")
        ax_soft.set_ylabel("expert step")
        ax_soft.set_title("Soft expected expert step")
        ax_soft.legend(fontsize=7, ncol=1, loc="upper left", framealpha=0.6)
        # Freeze y-limits before adding the animated vline
        y0, y1 = ax_soft.get_ylim()
        ax_soft.set_ylim(y0, y1)
        (vline,) = ax_soft.plot(
            [0.0, 0.0], [y0, y1], color="k", linestyle="--", linewidth=1.5, zorder=10
        )

        # Right panels: one image axis per video
        im_objs: list = []
        for i, vid_id in enumerate(anim_ids):
            r, c = divmod(i, n_cols_v)
            ax = fig_a.add_subplot(gs[r, 1 + c])
            ax.axis("off")
            ax.set_title(
                demo_name_map.get(vid_id, vid_id),
                fontsize=8,
                color=colors[video_ids.index(vid_id)],
            )
            im = ax.imshow(video_frames[vid_id][0])
            im_objs.append(im)

        fig_a.suptitle(
            f"Expert projection — {h5_stem}  ({n_vids} demos)", fontsize=11
        )
        fig_a.subplots_adjust(top=0.88, hspace=0.35, wspace=0.15)

        # ── render frames with imageio (libx264) ───────────────────────────
        try:
            import imageio  # noqa: PLC0415
        except ImportError as exc:
            print(
                f"[expert_projection] imageio not available ({exc}) – skipping animation. "
                "Install with: pip install imageio[ffmpeg]"
            )
            plt.close(fig_a)
            return

        mp4_path = save_dir / f"alignment_anim_{h5_stem}.mp4"
        try:
            writer = imageio.get_writer(
                str(mp4_path),
                fps=20,
                format="ffmpeg",
                codec="libx264",
                macro_block_size=1,
            )
        except Exception as exc:
            print(f"[expert_projection] could not open video writer ({exc}) – skipping animation")
            plt.close(fig_a)
            return

        print(f"[expert_projection] rendering {N_PTS} frames …")
        if _tsne_fig_anim is not None:
            print("[expert_projection] t-SNE panel will be stitched on the right")
        log_every = max(1, N_PTS // 10)
        try:
            for fi in range(N_PTS):
                t = fi / (N_PTS - 1)

                # ── update alignment panel ─────────────────────────────────
                vline.set_xdata([t, t])
                for im, vid_id in zip(im_objs, anim_ids):
                    frames = video_frames[vid_id]
                    fj = min(int(t * len(frames)), len(frames) - 1)
                    im.set_data(frames[fj])
                fig_a.canvas.draw()
                align_frame = np.asarray(fig_a.canvas.buffer_rgba()).copy()[:, :, :3]

                # ── update t-SNE panel (Gaussian highlight) ────────────────
                if _tsne_fig_anim is not None:
                    # Expert: darken solid color + enlarge marker ∝ Gaussian at current progress
                    _w_e = np.exp(-0.5 * ((_tsne_expert_prog_a - t) / _tsne_sigma) ** 2)
                    _al_e = _tsne_bg_al + (_tsne_hl_al - _tsne_bg_al) * _w_e
                    _base_e_c = np.array(_tsne_cmap_e_fn(0.55)[:3], dtype=np.float32)
                    _dark_e_c = np.array(_tsne_cmap_e_fn(0.92)[:3], dtype=np.float32)
                    _e_rgb = (1.0 - _w_e[:, None]) * _base_e_c + _w_e[:, None] * _dark_e_c
                    _e_rgba = np.concatenate([_e_rgb, _al_e[:, None]], axis=1).astype(np.float32)
                    _tsne_sc_expert.set_facecolor(_e_rgba)
                    _tsne_sc_expert.set_sizes(
                        np.full(len(_tsne_expert_prog_a), float(_pt_a)) * (1.0 + 4.0 * _w_e)
                    )

                    # NE + projected: darken solid color + enlarge marker ∝ Gaussian
                    for _sne3, _spr3, _vcol3, _np3 in zip(
                        _tsne_sc_ne, _tsne_sc_proj, _tsne_vcols_a, _tsne_ne_prog_a
                    ):
                        _w_v = np.exp(-0.5 * ((_np3 - t) / _tsne_sigma) ** 2)
                        _al_v = _tsne_bg_al + (_tsne_hl_al - _tsne_bg_al) * _w_v
                        _base_v = np.array(_vcol3[:3], dtype=np.float32)
                        _dark_v = _base_v * 0.45  # darken to ~45% brightness
                        _nv_rgb = (1.0 - _w_v[:, None]) * _base_v + _w_v[:, None] * _dark_v
                        _nv_rgba = np.concatenate([_nv_rgb, _al_v[:, None]], axis=1).astype(np.float32)
                        _sne3.set_facecolor(_nv_rgba)
                        _spr3.set_facecolor(_nv_rgba)
                        _base_sv = float(_pt_a) * 1.4
                        _sne3.set_sizes(np.full(len(_np3), _base_sv) * (1.0 + 4.0 * _w_v))
                        _spr3.set_sizes(np.full(len(_np3), _base_sv) * (1.0 + 4.0 * _w_v))

                    _tsne_fig_anim.canvas.draw()
                    tsne_frame = np.asarray(_tsne_fig_anim.canvas.buffer_rgba()).copy()[:, :, :3]

                    # Resize t-SNE frame to match alignment frame height
                    _ha, _wa = align_frame.shape[:2]
                    _ht, _wt = tsne_frame.shape[:2]
                    if _ha != _ht:
                        tsne_frame = cv2.resize(tsne_frame, (int(_wt * _ha / _ht), _ha))

                    frame = np.concatenate([align_frame, tsne_frame], axis=1)
                else:
                    frame = align_frame

                # libx264 requires even pixel dimensions
                h_px, w_px = frame.shape[:2]
                frame = frame[: h_px - h_px % 2, : w_px - w_px % 2]
                writer.append_data(frame)
                if fi % log_every == 0:
                    print(f"[expert_projection]   [{fi + 1:3d}/{N_PTS}]  t={t:.3f}")
        finally:
            writer.close()

        print(f"[expert_projection] saved alignment animation: {mp4_path}")
        plt.close(fig_a)
        if _tsne_fig_anim is not None:
            plt.close(_tsne_fig_anim)

    # ------------------------------------------------------------------
    # t-SNE: expert + non-expert + SNN-projected embeddings
    # ------------------------------------------------------------------

    def _save_tsne_visualization(
        self,
        output_h5_path,
        vis_output_dir: str,
        viz_video_ids: list,
        demo_name_map: dict = None,
        shared_tsne_bundle: dict | None = None,
    ) -> dict | None:
        """Save a t-SNE scatter rendered from a shared dataset-level fit."""
        if demo_name_map is None:
            demo_name_map = {}

        import matplotlib                                   # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt                    # noqa: PLC0415
        from matplotlib.collections import LineCollection  # noqa: PLC0415
        from matplotlib.lines import Line2D                # noqa: PLC0415

        h5_stem  = Path(output_h5_path).stem
        save_dir = Path(vis_output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if shared_tsne_bundle is None:
            shared_tsne_bundle = self._build_shared_tsne_bundle(
                output_h5_path=output_h5_path,
                viz_video_ids=viz_video_ids,
                demo_name_map=demo_name_map,
            )
        if shared_tsne_bundle is None:
            return None

        tsne_data = self._slice_shared_tsne_bundle(shared_tsne_bundle, viz_video_ids)
        if tsne_data is None:
            print("[expert_projection] tsne: no viz videos in shared fit – skipping")
            return None

        anim_ids = tsne_data["video_ids"]
        coords_expert = tsne_data["coords_expert"]
        coords_ne = tsne_data["coords_ne"]
        coords_proj = tsne_data["coords_proj"]
        expert_prog = tsne_data["expert_prog"]
        ne_prog_list = tsne_data["ne_prog_list"]
        ne_cmaps = tsne_data["ne_cmaps"]
        vid_colors = tsne_data["vid_colors"]
        labels = tsne_data["labels"]

        _plot_cfg = self.config.get("tsne_viz", {}).get("plot", {})
        _figsize = tsne_data["plot_figsize"]
        _pt_size = tsne_data["plot_pt_size"]
        _alpha = float(_plot_cfg.get("alpha", 0.75))
        _cmap_expert = tsne_data["cmap_expert_name"]
        _dpi = int(_plot_cfg.get("dpi", 150))
        axis_xlim = tsne_data["axis_xlim"]
        axis_ylim = tsne_data["axis_ylim"]

        print(
            "[expert_projection] tsne: rendering from shared fit "
            f"(videos={len(anim_ids)}, n={tsne_data['n_total']}, "
            f"perplexity={tsne_data['perplexity_used']:.1f})"
        )

        # ── plot ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=_figsize)

        # Shared normaliser for all gradient paths (progress ∈ [0, 1])
        _norm_prog = plt.Normalize(vmin=0.0, vmax=1.0)

        # Expert trajectory: gradient-coloured line (sorted by temporal progress)
        _sort_e = np.argsort(expert_prog)
        _ce_sorted = coords_expert[_sort_e]          # [N, 2]
        _prog_sorted = expert_prog[_sort_e]          # [N]
        _pts = _ce_sorted[:, np.newaxis, :]          # [N, 1, 2]
        _segs = np.concatenate([_pts[:-1], _pts[1:]], axis=1)  # [N-1, 2, 2]
        _seg_prog = (_prog_sorted[:-1] + _prog_sorted[1:]) / 2.0
        lc_e = LineCollection(
            _segs, cmap=_cmap_expert, norm=_norm_prog,
            linewidth=2.0, alpha=_alpha, zorder=2,
        )
        lc_e.set_array(_seg_prog)
        ax.add_collection(lc_e)
        plt.colorbar(lc_e, ax=ax, fraction=0.025, pad=0.01, label="Expert progress")

        # Per-video: NE latent path (gradient LineCollection) + scatter + projected
        leg_proxies = []
        leg_proxy_labels = []
        for i, (ne_c, proj_c, ne_cm, label) in enumerate(
            zip(coords_ne, coords_proj, ne_cmaps, labels)
        ):
            ne_prog = ne_prog_list[i]  # [N_q] temporal progress ∈ [0, 1]

            # Thin lines connecting each NE frame to its SNN-projected position
            _mid_col = ne_cm(0.5)
            for j in range(len(ne_c)):
                ax.plot(
                    [ne_c[j, 0], proj_c[j, 0]],
                    [ne_c[j, 1], proj_c[j, 1]],
                    color=_mid_col, linewidth=0.6, alpha=0.25, zorder=1,
                )

            # Non-expert latent path: adjacent frames connected as gradient LineCollection
            if len(ne_c) >= 2:
                _pts_ne = ne_c[:, np.newaxis, :]
                _segs_ne = np.concatenate([_pts_ne[:-1], _pts_ne[1:]], axis=1)
                _seg_prog_ne = (ne_prog[:-1] + ne_prog[1:]) / 2.0
                lc_ne = LineCollection(
                    _segs_ne, cmap=ne_cm, norm=_norm_prog,
                    linewidth=1.5, alpha=_alpha, zorder=3,
                )
                lc_ne.set_array(_seg_prog_ne)
                ax.add_collection(lc_ne)

            # NE frames: gradient-coloured circles (alpha baked into RGBA)
            _ne_rgba = ne_cm(ne_prog).copy()
            _ne_rgba[:, 3] = min(1.0, _alpha + 0.1)
            ax.scatter(
                ne_c[:, 0], ne_c[:, 1],
                c=_ne_rgba, s=_pt_size * 1.4, marker="o", zorder=4,
            )
            # Projected frames: same gradient colormap, triangle markers
            _proj_rgba = ne_cm(ne_prog).copy()
            _proj_rgba[:, 3] = max(0.0, _alpha - 0.2)
            ax.scatter(
                proj_c[:, 0], proj_c[:, 1],
                c=_proj_rgba, s=_pt_size * 1.4, marker="^", zorder=3,
            )

            # Explicit legend proxies (mid-progress colour of each NE cmap)
            _leg_col = ne_cm(0.6)
            leg_proxies.append(
                Line2D([0], [0], color=_leg_col, linewidth=2.0,
                       marker="o", markersize=4, label=label)
            )
            leg_proxy_labels.append(label)
            leg_proxies.append(
                Line2D([0], [0], color=_leg_col, linewidth=0,
                       marker="^", markersize=5, label=f"{label} (proj)")
            )
            leg_proxy_labels.append(f"{label} (proj)")

        ax.set_xlim(*axis_xlim)
        ax.set_ylim(*axis_ylim)

        # Legend: expert proxy + per-video NE/proj proxies
        expert_proxy = Line2D(
            [0], [0], color=plt.get_cmap(_cmap_expert)(0.65), linewidth=2.0, label="Expert",
        )
        ax.legend(
            [expert_proxy] + leg_proxies,
            ["Expert"] + leg_proxy_labels,
            fontsize=7, loc="upper left",
            bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.7,
        )

        ax.set_title(
            f"t-SNE: Expert / Non-expert (○) / Projected (▲)\n{h5_stem}",
            fontsize=10,
        )
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        fig.tight_layout()

        png_path = save_dir / f"tsne_projection_{h5_stem}.png"
        fig.savefig(png_path, dpi=_dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[expert_projection] saved t-SNE projection: {png_path}")
        return {
            "video_ids":        anim_ids,
            "coords_expert":    coords_expert,
            "coords_ne":        coords_ne,
            "coords_proj":      coords_proj,
            "expert_prog":      expert_prog,
            "ne_prog_list":     ne_prog_list,
            "ne_cmaps":         ne_cmaps,
            "vid_colors":       vid_colors,
            "plot_pt_size":     _pt_size,
            "plot_figsize":     _figsize,
            "cmap_expert_name": _cmap_expert,
            "axis_xlim":        axis_xlim,
            "axis_ylim":        axis_ylim,
            "perplexity_used":  tsne_data["perplexity_used"],
            "n_total":          tsne_data["n_total"],
        }
