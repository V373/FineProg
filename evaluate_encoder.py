"""Evaluation entry point for the PyTorch TCC project.

Pipeline:
    1. Load configs_v2/eval/<task>.yaml via ConfigV2
    2. Resolve input refs to embedding / dataset paths
    3. Load embeddings when the task consumes a single H5 input
    4. Build the requested evaluation task via build_task()
    5. Call task.evaluate(...)
    6. Print results

Usage:
    python evaluate.py --task kendalls_tau
    python evaluate.py --task expert_projection
"""


import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Ensure the projects root is on sys.path so that `fineprog` is importable
# both when the script is run directly and when imported as a module.
_PROJ_ROOT = Path(__file__).parent
_PROJECTS_ROOT = _PROJ_ROOT.parent  # .../projects/
if str(_PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECTS_ROOT))

from fineprog.algos.eval_task.base_task import build_task  # noqa: E402
# [v2] V2 config resolver (independent of old config system)
sys.path.insert(0, str(_PROJ_ROOT))
from utils.config_v2 import ConfigV2  # noqa: E402


# ---------------------------------------------------------------------------
# Embeddings H5 reader
# ---------------------------------------------------------------------------

def load_embeddings_h5(path: str) -> dict:
    """Read an embeddings H5 file produced by extract_embeddings.py.

    Expected H5 structure::

        /videos/<video_id>/
            embeddings         [T_out, D]  float32
            target_steps       [T_out]     int64
            phase_labels       [T_out]     int64   (optional)
            keyframe_labels    [T_out]     int64   (optional)
            attrs:
                seq_len        int
                action_id      int
                labeled        bool  (optional, defaults to False)

    Returns:
        dict with keys:
            "video_id"        – list[str]
            "embeddings"      – list[np.ndarray], each shape [T_i, D]
            "target_steps"    – list[np.ndarray], each shape [T_i]
            "seq_len"         – list[int]
            "action_id"       – list[int]
            "phase_labels"    – list[np.ndarray or None]
            "keyframe_labels" – list[np.ndarray or None]
            "labeled"         – list[bool]
    """
    dataset: dict = {
        "video_id":        [],
        "embeddings":      [],
        "target_steps":    [],
        "seq_len":         [],
        "action_id":       [],
        "phase_labels":    [],
        "keyframe_labels": [],
        "labeled":         [],
    }
    with h5py.File(path, "r") as f:
        videos_grp = f["videos"]
        for video_id in sorted(videos_grp.keys()):
            grp = videos_grp[video_id]
            dataset["video_id"].append(video_id)
            dataset["embeddings"].append(np.array(grp["embeddings"]))
            dataset["target_steps"].append(np.array(grp["target_steps"]))
            dataset["seq_len"].append(int(grp.attrs["seq_len"]))
            dataset["action_id"].append(int(grp.attrs["action_id"]))
            dataset["phase_labels"].append(
                np.array(grp["phase_labels"]) if "phase_labels" in grp else None
            )
            dataset["keyframe_labels"].append(
                np.array(grp["keyframe_labels"]) if "keyframe_labels" in grp else None
            )
            dataset["labeled"].append(bool(grp.attrs.get("labeled", False)))
    return dataset


# ---------------------------------------------------------------------------
# Kendall's Tau evaluation
# ---------------------------------------------------------------------------

def evaluate_kendall(embeddings_dataset: dict, eval_cfg: dict, output_dir: str = None) -> dict:
    """Build a KendallsTauTask from config and evaluate on the dataset.

    Args:
        embeddings_dataset: dict produced by load_embeddings_h5().
        eval_cfg:           Resolved Kendalls Tau config dict.
        output_dir:         Directory to save the KT heatmap PNG.  When None
                            the task falls back to its default path.

    Returns:
        Result dict with keys: "task_name", "metric_name", "metric_value".
    """
    task = build_task("kendalls_tau")
    # build_task passes config_path but KendallsTauTask does not parse it
    # internally yet; apply the config values directly.
    task.stride   = eval_cfg.get("kendall_stride",   1)
    task.distance = eval_cfg.get("kendall_distance", "sqeuclidean")
    if output_dir is not None:
        task.output_dir = output_dir
    return task.evaluate(embeddings_dataset)


# ---------------------------------------------------------------------------
# Programmatic entry point (used by in_training_eval and other callers)
# ---------------------------------------------------------------------------

def run_eval_task(task_name: str, resolved: dict) -> dict:
    """Execute a single eval task from a fully resolved config dict.

    Intended for programmatic calls (e.g. from ``utils/in_training_eval.py``).
    The caller is responsible for building ``resolved`` via
    ``ConfigV2().load_eval(task_name, overrides=...)``.

    Args:
        task_name: Task identifier — ``"latent_distance_heatmap"`` or ``"kendalls_tau"``.
        resolved:  Fully resolved config dict (all *_ref keys converted to paths).

    Returns:
        Result dict with at minimum: ``task_name``, ``metric_name``, ``metric_value``,
        plus task-specific keys (e.g. ``output_heatmap_path``).
    """
    if task_name == "latent_distance_heatmap":
        task = build_task("latent_distance_heatmap")
        task.configure(resolved)
        return task.evaluate(None)

    if task_name == "kendalls_tau":
        embedding_h5_path = resolved.get("embedding_h5_path")
        if not embedding_h5_path:
            raise ValueError(
                "[run_eval_task] 'embedding_h5_path' is required in resolved config "
                "for task 'kendalls_tau'."
            )
        embeddings_dataset = load_embeddings_h5(embedding_h5_path)
        task = build_task("kendalls_tau")
        task.configure(resolved)
        return task.evaluate(embeddings_dataset)

    raise NotImplementedError(
        f"[run_eval_task] Task '{task_name}' is not supported for programmatic calls. "
        "Supported: latent_distance_heatmap, kendalls_tau."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(task_name: str | None = None) -> None:
    """Full evaluation pipeline.  V2 mode uses ConfigV2 for path resolution."""

    # -----------------------------------------------------------------------
    # [v2] Config resolution via V2 resolver
    # -----------------------------------------------------------------------
    # task_name is supplied by __main__ from --task CLI arg; default is "kendalls_tau".
    if task_name is None:
        task_name = "kendalls_tau"

    print(f"[evaluate] [v2] task_name: {task_name}")

    # [v2] Resolve all paths for the requested task via ConfigV2
    _cfg_v2 = ConfigV2()
    resolved = _cfg_v2.load_eval(task_name)   # [v2] resolves embedding refs → absolute paths
    print(f"[evaluate] [v2] resolved eval config for '{task_name}':")
    _cfg_v2.print_config(resolved, f"eval/{task_name}")

    # -----------------------------------------------------------------------
    # Load embeddings (for tasks that need a single embedding H5)
    # -----------------------------------------------------------------------
    embeddings_dataset = None
    if task_name not in ("classification", "expert_projection", "latent_distance_heatmap"):
        embedding_save_path = resolved.get("embedding_h5_path")
        if not embedding_save_path:
            raise ValueError(
                f"[evaluate] [v2] 'embedding_h5_path' not resolved for task '{task_name}'."
            )
        print(f"[evaluate] embeddings: {embedding_save_path}")
        embeddings_dataset = load_embeddings_h5(embedding_save_path)
        n_videos = len(embeddings_dataset["video_id"])
        print(f"[evaluate] number of videos     : {n_videos}")
        if n_videos > 0:
            print(f"[evaluate] first embedding shape: {embeddings_dataset['embeddings'][0].shape}")

    # --- evaluate ---
    print(f"[evaluate] running task: {task_name}")

    if task_name == "expert_projection":
        print(f"[evaluate] expert_h5_path   : {resolved['expert_h5_path']}")
        print(f"[evaluate] nonexpert_h5_path: {resolved['nonexpert_h5_path']}")

        task = build_task("expert_projection")
        task.configure(resolved)  # [v2] resolved dict has same keys task.configure() expects
        result = task.evaluate(None)

    elif task_name == "latent_distance_heatmap":
        # [v2] resolved has embedding_h5_path, output_dir, selected_video_index, viz options
        print(f"[evaluate] embedding_h5_path    : {resolved['embedding_h5_path']}")
        print(f"[evaluate] selected_video_index : {resolved['selected_video_index']}")
        print(f"[evaluate] output_dir           : {resolved['output_dir']}")

        task = build_task("latent_distance_heatmap")
        task.configure(resolved)
        result = task.evaluate(None)

    elif task_name == "kendalls_tau":
        # [v1] result = evaluate_kendall(embeddings_dataset, eval_cfg, output_dir=kt_output_dir)
        task = build_task("kendalls_tau")  # [v2]
        task.configure(resolved)
        result = task.evaluate(embeddings_dataset)

    elif task_name == "classification":
        train_h5_path = resolved["classification_train_h5_path"]
        val_h5_path   = resolved["classification_val_h5_path"]

        print(f"[evaluate] classification train H5: {train_h5_path}")
        print(f"[evaluate] classification val   H5: {val_h5_path}")

        train_dataset = load_embeddings_h5(train_h5_path)
        val_dataset   = load_embeddings_h5(val_h5_path)

        print(f"[evaluate] train H5 videos: {len(train_dataset['video_id'])}")
        print(f"[evaluate] val   H5 videos: {len(val_dataset['video_id'])}")

        svm_c    = float(resolved.get("svm_c",    1.0))
        max_iter = int(resolved.get("max_iter", 10000))
        gen_tsne = bool(resolved.get("gen_tsne_phase_label", False))
        task = build_task("classification", svm_c=svm_c, max_iter=max_iter,
                          gen_tsne_phase_label=gen_tsne)
        result = task.evaluate({
            "train":           train_dataset,
            "val":             val_dataset,
            "_train_h5_path":  train_h5_path,
            "_val_h5_path":    val_h5_path,
        })

    else:
        raise NotImplementedError(
            f"Task '{task_name}' is not supported. "
            "Supported tasks: kendalls_tau, classification, expert_projection, latent_distance_heatmap."
        )

    # --- print results ---
    print()
    print("=" * 42)
    print(f"  task_name    : {result['task_name']}")
    print(f"  metric_name  : {result['metric_name']}")
    print(f"  metric_value : {result['metric_value']:.6f}")
    print("=" * 42)


# ---------------------------------------------------------------------------
# __main__ entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TCC embeddings on downstream tasks")
    parser.add_argument(
        "--task", type=str, default="kendalls_tau",
        choices=["kendalls_tau", "expert_projection", "classification", "latent_distance_heatmap"],
        help="[v2] Evaluation task (default: kendalls_tau). "
             "Config is read from configs_v2/eval/<task>.yaml.",
    )
    args = parser.parse_args()

    main(task_name=args.task)
