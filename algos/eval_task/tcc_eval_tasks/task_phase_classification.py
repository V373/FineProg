"""Phase Classification evaluation task.

Uses pre-extracted TCC embeddings as frozen features.  A linear SVM is trained
on labeled training videos and evaluated on labeled validation videos.

Reference: TCC paper §5 – Phase Classification.
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECTS_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
if _PROJECTS_ROOT not in sys.path:
    sys.path.insert(0, _PROJECTS_ROOT)

import datetime

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fineprog.algos.eval_task.base_task import BaseTask

# outputs/ lives under the fineprog project root (4 levels up from this file)
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_CM_OUTPUTS_DIR = os.path.join(_PROJ_ROOT, "outputs", "confusion_matrix")


def _save_confusion_heatmap(
    cm: np.ndarray,
    labels: list,
    accuracy: float,
    output_dir: str = None,
) -> str:
    """Row-normalize confusion matrix and save as a heatmap PNG."""
    if output_dir is None:
        output_dir = _CM_OUTPUTS_DIR

    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.where(row_sums == 0, 0.0, cm.astype(float) / row_sums)

    n = len(labels)
    fig_size = max(4, n * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(7, min(10, int(72 / n))), color=text_color)

    str_labels = [str(l) for l in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(str_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(str_labels, fontsize=8)
    ax.set_xlabel("Predicted phase", fontsize=9)
    ax.set_ylabel("True phase", fontsize=9)
    ax.set_title(f"Confusion matrix (row-normalized)  acc={accuracy:.4f}", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"confusion_matrix_{ts}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


class PhaseClassificationTask(BaseTask):
    """Train a linear SVM on labeled training frames and report per-frame
    phase classification accuracy on labeled validation frames.

    Args:
        svm_c:    Regularization parameter C for LinearSVC.
        max_iter: Maximum number of iterations for LinearSVC.
    """

    def __init__(self, svm_c: float = 1.0, max_iter: int = 10000, output_dir: str = None,
                 gen_tsne_phase_label: bool = False):
        super().__init__(task_name="classification")
        self.svm_c = svm_c
        self.max_iter = max_iter
        self.output_dir = output_dir
        self.gen_tsne_phase_label = gen_tsne_phase_label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_labeled_frames(self, dataset: dict, split_name: str):
        """Collect all valid labeled frames from a dataset dict.

        Args:
            dataset:    dict produced by load_embeddings_h5().
            split_name: "train" or "val" (used only for logging).

        Returns:
            Tuple (X, y, video_records):
                X            – np.ndarray [N, D]
                y            – np.ndarray [N]
                video_records– list of info dicts for each participating video
        """
        X_parts, y_parts = [], []
        video_records = []

        labeled_list  = dataset.get("labeled", [False] * len(dataset["video_id"]))
        phase_list    = dataset.get("phase_labels", [None] * len(dataset["video_id"]))

        for idx, vid_id in enumerate(dataset["video_id"]):
            embs   = dataset["embeddings"][idx]      # [T, D]
            phases = phase_list[idx]                  # [T] or None
            labeled = labeled_list[idx]

            if not labeled:
                print(f"[Classification][{split_name}] SKIP video_id={vid_id}: labeled attr is False")
                continue
            if phases is None:
                print(f"[Classification][{split_name}] SKIP video_id={vid_id}: missing phase_labels")
                continue

            mask = phases != -1
            valid_frames = int(mask.sum())
            if valid_frames == 0:
                print(f"[Classification][{split_name}] SKIP video_id={vid_id}: "
                      "no valid frames after filtering phase_labels == -1")
                continue

            X_parts.append(embs[mask])
            y_parts.append(phases[mask])
            unique_phases = np.unique(phases[mask]).tolist()
            video_records.append({
                "video_id":      vid_id,
                "n_frames":      valid_frames,
                "unique_phases": unique_phases,
                "emb_idx":       idx,
            })
            print(f"[Classification][{split_name}] ADD  video_id={vid_id} "
                  f"frames={valid_frames} phases={unique_phases}")

        if X_parts:
            X = np.concatenate(X_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
        else:
            X = np.empty((0, dataset["embeddings"][0].shape[-1] if dataset["embeddings"] else 128))
            y = np.empty((0,), dtype=np.int64)

        return X, y, video_records

    # ------------------------------------------------------------------
    # t-SNE phase-label H5 generator
    # ------------------------------------------------------------------

    def _generate_tsne_h5(self, data: dict, clf) -> str:
        """Write a combined embeddings + phase-label H5 for t-SNE visualization.

        Train labeled videos  → original phase labels,  is_ground_truth=True.
        Train unlabeled videos → SVM-predicted labels,  is_ground_truth=False.
        Val videos (all)      → SVM-predicted labels,  is_ground_truth=False.

        Val video groups are written after train groups.  If a val video_id
        collides with an existing train video_id, the val entry is keyed as
        "val_<video_id>" in the H5 file.

        Args:
            data: dict passed to evaluate(), containing keys "train", "val",
                  and optionally "_train_h5_path" / "_val_h5_path".
            clf:  Trained sklearn pipeline (StandardScaler + SVC).

        Returns:
            Path of the written H5 file.
        """
        train_dataset = data["train"]
        val_dataset   = data["val"]

        # Determine output directory: same as the train labeled H5 file
        train_h5_path = data.get("_train_h5_path")
        if train_h5_path:
            out_dir = os.path.dirname(os.path.abspath(train_h5_path))
        else:
            out_dir = _CM_OUTPUTS_DIR  # fallback

        out_path = os.path.join(out_dir, "embd_tsne_phase_label.h5")
        os.makedirs(out_dir, exist_ok=True)

        labeled_list_tr = train_dataset.get("labeled",      [False] * len(train_dataset["video_id"]))
        phase_list_tr   = train_dataset.get("phase_labels", [None]  * len(train_dataset["video_id"]))

        print(f"\n[Classification] Generating t-SNE phase-label H5 → {out_path}")

        with h5py.File(out_path, "w") as f:
            vg = f.create_group("videos")
            written_ids: set = set()

            # ---- training set ----
            for idx, vid_id in enumerate(train_dataset["video_id"]):
                embs    = train_dataset["embeddings"][idx]   # [T, D]
                labeled = labeled_list_tr[idx]
                phases  = phase_list_tr[idx]                 # ndarray or None

                if labeled and phases is not None:
                    phase_out       = phases.astype(np.int64)
                    is_ground_truth = True
                else:
                    phase_out       = clf.predict(embs).astype(np.int64)
                    is_ground_truth = False

                grp = vg.create_group(vid_id)
                grp.create_dataset("embeddings",   data=embs.astype(np.float32), compression="gzip")
                grp.create_dataset("phase_labels", data=phase_out,               compression="gzip")
                grp.attrs["is_ground_truth"] = is_ground_truth
                grp.attrs["data_type"]       = "train"
                written_ids.add(vid_id)
                print(f"[Classification][tsne_h5] train vid={vid_id} "
                      f"gt={is_ground_truth} frames={len(embs)}")

            # ---- val set ----
            for idx, vid_id in enumerate(val_dataset["video_id"]):
                embs      = val_dataset["embeddings"][idx]
                phase_out = clf.predict(embs).astype(np.int64)

                # Resolve collision with a train video that has the same ID
                h5_key = f"val_{vid_id}" if vid_id in written_ids else vid_id

                grp = vg.create_group(h5_key)
                grp.create_dataset("embeddings",   data=embs.astype(np.float32), compression="gzip")
                grp.create_dataset("phase_labels", data=phase_out,               compression="gzip")
                grp.attrs["is_ground_truth"] = False
                grp.attrs["data_type"]       = "val"
                written_ids.add(h5_key)
                print(f"[Classification][tsne_h5] val   vid={vid_id} "
                      f"(h5_key={h5_key}) frames={len(embs)}")

        print(f"[Classification] t-SNE phase-label H5 saved → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def evaluate(self, data: dict) -> dict:
        """Run phase classification evaluation.

        Args:
            data: dict with keys "train" and "val", each being a dataset dict
                  produced by load_embeddings_h5().

        Returns:
            dict with keys "task_name", "metric_name", "metric_value".
        """
        train_dataset = data["train"]
        val_dataset   = data["val"]

        print(f"\n[Classification] Collecting training frames ...")
        X_train, y_train, train_records = self._collect_labeled_frames(train_dataset, "train")

        print(f"\n[Classification] Collecting validation frames ...")
        X_val,   y_val,   val_records   = self._collect_labeled_frames(val_dataset,   "val")

        # ---- sanity checks ----
        n_train_labeled = len(train_records)
        n_val_labeled   = len(val_records)
        print(f"\n[Classification] train labeled videos : {n_train_labeled}")
        print(f"[Classification] val   labeled videos : {n_val_labeled}")
        print(f"[Classification] X_train shape        : {X_train.shape}")
        print(f"[Classification] X_val   shape        : {X_val.shape}")

        if X_train.shape[0] == 0:
            raise RuntimeError("[Classification] No valid training frames found. Cannot train SVM.")

        if X_val.shape[0] == 0:
            raise RuntimeError("[Classification] No valid validation frames found. Cannot evaluate.")

        train_unique = np.unique(y_train).tolist()
        val_unique   = np.unique(y_val).tolist()
        print(f"[Classification] y_train unique phases: {train_unique}")
        print(f"[Classification] y_val   unique phases: {val_unique}")

        if len(train_unique) < 2:
            raise RuntimeError(
                f"[Classification] Training set has only {len(train_unique)} phase class(es): "
                f"{train_unique}. SVM requires at least 2 classes."
            )

        # ---- train/val phase mismatch warning ----
        train_unique_set = set(train_unique)
        val_unique_set   = set(val_unique)
        missing_in_train = sorted(val_unique_set - train_unique_set)
        if missing_in_train:
            print(f"[Classification] WARNING: validation phases not present in training: {missing_in_train}")

        # ---- train SVC (RBF) ----
        print(f"\n[Classification] Training SVC-RBF (C={self.svm_c}, decision_function_shape='ovo') ...")
        clf = make_pipeline(
            StandardScaler(),
            SVC(
                C=self.svm_c,
                kernel="rbf",
                decision_function_shape="ovo",
            ),
        )
        clf.fit(X_train, y_train)
        print("[Classification] Training done.")

        # ---- overall accuracy ----
        y_pred   = clf.predict(X_val)
        accuracy = float(accuracy_score(y_val, y_pred))
        print(f"\n[Classification] Overall phase accuracy: {accuracy:.4f}")

        # ---- confusion matrix ----
        all_labels = sorted(train_unique_set | val_unique_set)
        cm = confusion_matrix(y_val, y_pred, labels=all_labels)
        print(f"\n[Classification] Confusion matrix labels: {all_labels}")
        print("[Classification] Confusion matrix:")
        print(cm)
        print("[Classification] Classification report:")
        print(classification_report(y_val, y_pred, labels=all_labels, zero_division=0))

        # ---- confusion matrix heatmap ----
        save_path = _save_confusion_heatmap(cm, all_labels, accuracy, output_dir=self.output_dir)
        print(f"[Classification] confusion matrix heatmap saved → {save_path}")

        # ---- optional: generate combined t-SNE phase-label H5 ----
        if self.gen_tsne_phase_label:
            self._generate_tsne_h5(data, clf)

        # ---- per-video accuracy ----
        labeled_list = val_dataset.get("labeled", [False] * len(val_dataset["video_id"]))
        phase_list   = val_dataset.get("phase_labels", [None] * len(val_dataset["video_id"]))

        print("\n[Classification] Per-video validation accuracy:")
        for idx, vid_id in enumerate(val_dataset["video_id"]):
            if not labeled_list[idx]:
                continue
            phases = phase_list[idx]
            if phases is None:
                continue
            mask = phases != -1
            if not mask.any():
                continue
            embs = val_dataset["embeddings"][idx]
            X_video   = embs[mask]
            y_video   = phases[mask]
            y_pred_v  = clf.predict(X_video)
            acc_video = float(accuracy_score(y_video, y_pred_v))
            unique_v  = np.unique(y_video).tolist()
            print(f"  [Classification][val] video_id={vid_id} "
                  f"frames={int(mask.sum())} phases={unique_v} acc={acc_video:.4f}")

        return {
            "task_name":    "classification",
            "metric_name":  "phase_accuracy",
            "metric_value": accuracy,
        }
