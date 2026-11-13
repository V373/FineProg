"""Train the single-frame R/M/G semantic classifier with cross-entropy only."""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

try:
    from .dataset_preparation.semantic_frame_dataset import SemanticFrameDataset
    from .models.semantic_classifier import LABELS, SemanticClassifier
    from .utils.config_v2 import ConfigV2
except ImportError:
    from dataset_preparation.semantic_frame_dataset import SemanticFrameDataset
    from models.semantic_classifier import LABELS, SemanticClassifier
    from utils.config_v2 import ConfigV2

ROOT = Path(__file__).resolve().parent


def classification_metrics(labels, predictions):
    """Accuracy, per-class precision / recall, confusion matrix (rows true, columns predicted)."""
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    hits = np.diag(confusion)

    def per_class(totals):
        # None: the class was never predicted (precision) or never present (recall).
        return {name: float(hit / total) if total else None
                for name, hit, total in zip(LABELS, hits, totals)}

    return {"accuracy": float(hits.sum() / confusion.sum()),
            "precision": per_class(confusion.sum(0)), "recall": per_class(confusion.sum(1)),
            "confusion_matrix": confusion.tolist(),
            "R_as_G": int(confusion[0, 2]), "G_as_R": int(confusion[2, 0]),
            "num_frames": int(confusion.sum())}


def _absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def run(config_path):
    resolver = ConfigV2()
    cfg = resolver.load_train(str(config_path))
    cfg["val_h5_path"] = resolver.resolve_dataset(cfg["val_dataset"])["processed_h5_path"]
    for key in ("train_labels_csv", "val_labels_csv"):
        cfg[key] = str(_absolute(cfg[key]))
    cfg["output_dir"] = str(_absolute(cfg.get("output_dir") or
                                      f"outputs/semantic/{datetime.now():%Y%m%d-%H%M%S}"))
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True)  # A run never overwrites an existing directory.
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    train_ds = SemanticFrameDataset(cfg["h5_path"], cfg["train_labels_csv"])
    val_ds = SemanticFrameDataset(cfg["val_h5_path"], cfg["val_labels_csv"])
    val_frames = torch.from_numpy(val_ds.frames).permute(0, 3, 1, 2)
    print(f"train frames R/M/G={np.bincount(train_ds.labels, minlength=3).tolist()} "
          f"val frames R/M/G={np.bincount(val_ds.labels, minlength=3).tolist()}", flush=True)

    model = SemanticClassifier(pretrained=cfg["pretrained"],
                               l2_normalize=cfg["l2_normalize"]).to(device).train()
    optimizer = torch.optim.Adam(model.param_groups(cfg["backbone_lr"], cfg["head_lr"]))
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                        sampler=train_ds.balanced_sampler(cfg["num_steps"] * cfg["batch_size"]))
    history = []

    def validate(step):
        probabilities, predictions = (x.cpu().numpy() for x in model.predict(val_frames))
        history.append({"step": step, **classification_metrics(val_ds.labels, predictions)})
        (output / "val_metrics.json").write_text(json.dumps(history, indent=2))
        print(f"validation step={step} accuracy={history[-1]['accuracy']:.4f} "
              f"confusion={history[-1]['confusion_matrix']}", flush=True)
        return probabilities, predictions

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "loss", "batch_accuracy"])
        for step, (frames, labels) in enumerate(loader, 1):
            frames, labels = frames.to(device), labels.to(device)
            logits = model(frames)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            writer.writerow([step, loss.item(), (logits.argmax(1) == labels).float().mean().item()])
            if step % cfg["eval_every"] == 0 or step == cfg["num_steps"]:
                file.flush()
                print(f"step={step} loss={loss.item():.6f}", flush=True)
                probabilities, predictions = validate(step)

    with (output / "val_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["video_id", "frame_idx", "label", "prediction", *(f"p_{name}" for name in LABELS)])
        for row in zip(val_ds.video_ids, val_ds.frame_idx, val_ds.labels, predictions, probabilities):
            writer.writerow([row[0], row[1], LABELS[row[2]], LABELS[row[3]], *(f"{p:.6f}" for p in row[4])])
    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "labels": list(LABELS)},
               output / "semantic_classifier.pt")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ConfigV2()._root / "train_semantic.yaml"))
    run(parser.parse_args().config)
