"""Three training entry points and offline teacher labels.

Each train function creates its output directory and returns the final checkpoint
Path. predict_labels creates its output directory and returns labels.csv. All
datasets and initial weights come from the supplied task/config, never a prior run.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import resolve_path
from .data import SemanticFrameDataset, VideoDataset, frame_tensor, read_labels, read_videos
from .losses import replay_loss, semantic_set_loss
from .models import LABELS, LocalEncoder, SemanticClassifier, load_local, load_semantic


def _start(output, cfg):
    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    output = Path(output)
    output.mkdir(parents=True)
    return output


def _write_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _loader(dataset, batch_size):
    if len(dataset) < batch_size:
        raise ValueError("Dataset has fewer videos than the requested batch size")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def _encode(model, frames, device):
    batch, time = frames.shape[:2]
    return model(frames.flatten(0, 1).to(device)).reshape(batch, time, -1)


def _replay(z, batch, cfg, device):
    return replay_loss(z, batch["target_steps"].to(device), batch["seq_len"].to(device),
                       **{key: cfg[key] for key in ("temperature", "variance_lambda",
                           "triplet_margin", "triplet_weight") if key in cfg})


def _save_local(model, path, cfg, step):
    torch.save({"model_state_dict": model.state_dict(),
                "encoder_config": {"single_frame_context": True, "context_size": 2,
                                   "embedding_dim": 128, "embedding_normalization": "l2"},
                "config": {key: str(value) if isinstance(value, Path) else value
                           for key, value in cfg.items()}, "step": step}, path)
    return path


def _train_step(loss, optimizer, scaler):
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def train_lower(task, cfg, output, device="cpu", max_steps=None):
    """Train expert-only local Stage1; return its final checkpoint Path."""
    output = _start(output, cfg)
    device = torch.device(device)
    dataset = VideoDataset(task["data"]["train"], cfg.get("clip_len", 20),
                           random_offset=cfg.get("random_offset", 1))
    loader = _loader(dataset, cfg.get("batch_size", 12))
    if loader.batch_size < 2:
        raise ValueError("Lower training needs at least two videos per batch for TCC")
    pretrained = resolve_path(cfg["pretrained_path"]) if cfg.get("pretrained_path") else None
    model = LocalEncoder(pretrained).to(device).train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg.get("learning_rate", 4e-4))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    total = cfg.get("num_epochs", 20000) * len(loader)
    total = min(total, max_steps) if max_steps is not None else total
    every = cfg.get("checkpoint_every_epochs", 2000) * len(loader)
    step = 0
    for epoch in range(cfg.get("num_epochs", 20000)):
        for batch in loader:
            step += 1
            with torch.autocast(device_type=device.type, enabled=amp):
                z = _encode(model, batch["frames"], device)
                loss = _replay(z, batch, cfg, device)
            _train_step(loss, optimizer, scaler)
            _write_csv(output / "loss.csv", {"step": step, "epoch": epoch + 1, "loss": float(loss.detach())})
            if step % every == 0 or step == total:
                name = "checkpoint.pt" if step == total else f"encoder_epoch{epoch + 1:06d}.pt"
                checkpoint = _save_local(model, output / name, cfg, step)
            if step == total:
                return checkpoint


@torch.no_grad()
def _classification_validation(model, dataset, device, batch_size=256):
    model.eval()
    confusion = torch.zeros(3, 3, dtype=torch.long)
    for frames, labels in DataLoader(dataset, batch_size=batch_size):
        predictions = model(frames.to(device)).argmax(-1).cpu()
        confusion += torch.bincount(labels * 3 + predictions, minlength=9).reshape(3, 3)
    model.train()
    totals = confusion.sum(1)
    return {"accuracy": float(confusion.diag().sum() / confusion.sum()),
            "recall": {name: float(confusion[i, i] / totals[i]) if totals[i] else None
                       for i, name in enumerate(LABELS)}, "confusion_matrix": confusion.tolist()}


def train_upper(task, cfg, output, device="cpu", max_steps=None):
    """Train the independent R/M/G classifier; return its final checkpoint Path."""
    output = _start(output, cfg)
    device = torch.device(device)
    dataset = SemanticFrameDataset(task["data"]["train"], task["labels"]["train"])
    validation = SemanticFrameDataset(task["data"]["valid"], task["labels"]["valid"])
    pretrained = resolve_path(cfg["pretrained_path"]) if cfg.get("pretrained_path") else None
    model = SemanticClassifier(pretrained,
                               cfg.get("l2_normalize", False)).to(device).train()
    optimizer = torch.optim.Adam(model.param_groups(cfg.get("backbone_lr", 1e-4), cfg.get("head_lr", 1e-3)))
    steps = cfg.get("num_steps", 400)
    steps = min(steps, max_steps) if max_steps is not None else steps
    batch_size = cfg.get("batch_size", 64)
    weights = 1.0 / np.bincount(dataset.labels, minlength=3)[dataset.labels]
    sampler = WeightedRandomSampler(torch.from_numpy(weights), steps * batch_size, replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    history = []
    for step, (frames, labels) in enumerate(loader, 1):
        logits = model(frames.to(device))
        loss = F.cross_entropy(logits, labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _write_csv(output / "loss.csv", {"step": step, "loss": float(loss.detach())})
        if step % cfg.get("eval_every", 100) == 0 or step == steps:
            history.append({"step": step, **_classification_validation(model, validation, device)})
            (output / "validation.json").write_text(json.dumps(history, indent=2))
    checkpoint = output / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "config": {**{key: str(value) if isinstance(value, Path) else value
                              for key, value in cfg.items()}, "l2_normalize": model.l2_normalize},
                "labels": list(LABELS)}, checkpoint)
    return checkpoint


def _cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def _encode_steps(model, pixels, steps, device, chunk_size):
    return torch.cat([model(frame_tensor(pixels[steps[start:start + chunk_size]]).to(device))
                      for start in range(0, len(steps), chunk_size)])


class FixedValidation:
    """Sample reference coordinates once, then validate every held-out frame."""

    def __init__(self, expert, validation, cfg):
        self.expert, self.validation = expert, validation
        self.tau = cfg.get("set_tau", 0.128)
        self.chunk_size = cfg.get("validation_chunk_size", 20)
        rng = np.random.default_rng(cfg.get("seed", 42))
        self.references = {}
        for vid in expert.video_ids:
            selected = []
            for category in range(3):
                candidates = np.flatnonzero(expert.labels[vid] == category)
                if len(candidates):
                    count = cfg.get("refs_per_class", 4)
                    selected.extend(rng.choice(candidates, count, replace=len(candidates) < count))
            if selected:
                self.references[vid] = np.asarray(selected)

    @torch.no_grad()
    def evaluate(self, model, device):
        was_training = model.training
        tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        model.eval()
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            refs = torch.cat([_encode_steps(model, self.expert.videos[vid], steps, device, self.chunk_size)
                              for vid, steps in self.references.items()])
            labels = torch.tensor(np.concatenate([self.expert.labels[vid][steps]
                                  for vid, steps in self.references.items()]), device=device)
            losses, targets = [], []
            for vid, pixels in self.validation.videos.items():
                z = _encode_steps(model, pixels, np.arange(len(pixels)), device, self.chunk_size)
                target = torch.as_tensor(self.validation.labels[vid], device=device)
                for start in range(0, len(z), 64):
                    q, t = z[start:start + 64].float(), target[start:start + 64]
                    same = t[:, None] == labels[None]
                    valid = ((t == 0) | (t == 2)) & same.any(1) & (~same).any(1)
                    logits = -(q[:, None] - refs[None]).square().sum(-1) / self.tau
                    nll = logits.logsumexp(-1) - logits.masked_fill(~same, -torch.inf).logsumexp(-1)
                    losses.append(nll[valid].cpu())
                    targets.append(t[valid].cpu())
            losses, targets = torch.cat(losses), torch.cat(targets)
            return {"set_loss": float(losses.mean()) if len(losses) else None,
                    **{f"{name}_loss": float(losses[targets == category].mean())
                       if (targets == category).any() else None for category, name in ((0, "r"), (2, "g"))},
                    "valid_queries": len(losses)}
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
            model.train(was_training)


def train_stage2(task, cfg, output, device="cpu", max_steps=None):
    """Fine-tune local Stage1 with teacher R/G sets; return final checkpoint Path.

    cfg['query_labels'] supplies pseudo labels for training IDs. Validation always
    reads task['labels'][query_split], keeping the held-out human labels separate.
    """
    specification = task["stage2"]
    train_ids, valid_ids = specification["train_ids"], specification["valid_ids"]
    if set(train_ids) & set(valid_ids):
        raise ValueError("Stage2 training and validation episodes must be disjoint")
    output = _start(output, cfg)
    device = torch.device(device)
    split = specification["query_split"]
    common = {"refs_per_class": cfg.get("refs_per_class", 4), "random_offset": cfg.get("random_offset", 1)}
    expert = VideoDataset(task["data"]["train"], cfg.get("clip_len", 20),
                          labels_path=task["labels"]["train"], role="expert", **common)
    query = VideoDataset(task["data"][split], cfg.get("query_steps", 20), train_ids,
                         resolve_path(cfg["query_labels"]), "query", **common)
    validation = VideoDataset(task["data"][split], cfg.get("query_steps", 20), valid_ids,
                              task["labels"][split], "query", **common)
    expert_loader = _loader(expert, cfg.get("expert_batch_size", 2))
    query_loader = _loader(query, cfg.get("query_batch_size", 1))
    expert_iter, query_iter = _cycle(expert_loader), _cycle(query_loader)
    model = load_local(task["weights"]["lower"], device).train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg.get("learning_rate", 4e-5))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=128)
    validator = FixedValidation(expert, validation, cfg)
    history = [{"step": 0, **validator.evaluate(model, device)}]
    total = cfg.get("num_epochs", 100) * len(expert_loader)
    total = min(total, max_steps) if max_steps is not None else total
    every = cfg.get("checkpoint_every_epochs", 25) * len(expert_loader)
    for step in range(1, total + 1):
        eb, qb = next(expert_iter), next(query_iter)
        with torch.autocast(device_type=device.type, enabled=amp):
            ez, qz = _encode(model, eb["frames"], device), _encode(model, qb["frames"], device)
            count = cfg.get("clip_len", 20)
            replay = _replay(ez[:, :count], {**eb, "target_steps": eb["target_steps"][:, :count]}, cfg, device)
            semantic = semantic_set_loss(qz.flatten(0, 1), ez[:, count:].flatten(0, 1),
                                          qb["labels"].to(device).flatten(),
                                          eb["labels"][:, count:].to(device).flatten(), cfg.get("set_tau", 0.128))
            loss = replay + cfg.get("set_weight", 1.0) * semantic
        _train_step(loss, optimizer, scaler)
        _write_csv(output / "loss.csv", {"step": step, "loss": float(loss.detach()),
                                        "replay": float(replay.detach()), "set": float(semantic.detach())})
        if step % every == 0 or step == total:
            name = "checkpoint.pt" if step == total else f"encoder_epoch{step // len(expert_loader):06d}.pt"
            checkpoint = _save_local(model, output / name, cfg, step)
            history.append({"step": step, **validator.evaluate(model, device)})
            (output / "validation.json").write_text(json.dumps(history, indent=2))
    return checkpoint


@torch.no_grad()
def predict_labels(task, checkpoint, split, output, device="cpu"):
    """Label every H5 frame independently with a frozen teacher; return CSV Path."""
    output = Path(output)
    output.mkdir(parents=True)
    model = load_semantic(checkpoint, device)
    destination = output / "labels.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "id", "frame_idx", "label", "label_id"])
        for vid, pixels in read_videos(task["data"][split]).items():
            for start in range(0, len(pixels), 256):
                labels = model(frame_tensor(pixels[start:start + 256]).to(device)).argmax(-1).cpu().tolist()
                writer.writerows([vid, vid, start + i, LABELS[label], label] for i, label in enumerate(labels))
    return destination
