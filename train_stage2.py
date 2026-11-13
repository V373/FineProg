"""Semantic-set finetuning of existing single-frame Local / HF1 checkpoints."""

import argparse
import csv
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from .algos.loss.encoder_loss import build_loss
    from .algos.loss.loss_semantic_set import semantic_set_loss
    from .dataset_preparation.stage2_dataset import Stage2VideoDataset, collate_stage2
    from .utils.config_v2 import ConfigV2
    from .utils.encoder_checkpoint import build_encoder_from_checkpoint, checkpoint_dict, encoder_config
    from .utils.stage2_validation import Stage2Validator
except ImportError:
    from algos.loss.encoder_loss import build_loss
    from algos.loss.loss_semantic_set import semantic_set_loss
    from dataset_preparation.stage2_dataset import Stage2VideoDataset, collate_stage2
    from utils.config_v2 import ConfigV2
    from utils.encoder_checkpoint import build_encoder_from_checkpoint, checkpoint_dict, encoder_config
    from utils.stage2_validation import Stage2Validator

ROOT = Path(__file__).resolve().parent


def build_stage2_encoder(checkpoint_path, device):
    """Restore all weights first; change training policy without rebuilding memory."""
    encoder = build_encoder_from_checkpoint(checkpoint_path, device)
    if not encoder.single_frame_context:
        raise ValueError("Pick Mango stage2 requires a single_frame_context=true source checkpoint")
    if encoder.memory is not None:
        if type(encoder.memory).__name__ != "StreamingMamba2" or encoder.memory.use_residual:
            raise ValueError("Stage2 supports local-only or single-branch no-residual Mamba2")
        encoder.memory_config.update(freeze_bn_stats=False, cache_local_embeddings=False,
                                     visual_chunk_size=20, checkpoint_visual=True)
    encoder._train_base = "only_bn"
    encoder._train_embedding = True
    encoder.train()
    encoder.configure_trainability()
    return encoder


def encode_batch(encoder, batch, device):
    if encoder.memory is not None:
        raw = [encoder.encode_local_episode(frames) for frames in batch["frames_full"]]
        return encoder.forward_episode(raw, batch["seq_len"], batch["target_steps"])
    frames = batch["frames"].to(device)
    frames = frames.float().div(255) if frames.dtype == torch.uint8 else frames
    return encoder(frames)


def compute_stage2_loss(encoder, expert_batch, query_batch, expert_loss, device,
                        set_weight=1.0, set_tau=0.128):
    if not np.isfinite(set_weight) or set_weight < 0:
        raise ValueError("set_weight must be finite and nonnegative")
    expert = encode_batch(encoder, expert_batch, device)
    query = encode_batch(encoder, query_batch, device)
    replay_count = expert_batch["replay_count"]
    replay = expert_loss(expert[:, :replay_count], {
        "target_steps": expert_batch["target_steps"][:, :replay_count].to(device),
        "seq_len": expert_batch["seq_len"].to(device),
    })
    semantic = semantic_set_loss(
        query.flatten(0, 1), expert[:, replay_count:].flatten(0, 1),
        query_batch["labels"].to(device).flatten(),
        expert_batch["labels"][:, replay_count:].to(device).flatten(), tau=set_tau)
    loss = replay["loss"] + set_weight * semantic["loss"]
    return {"loss": loss, "metrics": {
        **{f"expert/{k}": v for k, v in replay.get("metrics", {}).items()},
        **{f"set/{k}": v for k, v in semantic["metrics"].items()},
        "expert_loss": replay["loss"].detach(), "set_loss": semantic["loss"].detach(),
        "weighted_set_loss": (set_weight * semantic["loss"]).detach(),
        "loss": loss.detach(),
    }}


def _parameter_group(name, parameter):
    if name.startswith("backbone."):
        return "backbone_bn" if parameter.requires_grad else "backbone_frozen"
    if name.startswith("temporal_embedder."):
        return "embedder"
    return "memory_projection" if name.startswith("memory.proj.") else "memory"


def _norm(tensors):
    return sum(float(t.detach().float().square().sum()) for t in tensors) ** 0.5


def _cycle(loader):
    while True:
        yield from loader


def _absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def run(config_path):
    resolver = ConfigV2()
    cfg = resolver.load_train(str(config_path))
    for key in ("checkpoint_path", "expert_labels_csv", "nonexpert_labels_csv", "loss_config"):
        cfg[key] = str(_absolute(cfg[key]))
    cfg["output_dir"] = str(_absolute(cfg.get("output_dir") or
        f".vibe/.codex/astra/{datetime.now():%Y%m%d-%H%M%S}_stage2"))
    cfg["nonexpert_h5_path"] = resolver.resolve_dataset(cfg["nonexpert_dataset"])["processed_h5_path"]
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.csv").exists():
        raise FileExistsError(f"Use a new output_dir instead of overwriting {output}")
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    device = torch.device(cfg.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the real HF1 smoke requires CUDA")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    encoder = build_stage2_encoder(cfg["checkpoint_path"], device)
    initial = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    source = torch.load(cfg["checkpoint_path"], map_location="cpu", weights_only=True)
    source_state = source["model_state_dict"]
    initial_matches = all(torch.equal(v, source_state[k]) for k, v in initial.items())
    assert initial_matches, "Applying stage2 settings changed checkpoint weights/buffers"
    del source, source_state
    cfg.update(encoder_config=encoder_config(encoder), train_base="only_bn",
               train_embedding=True, single_frame_context=True, extract_backbone_cache=False)
    cfg["memory"] = dict(encoder.memory_config)
    # Snapshot actual loss files and labels so subsequent external edits cannot change this run.
    loss_source = Path(cfg["loss_config"])
    shutil.copytree(loss_source.parent, output / "loss")
    cfg["loss_config"] = str(output / "loss" / loss_source.name)
    for role in ("expert", "nonexpert"):
        dest = output / f"{role}_labels.csv"
        shutil.copy2(cfg[f"{role}_labels_csv"], dest)
        cfg[f"{role}_labels_csv"] = str(dest)
    common = dict(encoder_config=encoder_config(encoder),
                  refs_per_class=cfg.get("refs_per_class", 4),
                  query_steps=cfg.get("query_steps", 20),
                  sampling_strategy=cfg.get("sampling_strategy", "offset_uniform"),
                  random_offset=cfg.get("random_offset", 1))
    expert_ds = Stage2VideoDataset(cfg["h5_path"], cfg["expert_labels_csv"], role="expert",
                                  video_ids=cfg.get("expert_video_ids"), **common)
    query_ds = Stage2VideoDataset(cfg["nonexpert_h5_path"], cfg["nonexpert_labels_csv"], role="query",
                                 video_ids=cfg["nonexpert_video_ids"], **common)
    def loader(ds, batch_size):
        if len(ds) < batch_size:
            raise ValueError("Training subset is smaller than the configured batch")
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                          num_workers=0, collate_fn=collate_stage2)
    expert_loader = loader(expert_ds, cfg.get("expert_batch_size", 2))
    query_loader = loader(query_ds, cfg.get("nonexpert_batch_size", 1))
    steps_per_epoch = len(expert_loader)
    epoch_mode = cfg.get("num_epochs") is not None
    if epoch_mode:
        if int(cfg["num_epochs"]) < 1:
            raise ValueError("num_epochs must be positive")
        cfg["num_steps"] = int(cfg["num_epochs"]) * steps_per_epoch
    cfg["steps_per_epoch"] = steps_per_epoch
    checkpoint_every_epochs = int(cfg.get("checkpoint_every_epochs", 25))
    if checkpoint_every_epochs < 1:
        raise ValueError("checkpoint_every_epochs must be positive")
    validator = None
    validation_ids = cfg.get("validation_video_ids")
    if validation_ids:
        if set(validation_ids) & set(query_ds.video_ids):
            raise ValueError("Training and validation episodes must be disjoint")
        validation_ds = Stage2VideoDataset(
            cfg["nonexpert_h5_path"], cfg["nonexpert_labels_csv"], role="query",
            video_ids=validation_ids, **common)
        validator = Stage2Validator(expert_ds, validation_ds,
                                   refs_per_class=cfg.get("refs_per_class", 4), seed=seed,
                                   tau=cfg.get("set_tau", 0.128))
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    expert_iter, query_iter = _cycle(expert_loader), _cycle(query_loader)
    expert_loss = build_loss("composite", cfg["loss_config"]).to(device)
    optimizer = torch.optim.Adam([p for p in encoder.parameters() if p.requires_grad],
                                 lr=cfg.get("learning_rate", 4e-5))
    amp = device.type == "cuda"
    # Low-temperature TCC can overflow FP16 gradients at the default 65536 scale.
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=128)
    groups = {}
    for name, param in encoder.named_parameters():
        groups.setdefault(_parameter_group(name, param), []).append((name, param))
    bn_names = [name for name, _ in encoder.named_buffers() if name.endswith("running_mean")]
    count_names = [name for name, _ in encoder.named_buffers() if name.endswith("num_batches_tracked")]
    records, started = [], time.perf_counter()
    validation_history = []
    best_validation = None

    def save_encoder(step, epoch):
        name = f"encoder_epoch{epoch:06d}.pt" if epoch_mode else f"encoder_step{step:06d}.pt"
        path = output / name
        payload = checkpoint_dict(encoder, training_provenance={
            "stage": 2, "source_checkpoint": cfg["checkpoint_path"], "step": step, "epoch": epoch})
        payload["stage2_config"] = cfg
        torch.save(payload, path)
        return path

    def validate(step, epoch, checkpoint):
        nonlocal best_validation
        if validator is None:
            return
        evaluation = validator.evaluate(encoder, output / f"eval_epoch{epoch:06d}", epoch)
        item = {"epoch": epoch, "step": step, "checkpoint": str(checkpoint),
                **{k: evaluation[k] for k in ("set_loss", "r_loss", "g_loss", "positive_mass")}}
        validation_history.append(item)
        if best_validation is None or item["set_loss"] < best_validation["set_loss"]:
            best_validation = dict(item)
        (output / "validation_history.json").write_text(json.dumps(validation_history, indent=2))
        print(f"validation epoch={epoch} set={item['set_loss']:.6f} "
              f"R={item['r_loss']} G={item['g_loss']}", flush=True)

    validate(0, 0, cfg["checkpoint_path"])
    last_checkpoint_path, last_checkpoint_step = None, 0
    single_frame_observed = {"calls": 0, "all_equal": True}
    def check_single_frame(module, args):
        frames = args[0].detach().reshape(-1, encoder.context_size, *args[0].shape[1:])
        single_frame_observed["calls"] += 1
        single_frame_observed["all_equal"] &= bool(torch.equal(frames[:, 0], frames[:, 1]))
    hook = encoder.backbone.register_forward_pre_hook(check_single_frame)
    for step in range(1, cfg.get("num_steps", 20) + 1):
        tick = time.perf_counter()
        expert_batch, query_batch = next(expert_iter), next(query_iter)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            result = compute_stage2_loss(encoder, expert_batch, query_batch, expert_loss, device,
                                        cfg.get("set_weight", 1.0), cfg.get("set_tau", 0.128))
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        counts_forward = {k: v.clone() for k, v in encoder.named_buffers() if k in count_names}
        scaler.scale(result["loss"]).backward()
        scaler.unscale_(optimizer)
        row = {"step": step, "epoch": step / steps_per_epoch,
               **{k: float(v) for k, v in result["metrics"].items()}}
        for group, params in groups.items():
            grads = [p.grad for _, p in params if p.grad is not None]
            if any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError(f"Nonfinite {group} gradient at step {step}")
            row[f"grad/{group}"] = _norm(grads)
        assert all(torch.equal(v, counts_forward[k]) for k, v in encoder.named_buffers() if k in count_names), \
            "Activation recomputation updated BN statistics twice"
        scaler.step(optimizer)
        scaler.update()
        buffers = dict(encoder.named_buffers())
        row["bn_mean_delta_l2"] = _norm([buffers[k].cpu() - initial[k] for k in bn_names])
        row["bn_batches_delta"] = int(buffers[count_names[0]].cpu() - initial[count_names[0]])
        row["seconds"] = time.perf_counter() - tick
        row["query_video_ids"] = ",".join(query_batch["video_id"])
        records.append(row)
        with (output / "metrics.csv").open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if step == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f"step={step} loss={row['loss']:.6f} expert={row['expert_loss']:.6f} "
              f"set={row['set_loss']:.6f} time={row['seconds']:.2f}s", flush=True)
        if step == 1:
            hook.remove()
            assert single_frame_observed["all_equal"] and single_frame_observed["calls"] > 0
        if epoch_mode and step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            if epoch % checkpoint_every_epochs == 0 or step == cfg["num_steps"]:
                last_checkpoint_path = save_encoder(step, epoch)
                last_checkpoint_step = step
                validate(step, epoch, last_checkpoint_path)
    final = encoder.state_dict()
    updates = {group: _norm([p.detach().cpu() - initial[name] for name, p in params])
               for group, params in groups.items()}
    assert updates["backbone_frozen"] == 0
    for group in ("backbone_bn", "embedder") + (("memory", "memory_projection") if encoder.memory else ()):
        assert updates[group] > 0, f"No update in {group}"
    assert all(torch.isfinite(v).all() for v in final.values())
    if last_checkpoint_step != len(records):
        last_checkpoint_path = save_encoder(len(records), len(records) // steps_per_epoch)
    summary = {"initial_state_matches_source": initial_matches,
               "single_frame_context": encoder.single_frame_context,
               "single_frame_input_check": single_frame_observed,
               "parameter_update_l2": updates,
               "bn_mean_delta_l2": records[-1]["bn_mean_delta_l2"],
               "bn_batches_delta": records[-1]["bn_batches_delta"],
               "bn_recompute_count_unchanged": True,
               "steps": len(records), "seconds": time.perf_counter() - started,
               "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if amp else 0,
               "checkpoint": str(last_checkpoint_path),
               "epochs": len(records) / steps_per_epoch, "steps_per_epoch": steps_per_epoch,
               "best_validation": best_validation,
               "validation_video_ids": validation_ids or [],
               "expert_video_ids": expert_ds.video_ids, "nonexpert_video_ids": query_ds.video_ids}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ConfigV2()._root / "train_stage2.yaml"))
    run(parser.parse_args().config)
