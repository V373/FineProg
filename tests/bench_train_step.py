#!/usr/bin/env python3
"""Benchmark one FineProg encoder training step with the active V2 config."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.loss.encoder_loss import build_loss
from algos.loss.tcc.deterministic_alignment import (
    compute_deterministic_alignment_loss,
)
from algos.loss.tcc.loss_tcc import TCCLoss
from dataset_preparation.h5vid_dataset import build_dataloader
from models.encoder import TCCEncoder
from utils.config_v2 import ConfigV2


BOUNDARIES = (
    "start",
    "h2d+cast",
    "backbone",
    "embedder",
    "loss_fwd",
    "backward",
    "optimizer",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uint8", action="store_true")
    parser.add_argument("--vectorized-loss", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--losses-out", type=Path)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_alignment_impl(vectorized: bool):
    if not vectorized:
        return compute_deterministic_alignment_loss
    try:
        from algos.loss.tcc.deterministic_alignment import (
            compute_deterministic_alignment_loss_batched,
        )
    except ImportError as exc:
        raise RuntimeError(
            "--vectorized-loss requires compute_deterministic_alignment_loss_batched"
        ) from exc
    return compute_deterministic_alignment_loss_batched


def _bind_tcc_alignment(loss_module: torch.nn.Module, vectorized: bool) -> None:
    alignment_fn = _select_alignment_impl(vectorized)

    def _compute(self, embeddings, target_steps, seq_len, batch_size, clip_len):
        del batch_size, clip_len
        return alignment_fn(
            embs=embeddings,
            steps=target_steps,
            seq_lens=seq_len,
            similarity_type=self.similarity_type,
            loss_type=self.loss_type,
            temperature=self.softmax_temperature,
            label_smoothing=self.label_smoothing,
            variance_lambda=self.variance_lambda,
            huber_delta=self.huber_delta,
            normalize_indices=self.normalize_indices,
        )

    for module in loss_module.modules():
        if isinstance(module, TCCLoss):
            module._compute_deterministic_loss = types.MethodType(_compute, module)


def _forward_embedder(
    encoder: TCCEncoder,
    grouped_feats: torch.Tensor,
    channels_last: bool,
) -> torch.Tensor:
    embedder = encoder.temporal_embedder
    batch_size, clip_len, context_size, channels, height, width = grouped_feats.shape
    x = grouped_feats.permute(0, 1, 3, 2, 4, 5).reshape(
        batch_size * clip_len,
        channels,
        context_size,
        height,
        width,
    )
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    else:
        x = x.contiguous()
    x = embedder.relu_1(embedder.conv3d_1(x))
    x = embedder.relu_2(embedder.conv3d_2(x))
    x = F.adaptive_max_pool3d(x, output_size=1).reshape(
        batch_size * clip_len, embedder.hidden_channels
    )
    x = embedder.relu_fc1(embedder.fc1(x))
    x = embedder.relu_fc2(embedder.fc2(x))
    embeddings = embedder.proj(x).reshape(batch_size, clip_len, embedder.embed_dim)
    return encoder._apply_embedding_normalization(embeddings)


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
    }


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.warmup < 0 or args.steps < 1:
        raise ValueError("--warmup must be >= 0 and --steps must be >= 1")

    _seed_everything(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    config = ConfigV2()
    train_cfg = config.load_train()
    train_yaml = config._root / "train.yaml"
    loss_yaml = config._root / train_cfg["loss_config"]
    device = torch.device("cuda")

    dataloader = build_dataloader(
        config_path=str(train_yaml),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        shuffle=True,
        split="train",
        h5_path_override=train_cfg["h5_path"],
    )
    # DataLoader workers start on first iteration, so this explicit benchmark
    # override is visible to them even when the YAML contains the opposite value.
    dataloader.dataset.transport_frames_as_uint8 = args.uint8

    encoder = TCCEncoder(
        config_path=str(train_yaml),
        train_config_path=str(train_yaml),
    ).to(device)
    encoder.train()
    encoder.configure_trainability()
    if args.channels_last:
        encoder.backbone.to(memory_format=torch.channels_last)
        encoder.temporal_embedder.to(memory_format=torch.channels_last_3d)
    else:
        encoder.backbone.to(memory_format=torch.contiguous_format)
        encoder.temporal_embedder.to(memory_format=torch.contiguous_format)

    loss_module = build_loss(train_cfg["loss_name"], config_path=str(loss_yaml)).to(device)
    loss_module.train()
    _bind_tcc_alignment(loss_module, args.vectorized_loss)

    optimizer = torch.optim.Adam(
        [parameter for parameter in encoder.parameters() if parameter.requires_grad],
        lr=float(train_cfg["learning_rate"]),
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=True)

    iterator = iter(dataloader)

    def next_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            return next(iterator)

    def run_step(record: bool):
        batch = next_batch()
        events = [torch.cuda.Event(enable_timing=True) for _ in BOUNDARIES] if record else None
        if events:
            events[0].record()

        frames = batch["frames"].to(device, non_blocking=True)
        if args.uint8:
            frames = frames.float().div_(255.0)
        loss_batch = {
            "target_steps": batch["target_steps"].to(device, non_blocking=True),
            "seq_len": batch["seq_len"].to(device, non_blocking=True),
        }
        if events:
            events[1].record()

        with torch.amp.autocast(device_type="cuda", enabled=True):
            batch_size, clip_len, context_size, channels, height, width = frames.shape
            frames_flat = frames.reshape(
                batch_size * clip_len * context_size,
                channels,
                height,
                width,
            )
            if args.channels_last:
                frames_flat = frames_flat.contiguous(memory_format=torch.channels_last)
            backbone_flat = encoder.backbone(frames_flat)
        if events:
            events[2].record()

        with torch.amp.autocast(device_type="cuda", enabled=True):
            _, feature_channels, feature_height, feature_width = backbone_flat.shape
            grouped_feats = backbone_flat.reshape(
                batch_size,
                clip_len,
                context_size,
                feature_channels,
                feature_height,
                feature_width,
            )
            embeddings = _forward_embedder(encoder, grouped_feats, args.channels_last)
        if events:
            events[3].record()

        with torch.amp.autocast(device_type="cuda", enabled=True):
            output = loss_module(embeddings, loss_batch)
            loss = output["loss"]
        if events:
            events[4].record()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if events:
            events[5].record()
        scaler.step(optimizer)
        scaler.update()
        if events:
            events[6].record()
        metrics = output.get("metrics", {})
        tracked = {
            key: metrics[key].detach()
            for key in (
                "component_raw_loss/tcc",
                "component_raw_loss/temporal_triplet",
            )
            if key in metrics
        }
        return events, loss.detach(), tracked

    for _ in range(args.warmup):
        run_step(record=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    recorded = []
    recorded_losses = []
    recorded_metrics: dict[str, list[torch.Tensor]] = {}
    wall_start = time.perf_counter()
    for _ in range(args.steps):
        events, loss, metrics = run_step(record=True)
        recorded.append(events)
        recorded_losses.append(loss)
        for key, value in metrics.items():
            recorded_metrics.setdefault(key, []).append(value)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start

    segment_values: dict[str, list[float]] = {
        name: [] for name in BOUNDARIES[1:]
    }
    total_values: list[float] = []
    for events in recorded:
        for index, name in enumerate(BOUNDARIES[1:], start=1):
            segment_values[name].append(events[index - 1].elapsed_time(events[index]))
        total_values.append(events[0].elapsed_time(events[-1]))

    result = {
        "settings": {
            "config_root": str(config._root),
            "dataset": train_cfg["train_dataset"],
            "batch_size": int(train_cfg["batch_size"]),
            "clip_len": int(train_cfg["clip_len"]),
            "context_size": int(train_cfg["context_size"]),
            "uint8": args.uint8,
            "vectorized_loss": args.vectorized_loss,
            "cudnn_benchmark": args.cudnn_benchmark,
            "channels_last": args.channels_last,
            "warmup": args.warmup,
            "steps": args.steps,
        },
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "segments": {name: _summarize(values) for name, values in segment_values.items()},
        "total": _summarize(total_values),
        "wall_mean_ms": wall_seconds * 1000.0 / args.steps,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024.0**2),
    }
    losses = torch.stack(recorded_losses).float().cpu().tolist()
    metric_series = {
        key: torch.stack(values).float().cpu().tolist()
        for key, values in recorded_metrics.items()
    }
    if args.losses_out is not None:
        args.losses_out.write_text(
            json.dumps(
                {
                    "settings": result["settings"],
                    "losses": losses,
                    "metrics": metric_series,
                }
            ),
            encoding="utf-8",
        )

    print("\nTraining-step benchmark")
    print("segment              mean_ms     p50_ms     p90_ms")
    for name, stats in (*result["segments"].items(), ("total", result["total"])):
        print(
            f"{name:<20} {stats['mean_ms']:>9.3f} "
            f"{stats['p50_ms']:>10.3f} {stats['p90_ms']:>10.3f}"
        )
    print(f"wall mean           {result['wall_mean_ms']:>9.3f} ms/step")
    print(f"peak allocated      {result['peak_memory_mib']:>9.1f} MiB")
    print("BENCH_JSON=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
