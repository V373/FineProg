"""
Profile key training-time slices for the current V2 train config.

This script mirrors the real training setup from train.py as closely as possible,
then times the segments that matter most for wall-clock speed:

1. One-time backbone cache extraction (for the configured only_bn + cache path)
2. Steady-state cached training step
3. Reference full-encoder training step on raw frames
4. Isolated TemporalTripletLoss forward+backward under different fractions

Run from the mytcc project root:
    conda run -n fineprog python scripts/profile_train_timing.py

Recommended quick run:
    conda run -n fineprog python scripts/profile_train_timing.py --warmup 5 --steps 20
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.optim as optim

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algos.loss.contrastive.loss_temporal_triplet import TemporalTripletLoss
from algos.loss.encoder_loss import build_loss
from dataset_preparation.h5vid_dataset import (
    build_dataloader,
    build_feature_cache_dataloader,
)
from models.encoder import TCCEncoder
from train import extract_backbone_features
from utils.config_v2 import ConfigV2


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(
    device: torch.device,
    fn: Callable[[], Any],
) -> tuple[float, Any]:
    _sync(device)
    t0 = time.perf_counter()
    result = fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0, result


def _stats_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _fmt_ms(ms: float) -> str:
    return f"{ms:8.2f} ms"


def _fmt_pct(value: float) -> str:
    return f"{value:6.1f}%"


def _fmt_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _next_batch(data_iter, dataloader):
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter), data_iter


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cache_total_frames(dataset) -> int:
    return sum(entry["frames"].shape[0] for entry in dataset._frames_cache.values())


def _cache_total_bytes(cache: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in cache.values())


def _print_section_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _print_profile_summary(title: str, times: dict[str, list[float]], batch_size: int) -> None:
    _print_section_header(title)

    if not times.get("step_total_ms"):
        print("No measurements collected.")
        return

    step_stats = _stats_ms(times["step_total_ms"])
    mean_step_ms = step_stats["mean"]
    steps_per_sec = 1000.0 / mean_step_ms if mean_step_ms > 0 else float("nan")
    samples_per_sec = steps_per_sec * batch_size if mean_step_ms > 0 else float("nan")

    print(f"measured steps      : {len(times['step_total_ms'])}")
    print(f"mean step time      : {_fmt_ms(mean_step_ms)}")
    print(f"median step time    : {_fmt_ms(step_stats['median'])}")
    print(f"step-time std       : {_fmt_ms(step_stats['std'])}")
    print(f"throughput          : {steps_per_sec:8.3f} steps/s   {samples_per_sec:8.3f} samples/s")
    print()
    print(f"  {'segment':<24s} {'mean':>12s} {'median':>12s} {'share':>9s}")
    print(f"  {'-' * 24} {'-' * 12} {'-' * 12} {'-' * 9}")

    ordered_keys = [
        "batch_fetch_ms",
        "h2d_ms",
        "backbone_ms",
        "temporal_embedder_ms",
        "loss_ms",
        "zero_grad_ms",
        "backward_ms",
        "optimizer_ms",
        "step_total_ms",
    ]

    for key in ordered_keys:
        if key not in times:
            continue
        stats = _stats_ms(times[key])
        share = (stats["mean"] / mean_step_ms * 100.0) if mean_step_ms > 0 else float("nan")
        print(
            f"  {key:<24s} {_fmt_ms(stats['mean']):>12s} {_fmt_ms(stats['median']):>12s} {_fmt_pct(share):>9s}"
        )


def _profile_cached_steps(
    encoder: TCCEncoder,
    loss_module: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cache_dataloader,
    device: torch.device,
    autocast_device: str,
    use_amp: bool,
    warmup: int,
    steps: int,
) -> dict[str, list[float]]:
    times: dict[str, list[float]] = defaultdict(list)
    data_iter = iter(cache_dataloader)

    total_iters = warmup + steps
    encoder.train()
    encoder.configure_trainability()
    loss_module.train()

    for idx in range(total_iters):
        _sync(device)
        step_t0 = time.perf_counter()

        batch_fetch_ms, batch = _time_call(
            device,
            lambda: _next_batch(data_iter, cache_dataloader),
        )
        (batch, data_iter) = batch

        def _move_batch():
            backbone_feats = batch["backbone_feats"].to(device, non_blocking=True)
            loss_batch = {
                "target_steps": batch["target_steps"].to(device, non_blocking=True),
                "seq_len": batch["seq_len"].to(device, non_blocking=True),
            }
            return backbone_feats, loss_batch

        h2d_ms, moved = _time_call(device, _move_batch)
        backbone_feats, loss_batch = moved

        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            temporal_ms, embeddings = _time_call(
                device,
                lambda: encoder.forward_from_feats(backbone_feats),
            )
            loss_ms, out = _time_call(
                device,
                lambda: loss_module(embeddings, loss_batch),
            )
            loss = out["loss"]

        zero_grad_ms, _ = _time_call(device, optimizer.zero_grad)
        backward_ms, _ = _time_call(device, lambda: scaler.scale(loss).backward())
        optimizer_ms, _ = _time_call(device, lambda: (scaler.step(optimizer), scaler.update()))

        _sync(device)
        step_total_ms = (time.perf_counter() - step_t0) * 1000.0

        if idx >= warmup:
            times["batch_fetch_ms"].append(batch_fetch_ms)
            times["h2d_ms"].append(h2d_ms)
            times["temporal_embedder_ms"].append(temporal_ms)
            times["loss_ms"].append(loss_ms)
            times["zero_grad_ms"].append(zero_grad_ms)
            times["backward_ms"].append(backward_ms)
            times["optimizer_ms"].append(optimizer_ms)
            times["step_total_ms"].append(step_total_ms)

    return times


def _profile_full_reference_steps(
    encoder: TCCEncoder,
    loss_module: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    raw_dataloader,
    device: torch.device,
    autocast_device: str,
    use_amp: bool,
    warmup: int,
    steps: int,
) -> dict[str, list[float]]:
    times: dict[str, list[float]] = defaultdict(list)
    data_iter = iter(raw_dataloader)

    total_iters = warmup + steps
    encoder.train()
    encoder.configure_trainability()
    loss_module.train()

    for idx in range(total_iters):
        _sync(device)
        step_t0 = time.perf_counter()

        batch_fetch_ms, batch = _time_call(
            device,
            lambda: _next_batch(data_iter, raw_dataloader),
        )
        (batch, data_iter) = batch

        def _move_batch():
            frames = batch["frames"].to(device, non_blocking=True)
            loss_batch = {
                "target_steps": batch["target_steps"].to(device, non_blocking=True),
                "seq_len": batch["seq_len"].to(device, non_blocking=True),
            }
            return frames, loss_batch

        h2d_ms, moved = _time_call(device, _move_batch)
        frames, loss_batch = moved

        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            def _run_backbone():
                batch_size, clip_len, context_size, channels, height, width = frames.shape
                frames_flat = frames.reshape(batch_size * clip_len * context_size, channels, height, width)
                backbone_feats_flat = encoder.backbone(frames_flat)
                _, num_channels, feat_h, feat_w = backbone_feats_flat.shape
                return backbone_feats_flat.reshape(
                    batch_size,
                    clip_len,
                    context_size,
                    num_channels,
                    feat_h,
                    feat_w,
                )

            backbone_ms, grouped_backbone_feats = _time_call(device, _run_backbone)
            temporal_ms, embeddings = _time_call(
                device,
                lambda: encoder.temporal_embedder(grouped_backbone_feats),
            )
            loss_ms, out = _time_call(
                device,
                lambda: loss_module(embeddings, loss_batch),
            )
            loss = out["loss"]

        zero_grad_ms, _ = _time_call(device, optimizer.zero_grad)
        backward_ms, _ = _time_call(device, lambda: scaler.scale(loss).backward())
        optimizer_ms, _ = _time_call(device, lambda: (scaler.step(optimizer), scaler.update()))

        _sync(device)
        step_total_ms = (time.perf_counter() - step_t0) * 1000.0

        if idx >= warmup:
            times["batch_fetch_ms"].append(batch_fetch_ms)
            times["h2d_ms"].append(h2d_ms)
            times["backbone_ms"].append(backbone_ms)
            times["temporal_embedder_ms"].append(temporal_ms)
            times["loss_ms"].append(loss_ms)
            times["zero_grad_ms"].append(zero_grad_ms)
            times["backward_ms"].append(backward_ms)
            times["optimizer_ms"].append(optimizer_ms)
            times["step_total_ms"].append(step_total_ms)

    return times


def _profile_triplet_fraction_sweep(
    encoder: TCCEncoder,
    cache_dataloader,
    cfg: dict[str, Any],
    loss_yaml: str,
    device: torch.device,
    autocast_device: str,
    use_amp: bool,
    fractions: list[float],
    reps: int,
    warmup: int,
) -> dict[float, dict[str, float]]:
    if cfg.get("loss_name") != "temporal_triplet":
        return {}

    batch = next(iter(cache_dataloader))
    backbone_feats = batch["backbone_feats"].to(device, non_blocking=True)
    loss_batch = {
        "target_steps": batch["target_steps"].to(device, non_blocking=True),
        "seq_len": batch["seq_len"].to(device, non_blocking=True),
    }

    encoder.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
        base_embeddings = encoder.forward_from_feats(backbone_feats).detach()
    encoder.train()

    results: dict[float, dict[str, float]] = {}

    for fraction in fractions:
        loss_module = TemporalTripletLoss(
            config_path=loss_yaml,
            loss_cfg={"num_triplets_fraction": fraction},
        ).to(device)
        loss_module.train()

        samples: list[float] = []

        total_iters = warmup + reps
        for idx in range(total_iters):
            def _run_once():
                emb = base_embeddings.detach().requires_grad_(True)
                with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                    out = loss_module(emb, loss_batch)
                    loss = out["loss"]
                loss.backward()
                metrics = out["metrics"]
                return metrics["num_sampled_triplets"], metrics["num_valid_triplets"]

            elapsed_ms, (num_sampled, num_valid) = _time_call(device, _run_once)
            if idx >= warmup:
                samples.append(elapsed_ms)

        stats = _stats_ms(samples)
        results[fraction] = {
            "mean_ms": stats["mean"],
            "median_ms": stats["median"],
            "num_sampled_triplets": int(num_sampled),
            "num_valid_triplets": int(num_valid),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile key training-time slices for mytcc train.py")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup steps per benchmark block")
    parser.add_argument("--steps", type=int, default=20, help="Number of measured steps per benchmark block")
    parser.add_argument(
        "--full-reference-steps",
        type=int,
        default=10,
        help="Measured raw-frame reference steps (uses the non-cache path on the same config)",
    )
    parser.add_argument(
        "--full-reference-warmup",
        type=int,
        default=3,
        help="Warmup steps for the raw-frame reference block",
    )
    parser.add_argument(
        "--loss-reps",
        type=int,
        default=25,
        help="Measured repetitions for the isolated triplet-loss fraction sweep",
    )
    parser.add_argument(
        "--loss-warmup",
        type=int,
        default=5,
        help="Warmup repetitions for the isolated triplet-loss fraction sweep",
    )
    args = parser.parse_args()

    cfg_v2 = ConfigV2()
    train_cfg = cfg_v2.load_train()
    train_yaml = str(cfg_v2._root / "train.yaml")
    loss_yaml = str(cfg_v2._root / train_cfg.get("loss_config", "loss/loss_tcc.yaml"))

    seed = int(train_cfg.get("seed", 42))
    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    autocast_device = "cuda" if use_amp else "cpu"

    print("=" * 88)
    print("mytcc Training Timing Profile")
    print("=" * 88)
    print(f"device              : {device}")
    print(f"seed                : {seed}")
    print(f"dataset             : {train_cfg['train_dataset']}")
    print(f"h5_path             : {train_cfg['h5_path']}")
    print(f"loss_name           : {train_cfg['loss_name']}")
    print(f"loss_config         : {loss_yaml}")
    print(f"train_base          : {train_cfg['train_base']}")
    print(f"extract_backbone_cache: {train_cfg.get('extract_backbone_cache', False)}")
    print(f"batch_size          : {train_cfg['batch_size']}")
    print(f"num_workers         : {train_cfg['num_workers']}")
    print(f"clip_len            : {train_cfg['clip_len']}")
    print(f"context_size        : {train_cfg['context_size']}")
    print(f"context_stride      : {train_cfg['context_stride']}")
    print(f"warmup / steps      : {args.warmup} / {args.steps}")

    raw_dataloader = build_dataloader(
        config_path=train_yaml,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=True,
        split="train",
        h5_path_override=train_cfg["h5_path"],
    )

    encoder = TCCEncoder(
        config_path=train_yaml,
        train_config_path=train_yaml,
    ).to(device)
    encoder.train()
    encoder.configure_trainability()

    loss_module = build_loss(train_cfg["loss_name"], config_path=loss_yaml).to(device)
    loss_module.train()

    trainable_params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=train_cfg.get("learning_rate", 1e-4))
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    dataset = raw_dataloader.dataset
    total_frames = _cache_total_frames(dataset)
    print(f"dataset videos       : {len(dataset)}")
    print(f"dataset total frames : {total_frames}")
    print(f"steps per epoch      : {len(raw_dataloader)}")

    cache_extract_ms, feat_cache = _time_call(
        device,
        lambda: extract_backbone_features(encoder, dataset, device),
    )
    cache_extract_s = cache_extract_ms / 1000.0
    cache_frames_per_s = total_frames / cache_extract_s if cache_extract_s > 0 else float("nan")
    cache_bytes = _cache_total_bytes(feat_cache)
    cache_dataloader = build_feature_cache_dataloader(
        feat_cache,
        dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    _print_section_header("One-time Backbone Cache Extraction")
    print(f"total extraction time: {cache_extract_s:8.2f} s")
    print(f"frames per second    : {cache_frames_per_s:8.2f} frames/s")
    print(f"seconds per video    : {cache_extract_s / len(dataset):8.3f} s/video")
    print(f"cache resident size  : {_fmt_gib(cache_bytes)}")

    cached_times = _profile_cached_steps(
        encoder=encoder,
        loss_module=loss_module,
        optimizer=optimizer,
        scaler=scaler,
        cache_dataloader=cache_dataloader,
        device=device,
        autocast_device=autocast_device,
        use_amp=use_amp,
        warmup=args.warmup,
        steps=args.steps,
    )
    _print_profile_summary(
        "Configured Steady-State Training Step  (cache path)",
        cached_times,
        batch_size=train_cfg["batch_size"],
    )

    full_reference_times = _profile_full_reference_steps(
        encoder=encoder,
        loss_module=loss_module,
        optimizer=optimizer,
        scaler=scaler,
        raw_dataloader=raw_dataloader,
        device=device,
        autocast_device=autocast_device,
        use_amp=use_amp,
        warmup=args.full_reference_warmup,
        steps=args.full_reference_steps,
    )
    _print_profile_summary(
        "Reference Training Step  (raw frames, no cache)",
        full_reference_times,
        batch_size=train_cfg["batch_size"],
    )

    triplet_fraction_results = _profile_triplet_fraction_sweep(
        encoder=encoder,
        cache_dataloader=cache_dataloader,
        cfg=train_cfg,
        loss_yaml=loss_yaml,
        device=device,
        autocast_device=autocast_device,
        use_amp=use_amp,
        fractions=[1.0, 0.5, 0.1],
        reps=args.loss_reps,
        warmup=args.loss_warmup,
    )

    if triplet_fraction_results:
        _print_section_header("Isolated TemporalTripletLoss  (real embeddings from cached batch)")
        print(f"  {'fraction':<10s} {'mean':>12s} {'median':>12s} {'sampled':>10s} {'valid':>10s}")
        print(f"  {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10}")
        baseline = triplet_fraction_results[1.0]["mean_ms"]
        for fraction in [1.0, 0.5, 0.1]:
            result = triplet_fraction_results[fraction]
            speedup = baseline / result["mean_ms"] if result["mean_ms"] > 0 else float("nan")
            print(
                f"  {fraction:<10.1f} {_fmt_ms(result['mean_ms']):>12s} {_fmt_ms(result['median_ms']):>12s} "
                f"{result['num_sampled_triplets']:>10d} {result['num_valid_triplets']:>10d}"
                f"   speedup vs 1.0: {speedup:5.2f}x"
            )


if __name__ == "__main__":
    main()