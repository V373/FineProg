"""
Benchmark the current V2 cache-path training recipe.

Focused tests only:
1. num_workers sweep with OMP / MKL / OpenBLAS threads limited to 1.
2. torch.compile(mode="reduce-overhead") on the current cache path.

Run from the mytcc project root, preferably via:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 conda run -n fineprog \
    python scripts/benchmark_cache_recipe.py --warmup 10 --steps 40

This script mirrors the current train.yaml cache path as closely as possible,
but avoids wandb / checkpointing and uses short benchmark windows.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algos.loss.encoder_loss import build_loss
from dataset_preparation.h5vid_dataset import H5VideoDataset, build_feature_cache_dataloader
from models.encoder import TCCEncoder
from train import extract_backbone_features
from utils.config_v2 import ConfigV2


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(device: torch.device, fn):
    _sync(device)
    t0 = time.perf_counter()
    result = fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0, result


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


def _maybe_set_torch_threads(limit_threads: bool) -> None:
    if not limit_threads:
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _fmt_ms(value: float) -> str:
    return f"{value:8.2f} ms"


def _fmt_pct(value: float) -> str:
    return f"{value:7.2f}%"


def _fmt_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _cache_total_frames(dataset: H5VideoDataset) -> int:
    return sum(entry["frames"].shape[0] for entry in dataset._frames_cache.values())


def _cache_total_bytes(cache: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in cache.values())


def _shutdown_dataloader(dataloader) -> None:
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


def _print_header(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _build_train_objects(
    train_yaml: str,
    loss_yaml: str,
    train_cfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[TCCEncoder, nn.Module, optim.Optimizer, torch.amp.GradScaler]:
    _set_seed(seed)

    encoder = TCCEncoder(
        config_path=train_yaml,
        train_config_path=train_yaml,
    ).to(device)
    encoder.train()
    encoder.configure_trainability()

    loss_module = build_loss(train_cfg["loss_name"], config_path=loss_yaml).to(device)
    loss_module.train()

    optimizer = optim.Adam(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=train_cfg.get("learning_rate", 1e-4),
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))
    return encoder, loss_module, optimizer, scaler


def _profile_cached_steps_detailed(
    encoder: TCCEncoder,
    loss_module: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cache_dataloader,
    device: torch.device,
    warmup: int,
    steps: int,
) -> dict[str, list[float]]:
    times: dict[str, list[float]] = defaultdict(list)
    data_iter = iter(cache_dataloader)
    use_amp = device.type == "cuda"
    autocast_device = "cuda" if use_amp else "cpu"

    encoder.train()
    encoder.configure_trainability()
    loss_module.train()

    for idx in range(warmup + steps):
        _sync(device)
        step_t0 = time.perf_counter()

        batch_fetch_ms, batch = _time_call(
            device,
            lambda: _next_batch(data_iter, cache_dataloader),
        )
        batch, data_iter = batch

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


class CachePathStep(nn.Module):
    def __init__(self, encoder: TCCEncoder, loss_module: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.loss_module = loss_module

    def forward(
        self,
        backbone_feats: torch.Tensor,
        target_steps: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.encoder.forward_from_feats(backbone_feats)
        out = self.loss_module(
            embeddings,
            {
                "target_steps": target_steps,
                "seq_len": seq_len,
            },
        )
        return out["loss"]


def _profile_cached_steps_combined(
    step_module: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cache_dataloader,
    device: torch.device,
    warmup: int,
    steps: int,
    prime_once: bool,
) -> tuple[dict[str, list[float]], float | None]:
    times: dict[str, list[float]] = defaultdict(list)
    data_iter = iter(cache_dataloader)
    use_amp = device.type == "cuda"
    autocast_device = "cuda" if use_amp else "cpu"
    prime_ms: float | None = None

    step_module.train()

    if prime_once:
        batch, data_iter = _next_batch(data_iter, cache_dataloader)
        backbone_feats = batch["backbone_feats"].to(device, non_blocking=True)
        target_steps = batch["target_steps"].to(device, non_blocking=True)
        seq_len = batch["seq_len"].to(device, non_blocking=True)

        def _prime_step():
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                loss = step_module(backbone_feats, target_steps, seq_len)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        prime_ms, _ = _time_call(device, _prime_step)

    for idx in range(warmup + steps):
        _sync(device)
        step_t0 = time.perf_counter()

        batch_fetch_ms, batch = _time_call(
            device,
            lambda: _next_batch(data_iter, cache_dataloader),
        )
        batch, data_iter = batch

        def _move_batch():
            backbone_feats = batch["backbone_feats"].to(device, non_blocking=True)
            target_steps = batch["target_steps"].to(device, non_blocking=True)
            seq_len = batch["seq_len"].to(device, non_blocking=True)
            return backbone_feats, target_steps, seq_len

        h2d_ms, moved = _time_call(device, _move_batch)
        backbone_feats, target_steps, seq_len = moved

        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            forward_loss_ms, loss = _time_call(
                device,
                lambda: step_module(backbone_feats, target_steps, seq_len),
            )

        zero_grad_ms, _ = _time_call(device, optimizer.zero_grad)
        backward_ms, _ = _time_call(device, lambda: scaler.scale(loss).backward())
        optimizer_ms, _ = _time_call(device, lambda: (scaler.step(optimizer), scaler.update()))

        _sync(device)
        step_total_ms = (time.perf_counter() - step_t0) * 1000.0

        if idx >= warmup:
            times["batch_fetch_ms"].append(batch_fetch_ms)
            times["h2d_ms"].append(h2d_ms)
            times["forward_loss_ms"].append(forward_loss_ms)
            times["zero_grad_ms"].append(zero_grad_ms)
            times["backward_ms"].append(backward_ms)
            times["optimizer_ms"].append(optimizer_ms)
            times["step_total_ms"].append(step_total_ms)

    return times, prime_ms


def _print_worker_sweep_table(results: dict[int, dict[str, dict[str, float]]], current_workers: int) -> None:
    _print_header("num_workers Sweep  (cache path, eager)")
    print(
        f"{'workers':>7s} {'step mean':>12s} {'steps/s':>10s} {'batch':>10s} {'h2d':>10s} "
        f"{'embed':>10s} {'loss':>10s} {'bwd':>10s} {'opt':>10s} {'vs nw=4':>10s}"
    )
    print("-" * 96)

    baseline_ms = results[current_workers]["step_total_ms"]["mean"]
    best_workers = None
    best_ms = None
    for workers in sorted(results):
        row = results[workers]
        mean_step_ms = row["step_total_ms"]["mean"]
        steps_per_sec = 1000.0 / mean_step_ms if mean_step_ms > 0 else float("nan")
        improvement = (baseline_ms - mean_step_ms) / baseline_ms * 100.0
        print(
            f"{workers:7d} "
            f"{mean_step_ms:10.2f}ms "
            f"{steps_per_sec:10.3f} "
            f"{row['batch_fetch_ms']['mean']:10.2f} "
            f"{row['h2d_ms']['mean']:10.2f} "
            f"{row['temporal_embedder_ms']['mean']:10.2f} "
            f"{row['loss_ms']['mean']:10.2f} "
            f"{row['backward_ms']['mean']:10.2f} "
            f"{row['optimizer_ms']['mean']:10.2f} "
            f"{improvement:9.2f}%"
        )
        if best_ms is None or mean_step_ms < best_ms:
            best_ms = mean_step_ms
            best_workers = workers

    assert best_workers is not None and best_ms is not None
    best_improvement = (baseline_ms - best_ms) / baseline_ms * 100.0
    print()
    print(f"current train.yaml num_workers : {current_workers}")
    print(f"best measured num_workers      : {best_workers}")
    print(f"best step time improvement     : {_fmt_pct(best_improvement)}")


def _print_compile_table(
    eager_stats: dict[str, dict[str, float]],
    compiled_stats: dict[str, dict[str, float]],
    compile_prime_ms: float | None,
) -> None:
    _print_header('torch.compile Comparison  (cache path, mode="reduce-overhead")')
    print(f"{'segment':<18s} {'eager mean':>12s} {'compiled mean':>15s} {'improvement':>12s}")
    print("-" * 64)

    for key in [
        "batch_fetch_ms",
        "h2d_ms",
        "forward_loss_ms",
        "zero_grad_ms",
        "backward_ms",
        "optimizer_ms",
        "step_total_ms",
    ]:
        eager_mean = eager_stats[key]["mean"]
        compiled_mean = compiled_stats[key]["mean"]
        improvement = (eager_mean - compiled_mean) / eager_mean * 100.0
        print(
            f"{key:<18s} {eager_mean:10.2f}ms {compiled_mean:13.2f}ms {improvement:11.2f}%"
        )

    if compile_prime_ms is not None:
        print()
        print(f"compiled first-step / graph-build cost : {_fmt_ms(compile_prime_ms)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the current mytcc cache-path training recipe")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps per benchmark block")
    parser.add_argument("--steps", type=int, default=40, help="Measured steps per benchmark block")
    parser.add_argument(
        "--worker-sweep",
        type=str,
        default="0,1,2,4",
        help="Comma-separated num_workers sweep",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help='torch.compile mode for the compile comparison (default: "reduce-overhead")',
    )
    parser.add_argument(
        "--limit-torch-threads",
        action="store_true",
        default=False,
        help="Also force torch intra-op and inter-op threads to 1.",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        default=False,
        help="Skip the torch.compile comparison block.",
    )
    args = parser.parse_args()

    _maybe_set_torch_threads(limit_threads=args.limit_torch_threads)

    cfg_v2 = ConfigV2()
    train_cfg = cfg_v2.load_train()
    train_yaml = str(cfg_v2._root / "train.yaml")
    loss_yaml = str(cfg_v2._root / train_cfg.get("loss_config", "loss/loss_tcc.yaml"))
    seed = int(train_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worker_values = [int(item) for item in args.worker_sweep.split(",") if item.strip()]
    current_workers = int(train_cfg["num_workers"])

    _print_header("Benchmark Configuration")
    print(f"device                    : {device}")
    print(f"dataset                   : {train_cfg['train_dataset']}")
    print(f"h5_path                   : {train_cfg['h5_path']}")
    print(f"loss_name                 : {train_cfg['loss_name']}")
    print(f"loss_config               : {loss_yaml}")
    print(f"train_base                : {train_cfg['train_base']}")
    print(f"extract_backbone_cache    : {train_cfg.get('extract_backbone_cache', False)}")
    print(f"batch_size                : {train_cfg['batch_size']}")
    print(f"current num_workers       : {current_workers}")
    print(f"worker sweep              : {worker_values}")
    print(f"warmup / measured steps   : {args.warmup} / {args.steps}")
    print(f"OMP_NUM_THREADS           : {os.environ.get('OMP_NUM_THREADS', '<unset>')}")
    print(f"MKL_NUM_THREADS           : {os.environ.get('MKL_NUM_THREADS', '<unset>')}")
    print(f"OPENBLAS_NUM_THREADS      : {os.environ.get('OPENBLAS_NUM_THREADS', '<unset>')}")
    print(f"NUMEXPR_NUM_THREADS       : {os.environ.get('NUMEXPR_NUM_THREADS', '<unset>')}")
    print(f"limit torch threads       : {args.limit_torch_threads}")
    print(f"torch.get_num_threads()   : {torch.get_num_threads()}")

    dataset = H5VideoDataset(
        h5_path=train_cfg["h5_path"],
        config_path=train_yaml,
    )
    total_frames = _cache_total_frames(dataset)
    print(f"dataset videos            : {len(dataset)}")
    print(f"dataset total frames      : {total_frames}")
    print(f"estimated steps/epoch     : {len(dataset) // train_cfg['batch_size']}")

    extract_encoder, _, _, _ = _build_train_objects(
        train_yaml=train_yaml,
        loss_yaml=loss_yaml,
        train_cfg=train_cfg,
        device=device,
        seed=seed,
    )
    cache_extract_ms, feat_cache = _time_call(
        device,
        lambda: extract_backbone_features(extract_encoder, dataset, device),
    )
    cache_extract_s = cache_extract_ms / 1000.0
    cache_bytes = _cache_total_bytes(feat_cache)
    print(f"feature-cache extraction  : {cache_extract_s:8.2f} s")
    print(f"feature-cache size        : {_fmt_gib(cache_bytes)}")
    print(f"feature-cache throughput  : {total_frames / cache_extract_s:8.2f} frames/s")
    del extract_encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    worker_results: dict[int, dict[str, dict[str, float]]] = {}
    for workers in worker_values:
        cache_dataloader = build_feature_cache_dataloader(
            feat_cache,
            dataset,
            batch_size=train_cfg["batch_size"],
            num_workers=workers,
        )
        encoder, loss_module, optimizer, scaler = _build_train_objects(
            train_yaml=train_yaml,
            loss_yaml=loss_yaml,
            train_cfg=train_cfg,
            device=device,
            seed=seed,
        )

        detailed_times = _profile_cached_steps_detailed(
            encoder=encoder,
            loss_module=loss_module,
            optimizer=optimizer,
            scaler=scaler,
            cache_dataloader=cache_dataloader,
            device=device,
            warmup=args.warmup,
            steps=args.steps,
        )
        worker_results[workers] = {key: _stats(values) for key, values in detailed_times.items()}

        _shutdown_dataloader(cache_dataloader)
        del cache_dataloader, encoder, loss_module, optimizer, scaler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _print_worker_sweep_table(worker_results, current_workers=current_workers)

    if args.skip_compile:
        _print_header("torch.compile Comparison")
        print("Skipped by --skip-compile")
        return

    compile_workers = current_workers
    compile_dataloader = build_feature_cache_dataloader(
        feat_cache,
        dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=compile_workers,
    )

    eager_encoder, eager_loss, eager_optimizer, eager_scaler = _build_train_objects(
        train_yaml=train_yaml,
        loss_yaml=loss_yaml,
        train_cfg=train_cfg,
        device=device,
        seed=seed,
    )
    eager_step = CachePathStep(eager_encoder, eager_loss).to(device)
    eager_times, _ = _profile_cached_steps_combined(
        step_module=eager_step,
        optimizer=eager_optimizer,
        scaler=eager_scaler,
        cache_dataloader=compile_dataloader,
        device=device,
        warmup=args.warmup,
        steps=args.steps,
        prime_once=False,
    )
    eager_stats = {key: _stats(values) for key, values in eager_times.items()}

    del eager_step, eager_encoder, eager_loss, eager_optimizer, eager_scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    compiled_encoder, compiled_loss, compiled_optimizer, compiled_scaler = _build_train_objects(
        train_yaml=train_yaml,
        loss_yaml=loss_yaml,
        train_cfg=train_cfg,
        device=device,
        seed=seed,
    )
    compiled_step = CachePathStep(compiled_encoder, compiled_loss).to(device)
    compiled_step = torch.compile(compiled_step, mode=args.compile_mode)
    compiled_times, compile_prime_ms = _profile_cached_steps_combined(
        step_module=compiled_step,
        optimizer=compiled_optimizer,
        scaler=compiled_scaler,
        cache_dataloader=compile_dataloader,
        device=device,
        warmup=args.warmup,
        steps=args.steps,
        prime_once=True,
    )
    compiled_stats = {key: _stats(values) for key, values in compiled_times.items()}

    _print_compile_table(
        eager_stats=eager_stats,
        compiled_stats=compiled_stats,
        compile_prime_ms=compile_prime_ms,
    )

    _shutdown_dataloader(compile_dataloader)


if __name__ == "__main__":
    main()