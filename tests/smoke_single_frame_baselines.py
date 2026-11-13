#!/usr/bin/env python3
"""Manual CUDA smoke: two real optimizer steps for each single-frame baseline.

Run from any directory with the fineprog environment. This uses the real 36-video
pick-mango training H5, the existing composite loss, and no trainer or W&B session.
The joint stage always tests batch 12 of the twelve longest complete episodes.
Artifacts are smoke checkpoints only, never production training checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algos.loss.encoder_loss import build_loss
from dataset_preparation.h5vid_dataset import H5VideoDataset
from utils.config_v2 import ConfigV2
from utils.encoder_checkpoint import (
    build_encoder_from_checkpoint, build_training_encoder, checkpoint_dict, encoder_config,
)


def _clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _local_contexts(frames, steps):
    indices = torch.stack(((steps - 20).clamp_min(0), steps), dim=1)
    return frames[indices]


@torch.no_grad()
def _probe(encoder, frames):
    encoder.eval()
    if encoder.memory is None:
        contexts = _local_contexts(frames, torch.arange(20))
        return encoder(contexts.unsqueeze(0).cuda().float().div_(255)).cpu()
    raw = encoder.encode_local_episode(frames[:7])
    return encoder.forward_episode([raw], [7]).cpu()


def _check_gradients(encoder):
    norms = {}
    for name in ("backbone", "temporal_embedder", "memory"):
        module = getattr(encoder, name)
        if module is None:
            continue
        total = torch.zeros((), device="cuda")
        trainable_count = 0
        for parameter in module.parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, f"{name}: missing gradient"
                assert torch.isfinite(parameter.grad).all(), f"{name}: nonfinite gradient"
                total += parameter.grad.detach().abs().sum()
                trainable_count += parameter.numel()
            else:
                assert parameter.grad is None, f"{name}: frozen parameter has gradient"
        norms[name] = float(total)
        if trainable_count:
            assert norms[name] > 0, f"{name}: all gradients are zero"
    return norms


def _check_updates(encoder, initial, bn_initial, expected_bn_updates):
    changed = {"backbone": False, "temporal_embedder": False, "memory": False}
    for name, parameter in encoder.named_parameters():
        value = parameter.detach().cpu()
        assert torch.isfinite(value).all(), f"{name}: nonfinite parameter"
        equal = torch.equal(initial[name], value)
        if not parameter.requires_grad:
            assert equal, f"{name}: frozen parameter changed"
        elif not equal:
            changed[name.split(".", 1)[0]] = True
    for name, module in encoder.backbone.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            before = bn_initial[name]
            actual = int(module.num_batches_tracked) - before["count"]
            assert actual == expected_bn_updates, (name, actual, expected_bn_updates)
            if expected_bn_updates == 0:
                assert torch.equal(module.running_mean.cpu(), before["mean"])
                assert torch.equal(module.running_var.cpu(), before["var"])
    expected = ({"memory"} if encoder._train_base == "frozen" else
                {"backbone", "temporal_embedder"} |
                ({"memory"} if encoder.memory is not None else set()))
    assert all(changed[name] for name in expected), changed
    return changed


def _run_stage(name, config, dataset, selected, loss_module, output_dir):
    print(f"\n[smoke] {name}: starting", flush=True)
    _clear_cuda()
    torch.manual_seed(42)
    np.random.seed(42)
    encoder, effective = build_training_encoder(config, "cuda")
    encoder.train()
    encoder.configure_trainability()
    assert encoder.single_frame_context
    if encoder.memory is not None:
        assert not encoder.memory.use_residual and encoder.memory.history_frames == 1
        assert encoder.memory.normalize_input
        source = torch.load(config["memory"]["local_checkpoint_path"],
                            map_location="cpu", weights_only=True)
        for key, value in source["model_state_dict"].items():
            assert torch.equal(encoder.state_dict()[key].cpu(), value), key
        del source
    initial = {n: p.detach().cpu().clone() for n, p in encoder.named_parameters()}
    bn_initial = {
        n: {"count": int(m.num_batches_tracked), "mean": m.running_mean.cpu().clone(),
            "var": m.running_var.cpu().clone()}
        for n, m in encoder.backbone.named_modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    }
    optimizer = torch.optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=4e-4)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    frames = [torch.from_numpy(dataset._frames_cache[dataset.video_ids[i]]["frames"])
              .permute(0, 3, 1, 2) for i in selected]
    lengths = [len(x) for x in frames]
    frozen = encoder.memory is not None and encoder._train_base == "frozen"
    cache = None
    if frozen:
        encoder.eval()
        with torch.no_grad():
            cache = [encoder.encode_local_episode(x).float().cpu() for x in frames]
        encoder.train()
        assert all(not x.requires_grad for x in cache)
    step_results, skipped_attempts, attempts = [], [], []
    for attempt in range(1, 13):
        steps = torch.stack([torch.from_numpy(dataset._sample_offset_uniform(n)).long()
                             for n in lengths])
        batch = {"target_steps": steps.cuda(), "seq_len": torch.tensor(lengths, device="cuda")}
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        with torch.autocast("cuda"):
            if encoder.memory is None:
                contexts = torch.stack([_local_contexts(x, s) for x, s in zip(frames, steps)])
                embeddings = encoder(contexts.cuda().float().div_(255))
            else:
                raw = ([x.cuda() for x in cache] if frozen else
                       [encoder.encode_local_episode(x) for x in frames])
                assert all(x.requires_grad != frozen for x in raw)
                if not frozen:
                    raw[0].retain_grad()
                embeddings = encoder.forward_episode(raw, lengths, steps)
            loss = loss_module(embeddings, batch)["loss"]
        assert torch.isfinite(loss), f"{name}: nonfinite loss"
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nonfinite = sorted({n.split(".", 1)[0] for n, p in encoder.named_parameters()
                            if p.grad is not None and not torch.isfinite(p.grad).all()})
        expected_bn = (0 if frozen else attempt *
                       (1 if encoder.memory is None else sum(math.ceil(n / 20) for n in lengths)))
        item = {"attempt": attempt, "loss": float(loss.detach()), "scale_before": scale_before,
                "bn_updates_per_layer": expected_bn}
        if nonfinite:
            # Follow the trainer's ordinary AMP behavior: GradScaler skips this
            # optimizer update and backs off. Forward BN updates still count.
            before_skip = {n: p.detach().cpu().clone() for n, p in encoder.named_parameters()}
            scaler.step(optimizer)
            scaler.update()
            assert scaler.get_scale() < scale_before, "GradScaler did not back off"
            for parameter_name, parameter in encoder.named_parameters():
                assert torch.equal(parameter.detach().cpu(), before_skip[parameter_name]), parameter_name
            del before_skip
            item.update(skipped=True, nonfinite_modules=nonfinite, weights_unchanged_on_skip=True)
            skipped_attempts.append(item)
        else:
            gradient_l1 = _check_gradients(encoder)
            early_history_gradient = None
            if encoder.memory is not None and not frozen:
                assert 0 not in steps[0].tolist()  # offset_uniform excludes frame zero.
                history_grad = raw[0].grad[0] / scale_before
                assert torch.isfinite(history_grad).all() and history_grad.abs().sum() > 0
                early_history_gradient = float(history_grad.abs().sum())
            scaler.step(optimizer)
            scaler.update()
            changed = _check_updates(encoder, initial, bn_initial, expected_bn)
            item.update(step=len(step_results) + 1, skipped=False, gradient_l1=gradient_l1,
                        modules_changed=changed, unsampled_frame0_gradient_l1=early_history_gradient)
            step_results.append(item)
        torch.cuda.synchronize()
        item.update(seconds=time.perf_counter() - step_start, scale_after=scaler.get_scale())
        attempts.append(item)
        (output_dir / f"{name}_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
        print(f"[smoke] {name}: {json.dumps(item)}", flush=True)
        del embeddings, loss, batch
        if encoder.memory is None:
            del contexts
        else:
            del raw
        if len(step_results) == 2:
            break
    assert len(step_results) == 2, f"{name}: fewer than two finite updates within 12 AMP attempts"
    torch.cuda.synchronize()
    result = {
        "status": "passed", "batch_size": len(selected), "lengths": lengths,
        "seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "encoder_config": encoder_config(encoder), "effective_config": effective,
        "steps": step_results, "skipped_attempts": skipped_attempts,
        "attempt_count": len(attempts), "successful_optimizer_updates": len(step_results),
        "raw_embedding_cache": frozen,
    }
    path = output_dir / f"smoke_{name}.pt"
    torch.save(checkpoint_dict(encoder), path)
    expected = _probe(encoder, frames[0])
    del optimizer, scaler, initial, bn_initial, cache
    encoder.cpu()
    del encoder
    _clear_cuda()
    restored = build_encoder_from_checkpoint(path, "cuda").eval()
    assert restored.single_frame_context
    torch.testing.assert_close(_probe(restored, frames[0]), expected, rtol=2e-4, atol=2e-5)
    result.update(checkpoint_path=str(path), checkpoint_roundtrip=True)
    del restored
    _clear_cuda()
    print(f"[smoke] {name}: passed; peak allocated {result['peak_allocated_mib']:.0f} MiB", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Smoke results/checkpoints (default: unique /tmp directory)")
    parser.add_argument("--local-batch-size", type=int, default=2,
                        help="Local-only stage batch; both Mamba stages always use longest 12 videos")
    args = parser.parse_args()
    if not 2 <= args.local_batch_size <= 12:
        parser.error("--local-batch-size must be between 2 and 12")
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              Path(tempfile.mkdtemp(prefix="fineprog_single_frame_smoke_")))
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() or any(output.glob("smoke_*.pt")):
        parser.error("output directory already contains smoke results; choose a fresh directory")
    results = {"status": "running", "output_dir": str(output), "stages": {}}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("This smoke requires CUDA and installed official mamba-ssm kernels")
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True
        cfg = ConfigV2(ROOT / "configs_v2")
        h5_path = cfg.resolve_dataset("pick_mango_vid_36vid_train")["processed_h5_path"]
        dataset = H5VideoDataset(h5_path, clip_len=20, context_size=2, context_stride=20,
                                 random_offset=1, full_episode=True, transport_frames_as_uint8=True)
        selected = sorted(range(len(dataset)), key=lambda i: (
            -len(dataset._frames_cache[dataset.video_ids[i]]["frames"]), dataset.video_ids[i]))[:12]
        assert len(dataset) == 36 and len(selected) == 12
        weights_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(ResNet50_Weights.IMAGENET1K_V1.url).name
        pretrained = weights_file.is_file()  # Never initiate a network download for this test.
        base = {"pretrained": pretrained, "clip_len": 20, "context_size": 2, "context_stride": 20,
                "embedding_dim": 128, "embedding_normalization": "l2", "single_frame_context": True,
                "train_base": "only_bn", "train_embedding": True, "memory": {"enabled": False}}
        loss_module = build_loss("composite", config_path=str(
            ROOT / "configs_v2/loss/loss_composite_tcc_triplet.yaml")).cuda().train()
        gpu = torch.cuda.get_device_properties(0)
        results.update(gpu=gpu.name, gpu_total_mib=gpu.total_memory / 2**20,
                       dataset=h5_path, video_ids=[dataset.video_ids[i] for i in selected],
                       pretrained=pretrained, pretrained_cache_path=str(weights_file),
                       local_batch_size=args.local_batch_size, joint_batch_size=12)
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        results["stages"]["local"] = _run_stage(
            "local", base, dataset, selected[:args.local_batch_size], loss_module, output)
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        local_path = results["stages"]["local"]["checkpoint_path"]
        for name, frozen in (("frozen_hf1", True), ("trainable_hf1", False)):
            memory = {"enabled": True, "local_checkpoint_path": local_path, "dual_branch": False,
                      "history_frames": 1, "history_stride": 1, "use_residual": False,
                      "normalize_input": True, "cache_local_embeddings": frozen,
                      "visual_chunk_size": 20, "checkpoint_visual": True, "freeze_bn_stats": frozen}
            config = {**base, "pretrained": False, "train_base": "frozen" if frozen else "only_bn",
                      "train_embedding": not frozen, "memory": memory}
            results["stages"][name] = _run_stage(name, config, dataset, selected, loss_module, output)
            result_path.write_text(json.dumps(results, indent=2) + "\n")
        sources = [results["stages"][n]["encoder_config"]["memory_config"]["local_checkpoint_sha256"]
                   for n in ("frozen_hf1", "trainable_hf1")]
        assert sources[0] == sources[1]
        results.update(status="passed", shared_local_checkpoint_sha256=sources[0])
    except Exception as exc:
        results.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"[smoke] results: {result_path}", flush=True)


if __name__ == "__main__":
    main()
