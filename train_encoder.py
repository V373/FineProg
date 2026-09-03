"""
Minimal TCC training script (PyTorch).

Pipeline:
    dataset → dataloader → TCCEncoder → encoder_loss → backward → optimizer.step()

Goal (current stage):
    - Confirm loss computes correctly
    - Confirm loss backpropagates
    - Confirm training is stable across multiple steps

Usage:
    python train.py
"""

import os
import sys
import yaml
import argparse
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm


# Resolve project root so imports work regardless of working directory
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset_preparation.h5vid_dataset import build_dataloader, build_feature_cache_dataloader
from models.encoder import TCCEncoder
from algos.loss.encoder_loss import build_loss
# [v2] V2 config resolver (independent of old config system)
from utils.config_v2 import ConfigV2

# ---------------------------------------------------------------------------
# [v2] V2 config constants (resolved once at import time)
# ---------------------------------------------------------------------------
_CFG_V2       = ConfigV2()
_TRAIN_V2     = _CFG_V2.load_train()          # resolved train stage config dict
_V2_TRAIN_YAML = str(_CFG_V2._root / "train.yaml")   # path to configs_v2/train.yaml
_loss_name_v2  = _TRAIN_V2.get("loss_name",   "tcc")
_loss_cfg_file = _TRAIN_V2.get("loss_config", "loss/loss_tcc.yaml")
_V2_LOSS_YAML  = str(_CFG_V2._root / _loss_cfg_file) # path to configs_v2/<loss_config>


def _save_encoder_checkpoint(encoder: TCCEncoder, checkpoint_path: str) -> None:
    """Save encoder weights together with the output-normalization mode."""
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "embedding_normalization": encoder.embedding_normalization,
        },
        checkpoint_path,
    )


def _maybe_limit_cpu_threads(enabled: bool) -> None:
    """Optionally force host-side thread pools to 1.

    This is intended for the current small-batch cache-path training recipe,
    where limiting CPU-side parallelism can reduce contention between the main
    process and DataLoader workers.
    """
    if not enabled:
        return

    thread_limit = "1"
    env_vars = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )

    print("[train] limit_cpu_threads enabled; forcing host thread pools to 1")
    for var in env_vars:
        previous = os.environ.get(var)
        os.environ[var] = thread_limit
        if previous not in (None, thread_limit):
            print(f"[train] {var}: {previous} -> {thread_limit}")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        print(f"[train] torch.set_num_interop_threads(1) skipped: {exc}")

    print(f"[train] torch.get_num_threads(): {torch.get_num_threads()}")


# ---------------------------------------------------------------------------
# Backbone feature extraction for cache mode (only_bn + extract_backbone_cache)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_backbone_features(
    encoder: "TCCEncoder",
    h5_dataset,
    device: torch.device,
) -> dict:
    """
    Run backbone on every frame of every video and return a feature cache.

    The backbone is set to eval mode during extraction so that BatchNorm uses
    its running statistics (not the current batch), which avoids contaminating
    the running stats with extraction-time statistics.
    After extraction the encoder is restored to its original training state.

    Args:
        encoder:    TCCEncoder instance (must already be on device).
        h5_dataset: H5VideoDataset with _frames_cache populated.
        device:     Torch device to run the backbone on.

    Returns:
        cache: Dict[video_id -> Tensor[T, 1024, 14, 14]] stored on CPU in fp16.
    """
    import numpy as np

    # Save training state, switch backbone to eval so BN uses running stats
    was_training = encoder.training
    encoder.eval()

    cache = {}
    print(f"[extract_backbone_features] Extracting features for {len(h5_dataset.video_ids)} videos...")
    for vid in h5_dataset.video_ids:
        frames_np = h5_dataset._frames_cache[vid]["frames"]  # [T, 224, 224, 3] uint8
        T = frames_np.shape[0]
        # Normalize and transpose to [T, 3, 224, 224] float32
        frames_f32 = frames_np.astype(np.float32) / 255.0          # [T, 224, 224, 3]
        frames_t = torch.from_numpy(frames_f32).permute(0, 3, 1, 2)  # [T, 3, 224, 224]
        frames_t = frames_t.to(device)

        # Process in chunks of 64 to avoid OOM on large videos
        chunk_size = 64
        feats_list = []
        for start in range(0, T, chunk_size):
            chunk = frames_t[start: start + chunk_size]  # [C, 3, 224, 224]
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                    enabled=(device.type == "cuda")):
                feats = encoder.backbone(chunk)           # [C, 1024, 14, 14]
            feats_list.append(feats.cpu().half())

        cache[vid] = torch.cat(feats_list, dim=0)        # [T, 1024, 14, 14] fp16 CPU

    # Restore original training state and re-apply trainability rules
    if was_training:
        encoder.train()
        encoder.configure_trainability()

    print(f"[extract_backbone_features] Done. Cache holds {len(cache)} videos.")
    return cache


def train(
    num_epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    log_every: int = 10,
    num_workers: int = 0,
    checkpoint_every: int = 1000,
    checkpoint_dir: str = "checkpoints/encoder",
    h5_path: str = None,
    register: bool = False,
    register_alias: str | None = None,
):
    def _accumulate_metric(metric_sum: dict, key: str, value) -> None:
        if isinstance(value, torch.Tensor):
            metric_value = value.detach()
            if metric_value.ndim != 0:
                metric_value = metric_value.mean()
        else:
            metric_value = torch.tensor(float(value), device=device, dtype=torch.float32)
        if key in metric_sum:
            metric_sum[key] = metric_sum[key] + metric_value
        else:
            metric_sum[key] = metric_value

    def _finalize_metric_averages(metric_sum: dict, num_steps: int) -> dict:
        if num_steps <= 0:
            return {}
        metric_avg = {}
        denom = float(num_steps)
        for key, value in metric_sum.items():
            if isinstance(value, torch.Tensor):
                metric_avg[key] = (value / denom).cpu().item()
            else:
                metric_avg[key] = float(value) / denom
        return metric_avg

    _train_cfg = _TRAIN_V2                              # [v2] resolved train config
    _maybe_limit_cpu_threads(bool(_train_cfg.get("limit_cpu_threads", False)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[train] Using device: {device}")

    # Resolve checkpoint directory (run-specific subfolder)
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(_PROJECT_ROOT, checkpoint_dir)
    # checkpoint_dir will be finalized after wandb.init() provides the run name

    # Generate run name with timestamp.
    _backbone_name = _train_cfg.get("backbone_name", "resnet50_conv4c")
    _train_base = _train_cfg.get("train_base", "only_bn")
    # CLI --h5_path overrides the value resolved from configs_v2/train.yaml
    _effective_h5 = h5_path if h5_path is not None else _TRAIN_V2.get("h5_path", "unknown")
    _dataset_name = os.path.splitext(os.path.basename(_effective_h5))[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if _loss_name_v2 == "composite":
        # Append component aliases so run name reflects contents, e.g. COMPOSITE_TCC_TRIPLET
        try:
            with open(_V2_LOSS_YAML, "r") as _f:
                _comp_cfg = yaml.safe_load(_f) or {}
            _aliases = [c["alias"] for c in _comp_cfg.get("components", [])]
            _component_suffix = "_" + "_".join(a.upper() for a in _aliases) if _aliases else ""
        except Exception:
            _component_suffix = ""
        _loss_tag = "COMPOSITE" + _component_suffix
    else:
        _loss_tag = _loss_name_v2.replace("_", "-").upper()  # e.g. "TCC", "TEMPORAL-INFONCE"
    run_name = f"{_loss_tag}-{_dataset_name}-{_backbone_name}-{_train_base}-{timestamp}"

    wandb.init(
        project="mytcc",
        name=run_name,
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "log_every": log_every,
            "num_workers": num_workers,
            "limit_cpu_threads": bool(_train_cfg.get("limit_cpu_threads", False)),
            "checkpoint_every": checkpoint_every,
            "checkpoint_dir": checkpoint_dir,
        },
    )

    # Upload YAML configs so the run is fully reproducible
    _yaml_to_save = [_V2_TRAIN_YAML, _V2_LOSS_YAML]
    if _loss_name_v2 == "composite":
        # Also save each component's child YAML
        _loss_yaml_dir = os.path.dirname(_V2_LOSS_YAML)
        try:
            with open(_V2_LOSS_YAML, "r") as _f:
                _comp_cfg_for_save = yaml.safe_load(_f) or {}
            for _comp in _comp_cfg_for_save.get("components", []):
                _child = _comp.get("config_file")
                if _child:
                    _yaml_to_save.append(os.path.join(_loss_yaml_dir, _child))
        except Exception:
            pass
    _configs_root = os.path.dirname(_V2_TRAIN_YAML)  # configs_v2/
    for _ypath in _yaml_to_save:
        if os.path.isfile(_ypath):
            wandb.save(_ypath, base_path=_configs_root, policy="now")

    # Create run-specific checkpoint subfolder using wandb run name
    run_checkpoint_dir = os.path.join(checkpoint_dir, wandb.run.name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"[train] Checkpoint dir: {run_checkpoint_dir}")

    # ------------------------------------------------------------------
    # 1. DataLoader
    # ------------------------------------------------------------------
    _h5_override = h5_path if h5_path is not None else _TRAIN_V2["h5_path"]  # [v2]
    dataloader = build_dataloader(                                             # [v2]
        config_path=_V2_TRAIN_YAML,  # [v2] configs_v2/train.yaml has clip_len/context_size/etc.
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True, # 默认打乱训练，这样training dataset所有视频可以匹配到任意一个pair计算loss
        split="train",
        h5_path_override=_h5_override,  # [v2] resolved path always provided explicitly
    )

    # ------------------------------------------------------------------
    # 1.5. Fix random seed for reproducibility
    # ------------------------------------------------------------------
    _seed = _TRAIN_V2.get("seed", 42)  # [v2]
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    print(f"[train] Random seed: {_seed}")

    # ------------------------------------------------------------------
    # 2. Encoder
    # ------------------------------------------------------------------
    encoder = TCCEncoder(                      # [v2]
        config_path=_V2_TRAIN_YAML,            # [v2] has clip_len, context_size, context_stride
        train_config_path=_V2_TRAIN_YAML,      # [v2] has backbone_name, train_base, train_embedding, pretrained
    )
    encoder = encoder.to(device)
    encoder.train()
    # encoder.train() resets all submodule states; restore trainability rules.
    encoder.configure_trainability()

    # Print trainability info
    summary = encoder.get_trainable_parameter_groups_summary()
    total_params = summary["backbone_total"] + summary["embedder_total"]
    trainable_params = summary["backbone_trainable"] + summary["embedder_trainable"]
    print(f"[train] train_base      : {encoder._train_base}")
    print(f"[train] train_embedding : {encoder._train_embedding}")
    print(f"[train] limit_cpu_threads: {bool(_train_cfg.get('limit_cpu_threads', False))}")
    print(f"[train] total params    : {total_params:,}")
    print(f"[train] trainable params: {trainable_params:,}")

    # ------------------------------------------------------------------
    # 3. Loss module
    # ------------------------------------------------------------------
    # [v1] loss_module = build_loss("tcc", config_path=CONFIG_LOSS)
    loss_module = build_loss(_loss_name_v2, config_path=_V2_LOSS_YAML)  # [v2] config-driven
    loss_module = loss_module.to(device)
    loss_module.train()

    # ------------------------------------------------------------------
    # 4. Optimizer  (only trainable parameters)
    # ------------------------------------------------------------------
    trainable_params_list = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params_list, lr=learning_rate)

    # AMP GradScaler (no-op when CUDA is unavailable)
    _use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(device="cuda", enabled=_use_amp)
    _autocast_device = "cuda" if _use_amp else "cpu"
    _transport_frames_as_uint8 = bool(_train_cfg.get("transport_frames_as_uint8", False))

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------

    # Decide whether to use backbone feature cache (independent branch)
    # frozen   — backbone never changes; extract once, reuse forever, no refresh.
    # only_bn  — BN γ/β drift each step; refresh cache every _cache_refresh_every epochs.
    # [v1] _train_cfg_full = load_yaml(CONFIG_TRAIN)
    _train_cfg_full = _TRAIN_V2  # [v2]
    _use_feat_cache = (
        encoder._train_base in ("only_bn", "frozen")
        and _train_cfg_full.get("extract_backbone_cache", False)
    )
    _cache_refresh_every = int(_train_cfg_full.get("cache_refresh_every", 100))

    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch
    global_step = 0
    perf_train_seconds = 0.0
    perf_train_epochs = 0

    pbar = tqdm(
        total=num_epochs,
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
    )

    if _use_feat_cache:
        # ------------------------------------------------------------------
        # CACHE BRANCH — skip backbone in every training step.
        # frozen:   Extract once, never refresh (backbone is fully static).
        # only_bn:  Refresh every _cache_refresh_every epochs (BN γ/β drift).
        # Active when train_base in (frozen, only_bn) AND
        # extract_backbone_cache=True in configs_v2/train.yaml.
        # ------------------------------------------------------------------
        _is_frozen_cache = (encoder._train_base == "frozen")
        if _is_frozen_cache:
            print("[train] Backbone cache mode enabled (frozen — extract once, no refresh)")
        else:
            print(f"[train] Backbone cache mode enabled "
                  f"(only_bn — refresh every {_cache_refresh_every} epochs)")

        _feat_cache = extract_backbone_features(encoder, dataloader.dataset, device)
        _cache_dataloader = build_feature_cache_dataloader(
            _feat_cache,
            dataloader.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        for epoch in range(num_epochs):
            # frozen: features never change, skip refresh entirely.
            # only_bn: BN γ/β update each step, refresh periodically.
            if not _is_frozen_cache and epoch > 0 and epoch % _cache_refresh_every == 0:
                _feat_cache = extract_backbone_features(encoder, dataloader.dataset, device)
                _cache_dataloader = build_feature_cache_dataloader(
                    _feat_cache,
                    dataloader.dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )

            epoch_train_start = time.perf_counter()
            epoch_loss_sum = 0.0
            epoch_steps = 0
            epoch_metrics_sum: dict = {}

            for batch in _cache_dataloader:
                # backbone_feats: [B, clip_len, context_size, 1024, 14, 14] fp16
                backbone_feats = batch["backbone_feats"].to(device, non_blocking=True)
                loss_batch = {
                    "target_steps": batch["target_steps"].to(device, non_blocking=True),
                    "seq_len": batch["seq_len"].to(device, non_blocking=True),
                }

                with torch.amp.autocast(device_type=_autocast_device, enabled=_use_amp):
                    embeddings = encoder.forward_from_feats(backbone_feats)
                    out = loss_module(embeddings, loss_batch)
                    loss = out["loss"]

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                epoch_loss_sum += loss.detach()
                epoch_steps += 1
                for k, v in out.get("metrics", {}).items():
                    _accumulate_metric(epoch_metrics_sum, k, v)

            should_log = (epoch + 1) % log_every == 0
            if should_log:
                epoch_loss_avg = (epoch_loss_sum / max(epoch_steps, 1)).cpu().item()
                epoch_metrics_avg = _finalize_metric_averages(epoch_metrics_sum, epoch_steps)
            perf_train_seconds += time.perf_counter() - epoch_train_start
            perf_train_epochs += 1
            pbar.update(1)
            if should_log:
                seconds_per_epoch = perf_train_seconds / perf_train_epochs
                pbar.set_postfix(loss=f"{epoch_loss_avg:.4f}")
                wandb.log(
                    {
                        "loss": epoch_loss_avg,
                        **epoch_metrics_avg,
                        "perf/s_per_epoch": seconds_per_epoch,
                    },
                    step=epoch + 1,
                )
                perf_train_seconds = 0.0
                perf_train_epochs = 0

            if (epoch + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(run_checkpoint_dir, f"encoder_epoch{epoch+1:06d}.pt")
                _save_encoder_checkpoint(encoder, ckpt_path)
                tqdm.write(f"[train] Saved checkpoint: {ckpt_path}")
                # [v2] In-training eval hook (lazy import; only active when enabled=true)
                _ite_eval_cfg = _train_cfg_full.get("in_training_eval", {})
                if _ite_eval_cfg.get("enabled", False):
                    from utils.in_training_eval import run_in_training_eval  # noqa: PLC0415
                    run_in_training_eval(
                        encoder=encoder,
                        eval_cfg=_ite_eval_cfg,
                        train_cfg=_train_cfg_full,
                        run_checkpoint_dir=run_checkpoint_dir,
                        epoch=epoch,
                        checkpoint_count=(epoch + 1) // checkpoint_every,
                        device=device,
                        train_config_path=_V2_TRAIN_YAML,
                        checkpoint_path=ckpt_path,
                    )

    else:
        # ------------------------------------------------------------------
        # ORIGINAL BRANCH — full forward through backbone every step.
        # This is the only branch used for frozen, train_all, and
        # only_bn-without-cache modes. Zero changes from the pre-cache code.
        # ------------------------------------------------------------------
        for epoch in range(num_epochs):
            epoch_train_start = time.perf_counter()
            epoch_loss_sum = 0.0
            epoch_steps = 0
            epoch_metrics_sum: dict = {}

            for batch in dataloader:
                # Move tensors to device
                frames = batch["frames"].to(device, non_blocking=True)  # [B, clip_len, ctx, 3, H, W]
                if _transport_frames_as_uint8:
                    frames = frames.float().div_(255.0)
                loss_batch = {
                    "target_steps": batch["target_steps"].to(device, non_blocking=True),
                    "seq_len": batch["seq_len"].to(device, non_blocking=True),
                }

                # Forward: encoder + loss  (AMP autocast)
                with torch.amp.autocast(device_type=_autocast_device, enabled=_use_amp):
                    embeddings = encoder(frames)              # [B, clip_len, D]
                    out = loss_module(embeddings, loss_batch)
                    loss = out["loss"]

                # Backward (AMP scaler)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                epoch_loss_sum += loss.detach()
                epoch_steps += 1
                for k, v in out.get("metrics", {}).items():
                    _accumulate_metric(epoch_metrics_sum, k, v)

            should_log = (epoch + 1) % log_every == 0
            if should_log:
                epoch_loss_avg = (epoch_loss_sum / max(epoch_steps, 1)).cpu().item()
                epoch_metrics_avg = _finalize_metric_averages(epoch_metrics_sum, epoch_steps)
            perf_train_seconds += time.perf_counter() - epoch_train_start
            perf_train_epochs += 1
            pbar.update(1)
            if should_log:
                seconds_per_epoch = perf_train_seconds / perf_train_epochs
                pbar.set_postfix(loss=f"{epoch_loss_avg:.4f}")
                wandb.log(
                    {
                        "loss": epoch_loss_avg,
                        **epoch_metrics_avg,
                        "perf/s_per_epoch": seconds_per_epoch,
                    },
                    step=epoch + 1,
                )
                perf_train_seconds = 0.0
                perf_train_epochs = 0

            # Save checkpoint every checkpoint_every epochs
            if (epoch + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(run_checkpoint_dir, f"encoder_epoch{epoch+1:06d}.pt")
                _save_encoder_checkpoint(encoder, ckpt_path)
                tqdm.write(f"[train] Saved checkpoint: {ckpt_path}")
                # [v2] In-training eval hook (lazy import; only active when enabled=true)
                _ite_eval_cfg = _train_cfg_full.get("in_training_eval", {})
                if _ite_eval_cfg.get("enabled", False):
                    from utils.in_training_eval import run_in_training_eval  # noqa: PLC0415
                    run_in_training_eval(
                        encoder=encoder,
                        eval_cfg=_ite_eval_cfg,
                        train_cfg=_train_cfg_full,
                        run_checkpoint_dir=run_checkpoint_dir,
                        epoch=epoch,
                        checkpoint_count=(epoch + 1) // checkpoint_every,
                        device=device,
                        train_config_path=_V2_TRAIN_YAML,
                        checkpoint_path=ckpt_path,
                    )

    pbar.close()

    # Capture actual run name before wandb session closes
    _actual_run_name = wandb.run.name
    wandb.finish()
    print(f"[train] Run finished: run_name={_actual_run_name}, checkpoint_epoch={num_epochs}")
    print("[train] Training complete.")

    # [v2] Optional: register run into configs_v2/registry/runs.yaml
    if register:
        from utils.registry_v2 import RegistryV2
        _reg = RegistryV2()
        _train_dataset = _TRAIN_V2.get("dataset_ref", "") or _dataset_name
        _backbone = _TRAIN_V2.get("backbone_name", "resnet50_conv4c")
        _train_base = _TRAIN_V2.get("train_base", "only_bn")
        _alias = register_alias or _reg.suggest_run_alias(_train_dataset, num_epochs)
        _reg.register_run(
            alias            = _alias,
            run_name         = _actual_run_name,
            train_dataset    = _train_dataset,
            backbone         = _backbone,
            train_base       = _train_base,
            checkpoint_epoch = num_epochs,
            description      = f"{_train_dataset}, epoch {num_epochs}",
        )
        print(f"[train] [v2] Run registered as '{_alias}' in configs_v2/registry/runs.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCC training script")
    parser.add_argument(
        "--h5_path",
        type=str,
        default=None,
        help="Path to the H5 dataset file. Overrides h5_path resolved from configs_v2/train.yaml.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        default=False,
        help="[v2] After training, register the run into configs_v2/registry/runs.yaml.",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        dest="register_alias",
        help="[v2] Registry alias for the run (auto-suggested if not set). Requires --register.",
    )
    args = parser.parse_args()

    cfg = _TRAIN_V2  # [v2]

    # ------------------------------------------------------------------
    # Launch full training
    # ------------------------------------------------------------------
    train(
        num_epochs=cfg.get("num_epochs", 5),
        batch_size=cfg.get("batch_size", 2),
        learning_rate=cfg.get("learning_rate", 1e-4),
        log_every=cfg.get("log_every", 10),
        num_workers=cfg.get("num_workers", 0),
        checkpoint_every=cfg.get("checkpoint_every", 1000),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints/encoder"),
        h5_path=args.h5_path,
        register=args.register,
        register_alias=args.register_alias,
    )
