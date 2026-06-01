"""
Real-data integration test for CompositeEncoderLoss (TCC + TemporalInfoNCE).

Replicates the training data path from train.py end-to-end:

    build_dataloader()
        → batch: frames [B, T, Ctx, 3, 224, 224]
                 target_steps [B, T]
                 seq_len [B]
    TCCEncoder.forward(frames)
        → embeddings [B, T, D=128]
    build_loss("tcc")         (embeddings, loss_batch) → tcc_out
    build_loss("temporal_infonce") (embeddings, loss_batch) → infonce_out
    build_loss("composite")   (embeddings, loss_batch) → composite_out
    loss.backward()

Run from the mytcc project root:
    conda run -n fineprog python scripts/test_composite_loss.py
"""

import argparse
import sys
import os
import pathlib

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch

from extract_embeddings import load_checkpoint
from utils.config_v2 import ConfigV2
from dataset_preparation.h5vid_dataset import build_dataloader
from models.encoder import TCCEncoder
from algos.loss.encoder_loss import build_loss

PASS = "[PASS]"
FAIL = "[FAIL]"

_V2_DIR         = pathlib.Path(_root) / "configs_v2"
_V2_TRAIN_YAML  = str(_V2_DIR / "train.yaml")
_COMPOSITE_YAML = str(_V2_DIR / "loss" / "loss_composite_tcc_infonce.yaml")
_TCC_YAML       = str(_V2_DIR / "loss" / "loss_tcc.yaml")
_INFONCE_YAML   = str(_V2_DIR / "loss" / "loss_temporal_infonce.yaml")
_DEFAULT_ENCODER_SOURCE = "random_init"
_DEFAULT_TRAINED_RUN_REF = "can_ph_180_ep50k"

# Test-specific overrides (keep I/O light)
_TEST_BATCH_SIZE = 2
_TEST_NUM_WORKERS = 0


# ─────────────────────────────────────────────────────────────────────────────
# Setup: resolve config and build dataloader + encoder
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint_info(cfg, train_v2, checkpoint_run_ref=None, checkpoint_path=None):
    """Resolve checkpoint metadata for the trained-encoder branch."""
    if checkpoint_path:
        ckpt_path = pathlib.Path(checkpoint_path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return {
            "run_ref": None,
            "run_name": None,
            "checkpoint_epoch": None,
            "train_dataset": None,
            "checkpoint_path": str(ckpt_path),
        }

    run_ref = checkpoint_run_ref or _DEFAULT_TRAINED_RUN_REF
    run_info = cfg.resolve_run(run_ref)
    ckpt_path = pathlib.Path(run_info["checkpoint_path"]).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Resolved checkpoint for run_ref='{run_ref}' does not exist: {ckpt_path}"
        )

    run_train_dataset = run_info.get("train_dataset")
    train_dataset = train_v2.get("train_dataset")
    if run_train_dataset and train_dataset and run_train_dataset != train_dataset:
        raise ValueError(
            "Checkpoint train_dataset mismatch: "
            f"run_ref='{run_ref}' uses '{run_train_dataset}', "
            f"but train.yaml uses '{train_dataset}'"
        )

    run_info["run_ref"] = run_ref
    run_info["checkpoint_path"] = str(ckpt_path)
    return run_info


def _setup(encoder_source=_DEFAULT_ENCODER_SOURCE,
           checkpoint_run_ref=None,
           checkpoint_path=None):
    """
    Mirrors the setup sequence from train.py:
      1. Resolve train config via ConfigV2
      2. Build dataloader (batch_size=2, no workers)
      3. Build TCCEncoder (pretrained=False to keep test fast)
      4. Optionally load a trained checkpoint into the encoder
      5. Pull one batch and run encoder forward

    Returns
    -------
    embeddings  : [B, T, D]  float32, requires_grad=True
    loss_batch  : {"target_steps": [B,T], "seq_len": [B]}
    batch       : full raw batch for shape inspection
    device      : torch.device used
    """
    cfg      = ConfigV2()
    train_v2 = cfg.load_train()

    h5_path       = train_v2["h5_path"]
    train_dataset = train_v2["train_dataset"]
    clip_len      = train_v2["clip_len"]       # 20
    context_sz    = train_v2["context_size"]   # 2
    ctx_stride    = train_v2["context_stride"] # 15

    print(f"\n[setup] Dataset H5 : {h5_path}")
    print(f"[setup] train_dataset={train_dataset}")
    print(f"[setup] clip_len={clip_len}  context_size={context_sz}  "
          f"context_stride={ctx_stride}")

    # ── Dataloader ────────────────────────────────────────────────────
    dataloader = build_dataloader(
        config_path      = _V2_TRAIN_YAML,
        h5_path_override = h5_path,
        batch_size       = _TEST_BATCH_SIZE,
        num_workers      = _TEST_NUM_WORKERS,
        shuffle          = True,
        split            = "train",
    )
    batch = next(iter(dataloader))

    frames       = batch["frames"]        # [B, T, Ctx, 3, 224, 224]
    target_steps = batch["target_steps"]  # [B, T]   int32
    seq_len      = batch["seq_len"]       # [B]      int32

    print(f"\n[setup] ── batch shapes ──")
    print(f"  frames       : {tuple(frames.shape)}  dtype={frames.dtype}")
    print(f"  target_steps : {tuple(target_steps.shape)}  dtype={target_steps.dtype}")
    print(f"  seq_len      : {tuple(seq_len.shape)}  values={seq_len.tolist()}")
    print(f"  video_ids    : {batch['video_id']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device: {device}")

    if encoder_source not in (_DEFAULT_ENCODER_SOURCE, "trained_ckpt"):
        raise ValueError(
            f"Unknown encoder_source='{encoder_source}'. "
            "Expected 'random_init' or 'trained_ckpt'."
        )

    checkpoint_info = None
    if encoder_source == "trained_ckpt":
        checkpoint_info = _resolve_checkpoint_info(
            cfg,
            train_v2,
            checkpoint_run_ref=checkpoint_run_ref,
            checkpoint_path=checkpoint_path,
        )

    print(f"[setup] encoder source: {encoder_source}")
    if checkpoint_info is not None:
        if checkpoint_info.get("run_ref") is not None:
            print(f"[setup] checkpoint run_ref: {checkpoint_info['run_ref']}")
        if checkpoint_info.get("run_name") is not None:
            print(f"[setup] checkpoint run_name: {checkpoint_info['run_name']}")
        if checkpoint_info.get("checkpoint_epoch") is not None:
            print(f"[setup] checkpoint epoch: {checkpoint_info['checkpoint_epoch']}")
        print(f"[setup] checkpoint path: {checkpoint_info['checkpoint_path']}")

    # ── Encoder (pretrained=False: skip ImageNet weights for test speed) ──
    encoder = TCCEncoder(
        config_path       = _V2_TRAIN_YAML,
        train_config_path = _V2_TRAIN_YAML,
        pretrained        = False,
    ).to(device)

    if checkpoint_info is not None:
        load_checkpoint(encoder, checkpoint_info["checkpoint_path"], device)

    encoder.eval()

    with torch.no_grad():
        embeddings_no_grad = encoder(frames.to(device))   # [B, T, D=128]
    # Re-attach requires_grad so backward works through loss
    embeddings = embeddings_no_grad.detach().requires_grad_(True)

    print(f"\n[setup] ── encoder output ──")
    print(f"  embeddings : {tuple(embeddings.shape)}  dtype={embeddings.dtype}")
    print(f"  finite     : {torch.isfinite(embeddings).all().item()}")
    print(f"  mean={embeddings.mean():.4f}  std={embeddings.std():.4f}")

    loss_batch = {
        "target_steps": target_steps.to(device),
        "seq_len":       seq_len.to(device),
    }

    return embeddings, loss_batch, batch, device


# ─────────────────────────────────────────────────────────────────────────────
# Main test
# ─────────────────────────────────────────────────────────────────────────────

def _run_composite_real_data(encoder_source=_DEFAULT_ENCODER_SOURCE,
                             checkpoint_run_ref=None,
                             checkpoint_path=None):
    """
    One-shot integration test: real frames → encoder → TCC + InfoNCE + composite.

    Data flow
    ---------
    frames [B, T, Ctx, 3, 224, 224]          (H5VideoDataset / build_dataloader)
      ↓  TCCEncoder.forward()
    embeddings [B, T, D=128]                 (shared by all three loss paths)
      ↓  TCCLoss.forward()
    tcc_out: {"loss": scalar, "metrics": {loss_tcc, loss_total, ...}}
      ↓  TemporalInfoNCELoss.forward()       (same embeddings, per-video anchors)
    infonce_out: {"loss": scalar, "metrics": {num_valid_anchors, mean_pos_dist2, ...}}
      ↓  CompositeEncoderLoss.forward()      (passes embeddings to both children)
    composite_out: {"loss": w_tcc*tcc + w_inf*infonce,
                    "metrics": {component_raw_loss/tcc, component_raw_loss/temporal_infonce,
                                component_weighted_loss/tcc, component_weighted_loss/temporal_infonce,
                                tcc/<child_keys>, temporal_infonce/<child_keys>,
                                loss_composite, loss_total}}
      ↓  loss.backward()
    gradients back to embeddings

    Weights (from loss_composite_tcc_infonce.yaml):
      w_tcc     = 1.0
      w_infonce = 0.5
    """
    torch.manual_seed(42)
    print("\n" + "=" * 70)
    print("=== test_composite_real_data: real frames → encoder → composite loss ===")
    print("=" * 70)

    embeddings, loss_batch, batch, device = _setup(
        encoder_source=encoder_source,
        checkpoint_run_ref=checkpoint_run_ref,
        checkpoint_path=checkpoint_path,
    )
    B, T, D = embeddings.shape

    # ── Layer 1: TCC loss (standalone) ────────────────────────────────────
    print("\n── Layer 1: TCCLoss (standalone) ──────────────────────────────────")
    tcc_fn  = build_loss("tcc", config_path=_TCC_YAML)
    tcc_out = tcc_fn(embeddings, loss_batch)
    tcc_loss_val = tcc_out["loss"].item()
    print(f"  tcc_loss             : {tcc_loss_val:.6f}")
    print(f"  metrics[loss_tcc]    : {tcc_out['metrics']['loss_tcc']}")
    print(f"  metrics[loss_total]  : {tcc_out['metrics']['loss_total']}")
    print(f"  stochastic_matching  : {tcc_out['metrics']['tcc/stochastic_matching']}")
    assert torch.isfinite(tcc_out["loss"]), "TCC loss is not finite"

    # ── Layer 2: TemporalInfoNCE loss (standalone, per-video debug) ───────
    print("\n── Layer 2: TemporalInfoNCELoss (standalone) ──────────────────────")
    infonce_fn  = build_loss("temporal_infonce", config_path=_INFONCE_YAML)
    infonce_out = infonce_fn(embeddings, loss_batch)
    infonce_loss_val = infonce_out["loss"].item()
    print(f"  infonce_loss            : {infonce_loss_val:.6f}")
    print(f"  num_valid_anchors       : {infonce_out['metrics']['num_valid_anchors']}")
    print(f"  num_sampled_anchors     : {infonce_out['metrics']['num_sampled_anchors']}")
    print(f"  mean_pos_dist2          : {infonce_out['metrics']['mean_pos_dist2']:.4f}")
    print(f"  mean_neg_dist2          : {infonce_out['metrics']['mean_neg_dist2']:.4f}")
    assert torch.isfinite(infonce_out["loss"]), "InfoNCE loss is not finite"

    # ── Per-video debug: detailed InfoNCE analysis ───────────────────────
    print("\n  [debug] per-video detailed InfoNCE analysis:")
    with torch.no_grad():
        steps_f    = loss_batch["target_steps"].float()                        # [B, T]
        anchor_idx = infonce_fn._select_anchor_indices(T, embeddings.device)  # [A]
        A_size     = anchor_idx.shape[0]

        per_video_stats = []   # (video_id, n_valid, loss_sum, loss_mean)

        for b in range(B):
            z = embeddings[b]   # [T, D]
            s = steps_f[b]      # [T]

            step_range = (s.max() - s.min()).item()
            abs_pos    = infonce_fn.pos_threshold * step_range
            abs_neg    = infonce_fn.neg_threshold * step_range

            # ── All target_steps for this clip ────────────────────────
            print(f"\n  ── video {batch['video_id'][b]}  "
                  f"(seq_len={loss_batch['seq_len'][b].item()}) ──")
            print(f"    step_range={step_range:.1f}  "
                  f"abs_pos(thr={infonce_fn.pos_threshold})={abs_pos:.2f}  "
                  f"abs_neg(thr={infonce_fn.neg_threshold})={abs_neg:.2f}")
            steps_list = s.long().tolist()
            print(f"    all target_steps ({T} frames):")
            for i in range(0, T, 10):
                end   = min(i + 10, T)
                chunk = steps_list[i:end]
                print(f"      frame [{i:2d}-{end-1:2d}]: {chunk}")

            # ── Compute pairwise quantities for all anchors ────────────
            z_anc = z[anchor_idx]                                           # [A, D]
            s_anc = s[anchor_idx]                                           # [A]
            gap   = (s_anc.unsqueeze(1) - s.unsqueeze(0)).abs()            # [A, T]

            pos_mask = (gap > 0) & (gap <= abs_pos)                        # [A, T]
            neg_mask = (gap >= abs_neg)                                    # [A, T]

            dist2  = infonce_fn._squared_l2_dist(z_anc, z)                # [A, T]
            logits = -dist2 / infonce_fn.temperature                       # [A, T]

            has_pos = pos_mask.any(dim=1)                                  # [A]
            has_neg = neg_mask.any(dim=1)                                  # [A]
            valid   = has_pos & has_neg                                    # [A]

            NEG_INF   = torch.finfo(logits.dtype).min
            log_num   = torch.logsumexp(
                            logits.masked_fill(~pos_mask, NEG_INF), dim=1) # [A]
            log_denom = torch.logsumexp(
                            logits.masked_fill(~(pos_mask | neg_mask), NEG_INF), dim=1)  # [A]
            loss_anchors = log_denom - log_num                             # [A]

            n_valid   = int(valid.sum().item())
            loss_sum  = float(loss_anchors[valid].sum().item()) if n_valid > 0 else 0.0
            loss_mean = loss_sum / n_valid if n_valid > 0 else float("nan")
            print(f"    valid anchors: {n_valid}/{A_size}  "
                  f"per-video mean loss = {loss_mean:.6f}  sum = {loss_sum:.6f}")
            per_video_stats.append((batch["video_id"][b], n_valid, loss_sum, loss_mean))

            # ── (1) Example anchor: middle valid anchor ───────────────
            valid_positions = torch.where(valid)[0]
            if len(valid_positions) == 0:
                print(f"    (no valid anchor — skipping per-anchor detail)")
                continue

            ea_pos       = valid_positions[len(valid_positions) // 2].item()  # idx in anchor_idx
            ea_frame_idx = anchor_idx[ea_pos].item()                          # clip frame idx
            ea_step      = int(s[ea_frame_idx].item())

            print(f"\n    ── (1) example anchor: frame_idx={ea_frame_idx}  "
                  f"target_step={ea_step} ──")
            print(f"    {'f_idx':>5}  {'step':>6}  {'gap':>6}  "
                  f"{'type':>4}  {'logit':>9}  {'dist2':>9}")

            pos_frames = []
            neg_frames = []
            for j in range(T):
                g       = gap[ea_pos, j].item()
                is_pos  = bool(pos_mask[ea_pos, j].item())
                is_neg  = bool(neg_mask[ea_pos, j].item())
                logit_j = logits[ea_pos, j].item()
                dist2_j = dist2[ea_pos, j].item()
                if j == ea_frame_idx:
                    ftype = "self"
                elif is_pos:
                    ftype = "POS"
                    pos_frames.append((j, int(s[j].item()), logit_j))
                elif is_neg:
                    ftype = "NEG"
                    neg_frames.append((j, int(s[j].item()), logit_j))
                else:
                    ftype = "ign"
                print(f"    {j:>5}  {int(s[j].item()):>6}  "
                      f"{g:>6.1f}  {ftype:>4}  {logit_j:>9.4f}  {dist2_j:>9.4f}")

            # ── (2) InfoNCE logit / loss breakdown for this anchor ─────
            print(f"\n    ── (2) InfoNCE computation for anchor frame_idx={ea_frame_idx} ──")
            print(f"      pos frames ({len(pos_frames)}):")
            for f_idx, f_step, f_logit in pos_frames:
                print(f"        frame_idx={f_idx}  step={f_step}  logit={f_logit:.4f}")
            print(f"      neg frames ({len(neg_frames)}):")
            for f_idx, f_step, f_logit in neg_frames:
                print(f"        frame_idx={f_idx}  step={f_step}  logit={f_logit:.4f}")
            print(f"      log_num   = logsumexp(pos logits)     = {log_num[ea_pos].item():.4f}")
            print(f"      log_denom = logsumexp(pos+neg logits) = {log_denom[ea_pos].item():.4f}")
            print(f"      loss_anchor = log_denom - log_num     = {loss_anchors[ea_pos].item():.4f}")

        # ── (3) Per-video losses ──────────────────────────────────────
        print(f"\n  ── (3) per-video loss summary ──")
        total_valid = sum(n for _, n, _, _ in per_video_stats)
        total_sum   = sum(ls for _, _, ls, _ in per_video_stats)
        for vid, n_v, l_sum, l_mean in per_video_stats:
            print(f"    {vid}: n_valid={n_v}  "
                  f"loss_sum={l_sum:.6f}  loss_mean={l_mean:.6f}")

        # ── (4) Batch loss reconstruction ─────────────────────────────
        batch_loss_recon = total_sum / total_valid if total_valid > 0 else float("nan")
        print(f"\n  ── (4) batch InfoNCE loss ──")
        print(f"    total_valid_anchors = {total_valid}  "
              f"(= {' + '.join(str(n) for _, n, _, _ in per_video_stats)})")
        print(f"    batch_loss = total_anchor_loss_sum / total_valid")
        print(f"               = {total_sum:.6f} / {total_valid}")
        print(f"               = {batch_loss_recon:.6f}")
        print(f"    infonce_fn output  = {infonce_loss_val:.6f}")
        match = abs(batch_loss_recon - infonce_loss_val) < 1e-4
        print(f"    reconstructed == forward() ? {match}")

    # ── Layer 3: CompositeEncoderLoss ─────────────────────────────────────
    print("\n── Layer 3: CompositeEncoderLoss (TCC × 1.0 + InfoNCE × 0.5) ──────")
    comp_fn  = build_loss("composite", config_path=_COMPOSITE_YAML)
    # Pass a fresh require_grad view so backward goes through a clean graph
    emb_for_comp = embeddings.detach().requires_grad_(True)
    comp_out  = comp_fn(emb_for_comp, loss_batch)
    comp_loss = comp_out["loss"]
    mets      = comp_out["metrics"]

    w_tcc    = mets["component_weight/tcc"]            # 1.0
    w_inf    = mets["component_weight/temporal_infonce"]  # 0.5

    print(f"\n  weights:  tcc={w_tcc}  temporal_infonce={w_inf}")
    print(f"\n  component_raw_loss/tcc             : {mets['component_raw_loss/tcc']:.6f}")
    print(f"  component_raw_loss/temporal_infonce: {mets['component_raw_loss/temporal_infonce']:.6f}")
    print(f"  component_weighted_loss/tcc             : {mets['component_weighted_loss/tcc']:.6f}")
    print(f"  component_weighted_loss/temporal_infonce: {mets['component_weighted_loss/temporal_infonce']:.6f}")
    print(f"\n  loss_composite : {mets['loss_composite']:.6f}")
    print(f"  loss_total     : {mets['loss_total']:.6f}")

    # Verify namespaced child metrics are present
    print(f"\n  [namespaced child metrics]")
    for key in ("tcc/loss_tcc", "tcc/loss_total",
                "temporal_infonce/loss_temporal_infonce",
                "temporal_infonce/num_valid_anchors",
                "temporal_infonce/mean_pos_dist2",
                "temporal_infonce/mean_neg_dist2"):
        assert key in mets, f"Missing metric key: '{key}'"
        print(f"    {key} = {mets[key]}")

    # Verify no unqualified collision keys
    assert "loss_tcc"             not in mets, "loss_tcc must be prefixed"
    assert "loss_temporal_infonce" not in mets, "loss_temporal_infonce must be prefixed"

    # ── Numerical consistency: composite == w_tcc*tcc + w_inf*infonce ─────
    print("\n── Numerical consistency check ────────────────────────────────────")
    expected = w_tcc * mets["component_raw_loss/tcc"] + \
               w_inf * mets["component_raw_loss/temporal_infonce"]
    delta = abs(comp_loss.item() - expected)
    print(f"  w_tcc*loss_tcc + w_inf*loss_infonce = {expected:.6f}")
    print(f"  composite loss                      = {comp_loss.item():.6f}")
    print(f"  Δ                                   = {delta:.2e}")
    assert delta < 1e-4, f"composite loss mismatch: expected {expected:.6f}, got {comp_loss.item():.6f}"

    # Also verify component_raw_loss values match standalone runs
    delta_tcc = abs(mets["component_raw_loss/tcc"] - tcc_loss_val)
    delta_inf = abs(mets["component_raw_loss/temporal_infonce"] - infonce_loss_val)
    print(f"\n  standalone tcc    = {tcc_loss_val:.6f}  composite tcc    = {mets['component_raw_loss/tcc']:.6f}  Δ={delta_tcc:.2e}")
    print(f"  standalone infonce= {infonce_loss_val:.6f}  composite infonce= {mets['component_raw_loss/temporal_infonce']:.6f}  Δ={delta_inf:.2e}")
    assert delta_tcc < 1e-4, f"TCC raw loss mismatch between standalone and composite"
    assert delta_inf < 1e-4, f"InfoNCE raw loss mismatch between standalone and composite"

    # ── Backward ──────────────────────────────────────────────────────────
    print("\n── Backward pass ───────────────────────────────────────────────────")
    comp_loss.backward()
    assert emb_for_comp.grad is not None, "No gradient reached embeddings"
    assert torch.isfinite(emb_for_comp.grad).all(), "Non-finite gradients detected"
    grad_norm = emb_for_comp.grad.norm().item()
    print(f"  grad_norm = {grad_norm:.6f}  (finite: True)")

    print(f"\n{PASS} test_composite_real_data passed")


def test_composite_real_data():
    _run_composite_real_data()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the real-data composite-loss integration test."
    )
    parser.add_argument(
        "--encoder-source",
        choices=("random_init", "trained_ckpt"),
        default=_DEFAULT_ENCODER_SOURCE,
        help="Encoder weight source used to produce embeddings.",
    )
    parser.add_argument(
        "--checkpoint-run-ref",
        default=_DEFAULT_TRAINED_RUN_REF,
        help=(
            "Run alias from configs_v2/registry/runs.yaml used when "
            "--encoder-source=trained_ckpt and --checkpoint-path is not set."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help=(
            "Optional explicit checkpoint .pt path. Overrides --checkpoint-run-ref "
            "when provided."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(0)
    failures = []
    case_name = f"test_composite_real_data[{args.encoder_source}]"
    try:
        _run_composite_real_data(
            encoder_source=args.encoder_source,
            checkpoint_run_ref=args.checkpoint_run_ref,
            checkpoint_path=args.checkpoint_path,
        )
    except Exception as e:
        print(f"\n{FAIL} {case_name}: {e}")
        import traceback; traceback.print_exc()
        failures.append(case_name)

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All tests passed.")
