"""
Smoke test for TemporalInfoNCELoss.

Run from the mytcc project root:
    conda run -n fineprog python scripts/test_temporal_infonce_loss.py

Tests:
  1. Normal case: B=2, T=16, D=128 — deterministic anchors, finite loss, backward works
  2. Edge case: no negatives possible (small T, huge neg_threshold) — safe zero-like loss, no crash
  3. Stochastic anchor mode: num_anchors=4 — loss finite, correct num_sampled_anchors
  4. build_loss factory: both name aliases work with new YAML
"""

import sys
import os

# Ensure project root is importable
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from algos.loss.contrastive.loss_temporal_infonce import TemporalInfoNCELoss
from algos.loss.encoder_loss import build_loss

PASS = "[PASS]"
FAIL = "[FAIL]"

# ── Hardcoded paths / params for Test 6 (real embeddings) ─────────────────────
_REAL_EMBD_H5 = os.path.join(
    _root,
    "datasets", "embeddings",
    "TCC-robomimic_can_ph-180vid_train-resnet50_conv4c-only_bn-20260508-234455",
    "robomimic_can_ph-180vid_train-embd.h5",
)
_REAL_V2_CONFIG  = os.path.join(_root, "configs_v2", "loss", "loss_temporal_infonce.yaml")
_REAL_BATCH_SIZE = 4   # number of videos to pull from the H5
_REAL_CLIP_T     = 80  # clip each video to this many timesteps
                       # (all chosen videos have >= 80 frames)


def make_batch(B, T, device="cpu"):
    target_steps = torch.stack(
        [torch.sort(torch.randperm(50)[:T])[0] for _ in range(B)]
    ).to(device=device, dtype=torch.int64)
    seq_len = torch.full((B,), 50, device=device)
    return {"target_steps": target_steps, "seq_len": seq_len}


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Normal forward + backward, deterministic anchors
# ──────────────────────────────────────────────────────────────────────────────
def test_normal_forward():
    print("\n=== Test 1: normal forward (deterministic anchors) ===")
    B, T, D = 2, 16, 128
    cfg = {
        "pos_threshold": 0.07,   # ~7% of step range ≈ 3 frames for range~45
        "neg_threshold": 0.18,   # ~18% of step range ≈ 8 frames for range~45
        "temperature": 0.2,
        "squared_l2": True,
        "anchor_sampling": "deterministic",
    }
    loss_fn = TemporalInfoNCELoss(loss_cfg=cfg)
    emb = torch.randn(B, T, D, requires_grad=True)
    batch = make_batch(B, T)

    out = loss_fn(emb, batch)
    loss = out["loss"]
    metrics = out["metrics"]

    assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"
    assert metrics["num_valid_anchors"] > 0, "Expected some valid anchors"
    assert metrics["num_sampled_anchors"] == T, f"Deterministic: expected A={T}, got {metrics['num_sampled_anchors']}"

    loss.backward()
    assert emb.grad is not None, "Expected gradient to flow"
    assert torch.isfinite(emb.grad).all(), "Expected finite gradients"

    print(f"  loss={loss.item():.4f}  num_valid_anchors={metrics['num_valid_anchors']}  "
          f"num_sampled={metrics['num_sampled_anchors']}")
    print(f"  mean_pos_dist2={metrics['mean_pos_dist2']:.4f}  "
          f"mean_neg_dist2={metrics['mean_neg_dist2']:.4f}")
    print(f"{PASS} Test 1 passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: No-negative edge case (huge neg_threshold)
# ──────────────────────────────────────────────────────────────────────────────
def test_no_negatives():
    print("\n=== Test 2: no-negative edge case ===")
    B, T, D = 2, 4, 32
    cfg = {
        "pos_threshold": 0.02,
        "neg_threshold": 2.0,    # > 1.0 → abs_neg > step_range, impossible to satisfy
        "temperature": 0.2,
        "squared_l2": True,
        "anchor_sampling": "deterministic",
    }
    loss_fn = TemporalInfoNCELoss(loss_cfg=cfg)
    emb = torch.randn(B, T, D, requires_grad=True)
    batch = make_batch(B, T)

    out = loss_fn(emb, batch)
    loss = out["loss"]
    metrics = out["metrics"]

    # Should not crash; loss should be zero (safe fallback)
    assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
    assert float(loss.item()) == 0.0, f"Expected 0.0 fallback loss, got {loss.item()}"
    assert metrics["num_valid_anchors"] == 0, "Expected 0 valid anchors"

    loss.backward()   # must not crash even with zero loss
    print(f"  loss={loss.item()}  num_valid_anchors={metrics['num_valid_anchors']}")
    print(f"{PASS} Test 2 passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Stochastic anchor mode
# ──────────────────────────────────────────────────────────────────────────────
def test_stochastic_anchors():
    print("\n=== Test 3: stochastic anchor mode ===")
    B, T, D = 3, 20, 64
    NUM_ANCHORS = 4
    cfg = {
        "pos_threshold": 0.05,   # ~5% of step range
        "neg_threshold": 0.20,   # ~20% of step range
        "temperature": 0.2,
        "squared_l2": True,
        "anchor_sampling": "stochastic",
        "num_anchors": NUM_ANCHORS,
    }
    loss_fn = TemporalInfoNCELoss(loss_cfg=cfg)
    emb = torch.randn(B, T, D, requires_grad=True)
    batch = make_batch(B, T)

    out = loss_fn(emb, batch)
    loss = out["loss"]
    metrics = out["metrics"]

    assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
    assert metrics["num_sampled_anchors"] == NUM_ANCHORS, \
        f"Expected num_sampled_anchors={NUM_ANCHORS}, got {metrics['num_sampled_anchors']}"

    if metrics["num_valid_anchors"] > 0:
        loss.backward()
        assert emb.grad is not None
        assert torch.isfinite(emb.grad).all()

    print(f"  loss={loss.item():.4f}  num_sampled_anchors={metrics['num_sampled_anchors']}  "
          f"num_valid={metrics['num_valid_anchors']}")
    print(f"{PASS} Test 3 passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: build_loss factory with both name aliases
# ──────────────────────────────────────────────────────────────────────────────
def test_build_loss_factory():
    print("\n=== Test 4: build_loss factory (name aliases) ===")
    import pathlib
    yaml_path = str(pathlib.Path(_root) / "configs_v2" / "loss_temporal_infonce.yaml")

    for name in ("temporal_infonce", "temporal_contrastive_infonce"):
        m = build_loss(name, config_path=yaml_path)
        assert isinstance(m, TemporalInfoNCELoss), \
            f"Expected TemporalInfoNCELoss, got {type(m)}"

        B, T, D = 2, 10, 64
        emb = torch.randn(B, T, D, requires_grad=True)
        batch = make_batch(B, T)
        out = m(emb, batch)
        assert "loss" in out and "metrics" in out

        print(f"  '{name}': loss={out['loss'].item():.4f}  "
              f"num_valid={out['metrics']['num_valid_anchors']}")

    print(f"{PASS} Test 4 passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: normalize_embeddings=True
# ──────────────────────────────────────────────────────────────────────────────
def test_normalized_embeddings():
    print("\n=== Test 5: normalize_embeddings=True ===")
    B, T, D = 2, 12, 64
    cfg = {
        "pos_threshold": 0.05,
        "neg_threshold": 0.15,
        "temperature": 0.07,
        "squared_l2": False,   # also test L2 (not squared)
        "normalize_embeddings": True,
        "anchor_sampling": "deterministic",
    }
    loss_fn = TemporalInfoNCELoss(loss_cfg=cfg)
    emb = torch.randn(B, T, D, requires_grad=True)
    batch = make_batch(B, T)

    out = loss_fn(emb, batch)
    loss = out["loss"]

    assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
    if out["metrics"]["num_valid_anchors"] > 0:
        loss.backward()
        assert emb.grad is not None

    print(f"  loss={loss.item():.4f}  num_valid={out['metrics']['num_valid_anchors']}")
    print(f"{PASS} Test 5 passed")


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Real embeddings loaded from H5, loss built directly from V2 config
# ──────────────────────────────────────────────────────────────────────────────
def test_real_embeddings():
    print("\n=== Test 6: real embeddings from H5 + V2 config ===")
    import h5py

    if not os.path.exists(_REAL_EMBD_H5):
        print(f"  Skipping: H5 not found at {_REAL_EMBD_H5}")
        print(f"{PASS} Test 6 skipped (file absent)")
        return

    with h5py.File(_REAL_EMBD_H5, "r") as f:
        vids = sorted(f["videos"].keys())[:_REAL_BATCH_SIZE]
        embs_list  = []
        steps_list = []
        for vid in vids:
            raw_emb   = f[f"videos/{vid}/embeddings"][:]   # (T_v, D)
            raw_steps = f[f"videos/{vid}/target_steps"][:] # (T_v,)
            assert raw_emb.shape[0] >= _REAL_CLIP_T, (
                f"Video {vid} has only {raw_emb.shape[0]} frames, need >= {_REAL_CLIP_T}"
            )
            embs_list.append(torch.tensor(raw_emb[:_REAL_CLIP_T],   dtype=torch.float32))
            steps_list.append(torch.tensor(raw_steps[:_REAL_CLIP_T], dtype=torch.int64))

    embeddings   = torch.stack(embs_list,  dim=0).requires_grad_(True)  # [B, T, D]
    target_steps = torch.stack(steps_list, dim=0)                        # [B, T]
    seq_len      = torch.full((len(vids),), _REAL_CLIP_T, dtype=torch.int64)

    B, T, D = embeddings.shape
    print(f"  loaded: B={B}, T={T}, D={D}  videos={vids}")

    # Build loss purely from V2 config — no inline overrides
    loss_fn = TemporalInfoNCELoss(config_path=_REAL_V2_CONFIG)
    batch   = {"target_steps": target_steps, "seq_len": seq_len}
    out     = loss_fn(embeddings, batch)
    loss    = out["loss"]
    metrics = out["metrics"]

    assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
    assert metrics["num_valid_anchors"] > 0, \
        "Expected valid anchors with real data; check pos/neg thresholds vs actual target_steps"

    # ── Debug: per-video intermediate statistics ──────────────────────────────
    print("\n  [debug] ── per-video intermediate statistics ──")
    with torch.no_grad():
        anchor_idx = loss_fn._select_anchor_indices(T, embeddings.device)
        A = anchor_idx.shape[0]
        print(f"  anchor_sampling={loss_fn.anchor_sampling}  "
              f"A(num_sampled_anchors)={A}  T={T}")

        NEG_INF = torch.finfo(torch.float32).min

        for b, vid in enumerate(vids):
            z = embeddings[b].detach()          # [T, D]
            s = target_steps[b].float()         # [T]

            # Temporal thresholds
            s_min, s_max    = s.min().item(), s.max().item()
            step_range      = s_max - s_min
            abs_pos         = loss_fn.pos_threshold * step_range
            abs_neg         = loss_fn.neg_threshold * step_range

            # Anchor embeddings / steps
            z_anchor = z[anchor_idx]            # [A, D]
            s_anchor = s[anchor_idx]            # [A]

            # Gap matrix [A, T]
            gap = (s_anchor.unsqueeze(1) - s.unsqueeze(0)).abs()

            # Masks [A, T]
            pos_mask = (gap > 0) & (gap <= abs_pos)
            neg_mask = gap >= abs_neg
            valid    = pos_mask.any(dim=1) & neg_mask.any(dim=1)

            pos_counts = pos_mask.sum(dim=1)    # [A]  int
            neg_counts = neg_mask.sum(dim=1)    # [A]

            # Distances and logits [A, T]
            dist2  = loss_fn._squared_l2_dist(z_anchor, z)
            logits = -dist2 / loss_fn.temperature

            # Per-anchor InfoNCE values for valid anchors
            logits_pos  = logits.masked_fill(~pos_mask, NEG_INF)
            logits_dnom = logits.masked_fill(~(pos_mask | neg_mask), NEG_INF)
            log_num     = torch.logsumexp(logits_pos,  dim=1)   # [A]
            log_den     = torch.logsumexp(logits_dnom, dim=1)   # [A]
            per_anchor_loss = (log_den - log_num)               # [A]

            valid_loss = per_anchor_loss[valid]

            # Pick the middle anchor as a concrete example
            mid = A // 2
            mid_pos_idx = torch.where(pos_mask[mid])[0].tolist()
            mid_neg_idx_head = torch.where(neg_mask[mid])[0][:8].tolist()

            print(f"\n  [video {vid}]")
            print(f"    target_steps[:12]   = {target_steps[b, :12].tolist()}")
            print(f"    min_step={int(s_min)}  max_step={int(s_max)}  "
                  f"step_range={step_range:.1f}")
            print(f"    abs_pos={abs_pos:.3f}  abs_neg={abs_neg:.3f}  "
                  f"(pos_thr={loss_fn.pos_threshold}, neg_thr={loss_fn.neg_threshold})")
            print(f"    valid_anchors={int(valid.sum())} / {A}  "
                  f"avg_pos_per_anchor={pos_counts.float().mean():.2f}  "
                  f"avg_neg_per_anchor={neg_counts.float().mean():.2f}")
            print(f"    mean_pos_dist2={dist2[valid][pos_mask[valid]].mean():.4f}  "
                  f"mean_neg_dist2={dist2[valid][neg_mask[valid]].mean():.4f}")
            print(f"    per_anchor_loss  min={valid_loss.min():.6f}  "
                  f"max={valid_loss.max():.6f}  "
                  f"mean={valid_loss.mean():.6f}")
            print(f"    ── example anchor (index={int(anchor_idx[mid])}, "
                  f"target_step={int(s_anchor[mid])}) ──")
            print(f"      pos_indices           = {mid_pos_idx}")
            print(f"      neg_indices[:8]       = {mid_neg_idx_head}")
            print(f"      pos_count={int(pos_counts[mid])}  neg_count={int(neg_counts[mid])}")
            print(f"      max_pos_logit={logits[mid][pos_mask[mid]].max():.4f}  "
                  f"max_neg_logit={logits[mid][neg_mask[mid]].max():.4f}")
            print(f"      log_num={log_num[mid]:.6f}  log_den={log_den[mid]:.6f}  "
                  f"anchor_loss={per_anchor_loss[mid]:.6f}")

    print()
    # ─────────────────────────────────────────────────────────────────────────

    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all(), "Non-finite gradients detected"

    print(f"  loss={loss.item():.6f}  num_valid_anchors={metrics['num_valid_anchors']}  "
          f"num_sampled_anchors={metrics['num_sampled_anchors']}")
    print(f"  mean_pos_dist2={metrics['mean_pos_dist2']:.4f}  "
          f"mean_neg_dist2={metrics['mean_neg_dist2']:.4f}")
    print(f"{PASS} Test 6 passed")


if __name__ == "__main__":
    torch.manual_seed(0)
    failures = []
    for fn in (
        # test_normal_forward,
        # test_no_negatives,
        # test_stochastic_anchors,
        # test_build_loss_factory,
        # test_normalized_embeddings,
        test_real_embeddings,
    ):
        try:
            fn()
        except Exception as e:
            print(f"{FAIL} {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failures.append(fn.__name__)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All tests passed.")
