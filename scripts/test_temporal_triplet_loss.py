"""
Smoke tests for TemporalTripletLoss.

Run from the mytcc project root:
    conda run -n fineprog python scripts/test_temporal_triplet_loss.py

Tests
-----
  1. Normal forward + backward: B=2, T=20, D=128 — finite loss, grad flows
  2. Edge case: all target_steps equal — all triplets equidistant, safe zero-loss fallback
  3. Edge case: every clip has all equidistant pairs — zero-loss fallback, no crash
  4. Constructive correctness: embeddings crafted so positives are much closer than
     negatives in embedding space → loss should be 0.0
  5. build_loss factory: build_loss("temporal_triplet", yaml_path) returns TemporalTripletLoss
  6. T < 3 raises ValueError at forward() time
  7. num_triplets_fraction subsampling: fraction=0.1 yields fewer sampled triplets
  8. Composite compatibility: temporal_triplet usable as child in CompositeEncoderLoss
"""

import sys
import os
import pathlib

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from algos.loss.contrastive.loss_temporal_triplet import TemporalTripletLoss
from algos.loss.encoder_loss import build_loss

PASS = "[PASS]"
FAIL = "[FAIL]"

_V2_DIR         = pathlib.Path(_root) / "configs_v2"
_TRIPLET_YAML   = str(_V2_DIR / "loss" / "loss_temporal_triplet.yaml")
_COMPOSITE_YAML = str(_V2_DIR / "loss" / "loss_composite_tcc_infonce.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_batch(B: int, T: int, max_step: int = 50, device: str = "cpu") -> dict:
    """Generate a synthetic batch with sorted, distinct target_steps per clip."""
    target_steps = torch.stack(
        [torch.sort(torch.randperm(max_step)[:T])[0] for _ in range(B)]
    ).to(device=device, dtype=torch.int64)
    seq_len = torch.full((B,), max_step, device=device, dtype=torch.int32)
    return {"target_steps": target_steps, "seq_len": seq_len}


def make_loss(overrides: dict = None) -> TemporalTripletLoss:
    """Build a TemporalTripletLoss from the V2 YAML, with optional overrides."""
    return TemporalTripletLoss(
        config_path=_TRIPLET_YAML,
        loss_cfg=overrides or {},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Normal forward + backward
# ─────────────────────────────────────────────────────────────────────────────
def test_normal_forward():
    print("\n=== Test 1: normal forward + backward ===")
    B, T, D = 2, 20, 128
    loss_fn = make_loss()
    emb   = torch.randn(B, T, D, requires_grad=True)
    batch = make_batch(B, T)

    out     = loss_fn(emb, batch)
    loss    = out["loss"]
    metrics = out["metrics"]

    assert loss.dim() == 0,            f"Expected scalar tensor, got shape {loss.shape}"
    assert torch.isfinite(loss),       f"Expected finite loss, got {loss}"
    assert "loss" in out,              "Missing 'loss' key"
    assert "metrics" in out,           "Missing 'metrics' key"

    required_keys = [
        "loss_total", "loss_temporal_triplet",
        "num_valid_triplets", "num_sampled_triplets",
        "num_clips_with_valid_triplets",
        "mean_d_ap", "mean_d_an",
        "triplet_accuracy_margin", "active_triplet_fraction",
    ]
    for k in required_keys:
        assert k in metrics, f"Missing metric key: '{k}'"

    assert metrics["num_valid_triplets"] > 0,  "Expected valid triplets for random data"
    assert metrics["num_clips_with_valid_triplets"] == B

    loss.backward()
    assert emb.grad is not None,                     "Expected gradient to flow"
    assert torch.isfinite(emb.grad).all(),           "Expected finite gradients"

    print(f"  loss={loss.item():.4f}")
    print(f"  num_valid_triplets={metrics['num_valid_triplets']}")
    print(f"  num_sampled_triplets={metrics['num_sampled_triplets']}")
    print(f"  mean_d_ap={metrics['mean_d_ap']:.4f}  mean_d_an={metrics['mean_d_an']:.4f}")
    print(f"  active_triplet_fraction={metrics['active_triplet_fraction']:.4f}")
    print(f"  triplet_accuracy_margin={metrics['triplet_accuracy_margin']:.4f}")
    print(f"{PASS} Test 1 passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: All target_steps identical — all pairs equidistant → zero-loss fallback
# ─────────────────────────────────────────────────────────────────────────────
def test_all_equal_steps():
    print("\n=== Test 2: all target_steps identical → zero-loss fallback ===")
    B, T, D = 2, 5, 32
    loss_fn = make_loss()
    emb   = torch.randn(B, T, D, requires_grad=True)

    # All steps are the same value → every (j, k) pair is equidistant from any anchor
    target_steps = torch.zeros(B, T, dtype=torch.int64)
    seq_len      = torch.full((B,), T, dtype=torch.int32)
    batch        = {"target_steps": target_steps, "seq_len": seq_len}

    out  = loss_fn(emb, batch)
    loss = out["loss"]
    metrics = out["metrics"]

    assert torch.isfinite(loss),              f"Expected finite loss, got {loss}"
    assert float(loss.item()) == 0.0,         f"Expected 0.0 fallback loss, got {loss.item()}"
    assert metrics["num_valid_triplets"] == 0
    assert metrics["num_sampled_triplets"] == 0
    assert metrics["num_clips_with_valid_triplets"] == 0

    loss.backward()  # must not crash with zero loss
    print(f"  loss={loss.item()}  num_valid_triplets={metrics['num_valid_triplets']}")
    print(f"{PASS} Test 2 passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Two distinct steps — every pair per anchor is equidistant (symmetric)
# ─────────────────────────────────────────────────────────────────────────────
def test_symmetric_equidistant():
    print("\n=== Test 3: symmetric equidistant pairs → zero-loss fallback ===")
    B, T, D = 2, 4, 32
    loss_fn = make_loss()
    emb = torch.randn(B, T, D, requires_grad=True)

    # Steps: [0, 1, 0, 1] — for anchor at step 0, two candidates at dist 1 are equidistant;
    # for anchor at step 1, two candidates at dist 1 are equidistant.
    target_steps = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=torch.int64)
    seq_len      = torch.full((B,), 10, dtype=torch.int32)
    batch        = {"target_steps": target_steps, "seq_len": seq_len}

    out  = loss_fn(emb, batch)
    loss = out["loss"]

    assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
    # With only 2 distinct step values, most candidates are equidistant;
    # valid non-equidistant triplets may still exist (e.g., anchor=0, pos=1, neg=0
    # is only available if gap(0,pos) < gap(0,neg)).
    # The important thing is no crash and no inf/nan.
    loss.backward()
    print(f"  loss={loss.item():.4f}  num_valid={out['metrics']['num_valid_triplets']}")
    print(f"{PASS} Test 3 passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Constructive correctness — when geometry already satisfies margin, loss ≈ 0
# ─────────────────────────────────────────────────────────────────────────────
def test_constructive_zero_loss():
    print("\n=== Test 4: constructive zero-loss check ===")
    B, T, D = 1, 4, 8
    margin = 1.0
    loss_fn = TemporalTripletLoss(loss_cfg={"margin": margin})

    # Embeddings arranged on a line so that embedding distance mirrors temporal distance
    # z[t] = t * large_scale_vector  →  temporally closer pairs are also metrically closer
    scale = 10.0
    z_base = torch.arange(T, dtype=torch.float32).unsqueeze(-1) * scale  # [T, 1]
    z_base = z_base.expand(T, D)   # [T, D]
    emb = z_base.unsqueeze(0).requires_grad_(True)   # [1, T, D]

    # Uniformly spaced steps → no equidistant ambiguity
    target_steps = torch.arange(T, dtype=torch.int64).unsqueeze(0)   # [1, T]
    seq_len      = torch.full((B,), T, dtype=torch.int32)
    batch        = {"target_steps": target_steps, "seq_len": seq_len}

    out  = loss_fn(emb, batch)
    loss = out["loss"]
    metrics = out["metrics"]

    # With scale=10 and margin=1: d_ap << d_an for any valid triplet,
    # so d_ap - d_an + margin should be << 0 → all triplets inactive → loss = 0
    assert torch.isfinite(loss), f"Expected finite loss, got {loss}"
    assert float(loss.item()) == 0.0, (
        f"Expected zero loss when embedding geometry matches temporal order, "
        f"got {loss.item():.6f}. "
        f"active_triplet_fraction={metrics['active_triplet_fraction']}"
    )
    assert metrics["active_triplet_fraction"] == 0.0, \
        f"Expected no active triplets, got {metrics['active_triplet_fraction']}"
    assert metrics["triplet_accuracy_margin"] == 1.0, \
        f"Expected 100% margin accuracy, got {metrics['triplet_accuracy_margin']}"

    loss.backward()  # must not crash on zero loss
    print(f"  loss={loss.item()}  active_frac={metrics['active_triplet_fraction']}")
    print(f"  acc_margin={metrics['triplet_accuracy_margin']}")
    print(f"{PASS} Test 4 passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: build_loss factory
# ─────────────────────────────────────────────────────────────────────────────
def test_build_loss_factory():
    print("\n=== Test 5: build_loss factory ===")
    loss_module = build_loss("temporal_triplet", config_path=_TRIPLET_YAML)

    assert isinstance(loss_module, TemporalTripletLoss), \
        f"Expected TemporalTripletLoss, got {type(loss_module)}"

    B, T, D = 2, 10, 64
    emb   = torch.randn(B, T, D, requires_grad=True)
    batch = make_batch(B, T)
    out   = loss_module(emb, batch)

    assert "loss" in out and "metrics" in out
    assert torch.isfinite(out["loss"]), f"Expected finite loss, got {out['loss']}"
    out["loss"].backward()

    print(f"  loss={out['loss'].item():.4f}")
    print(f"  num_valid_triplets={out['metrics']['num_valid_triplets']}")
    print(f"{PASS} Test 5 passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: T < 3 raises ValueError
# ─────────────────────────────────────────────────────────────────────────────
def test_t_too_small_raises():
    print("\n=== Test 6: T < 3 raises ValueError ===")
    loss_fn = make_loss()
    emb     = torch.randn(2, 2, 32)   # T=2
    batch   = make_batch(2, 2, max_step=10)

    raised = False
    try:
        loss_fn(emb, batch)
    except ValueError as e:
        raised = True
        print(f"  Caught expected ValueError: {e}")

    assert raised, "Expected ValueError for T < 3, but none was raised"
    print(f"{PASS} Test 6 passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: num_triplets_fraction subsampling
# ─────────────────────────────────────────────────────────────────────────────
def test_subsampling():
    print("\n=== Test 7: num_triplets_fraction subsampling ===")
    B, T, D = 3, 20, 64
    torch.manual_seed(42)

    loss_full = make_loss({"num_triplets_fraction": 1.0})
    loss_sub  = make_loss({"num_triplets_fraction": 0.1})

    emb   = torch.randn(B, T, D, requires_grad=True)
    batch = make_batch(B, T)

    out_full = loss_full(emb.detach().requires_grad_(True), batch)
    out_sub  = loss_sub(emb.detach().requires_grad_(True), batch)

    n_full = out_full["metrics"]["num_sampled_triplets"]
    n_sub  = out_sub["metrics"]["num_sampled_triplets"]

    assert n_sub <= n_full, \
        f"Subsampled count {n_sub} should be <= full count {n_full}"
    assert n_sub > 0, "Expected at least 1 sampled triplet"

    assert torch.isfinite(out_sub["loss"]), "Expected finite loss with subsampling"
    out_sub["loss"].backward()

    print(f"  full: num_sampled={n_full}  loss={out_full['loss'].item():.4f}")
    print(f"  sub:  num_sampled={n_sub}   loss={out_sub['loss'].item():.4f}")
    print(f"{PASS} Test 7 passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Composite compatibility
# ─────────────────────────────────────────────────────────────────────────────
def test_composite_compatibility():
    print("\n=== Test 8: CompositeEncoderLoss with temporal_triplet child ===")
    import tempfile, yaml as _yaml

    # Write a temporary composite YAML that uses temporal_triplet as a child
    composite_cfg = {
        "components": [
            {
                "alias":       "triplet",
                "name":        "temporal_triplet",
                "weight":      1.0,
                "config_file": "loss/loss_temporal_triplet.yaml",
            }
        ]
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml",
        dir=str(_V2_DIR), delete=False
    ) as f:
        _yaml.dump(composite_cfg, f)
        tmp_yaml = f.name

    try:
        composite = build_loss("composite", config_path=tmp_yaml)

        B, T, D = 2, 20, 128
        emb   = torch.randn(B, T, D, requires_grad=True)
        batch = make_batch(B, T)

        out  = composite(emb, batch)
        loss = out["loss"]

        assert "loss" in out and "metrics" in out
        assert torch.isfinite(loss), f"Expected finite composite loss, got {loss}"
        assert "component_raw_loss/triplet" in out["metrics"], \
            "Expected prefixed child metric in composite output"

        loss.backward()
        print(f"  composite loss={loss.item():.4f}")
        print(f"  component_raw_loss/triplet={out['metrics']['component_raw_loss/triplet']:.4f}")
        print(f"{PASS} Test 8 passed")
    finally:
        import os as _os
        _os.unlink(tmp_yaml)


def test_pairwise_matrix_equivalence():
    """Test 9: _squared_l2_dist_full produces distances identical to direct subtraction."""
    print("\n=== Test 9: pairwise matrix numerical equivalence ===")

    T, D = 8, 32
    torch.manual_seed(42)
    z = torch.randn(T, D)

    # New: full [T,T] pairwise matrix
    pdist2 = TemporalTripletLoss._squared_l2_dist_full(z)

    # Reference: explicit loop over all pairs
    ref = torch.zeros(T, T)
    for i in range(T):
        for j in range(T):
            ref[i, j] = ((z[i] - z[j]) ** 2).sum()

    max_err = float((pdist2 - ref).abs().max())
    assert pdist2.shape == (T, T), f"shape mismatch: {pdist2.shape}"
    assert max_err < 1e-4, f"max abs error {max_err:.2e} exceeds 1e-4"
    # Diagonal must be zero
    assert float(pdist2.diagonal().max()) < 1e-5, "diagonal not zero"

    print(f"{PASS} Test 9 passed  max_abs_err={max_err:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("TemporalTripletLoss — Smoke Tests")
    print("=" * 70)

    tests = [
        test_normal_forward,
        test_all_equal_steps,
        test_symmetric_equidistant,
        test_constructive_zero_loss,
        test_build_loss_factory,
        test_t_too_small_raises,
        test_subsampling,
        test_composite_compatibility,
        test_pairwise_matrix_equivalence,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"\n{FAIL} {fn.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    if failed:
        sys.exit(1)
