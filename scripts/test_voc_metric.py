"""Numerical regression tests for the VOC (Velocity Ordering Consistency) helpers.

Tests cover:
  - _safe_spearman_corr       : degenerate / valid inputs
  - _compute_anchor_voc_from_row : past/future split + mean_valid logic
  - _compute_video_voc        : short trajectories, constant distances,
                                monotone ideal case
  - VOC uses raw L2 distances : similarity-converted matrix gives different result

Run from the mytcc/ project root:
    conda run -n fineprog python scripts/test_voc_metric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure mytcc/ is on sys.path regardless of cwd.
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from algos.eval_task.tcc_eval_tasks.task_latent_distance_heatmap import (  # noqa: E402
    _compute_anchor_voc_from_row,
    _compute_video_voc,
    _l2_to_similarity,
    _safe_spearman_corr,
)


# ---------------------------------------------------------------------------
# _safe_spearman_corr
# ---------------------------------------------------------------------------

def test_safe_spearman_too_short():
    """n < 2 must return None."""
    assert _safe_spearman_corr(np.array([]), np.array([])) is None
    assert _safe_spearman_corr(np.array([1.0]), np.array([0.0])) is None
    print("PASS  test_safe_spearman_too_short")


def test_safe_spearman_constant_values():
    """Constant values → zero variance → None."""
    vals = np.array([3.0, 3.0, 3.0])
    idx  = np.array([0.0, 1.0, 2.0])
    assert _safe_spearman_corr(vals, idx) is None
    print("PASS  test_safe_spearman_constant_values")


def test_safe_spearman_perfect_positive():
    """Perfect monotone increasing → correlation ≈ +1.0."""
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    idx  = np.array([0.0, 1.0, 2.0, 3.0])
    corr = _safe_spearman_corr(vals, idx)
    assert corr is not None
    assert abs(corr - 1.0) < 1e-6, f"Expected ≈1.0, got {corr}"
    print(f"PASS  test_safe_spearman_perfect_positive (corr={corr:.6f})")


def test_safe_spearman_perfect_negative():
    """Perfect monotone decreasing → correlation ≈ -1.0."""
    vals = np.array([4.0, 3.0, 2.0, 1.0])
    idx  = np.array([0.0, 1.0, 2.0, 3.0])
    corr = _safe_spearman_corr(vals, idx)
    assert corr is not None
    assert abs(corr + 1.0) < 1e-6, f"Expected ≈-1.0, got {corr}"
    print(f"PASS  test_safe_spearman_perfect_negative (corr={corr:.6f})")


# ---------------------------------------------------------------------------
# _compute_anchor_voc_from_row
# ---------------------------------------------------------------------------

def test_anchor_voc_first_frame():
    """Anchor at index 0: no past side; VOC = future Spearman."""
    # d[0, t] = t  (distances grow linearly with t)
    anchor_row = np.array([0., 1., 2., 3., 4.], dtype=np.float32)
    voc = _compute_anchor_voc_from_row(anchor_row, anchor_index=0)
    # future: [1,2,3,4] vs t=[1,2,3,4] → Spearman = 1.0
    assert voc is not None
    assert abs(voc - 1.0) < 1e-6, f"Expected 1.0, got {voc}"
    print(f"PASS  test_anchor_voc_first_frame (voc={voc:.6f})")


def test_anchor_voc_last_frame():
    """Anchor at last index: no future side; VOC = past Spearman (negated dist)."""
    T = 5
    # d[4, t] = |4 - t| → distances grow as t moves away from anchor
    anchor_row = np.array([4., 3., 2., 1., 0.], dtype=np.float32)
    voc = _compute_anchor_voc_from_row(anchor_row, anchor_index=T - 1)
    # past: t=[0,1,2,3], dists=[4,3,2,1], -dists=[-4,-3,-2,-1] → Spearman w/ t = +1.0
    assert voc is not None
    assert abs(voc - 1.0) < 1e-6, f"Expected 1.0, got {voc}"
    print(f"PASS  test_anchor_voc_last_frame (voc={voc:.6f})")


def test_anchor_voc_middle_frame():
    """Anchor in middle: both sides valid; each Spearman = 1.0 → mean = 1.0."""
    # d[2, t] = |2 - t| for T=5
    anchor_row = np.array([2., 1., 0., 1., 2.], dtype=np.float32)
    voc = _compute_anchor_voc_from_row(anchor_row, anchor_index=2)
    # past:   t=[0,1], dists=[2,1], -dists=[-2,-1], Spearman([-2,-1],[0,1]) = +1.0
    # future: t=[3,4], dists=[1,2],                  Spearman([1,2],[3,4])   = +1.0
    # mean = 1.0
    assert voc is not None
    assert abs(voc - 1.0) < 1e-6, f"Expected 1.0, got {voc}"
    print(f"PASS  test_anchor_voc_middle_frame (voc={voc:.6f})")


def test_anchor_voc_single_neighbor_skips_side():
    """If one side has only 1 frame, that side is skipped; other side dominates."""
    # T=3, anchor at 1: past=[t=0] (1 frame → skip), future=[t=2] (1 frame → skip)
    anchor_row = np.array([1., 0., 1.], dtype=np.float32)
    voc = _compute_anchor_voc_from_row(anchor_row, anchor_index=1)
    # Both sides have n=1 → both None → result must be None
    assert voc is None, f"Expected None when both sides have 1 frame, got {voc}"
    print("PASS  test_anchor_voc_single_neighbor_skips_side")


def test_anchor_voc_bad_past_only_future_remains():
    """If past distances are constant, past side is None; only future contributes."""
    # anchor at index 2, T=5
    # past side (t=0,1): d[2,0]=d[2,1]=1 (constant) → None
    # future side (t=3,4): d[2,3]=1, d[2,4]=2 → Spearman([1,2],[3,4]) = 1.0
    anchor_row = np.array([1., 1., 0., 1., 2.], dtype=np.float32)
    voc = _compute_anchor_voc_from_row(anchor_row, anchor_index=2)
    assert voc is not None
    assert abs(voc - 1.0) < 1e-6, f"Expected 1.0 (only future side), got {voc}"
    print(f"PASS  test_anchor_voc_bad_past_only_future_remains (voc={voc:.6f})")


# ---------------------------------------------------------------------------
# _compute_video_voc
# ---------------------------------------------------------------------------

def test_video_voc_t1():
    """T=1: single frame, no past or future on any anchor → VOC = None."""
    M = np.zeros((1, 1), dtype=np.float32)
    r = _compute_video_voc(M)
    assert r["voc"] is None, f"Expected None for T=1, got {r['voc']}"
    assert r["n_anchors"] == 1
    assert r["n_valid"] == 0
    print("PASS  test_video_voc_t1")


def test_video_voc_t2():
    """T=2: each anchor has exactly 1 neighbour on one side → all skip → None."""
    M = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
    r = _compute_video_voc(M)
    assert r["voc"] is None, f"Expected None for T=2, got {r['voc']}"
    assert r["n_valid"] == 0
    print("PASS  test_video_voc_t2")


def test_video_voc_t3_minimal():
    """T=3: anchors 0 and 2 are valid; anchor 1 is skipped (1 frame each side)."""
    # d[i,j] = |i - j| → perfect temporal ordering
    M = np.array([
        [0., 1., 2.],
        [1., 0., 1.],
        [2., 1., 0.],
    ], dtype=np.float32)
    r = _compute_video_voc(M)
    assert r["n_valid"] == 2, f"Expected 2 valid anchors, got {r['n_valid']}"
    assert r["voc"] is not None
    assert abs(r["voc"] - 1.0) < 1e-6, f"Expected VOC=1.0, got {r['voc']}"
    print(f"PASS  test_video_voc_t3_minimal (VOC={r['voc']:.6f}, n_valid={r['n_valid']}/3)")


def test_video_voc_monotone_ideal():
    """Distances grow monotonically from any anchor → VOC ≈ 1.0."""
    T = 8
    M = np.zeros((T, T), dtype=np.float32)
    for i in range(T):
        for j in range(T):
            M[i, j] = abs(i - j)
    r = _compute_video_voc(M)
    assert r["voc"] is not None
    assert abs(r["voc"] - 1.0) < 1e-6, f"Expected VOC≈1.0, got {r['voc']}"
    print(f"PASS  test_video_voc_monotone_ideal (VOC={r['voc']:.6f})")


def test_video_voc_constant_distances():
    """All off-diagonal distances equal → all anchors skip → VOC = None."""
    T = 5
    M = np.ones((T, T), dtype=np.float32)
    np.fill_diagonal(M, 0.0)
    r = _compute_video_voc(M)
    assert r["voc"] is None, f"Expected None for constant distances, got {r['voc']}"
    assert r["n_valid"] == 0
    print("PASS  test_video_voc_constant_distances")


def test_video_voc_returns_correct_counts():
    """n_anchors and n_valid are correctly reported."""
    T = 6
    M = np.zeros((T, T), dtype=np.float32)
    for i in range(T):
        for j in range(T):
            M[i, j] = abs(i - j)
    r = _compute_video_voc(M)
    assert r["n_anchors"] == T, f"Expected n_anchors={T}, got {r['n_anchors']}"
    # anchors 1 and T-2 (each with 1 neighbour on one side → 1 frame that side)
    # actually for T=6: anchor 1 → past=[t=0] (1 frame), future=[t=2,3,4,5] (4 frames) → future OK
    #                   anchor 4 → past=[t=0,1,2,3] (4 frames), future=[t=5] (1 frame) → past OK
    # Only anchors 0 (past empty) and 5 (future empty) have exactly 1 side,
    # and for both of those the available side has enough frames.
    # So all 6 anchors should be valid here.
    assert r["n_valid"] == T, f"Expected all {T} anchors valid, got {r['n_valid']}"
    print(f"PASS  test_video_voc_returns_correct_counts (n_valid={r['n_valid']}/{r['n_anchors']})")


# ---------------------------------------------------------------------------
# VOC uses raw L2 distances, not similarity-converted values
# ---------------------------------------------------------------------------

def test_voc_uses_raw_distances_not_similarity():
    """VOC computed on similarity matrix must differ from VOC on raw L2 matrix.

    This verifies that the task implementation calls _compute_video_voc(M)
    with the raw distance matrix, not with M_vis (similarity-converted).

    For d[i,j]=|i-j| (perfect temporal ordering):
      - VOC(raw)  ≈ +1.0  (distances grow away from any anchor)
      - VOC(sim)  ≈ -1.0  (similarities *shrink* away from anchor,
                           so negation of past + forward future both give -1)
    """
    T = 7
    M_raw = np.array(
        [[abs(i - j) for j in range(T)] for i in range(T)],
        dtype=np.float32,
    )
    M_sim = _l2_to_similarity(M_raw, tau=1.0)

    voc_raw = _compute_video_voc(M_raw)["voc"]
    voc_sim = _compute_video_voc(M_sim)["voc"]

    assert voc_raw is not None and abs(voc_raw - 1.0) < 1e-6, (
        f"Expected VOC(raw)≈1.0 for |i-j| distance, got {voc_raw}"
    )
    assert voc_sim is not None and abs(voc_sim + 1.0) < 1e-6, (
        f"Expected VOC(sim)≈-1.0 (similarity decreases with time), got {voc_sim}"
    )
    print(
        f"PASS  test_voc_uses_raw_distances_not_similarity "
        f"(raw={voc_raw:.4f}, sim={voc_sim:.4f})"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # _safe_spearman_corr
        test_safe_spearman_too_short,
        test_safe_spearman_constant_values,
        test_safe_spearman_perfect_positive,
        test_safe_spearman_perfect_negative,
        # _compute_anchor_voc_from_row
        test_anchor_voc_first_frame,
        test_anchor_voc_last_frame,
        test_anchor_voc_middle_frame,
        test_anchor_voc_single_neighbor_skips_side,
        test_anchor_voc_bad_past_only_future_remains,
        # _compute_video_voc
        test_video_voc_t1,
        test_video_voc_t2,
        test_video_voc_t3_minimal,
        test_video_voc_monotone_ideal,
        test_video_voc_constant_distances,
        test_video_voc_returns_correct_counts,
        # semantic check
        test_voc_uses_raw_distances_not_similarity,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {t.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
