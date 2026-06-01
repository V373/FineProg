"""Regression tests for _collect_image_payload and _log_eval_to_wandb.

Run from the mytcc/ project root:
    conda run -n fineprog python scripts/test_in_training_eval_logging.py

Tests cover:
  - latent_distance_heatmap  plot_mode: heatmap
  - latent_distance_heatmap  plot_mode: anchor_distance_curves
  - latent_distance_heatmap  plot_mode: both
  - latent_distance_heatmap  missing / None image paths
  - kendalls_tau             backward compatibility (heatmap only)
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure mytcc/ is on sys.path regardless of cwd.
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from utils.in_training_eval import _collect_image_payload  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(path: str) -> str:
    """Write a minimal 1-byte file so os.path.isfile() is True."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x89PNG")
    return path


def _install_fake_wandb():
    """Inject a fake wandb module so import inside _collect_image_payload works."""
    fake = types.ModuleType("wandb")
    fake.Image = lambda path: f"<Image:{path}>"  # simple sentinel
    fake.run = MagicMock()  # non-None so the guard in _log_eval_to_wandb passes
    fake.log = MagicMock()
    sys.modules["wandb"] = fake
    return fake


# ---------------------------------------------------------------------------
# Test: heatmap-only mode
# ---------------------------------------------------------------------------

def test_heatmap_only():
    _install_fake_wandb()
    with tempfile.TemporaryDirectory() as td:
        hm = _make_png(os.path.join(td, "heatmap_0.png"))
        result = {
            "output_heatmap_path": hm,
            "output_heatmap_paths": [hm],
            "output_curve_path": None,
            "output_curve_paths": [],
        }
        payload = _collect_image_payload("latent_distance_heatmap", result)
    hm_keys = [k for k in payload if "/heatmap_" in k]
    cv_keys  = [k for k in payload if "/curve_" in k]
    assert len(hm_keys) == 1, f"Expected 1 heatmap key, got {hm_keys}"
    assert len(cv_keys) == 0, f"Expected 0 curve keys, got {cv_keys}"
    print("PASS  test_heatmap_only")


# ---------------------------------------------------------------------------
# Test: anchor_distance_curves-only mode
# ---------------------------------------------------------------------------

def test_curves_only():
    _install_fake_wandb()
    with tempfile.TemporaryDirectory() as td:
        cv = _make_png(os.path.join(td, "curve_0.png"))
        result = {
            "output_heatmap_path": None,
            "output_heatmap_paths": [],
            "output_curve_path": cv,
            "output_curve_paths": [cv],
        }
        payload = _collect_image_payload("latent_distance_heatmap", result)
    hm_keys = [k for k in payload if "/heatmap_" in k]
    cv_keys  = [k for k in payload if "/curve_" in k]
    assert len(hm_keys) == 0, f"Expected 0 heatmap keys, got {hm_keys}"
    assert len(cv_keys) == 1, f"Expected 1 curve key, got {cv_keys}"
    print("PASS  test_curves_only")


# ---------------------------------------------------------------------------
# Test: both mode
# ---------------------------------------------------------------------------

def test_both():
    _install_fake_wandb()
    with tempfile.TemporaryDirectory() as td:
        hm = _make_png(os.path.join(td, "heatmap_0.png"))
        cv = _make_png(os.path.join(td, "curve_0.png"))
        result = {
            "output_heatmap_path": hm,
            "output_heatmap_paths": [hm],
            "output_curve_path": cv,
            "output_curve_paths": [cv],
        }
        payload = _collect_image_payload("latent_distance_heatmap", result)
    hm_keys = [k for k in payload if "/heatmap_" in k]
    cv_keys  = [k for k in payload if "/curve_" in k]
    assert len(hm_keys) == 1, f"Expected 1 heatmap key, got {hm_keys}"
    assert len(cv_keys) == 1, f"Expected 1 curve key, got {cv_keys}"
    print("PASS  test_both")


# ---------------------------------------------------------------------------
# Test: all-video mode (multiple paths per family)
# ---------------------------------------------------------------------------

def test_all_videos():
    _install_fake_wandb()
    with tempfile.TemporaryDirectory() as td:
        hms = [_make_png(os.path.join(td, f"heatmap_{i}.png")) for i in range(3)]
        cvs = [_make_png(os.path.join(td, f"curve_{i}.png")) for i in range(3)]
        result = {
            "output_heatmap_path": None,
            "output_heatmap_paths": hms,
            "output_curve_path": None,
            "output_curve_paths": cvs,
        }
        payload = _collect_image_payload("latent_distance_heatmap", result)
    hm_keys = [k for k in payload if "/heatmap_" in k]
    cv_keys  = [k for k in payload if "/curve_" in k]
    assert len(hm_keys) == 3, f"Expected 3 heatmap keys, got {hm_keys}"
    assert len(cv_keys) == 3, f"Expected 3 curve keys, got {cv_keys}"
    print("PASS  test_all_videos")


# ---------------------------------------------------------------------------
# Test: missing / None paths are silently skipped
# ---------------------------------------------------------------------------

def test_missing_paths():
    _install_fake_wandb()
    result = {
        "output_heatmap_path": None,
        "output_heatmap_paths": [None, "/nonexistent/file.png"],
        "output_curve_path": None,
        "output_curve_paths": [],
    }
    payload = _collect_image_payload("latent_distance_heatmap", result)
    assert payload == {}, f"Expected empty payload for missing paths, got {payload}"
    print("PASS  test_missing_paths")


# ---------------------------------------------------------------------------
# Test: kendalls_tau backward compatibility (only heatmap key in result)
# ---------------------------------------------------------------------------

def test_kendalls_tau_compat():
    _install_fake_wandb()
    with tempfile.TemporaryDirectory() as td:
        hm = _make_png(os.path.join(td, "kendall_heatmap.png"))
        result = {
            "output_heatmap_path": hm,
            "output_heatmap_paths": [hm],
            # No curve keys — task never emits them
        }
        payload = _collect_image_payload("kendalls_tau", result)
    hm_keys = [k for k in payload if "/heatmap_" in k]
    cv_keys  = [k for k in payload if "/curve_" in k]
    assert len(hm_keys) == 1, f"Expected 1 heatmap key, got {hm_keys}"
    assert len(cv_keys) == 0, f"Expected 0 curve keys (task doesn't emit curves), got {cv_keys}"
    print("PASS  test_kendalls_tau_compat")


# ---------------------------------------------------------------------------
# Test: n_imgs count matches both image families
# ---------------------------------------------------------------------------

def test_n_imgs_count():
    """The printed summary should count both heatmap and curve images."""
    fake_wnd = _install_fake_wandb()
    mock_log_calls: list = []
    fake_wnd.log = lambda payload, step=None: mock_log_calls.append((dict(payload), step))

    from utils.in_training_eval import _log_eval_to_wandb

    with tempfile.TemporaryDirectory() as td:
        hm = _make_png(os.path.join(td, "hm.png"))
        cv = _make_png(os.path.join(td, "cv.png"))
        result = {
            "metric_name": "voc_spearman",
            "metric_value": 0.42,
            "voc_n_valid_anchors": 18,
            "voc_n_total_anchors": 20,
            "mean_offdiag_l2_distance": 1.23,
            "output_heatmap_path": hm,
            "output_heatmap_paths": [hm],
            "output_curve_path": cv,
            "output_curve_paths": [cv],
        }

        printed: list[str] = []
        import builtins
        _orig_print = builtins.print
        def _cap_print(*args, **kwargs):
            printed.append(" ".join(str(a) for a in args))
            _orig_print(*args, **kwargs)
        builtins.print = _cap_print
        try:
            _log_eval_to_wandb(
                task_name="latent_distance_heatmap",
                result=result,
                epoch=4999,
                log_images=True,
            )
        finally:
            builtins.print = _orig_print

    # Summary line must say "+ 2 image(s)" (1 heatmap + 1 curve)
    summary_lines = [l for l in printed if "Logged to wandb" in l]
    assert summary_lines, "No summary line printed"
    assert "+ 2 image(s)" in summary_lines[0], (
        f"Expected '+ 2 image(s)' in summary, got: {summary_lines[0]!r}"
    )
    # wandb.log must have been called with both image keys
    assert mock_log_calls, "wandb.log was never called"
    logged_payload = mock_log_calls[0][0]
    hm_keys = [k for k in logged_payload if "/heatmap_" in k]
    cv_keys  = [k for k in logged_payload if "/curve_" in k]
    assert len(hm_keys) == 1 and len(cv_keys) == 1, (
        f"Expected 1 heatmap + 1 curve in payload; got hm={hm_keys} cv={cv_keys}"
    )
    print("PASS  test_n_imgs_count")


# ---------------------------------------------------------------------------
# Test: video_id labels (all-mode per_video_results and single-mode selected_video_id)
# ---------------------------------------------------------------------------

def test_video_id_labels():
    """Keys should include vid<video_id> when the result dict provides video ids."""
    _install_fake_wandb()
    with tempfile.TemporaryDirectory() as td:
        hms = [_make_png(os.path.join(td, f"hm{i}.png")) for i in range(4)]
        cvs = [_make_png(os.path.join(td, f"cv{i}.png")) for i in range(4)]
        # Simulate "all" mode result with per_video_results
        per_video = [
            {"video_id": "000001"},
            {"video_id": "000002"},
            {"video_id": "000003"},
            {"video_id": "000004"},
        ]
        result = {
            "output_heatmap_paths": hms,
            "output_curve_paths":   cvs,
            "per_video_results":    per_video,
        }
        payload = _collect_image_payload("latent_distance_heatmap", result)

    expected_suffixes = ["vid000001", "vid000002", "vid000003", "vid000004"]
    for sfx in expected_suffixes:
        hm_key = f"eval/train/latent_distance_heatmap/heatmap_{sfx}"
        cv_key = f"eval/train/latent_distance_heatmap/curve_{sfx}"
        assert hm_key in payload, f"Missing expected key: {hm_key}; payload keys: {list(payload)}"
        assert cv_key in payload, f"Missing expected key: {cv_key}; payload keys: {list(payload)}"
    assert len(payload) == 8, f"Expected 8 keys total, got {len(payload)}: {list(payload)}"
    print("PASS  test_video_id_labels")


def test_single_video_id_label():
    """Single-video mode: key suffix should be vid<selected_video_id>."""
    _install_fake_wandb()
    with tempfile.TemporaryDirectory() as td:
        hm = _make_png(os.path.join(td, "hm.png"))
        cv = _make_png(os.path.join(td, "cv.png"))
        result = {
            "output_heatmap_path":  hm,
            "output_heatmap_paths": [hm],
            "output_curve_path":    cv,
            "output_curve_paths":   [cv],
            "selected_video_id":    "000003",
        }
        payload = _collect_image_payload("latent_distance_heatmap", result)

    assert "eval/train/latent_distance_heatmap/heatmap_vid000003" in payload, \
        f"Expected heatmap_vid000003; got {list(payload)}"
    assert "eval/train/latent_distance_heatmap/curve_vid000003" in payload, \
        f"Expected curve_vid000003; got {list(payload)}"
    print("PASS  test_single_video_id_label")


# ---------------------------------------------------------------------------
# Test: VOC extra fields do not disturb image-payload collection
# ---------------------------------------------------------------------------

def test_voc_extra_fields_no_img_impact():
    """New VOC diagnostic fields must not affect image payload keys or count."""
    _install_fake_wandb()
    with tempfile.TemporaryDirectory() as td:
        hm = _make_png(os.path.join(td, "hm.png"))
        cv = _make_png(os.path.join(td, "cv.png"))
        result = {
            # Primary VOC fields
            "metric_name":            "voc_spearman",
            "metric_value":           0.78,
            "voc_n_valid_anchors":    47,
            "voc_n_total_anchors":    50,
            # Legacy auxiliary field
            "mean_offdiag_l2_distance": 3.14,
            # Image paths
            "output_heatmap_path":    hm,
            "output_heatmap_paths":   [hm],
            "output_curve_path":      cv,
            "output_curve_paths":     [cv],
            "selected_video_id":      "000007",
        }
        payload = _collect_image_payload("latent_distance_heatmap", result)

    hm_keys = [k for k in payload if "/heatmap_" in k]
    cv_keys  = [k for k in payload if "/curve_" in k]
    # Exactly 1 heatmap + 1 curve, both labelled with the video id
    assert len(hm_keys) == 1, f"Expected 1 heatmap key, got {hm_keys}"
    assert len(cv_keys) == 1, f"Expected 1 curve key, got {cv_keys}"
    assert "vid000007" in hm_keys[0], f"Expected vid000007 in {hm_keys[0]}"
    assert "vid000007" in cv_keys[0], f"Expected vid000007 in {cv_keys[0]}"
    # No spurious keys from the VOC diagnostic fields
    extra_keys = [k for k in payload if "/heatmap_" not in k and "/curve_" not in k]
    assert not extra_keys, f"Unexpected payload keys from VOC fields: {extra_keys}"
    print("PASS  test_voc_extra_fields_no_img_impact")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_heatmap_only,
        test_curves_only,
        test_both,
        test_all_videos,
        test_missing_paths,
        test_kendalls_tau_compat,
        test_n_imgs_count,
        test_video_id_labels,
        test_single_video_id_label,
        test_voc_extra_fields_no_img_impact,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {t.__name__}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*48}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)

