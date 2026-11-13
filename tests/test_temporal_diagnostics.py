"""Offline temporal artifacts, causal masks, visualizations and isolated uploads."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

_PROJECTS = Path(__file__).resolve().parents[2]
if str(_PROJECTS) not in sys.path:
    sys.path.insert(0, str(_PROJECTS))

from fineprog.utils import temporal_diagnostics as td
from fineprog.algos.eval_task.tcc_eval_tasks import task_gaussian_progress_pred as pred


def _diagnostics(size=3, mode="learned"):
    valid = np.arange(size)[None, :] < np.arange(size)[:, None]
    lengths = valid.sum(1)
    attention = valid.astype(np.float32) / np.maximum(lengths[:, None], 1)
    if mode == "none":
        attention[:] = 0
    history_norm = np.where(lengths > 0, 0.4, 0.0)
    if mode == "none":
        history_norm[:] = 0
    return {
        "attention": attention.astype(np.float32), "attention_valid": valid,
        "history_lengths": lengths, "frame_indices": np.arange(size),
        "branch_weights": np.tile([0.1, 0.9], (size, 1)),
        "branch_difference": np.full(size, 0.3),
        "current_component_norm": np.full(size, 2.0),
        "history_component_norm": history_norm,
        "history_injection_ratio": history_norm / 2,
    }


def _write_diagnostics(group, size, mode):
    group.attrs["attention_mode"] = mode
    group.attrs["encoder_config"] = json.dumps({"embedding_dim": 4})
    for key, value in _diagnostics(size, mode).items():
        group.create_dataset(key, data=value, compression="gzip")


def _prediction(tmp_path, *, mode="learned", ids=("b", "a"), size=3):
    processed = tmp_path / "frames.h5"
    with h5py.File(processed, "w") as file:
        for video_id in ids:
            group = file.create_group(f"videos/{video_id}")
            frames = np.zeros((size, 24, 32, 3), dtype=np.uint8)
            frames[..., 0] = np.arange(size, dtype=np.uint8)[:, None, None] * 50
            frames[..., 1] = 100
            group.create_dataset("frames", data=frames)
            group.attrs["fps"] = 7.0
    prediction = tmp_path / "prediction.h5"
    with h5py.File(prediction, "w") as file:
        file.attrs["checkpoint_sha256"] = "a" * 64
        file.attrs["processed_h5_path"] = str(processed)
        file.attrs["gaussian_model_h5_path"] = "new-model.h5"
        file.attrs["training_provenance"] = json.dumps({
            "run_id": "train123", "run_name": "source-training", "project": "source-project",
            "entity": "source-entity", "epoch": 2000,
        })
        for video_id in ids:
            group = file.create_group(f"nonexperts/{video_id}")
            raw = np.linspace(0.2, 0.9, size)
            ood = np.zeros(size, dtype=bool)
            if size > 2:
                ood[1] = True
            for key, values in {
                "target_steps": np.arange(size), "progress_mean": raw,
                "progress_gated": td._hold_progress(raw, ood),
                "is_ood_raw": ood, "is_ood": ood,
                "posterior": np.stack([1 - raw, raw], axis=1),
                "keyframe_labels": np.where(np.arange(size) == size - 1, 0, -1),
            }.items():
                group.create_dataset(key, data=values)
            _write_diagnostics(group.create_group("temporal_diagnostics"), size, mode)
    return prediction


@pytest.mark.parametrize("mode", ["learned", "uniform", "none"])
def test_attention_mask_contract_and_empty_history(mode):
    arrays = td.validate_temporal_diagnostics(_diagnostics(4, mode), attention_mode=mode)
    assert arrays["history_lengths"].tolist() == [0, 1, 2, 3]
    assert not arrays["attention"][0].any()
    td.validate_temporal_diagnostics(_diagnostics(1, mode), attention_mode=mode)


@pytest.mark.parametrize("fault", ["future", "missing_mask", "row_sum", "weights", "nan", "sample_stride"])
def test_invalid_diagnostics_rejected(fault):
    data = _diagnostics()
    if fault == "future":
        data["attention_valid"][1, 2] = True
    elif fault == "missing_mask":
        del data["attention_valid"]
    elif fault == "row_sum":
        data["attention"][2, 0] = 0.1
    elif fault == "weights":
        data["branch_weights"][0] = [0.2, 0.9]
    elif fault == "nan":
        data["history_injection_ratio"][0] = np.nan
    else:
        data["frame_indices"] *= 2
    with pytest.raises(ValueError):
        td.validate_temporal_diagnostics(data)


def test_fixed_intervals_ties_weighted_median_and_empty_rows():
    row = np.array([0.1, 0.3, 0.2, 0.2, 0.2, 0.0])
    valid = np.array([True, True, True, True, True, False])
    intervals = td.summarize_history_intervals(row, valid, interval_frames=2, num_intervals=3)
    assert [(x["start"], x["end"], x["representative"]) for x in intervals] == [(0, 2, 1), (2, 4, 2), (4, 5, 4)]
    assert [x["mass"] for x in intervals] == pytest.approx([0.4, 0.4, 0.2])
    assert td.summarize_history_intervals(row * 0, valid) == []
    assert td.summarize_history_intervals(row, valid & False) == []


def test_explicit_selection_and_config_requirements():
    with pytest.raises(ValueError, match="nonempty"):
        td.parse_temporal_config({"enabled": True})
    with pytest.raises(ValueError, match="explicit list"):
        td.parse_temporal_config({"video_ids": "all"})
    cfg = td.parse_temporal_config({"enabled": True, "video_ids": ["b", "a"]})
    assert td.select_video_ids(cfg, ["a", "b", "c"]) == ["b", "a"]
    with pytest.raises(ValueError, match="missing"):
        td.select_video_ids(cfg, ["a"])
    with pytest.raises(ValueError, match="requires save_posterior"):
        pred._parse_pred_config({"gaussian_model_h5_path": "g", "nonexpert_h5_path": "q",
                                 "temporal_diagnostics": {"enabled": True, "video_ids": ["a"]}})


def test_h5_roundtrip_baseline_alignment_and_source_hashes(tmp_path):
    path = _prediction(tmp_path)
    with h5py.File(path, "r") as file:
        trajectory = td.read_prediction_trajectory(file, "a")
        metadata = td.read_prediction_metadata(file)
        np.testing.assert_array_equal(trajectory["progress_gated_raw"], [0.2, 0.2, 0.9])
        for key, value in _diagnostics().items():
            np.testing.assert_array_equal(trajectory["diagnostics"][key], value)
        baseline = td.read_prediction_trajectory(file, "a", diagnostics=False)
    baseline_meta = {**metadata, "checkpoint_sha256": "b" * 64, "gaussian_model_h5_path": "baseline.h5"}
    td.validate_baseline_alignment(trajectory, baseline, metadata, baseline_meta)
    with pytest.raises(ValueError, match="processed_h5_path"):
        td.validate_baseline_alignment(trajectory, baseline, metadata, {**baseline_meta, "processed_h5_path": "other.h5"})
    with pytest.raises(ValueError, match="target_steps"):
        td.validate_baseline_alignment(trajectory, {**baseline, "target_steps": np.array([0, 1, 4])}, metadata, baseline_meta)
    model = tmp_path / "gaussian.h5"
    calibration = tmp_path / "calibration.h5"
    for artifact in (model, calibration):
        with h5py.File(artifact, "w") as file:
            file.attrs["checkpoint_sha256"] = "a" * 64
    assert td.validate_prediction_sources(str(path), str(model), str(calibration)) == "a" * 64
    with h5py.File(calibration, "a") as file:
        file.attrs["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="mismatch"):
        td.validate_prediction_sources(str(path), str(model), str(calibration))


def test_diagnostic_copy_does_not_change_values(tmp_path):
    source_path = tmp_path / "embedding.h5"
    destination_path = tmp_path / "prediction.h5"
    with h5py.File(source_path, "w") as source, h5py.File(destination_path, "w") as destination:
        src = source.create_group("videos/a")
        _write_diagnostics(src.create_group("temporal_diagnostics"), 3, "uniform")
        dst = destination.create_group("nonexperts/a")
        td.copy_trajectory_diagnostics(src, dst, np.arange(3))
        copied = td.read_temporal_diagnostics(dst["temporal_diagnostics"])
        np.testing.assert_array_equal(copied["attention"], _diagnostics()["attention"])
        assert copied["attention_mode"] == "uniform"
        with pytest.raises(ValueError, match="target_steps mismatch"):
            td.copy_trajectory_diagnostics(src, destination.create_group("nonexperts/b"), np.arange(3) * 2)


@pytest.mark.parametrize("mode,size", [("learned", 3), ("none", 1)])
def test_small_png_and_mp4_rendering(tmp_path, mode, size):
    import cv2
    prediction = _prediction(tmp_path, mode=mode, ids=("a",), size=size)
    result = td.render_prediction_diagnostics(prediction, {"enabled": True, "video_ids": ["a"]})
    manifest = json.loads(Path(result["temporal_manifest_path"]).read_text())
    assert manifest["local_status"] == "complete"
    assert manifest["upload_status"] == "not_requested"
    assert manifest["metrics"]["real_frame_count"] == size
    assert manifest["metrics"]["nonempty_history_count"] == size - 1
    if size == 1:
        assert "history_injection_ratio_mean" not in manifest["metrics"]
    else:
        assert manifest["metrics"]["history_injection_ratio_mean"] == pytest.approx(0.2)
    entry = manifest["per_video_results"][0]
    assert Path(entry["png_path"]).stat().st_size > 1000
    capture = cv2.VideoCapture(entry["video_path"])
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == size
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(7.0)
        ok, frame = capture.read()
        assert ok and frame.shape[0] > 1000 and frame.shape[1] >= 1000
        assert frame.std() > 5
    finally:
        capture.release()


def _fake_wandb(*, fail=False):
    run = SimpleNamespace(id="evaluation123", url="https://test.invalid/eval", summary={},
                          payloads=[], finished=False)
    run.log = lambda payload: run.payloads.append(payload)
    run.finish = lambda: setattr(run, "finished", True)
    module = SimpleNamespace(run=object(), calls=[], Image=lambda path: ("image", path),
                             Video=lambda path, **kwargs: ("video", path))
    def init(**kwargs):
        module.calls.append(kwargs)
        if fail:
            raise RuntimeError("simulated offline network")
        return run
    module.init = init
    return module, run


def test_upload_is_local_first_independent_bounded_and_retry_only_reads_manifest(tmp_path, monkeypatch):
    prediction = _prediction(tmp_path)
    failing, _ = _fake_wandb(fail=True)
    monkeypatch.setitem(sys.modules, "wandb", failing)
    result = td.render_prediction_diagnostics(prediction, {
        "enabled": True, "video_ids": ["b", "a"], "save_mp4": False, "log_to_wandb": True,
    })
    path = Path(result["temporal_manifest_path"])
    failed = json.loads(path.read_text())
    assert failed["local_status"] == "complete" and failed["upload_status"] == "failed"
    assert len(failed["per_video_results"]) == 2
    assert all(Path(entry["png_path"]).is_file() for entry in failed["per_video_results"])
    monkeypatch.setattr(td, "render_temporal_png", lambda *a, **k: pytest.fail("upload must not render"))
    module, run = _fake_wandb()
    uploaded = td.upload_offline_summary(path, wandb_module=module)
    assert uploaded["upload_status"] == "complete" and run.finished
    call = module.calls[0]
    assert call["job_type"] == "eval" and call["reinit"] == "create_new"
    assert call["resume"] == "never" and call["id"] != "train123"
    assert call["project"] == "source-project" and call["entity"] == "source-entity"
    assert call["group"] == "train123"
    payload = run.payloads[0]
    media = {key: value for key, value in payload.items() if isinstance(value, tuple)}
    assert list(media) == ["temporal/b/summary"]
    assert run.summary["local_trajectory_count"] == 2


def test_video_upload_is_explicit_and_reusing_training_run_is_rejected(tmp_path):
    prediction = _prediction(tmp_path, ids=("a",))
    result = td.render_prediction_diagnostics(prediction, {"enabled": True, "video_ids": ["a"],
                                                         "save_png": False, "save_mp4": False})
    path = Path(result["temporal_manifest_path"])
    manifest = json.loads(path.read_text())
    manifest["config"]["wandb_log_video"] = True
    manifest["per_video_results"][0]["video_path"] = str(tmp_path / "saved.mp4")
    path.write_text(json.dumps(manifest))
    module, run = _fake_wandb()
    td.upload_offline_summary(path, wandb_module=module)
    assert run.payloads[0]["temporal/a/video"][0] == "video"
    polluted, existing_run = _fake_wandb()
    polluted.run = existing_run
    uploaded = td.upload_offline_summary(path, wandb_module=polluted)
    assert uploaded["upload_status"] == "failed"
    assert not existing_run.payloads and not existing_run.finished
    # A different Run object is still unsafe if its ID aliases the training run.
    aliased, aliased_run = _fake_wandb()
    aliased_run.id = "train123"
    uploaded = td.upload_offline_summary(path, wandb_module=aliased)
    assert uploaded["upload_status"] == "failed"
    assert not aliased_run.payloads and not aliased_run.finished


def test_intervention_summary_uses_the_same_selected_trajectory_limit(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "local_status": "complete", "config": {"enabled": True, "video_ids": ["a", "b"]},
        "provenance": {}, "per_video_results": [
            {"video_id": "a", "png_path": "a.png", "intervention_png_path": "a_intervention.png"},
            {"video_id": "b", "png_path": "b.png", "intervention_png_path": "b_intervention.png"},
        ],
    }))
    module, run = _fake_wandb()
    result = td.upload_offline_summary(path, wandb_module=module)
    assert result["upload_status"] == "complete"
    assert set(run.payloads[0]) == {"temporal/a/summary", "temporal/a/intervention"}


def test_raw_gating_preserved_when_final_gap_filter_changes_ood():
    # The same raw distance sequence is used in both calls; only post-episode
    # gap filling changes final OOD, never posterior or online gated progress.
    model = {"bin_means": np.array([[0.0], [1.0]]), "cholesky_factors": np.ones((2, 1, 1)),
             "bin_log_determinants": np.zeros(2), "embedding_dim": 1,
             "num_bins": 2, "bin_progress_values": np.array([0.0, 1.0])}
    query = np.array([[10.0], [10.0], [0.0], [10.0], [10.0]])
    kwargs = dict(query_embeddings=query, model=model, posterior_temperature=1.0, entropy_epsilon=1e-12,
                  calibration_distance_bins=[np.array([0.5, 1.5, 2.0])] * 2, ood_p_value_threshold=0.3)
    raw = pred._infer_one_trajectory(**kwargs)
    final = pred._infer_one_trajectory(**kwargs, enable_ood_filter=True, ood_filter_max_gap=1,
                                       ood_filter_min_ood_run=2)
    np.testing.assert_array_equal(final["progress_gated_raw"], raw["progress_gated"])
    np.testing.assert_array_equal(final["posterior"], raw["posterior"])
    assert not raw["is_ood"][2] and final["is_ood"][2]
    assert np.all(final["progress_gated"] == 0)


def test_gaussian_prediction_diagnostics_preserve_all_numeric_outputs(tmp_path, monkeypatch):
    model_path = tmp_path / "model.h5"
    query_path = tmp_path / "query.h5"
    calibration_path = tmp_path / "calibration.h5"
    with h5py.File(model_path, "w") as file:
        file.attrs.update(embedding_normalization="none", enable_pca=False,
                          input_embedding_dim=1, embedding_dim=1, checkpoint_sha256="a" * 64)
        model = file.create_group("model")
        for key, value in {
            "bin_progress_values": np.array([0., 1.]), "bin_means": np.array([[0.], [1.]]),
            "bin_independent_covariances": np.ones((2, 1, 1)), "shared_covariance": np.ones((1, 1)),
            "bin_final_covariances": np.ones((2, 1, 1)), "bin_log_determinants": np.zeros(2),
            "bin_counts": np.array([2, 2]),
        }.items():
            model.create_dataset(key, data=value)
    for path, values in ((query_path, [0., 5., 1.]), (calibration_path, [0., 0.25, 0.75, 1.])):
        with h5py.File(path, "w") as file:
            file.attrs.update(embedding_normalization="none", checkpoint_sha256="a" * 64,
                              processed_h5_path=str(tmp_path / "frames.h5"))
            group = file.create_group("videos/a")
            group.create_dataset("embeddings", data=np.array(values)[:, None])
            group.create_dataset("target_steps", data=np.arange(len(values)))
            if path == query_path:
                _write_diagnostics(group.create_group("temporal_diagnostics"), 3, "learned")
                group.create_dataset("keyframe_labels", data=np.array([-1, -1, 0]))
    counter = iter(range(2))
    def output_paths(**kwargs):
        directory = tmp_path / f"out{next(counter)}"
        return directory, directory / "pred.h5", directory / "pred.h5.tmp"
    monkeypatch.setattr(pred, "_build_output_paths", output_paths)
    outputs = []
    for enabled in (False, True):
        task = pred.GaussianProgressPredTask()
        task.configure({"gaussian_model_h5_path": str(model_path), "nonexpert_h5_path": str(query_path),
                        "calibration_h5_path": str(calibration_path), "enable_calibration": True,
                        "ood_p_value_threshold": 0.4, "posterior_temperature": 1.0,
                        "temporal_diagnostics": {"enabled": enabled, "video_ids": ["a"],
                                                 "save_png": False, "save_mp4": False}})
        outputs.append(task.evaluate())
    with h5py.File(outputs[0]["output_h5_path"], "r") as plain, h5py.File(outputs[1]["output_h5_path"], "r") as diag:
        for key in plain["nonexperts/a"]:
            np.testing.assert_array_equal(plain[f"nonexperts/a/{key}"][:], diag[f"nonexperts/a/{key}"][:])
        assert "temporal_diagnostics" not in plain["nonexperts/a"]
        assert "temporal_diagnostics" in diag["nonexperts/a"]
        assert diag.attrs["checkpoint_sha256"] == "a" * 64
        assert "progress_gated_raw" in diag["nonexperts/a"]
    assert Path(outputs[1]["temporal_manifest_path"]).is_file()


def test_attention_panels_share_current_time_axis_and_vertical_cursors(tmp_path):
    prediction = _prediction(tmp_path, ids=("a",), size=4)
    with h5py.File(prediction, "r") as file:
        data = td.read_prediction_trajectory(file, "a")
    fig, cursors, retrieval_cursors = td._overview_figure(data, None, 4.0)
    try:
        heatmap_axes = [axis for axis in fig.axes if axis.images]
        for axis, expected in zip(heatmap_axes[:2], (
            data["diagnostics"]["attention"].T,
            (data["diagnostics"]["attention"] * data["diagnostics"]["history_lengths"][:, None]).T,
        )):
            displayed = axis.images[0].get_array()
            np.testing.assert_array_equal(displayed.data, expected)
            np.testing.assert_array_equal(np.ma.getmaskarray(displayed), ~data["diagnostics"]["attention_valid"].T)
            assert axis.get_xlabel() == "Current frame t"
            assert axis.get_ylabel() == "Historical frame j"
        for cursor in cursors + retrieval_cursors:
            np.testing.assert_array_equal(cursor.get_xdata(), [0, 0])
            np.testing.assert_array_equal(cursor.get_ydata(), [0, 1])
    finally:
        td._pyplot().close(fig)


def test_phase_changes_are_events_only_between_annotated_phases():
    phases = np.array([-1, 0, 0, 1, -1, 2, 3])
    expected = [(3, "phase 0 → 1"), (6, "phase 2 → 3")]
    assert td._event_markers({"phase_labels": phases}) == expected
    assert td._event_markers({"phase_labels": phases, "keyframe_labels": np.full(7, -1)}) == expected
    keyframes = np.array([-1, -1, 4, -1, -1, -1, -1])
    assert td._event_markers({"phase_labels": phases, "keyframe_labels": keyframes}) == [(2, "event 4")]


def test_missing_fps_uses_explicit_playback_rate_without_inventing_seconds(tmp_path, monkeypatch):
    import cv2
    from matplotlib.axes import Axes
    prediction = _prediction(tmp_path, ids=("a",), size=2)
    with h5py.File(tmp_path / "frames.h5", "a") as file:
        del file["videos/a"].attrs["fps"]
        assert td._source_fps(file["videos/a"], "a") is None
    titles, captions = [], []
    original_title, original_put_text = Axes.set_title, cv2.putText
    def record_title(self, label, *args, **kwargs):
        titles.append(label)
        return original_title(self, label, *args, **kwargs)
    def record_caption(image, text, *args, **kwargs):
        captions.append(text)
        return original_put_text(image, text, *args, **kwargs)
    monkeypatch.setattr(Axes, "set_title", record_title)
    monkeypatch.setattr(cv2, "putText", record_caption)
    result = td.render_prediction_diagnostics(prediction, {"enabled": True, "video_ids": ["a"], "save_png": False})
    manifest = json.loads(Path(result["temporal_manifest_path"]).read_text())
    entry = manifest["per_video_results"][0]
    assert entry["source_fps"] is None and entry["playback_fps"] == 10.0
    assert any("source FPS unknown, playback only: 10 fps" in title for title in titles)
    assert "Current frame 1" in captions
    assert "frame 0" in captions
    assert not any("0.00s" in text or "0.10s" in text for text in titles + captions)
    capture = cv2.VideoCapture(entry["video_path"])
    try:
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0)
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 2
    finally:
        capture.release()


def test_old_prediction_never_borrows_identity_from_a_mutable_embedding_path(tmp_path):
    embedding = tmp_path / "overwritten_embedding.h5"
    prediction = tmp_path / "old_prediction.h5"
    with h5py.File(embedding, "w") as file:
        file.attrs.update(checkpoint_sha256="b" * 64, checkpoint_path="new_checkpoint.pt",
                          encoder_config=json.dumps({"embedding_dim": 999}),
                          training_provenance=json.dumps({"run_id": "new_training_run"}),
                          processed_h5_path="source_frames.h5")
    with h5py.File(prediction, "w") as file:
        file.attrs["nonexpert_h5_path"] = str(embedding)
    with h5py.File(prediction, "r") as file:
        metadata = td.read_prediction_metadata(file)
    assert metadata["checkpoint_provenance_status"] == "unverified"
    assert "checkpoint_sha256" not in metadata and "checkpoint_path" not in metadata
    assert "encoder_config" not in metadata and "training_provenance" not in metadata
    assert metadata["processed_h5_path"] == "source_frames.h5"
    assert metadata["processed_h5_path_source"] == "inferred_from_nonexpert_h5"
    with h5py.File(prediction, "a") as file:
        file.attrs["checkpoint_sha256"] = "a" * 64
        file.attrs["encoder_config"] = json.dumps({"embedding_dim": 4})
    with h5py.File(prediction, "r") as file:
        pinned = td.read_prediction_metadata(file)
    assert pinned["checkpoint_sha256"] == "a" * 64
    assert pinned["encoder_config"]["embedding_dim"] == 4
    assert "training_provenance" not in pinned


def test_diagnostic_subgroup_hash_cannot_disagree_with_query_root(tmp_path):
    with h5py.File(tmp_path / "query.h5", "w") as source, h5py.File(tmp_path / "pred.h5", "w") as output:
        source.attrs["checkpoint_sha256"] = "a" * 64
        video = source.create_group("videos/a")
        group = video.create_group("temporal_diagnostics")
        _write_diagnostics(group, 3, "learned")
        group.attrs["checkpoint_sha256"] = "b" * 64
        with pytest.raises(ValueError, match="subgroup checkpoint_sha256"):
            td.copy_trajectory_diagnostics(video, output.create_group("bad"), np.arange(3))
        group.attrs["checkpoint_sha256"] = "a" * 64
        td.copy_trajectory_diagnostics(video, output.create_group("good"), np.arange(3))
        assert output["good/temporal_diagnostics"].attrs["checkpoint_sha256"] == "a" * 64


def test_baseline_is_verified_against_migrated_short_checkpoint_when_pinned(tmp_path):
    current = {"video_id": "a", "target_steps": np.arange(3)}
    baseline = dict(current)
    source_path = str(tmp_path / "frames.h5")
    current_metadata = {"processed_h5_path": source_path, "encoder_config": {
        "memory_config": {"source_checkpoint_sha256": "a" * 64},
    }}
    baseline_metadata = {"processed_h5_path": source_path, "checkpoint_sha256": "b" * 64}
    with pytest.raises(ValueError, match="source_checkpoint_sha256"):
        td.validate_baseline_alignment(current, baseline, current_metadata, baseline_metadata)
    baseline_metadata["checkpoint_sha256"] = "a" * 64
    td.validate_baseline_alignment(current, baseline, current_metadata, baseline_metadata)
    assert baseline_metadata["source_checkpoint_verification"] == "verified"
    del baseline_metadata["checkpoint_sha256"]
    td.validate_baseline_alignment(current, baseline, current_metadata, baseline_metadata)
    assert baseline_metadata["source_checkpoint_verification"] == "unverified"
    assert baseline["source_checkpoint_verification"] == "unverified"
    assert "checkpoint_sha256" not in baseline_metadata
