"""Numerical and portable image-to-progress checks using generated tiny data."""

import csv
from pathlib import Path

import h5py
import numpy as np
import pytest

from fineprog import progress


@pytest.fixture
def expert():
    rng = np.random.default_rng(42)
    return [rng.normal(size=(30 + index * 3, 5)) + np.linspace(0, 1, 30 + index * 3)[:, None]
            for index in range(3)]


def fitted(expert, **kwargs):
    cfg = {"num_bins": 3, "pca_dim": 2, "variance_floor": 0.02,
           "temperature": 2, "ood_threshold": 0.5, **kwargs}
    model = progress.fit_gaussian(expert, cfg)
    model["calibration_bins"] = progress.calibrate(model, expert)
    return model


def test_progress_time_and_final_bin():
    np.testing.assert_array_equal(progress.temporal_progress(3, [5, 7, 9]), [0, 0.5, 1])
    np.testing.assert_array_equal(progress._bin_indices(np.array([0, 0.5, 1]), 2), [0, 1, 1])
    np.testing.assert_array_equal(progress.temporal_progress(3, [1, 1, 1]), [0, 0.5, 1])
    with pytest.raises(ValueError, match="at least two"):
        progress.temporal_progress(1)


def test_full_covariance_and_unbiased_divisor():
    trajectories = [np.array([[0, 0], [2, 4], [5, 8], [7, 12]], dtype=float)]
    model = progress.fit_gaussian(trajectories, {
        "enable_pca": False, "num_bins": 2, "variance_floor": 0.01})
    np.testing.assert_array_equal(model["bin_means"], [[1, 2], [6, 10]])
    np.testing.assert_allclose(model["bin_final_covariances"],
                              [[[2.008, 3.996], [3.996, 8.002]], [[2.008, 3.996], [3.996, 8.002]]],
                              rtol=1e-13, atol=1e-13)
    assert np.linalg.eigvalsh(model["bin_final_covariances"]).min() >= 0.01 - 1e-12
    np.testing.assert_array_equal(model["bin_counts"], [2, 2])


def test_pca_covariance_and_matching_calibration_bins(expert):
    model = fitted(expert)
    joined = np.concatenate(expert)
    np.testing.assert_allclose(model["pca_mean"], joined.mean(0), atol=1e-15)
    distances = [progress.squared_mahalanobis(model, progress.project_embeddings(model, x))
                 for x in expert]
    for index in range(3):
        expected = np.concatenate([distance[progress._bin_indices(
            progress.temporal_progress(len(x)), 3) == index, index]
            for x, distance in zip(expert, distances)])
        np.testing.assert_array_equal(model["calibration_bins"][index], expected)
    np.testing.assert_array_equal(model["bin_progress_values"], [0, 0.5, 1])


def test_conformal_strict_ties_and_hold_last_id():
    actual = progress.conformal_p_values(np.array([[1, 2], [2, 1], [9, 9]]),
                                         [np.array([1, 2, 2]), np.array([1, 2, 3])])
    np.testing.assert_array_equal(actual, [[0.75, 0.5], [0.25, 0.75], [0.25, 0.25]])
    np.testing.assert_array_equal(progress.gate_progress(
        np.array([0.1, 0.4, 0.9, 0.8]), [True, False, True, False]), [0, 0.4, 0.4, 0.8])


def test_posterior_and_online_chunk_equivalence(expert):
    model = fitted(expert)
    query = np.concatenate([expert[0][:4], np.full((1, 5), 1e4), expert[1][:3]])
    result = progress.predict_embeddings(model, query)
    logits = result["log_likelihood"] / model["temperature"]
    posterior = np.exp(logits - logits.max(1, keepdims=True))
    posterior /= posterior.sum(1, keepdims=True)
    np.testing.assert_array_equal(result["posterior"], posterior)
    np.testing.assert_allclose(posterior.sum(1), 1, atol=1e-15)
    assert result["is_ood"][4]
    predictions, last = [], 0
    for frame in query:
        prediction = progress.predict_embeddings(model, frame[None], last_progress=last)
        last = prediction["gated_progress"][0]
        predictions.append(last)
    np.testing.assert_allclose(predictions, result["gated_progress"], rtol=1e-12, atol=1e-12)


def test_portable_single_artifact_roundtrip(expert, tmp_path):
    model = fitted(expert)
    path = progress.save_progress_model(model, tmp_path / "progress.h5")
    reloaded = progress.load_progress_model(path)
    for key, values in progress.predict_embeddings(model, expert[0]).items():
        np.testing.assert_array_equal(progress.predict_embeddings(reloaded, expert[0])[key], values)
    with h5py.File(path) as file:
        assert "calibration" in file and "pca_components" in file["model"]
        assert not any("path" in name or "checkpoint" in name for name in file.attrs)


def test_reject_empty_temporal_bins():
    with pytest.raises(ValueError, match=">=2 samples"):
        progress.fit_gaussian([np.zeros((2, 3))], {"enable_pca": False, "num_bins": 3})


def test_checkpoint_hash_rejects_mismatched_weights(expert, tmp_path):
    path = tmp_path / "encoder.pt"
    path.write_bytes(b"selected local weights")
    model = fitted(expert)
    model["checkpoint_sha256"] = progress._checkpoint_sha256(path)
    progress._check_checkpoint(model, path)
    path.write_bytes(b"other local weights")
    with pytest.raises(ValueError, match="SHA256"):
        progress._check_checkpoint(model, path)


def test_h5_pipeline_and_frame_estimator_use_identical_math(monkeypatch, tmp_path):
    import torch

    class PixelEncoder(torch.nn.Module):
        def forward(self, images):
            assert images.dtype == torch.uint8
            values = images[:, :, 0, 0].float() / 255
            return torch.nn.functional.normalize(torch.cat([values, values[:, :1].square()], 1), dim=1)

    monkeypatch.setattr(progress, "_load_encoder", lambda checkpoint, device: PixelEncoder().eval())
    monkeypatch.setattr(progress, "_resolve", Path)
    monkeypatch.setattr(progress, "_checkpoint_sha256", lambda path: "0" * 64)
    rng = np.random.default_rng(9)
    h5_path = tmp_path / "rgb.h5"
    frames_by_id = {}
    with h5py.File(h5_path, "w") as file:
        for video_id in ("000001", "000002"):
            colors = rng.integers(1, 255, size=(12, 1, 1, 3), dtype=np.uint8)
            frames = np.broadcast_to(colors, (12, 224, 224, 3)).copy()
            frames_by_id[video_id] = frames
            file.create_dataset(f"videos/{video_id}/frames", data=frames)
    task = {"data": {"train": h5_path, "calibration": h5_path, "valid": h5_path}}
    cfg = {"num_bins": 2, "pca_dim": 2, "temperature": 5, "ood_threshold": 0.5, "batch_size": 4}
    model_path = progress.fit_progress(task, cfg, "unused.pt", tmp_path / "fit")
    output = progress.predict_progress_file(task, cfg, "unused.pt", model_path, "valid", tmp_path / "pred")
    estimator = progress.ProgressEstimator("unused.pt", model_path)
    with np.load(output / "000001.npz") as offline:
        for index, frame in enumerate(frames_by_id["000001"]):
            online = estimator.predict_frame(frame)
            assert online["is_ood"] == offline["is_ood"][index]
            for key in ("raw_progress", "gated_progress", "max_p", "posterior"):
                np.testing.assert_allclose(online[key], offline[key][index], rtol=1e-12, atol=1e-12)
    estimator.last_progress = 0.9
    estimator.reset()
    assert estimator.last_progress == 0
    rows = list(csv.DictReader((output / "000001.csv").open()))
    assert len(rows) == 12 and rows[0]["frame_idx"] == "0"
    assert (output / "000001.png").is_file()
