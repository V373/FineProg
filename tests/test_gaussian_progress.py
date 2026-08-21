from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

_PROJECTS_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECTS_ROOT))

from fineprog.algos.eval_task.base_task import build_task
from fineprog.algos.eval_task.tcc_eval_tasks import task_gaussian_progress_pred
from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_pred import (
    GaussianProgressPredTask,
    _apply_model_pca,
    _build_confidence_figure,
    _build_progress_figure,
    _compute_conformal_p_values,
    _compute_gaussian_log_likelihood,
    _compute_progress_gated,
    _compute_progress_posterior,
    _compute_squared_mahalanobis,
    _infer_one_trajectory,
    _parse_pred_config,
    _processed_h5_path_from_embedding,
    _read_gaussian_model,
    _resolve_raw_video_path,
    _save_confidence_figure,
    _save_prediction_video,
    _save_progress_figure,
)
from fineprog.utils.config_v2 import ConfigV2


def _write_gaussian_model(
    path: Path,
    means: np.ndarray,
    final_covariances: np.ndarray,
    progress_values: np.ndarray | None = None,
    log_determinants: np.ndarray | None = None,
    embedding_normalization: str | None = None,
    enable_pca: bool | None = None,
    input_embedding_dim: int | None = None,
    pca_mean: np.ndarray | None = None,
    pca_components: np.ndarray | None = None,
) -> None:
    means = np.asarray(means, dtype=np.float64)
    final_covariances = np.asarray(final_covariances, dtype=np.float64)
    num_bins, embedding_dim = means.shape
    if progress_values is None:
        progress_values = np.linspace(0.0, 1.0, num_bins, dtype=np.float64)
    if log_determinants is None:
        log_determinants = np.linalg.slogdet(final_covariances)[1]

    with h5py.File(path, "w") as model_file:
        if embedding_normalization is not None:
            model_file.attrs["embedding_normalization"] = embedding_normalization
        if enable_pca is not None:
            model_file.attrs["enable_pca"] = enable_pca
            model_file.attrs["embedding_dim"] = embedding_dim
        if input_embedding_dim is not None:
            model_file.attrs["input_embedding_dim"] = input_embedding_dim
        model_group = model_file.create_group("model")
        model_group.create_dataset(
            "bin_progress_values",
            data=np.asarray(progress_values, dtype=np.float64),
        )
        model_group.create_dataset("bin_means", data=means)
        # These three datasets are part of the fitted-model schema but online
        # inference must not use them to rebuild the final covariance.
        model_group.create_dataset(
            "bin_independent_covariances",
            data=np.stack(
                [
                    (10.0 + bin_index) * np.eye(embedding_dim)
                    for bin_index in range(num_bins)
                ]
            ),
        )
        model_group.create_dataset(
            "shared_covariance",
            data=20.0 * np.eye(embedding_dim),
        )
        model_group.create_dataset(
            "bin_final_covariances",
            data=final_covariances,
        )
        model_group.create_dataset(
            "bin_log_determinants",
            data=np.asarray(log_determinants, dtype=np.float64),
        )
        model_group.create_dataset(
            "bin_counts",
            data=np.full(num_bins, 2, dtype=np.int64),
        )
        if pca_mean is not None:
            model_group.create_dataset("pca_mean", data=pca_mean)
        if pca_components is not None:
            model_group.create_dataset("pca_components", data=pca_components)


def _write_nonexpert_h5(
    path: Path,
    records: dict[str, tuple[np.ndarray, np.ndarray]],
    embedding_normalization: str | None = None,
) -> None:
    with h5py.File(path, "w") as nonexpert_file:
        if embedding_normalization is not None:
            nonexpert_file.attrs["embedding_normalization"] = embedding_normalization
        videos_group = nonexpert_file.create_group("videos")
        for video_id, (embeddings, target_steps) in records.items():
            video_group = videos_group.create_group(video_id)
            video_group.create_dataset("embeddings", data=embeddings)
            video_group.create_dataset("target_steps", data=target_steps)


def _identity_model(tmp_path: Path) -> tuple[Path, dict]:
    model_path = tmp_path / "model.h5"
    means = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    covariances = np.stack([np.eye(2), np.eye(2)])
    _write_gaussian_model(model_path, means, covariances)
    return model_path, _read_gaussian_model(str(model_path))


def test_config_v2_and_task_factory_registration():
    resolved = ConfigV2().load_eval("gaussian_progress_pred")
    assert set(resolved) == {
        "gaussian_model_h5_path",
        "nonexpert_h5_path",
        "posterior_temperature",
        "entropy_epsilon",
        "enable_calibration",
        "calibration_h5_path",
        "ood_p_value_threshold",
        "save_posterior",
        "enable_visualization",
        "enable_video",
    }
    assert Path(resolved["gaussian_model_h5_path"]).is_absolute()
    assert Path(resolved["nonexpert_h5_path"]).is_absolute()
    assert Path(resolved["calibration_h5_path"]).is_absolute()
    assert Path(resolved["calibration_h5_path"]).name == (
        "fruit_expert_videos-20vid_valid-embd.h5"
    )
    parsed = _parse_pred_config(resolved)
    assert np.isfinite(parsed["posterior_temperature"])
    assert parsed["posterior_temperature"] > 0.0
    assert parsed["entropy_epsilon"] == 1.0e-12
    assert parsed["enable_calibration"] is True
    assert parsed["calibration_h5_path"] == resolved["calibration_h5_path"]
    assert parsed["ood_p_value_threshold"] == 0.3
    assert resolved["save_posterior"] is True
    assert resolved["enable_visualization"] is True
    assert resolved["enable_video"] is True
    defaults = _parse_pred_config(
        {
            "gaussian_model_h5_path": "model.h5",
            "nonexpert_h5_path": "query.h5",
        }
    )
    assert not defaults["enable_visualization"]
    assert not defaults["enable_video"]
    assert isinstance(build_task("gaussian_progress_pred"), GaussianProgressPredTask)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("posterior_temperature", 0.0),
        ("posterior_temperature", -1.0),
        ("posterior_temperature", np.nan),
        ("entropy_epsilon", 0.0),
        ("entropy_epsilon", np.inf),
        ("enable_calibration", "false"),
        ("save_posterior", "true"),
        ("enable_visualization", "true"),
        ("enable_video", "true"),
    ],
)
def test_invalid_prediction_config_is_rejected(key, value):
    config = {
        "gaussian_model_h5_path": "model.h5",
        "nonexpert_h5_path": "query.h5",
        key: value,
    }
    with pytest.raises(ValueError):
        _parse_pred_config(config)


def test_video_config_requires_visualization():
    with pytest.raises(ValueError, match="enable_visualization=true"):
        _parse_pred_config(
            {
                "gaussian_model_h5_path": "model.h5",
                "nonexpert_h5_path": "query.h5",
                "enable_video": True,
            }
        )

    parsed = _parse_pred_config(
        {
            "gaussian_model_h5_path": "model.h5",
            "nonexpert_h5_path": "query.h5",
            "enable_calibration": True,
            "calibration_h5_path": "calibration.h5",
            "ood_p_value_threshold": 0.5,
            "enable_visualization": True,
            "enable_video": True,
        }
    )
    assert parsed["enable_visualization"] is True
    assert parsed["enable_video"] is True


def test_calibration_config_requires_path_only_when_enabled():
    with pytest.raises(ValueError, match="calibration_h5_path is required"):
        _parse_pred_config(
            {
                "gaussian_model_h5_path": "model.h5",
                "nonexpert_h5_path": "query.h5",
                "enable_calibration": True,
            }
        )

    parsed = _parse_pred_config(
        {
            "gaussian_model_h5_path": "model.h5",
            "nonexpert_h5_path": "query.h5",
            "enable_calibration": True,
            "calibration_h5_path": "calibration.h5",
            "ood_p_value_threshold": 0.5,
        }
    )
    assert parsed["enable_calibration"] is True
    assert parsed["calibration_h5_path"] == "calibration.h5"
    assert parsed["ood_p_value_threshold"] == 0.5

    parsed = _parse_pred_config(
        {
            "gaussian_model_h5_path": "model.h5",
            "nonexpert_h5_path": "query.h5",
            "enable_calibration": False,
        }
    )
    assert parsed["enable_calibration"] is False
    assert parsed["calibration_h5_path"] is None
    assert parsed["ood_p_value_threshold"] is None


def test_calibration_requires_valid_ood_p_value_threshold():
    config = {
        "gaussian_model_h5_path": "model.h5",
        "nonexpert_h5_path": "query.h5",
        "enable_calibration": True,
        "calibration_h5_path": "calibration.h5",
    }
    with pytest.raises(ValueError, match="ood_p_value_threshold is required"):
        _parse_pred_config(config)

    for threshold in (0.0, 1.0, -0.1, np.nan, np.inf, "invalid"):
        with pytest.raises(ValueError, match="ood_p_value_threshold"):
            _parse_pred_config({**config, "ood_p_value_threshold": threshold})


def test_visualization_requires_saved_posterior_and_calibration():
    with pytest.raises(ValueError, match="requires save_posterior=true"):
        _parse_pred_config(
            {
                "gaussian_model_h5_path": "model.h5",
                "nonexpert_h5_path": "query.h5",
                "save_posterior": False,
                "enable_visualization": True,
            }
        )
    with pytest.raises(ValueError, match="requires enable_calibration=true"):
        _parse_pred_config(
            {
                "gaussian_model_h5_path": "model.h5",
                "nonexpert_h5_path": "query.h5",
                "enable_visualization": True,
            }
        )


def test_model_shapes_and_cholesky_are_loaded(tmp_path):
    model_path, model = _identity_model(tmp_path)
    assert model_path.is_file()
    assert model["num_bins"] == 2
    assert model["enable_pca"] is False
    assert model["input_embedding_dim"] == 2
    assert model["embedding_dim"] == 2
    assert model["pca_mean"] is None
    assert model["pca_components"] is None
    assert model["bin_progress_values"].shape == (2,)
    assert model["bin_means"].shape == (2, 2)
    assert model["bin_final_covariances"].shape == (2, 2, 2)
    assert model["bin_log_determinants"].shape == (2,)
    np.testing.assert_allclose(
        model["cholesky_factors"],
        np.stack([np.eye(2), np.eye(2)]),
    )


def test_pca_model_is_loaded_and_applied_with_saved_parameters(tmp_path):
    model_path = tmp_path / "pca_model.h5"
    pca_mean = np.array([1.0, 2.0, 3.0])
    pca_components = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    _write_gaussian_model(
        model_path,
        means=np.array([[0.0, 0.0], [4.0, 2.0]]),
        final_covariances=np.stack([np.eye(2), np.eye(2)]),
        enable_pca=True,
        input_embedding_dim=3,
        pca_mean=pca_mean,
        pca_components=pca_components,
    )

    model = _read_gaussian_model(str(model_path))
    assert model["enable_pca"] is True
    assert model["input_embedding_dim"] == 3
    assert model["embedding_dim"] == 2
    np.testing.assert_allclose(model["pca_mean"], pca_mean)
    np.testing.assert_allclose(model["pca_components"], pca_components)

    raw_query = np.array([[5.0, 4.0, 100.0], [1.0, 2.0, 3.0]])
    projected_query = _apply_model_pca(raw_query, model)
    np.testing.assert_allclose(projected_query, [[4.0, 2.0], [0.0, 0.0]])
    assert projected_query.dtype == np.dtype("float64")


def test_pca_model_schema_inconsistencies_are_rejected(tmp_path):
    means = np.array([[0.0, 0.0], [4.0, 2.0]])
    covariances = np.stack([np.eye(2), np.eye(2)])
    pca_mean = np.array([1.0, 2.0, 3.0])
    pca_components = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    missing_component_path = tmp_path / "missing_component.h5"
    _write_gaussian_model(
        missing_component_path,
        means,
        covariances,
        enable_pca=True,
        input_embedding_dim=3,
        pca_mean=pca_mean,
    )
    with pytest.raises(ValueError, match="pca_components"):
        _read_gaussian_model(str(missing_component_path))

    bad_shape_path = tmp_path / "bad_shape.h5"
    _write_gaussian_model(
        bad_shape_path,
        means,
        covariances,
        enable_pca=True,
        input_embedding_dim=3,
        pca_mean=pca_mean,
        pca_components=np.eye(2),
    )
    with pytest.raises(ValueError, match="pca_components must have shape"):
        _read_gaussian_model(str(bad_shape_path))

    nonfinite_path = tmp_path / "nonfinite_pca.h5"
    _write_gaussian_model(
        nonfinite_path,
        means,
        covariances,
        enable_pca=True,
        input_embedding_dim=3,
        pca_mean=np.array([1.0, np.nan, 3.0]),
        pca_components=pca_components,
    )
    with pytest.raises(ValueError, match="pca_mean contains NaN or Inf"):
        _read_gaussian_model(str(nonfinite_path))

    disabled_with_pca_path = tmp_path / "disabled_with_pca.h5"
    _write_gaussian_model(
        disabled_with_pca_path,
        means,
        covariances,
        enable_pca=False,
        input_embedding_dim=2,
        pca_mean=pca_mean,
        pca_components=pca_components,
    )
    with pytest.raises(ValueError, match="enable_pca=false"):
        _read_gaussian_model(str(disabled_with_pca_path))

    missing_input_dim_path = tmp_path / "missing_input_dim.h5"
    _write_gaussian_model(
        missing_input_dim_path,
        means,
        covariances,
        enable_pca=True,
        pca_mean=pca_mean,
        pca_components=pca_components,
    )
    with pytest.raises(ValueError, match="input_embedding_dim"):
        _read_gaussian_model(str(missing_input_dim_path))


def test_model_validation_rejects_missing_dataset_and_non_pd_covariance(tmp_path):
    missing_path = tmp_path / "missing.h5"
    means = np.array([[0.0, 0.0], [1.0, 0.0]])
    covariances = np.stack([np.eye(2), np.eye(2)])
    _write_gaussian_model(missing_path, means, covariances)
    with h5py.File(missing_path, "a") as model_file:
        del model_file["model/bin_counts"]
    with pytest.raises(ValueError, match="bin_counts"):
        _read_gaussian_model(str(missing_path))

    non_pd_path = tmp_path / "non_pd.h5"
    non_pd_covariances = np.stack([np.eye(2), np.array([[1.0, 2.0], [2.0, 1.0]])])
    _write_gaussian_model(
        non_pd_path,
        means,
        non_pd_covariances,
        log_determinants=np.zeros(2),
    )
    with pytest.raises(ValueError, match="Cholesky decomposition failed"):
        _read_gaussian_model(str(non_pd_path))


def test_mahalanobis_log_likelihood_and_posterior_math(tmp_path):
    _, model = _identity_model(tmp_path)
    query = np.array([[0.1, 0.0], [4.0, 0.0]], dtype=np.float64)

    squared_mahalanobis = _compute_squared_mahalanobis(
        query,
        model["bin_means"],
        model["cholesky_factors"],
    )
    np.testing.assert_allclose(
        squared_mahalanobis,
        [[0.01, 15.21], [16.0, 0.0]],
    )
    assert np.isfinite(squared_mahalanobis).all()
    assert np.all(squared_mahalanobis >= 0.0)

    log_likelihood = _compute_gaussian_log_likelihood(
        squared_mahalanobis,
        model["bin_log_determinants"],
        model["embedding_dim"],
    )
    expected = -0.5 * (squared_mahalanobis + 2.0 * np.log(2.0 * np.pi))
    np.testing.assert_allclose(log_likelihood, expected)

    posterior = _compute_progress_posterior(log_likelihood, 1.0)
    assert np.all(posterior >= 0.0)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
    assert posterior[0, 0] > posterior[0, 1]
    assert posterior[1, 1] > posterior[1, 0]


def test_conformal_p_value_uses_strict_less_than():
    squared_mahalanobis = np.array([[2.0, 3.0], [5.0, 0.0]], dtype=np.float64)
    calibration_distance_bins = [
        np.array([2.0, 5.0], dtype=np.float64),
        np.array([1.0, 3.0, 3.0], dtype=np.float64),
    ]

    conformal_p_value = _compute_conformal_p_values(
        squared_mahalanobis,
        calibration_distance_bins,
    )

    np.testing.assert_allclose(
        conformal_p_value,
        [[2.0 / 3.0, 1.0 / 4.0], [1.0 / 3.0, 1.0]],
    )
    assert conformal_p_value.dtype == np.dtype("float64")


def test_progress_gated_holds_last_in_distribution_progress():
    progress_mean = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    is_ood = np.array([True, True, False, True, True, False, True])

    progress_gated = _compute_progress_gated(progress_mean, is_ood)

    np.testing.assert_allclose(
        progress_gated,
        [0.0, 0.0, 0.3, 0.3, 0.3, 0.6, 0.6],
    )


def test_saved_log_determinants_control_the_posterior(tmp_path):
    model_path = tmp_path / "saved_logdet.h5"
    means = np.zeros((2, 2), dtype=np.float64)
    covariances = np.stack([np.eye(2), np.eye(2)])
    _write_gaussian_model(
        model_path,
        means,
        covariances,
        log_determinants=np.array([0.0, 4.0]),
    )
    model = _read_gaussian_model(str(model_path))
    inferred = _infer_one_trajectory(
        np.zeros((1, 2)),
        model,
        posterior_temperature=1.0,
        entropy_epsilon=1.0e-12,
    )

    # Equal means and final covariances would produce 0.5/0.5 if the online
    # implementation recomputed determinants instead of using the saved values.
    assert inferred["posterior"][0, 0] == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))
    np.testing.assert_allclose(
        inferred["gaussian_log_likelihood"],
        [[-np.log(2.0 * np.pi), -np.log(2.0 * np.pi) - 2.0]],
    )
    assert inferred["map_bin"][0] == 0


def test_near_mean_prediction_and_far_query_support(tmp_path):
    _, model = _identity_model(tmp_path)
    queries = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [100.0, 100.0],
        ]
    )
    inferred = _infer_one_trajectory(
        queries,
        model,
        posterior_temperature=1.0,
        entropy_epsilon=1.0e-12,
    )

    assert inferred["map_bin"].tolist()[:2] == [0, 1]
    assert inferred["posterior"][0, 0] > inferred["posterior"][0, 1]
    assert inferred["posterior"][1, 1] > inferred["posterior"][1, 0]
    assert inferred["min_mahalanobis_sq"][2] > max(inferred["min_mahalanobis_sq"][:2])
    assert np.all(inferred["progress_mean"] >= 0.0)
    assert np.all(inferred["progress_mean"] <= 1.0)


def test_end_to_end_h5_schema_and_frame_weighted_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(task_gaussian_progress_pred, "_PROJ_ROOT", tmp_path)
    model_path, model = _identity_model(tmp_path)
    nonexpert_path = tmp_path / "query.h5"
    records = {
        "000002": (
            np.array([[4.0, 0.0], [3.5, 0.0], [100.0, 100.0]]),
            np.array([0, 2, 4], dtype=np.int64),
        ),
        "000001": (
            np.array([[0.0, 0.0], [0.5, 0.0]]),
            np.array([1, 3], dtype=np.int64),
        ),
    }
    _write_nonexpert_h5(nonexpert_path, records)

    task = GaussianProgressPredTask()
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(nonexpert_path),
            "posterior_temperature": 1.0,
            "entropy_epsilon": 1.0e-12,
            "enable_calibration": False,
            "save_posterior": True,
        }
    )
    result = task.evaluate(None)

    assert set(result) == {
        "task_name",
        "metric_name",
        "metric_value",
        "output_h5_path",
    }
    assert result["task_name"] == "gaussian_progress_pred"
    assert result["metric_name"] == "global_mean_min_mahalanobis_sq"
    output_path = Path(result["output_h5_path"])
    assert output_path.is_file()
    assert output_path.name.startswith("gaussian_progress_pred-")
    assert output_path.parent.parent.name == nonexpert_path.stem
    assert output_path.parent.parent.parent.name == model_path.stem
    assert not list(output_path.parent.glob("*.tmp"))

    expected_video_datasets = {
        "target_steps",
        "progress_label",
        "progress_mean",
        "progress_variance",
        "posterior_entropy",
        "normalized_posterior_entropy",
        "map_bin",
        "map_progress",
        "min_mahalanobis_sq",
        "nearest_mahalanobis_bin",
        "gaussian_log_likelihood",
        "posterior",
    }
    support_sum = 0.0
    total_steps = 0
    with h5py.File(output_path, "r") as output_file:
        assert set(output_file.keys()) == {"model", "nonexperts"}
        assert output_file.attrs["task_name"] == "gaussian_progress_pred"
        assert output_file.attrs["embedding_normalization"] == "none"
        assert output_file.attrs["num_bins"] == 2
        assert not bool(output_file.attrs["enable_pca"])
        assert output_file.attrs["input_embedding_dim"] == 2
        assert output_file.attrs["embedding_dim"] == 2
        assert not bool(output_file.attrs["enable_calibration"])
        assert "ood_p_value_threshold" not in output_file.attrs
        assert bool(output_file.attrs["save_posterior"])
        assert "backward_delta_threshold" not in output_file.attrs

        output_model = output_file["model"]
        assert set(output_model.keys()) == {
            "bin_progress_values",
            "bin_means",
            "bin_final_covariances",
            "bin_log_determinants",
        }
        assert output_model["bin_means"].compression == "gzip"
        assert output_model["bin_final_covariances"].compression == "gzip"
        np.testing.assert_allclose(
            output_model["bin_final_covariances"][:],
            model["bin_final_covariances"],
        )

        assert list(output_file["nonexperts"].keys()) == ["000001", "000002"]
        for video_id, (embeddings, target_steps) in records.items():
            video_group = output_file["nonexperts"][video_id]
            assert set(video_group.keys()) == expected_video_datasets
            num_steps = embeddings.shape[0]
            for dataset_name in expected_video_datasets - {
                "posterior",
                "gaussian_log_likelihood",
            }:
                assert video_group[dataset_name].shape == (num_steps,)
            assert video_group["posterior"].shape == (num_steps, 2)
            assert video_group["posterior"].compression == "gzip"
            assert video_group["gaussian_log_likelihood"].shape == (num_steps, 2)
            assert video_group["gaussian_log_likelihood"].compression == "gzip"
            assert video_group["progress_mean"].dtype == np.dtype("float64")
            assert video_group["map_bin"].dtype == np.dtype("int64")
            np.testing.assert_array_equal(video_group["target_steps"][:], target_steps)
            np.testing.assert_allclose(
                video_group["progress_label"][:],
                video_group["progress_mean"][:],
            )
            np.testing.assert_allclose(
                video_group["posterior"][:].sum(axis=1),
                1.0,
            )
            assert "progress_delta" not in video_group
            assert "backward_progress" not in video_group
            support_sum += float(video_group["min_mahalanobis_sq"][:].sum())
            total_steps += num_steps

    assert result["metric_value"] == pytest.approx(support_sum / total_steps)


def test_calibration_end_to_end_uses_matching_gaussian_column(tmp_path, monkeypatch):
    monkeypatch.setattr(task_gaussian_progress_pred, "_PROJ_ROOT", tmp_path)
    model_path = tmp_path / "model.h5"
    _write_gaussian_model(
        model_path,
        means=np.array([[0.0, 0.0], [10.0, 0.0]]),
        final_covariances=np.stack([np.eye(2), np.eye(2)]),
    )
    calibration_path = tmp_path / "calibration.h5"
    _write_nonexpert_h5(
        calibration_path,
        {
            "calibration_video": (
                np.array(
                    [
                        [9.0, 0.0],
                        [8.0, 0.0],
                        [1.0, 0.0],
                        [2.0, 0.0],
                        [3.0, 0.0],
                    ]
                ),
                np.arange(5),
            )
        },
    )
    nonexpert_path = tmp_path / "query.h5"
    _write_nonexpert_h5(
        nonexpert_path,
        {
            "video": (
                np.array(
                    [
                        [9.0, 0.0],
                        [8.0, 0.0],
                        [1.0, 0.0],
                        [2.0, 0.0],
                        [3.0, 0.0],
                        [0.0, 0.0],
                        [18.0, 0.0],
                        [19.0, 0.0],
                    ]
                ),
                np.arange(8),
            )
        },
    )

    task = GaussianProgressPredTask()
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(nonexpert_path),
            "enable_calibration": True,
            "calibration_h5_path": str(calibration_path),
            "ood_p_value_threshold": 0.5,
            "save_posterior": False,
        }
    )
    output_path = Path(task.evaluate(None)["output_h5_path"])

    with h5py.File(output_path, "r") as output_file:
        assert bool(output_file.attrs["enable_calibration"])
        assert output_file.attrs["calibration_h5_path"] == str(
            calibration_path.resolve()
        )
        assert output_file.attrs["ood_p_value_threshold"] == 0.5
        video_group = output_file["nonexperts/video"]
        assert "posterior" not in video_group
        assert "conformal_p_value" in video_group
        conformal_dataset = video_group["conformal_p_value"]
        assert conformal_dataset.shape == (8, 2)
        assert conformal_dataset.dtype == np.dtype("float64")
        assert conformal_dataset.compression == "gzip"
        np.testing.assert_allclose(
            conformal_dataset[:],
            [
                [1.0 / 3.0, 1.0],
                [2.0 / 3.0, 1.0],
                [1.0, 1.0 / 4.0],
                [1.0, 1.0 / 2.0],
                [1.0, 3.0 / 4.0],
                [1.0, 1.0 / 4.0],
                [1.0 / 3.0, 1.0 / 2.0],
                [1.0 / 3.0, 1.0 / 4.0],
            ],
        )
        is_ood_dataset = video_group["is_ood"]
        assert is_ood_dataset.shape == (8,)
        assert is_ood_dataset.dtype == np.dtype("bool")
        np.testing.assert_array_equal(
            is_ood_dataset[:],
            [False, False, False, False, False, False, False, True],
        )
        progress_gated_dataset = video_group["progress_gated"]
        assert progress_gated_dataset.shape == (8,)
        assert progress_gated_dataset.dtype == np.dtype("float64")
        progress_mean = video_group["progress_mean"][:]
        expected_progress_gated = progress_mean.copy()
        expected_progress_gated[-1] = progress_mean[-2]
        np.testing.assert_allclose(
            progress_gated_dataset[:],
            expected_progress_gated,
        )


def test_calibration_rejects_empty_temporal_bin(tmp_path, monkeypatch):
    monkeypatch.setattr(task_gaussian_progress_pred, "_PROJ_ROOT", tmp_path)
    model_path = tmp_path / "three_bin_model.h5"
    _write_gaussian_model(
        model_path,
        means=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        final_covariances=np.stack([np.eye(2)] * 3),
    )
    calibration_path = tmp_path / "sparse_calibration.h5"
    _write_nonexpert_h5(
        calibration_path,
        {"video": (np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([0, 1]))},
    )
    nonexpert_path = tmp_path / "query.h5"
    _write_nonexpert_h5(
        nonexpert_path,
        {"video": (np.array([[0.0, 0.0]]), np.array([0]))},
    )

    task = GaussianProgressPredTask()
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(nonexpert_path),
            "enable_calibration": True,
            "calibration_h5_path": str(calibration_path),
            "ood_p_value_threshold": 0.5,
        }
    )
    with pytest.raises(ValueError, match="calibration.*bin"):
        task.evaluate(None)


def test_visualization_writes_two_figures_per_nonexpert(tmp_path, monkeypatch):
    monkeypatch.setattr(task_gaussian_progress_pred, "_PROJ_ROOT", tmp_path)
    model_path, _ = _identity_model(tmp_path)
    calibration_path = tmp_path / "calibration_visualization.h5"
    _write_nonexpert_h5(
        calibration_path,
        {
            "calibration_video": (
                np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
                np.arange(4),
            )
        },
    )
    nonexpert_path = tmp_path / "robomimic_test-2vid-embd.h5"
    _write_nonexpert_h5(
        nonexpert_path,
        {
            "000002": (
                np.array([[4.0, 0.0], [3.5, 0.0], [3.0, 0.0]]),
                np.arange(3),
            ),
            "000001": (
                np.array([[0.0, 0.0], [0.5, 0.0]]),
                np.arange(2),
            ),
        },
    )
    processed_path = tmp_path / "datasets/processed/robomimic_test-2vid.h5"
    processed_path.parent.mkdir(parents=True)
    raw_dir = tmp_path / "datasets/raw/robomimic_test"
    raw_dir.mkdir(parents=True)
    with h5py.File(processed_path, "w") as processed_file:
        videos_group = processed_file.create_group("videos")
        for video_id, frame_count in (("000001", 2), ("000002", 3)):
            action_name = f"demo_{int(video_id)}"
            raw_path = raw_dir / f"{action_name}.mp4"
            _write_rgb_mp4(
                raw_path,
                [
                    np.full(
                        (16, 16, 3),
                        (frame_index * 40, 80, 160),
                        dtype=np.uint8,
                    )
                    for frame_index in range(frame_count)
                ],
                fps=8.0,
            )
            video_group = videos_group.create_group(video_id)
            video_group.create_dataset(
                "frames",
                data=np.zeros((frame_count, 16, 16, 3), dtype=np.uint8),
            )
            video_group.attrs["action_name"] = action_name
            video_group.attrs["path"] = (
                f"/stale/project/datasets/raw/robomimic_test/{action_name}.mp4"
            )
            video_group.attrs["num_frames"] = frame_count

    task = GaussianProgressPredTask()
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(nonexpert_path),
            "enable_calibration": True,
            "calibration_h5_path": str(calibration_path),
            "ood_p_value_threshold": 0.5,
            "enable_visualization": True,
            "enable_video": True,
        }
    )
    output_path = Path(task.evaluate(None)["output_h5_path"])
    progress_paths = sorted((output_path.parent / "figures/progress").glob("*.png"))
    confidence_paths = sorted((output_path.parent / "figures/confidence").glob("*.png"))
    video_paths = sorted((output_path.parent / "videos").glob("*.mp4"))

    assert [path.stem for path in progress_paths] == ["000001", "000002"]
    assert [path.stem for path in confidence_paths] == ["000001", "000002"]
    assert [path.stem for path in video_paths] == ["000001", "000002"]
    assert all(
        path.stat().st_size > 0
        for path in progress_paths + confidence_paths + video_paths
    )


def test_visualization_uses_requested_axes_and_color_norms(tmp_path, monkeypatch):
    from matplotlib.colors import BoundaryNorm, Normalize, SymLogNorm
    from matplotlib.figure import Figure

    saved_figures = []

    def record_figure(figure, path, *args, **kwargs):  # noqa: ARG001
        saved_figures.append(figure)

    monkeypatch.setattr(Figure, "savefig", record_figure)
    frame_steps = np.arange(3, dtype=np.float64)
    bin_progress_values = np.array([0.0, 1.0])
    posterior = np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])
    conformal_p_value = np.array([[0.2, 0.6], [0.4, 0.8], [0.3, 0.7]])
    is_ood = np.array([False, True, False])
    likelihood = np.array([[-100.0, 2.0], [-1.0, 0.0], [3.0, -10.0]])

    _save_progress_figure(
        "video",
        frame_steps,
        bin_progress_values,
        np.array([0.2, 0.5, 0.9]),
        np.array([0.2, 0.2, 0.9]),
        posterior,
        is_ood,
        tmp_path / "progress.png",
    )
    _save_confidence_figure(
        "video",
        frame_steps,
        bin_progress_values,
        is_ood,
        conformal_p_value,
        likelihood,
        tmp_path / "confidence.png",
    )

    progress_ood_heatmap = saved_figures[0].axes[0].collections[0]
    progress_heatmap = saved_figures[0].axes[3].collections[0]
    ood_heatmap = saved_figures[1].axes[0].collections[0]
    p_value_heatmap = saved_figures[1].axes[1].collections[0]
    likelihood_heatmap = saved_figures[1].axes[2].collections[0]
    assert isinstance(progress_heatmap.norm, Normalize)
    assert progress_heatmap.norm.vmin == pytest.approx(posterior.min())
    assert progress_heatmap.norm.vmax == pytest.approx(posterior.max())
    assert progress_heatmap.cmap.name == "Blues"
    assert isinstance(progress_ood_heatmap.norm, BoundaryNorm)
    np.testing.assert_array_equal(
        np.asarray(progress_ood_heatmap.get_array()).reshape(-1),
        is_ood.astype(np.int8),
    )
    assert isinstance(ood_heatmap.norm, BoundaryNorm)
    np.testing.assert_array_equal(
        np.asarray(ood_heatmap.get_array()).reshape(-1),
        is_ood.astype(np.int8),
    )
    np.testing.assert_allclose(ood_heatmap.cmap(0), (1.0, 1.0, 1.0, 1.0))
    np.testing.assert_allclose(
        ood_heatmap.cmap(1),
        (214.0 / 255.0, 39.0 / 255.0, 40.0 / 255.0, 1.0),
    )
    assert saved_figures[1].axes[0].get_ylabel() == ""
    assert not saved_figures[1].axes[0].get_yticks().size
    assert isinstance(p_value_heatmap.norm, Normalize)
    assert not isinstance(p_value_heatmap.norm, SymLogNorm)
    assert p_value_heatmap.norm.vmin == pytest.approx(conformal_p_value.min())
    assert p_value_heatmap.norm.vmax == pytest.approx(conformal_p_value.max())
    assert p_value_heatmap.cmap.name == "Blues"
    assert not saved_figures[1].axes[1].lines
    assert saved_figures[1].axes[1].get_ylabel() == "Progress"
    assert isinstance(likelihood_heatmap.norm, SymLogNorm)
    assert likelihood_heatmap.norm.linthresh == 1.0
    assert likelihood_heatmap.norm.vmin == -100.0
    assert likelihood_heatmap.norm.vmax == 3.0
    assert likelihood_heatmap.cmap.name == "Blues"
    progress_figure = saved_figures[0]
    progress_figure.canvas.draw()
    assert len(progress_figure.axes) == 5
    np.testing.assert_allclose(progress_figure.get_size_inches(), [8.0, 6.5])
    assert progress_figure.dpi == 300
    progress_ood_box = progress_figure.axes[0].get_position()
    progress_gated_box = progress_figure.axes[1].get_position()
    progress_mean_box = progress_figure.axes[2].get_position()
    progress_heatmap_box = progress_figure.axes[3].get_position()
    assert progress_figure.axes[0].get_xlim() == pytest.approx((0.0, 2.0))
    assert progress_figure.axes[1].get_ylabel() == "Progress gated"
    assert progress_figure.axes[2].get_ylabel() == "Progress mean"
    assert progress_ood_box.x0 == pytest.approx(progress_gated_box.x0)
    assert progress_gated_box.x0 == pytest.approx(progress_mean_box.x0)
    assert progress_mean_box.x0 == pytest.approx(progress_heatmap_box.x0)
    assert progress_ood_box.width == pytest.approx(progress_gated_box.width)
    assert progress_gated_box.width == pytest.approx(progress_mean_box.width)
    assert progress_mean_box.width == pytest.approx(progress_heatmap_box.width)
    assert progress_gated_box.height == pytest.approx(progress_mean_box.height)
    assert progress_mean_box.height == pytest.approx(progress_heatmap_box.height)
    assert progress_ood_box.height == pytest.approx(progress_gated_box.height / 6.0)
    assert not progress_figure.axes[0].xaxis.get_ticklabels()
    assert not progress_figure.axes[1].xaxis.get_ticklabels()
    assert not progress_figure.axes[2].xaxis.get_ticklabels()
    assert progress_figure.axes[4].get_ylabel() == "Posterior probability"
    assert progress_figure.axes[4].get_position().x0 > progress_heatmap_box.x1

    confidence_figure = saved_figures[1]
    confidence_figure.canvas.draw()
    assert len(confidence_figure.axes) == 5
    ood_box = confidence_figure.axes[0].get_position()
    p_value_box = confidence_figure.axes[1].get_position()
    likelihood_box = confidence_figure.axes[2].get_position()
    assert confidence_figure.axes[0].get_xlim() == pytest.approx((0.0, 2.0))
    assert confidence_figure.axes[1].get_ylim() == pytest.approx((0.0, 1.0))
    assert confidence_figure.axes[2].get_ylim() == pytest.approx((0.0, 1.0))
    assert ood_box.x0 == pytest.approx(p_value_box.x0)
    assert p_value_box.x0 == pytest.approx(likelihood_box.x0)
    assert ood_box.width == pytest.approx(p_value_box.width)
    assert p_value_box.width == pytest.approx(likelihood_box.width)
    assert p_value_box.height == pytest.approx(likelihood_box.height)
    assert ood_box.height == pytest.approx(p_value_box.height / 6.0)
    assert not confidence_figure.axes[0].xaxis.get_ticklabels()
    assert not confidence_figure.axes[1].xaxis.get_ticklabels()
    assert confidence_figure.axes[3].get_ylabel() == "Conformal p-value"
    assert confidence_figure.axes[4].get_ylabel() == "Gaussian log likelihood"
    assert confidence_figure.axes[3].get_position().x0 > p_value_box.x1
    assert confidence_figure.axes[4].get_position().x0 > likelihood_box.x1
    assert confidence_figure._suptitle.get_text() == "Progress confidence — video"


def test_confidence_figure_rejects_misaligned_ood_shape(tmp_path):
    with pytest.raises(ValueError, match="is_ood shape must match"):
        _save_confidence_figure(
            "video",
            np.arange(3, dtype=np.float64),
            np.array([0.0, 1.0]),
            np.array([True, False]),
            np.ones((3, 2)),
            np.ones((3, 2)),
            tmp_path / "confidence.png",
        )


def test_video_figures_are_compact_titleless_and_have_cursor_lines():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_steps = np.arange(3, dtype=np.float64)
    bin_progress_values = np.array([0.0, 1.0])
    progress_figure, progress_lines = _build_progress_figure(
        plt=plt,
        video_id="video",
        frame_steps=frame_steps,
        bin_progress_values=bin_progress_values,
        progress_mean=np.array([0.1, 0.5, 0.9]),
        posterior=np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
        progress_curve_label="Progress gated",
        figsize=(4.0, 2.0),
        show_title=False,
        cursor_x=1.0,
    )
    confidence_figure, confidence_lines = _build_confidence_figure(
        plt=plt,
        video_id="video",
        frame_steps=frame_steps,
        bin_progress_values=bin_progress_values,
        is_ood=np.array([False, True, False]),
        conformal_p_value=np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
        gaussian_log_likelihood=np.array([[-3.0, -1.0], [-2.0, -2.0], [-1.0, -3.0]]),
        figsize=(4.0, 2.25),
        show_title=False,
        cursor_x=1.0,
    )
    try:
        np.testing.assert_allclose(progress_figure.get_size_inches(), [4.0, 2.0])
        np.testing.assert_allclose(confidence_figure.get_size_inches(), [4.0, 2.25])
        assert progress_figure.axes[0].get_title() == ""
        assert progress_figure.axes[0].get_ylabel() == "Progress gated"
        assert confidence_figure._suptitle is None
        assert len(progress_lines) == 2
        assert len(confidence_lines) == 3
        assert not progress_figure.axes[2].lines
        assert not confidence_figure.axes[3].lines
        assert not confidence_figure.axes[4].lines
        for line in progress_lines + confidence_lines:
            assert line.get_color() == "black"
            assert line.get_linestyle() == "--"
            assert line.get_linewidth() == pytest.approx(0.8)
            np.testing.assert_allclose(line.get_xdata(), [1.0, 1.0])
    finally:
        plt.close(progress_figure)
        plt.close(confidence_figure)


def _write_rgb_mp4(path: Path, frames_rgb: list[np.ndarray], fps: float) -> None:
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        str(path),
        fps=fps,
        format="ffmpeg",
        codec="libx264",
        macro_block_size=1,
    )
    try:
        for frame in frames_rgb:
            writer.append_data(frame)
    finally:
        writer.close()


def test_raw_video_mapping_uses_current_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(task_gaussian_progress_pred, "_PROJ_ROOT", tmp_path)
    processed_path = tmp_path / "datasets/processed/robomimic_can_mh-2vid-worse.h5"
    processed_path.parent.mkdir(parents=True)
    raw_path = tmp_path / "datasets/raw/robomimic_can_mh/demo_50.mp4"
    raw_path.parent.mkdir(parents=True)
    raw_path.touch()
    with h5py.File(processed_path, "w") as processed_file:
        group = processed_file.create_group("videos/000001")
        group.create_dataset("frames", data=np.zeros((3, 2, 2, 3), dtype=np.uint8))
        group.attrs["action_name"] = "demo_50"
        group.attrs["path"] = "/stale/project/datasets/raw/robomimic_can_mh/demo_50.mp4"
        group.attrs["num_frames"] = 3

    inferred = _processed_h5_path_from_embedding(
        "/embedding/run/robomimic_can_mh-2vid-worse-embd-ep010000.h5"
    )
    assert inferred == processed_path
    with h5py.File(processed_path, "r") as processed_file:
        resolved, frame_count = _resolve_raw_video_path(
            processed_file["videos"], "000001"
        )
    assert resolved == raw_path
    assert frame_count == 3


def test_prediction_video_end_to_end_and_alignment_fail_fast(tmp_path):
    import cv2

    raw_path = tmp_path / "raw.mp4"
    frames = [
        np.full((16, 16, 3), (255, 0, 0), dtype=np.uint8),
        np.full((16, 16, 3), (0, 255, 0), dtype=np.uint8),
        np.full((16, 16, 3), (0, 0, 255), dtype=np.uint8),
    ]
    _write_rgb_mp4(raw_path, frames, fps=7.0)
    output_path = tmp_path / "video.mp4"
    output_path.write_bytes(b"old video")
    common = {
        "video_id": "000001",
        "raw_video_path": raw_path,
        "processed_frame_count": 3,
        "target_steps": np.arange(3),
        "bin_progress_values": np.array([0.0, 1.0]),
        "progress_gated": np.array([0.1, 0.1, 0.9]),
        "posterior": np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
        "is_ood": np.array([False, True, False]),
        "conformal_p_value": np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
        "gaussian_log_likelihood": np.array([[-3.0, -1.0], [-2.0, -2.0], [-1.0, -3.0]]),
        "output_path": output_path,
    }
    _save_prediction_video(**common)
    assert output_path.read_bytes() != b"old video"
    assert not (tmp_path / ".video.tmp.mp4").exists()

    capture = cv2.VideoCapture(str(output_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(7.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first_bgr = capture.read()
    capture.release()
    assert ok
    assert width % 2 == 0
    assert height % 2 == 0
    assert width > height * 3
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    assert first_rgb[height // 2, height // 2, 0] > 200

    bad = dict(common)
    bad["target_steps"] = np.array([0, 2, 4])
    bad["output_path"] = tmp_path / "bad.mp4"
    with pytest.raises(ValueError, match="target_steps must equal arange"):
        _save_prediction_video(**bad)
    assert not bad["output_path"].exists()


def test_end_to_end_pca_query_projection_and_output_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(task_gaussian_progress_pred, "_PROJ_ROOT", tmp_path)
    model_path = tmp_path / "pca_model.h5"
    pca_mean = np.array([1.0, 2.0, 3.0])
    pca_components = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    _write_gaussian_model(
        model_path,
        means=np.array([[0.0, 0.0], [4.0, 2.0]]),
        final_covariances=np.stack([np.eye(2), np.eye(2)]),
        embedding_normalization="none",
        enable_pca=True,
        input_embedding_dim=3,
        pca_mean=pca_mean,
        pca_components=pca_components,
    )
    raw_query = np.array(
        [[5.0, 4.0, 100.0], [1.0, 2.0, 3.0]],
        dtype=np.float64,
    )
    query_path = tmp_path / "pca_query.h5"
    _write_nonexpert_h5(
        query_path,
        {"video": (raw_query, np.array([10, 20]))},
        embedding_normalization="none",
    )

    task = GaussianProgressPredTask()
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(query_path),
        }
    )
    result = task.evaluate(None)

    model = _read_gaussian_model(str(model_path))
    projected_query = np.array([[4.0, 2.0], [0.0, 0.0]])
    expected = _infer_one_trajectory(
        projected_query,
        model,
        posterior_temperature=1.0,
        entropy_epsilon=1.0e-12,
    )
    with h5py.File(result["output_h5_path"], "r") as output_file:
        assert bool(output_file.attrs["enable_pca"])
        assert output_file.attrs["input_embedding_dim"] == 3
        assert output_file.attrs["embedding_dim"] == 2
        np.testing.assert_allclose(
            output_file["model/pca_mean"][:],
            pca_mean,
        )
        np.testing.assert_allclose(
            output_file["model/pca_components"][:],
            pca_components,
        )
        np.testing.assert_allclose(
            output_file["nonexperts/video/gaussian_log_likelihood"][:],
            expected["gaussian_log_likelihood"],
        )
        np.testing.assert_allclose(
            output_file["nonexperts/video/posterior"][:],
            expected["posterior"],
        )
        np.testing.assert_array_equal(
            output_file["nonexperts/video/map_bin"][:],
            expected["map_bin"],
        )


def test_pca_query_normalization_is_checked_before_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(task_gaussian_progress_pred, "_PROJ_ROOT", tmp_path)
    model_path = tmp_path / "pca_l2_model.h5"
    _write_gaussian_model(
        model_path,
        means=np.array([[0.0, 0.0], [1.0, 0.0]]),
        final_covariances=np.stack([np.eye(2), np.eye(2)]),
        embedding_normalization="l2",
        enable_pca=True,
        input_embedding_dim=3,
        pca_mean=np.zeros(3),
        pca_components=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    query_path = tmp_path / "bad_pca_l2_query.h5"
    _write_nonexpert_h5(
        query_path,
        {
            "video": (
                np.array([[1.0, 0.0, 1.0]]),
                np.array([0]),
            )
        },
        embedding_normalization="l2",
    )

    task = GaussianProgressPredTask()
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(query_path),
        }
    )
    with pytest.raises(ValueError, match="not unit norm"):
        task.evaluate(None)


def test_l2_model_and_query_match_and_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(task_gaussian_progress_pred, "_PROJ_ROOT", tmp_path)
    model_path = tmp_path / "model_l2.h5"
    _write_gaussian_model(
        model_path,
        means=np.array([[1.0, 0.0], [0.0, 1.0]]),
        final_covariances=np.stack([np.eye(2), np.eye(2)]),
        embedding_normalization="l2",
    )
    query_l2_path = tmp_path / "query_l2.h5"
    _write_nonexpert_h5(
        query_l2_path,
        {
            "video": (
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([0, 1]),
            )
        },
        embedding_normalization="l2",
    )

    task = GaussianProgressPredTask()
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(query_l2_path),
        }
    )
    result = task.evaluate(None)
    with h5py.File(result["output_h5_path"], "r") as output_file:
        assert output_file.attrs["embedding_normalization"] == "l2"

    query_none_path = tmp_path / "query_none.h5"
    _write_nonexpert_h5(
        query_none_path,
        {
            "video": (
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([0, 1]),
            )
        },
        embedding_normalization="none",
    )
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(query_none_path),
        }
    )
    with pytest.raises(ValueError, match="embedding_normalization mismatch"):
        task.evaluate(None)

    query_bad_path = tmp_path / "query_bad_l2.h5"
    _write_nonexpert_h5(
        query_bad_path,
        {
            "video": (
                np.array([[2.0, 0.0], [0.0, 1.0]]),
                np.array([0, 1]),
            )
        },
        embedding_normalization="l2",
    )
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(query_bad_path),
        }
    )
    with pytest.raises(ValueError, match="not unit norm"):
        task.evaluate(None)


def test_save_posterior_false_omits_only_posterior(tmp_path, monkeypatch):
    monkeypatch.setattr(task_gaussian_progress_pred, "_PROJ_ROOT", tmp_path)
    model_path, _ = _identity_model(tmp_path)
    nonexpert_path = tmp_path / "query_no_posterior.h5"
    _write_nonexpert_h5(
        nonexpert_path,
        {
            "video": (
                np.array([[0.0, 0.0], [1.0, 0.0]]),
                np.array([0, 1]),
            )
        },
    )
    task = GaussianProgressPredTask()
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(nonexpert_path),
            "save_posterior": False,
        }
    )
    result = task.evaluate(None)

    with h5py.File(result["output_h5_path"], "r") as output_file:
        video_group = output_file["nonexperts/video"]
        assert "posterior" not in video_group
        assert "gaussian_log_likelihood" in video_group
        assert video_group["gaussian_log_likelihood"].shape == (2, 2)
        assert video_group["gaussian_log_likelihood"].compression == "gzip"
        assert "progress_mean" in video_group
        assert "min_mahalanobis_sq" in video_group
        assert not bool(output_file.attrs["save_posterior"])


def test_nonexpert_dimension_and_nonfinite_values_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(task_gaussian_progress_pred, "_PROJ_ROOT", tmp_path)
    model_path, _ = _identity_model(tmp_path)

    dimension_path = tmp_path / "wrong_dimension.h5"
    _write_nonexpert_h5(
        dimension_path,
        {"video": (np.zeros((2, 3)), np.array([0, 1]))},
    )
    task = GaussianProgressPredTask()
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(dimension_path),
        }
    )
    with pytest.raises(ValueError, match="dimension mismatch"):
        task.evaluate(None)

    nonfinite_path = tmp_path / "nonfinite.h5"
    _write_nonexpert_h5(
        nonfinite_path,
        {
            "video": (
                np.array([[0.0, 0.0], [np.nan, 0.0]]),
                np.array([0, 1]),
            )
        },
    )
    task.configure(
        {
            "gaussian_model_h5_path": str(model_path),
            "nonexpert_h5_path": str(nonfinite_path),
        }
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        task.evaluate(None)
