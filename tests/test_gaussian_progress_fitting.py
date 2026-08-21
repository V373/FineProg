from pathlib import Path
import sys
import types

import h5py
import numpy as np
import pytest

_PROJECTS_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECTS_ROOT))

from fineprog.algos.eval_task.base_task import build_task
from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_fitting import (
    GaussianProgressFittingTask,
    _apply_covariance_floor_and_regularization,
    _apply_saved_model_pca,
    _build_covariance_artifact_tag,
    _compute_temporal_progress,
    _fit_gaussian_progress_model,
    _fit_model_pca,
    _fit_open_tsne_reference_projection,
    _fit_tsne_projection,
    _parse_fitting_config,
    _parse_visualization_config,
    _read_expert_trajectories,
)
from fineprog.utils.config_v2 import ConfigV2


def _base_config(**overrides):
    config = {
        "enable_pca": False,
        "pca_dim": 8,
        "num_bins": 2,
        "covariance_mode": "independent",
        "shared_covariance_weight": 0.5,
        "variance_floor": 1.0e-8,
        "enable_covariance_regularization": False,
        "covariance_regularization": 1.0e-6,
        "covariance_rank_tolerance": 1.0e-5,
        "enable_calibration": False,
    }
    config.update(overrides)
    return _parse_fitting_config(config)


def _write_expert_h5(
    path: Path,
    records: dict[str, tuple[np.ndarray, np.ndarray | None]],
    embedding_normalization: str | None = None,
):
    with h5py.File(path, "w") as h5_file:
        if embedding_normalization is not None:
            h5_file.attrs["embedding_normalization"] = embedding_normalization
        videos = h5_file.create_group("videos")
        for video_id, (embeddings, target_steps) in records.items():
            group = videos.create_group(video_id)
            group.create_dataset("embeddings", data=embeddings)
            if target_steps is not None:
                group.create_dataset("target_steps", data=target_steps)


def test_end_to_end_exact_statistics_and_h5_schema(tmp_path):
    expert_h5_path = tmp_path / "expert.h5"
    _write_expert_h5(
        expert_h5_path,
        {
            "000001": (
                np.array(
                    [[0.0, 0.0], [2.0, 2.0], [10.0, 0.0], [12.0, 2.0]],
                    dtype=np.float64,
                ),
                np.array([0, 1, 2, 3]),
            )
        },
    )

    task = GaussianProgressFittingTask()
    task.configure(
        {
            "expert_h5_path": str(expert_h5_path),
            "output_dir": str(tmp_path / "outputs"),
            **_base_config(variance_floor=0.1),
        }
    )
    result = task.evaluate(None)

    assert result["metric_name"] == "num_bins"
    assert result["metric_value"] == 2.0
    assert result["bin_counts"] == [2, 2]
    assert result["bin_final_covariance_ranks"] == [2, 2]
    assert result["bin_pre_pca_final_covariance_ranks"] is None
    assert result["num_expert_features"] == 4
    assert not result["enable_pca"]
    assert result["input_embedding_dim"] == 2
    assert result["embedding_dim"] == 2
    assert not result["visualization_enabled"]
    assert result["output_visualization_path"] is None
    assert result["output_real_visualization_path"] is None
    assert result["output_real_video_idx_visualization_path"] is None
    assert result["visualization_num_tsne_points"] == 0
    assert result["real_visualization_num_tsne_points"] == 0
    assert "covariance_mode-independent" in Path(result["output_h5_path"]).name
    assert Path(result["output_h5_path"]).parent == Path(result["output_dir"])
    assert Path(result["output_dir"]).parent == tmp_path / "outputs"
    assert len(Path(result["output_dir"]).name) == len("YYYYMMDD-HHMMSS")

    expected_covariance = np.array([[2.0, 2.0], [2.0, 2.0]])
    with h5py.File(result["output_h5_path"], "r") as model_h5:
        assert set(model_h5.keys()) == {"model"}
        model_group = model_h5["model"]
        assert set(model_group.keys()) == {
            "bin_progress_values",
            "bin_means",
            "bin_independent_covariances",
            "shared_covariance",
            "bin_final_covariances",
            "bin_final_covariance_ranks",
            "bin_log_determinants",
            "bin_counts",
        }
        np.testing.assert_allclose(model_group["bin_progress_values"][:], [0.0, 1.0])
        np.testing.assert_allclose(model_group["bin_means"][:], [[1.0, 1.0], [11.0, 1.0]])
        np.testing.assert_allclose(
            model_group["bin_independent_covariances"][:],
            np.stack([expected_covariance, expected_covariance]),
        )
        np.testing.assert_allclose(model_group["shared_covariance"][:], expected_covariance)
        np.testing.assert_array_equal(model_group["bin_counts"][:], [2, 2])
        final_covariances = model_group["bin_final_covariances"][:]
        assert final_covariances.shape == (2, 2, 2)
        assert np.all(np.linalg.eigvalsh(final_covariances) > 0.0)
        np.testing.assert_array_equal(
            model_group["bin_final_covariance_ranks"][:], [2, 2]
        )
        signs, expected_logdets = np.linalg.slogdet(final_covariances)
        np.testing.assert_array_equal(signs, np.ones(2))
        np.testing.assert_allclose(
            model_group["bin_log_determinants"][:], expected_logdets
        )

        assert model_h5.attrs["task_name"] == "gaussian_progress_fitting"
        assert model_h5.attrs["embedding_normalization"] == "none"
        assert model_h5.attrs["num_expert_videos"] == 1
        assert model_h5.attrs["num_expert_features"] == 4
        assert model_h5.attrs["num_bins"] == 2
        assert not bool(model_h5.attrs["enable_pca"])
        assert model_h5.attrs["input_embedding_dim"] == 2
        assert model_h5.attrs["embedding_dim"] == 2
        assert model_h5.attrs["covariance_mode"] == "independent"
        assert model_h5.attrs["covariance_rank_tolerance"] == pytest.approx(1.0e-5)
        assert not bool(model_h5.attrs["enable_calibration"])


def test_model_pca_drives_gaussian_statistics_and_h5_schema(tmp_path):
    expert_h5_path = tmp_path / "expert_pca.h5"
    raw_embeddings = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.5, 3.0],
            [8.0, 1.0, 2.0],
            [9.0, 3.0, 4.0],
            [10.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    target_steps = np.arange(raw_embeddings.shape[0])
    _write_expert_h5(
        expert_h5_path,
        {"000001": (raw_embeddings, target_steps)},
    )

    fitting_config = _base_config(
        enable_pca=True,
        pca_dim=2,
        variance_floor=0.1,
    )
    expected_trajectories, expected_pca = _fit_model_pca(
        [raw_embeddings],
        pca_dim=2,
    )
    expected_model = _fit_gaussian_progress_model(
        expected_trajectories,
        [target_steps / float(target_steps[-1])],
        fitting_config,
    )
    expected_pre_pca_model = _fit_gaussian_progress_model(
        [raw_embeddings],
        [target_steps / float(target_steps[-1])],
        fitting_config,
    )

    task = GaussianProgressFittingTask()
    task.configure(
        {
            "expert_h5_path": str(expert_h5_path),
            "output_dir": str(tmp_path / "outputs_pca"),
            **fitting_config,
        }
    )
    result = task.evaluate(None)

    assert result["enable_pca"]
    assert result["input_embedding_dim"] == 3
    assert result["embedding_dim"] == 2
    assert result["bin_final_covariance_ranks"] == expected_model[
        "bin_final_covariance_ranks"
    ].tolist()
    assert result["bin_pre_pca_final_covariance_ranks"] == expected_pre_pca_model[
        "bin_final_covariance_ranks"
    ].tolist()
    with h5py.File(result["output_h5_path"], "r") as model_h5:
        model_group = model_h5["model"]
        assert bool(model_h5.attrs["enable_pca"])
        assert model_h5.attrs["input_embedding_dim"] == 3
        assert model_h5.attrs["embedding_dim"] == 2
        assert model_group["pca_mean"].shape == (3,)
        assert model_group["pca_components"].shape == (2, 3)
        np.testing.assert_allclose(
            model_group["pca_mean"][:],
            expected_pca["pca_mean"],
        )
        np.testing.assert_allclose(
            model_group["pca_components"][:],
            expected_pca["pca_components"],
        )
        np.testing.assert_allclose(
            model_group["bin_means"][:],
            expected_model["bin_means"],
        )
        np.testing.assert_allclose(
            model_group["bin_independent_covariances"][:],
            expected_model["bin_independent_covariances"],
        )
        assert model_group["bin_final_covariances"].shape == (2, 2, 2)
        np.testing.assert_array_equal(
            model_group["bin_final_covariance_ranks"][:],
            expected_model["bin_final_covariance_ranks"],
        )
        np.testing.assert_array_equal(
            model_group["bin_pre_pca_final_covariance_ranks"][:],
            expected_pre_pca_model["bin_final_covariance_ranks"],
        )

        replayed = _apply_saved_model_pca(
            [raw_embeddings],
            model_group["pca_mean"][:],
            model_group["pca_components"][:],
        )
        np.testing.assert_allclose(replayed[0], expected_trajectories[0])


def test_l2_metadata_is_propagated_to_gaussian_model(tmp_path):
    expert_h5_path = tmp_path / "expert_l2.h5"
    _write_expert_h5(
        expert_h5_path,
        {
            "000001": (
                np.array(
                    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                    dtype=np.float64,
                ),
                np.array([0, 1, 2, 3]),
            )
        },
        embedding_normalization="l2",
    )

    task = GaussianProgressFittingTask()
    task.configure(
        {
            "expert_h5_path": str(expert_h5_path),
            "output_dir": str(tmp_path / "outputs_l2"),
            **_base_config(variance_floor=0.1),
        }
    )
    result = task.evaluate(None)

    with h5py.File(result["output_h5_path"], "r") as model_h5:
        assert model_h5.attrs["embedding_normalization"] == "l2"


@pytest.mark.parametrize(
    ("target_steps", "expected_reason"),
    [
        (None, "missing target_steps"),
        (np.array([0, 1]), "shape mismatch"),
        (np.array([5, 5, 5]), "first and last values are equal"),
        (np.array([[0], [1], [2]]), "shape mismatch"),
    ],
)
def test_temporal_progress_fallbacks(target_steps, expected_reason):
    progress, reason = _compute_temporal_progress("video", 3, target_steps)
    np.testing.assert_allclose(progress, [0.0, 0.5, 1.0])
    assert expected_reason in reason


def test_progress_bin_boundaries_include_one_in_last_bin():
    embeddings = [np.array([[0.0], [1.0], [2.0], [3.0]])]
    progresses = [np.array([0.0, 0.49, 0.5, 1.0])]
    model = _fit_gaussian_progress_model(embeddings, progresses, _base_config())
    np.testing.assert_array_equal(model["bin_counts"], [2, 2])


def test_final_covariance_rank_uses_configured_tolerance():
    embeddings = [
        np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 0.0], [12.0, 2.0]])
    ]
    progresses = [np.array([0.0, 0.25, 0.75, 1.0])]
    model = _fit_gaussian_progress_model(
        embeddings,
        progresses,
        _base_config(
            variance_floor=1.0e-8,
            covariance_rank_tolerance=1.0e-5,
        ),
    )
    np.testing.assert_array_equal(model["bin_final_covariance_ranks"], [1, 1])


def test_temporal_progress_rejects_nonfinite_and_out_of_range_values():
    with pytest.raises(ValueError, match="NaN or Inf"):
        _compute_temporal_progress("video", 3, np.array([0.0, np.nan, 2.0]))
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        _compute_temporal_progress("video", 3, np.array([0.0, 2.0, 1.0]))


def test_pooled_covariance_uses_within_bin_residuals_not_global_mean():
    embeddings = [
        np.array([[0.0], [2.0], [100.0], [102.0]], dtype=np.float64)
    ]
    progresses = [np.array([0.0, 0.25, 0.75, 1.0])]
    model = _fit_gaussian_progress_model(embeddings, progresses, _base_config())
    np.testing.assert_allclose(model["shared_covariance"], [[2.0]])
    assert float(np.var(embeddings[0], ddof=1)) > 2.0


def test_shared_and_weighted_covariance_modes():
    embeddings = [
        np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [10.0, 0.0],
                [10.0, 2.0],
                [12.0, 1.0],
            ]
        )
    ]
    progresses = [np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])]

    shared_model = _fit_gaussian_progress_model(
        embeddings,
        progresses,
        _base_config(covariance_mode="shared"),
    )
    np.testing.assert_allclose(
        shared_model["bin_final_covariances"][0],
        shared_model["bin_final_covariances"][1],
    )

    weighted_config = _base_config(
        covariance_mode="independent_shared_weighted",
        shared_covariance_weight=0.5,
    )
    weighted_model = _fit_gaussian_progress_model(
        embeddings,
        progresses,
        weighted_config,
    )
    for bin_index in range(2):
        expected_base = 0.5 * (
            weighted_model["bin_independent_covariances"][bin_index]
            + weighted_model["shared_covariance"]
        )
        expected_final = _apply_covariance_floor_and_regularization(
            expected_base,
            variance_floor=weighted_config["variance_floor"],
            enable_regularization=False,
            covariance_regularization=weighted_config["covariance_regularization"],
        )
        np.testing.assert_allclose(
            weighted_model["bin_final_covariances"][bin_index], expected_final
        )


def test_eigenvalue_floor_and_regularization_preserve_full_covariance():
    covariance = np.array([[1.0, 1.0], [1.0, 1.0]])
    floored = _apply_covariance_floor_and_regularization(
        covariance,
        variance_floor=0.25,
        enable_regularization=False,
        covariance_regularization=0.5,
    )
    regularized = _apply_covariance_floor_and_regularization(
        covariance,
        variance_floor=0.25,
        enable_regularization=True,
        covariance_regularization=0.5,
    )
    assert floored[0, 1] != 0.0
    np.testing.assert_allclose(regularized, floored + 0.5 * np.eye(2))
    assert np.linalg.eigvalsh(floored).min() == pytest.approx(0.25)
    assert np.linalg.eigvalsh(regularized).min() == pytest.approx(0.75)


@pytest.mark.parametrize(
    "override",
    [
        {"enable_pca": "yes"},
        {"pca_dim": 0},
        {"pca_dim": 2.5},
        {"pca_dim": True},
        {"num_bins": 1},
        {"num_bins": 2.5},
        {"covariance_mode": "diagonal"},
        {"shared_covariance_weight": -0.1},
        {"shared_covariance_weight": 1.1},
        {"variance_floor": 0.0},
        {"covariance_regularization": -1.0},
        {"covariance_rank_tolerance": -1.0},
        {"covariance_rank_tolerance": np.nan},
        {"covariance_rank_tolerance": True},
    ],
)
def test_invalid_fitting_config_is_rejected(override):
    with pytest.raises(ValueError):
        _base_config(**override)


def test_calibration_true_is_explicitly_rejected():
    with pytest.raises(NotImplementedError, match="not supported"):
        _base_config(enable_calibration=True)


def test_model_pca_rejects_dimension_not_smaller_than_input():
    embeddings = [np.zeros((4, 3), dtype=np.float64)]
    with pytest.raises(ValueError, match=r"input_embedding_dim - 1"):
        _fit_model_pca(embeddings, pca_dim=3)


@pytest.mark.parametrize(
    ("mode", "expected_tag"),
    [
        ("independent", "covariance_mode-independent"),
        ("shared", "covariance_mode-shared"),
        (
            "independent_shared_weighted",
            "covariance_mode-independent_shared_weighted-shared_covariance_weight-0.5",
        ),
    ],
)
def test_covariance_artifact_tag(mode, expected_tag):
    config = _base_config(covariance_mode=mode, shared_covariance_weight=0.5)
    assert _build_covariance_artifact_tag(config) == expected_tag


def test_undersampled_bins_report_bin_indices_and_counts():
    embeddings = [np.array([[0.0], [1.0], [2.0]])]
    progresses = [np.array([0.0, 0.1, 1.0])]
    with pytest.raises(ValueError, match=r"bin 1: 1"):
        _fit_gaussian_progress_model(embeddings, progresses, _base_config())


def test_h5_validation_rejects_missing_videos_and_nonfinite_embeddings(tmp_path):
    missing_videos_path = tmp_path / "missing_videos.h5"
    with h5py.File(missing_videos_path, "w"):
        pass
    with pytest.raises(ValueError, match="/videos"):
        _read_expert_trajectories(str(missing_videos_path))

    nonfinite_path = tmp_path / "nonfinite.h5"
    _write_expert_h5(
        nonfinite_path,
        {"000001": (np.array([[0.0], [np.nan]]), np.array([0, 1]))},
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        _read_expert_trajectories(str(nonfinite_path))


def test_h5_validation_rejects_inconsistent_embedding_dimensions(tmp_path):
    path = tmp_path / "inconsistent_dims.h5"
    _write_expert_h5(
        path,
        {
            "000001": (np.zeros((2, 2)), np.array([0, 1])),
            "000002": (np.zeros((2, 3)), np.array([0, 1])),
        },
    )
    with pytest.raises(ValueError, match="inconsistent embedding dimension"):
        _read_expert_trajectories(str(path))


def test_h5_validation_rejects_non_matrix_and_singleton_trajectories(tmp_path):
    non_matrix_path = tmp_path / "non_matrix.h5"
    _write_expert_h5(
        non_matrix_path,
        {"000001": (np.zeros(3), np.array([0, 1, 2]))},
    )
    with pytest.raises(ValueError, match=r"shape \[T, D\]"):
        _read_expert_trajectories(str(non_matrix_path))

    singleton_path = tmp_path / "singleton.h5"
    _write_expert_h5(
        singleton_path,
        {"000001": (np.zeros((1, 2)), np.array([0]))},
    )
    with pytest.raises(ValueError, match="T>=2"):
        _read_expert_trajectories(str(singleton_path))


def test_config_v2_and_task_factory_registration():
    resolved = ConfigV2().load_eval("gaussian_progress_fitting")
    assert Path(resolved["expert_h5_path"]).is_absolute()
    assert Path(resolved["expert_h5_path"]).suffix == ".h5"
    assert Path(resolved["output_dir"]).is_absolute()
    assert "gaussian_progress_fitting" in resolved["output_dir"]
    assert resolved["enable_pca"] is True
    assert resolved["pca_dim"] == 32
    assert resolved["covariance_rank_tolerance"] == pytest.approx(1.0e-5)
    assert isinstance(resolved["enable_visualization"], bool)
    assert resolved["tsne_viz"]["tsne"] == {
        "use_open_tsne": False,
        "perplexity_mode": "config_clamped",
        "perplexity": 1000,
        "learning_rate": "auto",
        "init": "pca",
        "max_iter": 5000,
    }
    assert resolved["tsne_viz"]["plot"]["enable_real_only_debug"] is False
    assert resolved["tsne_viz"]["plot"]["enable_real_video_idx_plot"] is False
    assert isinstance(build_task("gaussian_progress_fitting"), GaussianProgressFittingTask)


def test_config_v2_explicit_paths_override_registry_reference(tmp_path):
    explicit_h5 = tmp_path / "custom_expert.h5"
    explicit_output = tmp_path / "custom_output"
    resolved = ConfigV2().load_eval(
        "gaussian_progress_fitting",
        overrides={
            "expert_h5_path": str(explicit_h5),
            "output_dir": str(explicit_output),
        },
    )
    assert resolved["expert_h5_path"] == str(explicit_h5)
    assert resolved["output_dir"] == str(explicit_output)


@pytest.mark.parametrize(
    "override",
    [
        {"enable_visualization": "yes"},
        {"enable_visualization": True, "tsne_viz": {"gaussian_samples_per_bin": 1}},
        {
            "enable_visualization": True,
            "tsne_viz": {"gaussian_samples_per_bin": "match_real_bin_count"},
        },
        {
            "enable_visualization": True,
            "tsne_viz": {"tsne": {"perplexity": 0}},
        },
        {
            "enable_visualization": True,
            "tsne_viz": {"contour": {"mass_levels": [0.8, 0.5]}},
        },
        {
            "enable_visualization": True,
            "tsne_viz": {"tsne": {"use_open_tsne": "yes"}},
        },
        {
            "enable_visualization": True,
            "tsne_viz": {"plot": {"enable_real_video_idx_plot": "yes"}},
        },
        {
            "enable_visualization": True,
            "tsne_viz": {"plot": {"enable_real_only_debug": "yes"}},
        },
    ],
)
def test_invalid_visualization_config_is_rejected(override):
    with pytest.raises(ValueError):
        _parse_visualization_config(override)


def test_sklearn_visualization_skips_real_only_debug_by_default(
    tmp_path, monkeypatch
):
    expert_h5_path = tmp_path / "expert.h5"
    _write_expert_h5(
        expert_h5_path,
        {
            "000001": (
                np.array(
                    [
                        [0.0, 0.0, 0.3],
                        [1.0, 0.2, 1.1],
                        [0.2, 1.0, -0.4],
                        [0.6, 0.7, 0.8],
                    ]
                ),
                np.arange(4),
            ),
            "000002": (
                np.array(
                    [
                        [4.0, 4.0, 1.5],
                        [5.0, 4.2, -0.2],
                        [4.2, 5.0, 2.2],
                        [4.6, 4.7, 0.8],
                    ]
                ),
                np.arange(4),
            ),
        },
    )

    tsne_calls = []
    save_calls = []
    real_scatter_markers = []
    contour_save_counts = []

    class FakeTSNE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, values):
            tsne_calls.append((self.kwargs, values.shape))
            return np.asarray(values[:, :2], dtype=np.float64)

    monkeypatch.setattr("sklearn.manifold.TSNE", FakeTSNE)
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    original_savefig = Figure.savefig
    original_scatter = Axes.scatter
    original_contour = Axes.contour

    def recording_savefig(figure, path, *args, **kwargs):
        save_calls.append(Path(path).name)
        return original_savefig(figure, path, *args, **kwargs)

    def recording_scatter(axis, *args, **kwargs):
        if not save_calls:
            real_scatter_markers.append(kwargs.get("marker"))
        return original_scatter(axis, *args, **kwargs)

    def recording_contour(axis, *args, **kwargs):
        contour_save_counts.append(len(save_calls))
        return original_contour(axis, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", recording_savefig)
    monkeypatch.setattr(Axes, "scatter", recording_scatter)
    monkeypatch.setattr(Axes, "contour", recording_contour)
    task = GaussianProgressFittingTask()
    task.configure(
        {
            "expert_h5_path": str(expert_h5_path),
            "output_dir": str(tmp_path / "outputs"),
            **_base_config(enable_pca=True, pca_dim=2),
            "enable_visualization": True,
            "tsne_viz": {
                "gaussian_samples_per_bin": "match_real_bin_counts",
                "random_seed": 7,
                "preprocessing": {
                    "standardize": True,
                    "use_pca_before_tsne": False,
                },
                "tsne": {
                    "use_open_tsne": False,
                    "perplexity_mode": "config_clamped",
                    "perplexity": 500,
                    "learning_rate": "auto",
                    "init": "pca",
                    "max_iter": 250,
                },
                "plot": {"dpi": 50},
            },
        }
    )
    result = task.evaluate(None)

    assert len(tsne_calls) == 1
    assert tsne_calls[0][1] == (18, 2)
    assert tsne_calls[0][0]["perplexity"] == pytest.approx(17 / 3)
    for setting in (
        "learning_rate",
        "init",
        "random_state",
        "max_iter",
        "method",
        "angle",
        "n_jobs",
    ):
        assert setting in tsne_calls[0][0]
    assert len(save_calls) == 1
    assert save_calls[0].startswith("gaussian_progress_tsne_contours-")
    assert real_scatter_markers == ["o", "X", "o", "X"]
    assert contour_save_counts and set(contour_save_counts) == {0}
    assert result["visualization_enabled"]
    assert result["visualization_num_real_points"] == 8
    assert result["visualization_num_synthetic_points"] == 8
    assert result["visualization_num_tsne_points"] == 18
    assert result["output_real_visualization_path"] is None
    assert result["output_real_video_idx_visualization_path"] is None
    assert result["real_visualization_num_tsne_points"] == 0
    assert result["real_visualization_perplexity_used"] is None
    visualization_path = Path(result["output_visualization_path"])
    assert "covariance_mode-independent" in visualization_path.name
    assert visualization_path.is_file()
    assert visualization_path.stat().st_size > 0


def test_open_tsne_fits_real_once_and_reuses_coordinates(tmp_path, monkeypatch):
    expert_h5_path = tmp_path / "expert.h5"
    _write_expert_h5(
        expert_h5_path,
        {
            "000001": (
                np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.2],
                        [0.2, 1.0],
                    ]
                ),
                np.arange(3),
            ),
            "000002": (
                np.array(
                    [
                        [4.0, 4.0],
                        [5.0, 4.2],
                        [4.2, 5.0],
                    ]
                ),
                np.arange(3),
            ),
        },
    )

    open_tsne_calls = []
    transform_shapes = []

    class FakeEmbedding:
        def __init__(self, coordinates):
            self.coordinates = np.asarray(coordinates, dtype=np.float64)

        def __array__(self, dtype=None, copy=None):
            del copy
            return np.asarray(self.coordinates, dtype=dtype)

        def transform(self, values):
            transform_shapes.append(values.shape)
            return np.asarray(values[:, :2], dtype=np.float64)

    class FakeOpenTSNE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, values):
            open_tsne_calls.append((self.kwargs, values.shape, values.copy()))
            return FakeEmbedding(values[:, :2])

    fake_module = types.ModuleType("openTSNE")
    fake_module.TSNE = FakeOpenTSNE
    monkeypatch.setitem(sys.modules, "openTSNE", fake_module)

    def fail_sklearn_tsne(**kwargs):
        raise AssertionError(f"sklearn TSNE must not run: {kwargs}")

    monkeypatch.setattr("sklearn.manifold.TSNE", fail_sklearn_tsne)

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    save_calls = []
    real_scatter_coordinates = []
    video_idx_colors = []
    original_savefig = Figure.savefig
    original_scatter = Axes.scatter

    def recording_savefig(figure, path, *args, **kwargs):
        save_calls.append(Path(path).name)
        return original_savefig(figure, path, *args, **kwargs)

    def recording_scatter(axis, x, y, *args, **kwargs):
        if kwargs.get("marker") == "o":
            if "c" in kwargs:
                video_idx_colors.append(np.asarray(kwargs["c"]))
            else:
                real_scatter_coordinates.append(np.column_stack([x, y]))
        return original_scatter(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", recording_savefig)
    monkeypatch.setattr(Axes, "scatter", recording_scatter)

    task = GaussianProgressFittingTask()
    task.configure(
        {
            "expert_h5_path": str(expert_h5_path),
            "output_dir": str(tmp_path / "outputs"),
            **_base_config(),
            "enable_visualization": True,
            "tsne_viz": {
                "gaussian_samples_per_bin": 20,
                "random_seed": 7,
                "preprocessing": {
                    "standardize": True,
                    "use_pca_before_tsne": False,
                },
                "tsne": {
                    "use_open_tsne": True,
                    "perplexity_mode": "config_clamped",
                    "perplexity": 500,
                    "learning_rate": "auto",
                    "init": "pca",
                    "max_iter": 250,
                },
                "plot": {
                    "enable_real_only_debug": True,
                    "enable_real_video_idx_plot": True,
                    "dpi": 50,
                },
            },
        }
    )
    result = task.evaluate(None)

    assert len(open_tsne_calls) == 1
    kwargs, fit_shape, processed_real = open_tsne_calls[0]
    assert fit_shape == (6, 2)
    np.testing.assert_allclose(processed_real.mean(axis=0), 0.0, atol=1.0e-12)
    assert kwargs["perplexity"] == 5.0
    assert kwargs["initialization"] == "pca"
    assert kwargs["n_iter"] == 250
    assert transform_shapes == [(42, 2)]
    assert save_calls[0].startswith("gaussian_progress_tsne_real_only-")
    assert save_calls[1].startswith(
        "gaussian_progress_tsne_real_only_video_idx-"
    )
    assert save_calls[2].startswith("gaussian_progress_tsne_contours-")
    assert len(video_idx_colors) == 1
    np.testing.assert_array_equal(video_idx_colors[0], [0, 0, 0, 1, 1, 1])
    assert len(real_scatter_coordinates) == 4
    np.testing.assert_allclose(
        real_scatter_coordinates[0], real_scatter_coordinates[2]
    )
    np.testing.assert_allclose(
        real_scatter_coordinates[1], real_scatter_coordinates[3]
    )
    assert result["visualization_num_tsne_points"] == 6
    assert result["visualization_num_real_points"] == 6
    assert result["visualization_num_synthetic_points"] == 40
    assert result["visualization_perplexity_used"] == 5.0
    assert result["real_visualization_num_tsne_points"] == 6
    assert result["real_visualization_perplexity_used"] == 5.0
    video_idx_path = Path(result["output_real_video_idx_visualization_path"])
    assert video_idx_path.is_file()
    assert video_idx_path.stat().st_size > 0


def test_open_tsne_defaults_to_disabled_for_legacy_configs():
    parsed = _parse_visualization_config({"enable_visualization": True})
    assert not parsed["use_open_tsne"]
    assert not parsed["enable_real_only_debug"]
    assert not parsed["enable_real_video_idx_plot"]


def test_visualization_config_accepts_matching_real_bin_counts():
    parsed = _parse_visualization_config(
        {
            "enable_visualization": True,
            "tsne_viz": {
                "gaussian_samples_per_bin": "match_real_bin_counts",
            },
        }
    )
    assert parsed["gaussian_samples_per_bin"] is None
    assert parsed["match_real_bin_counts"]


def test_one_dimensional_tsne_uses_random_initialization(monkeypatch):
    calls = {}

    class FakeSklearnTSNE:
        def __init__(self, **kwargs):
            calls["sklearn_init"] = kwargs["init"]

        def fit_transform(self, values):
            return np.zeros((values.shape[0], 2), dtype=np.float64)

    class FakeEmbedding:
        def __init__(self, num_rows):
            self.coordinates = np.zeros((num_rows, 2), dtype=np.float64)

        def __array__(self, dtype=None, copy=None):
            del copy
            return np.asarray(self.coordinates, dtype=dtype)

        def transform(self, values):
            return np.zeros((values.shape[0], 2), dtype=np.float64)

    class FakeOpenTSNE:
        def __init__(self, **kwargs):
            calls["open_init"] = kwargs["initialization"]

        def fit(self, values):
            return FakeEmbedding(values.shape[0])

    fake_module = types.ModuleType("openTSNE")
    fake_module.TSNE = FakeOpenTSNE
    monkeypatch.setitem(sys.modules, "openTSNE", fake_module)
    monkeypatch.setattr("sklearn.manifold.TSNE", FakeSklearnTSNE)

    visualization_config = {
        "standardize": False,
        "use_pca_before_tsne": False,
        "pca_dim": 1,
        "perplexity_mode": "config_clamped",
        "perplexity": 2.0,
        "learning_rate": "auto",
        "init": "pca",
        "random_seed": 7,
        "max_iter": 250,
    }
    real_features = np.arange(6, dtype=np.float64).reshape(-1, 1)
    out_of_sample_features = np.array([[0.5], [1.5]], dtype=np.float64)

    sklearn_coordinates, _ = _fit_tsne_projection(
        real_features,
        visualization_config,
        fit_label="one-dimensional test",
    )
    open_coordinates, transformed_coordinates, _ = (
        _fit_open_tsne_reference_projection(
            real_features,
            out_of_sample_features,
            visualization_config,
        )
    )

    assert calls == {"sklearn_init": "random", "open_init": "random"}
    assert sklearn_coordinates.shape == (6, 2)
    assert open_coordinates.shape == (6, 2)
    assert transformed_coordinates.shape == (2, 2)
