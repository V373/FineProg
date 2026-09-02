"""Online progress prediction from an offline-fitted Gaussian H5 model.

This task never fits or modifies Gaussian statistics. It consumes the final
covariance matrices stored by the offline fitting task and applies them to
calibration and non-expert embeddings.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import solve_triangular

from fineprog.algos.eval_task.base_task import BaseTask
from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_fitting import (
    _compute_temporal_progress,
)
from fineprog.utils.embedding_normalization import (
    read_embedding_normalization,
    validate_embeddings_for_normalization,
)

_PROJ_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DATASET_NAMES = (
    "bin_progress_values",
    "bin_means",
    "bin_independent_covariances",
    "shared_covariance",
    "bin_final_covariances",
    "bin_log_determinants",
    "bin_counts",
)
_PROGRESS_TOLERANCE = 1.0e-12
_POSTERIOR_RTOL = 1.0e-10
_POSTERIOR_ATOL = 1.0e-12
_SYMMETRY_RTOL = 1.0e-8
_SYMMETRY_ATOL = 1.0e-10


def _read_positive_integer_attr(
    attrs: h5py.AttributeManager,
    name: str,
) -> int:
    """Read one required positive-integer model attribute."""
    if name not in attrs:
        raise ValueError(
            f"[gaussian_progress_pred] PCA Gaussian model is missing root "
            f"attribute {name!r}."
        )
    value = attrs[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"[gaussian_progress_pred] root attribute {name!r} must be a "
            f"positive integer; got {value!r}."
        )
    value = int(value)
    if value < 1:
        raise ValueError(
            f"[gaussian_progress_pred] root attribute {name!r} must be >=1; "
            f"got {value}."
        )
    return value


def _parse_pred_config(config: dict) -> dict:
    """Validate and normalize the online-inference configuration."""
    gaussian_model_h5_path = config.get("gaussian_model_h5_path")
    if not gaussian_model_h5_path:
        raise ValueError("[gaussian_progress_pred] gaussian_model_h5_path is required.")

    nonexpert_h5_path = config.get("nonexpert_h5_path")
    if not nonexpert_h5_path:
        raise ValueError("[gaussian_progress_pred] nonexpert_h5_path is required.")

    try:
        posterior_temperature = float(config.get("posterior_temperature", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "[gaussian_progress_pred] posterior_temperature must be a finite "
            "number > 0."
        ) from exc
    if not np.isfinite(posterior_temperature) or posterior_temperature <= 0.0:
        raise ValueError(
            "[gaussian_progress_pred] posterior_temperature must be finite and > 0; "
            f"got {posterior_temperature}."
        )

    try:
        entropy_epsilon = float(config.get("entropy_epsilon", 1.0e-12))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "[gaussian_progress_pred] entropy_epsilon must be a finite number > 0."
        ) from exc
    if not np.isfinite(entropy_epsilon) or entropy_epsilon <= 0.0:
        raise ValueError(
            "[gaussian_progress_pred] entropy_epsilon must be finite and > 0; "
            f"got {entropy_epsilon}."
        )

    enable_calibration = config.get("enable_calibration", False)
    if not isinstance(enable_calibration, (bool, np.bool_)):
        raise ValueError("[gaussian_progress_pred] enable_calibration must be boolean.")
    enable_calibration = bool(enable_calibration)
    calibration_h5_path = config.get("calibration_h5_path")
    if enable_calibration and not calibration_h5_path:
        raise ValueError(
            "[gaussian_progress_pred] calibration_h5_path is required when "
            "enable_calibration=true."
        )
    ood_p_value_threshold = None
    if enable_calibration:
        if config.get("ood_p_value_threshold") is None:
            raise ValueError(
                "[gaussian_progress_pred] ood_p_value_threshold is required "
                "when enable_calibration=true."
            )
        try:
            ood_p_value_threshold = float(config["ood_p_value_threshold"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "[gaussian_progress_pred] ood_p_value_threshold must be a "
                "finite number in (0, 1)."
            ) from exc
        if (
            not np.isfinite(ood_p_value_threshold)
            or ood_p_value_threshold <= 0.0
            or ood_p_value_threshold >= 1.0
        ):
            raise ValueError(
                "[gaussian_progress_pred] ood_p_value_threshold must be finite "
                f"and in (0, 1); got {ood_p_value_threshold}."
            )

    save_posterior = config.get("save_posterior", True)
    if not isinstance(save_posterior, (bool, np.bool_)):
        raise ValueError("[gaussian_progress_pred] save_posterior must be boolean.")

    enable_visualization = config.get("enable_visualization", False)
    if not isinstance(enable_visualization, (bool, np.bool_)):
        raise ValueError(
            "[gaussian_progress_pred] enable_visualization must be boolean."
        )
    enable_visualization = bool(enable_visualization)
    enable_video = config.get("enable_video", False)
    if not isinstance(enable_video, (bool, np.bool_)):
        raise ValueError("[gaussian_progress_pred] enable_video must be boolean.")
    enable_video = bool(enable_video)
    if enable_video and not enable_visualization:
        raise ValueError(
            "[gaussian_progress_pred] enable_video=true requires "
            "enable_visualization=true."
        )
    if enable_visualization and not bool(save_posterior):
        raise ValueError(
            "[gaussian_progress_pred] enable_visualization=true requires "
            "save_posterior=true."
        )
    if enable_visualization and not enable_calibration:
        raise ValueError(
            "[gaussian_progress_pred] enable_visualization=true requires "
            "enable_calibration=true so the confidence figure can use "
            "conformal p-values."
        )

    return {
        "gaussian_model_h5_path": str(gaussian_model_h5_path),
        "nonexpert_h5_path": str(nonexpert_h5_path),
        "posterior_temperature": posterior_temperature,
        "entropy_epsilon": entropy_epsilon,
        "enable_calibration": enable_calibration,
        "calibration_h5_path": (
            str(calibration_h5_path) if calibration_h5_path else None
        ),
        "ood_p_value_threshold": ood_p_value_threshold,
        "save_posterior": bool(save_posterior),
        "enable_visualization": enable_visualization,
        "enable_video": enable_video,
    }


def _read_gaussian_model(gaussian_model_h5_path: str) -> dict:
    """Load and validate the saved online-inference Gaussian statistics."""
    model_path = Path(gaussian_model_h5_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"[gaussian_progress_pred] Gaussian model H5 not found: {model_path}"
        )

    with h5py.File(model_path, "r") as model_file:
        embedding_normalization = read_embedding_normalization(
            model_file,
            str(model_path),
        )
        if "enable_pca" in model_file.attrs:
            enable_pca_attr = model_file.attrs["enable_pca"]
            if not isinstance(enable_pca_attr, (bool, np.bool_)):
                raise ValueError(
                    "[gaussian_progress_pred] root attribute 'enable_pca' "
                    f"must be boolean; got {enable_pca_attr!r}."
                )
            enable_pca = bool(enable_pca_attr)
        else:
            # Backward compatibility for Gaussian models written before PCA
            # metadata was added.
            enable_pca = False

        if "model" not in model_file or not isinstance(model_file["model"], h5py.Group):
            raise ValueError(
                "[gaussian_progress_pred] Gaussian model H5 must contain /model."
            )
        model_group = model_file["model"]

        missing_names = [
            name for name in _MODEL_DATASET_NAMES if name not in model_group
        ]
        if missing_names:
            raise ValueError(
                "[gaussian_progress_pred] Gaussian model H5 is missing required "
                f"/model datasets: {missing_names}."
            )
        non_dataset_names = [
            name
            for name in _MODEL_DATASET_NAMES
            if not isinstance(model_group[name], h5py.Dataset)
        ]
        if non_dataset_names:
            raise ValueError(
                "[gaussian_progress_pred] Required /model entries must be datasets: "
                f"{non_dataset_names}."
            )

        present_pca_names = [
            name for name in ("pca_mean", "pca_components") if name in model_group
        ]
        if enable_pca:
            missing_pca_names = [
                name
                for name in ("pca_mean", "pca_components")
                if name not in model_group
            ]
            if missing_pca_names:
                raise ValueError(
                    "[gaussian_progress_pred] enable_pca=true requires /model "
                    f"datasets: {missing_pca_names}."
                )
            non_dataset_pca_names = [
                name
                for name in ("pca_mean", "pca_components")
                if not isinstance(model_group[name], h5py.Dataset)
            ]
            if non_dataset_pca_names:
                raise ValueError(
                    "[gaussian_progress_pred] PCA /model entries must be "
                    f"datasets: {non_dataset_pca_names}."
                )
        elif present_pca_names:
            raise ValueError(
                "[gaussian_progress_pred] enable_pca=false is inconsistent with "
                f"saved PCA datasets: {present_pca_names}."
            )

        try:
            bin_progress_values = np.asarray(
                model_group["bin_progress_values"][:], dtype=np.float64
            )
            bin_means = np.asarray(model_group["bin_means"][:], dtype=np.float64)
            bin_final_covariances = np.asarray(
                model_group["bin_final_covariances"][:], dtype=np.float64
            )
            bin_log_determinants = np.asarray(
                model_group["bin_log_determinants"][:], dtype=np.float64
            )
            if enable_pca:
                pca_mean = np.asarray(model_group["pca_mean"][:], dtype=np.float64)
                pca_components = np.asarray(
                    model_group["pca_components"][:], dtype=np.float64
                )
            else:
                pca_mean = None
                pca_components = None
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "[gaussian_progress_pred] Online model datasets must be numeric."
            ) from exc

        if enable_pca:
            root_embedding_dim = _read_positive_integer_attr(
                model_file.attrs,
                "embedding_dim",
            )
            input_embedding_dim = _read_positive_integer_attr(
                model_file.attrs,
                "input_embedding_dim",
            )
        else:
            root_embedding_dim = (
                _read_positive_integer_attr(model_file.attrs, "embedding_dim")
                if "embedding_dim" in model_file.attrs
                else None
            )
            input_embedding_dim = (
                _read_positive_integer_attr(
                    model_file.attrs,
                    "input_embedding_dim",
                )
                if "input_embedding_dim" in model_file.attrs
                else None
            )

    if bin_progress_values.ndim != 1:
        raise ValueError(
            "[gaussian_progress_pred] bin_progress_values must have shape [K]; "
            f"got {bin_progress_values.shape}."
        )
    num_bins = int(bin_progress_values.shape[0])
    if num_bins < 2:
        raise ValueError(
            "[gaussian_progress_pred] Gaussian model must contain at least two "
            f"progress bins; got K={num_bins}."
        )

    if bin_means.ndim != 2 or bin_means.shape[0] != num_bins:
        raise ValueError(
            "[gaussian_progress_pred] bin_means must have shape [K, D]; "
            f"got {bin_means.shape} for K={num_bins}."
        )
    embedding_dim = int(bin_means.shape[1])
    if embedding_dim < 1:
        raise ValueError(
            "[gaussian_progress_pred] bin_means must have D>=1; "
            f"got shape {bin_means.shape}."
        )
    if root_embedding_dim is not None and root_embedding_dim != embedding_dim:
        raise ValueError(
            "[gaussian_progress_pred] root attribute 'embedding_dim' does not "
            f"match bin_means: attr={root_embedding_dim}, D={embedding_dim}."
        )

    if enable_pca:
        if input_embedding_dim <= embedding_dim:
            raise ValueError(
                "[gaussian_progress_pred] PCA input_embedding_dim must be "
                "greater than Gaussian embedding_dim; "
                f"got D_in={input_embedding_dim}, D_model={embedding_dim}."
            )
        expected_pca_mean_shape = (input_embedding_dim,)
        if pca_mean.shape != expected_pca_mean_shape:
            raise ValueError(
                "[gaussian_progress_pred] pca_mean must have shape "
                f"{expected_pca_mean_shape}; got {pca_mean.shape}."
            )
        expected_pca_components_shape = (
            embedding_dim,
            input_embedding_dim,
        )
        if pca_components.shape != expected_pca_components_shape:
            raise ValueError(
                "[gaussian_progress_pred] pca_components must have shape "
                f"{expected_pca_components_shape}; got {pca_components.shape}."
            )
        if not np.isfinite(pca_mean).all():
            raise ValueError("[gaussian_progress_pred] pca_mean contains NaN or Inf.")
        if not np.isfinite(pca_components).all():
            raise ValueError(
                "[gaussian_progress_pred] pca_components contains NaN or Inf."
            )
    else:
        if input_embedding_dim is not None and input_embedding_dim != embedding_dim:
            raise ValueError(
                "[gaussian_progress_pred] enable_pca=false requires "
                "input_embedding_dim to equal embedding_dim; "
                f"got D_in={input_embedding_dim}, D={embedding_dim}."
            )
        input_embedding_dim = embedding_dim

    expected_covariance_shape = (num_bins, embedding_dim, embedding_dim)
    if bin_final_covariances.shape != expected_covariance_shape:
        raise ValueError(
            "[gaussian_progress_pred] bin_final_covariances must have shape "
            f"{expected_covariance_shape}; got {bin_final_covariances.shape}."
        )
    if bin_log_determinants.shape != (num_bins,):
        raise ValueError(
            "[gaussian_progress_pred] bin_log_determinants must have shape "
            f"({num_bins},); got {bin_log_determinants.shape}."
        )

    model_arrays = {
        "bin_progress_values": bin_progress_values,
        "bin_means": bin_means,
        "bin_final_covariances": bin_final_covariances,
        "bin_log_determinants": bin_log_determinants,
    }
    for name, array in model_arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"[gaussian_progress_pred] {name} contains NaN or Inf.")

    if np.any(bin_progress_values < 0.0) or np.any(bin_progress_values > 1.0):
        raise ValueError(
            "[gaussian_progress_pred] bin_progress_values must lie in [0, 1]."
        )

    cholesky_factors = np.empty_like(bin_final_covariances)
    for bin_index, covariance in enumerate(bin_final_covariances):
        if not np.allclose(
            covariance,
            covariance.T,
            rtol=_SYMMETRY_RTOL,
            atol=_SYMMETRY_ATOL,
        ):
            max_error = float(np.max(np.abs(covariance - covariance.T)))
            raise ValueError(
                "[gaussian_progress_pred] bin_final_covariances"
                f"[{bin_index}] is not symmetric; max asymmetry={max_error}."
            )
        try:
            cholesky_factors[bin_index] = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "[gaussian_progress_pred] Cholesky decomposition failed for "
                f"bin_final_covariances[{bin_index}]."
            ) from exc

    return {
        **model_arrays,
        "cholesky_factors": cholesky_factors,
        "num_bins": num_bins,
        "enable_pca": enable_pca,
        "input_embedding_dim": input_embedding_dim,
        "embedding_dim": embedding_dim,
        "embedding_normalization": embedding_normalization,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
    }


def _apply_model_pca(
    raw_query_embeddings: np.ndarray,
    model: dict,
) -> np.ndarray:
    """Project raw queries with the PCA parameters fitted on expert data."""
    if not model["enable_pca"]:
        return raw_query_embeddings

    projected_query_embeddings = (
        raw_query_embeddings - model["pca_mean"][np.newaxis, :]
    ) @ model["pca_components"].T
    expected_shape = (
        int(raw_query_embeddings.shape[0]),
        int(model["embedding_dim"]),
    )
    if projected_query_embeddings.shape != expected_shape:
        raise ValueError(
            "[gaussian_progress_pred] PCA-projected query embeddings must have "
            f"shape {expected_shape}; got {projected_query_embeddings.shape}."
        )
    if not np.isfinite(projected_query_embeddings).all():
        raise ValueError(
            "[gaussian_progress_pred] PCA-projected query embeddings contain "
            "NaN or Inf."
        )
    return np.asarray(projected_query_embeddings, dtype=np.float64)


def _compute_squared_mahalanobis(
    query_embeddings: np.ndarray,
    bin_means: np.ndarray,
    cholesky_factors: np.ndarray,
) -> np.ndarray:
    """Compute [T, K] squared Mahalanobis distances without matrix inverses."""
    num_query_steps = int(query_embeddings.shape[0])
    num_bins = int(bin_means.shape[0])
    distances = np.empty((num_query_steps, num_bins), dtype=np.float64)

    for bin_index in range(num_bins):
        deltas = query_embeddings - bin_means[bin_index]
        whitened_deltas = solve_triangular(
            cholesky_factors[bin_index],
            deltas.T,
            lower=True,
            check_finite=False,
        )
        distances[:, bin_index] = np.einsum(
            "dt,dt->t",
            whitened_deltas,
            whitened_deltas,
            optimize=True,
        )

    if not np.isfinite(distances).all():
        raise ValueError(
            "[gaussian_progress_pred] squared Mahalanobis distances contain "
            "NaN or Inf."
        )
    if np.any(distances < 0.0):
        raise ValueError(
            "[gaussian_progress_pred] squared Mahalanobis distances contain a "
            "negative value."
        )
    return distances


def _compute_conformal_p_values(
    squared_mahalanobis: np.ndarray,
    calibration_distance_bins: list[np.ndarray],
) -> np.ndarray:
    """Calibrate each Gaussian-bin distance against its temporal-bin samples."""
    num_query_steps, num_bins = squared_mahalanobis.shape
    if len(calibration_distance_bins) != num_bins:
        raise ValueError(
            "[gaussian_progress_pred] calibration distance-bin count must match "
            f"K={num_bins}; got {len(calibration_distance_bins)}."
        )

    p_values = np.empty((num_query_steps, num_bins), dtype=np.float64)
    for bin_index, calibration_distances in enumerate(calibration_distance_bins):
        calibration_distances = np.asarray(calibration_distances, dtype=np.float64)
        if calibration_distances.ndim != 1 or calibration_distances.size == 0:
            raise ValueError(
                "[gaussian_progress_pred] every calibration bin must contain "
                f"at least one distance; bin {bin_index} has shape "
                f"{calibration_distances.shape}."
            )
        counts = np.sum(
            squared_mahalanobis[:, bin_index, np.newaxis]
            < calibration_distances[np.newaxis, :],
            axis=1,
        )
        p_values[:, bin_index] = (1.0 + counts) / (1.0 + calibration_distances.size)

    if (
        not np.isfinite(p_values).all()
        or np.any(p_values <= 0.0)
        or np.any(p_values > 1.0)
    ):
        raise ValueError(
            "[gaussian_progress_pred] conformal p-values must be finite and in "
            "(0, 1]."
        )
    return p_values


def _compute_progress_gated(
    progress_mean: np.ndarray,
    is_ood: np.ndarray,
) -> np.ndarray:
    """Hold the last in-distribution progress across OOD frames."""
    progress_gated = np.empty_like(progress_mean)
    last_progress = 0.0
    for frame_index in range(progress_mean.shape[0]):
        if not is_ood[frame_index]:
            last_progress = progress_mean[frame_index]
        progress_gated[frame_index] = last_progress
    return progress_gated


def _compute_gaussian_log_likelihood(
    squared_mahalanobis: np.ndarray,
    bin_log_determinants: np.ndarray,
    embedding_dim: int,
) -> np.ndarray:
    """Compute Gaussian log likelihoods using the saved log determinants."""
    gaussian_constant = float(embedding_dim) * np.log(2.0 * np.pi)
    log_likelihood = -0.5 * (
        squared_mahalanobis + bin_log_determinants[np.newaxis, :] + gaussian_constant
    )
    if not np.isfinite(log_likelihood).all():
        raise ValueError(
            "[gaussian_progress_pred] Gaussian log likelihood contains NaN or Inf."
        )
    return log_likelihood


def _compute_progress_posterior(
    log_likelihood: np.ndarray,
    posterior_temperature: float,
) -> np.ndarray:
    """Apply temperature scaling and a numerically stable uniform-prior softmax."""
    logits = log_likelihood / posterior_temperature
    if not np.isfinite(logits).all():
        raise ValueError(
            "[gaussian_progress_pred] posterior logits contain NaN or Inf."
        )

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    unnormalized = np.exp(shifted_logits)
    denominators = np.sum(unnormalized, axis=1, keepdims=True)
    if not np.isfinite(denominators).all() or np.any(denominators <= 0.0):
        raise ValueError(
            "[gaussian_progress_pred] posterior softmax denominator is invalid."
        )
    posterior = unnormalized / denominators

    if not np.isfinite(posterior).all():
        raise ValueError("[gaussian_progress_pred] posterior contains NaN or Inf.")
    if np.any(posterior < 0.0):
        raise ValueError(
            "[gaussian_progress_pred] posterior contains a negative value."
        )
    if not np.allclose(
        np.sum(posterior, axis=1),
        1.0,
        rtol=_POSTERIOR_RTOL,
        atol=_POSTERIOR_ATOL,
    ):
        max_error = float(np.max(np.abs(np.sum(posterior, axis=1) - 1.0)))
        raise ValueError(
            "[gaussian_progress_pred] posterior rows do not sum to one; "
            f"max error={max_error}."
        )
    return posterior


def _infer_one_trajectory(
    query_embeddings: np.ndarray,
    model: dict,
    posterior_temperature: float,
    entropy_epsilon: float,
    calibration_distance_bins: list[np.ndarray] | None = None,
    ood_p_value_threshold: float | None = None,
) -> dict:
    """Infer progress and support diagnostics for one non-expert trajectory."""
    squared_mahalanobis = _compute_squared_mahalanobis(
        query_embeddings=query_embeddings,
        bin_means=model["bin_means"],
        cholesky_factors=model["cholesky_factors"],
    )
    log_likelihood = _compute_gaussian_log_likelihood(
        squared_mahalanobis=squared_mahalanobis,
        bin_log_determinants=model["bin_log_determinants"],
        embedding_dim=model["embedding_dim"],
    )
    posterior = _compute_progress_posterior(
        log_likelihood=log_likelihood,
        posterior_temperature=posterior_temperature,
    )

    bin_progress_values = model["bin_progress_values"]
    raw_progress_mean = posterior @ bin_progress_values
    if not np.isfinite(raw_progress_mean).all():
        raise ValueError("[gaussian_progress_pred] progress mean contains NaN or Inf.")
    progress_min = float(raw_progress_mean.min())
    progress_max = float(raw_progress_mean.max())
    if progress_min < -_PROGRESS_TOLERANCE or progress_max > 1.0 + _PROGRESS_TOLERANCE:
        raise ValueError(
            "[gaussian_progress_pred] progress mean lies outside [0, 1]; "
            f"range=[{progress_min}, {progress_max}]."
        )
    progress_mean = np.clip(raw_progress_mean, 0.0, 1.0)

    map_bin = np.argmax(posterior, axis=1).astype(np.int64)
    map_progress = bin_progress_values[map_bin]
    centered_progress = bin_progress_values[np.newaxis, :] - progress_mean[:, None]
    progress_variance = np.sum(
        posterior * np.square(centered_progress),
        axis=1,
    )
    posterior_entropy = -np.sum(
        posterior * np.log(posterior + entropy_epsilon),
        axis=1,
    )
    normalized_posterior_entropy = posterior_entropy / np.log(float(model["num_bins"]))

    min_mahalanobis_sq = np.min(squared_mahalanobis, axis=1)
    nearest_mahalanobis_bin = np.argmin(squared_mahalanobis, axis=1).astype(np.int64)

    output_arrays = {
        "gaussian_log_likelihood": log_likelihood,
        "progress_mean": progress_mean,
        "progress_variance": progress_variance,
        "posterior_entropy": posterior_entropy,
        "normalized_posterior_entropy": normalized_posterior_entropy,
        "map_bin": map_bin,
        "map_progress": map_progress,
        "min_mahalanobis_sq": min_mahalanobis_sq,
        "nearest_mahalanobis_bin": nearest_mahalanobis_bin,
        "posterior": posterior,
    }
    if calibration_distance_bins is not None:
        if ood_p_value_threshold is None:
            raise ValueError(
                "[gaussian_progress_pred] ood_p_value_threshold is required "
                "for conformal OOD prediction."
            )
        conformal_p_value = _compute_conformal_p_values(
            squared_mahalanobis=squared_mahalanobis,
            calibration_distance_bins=calibration_distance_bins,
        )
        output_arrays["conformal_p_value"] = conformal_p_value
        is_ood = np.all(
            conformal_p_value < ood_p_value_threshold,
            axis=1,
        )
        output_arrays["is_ood"] = is_ood
        output_arrays["progress_gated"] = _compute_progress_gated(
            progress_mean,
            is_ood,
        )
    for name, array in output_arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(
                f"[gaussian_progress_pred] inferred {name} contains NaN or Inf."
            )
    if np.any(progress_variance < 0.0):
        raise ValueError(
            "[gaussian_progress_pred] progress variance contains a negative value."
        )
    return output_arrays


def _read_nonexpert_video(
    videos_group: h5py.Group,
    video_id: str,
    expected_input_embedding_dim: int,
    embedding_normalization: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate one non-expert embedding trajectory."""
    video_group = videos_group[video_id]
    if not isinstance(video_group, h5py.Group):
        raise ValueError(
            f"[gaussian_progress_pred] /videos/{video_id} must be a group."
        )
    missing_names = [
        name for name in ("embeddings", "target_steps") if name not in video_group
    ]
    if missing_names:
        raise ValueError(
            f"[gaussian_progress_pred] /videos/{video_id} is missing datasets: "
            f"{missing_names}."
        )

    try:
        query_embeddings = np.asarray(video_group["embeddings"][:], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[gaussian_progress_pred] /videos/{video_id}/embeddings must be numeric."
        ) from exc
    if query_embeddings.ndim != 2:
        raise ValueError(
            f"[gaussian_progress_pred] /videos/{video_id}/embeddings must have "
            f"shape [T_q, D]; got {query_embeddings.shape}."
        )
    num_query_steps, embedding_dim = query_embeddings.shape
    if num_query_steps < 1:
        raise ValueError(
            f"[gaussian_progress_pred] /videos/{video_id}/embeddings must have "
            "T_q>=1."
        )
    if embedding_dim != expected_input_embedding_dim:
        raise ValueError(
            f"[gaussian_progress_pred] /videos/{video_id} embedding dimension "
            "mismatch before PCA: expected "
            f"D_in={expected_input_embedding_dim}, got D={embedding_dim}."
        )
    if not np.isfinite(query_embeddings).all():
        raise ValueError(
            f"[gaussian_progress_pred] /videos/{video_id}/embeddings contains "
            "NaN or Inf."
        )
    validate_embeddings_for_normalization(
        query_embeddings,
        embedding_normalization,
        f"{videos_group.file.filename}:/videos/{video_id}/embeddings",
    )

    target_steps = np.asarray(video_group["target_steps"][:])
    if target_steps.shape != (num_query_steps,):
        raise ValueError(
            f"[gaussian_progress_pred] /videos/{video_id}/target_steps must have "
            f"shape ({num_query_steps},); got {target_steps.shape}."
        )
    if not np.issubdtype(target_steps.dtype, np.number):
        raise ValueError(
            f"[gaussian_progress_pred] /videos/{video_id}/target_steps must be numeric."
        )
    if not np.isfinite(target_steps).all():
        raise ValueError(
            f"[gaussian_progress_pred] /videos/{video_id}/target_steps contains "
            "NaN or Inf."
        )
    return query_embeddings, target_steps


def _read_calibration_distance_bins(
    calibration_h5_path: str,
    model: dict,
) -> list[np.ndarray]:
    """Read calibration trajectories and collect matching-bin distances."""
    calibration_path = Path(calibration_h5_path).expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(
            "[gaussian_progress_pred] calibration H5 not found: " f"{calibration_path}"
        )

    num_bins = int(model["num_bins"])
    per_bin_parts: list[list[np.ndarray]] = [[] for _ in range(num_bins)]
    with h5py.File(calibration_path, "r") as calibration_file:
        calibration_normalization = read_embedding_normalization(
            calibration_file,
            str(calibration_path),
        )
        if calibration_normalization != model["embedding_normalization"]:
            raise ValueError(
                "[gaussian_progress_pred] embedding_normalization mismatch: "
                f"Gaussian model is {model['embedding_normalization']!r}, "
                f"calibration H5 is {calibration_normalization!r}."
            )
        if "videos" not in calibration_file or not isinstance(
            calibration_file["videos"], h5py.Group
        ):
            raise ValueError(
                "[gaussian_progress_pred] calibration H5 must contain /videos."
            )
        videos_group = calibration_file["videos"]
        video_ids = sorted(videos_group.keys())
        if not video_ids:
            raise ValueError(
                "[gaussian_progress_pred] calibration H5 has an empty /videos " "group."
            )

        for video_id in video_ids:
            raw_embeddings, target_steps = _read_nonexpert_video(
                videos_group=videos_group,
                video_id=video_id,
                expected_input_embedding_dim=model["input_embedding_dim"],
                embedding_normalization=calibration_normalization,
            )
            embeddings = _apply_model_pca(raw_embeddings, model)
            distances = _compute_squared_mahalanobis(
                query_embeddings=embeddings,
                bin_means=model["bin_means"],
                cholesky_factors=model["cholesky_factors"],
            )
            progress, _ = _compute_temporal_progress(
                video_id=video_id,
                num_embeddings=embeddings.shape[0],
                target_steps=target_steps,
            )
            bin_indices = np.minimum(
                np.floor(num_bins * progress).astype(np.int64),
                num_bins - 1,
            )
            for bin_index in range(num_bins):
                selected = distances[bin_indices == bin_index, bin_index]
                if selected.size:
                    per_bin_parts[bin_index].append(selected)

    empty_bins = [
        bin_index for bin_index, parts in enumerate(per_bin_parts) if not parts
    ]
    if empty_bins:
        raise ValueError(
            "[gaussian_progress_pred] calibration temporal bins have no "
            f"samples: {empty_bins}."
        )
    return [np.concatenate(parts) for parts in per_bin_parts]


def _build_output_paths(
    gaussian_model_h5_path: str,
    nonexpert_h5_path: str,
    timestamp: str,
) -> tuple[Path, Path, Path]:
    """Build the self-contained output directory, final H5, and temporary H5."""
    output_dir = (
        _PROJ_ROOT
        / "outputs"
        / "gaussian_progress_pred"
        / Path(gaussian_model_h5_path).stem
        / Path(nonexpert_h5_path).stem
        / timestamp
    )
    output_h5_path = output_dir / f"gaussian_progress_pred-{timestamp}.h5"
    temporary_h5_path = output_dir / f".{output_h5_path.name}.tmp"
    return output_dir, output_h5_path, temporary_h5_path


def _normalized_cell_edges(values: np.ndarray) -> np.ndarray:
    """Return cell edges for ordered sample centers, preserving the axis range."""
    edges = np.empty(values.size + 1, dtype=np.float64)
    if values.size == 0:
        return edges
    edges[0] = values[0] - 0.5
    edges[-1] = values[-1] + 0.5
    if values.size > 1:
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    return edges


def _frame_tick_values(num_frames: int) -> tuple[np.ndarray, list[str]]:
    """Build a compact, frame-based tick grid for a single video's x-axis."""
    if num_frames <= 1:
        ticks = np.asarray([0.0], dtype=np.float64)
        labels = ["0"]
        return ticks, labels

    tick_count = min(5, num_frames)
    tick_positions = np.linspace(0.0, float(num_frames - 1), tick_count)
    tick_positions = np.rint(tick_positions).astype(np.int64)
    unique_positions = np.unique(tick_positions)
    return unique_positions.astype(np.float64), [
        str(int(pos)) for pos in unique_positions
    ]


def _create_progress_axes(plt, figsize: tuple[float, float] = (8.0, 4.0)):
    """Create aligned progress axes with one external heatmap colorbar."""
    figure = plt.figure(figsize=figsize, dpi=300, constrained_layout=True)
    layout = figure.add_gridspec(
        2,
        2,
        height_ratios=(1, 1),
        width_ratios=(1, 0.035),
        hspace=0.12,
        wspace=0.12,
    )
    curve_axis = figure.add_subplot(layout[0, 0])
    heatmap_axis = figure.add_subplot(layout[1, 0], sharex=curve_axis)
    colorbar_axis = figure.add_subplot(layout[1, 1])

    curve_axis.tick_params(axis="x", which="both", labelbottom=False)
    return figure, curve_axis, heatmap_axis, colorbar_axis


def _create_gated_progress_axes(
    plt,
    figsize: tuple[float, float] = (8.0, 6.5),
):
    """Create aligned OOD, gated-progress, mean-progress, and heatmap axes."""
    figure = plt.figure(figsize=figsize, dpi=300, constrained_layout=True)
    layout = figure.add_gridspec(
        4,
        2,
        height_ratios=(1, 6, 6, 6),
        width_ratios=(1, 0.035),
        hspace=0.10,
        wspace=0.12,
    )
    ood_axis = figure.add_subplot(layout[0, 0])
    gated_curve_axis = figure.add_subplot(layout[1, 0], sharex=ood_axis)
    curve_axis = figure.add_subplot(layout[2, 0], sharex=ood_axis)
    heatmap_axis = figure.add_subplot(layout[3, 0], sharex=ood_axis)
    colorbar_axis = figure.add_subplot(layout[3, 1])

    for axis in (ood_axis, gated_curve_axis, curve_axis):
        axis.tick_params(
            axis="x",
            which="both",
            bottom=False,
            labelbottom=False,
        )
    return (
        figure,
        ood_axis,
        gated_curve_axis,
        curve_axis,
        heatmap_axis,
        colorbar_axis,
    )


def _create_confidence_axes(
    plt,
    figsize: tuple[float, float] = (8.0, 4.5),
):
    """Create aligned OOD, p-value, and likelihood axes."""
    figure = plt.figure(figsize=figsize, dpi=300, constrained_layout=True)
    layout = figure.add_gridspec(
        3,
        2,
        height_ratios=(1, 6, 6),
        width_ratios=(1, 0.035),
        hspace=0.10,
        wspace=0.12,
    )
    ood_axis = figure.add_subplot(layout[0, 0])
    p_value_axis = figure.add_subplot(layout[1, 0], sharex=ood_axis)
    likelihood_axis = figure.add_subplot(layout[2, 0], sharex=ood_axis)
    p_value_colorbar_axis = figure.add_subplot(layout[1, 1])
    likelihood_colorbar_axis = figure.add_subplot(layout[2, 1])

    for axis in (ood_axis, p_value_axis):
        axis.tick_params(
            axis="x",
            which="both",
            bottom=False,
            labelbottom=False,
        )
    return (
        figure,
        ood_axis,
        p_value_axis,
        likelihood_axis,
        p_value_colorbar_axis,
        likelihood_colorbar_axis,
    )


def _build_progress_figure(
    plt,
    video_id: str,
    frame_steps: np.ndarray,
    bin_progress_values: np.ndarray,
    progress_mean: np.ndarray,
    posterior: np.ndarray,
    progress_gated: np.ndarray | None = None,
    is_ood: np.ndarray | None = None,
    progress_curve_label: str = "Progress mean",
    figsize: tuple[float, float] = (8.0, 6.5),
    show_title: bool = True,
    cursor_x: float | None = None,
):
    """Build a posterior-progress figure, optionally with OOD gating panels."""
    if (progress_gated is None) != (is_ood is None):
        raise ValueError(
            "[gaussian_progress_pred] progress figure requires both "
            "progress_gated and is_ood."
        )

    frame_edges = _normalized_cell_edges(frame_steps)
    progress_edges = _normalized_cell_edges(bin_progress_values)
    posterior_min = float(np.min(posterior))
    posterior_max = float(np.max(posterior))
    x_ticks, x_tick_labels = _frame_tick_values(frame_steps.size)

    if progress_gated is None:
        figure, curve_axis, heatmap_axis, colorbar_axis = _create_progress_axes(
            plt,
            figsize=figsize,
        )
        curve_axis.plot(frame_steps, progress_mean, color="tab:blue")
        curve_axis.set_ylabel(progress_curve_label)
        curve_axis.set_ylim(0.0, 1.0)
        title_axis = curve_axis
        cursor_axes = (curve_axis, heatmap_axis)
    else:
        from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: PLC0415

        progress_gated = np.asarray(progress_gated)
        is_ood = np.asarray(is_ood)
        if progress_gated.shape != frame_steps.shape:
            raise ValueError(
                "[gaussian_progress_pred] progress figure progress_gated shape "
                f"must match frame_steps shape {frame_steps.shape}; got "
                f"{progress_gated.shape}."
            )
        if is_ood.shape != frame_steps.shape:
            raise ValueError(
                "[gaussian_progress_pred] progress figure is_ood shape must "
                f"match frame_steps shape {frame_steps.shape}; got {is_ood.shape}."
            )
        (
            figure,
            ood_axis,
            gated_curve_axis,
            curve_axis,
            heatmap_axis,
            colorbar_axis,
        ) = _create_gated_progress_axes(plt, figsize=figsize)
        ood_axis.pcolormesh(
            frame_edges,
            np.asarray([0.0, 1.0]),
            is_ood.astype(np.int8, copy=False)[np.newaxis, :],
            cmap=ListedColormap(("white", "#d62728")),
            norm=BoundaryNorm((-0.5, 0.5, 1.5), ncolors=2),
        )
        ood_axis.set_ylim(0.0, 1.0)
        ood_axis.set_yticks([])
        ood_axis.set_ylabel("")

        gated_curve_axis.plot(frame_steps, progress_gated, color="tab:blue")
        gated_curve_axis.set_ylabel("Progress gated")
        gated_curve_axis.set_ylim(0.0, 1.0)
        curve_axis.plot(frame_steps, progress_mean, color="tab:blue")
        curve_axis.set_ylabel("Progress mean")
        curve_axis.set_ylim(0.0, 1.0)
        title_axis = ood_axis
        cursor_axes = (ood_axis, gated_curve_axis, curve_axis, heatmap_axis)

    if show_title:
        title_axis.set_title(f"Posterior progress — {video_id}")
    title_axis.set_xlim(frame_steps[0], frame_steps[-1])

    heatmap = heatmap_axis.pcolormesh(
        frame_edges,
        progress_edges,
        posterior.T,
        cmap="Blues",
        vmin=posterior_min,
        vmax=posterior_max,
    )
    heatmap_axis.set_xlabel("Frame number")
    heatmap_axis.set_ylabel("Progress")
    heatmap_axis.set_xlim(frame_steps[0], frame_steps[-1])
    heatmap_axis.set_ylim(0.0, 1.0)
    heatmap_axis.set_xticks(x_ticks)
    heatmap_axis.set_xticklabels(x_tick_labels)
    figure.colorbar(
        heatmap,
        cax=colorbar_axis,
        label="Posterior probability",
    )

    cursor_lines = []
    if cursor_x is not None:
        for axis in cursor_axes:
            cursor_lines.append(
                axis.axvline(
                    cursor_x,
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    zorder=10,
                )
            )
    return figure, cursor_lines


def _save_progress_figure(
    video_id: str,
    frame_steps: np.ndarray,
    bin_progress_values: np.ndarray,
    progress_mean: np.ndarray,
    progress_gated: np.ndarray,
    posterior: np.ndarray,
    is_ood: np.ndarray,
    output_path: Path,
) -> None:
    """Save the OOD-gated posterior-progress figure."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure, _ = _build_progress_figure(
        plt=plt,
        video_id=video_id,
        frame_steps=frame_steps,
        bin_progress_values=bin_progress_values,
        progress_mean=progress_mean,
        posterior=posterior,
        progress_gated=progress_gated,
        is_ood=is_ood,
    )
    try:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def _build_confidence_figure(
    plt,
    video_id: str,
    frame_steps: np.ndarray,
    bin_progress_values: np.ndarray,
    is_ood: np.ndarray,
    conformal_p_value: np.ndarray,
    gaussian_log_likelihood: np.ndarray,
    figsize: tuple[float, float] = (8.0, 4.5),
    show_title: bool = True,
    cursor_x: float | None = None,
):
    """Build OOD, conformal p-value, and likelihood heatmaps."""
    from matplotlib.colors import (  # noqa: PLC0415
        BoundaryNorm,
        ListedColormap,
        SymLogNorm,
    )

    is_ood = np.asarray(is_ood)
    if is_ood.shape != frame_steps.shape:
        raise ValueError(
            "[gaussian_progress_pred] confidence figure is_ood shape must "
            f"match frame_steps shape {frame_steps.shape}; got {is_ood.shape}."
        )

    frame_edges = _normalized_cell_edges(frame_steps)
    progress_edges = _normalized_cell_edges(bin_progress_values)
    p_value_min = float(np.min(conformal_p_value))
    p_value_max = float(np.max(conformal_p_value))
    likelihood_min = float(np.min(gaussian_log_likelihood))
    likelihood_max = float(np.max(gaussian_log_likelihood))
    (
        figure,
        ood_axis,
        p_value_axis,
        likelihood_axis,
        p_value_colorbar_axis,
        likelihood_colorbar_axis,
    ) = _create_confidence_axes(plt, figsize=figsize)
    x_ticks, x_tick_labels = _frame_tick_values(frame_steps.size)
    ood_axis.pcolormesh(
        frame_edges,
        np.asarray([0.0, 1.0]),
        is_ood.astype(np.int8, copy=False)[np.newaxis, :],
        cmap=ListedColormap(("white", "#d62728")),
        norm=BoundaryNorm((-0.5, 0.5, 1.5), ncolors=2),
    )
    ood_axis.set_xlim(frame_steps[0], frame_steps[-1])
    ood_axis.set_ylim(0.0, 1.0)
    ood_axis.set_yticks([])
    ood_axis.set_ylabel("")

    p_value_heatmap = p_value_axis.pcolormesh(
        frame_edges,
        progress_edges,
        conformal_p_value.T,
        cmap="Blues",
        vmin=p_value_min,
        vmax=p_value_max,
    )
    p_value_axis.set_ylabel("Progress")
    p_value_axis.set_xlim(frame_steps[0], frame_steps[-1])
    p_value_axis.set_ylim(0.0, 1.0)
    p_value_axis.set_xticks(x_ticks)
    p_value_axis.set_xticklabels(x_tick_labels)
    figure.colorbar(
        p_value_heatmap,
        cax=p_value_colorbar_axis,
        label="Conformal p-value",
    )

    likelihood_heatmap = likelihood_axis.pcolormesh(
        frame_edges,
        progress_edges,
        gaussian_log_likelihood.T,
        cmap="Blues",
        norm=SymLogNorm(
            linthresh=1.0,
            linscale=1.0,
            base=10,
            vmin=likelihood_min,
            vmax=likelihood_max,
        ),
    )
    likelihood_axis.set_xlabel("Frame number")
    likelihood_axis.set_ylabel("Progress")
    likelihood_axis.set_xlim(frame_steps[0], frame_steps[-1])
    likelihood_axis.set_ylim(0.0, 1.0)
    likelihood_axis.set_xticks(x_ticks)
    likelihood_axis.set_xticklabels(x_tick_labels)
    figure.colorbar(
        likelihood_heatmap,
        cax=likelihood_colorbar_axis,
        label="Gaussian log likelihood",
    )
    if show_title:
        figure.suptitle(f"Progress confidence — {video_id}")

    cursor_lines = []
    if cursor_x is not None:
        for axis in (ood_axis, p_value_axis, likelihood_axis):
            cursor_lines.append(
                axis.axvline(
                    cursor_x,
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    zorder=10,
                )
            )
    return figure, cursor_lines


def _save_confidence_figure(
    video_id: str,
    frame_steps: np.ndarray,
    bin_progress_values: np.ndarray,
    is_ood: np.ndarray,
    conformal_p_value: np.ndarray,
    gaussian_log_likelihood: np.ndarray,
    output_path: Path,
) -> None:
    """Save OOD, conformal p-value, and Gaussian likelihood heatmaps."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure, _ = _build_confidence_figure(
        plt=plt,
        video_id=video_id,
        frame_steps=frame_steps,
        bin_progress_values=bin_progress_values,
        is_ood=is_ood,
        conformal_p_value=conformal_p_value,
        gaussian_log_likelihood=gaussian_log_likelihood,
    )
    try:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def _text_attr(attrs: h5py.AttributeManager, name: str, context: str) -> str:
    """Read one required, non-empty text attribute."""
    if name not in attrs:
        raise ValueError(
            f"[gaussian_progress_pred] {context} is missing attribute {name!r}."
        )
    value = attrs[name]
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"[gaussian_progress_pred] {context} attribute {name!r} must be "
            f"non-empty text; got {value!r}."
        )
    return value


def _processed_h5_path_from_embedding(nonexpert_h5_path: str) -> Path:
    """Locate the processed H5 associated with one embedding H5."""
    embedding_stem = Path(nonexpert_h5_path).stem
    if "-embd" not in embedding_stem:
        raise ValueError(
            "[gaussian_progress_pred] nonexpert embedding H5 stem must contain "
            f"'-embd' to locate its processed H5; got {embedding_stem!r}."
        )
    processed_stem = embedding_stem.split("-embd", maxsplit=1)[0]
    processed_root = _PROJ_ROOT / "datasets" / "processed"
    candidate_paths = (
        processed_root / f"{processed_stem}.h5",
        processed_root / "metaworld" / f"{processed_stem}.h5",
    )
    processed_h5_path = next(
        (path for path in candidate_paths if path.is_file()),
        None,
    )
    if processed_h5_path is None:
        raise FileNotFoundError(
            "[gaussian_progress_pred] processed H5 inferred from the nonexpert "
            "embedding H5 was not found. Checked: "
            + ", ".join(str(path) for path in candidate_paths)
        )
    return processed_h5_path


def _resolve_raw_video_path(
    processed_videos_group: h5py.Group,
    video_id: str,
) -> tuple[Path, int]:
    """Resolve one processed video ID to its current raw MP4 and frame count."""
    if video_id not in processed_videos_group:
        raise ValueError(
            f"[gaussian_progress_pred] processed H5 is missing /videos/{video_id}."
        )
    processed_video_group = processed_videos_group[video_id]
    if not isinstance(processed_video_group, h5py.Group):
        raise ValueError(
            f"[gaussian_progress_pred] processed /videos/{video_id} must be a group."
        )
    context = f"processed /videos/{video_id}"
    action_name = _text_attr(processed_video_group.attrs, "action_name", context)
    stale_source_path = Path(_text_attr(processed_video_group.attrs, "path", context))
    if stale_source_path.stem != action_name:
        raise ValueError(
            f"[gaussian_progress_pred] {context} action_name {action_name!r} "
            f"does not match path {str(stale_source_path)!r}."
        )
    raw_dir_name = stale_source_path.parent.name
    if raw_dir_name in ("", ".", ".."):
        raise ValueError(
            f"[gaussian_progress_pred] {context} path does not identify a raw "
            f"video directory: {str(stale_source_path)!r}."
        )
    raw_video_path = stale_source_path
    if not raw_video_path.is_file():
        raw_video_path = (
            _PROJ_ROOT / "datasets" / "raw" / raw_dir_name / stale_source_path.name
        )
    if not raw_video_path.is_file():
        raise FileNotFoundError(
            f"[gaussian_progress_pred] raw video not found: {raw_video_path}"
        )

    if "frames" not in processed_video_group or not isinstance(
        processed_video_group["frames"], h5py.Dataset
    ):
        raise ValueError(
            f"[gaussian_progress_pred] {context} must contain a frames dataset."
        )
    processed_frame_count = int(processed_video_group["frames"].shape[0])
    if "num_frames" not in processed_video_group.attrs:
        raise ValueError(
            f"[gaussian_progress_pred] {context} is missing attribute 'num_frames'."
        )
    num_frames_attr = processed_video_group.attrs["num_frames"]
    if isinstance(num_frames_attr, (bool, np.bool_)) or not isinstance(
        num_frames_attr,
        (int, np.integer),
    ):
        raise ValueError(
            f"[gaussian_progress_pred] {context} attribute 'num_frames' must be "
            f"an integer; got {num_frames_attr!r}."
        )
    if int(num_frames_attr) != processed_frame_count:
        raise ValueError(
            f"[gaussian_progress_pred] {context} frame count mismatch: "
            f"frames has {processed_frame_count}, num_frames is {num_frames_attr}."
        )
    return raw_video_path, processed_frame_count


def _resize_rgb_to_height(frame: np.ndarray, target_height: int, cv2) -> np.ndarray:
    """Resize one RGB image proportionally to a fixed height."""
    height, width = frame.shape[:2]
    target_width = max(1, int(round(width * target_height / float(height))))
    return cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA if target_height < height else cv2.INTER_LINEAR,
    )


def _save_prediction_video(
    video_id: str,
    raw_video_path: Path,
    processed_frame_count: int,
    target_steps: np.ndarray,
    bin_progress_values: np.ndarray,
    progress_gated: np.ndarray,
    posterior: np.ndarray,
    is_ood: np.ndarray,
    conformal_p_value: np.ndarray,
    gaussian_log_likelihood: np.ndarray,
    output_path: Path,
) -> None:
    """Render one raw-video, progress, and confidence composite MP4."""
    import cv2  # noqa: PLC0415
    import imageio.v2 as imageio  # noqa: PLC0415
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    target_steps = np.asarray(target_steps)
    prediction_frame_count = int(progress_gated.shape[0])
    expected_steps = np.arange(prediction_frame_count, dtype=np.int64)
    if target_steps.shape != expected_steps.shape or not np.array_equal(
        target_steps,
        expected_steps,
    ):
        raise ValueError(
            f"[gaussian_progress_pred] video {video_id} requires one prediction "
            "per raw frame: target_steps must equal arange(T)."
        )
    if processed_frame_count != prediction_frame_count:
        raise ValueError(
            f"[gaussian_progress_pred] video {video_id} frame count mismatch: "
            f"processed={processed_frame_count}, prediction={prediction_frame_count}."
        )

    capture = cv2.VideoCapture(str(raw_video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"[gaussian_progress_pred] failed to open raw video: {raw_video_path}"
        )
    raw_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    raw_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if raw_frame_count != processed_frame_count:
        capture.release()
        raise ValueError(
            f"[gaussian_progress_pred] video {video_id} frame count mismatch: "
            f"raw={raw_frame_count}, processed={processed_frame_count}, "
            f"prediction={prediction_frame_count}."
        )
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        capture.release()
        raise ValueError(
            f"[gaussian_progress_pred] raw video FPS must be positive; "
            f"got {source_fps} for {raw_video_path}."
        )
    target_height = raw_height - raw_height % 2
    if target_height < 2:
        capture.release()
        raise ValueError(
            f"[gaussian_progress_pred] raw video height must be >=2; "
            f"got {raw_height} for {raw_video_path}."
        )

    frame_steps = expected_steps.astype(np.float64)
    with matplotlib.rc_context(
        {
            "font.size": 6.0,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
        }
    ):
        progress_figure, progress_lines = _build_progress_figure(
            plt=plt,
            video_id=video_id,
            frame_steps=frame_steps,
            bin_progress_values=bin_progress_values,
            progress_mean=progress_gated,
            posterior=posterior,
            progress_curve_label="Progress gated",
            figsize=(4.0, 2.0),
            show_title=False,
            cursor_x=0.0,
        )
        confidence_figure, confidence_lines = _build_confidence_figure(
            plt=plt,
            video_id=video_id,
            frame_steps=frame_steps,
            bin_progress_values=bin_progress_values,
            is_ood=is_ood,
            conformal_p_value=conformal_p_value,
            gaussian_log_likelihood=gaussian_log_likelihood,
            figsize=(4.0, 2.25),
            show_title=False,
            cursor_x=0.0,
        )

    for line in progress_lines + confidence_lines:
        line.set_animated(True)
    progress_figure.canvas.draw()
    progress_background = progress_figure.canvas.copy_from_bbox(progress_figure.bbox)
    confidence_figure.canvas.draw()
    confidence_background = confidence_figure.canvas.copy_from_bbox(
        confidence_figure.bbox
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    if temporary_path.exists():
        temporary_path.unlink()
    writer = None
    try:
        writer = imageio.get_writer(
            str(temporary_path),
            fps=source_fps,
            format="ffmpeg",
            codec="libx264",
            macro_block_size=1,
        )
        for frame_index in range(prediction_frame_count):
            ok, raw_bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"[gaussian_progress_pred] raw video {raw_video_path} ended "
                    f"before frame {frame_index}."
                )
            for line in progress_lines + confidence_lines:
                line.set_xdata([frame_index, frame_index])

            progress_figure.canvas.restore_region(progress_background)
            for line in progress_lines:
                line.axes.draw_artist(line)
            progress_figure.canvas.blit(progress_figure.bbox)
            progress_panel = np.asarray(progress_figure.canvas.buffer_rgba()).copy()[
                :, :, :3
            ]
            confidence_figure.canvas.restore_region(confidence_background)
            for line in confidence_lines:
                line.axes.draw_artist(line)
            confidence_figure.canvas.blit(confidence_figure.bbox)
            confidence_panel = np.asarray(
                confidence_figure.canvas.buffer_rgba()
            ).copy()[:, :, :3]
            raw_panel = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

            panels = [
                _resize_rgb_to_height(raw_panel, target_height, cv2),
                _resize_rgb_to_height(progress_panel, target_height, cv2),
                _resize_rgb_to_height(confidence_panel, target_height, cv2),
            ]
            combined = np.concatenate(panels, axis=1)
            if combined.shape[1] % 2:
                combined = combined[:, :-1]
            writer.append_data(np.ascontiguousarray(combined, dtype=np.uint8))

        extra_ok, _ = capture.read()
        if extra_ok:
            raise ValueError(
                f"[gaussian_progress_pred] raw video {raw_video_path} contains "
                "more frames than reported."
            )
        writer.close()
        writer = None
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            writer = None
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.close()
        plt.close(progress_figure)
        plt.close(confidence_figure)

    temporary_path.replace(output_path)


def _save_prediction_videos(output_h5_path: Path) -> None:
    """Render a three-panel MP4 for every prediction in one output H5."""
    output_h5_path = Path(output_h5_path).expanduser().resolve()
    if not output_h5_path.is_file():
        raise FileNotFoundError(
            f"[gaussian_progress_pred] prediction H5 not found: {output_h5_path}"
        )
    videos_dir = output_h5_path.parent / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5_path, "r") as output_file:
        nonexpert_h5_path = _text_attr(
            output_file.attrs,
            "nonexpert_h5_path",
            "prediction H5 root",
        )
        processed_h5_path = _processed_h5_path_from_embedding(nonexpert_h5_path)
        bin_progress_values = np.asarray(
            output_file["model/bin_progress_values"][:], dtype=np.float64
        )
        nonexperts_group = output_file["nonexperts"]

        with h5py.File(processed_h5_path, "r") as processed_file:
            if "videos" not in processed_file or not isinstance(
                processed_file["videos"], h5py.Group
            ):
                raise ValueError(
                    "[gaussian_progress_pred] processed H5 must contain /videos."
                )
            processed_videos_group = processed_file["videos"]
            for video_id in sorted(nonexperts_group.keys()):
                video_group = nonexperts_group[video_id]
                raw_video_path, processed_frame_count = _resolve_raw_video_path(
                    processed_videos_group,
                    video_id,
                )
                print(
                    f"[gaussian_progress_pred] rendering video {video_id}: "
                    f"{raw_video_path.name}"
                )
                _save_prediction_video(
                    video_id=video_id,
                    raw_video_path=raw_video_path,
                    processed_frame_count=processed_frame_count,
                    target_steps=np.asarray(video_group["target_steps"][:]),
                    bin_progress_values=bin_progress_values,
                    progress_gated=np.asarray(
                        video_group["progress_gated"][:], dtype=np.float64
                    ),
                    posterior=np.asarray(video_group["posterior"][:], dtype=np.float64),
                    is_ood=np.asarray(video_group["is_ood"][:], dtype=np.bool_),
                    conformal_p_value=np.asarray(
                        video_group["conformal_p_value"][:], dtype=np.float64
                    ),
                    gaussian_log_likelihood=np.asarray(
                        video_group["gaussian_log_likelihood"][:],
                        dtype=np.float64,
                    ),
                    output_path=videos_dir / f"{video_id}.mp4",
                )

    print(f"[gaussian_progress_pred] videos              : {videos_dir}")


def _save_prediction_visualizations(output_h5_path: Path) -> None:
    """Render both requested figures for every saved non-expert prediction."""
    figures_dir = output_h5_path.parent / "figures"
    progress_dir = figures_dir / "progress"
    confidence_dir = figures_dir / "confidence"
    progress_dir.mkdir(parents=True, exist_ok=False)
    confidence_dir.mkdir()

    with h5py.File(output_h5_path, "r") as output_file:
        bin_progress_values = np.asarray(
            output_file["model/bin_progress_values"][:], dtype=np.float64
        )
        nonexperts_group = output_file["nonexperts"]
        video_ids = sorted(nonexperts_group.keys())
        for video_id in video_ids:
            video_group = nonexperts_group[video_id]
            progress_mean = np.asarray(
                video_group["progress_mean"][:], dtype=np.float64
            )
            progress_gated = np.asarray(
                video_group["progress_gated"][:], dtype=np.float64
            )
            posterior = np.asarray(video_group["posterior"][:], dtype=np.float64)
            conformal_p_value = np.asarray(
                video_group["conformal_p_value"][:], dtype=np.float64
            )
            is_ood = np.asarray(video_group["is_ood"][:], dtype=np.bool_)
            gaussian_log_likelihood = np.asarray(
                video_group["gaussian_log_likelihood"][:], dtype=np.float64
            )
            frame_steps = np.arange(progress_mean.shape[0], dtype=np.float64)

            _save_progress_figure(
                video_id=video_id,
                frame_steps=frame_steps,
                bin_progress_values=bin_progress_values,
                progress_mean=progress_mean,
                progress_gated=progress_gated,
                posterior=posterior,
                is_ood=is_ood,
                output_path=progress_dir / f"{video_id}.png",
            )
            _save_confidence_figure(
                video_id=video_id,
                frame_steps=frame_steps,
                bin_progress_values=bin_progress_values,
                is_ood=is_ood,
                conformal_p_value=conformal_p_value,
                gaussian_log_likelihood=gaussian_log_likelihood,
                output_path=confidence_dir / f"{video_id}.png",
            )

    print(f"[gaussian_progress_pred] progress figures     : {progress_dir}")
    print(f"[gaussian_progress_pred] confidence figures   : {confidence_dir}")


class GaussianProgressPredTask(BaseTask):
    """Predict continuous progress from saved progress-conditioned Gaussians."""

    def __init__(self):
        super().__init__(task_name="gaussian_progress_pred", downstream_task=False)
        self.config: dict = {}

    def configure(self, config: dict) -> None:
        """Store the resolved V2 evaluation config."""
        self.config = dict(config)

    def evaluate(self, embeddings_dataset=None) -> dict:  # noqa: ARG002
        if not self.config:
            raise ValueError(
                "[gaussian_progress_pred] task must be configured before evaluate()."
            )
        pred_config = _parse_pred_config(self.config)
        gaussian_model_h5_path = str(
            Path(pred_config["gaussian_model_h5_path"]).expanduser().resolve()
        )
        nonexpert_h5_path = str(
            Path(pred_config["nonexpert_h5_path"]).expanduser().resolve()
        )
        nonexpert_path = Path(nonexpert_h5_path)
        if not nonexpert_path.is_file():
            raise FileNotFoundError(
                f"[gaussian_progress_pred] non-expert H5 not found: {nonexpert_path}"
            )

        model = _read_gaussian_model(gaussian_model_h5_path)
        calibration_h5_path = None
        calibration_distance_bins = None
        if pred_config["enable_calibration"]:
            calibration_h5_path = str(
                Path(pred_config["calibration_h5_path"]).expanduser().resolve()
            )
            calibration_distance_bins = _read_calibration_distance_bins(
                calibration_h5_path=calibration_h5_path,
                model=model,
            )
        print()
        print(
            "[gaussian_progress_pred] gaussian_model_h5_path:",
            gaussian_model_h5_path,
        )
        print(
            "[gaussian_progress_pred] nonexpert_h5_path    :",
            nonexpert_h5_path,
        )
        if calibration_h5_path is not None:
            print(
                "[gaussian_progress_pred] calibration_h5_path  :",
                calibration_h5_path,
            )
            print(
                "[gaussian_progress_pred] calibration counts   :",
                [values.size for values in calibration_distance_bins],
            )
        print(
            "[gaussian_progress_pred] model shape          : "
            f"K={model['num_bins']}, D_in={model['input_embedding_dim']}, "
            f"D_model={model['embedding_dim']}, PCA={model['enable_pca']}"
        )

        with h5py.File(nonexpert_path, "r") as nonexpert_file:
            query_embedding_normalization = read_embedding_normalization(
                nonexpert_file,
                str(nonexpert_path),
            )
            if query_embedding_normalization != model["embedding_normalization"]:
                raise ValueError(
                    "[gaussian_progress_pred] embedding_normalization mismatch: "
                    f"Gaussian model is {model['embedding_normalization']!r}, "
                    f"query H5 is {query_embedding_normalization!r}."
                )
            if "videos" not in nonexpert_file or not isinstance(
                nonexpert_file["videos"], h5py.Group
            ):
                raise ValueError(
                    "[gaussian_progress_pred] non-expert H5 must contain /videos."
                )
            videos_group = nonexpert_file["videos"]
            video_ids = sorted(videos_group.keys())
            if not video_ids:
                raise ValueError(
                    "[gaussian_progress_pred] non-expert H5 has an empty /videos group."
                )

            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir, output_h5_path, temporary_h5_path = _build_output_paths(
                gaussian_model_h5_path=gaussian_model_h5_path,
                nonexpert_h5_path=nonexpert_h5_path,
                timestamp=timestamp,
            )
            output_dir.mkdir(parents=True, exist_ok=False)
            print(f"[gaussian_progress_pred] non-expert videos    : {len(video_ids)}")
            print(f"[gaussian_progress_pred] output_h5_path      : {output_h5_path}")

            support_sum = 0.0
            total_query_steps = 0
            try:
                with h5py.File(temporary_h5_path, "w-") as output_file:
                    output_file.attrs["task_name"] = "gaussian_progress_pred"
                    output_file.attrs["embedding_normalization"] = model[
                        "embedding_normalization"
                    ]
                    output_file.attrs["gaussian_model_h5_path"] = gaussian_model_h5_path
                    output_file.attrs["nonexpert_h5_path"] = nonexpert_h5_path
                    output_file.attrs["num_bins"] = int(model["num_bins"])
                    output_file.attrs["enable_pca"] = bool(model["enable_pca"])
                    output_file.attrs["input_embedding_dim"] = int(
                        model["input_embedding_dim"]
                    )
                    output_file.attrs["embedding_dim"] = int(model["embedding_dim"])
                    output_file.attrs["posterior_temperature"] = float(
                        pred_config["posterior_temperature"]
                    )
                    output_file.attrs["entropy_epsilon"] = float(
                        pred_config["entropy_epsilon"]
                    )
                    output_file.attrs["enable_calibration"] = bool(
                        pred_config["enable_calibration"]
                    )
                    if calibration_h5_path is not None:
                        output_file.attrs["calibration_h5_path"] = calibration_h5_path
                        output_file.attrs["ood_p_value_threshold"] = float(
                            pred_config["ood_p_value_threshold"]
                        )
                    output_file.attrs["save_posterior"] = bool(
                        pred_config["save_posterior"]
                    )

                    output_model_group = output_file.create_group("model")
                    output_model_group.create_dataset(
                        "bin_progress_values",
                        data=model["bin_progress_values"],
                    )
                    output_model_group.create_dataset(
                        "bin_means",
                        data=model["bin_means"],
                        compression="gzip",
                    )
                    output_model_group.create_dataset(
                        "bin_final_covariances",
                        data=model["bin_final_covariances"],
                        compression="gzip",
                    )
                    output_model_group.create_dataset(
                        "bin_log_determinants",
                        data=model["bin_log_determinants"],
                    )
                    if model["enable_pca"]:
                        output_model_group.create_dataset(
                            "pca_mean",
                            data=model["pca_mean"],
                        )
                        output_model_group.create_dataset(
                            "pca_components",
                            data=model["pca_components"],
                        )

                    output_nonexperts_group = output_file.create_group("nonexperts")
                    for video_id in video_ids:
                        raw_query_embeddings, target_steps = _read_nonexpert_video(
                            videos_group=videos_group,
                            video_id=video_id,
                            expected_input_embedding_dim=model["input_embedding_dim"],
                            embedding_normalization=query_embedding_normalization,
                        )
                        query_embeddings = _apply_model_pca(
                            raw_query_embeddings=raw_query_embeddings,
                            model=model,
                        )
                        inferred = _infer_one_trajectory(
                            query_embeddings=query_embeddings,
                            model=model,
                            posterior_temperature=pred_config["posterior_temperature"],
                            entropy_epsilon=pred_config["entropy_epsilon"],
                            calibration_distance_bins=calibration_distance_bins,
                            ood_p_value_threshold=pred_config["ood_p_value_threshold"],
                        )

                        output_video_group = output_nonexperts_group.create_group(
                            video_id
                        )
                        output_video_group.create_dataset(
                            "target_steps", data=target_steps
                        )
                        output_video_group.create_dataset(
                            "progress_label", data=inferred["progress_mean"]
                        )
                        output_video_group.create_dataset(
                            "progress_mean", data=inferred["progress_mean"]
                        )
                        output_video_group.create_dataset(
                            "progress_variance", data=inferred["progress_variance"]
                        )
                        output_video_group.create_dataset(
                            "posterior_entropy", data=inferred["posterior_entropy"]
                        )
                        output_video_group.create_dataset(
                            "normalized_posterior_entropy",
                            data=inferred["normalized_posterior_entropy"],
                        )
                        output_video_group.create_dataset(
                            "map_bin", data=inferred["map_bin"]
                        )
                        output_video_group.create_dataset(
                            "map_progress", data=inferred["map_progress"]
                        )
                        output_video_group.create_dataset(
                            "min_mahalanobis_sq",
                            data=inferred["min_mahalanobis_sq"],
                        )
                        output_video_group.create_dataset(
                            "nearest_mahalanobis_bin",
                            data=inferred["nearest_mahalanobis_bin"],
                        )
                        output_video_group.create_dataset(
                            "gaussian_log_likelihood",
                            data=inferred["gaussian_log_likelihood"],
                            compression="gzip",
                        )
                        if pred_config["save_posterior"]:
                            output_video_group.create_dataset(
                                "posterior",
                                data=inferred["posterior"],
                                compression="gzip",
                            )
                        if pred_config["enable_calibration"]:
                            output_video_group.create_dataset(
                                "conformal_p_value",
                                data=inferred["conformal_p_value"],
                                compression="gzip",
                            )
                            output_video_group.create_dataset(
                                "is_ood",
                                data=inferred["is_ood"],
                            )
                            output_video_group.create_dataset(
                                "progress_gated",
                                data=inferred["progress_gated"],
                            )

                        current_steps = int(query_embeddings.shape[0])
                        current_support_sum = float(
                            np.sum(
                                inferred["min_mahalanobis_sq"],
                                dtype=np.float64,
                            )
                        )
                        support_sum += current_support_sum
                        total_query_steps += current_steps
                        print(
                            f"  [video {video_id}] T_q={current_steps}, "
                            "mean_min_mahalanobis_sq="
                            f"{current_support_sum / current_steps:.6f}"
                        )

                temporary_h5_path.replace(output_h5_path)
            except Exception:
                if temporary_h5_path.exists():
                    temporary_h5_path.unlink()
                raise

        if pred_config["enable_visualization"]:
            _save_prediction_visualizations(output_h5_path)
        if pred_config["enable_video"]:
            _save_prediction_videos(output_h5_path)

        if total_query_steps < 1:
            raise ValueError(
                "[gaussian_progress_pred] no non-expert query steps were evaluated."
            )
        global_mean_min_mahalanobis_sq = support_sum / float(total_query_steps)
        print(
            "[gaussian_progress_pred] global_mean_min_mahalanobis_sq: "
            f"{global_mean_min_mahalanobis_sq:.6f}"
        )
        print("[gaussian_progress_pred] saved output H5: " f"{output_h5_path}")
        return {
            "task_name": "gaussian_progress_pred",
            "metric_name": "global_mean_min_mahalanobis_sq",
            "metric_value": float(global_mean_min_mahalanobis_sq),
            "output_h5_path": str(output_h5_path),
        }
