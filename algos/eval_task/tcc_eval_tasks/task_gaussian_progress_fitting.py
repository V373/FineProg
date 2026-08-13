"""Offline fitting for a progress-conditioned Gaussian embedding model.

This evaluation task consumes only a standard multi-video expert embedding H5.
It assigns expert features to temporal-progress bins, estimates one full
Gaussian covariance per bin plus a pooled within-bin covariance, and writes the
fitted statistics to an H5 model artifact.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path

import h5py
import numpy as np

from fineprog.algos.eval_task.base_task import BaseTask
from fineprog.utils.embedding_normalization import (
    read_embedding_normalization,
    validate_embeddings_for_normalization,
)


_PROJ_ROOT = Path(__file__).resolve().parents[3]
_SUPPORTED_COVARIANCE_MODES = {
    "independent",
    "shared",
    "independent_shared_weighted",
}
_PROGRESS_TOLERANCE = 1.0e-12
_SYMMETRY_RTOL = 1.0e-8
_SYMMETRY_ATOL = 1.0e-10


def _parse_fitting_config(config: dict) -> dict:
    """Validate and normalize the Gaussian fitting configuration."""
    enable_pca = config.get("enable_pca", False)
    if not isinstance(enable_pca, (bool, np.bool_)):
        raise ValueError(
            "[gaussian_progress_fitting] enable_pca must be boolean."
        )
    enable_pca = bool(enable_pca)

    raw_pca_dim = config.get("pca_dim", 8)
    if isinstance(raw_pca_dim, bool):
        raise ValueError(
            "[gaussian_progress_fitting] pca_dim must be a positive integer."
        )
    try:
        pca_dim_float = float(raw_pca_dim)
        pca_dim = int(raw_pca_dim)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "[gaussian_progress_fitting] pca_dim must be a positive integer."
        ) from exc
    if (
        not np.isfinite(pca_dim_float)
        or pca_dim_float != pca_dim
        or pca_dim < 1
    ):
        raise ValueError(
            "[gaussian_progress_fitting] pca_dim must be a positive integer; "
            f"got {raw_pca_dim!r}."
        )

    raw_num_bins = config.get("num_bins", 20)
    if isinstance(raw_num_bins, bool):
        raise ValueError("[gaussian_progress_fitting] num_bins must be an integer >= 2.")
    try:
        num_bins_float = float(raw_num_bins)
        num_bins = int(raw_num_bins)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "[gaussian_progress_fitting] num_bins must be an integer >= 2."
        ) from exc
    if not np.isfinite(num_bins_float) or num_bins_float != num_bins or num_bins < 2:
        raise ValueError(
            f"[gaussian_progress_fitting] num_bins must be an integer >= 2; got {raw_num_bins!r}."
        )

    covariance_mode = str(config.get("covariance_mode", "independent")).strip().lower()
    if covariance_mode not in _SUPPORTED_COVARIANCE_MODES:
        raise ValueError(
            "[gaussian_progress_fitting] covariance_mode must be one of "
            f"{sorted(_SUPPORTED_COVARIANCE_MODES)}; got {covariance_mode!r}."
        )

    shared_weight = float(config.get("shared_covariance_weight", 0.5))
    if not np.isfinite(shared_weight) or not 0.0 <= shared_weight <= 1.0:
        raise ValueError(
            "[gaussian_progress_fitting] shared_covariance_weight must be finite "
            f"and in [0, 1]; got {shared_weight}."
        )

    variance_floor = float(config.get("variance_floor", 1.0e-8))
    if not np.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError(
            "[gaussian_progress_fitting] variance_floor must be finite and > 0; "
            f"got {variance_floor}."
        )

    enable_regularization = config.get("enable_covariance_regularization", False)
    if not isinstance(enable_regularization, (bool, np.bool_)):
        raise ValueError(
            "[gaussian_progress_fitting] enable_covariance_regularization must be boolean."
        )
    enable_regularization = bool(enable_regularization)

    covariance_regularization = float(config.get("covariance_regularization", 1.0e-6))
    if not np.isfinite(covariance_regularization) or covariance_regularization < 0.0:
        raise ValueError(
            "[gaussian_progress_fitting] covariance_regularization must be finite "
            f"and >= 0; got {covariance_regularization}."
        )

    raw_covariance_rank_tolerance = config.get(
        "covariance_rank_tolerance", 1.0e-5
    )
    if isinstance(raw_covariance_rank_tolerance, (bool, np.bool_)):
        raise ValueError(
            "[gaussian_progress_fitting] covariance_rank_tolerance must be a "
            "finite number >= 0."
        )
    try:
        covariance_rank_tolerance = float(raw_covariance_rank_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "[gaussian_progress_fitting] covariance_rank_tolerance must be a "
            "finite number >= 0."
        ) from exc
    if (
        not np.isfinite(covariance_rank_tolerance)
        or covariance_rank_tolerance < 0.0
    ):
        raise ValueError(
            "[gaussian_progress_fitting] covariance_rank_tolerance must be a "
            "finite number >= 0; "
            f"got {raw_covariance_rank_tolerance!r}."
        )

    enable_calibration = config.get("enable_calibration", False)
    if not isinstance(enable_calibration, (bool, np.bool_)):
        raise ValueError("[gaussian_progress_fitting] enable_calibration must be boolean.")
    if bool(enable_calibration):
        raise NotImplementedError(
            "[gaussian_progress_fitting] enable_calibration=true is not supported "
            "during offline fitting phase 1."
        )

    return {
        "enable_pca": enable_pca,
        "pca_dim": pca_dim,
        "num_bins": num_bins,
        "covariance_mode": covariance_mode,
        "shared_covariance_weight": shared_weight,
        "variance_floor": variance_floor,
        "enable_covariance_regularization": enable_regularization,
        "covariance_regularization": covariance_regularization,
        "covariance_rank_tolerance": covariance_rank_tolerance,
        "enable_calibration": False,
    }


def _build_covariance_artifact_tag(fitting_config: dict) -> str:
    """Build a readable filename tag for the fitted covariance configuration."""
    covariance_mode = fitting_config["covariance_mode"]
    tag = f"covariance_mode-{covariance_mode}"
    if covariance_mode == "independent_shared_weighted":
        shared_weight = format(
            float(fitting_config["shared_covariance_weight"]), ".12g"
        )
        tag += f"-shared_covariance_weight-{shared_weight}"
    return tag


def _compute_temporal_progress(
    video_id: str,
    num_embeddings: int,
    target_steps: np.ndarray | None,
) -> tuple[np.ndarray, str | None]:
    """Compute normalized temporal progress and report any fallback reason."""
    if num_embeddings < 2:
        raise ValueError(
            f"[gaussian_progress_fitting] video '{video_id}' must contain at least "
            f"2 embeddings; got T={num_embeddings}."
        )

    fallback_reason = None
    if target_steps is None:
        fallback_reason = "missing target_steps"
    else:
        steps_raw = np.asarray(target_steps)
        if steps_raw.ndim != 1 or steps_raw.shape[0] != num_embeddings:
            fallback_reason = (
                f"target_steps shape mismatch: expected ({num_embeddings},), "
                f"got {steps_raw.shape}"
            )
        else:
            try:
                steps = steps_raw.astype(np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"[gaussian_progress_fitting] video '{video_id}' target_steps "
                    "must be numeric."
                ) from exc
            if not np.isfinite(steps).all():
                raise ValueError(
                    f"[gaussian_progress_fitting] video '{video_id}' target_steps "
                    "contains NaN or Inf."
                )
            if steps[-1] == steps[0]:
                fallback_reason = "target_steps first and last values are equal"

    if fallback_reason is not None:
        progress = np.arange(num_embeddings, dtype=np.float64) / float(num_embeddings - 1)
    else:
        progress = (steps - steps[0]) / (steps[-1] - steps[0])

    if not np.isfinite(progress).all():
        raise ValueError(
            f"[gaussian_progress_fitting] video '{video_id}' normalized progress "
            "contains NaN or Inf."
        )
    progress_min = float(progress.min())
    progress_max = float(progress.max())
    if progress_min < -_PROGRESS_TOLERANCE or progress_max > 1.0 + _PROGRESS_TOLERANCE:
        raise ValueError(
            f"[gaussian_progress_fitting] video '{video_id}' normalized progress "
            f"must lie in [0, 1]; got range [{progress_min}, {progress_max}]."
        )
    return np.clip(progress, 0.0, 1.0), fallback_reason


def _read_expert_trajectories(
    expert_h5_path: str,
) -> tuple[
    list[str],
    list[np.ndarray],
    list[np.ndarray],
    list[tuple[str, str]],
    str,
]:
    """Read and validate all expert embedding trajectories from an H5 file."""
    path = Path(expert_h5_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"[gaussian_progress_fitting] expert H5 file not found: {path}"
        )

    video_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    progresses: list[np.ndarray] = []
    fallbacks: list[tuple[str, str]] = []
    embedding_dim: int | None = None

    with h5py.File(path, "r") as h5_file:
        embedding_normalization = read_embedding_normalization(
            h5_file,
            str(path),
        )
        if "videos" not in h5_file or not isinstance(h5_file["videos"], h5py.Group):
            raise ValueError(
                f"[gaussian_progress_fitting] expert H5 must contain a /videos group: {path}"
            )
        videos_group = h5_file["videos"]
        sorted_video_ids = sorted(videos_group.keys())
        if not sorted_video_ids:
            raise ValueError(
                f"[gaussian_progress_fitting] expert H5 has an empty /videos group: {path}"
            )

        for video_id in sorted_video_ids:
            video_group = videos_group[video_id]
            if not isinstance(video_group, h5py.Group) or "embeddings" not in video_group:
                raise ValueError(
                    f"[gaussian_progress_fitting] video '{video_id}' is missing embeddings."
                )
            try:
                video_embeddings = np.asarray(
                    video_group["embeddings"], dtype=np.float64
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"[gaussian_progress_fitting] video '{video_id}' embeddings "
                    "must be numeric."
                ) from exc
            if video_embeddings.ndim != 2:
                raise ValueError(
                    f"[gaussian_progress_fitting] video '{video_id}' embeddings must "
                    f"have shape [T, D]; got {video_embeddings.shape}."
                )
            num_embeddings, current_dim = video_embeddings.shape
            if num_embeddings < 2 or current_dim < 1:
                raise ValueError(
                    f"[gaussian_progress_fitting] video '{video_id}' embeddings must "
                    f"have T>=2 and D>=1; got {video_embeddings.shape}."
                )
            if embedding_dim is None:
                embedding_dim = current_dim
            elif current_dim != embedding_dim:
                raise ValueError(
                    f"[gaussian_progress_fitting] inconsistent embedding dimension for "
                    f"video '{video_id}': expected D={embedding_dim}, got D={current_dim}."
                )
            if not np.isfinite(video_embeddings).all():
                raise ValueError(
                    f"[gaussian_progress_fitting] video '{video_id}' embeddings "
                    "contains NaN or Inf."
                )
            validate_embeddings_for_normalization(
                video_embeddings,
                embedding_normalization,
                f"{path}:/videos/{video_id}/embeddings",
            )

            target_steps = (
                np.asarray(video_group["target_steps"])
                if "target_steps" in video_group
                else None
            )
            progress, fallback_reason = _compute_temporal_progress(
                video_id=video_id,
                num_embeddings=num_embeddings,
                target_steps=target_steps,
            )
            if fallback_reason is not None:
                fallbacks.append((video_id, fallback_reason))

            video_ids.append(video_id)
            embeddings.append(video_embeddings)
            progresses.append(progress)

    return (
        video_ids,
        embeddings,
        progresses,
        fallbacks,
        embedding_normalization,
    )


def _split_transformed_embeddings(
    transformed_embeddings: np.ndarray,
    source_embeddings: list[np.ndarray],
) -> list[np.ndarray]:
    """Split concatenated transformed features back into source trajectories."""
    split_embeddings: list[np.ndarray] = []
    offset = 0
    for trajectory in source_embeddings:
        next_offset = offset + trajectory.shape[0]
        split_embeddings.append(transformed_embeddings[offset:next_offset])
        offset = next_offset
    if offset != transformed_embeddings.shape[0]:
        raise ValueError(
            "[gaussian_progress_fitting] transformed PCA feature count does not "
            "match the source trajectories."
        )
    return split_embeddings


def _fit_model_pca(
    embeddings: list[np.ndarray],
    pca_dim: int,
) -> tuple[list[np.ndarray], dict]:
    """Fit PCA on all expert frames and project every trajectory."""
    from sklearn.decomposition import PCA  # noqa: PLC0415

    if not embeddings:
        raise ValueError(
            "[gaussian_progress_fitting] PCA requires at least one trajectory."
        )
    all_embeddings = np.concatenate(embeddings, axis=0).astype(
        np.float64, copy=False
    )
    if all_embeddings.ndim != 2 or not np.isfinite(all_embeddings).all():
        raise ValueError(
            "[gaussian_progress_fitting] PCA input must be a finite [N, D] array."
        )
    num_features, input_embedding_dim = all_embeddings.shape
    max_pca_dim = min(num_features, input_embedding_dim - 1)
    if pca_dim > max_pca_dim:
        raise ValueError(
            "[gaussian_progress_fitting] when enable_pca=true, pca_dim must "
            "satisfy 1 <= pca_dim <= min(num_expert_features, "
            f"input_embedding_dim - 1)={max_pca_dim}; got pca_dim={pca_dim}, "
            f"num_expert_features={num_features}, "
            f"input_embedding_dim={input_embedding_dim}."
        )

    pca = PCA(
        n_components=pca_dim,
        whiten=False,
        svd_solver="full",
    )
    transformed = np.asarray(
        pca.fit_transform(all_embeddings),
        dtype=np.float64,
    )
    pca_mean = np.asarray(pca.mean_, dtype=np.float64)
    pca_components = np.asarray(pca.components_, dtype=np.float64)
    if (
        transformed.shape != (num_features, pca_dim)
        or pca_mean.shape != (input_embedding_dim,)
        or pca_components.shape != (pca_dim, input_embedding_dim)
        or not np.isfinite(transformed).all()
        or not np.isfinite(pca_mean).all()
        or not np.isfinite(pca_components).all()
    ):
        raise ValueError(
            "[gaussian_progress_fitting] fitted PCA produced invalid parameters "
            "or transformed embeddings."
        )

    return _split_transformed_embeddings(transformed, embeddings), {
        "input_embedding_dim": input_embedding_dim,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
    }


def _apply_saved_model_pca(
    embeddings: list[np.ndarray],
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
) -> list[np.ndarray]:
    """Project trajectories with PCA parameters saved in a fitted model."""
    if not embeddings:
        raise ValueError(
            "[gaussian_progress_fitting] saved PCA transform requires at least "
            "one trajectory."
        )
    pca_mean = np.asarray(pca_mean, dtype=np.float64)
    pca_components = np.asarray(pca_components, dtype=np.float64)
    all_embeddings = np.concatenate(embeddings, axis=0).astype(
        np.float64, copy=False
    )
    if (
        pca_mean.ndim != 1
        or pca_components.ndim != 2
        or pca_components.shape[1] != pca_mean.shape[0]
        or all_embeddings.shape[1] != pca_mean.shape[0]
    ):
        raise ValueError(
            "[gaussian_progress_fitting] expert embeddings and saved PCA "
            "parameters have inconsistent dimensions."
        )
    transformed = (all_embeddings - pca_mean[np.newaxis, :]) @ pca_components.T
    if not np.isfinite(transformed).all():
        raise ValueError(
            "[gaussian_progress_fitting] saved PCA transform produced NaN or Inf."
        )
    return _split_transformed_embeddings(transformed, embeddings)


def _validate_covariance_matrix(
    covariance: np.ndarray,
    embedding_dim: int,
    name: str,
    require_positive_definite: bool = False,
) -> None:
    """Validate full covariance shape, finiteness, symmetry, and optional PD."""
    if covariance.shape != (embedding_dim, embedding_dim):
        raise ValueError(
            f"[gaussian_progress_fitting] {name} must have shape "
            f"({embedding_dim}, {embedding_dim}); got {covariance.shape}."
        )
    if not np.isfinite(covariance).all():
        raise ValueError(f"[gaussian_progress_fitting] {name} contains NaN or Inf.")
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=_SYMMETRY_RTOL,
        atol=_SYMMETRY_ATOL,
    ):
        max_error = float(np.max(np.abs(covariance - covariance.T)))
        raise ValueError(
            f"[gaussian_progress_fitting] {name} is not numerically symmetric; "
            f"max asymmetry={max_error}."
        )
    if require_positive_definite:
        eigenvalues = np.linalg.eigvalsh(covariance)
        if not np.isfinite(eigenvalues).all() or float(eigenvalues.min()) <= 0.0:
            raise ValueError(
                f"[gaussian_progress_fitting] {name} must be strictly positive "
                f"definite; min eigenvalue={float(eigenvalues.min())}."
            )


def _apply_covariance_floor_and_regularization(
    covariance: np.ndarray,
    variance_floor: float,
    enable_regularization: bool,
    covariance_regularization: float,
) -> np.ndarray:
    """Apply an eigenvalue floor, then optional isotropic regularization."""
    symmetric_covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariance)
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise ValueError(
            "[gaussian_progress_fitting] covariance eigendecomposition produced "
            "NaN or Inf."
        )
    floored_eigenvalues = np.maximum(eigenvalues, variance_floor)
    floored_covariance = (
        eigenvectors * floored_eigenvalues[np.newaxis, :]
    ) @ eigenvectors.T
    floored_covariance = 0.5 * (floored_covariance + floored_covariance.T)

    if enable_regularization:
        floored_covariance = floored_covariance + (
            covariance_regularization * np.eye(covariance.shape[0], dtype=np.float64)
        )
    return 0.5 * (floored_covariance + floored_covariance.T)


def _assign_progress_bins(
    embeddings: list[np.ndarray],
    progresses: list[np.ndarray],
    num_bins: int,
) -> dict:
    """Concatenate trajectories and apply the canonical progress-bin rule."""
    if not embeddings or len(embeddings) != len(progresses):
        raise ValueError(
            "[gaussian_progress_fitting] at least one matched embedding/progress "
            "trajectory is required."
        )

    all_embeddings = np.concatenate(embeddings, axis=0).astype(np.float64, copy=False)
    all_progress = np.concatenate(progresses, axis=0).astype(np.float64, copy=False)
    embedding_dim = all_embeddings.shape[1]

    if all_embeddings.shape[0] != all_progress.shape[0]:
        raise ValueError(
            "[gaussian_progress_fitting] concatenated embeddings and progress "
            "lengths do not match."
        )
    if not np.isfinite(all_progress).all() or np.any(all_progress < 0.0) or np.any(all_progress > 1.0):
        raise ValueError(
            "[gaussian_progress_fitting] normalized progress must be finite and in [0, 1]."
        )

    bin_indices = np.minimum(
        np.floor(num_bins * all_progress).astype(np.int64),
        num_bins - 1,
    )
    if np.any(bin_indices < 0) or np.any(bin_indices >= num_bins):
        raise ValueError(
            "[gaussian_progress_fitting] computed bin indices fall outside [0, K-1]."
        )
    bin_counts = np.bincount(bin_indices, minlength=num_bins).astype(np.int64)
    undersampled = [
        (bin_index, int(count))
        for bin_index, count in enumerate(bin_counts)
        if count < 2
    ]
    if undersampled:
        detail = ", ".join(
            f"bin {bin_index}: {count}" for bin_index, count in undersampled
        )
        raise ValueError(
            "[gaussian_progress_fitting] every progress bin requires at least "
            f"2 samples; undersampled bins: {detail}."
        )

    return {
        "all_embeddings": all_embeddings,
        "all_progress": all_progress,
        "bin_indices": bin_indices,
        "bin_counts": bin_counts,
        "embedding_dim": embedding_dim,
    }


def _fit_gaussian_progress_model(
    embeddings: list[np.ndarray],
    progresses: list[np.ndarray],
    fitting_config: dict,
) -> dict:
    """Fit full per-bin and pooled Gaussian covariance statistics."""
    num_bins = fitting_config["num_bins"]
    assigned = _assign_progress_bins(embeddings, progresses, num_bins)
    all_embeddings = assigned["all_embeddings"]
    bin_indices = assigned["bin_indices"]
    bin_counts = assigned["bin_counts"]
    embedding_dim = assigned["embedding_dim"]

    bin_means = np.empty((num_bins, embedding_dim), dtype=np.float64)
    bin_independent_covariances = np.empty(
        (num_bins, embedding_dim, embedding_dim), dtype=np.float64
    )
    bin_residual_scatters = np.empty_like(bin_independent_covariances)

    for bin_index in range(num_bins):
        bin_embeddings = all_embeddings[bin_indices == bin_index]
        bin_mean = bin_embeddings.mean(axis=0)
        residuals = bin_embeddings - bin_mean
        residual_scatter = residuals.T @ residuals
        bin_means[bin_index] = bin_mean
        bin_residual_scatters[bin_index] = residual_scatter
        bin_independent_covariances[bin_index] = residual_scatter / float(
            bin_embeddings.shape[0] - 1
        )

    pooled_denominator = int(np.sum(bin_counts - 1))
    shared_covariance = bin_residual_scatters.sum(axis=0) / float(pooled_denominator)

    for bin_index, covariance in enumerate(bin_independent_covariances):
        _validate_covariance_matrix(
            covariance,
            embedding_dim,
            f"bin_independent_covariances[{bin_index}]",
        )
    _validate_covariance_matrix(
        shared_covariance,
        embedding_dim,
        "shared_covariance",
    )

    covariance_mode = fitting_config["covariance_mode"]
    shared_weight = fitting_config["shared_covariance_weight"]
    bin_final_covariances = np.empty_like(bin_independent_covariances)
    bin_log_determinants = np.empty(num_bins, dtype=np.float64)
    bin_final_covariance_ranks = np.empty(num_bins, dtype=np.int64)

    for bin_index in range(num_bins):
        if covariance_mode == "independent":
            base_covariance = bin_independent_covariances[bin_index]
        elif covariance_mode == "shared":
            base_covariance = shared_covariance
        else:
            base_covariance = (
                (1.0 - shared_weight) * bin_independent_covariances[bin_index]
                + shared_weight * shared_covariance
            )

        final_covariance = _apply_covariance_floor_and_regularization(
            covariance=base_covariance,
            variance_floor=fitting_config["variance_floor"],
            enable_regularization=fitting_config[
                "enable_covariance_regularization"
            ],
            covariance_regularization=fitting_config[
                "covariance_regularization"
            ],
        )
        _validate_covariance_matrix(
            final_covariance,
            embedding_dim,
            f"bin_final_covariances[{bin_index}]",
            require_positive_definite=True,
        )
        determinant_sign, log_determinant = np.linalg.slogdet(final_covariance)
        if determinant_sign <= 0.0 or not np.isfinite(log_determinant):
            raise ValueError(
                f"[gaussian_progress_fitting] invalid log determinant for bin "
                f"{bin_index}: sign={determinant_sign}, logdet={log_determinant}."
            )
        bin_final_covariances[bin_index] = final_covariance
        bin_log_determinants[bin_index] = log_determinant
        bin_final_covariance_ranks[bin_index] = np.linalg.matrix_rank(
            final_covariance,
            tol=fitting_config["covariance_rank_tolerance"],
        )

    bin_progress_values = np.arange(num_bins, dtype=np.float64) / float(num_bins - 1)
    return {
        "bin_progress_values": bin_progress_values,
        "bin_means": bin_means,
        "bin_independent_covariances": bin_independent_covariances,
        "shared_covariance": shared_covariance,
        "bin_final_covariances": bin_final_covariances,
        "bin_final_covariance_ranks": bin_final_covariance_ranks,
        "bin_log_determinants": bin_log_determinants,
        "bin_counts": bin_counts,
        "embedding_dim": embedding_dim,
        "num_expert_features": int(all_embeddings.shape[0]),
    }


def _write_gaussian_model_h5(
    output_h5_path: Path,
    model: dict,
    fitting_config: dict,
    expert_h5_path: str,
    num_expert_videos: int,
    embedding_normalization: str,
) -> None:
    """Write the validated Gaussian model to a new H5 artifact."""
    with h5py.File(output_h5_path, "w-") as output_file:
        output_file.attrs["task_name"] = "gaussian_progress_fitting"
        output_file.attrs["embedding_normalization"] = embedding_normalization
        output_file.attrs["expert_h5_path"] = str(expert_h5_path)
        output_file.attrs["num_expert_videos"] = int(num_expert_videos)
        output_file.attrs["num_expert_features"] = int(model["num_expert_features"])
        output_file.attrs["num_bins"] = int(fitting_config["num_bins"])
        output_file.attrs["enable_pca"] = bool(model["enable_pca"])
        output_file.attrs["input_embedding_dim"] = int(
            model["input_embedding_dim"]
        )
        output_file.attrs["embedding_dim"] = int(model["embedding_dim"])
        output_file.attrs["covariance_mode"] = fitting_config["covariance_mode"]
        output_file.attrs["shared_covariance_weight"] = float(
            fitting_config["shared_covariance_weight"]
        )
        output_file.attrs["variance_floor"] = float(fitting_config["variance_floor"])
        output_file.attrs["enable_covariance_regularization"] = bool(
            fitting_config["enable_covariance_regularization"]
        )
        output_file.attrs["covariance_regularization"] = float(
            fitting_config["covariance_regularization"]
        )
        output_file.attrs["covariance_rank_tolerance"] = float(
            fitting_config["covariance_rank_tolerance"]
        )
        output_file.attrs["enable_calibration"] = False

        model_group = output_file.create_group("model")
        model_group.create_dataset(
            "bin_progress_values", data=model["bin_progress_values"]
        )
        model_group.create_dataset("bin_means", data=model["bin_means"])
        model_group.create_dataset(
            "bin_independent_covariances",
            data=model["bin_independent_covariances"],
        )
        model_group.create_dataset(
            "shared_covariance", data=model["shared_covariance"]
        )
        model_group.create_dataset(
            "bin_final_covariances", data=model["bin_final_covariances"]
        )
        model_group.create_dataset(
            "bin_final_covariance_ranks",
            data=model["bin_final_covariance_ranks"],
        )
        if model["enable_pca"]:
            model_group.create_dataset(
                "bin_pre_pca_final_covariance_ranks",
                data=model["bin_pre_pca_final_covariance_ranks"],
            )
        model_group.create_dataset(
            "bin_log_determinants", data=model["bin_log_determinants"]
        )
        model_group.create_dataset("bin_counts", data=model["bin_counts"])
        if model["enable_pca"]:
            model_group.create_dataset("pca_mean", data=model["pca_mean"])
            model_group.create_dataset(
                "pca_components", data=model["pca_components"]
            )


def _parse_visualization_config(config: dict) -> dict:
    """Validate the optional t-SNE visualization configuration."""
    enabled = config.get("enable_visualization", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError(
            "[gaussian_progress_fitting] enable_visualization must be boolean."
        )
    if not bool(enabled):
        return {"enabled": False}

    tsne_viz = config.get("tsne_viz", {})
    if not isinstance(tsne_viz, dict):
        raise ValueError("[gaussian_progress_fitting] tsne_viz must be a mapping.")

    gaussian_samples_per_bin = tsne_viz.get("gaussian_samples_per_bin", 200)
    if isinstance(gaussian_samples_per_bin, bool):
        raise ValueError(
            "[gaussian_progress_fitting] gaussian_samples_per_bin must be an integer >= 20."
        )
    try:
        gaussian_samples_float = float(gaussian_samples_per_bin)
        gaussian_samples_per_bin = int(gaussian_samples_per_bin)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "[gaussian_progress_fitting] gaussian_samples_per_bin must be an integer >= 20."
        ) from exc
    if (
        not np.isfinite(gaussian_samples_float)
        or gaussian_samples_float != gaussian_samples_per_bin
        or gaussian_samples_per_bin < 20
    ):
        raise ValueError(
            "[gaussian_progress_fitting] gaussian_samples_per_bin must be an "
            f"integer >= 20; got {gaussian_samples_per_bin!r}."
        )

    random_seed = int(tsne_viz.get("random_seed", 42))
    preprocessing = tsne_viz.get("preprocessing", {})
    tsne_config = tsne_viz.get("tsne", {})
    contour_config = tsne_viz.get("contour", {})
    plot_config = tsne_viz.get("plot", {})
    for name, section in (
        ("preprocessing", preprocessing),
        ("tsne", tsne_config),
        ("contour", contour_config),
        ("plot", plot_config),
    ):
        if not isinstance(section, dict):
            raise ValueError(
                f"[gaussian_progress_fitting] tsne_viz.{name} must be a mapping."
            )

    use_open_tsne = tsne_config.get("use_open_tsne", False)
    if not isinstance(use_open_tsne, (bool, np.bool_)):
        raise ValueError(
            "[gaussian_progress_fitting] tsne_viz.tsne.use_open_tsne must be boolean."
        )

    standardize = preprocessing.get("standardize", True)
    use_pca = preprocessing.get("use_pca_before_tsne", True)
    if not isinstance(standardize, (bool, np.bool_)) or not isinstance(
        use_pca, (bool, np.bool_)
    ):
        raise ValueError(
            "[gaussian_progress_fitting] preprocessing standardize/use_pca flags "
            "must be boolean."
        )
    pca_dim = int(preprocessing.get("pca_dim", 50))
    if pca_dim < 1:
        raise ValueError("[gaussian_progress_fitting] pca_dim must be >= 1.")

    perplexity_mode = str(
        tsne_config.get("perplexity_mode", "config_clamped")
    ).strip().lower()
    if perplexity_mode not in {
        "config",
        "config_clamped",
        "clamped_config",
        "clamp",
        "max_safe",
    }:
        raise ValueError(
            "[gaussian_progress_fitting] unsupported perplexity_mode: "
            f"{perplexity_mode!r}."
        )
    perplexity = float(tsne_config.get("perplexity", 500))
    if not np.isfinite(perplexity) or perplexity <= 0.0:
        raise ValueError("[gaussian_progress_fitting] perplexity must be finite and > 0.")
    learning_rate = tsne_config.get("learning_rate", "auto")
    if isinstance(learning_rate, str):
        if learning_rate != "auto":
            raise ValueError(
                "[gaussian_progress_fitting] learning_rate string must be 'auto'."
            )
    else:
        learning_rate = float(learning_rate)
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "[gaussian_progress_fitting] learning_rate must be 'auto' or > 0."
            )
    init = str(tsne_config.get("init", "pca"))
    if init not in {"pca", "random"}:
        raise ValueError(
            "[gaussian_progress_fitting] tsne init must be 'pca' or 'random'."
        )
    max_iter = int(tsne_config.get("max_iter", 2000))
    if max_iter < 250:
        raise ValueError("[gaussian_progress_fitting] tsne max_iter must be >= 250.")

    mass_levels = np.asarray(
        contour_config.get("mass_levels", [0.50, 0.80, 0.95]), dtype=np.float64
    )
    if (
        mass_levels.ndim != 1
        or mass_levels.size < 1
        or not np.isfinite(mass_levels).all()
        or np.any(mass_levels <= 0.0)
        or np.any(mass_levels >= 1.0)
        or np.any(np.diff(mass_levels) <= 0.0)
    ):
        raise ValueError(
            "[gaussian_progress_fitting] contour mass_levels must be strictly "
            "increasing values in (0, 1)."
        )

    figsize_raw = plot_config.get("figsize", [10, 8])
    if not isinstance(figsize_raw, (list, tuple)) or len(figsize_raw) != 2:
        raise ValueError(
            "[gaussian_progress_fitting] plot figsize must contain [width, height]."
        )
    figsize = tuple(float(value) for value in figsize_raw)
    point_size = float(plot_config.get("point_size", 20))
    alpha = float(plot_config.get("alpha", 0.75))
    dpi = int(plot_config.get("dpi", 300))
    enable_real_only_debug = plot_config.get("enable_real_only_debug", False)
    if not isinstance(enable_real_only_debug, (bool, np.bool_)):
        raise ValueError(
            "[gaussian_progress_fitting] "
            "tsne_viz.plot.enable_real_only_debug must be boolean."
        )
    enable_real_video_idx_plot = plot_config.get(
        "enable_real_video_idx_plot", False
    )
    if not isinstance(enable_real_video_idx_plot, (bool, np.bool_)):
        raise ValueError(
            "[gaussian_progress_fitting] "
            "tsne_viz.plot.enable_real_video_idx_plot must be boolean."
        )
    if (
        min(figsize) <= 0.0
        or point_size <= 0.0
        or not 0.0 < alpha <= 1.0
        or dpi <= 0
    ):
        raise ValueError(
            "[gaussian_progress_fitting] plot dimensions, point_size, alpha, "
            "and dpi must be positive (alpha <= 1)."
        )

    return {
        "enabled": True,
        "gaussian_samples_per_bin": gaussian_samples_per_bin,
        "random_seed": random_seed,
        "use_open_tsne": bool(use_open_tsne),
        "standardize": bool(standardize),
        "use_pca_before_tsne": bool(use_pca),
        "pca_dim": pca_dim,
        "perplexity_mode": perplexity_mode,
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "init": init,
        "max_iter": max_iter,
        "mass_levels": mass_levels,
        "figsize": figsize,
        "point_size": point_size,
        "alpha": alpha,
        "dpi": dpi,
        "enable_real_only_debug": bool(enable_real_only_debug),
        "enable_real_video_idx_plot": bool(enable_real_video_idx_plot),
        "colormap": str(plot_config.get("cmap_progress_bins", "turbo")),
    }


def _read_gaussian_model_for_visualization(
    gaussian_model_h5_path: Path,
    expert_h5_path: str,
) -> dict:
    """Read and validate the saved model statistics needed for visualization."""
    required_datasets = {
        "bin_progress_values",
        "bin_means",
        "bin_final_covariances",
        "bin_counts",
    }
    with h5py.File(gaussian_model_h5_path, "r") as model_file:
        if str(model_file.attrs.get("task_name", "")) != "gaussian_progress_fitting":
            raise ValueError(
                "[gaussian_progress_fitting] visualization model H5 has an "
                "unexpected task_name attr."
            )
        stored_expert_path = model_file.attrs.get("expert_h5_path", "")
        if not stored_expert_path or (
            Path(str(stored_expert_path)).expanduser().resolve()
            != Path(expert_h5_path).expanduser().resolve()
        ):
            raise ValueError(
                "[gaussian_progress_fitting] visualization expert H5 does not "
                "match the model expert_h5_path attr."
            )
        if "model" not in model_file:
            raise ValueError(
                "[gaussian_progress_fitting] visualization model H5 is missing /model."
            )
        model_group = model_file["model"]
        missing = sorted(required_datasets - set(model_group.keys()))
        if missing:
            raise ValueError(
                "[gaussian_progress_fitting] visualization model H5 is missing "
                f"datasets: {missing}."
            )

        num_bins = int(model_file.attrs["num_bins"])
        embedding_dim = int(model_file.attrs["embedding_dim"])
        enable_pca = bool(model_file.attrs.get("enable_pca", False))
        if enable_pca and "input_embedding_dim" not in model_file.attrs:
            raise ValueError(
                "[gaussian_progress_fitting] PCA visualization model is missing "
                "the input_embedding_dim attr."
            )
        input_embedding_dim = int(
            model_file.attrs.get("input_embedding_dim", embedding_dim)
        )
        pca_dataset_names = {"pca_mean", "pca_components"}
        present_pca_datasets = pca_dataset_names.intersection(model_group.keys())
        if enable_pca and present_pca_datasets != pca_dataset_names:
            missing_pca = sorted(pca_dataset_names - present_pca_datasets)
            raise ValueError(
                "[gaussian_progress_fitting] PCA visualization model is missing "
                f"datasets: {missing_pca}."
            )
        if not enable_pca and present_pca_datasets:
            raise ValueError(
                "[gaussian_progress_fitting] non-PCA visualization model must "
                "not contain pca_mean or pca_components."
            )
        model = {
            "num_bins": num_bins,
            "enable_pca": enable_pca,
            "input_embedding_dim": input_embedding_dim,
            "embedding_dim": embedding_dim,
            "num_expert_features": int(model_file.attrs["num_expert_features"]),
            "covariance_mode": str(model_file.attrs["covariance_mode"]),
            "bin_progress_values": np.asarray(
                model_group["bin_progress_values"], dtype=np.float64
            ),
            "bin_means": np.asarray(model_group["bin_means"], dtype=np.float64),
            "bin_final_covariances": np.asarray(
                model_group["bin_final_covariances"], dtype=np.float64
            ),
            "bin_counts": np.asarray(model_group["bin_counts"], dtype=np.int64),
            "pca_mean": (
                np.asarray(model_group["pca_mean"], dtype=np.float64)
                if enable_pca
                else None
            ),
            "pca_components": (
                np.asarray(model_group["pca_components"], dtype=np.float64)
                if enable_pca
                else None
            ),
        }

    if num_bins < 2 or embedding_dim < 1 or input_embedding_dim < 1:
        raise ValueError(
            "[gaussian_progress_fitting] visualization model dimensions must be "
            "positive and num_bins must be >= 2."
        )
    if enable_pca:
        if input_embedding_dim <= embedding_dim:
            raise ValueError(
                "[gaussian_progress_fitting] PCA visualization model requires "
                "input_embedding_dim > embedding_dim."
            )
        expected_pca_shapes = {
            "pca_mean": (input_embedding_dim,),
            "pca_components": (embedding_dim, input_embedding_dim),
        }
        for name, expected_shape in expected_pca_shapes.items():
            if model[name].shape != expected_shape:
                raise ValueError(
                    f"[gaussian_progress_fitting] model {name} must have shape "
                    f"{expected_shape}; got {model[name].shape}."
                )
            if not np.isfinite(model[name]).all():
                raise ValueError(
                    f"[gaussian_progress_fitting] model {name} contains NaN or Inf."
                )
    elif input_embedding_dim != embedding_dim:
        raise ValueError(
            "[gaussian_progress_fitting] non-PCA visualization model requires "
            "input_embedding_dim == embedding_dim."
        )

    expected_shapes = {
        "bin_progress_values": (num_bins,),
        "bin_means": (num_bins, embedding_dim),
        "bin_final_covariances": (num_bins, embedding_dim, embedding_dim),
        "bin_counts": (num_bins,),
    }
    for name, expected_shape in expected_shapes.items():
        if model[name].shape != expected_shape:
            raise ValueError(
                f"[gaussian_progress_fitting] model {name} must have shape "
                f"{expected_shape}; got {model[name].shape}."
            )
        if not np.isfinite(model[name]).all():
            raise ValueError(
                f"[gaussian_progress_fitting] model {name} contains NaN or Inf."
            )

    if np.any(model["bin_counts"] < 2):
        raise ValueError(
            "[gaussian_progress_fitting] model bin_counts contains a value < 2."
        )
    for bin_index, covariance in enumerate(model["bin_final_covariances"]):
        _validate_covariance_matrix(
            covariance,
            embedding_dim,
            f"visualization bin_final_covariances[{bin_index}]",
            require_positive_definite=True,
        )
    return model


def _sample_saved_gaussians(model: dict, samples_per_bin: int, random_seed: int) -> np.ndarray:
    """Draw deterministic high-dimensional samples from every saved Gaussian."""
    rng = np.random.default_rng(random_seed)
    samples = np.empty(
        (model["num_bins"], samples_per_bin, model["embedding_dim"]),
        dtype=np.float64,
    )
    for bin_index in range(model["num_bins"]):
        try:
            cholesky_factor = np.linalg.cholesky(
                model["bin_final_covariances"][bin_index]
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"[gaussian_progress_fitting] Cholesky sampling failed for bin "
                f"{bin_index}."
            ) from exc
        standard_normal = rng.standard_normal(
            (samples_per_bin, model["embedding_dim"])
        )
        samples[bin_index] = (
            standard_normal @ cholesky_factor.T + model["bin_means"][bin_index]
        )
    if not np.isfinite(samples).all():
        raise ValueError(
            "[gaussian_progress_fitting] sampled Gaussian features contain NaN or Inf."
        )
    return samples


def _evaluate_projected_kde(
    projected_samples: np.ndarray,
    mass_levels: np.ndarray,
    grid_size: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Evaluate a 2D KDE and convert probability masses to density thresholds."""
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    centered = projected_samples - projected_samples.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("projected Gaussian samples are rank deficient in 2D")
    try:
        kde = gaussian_kde(projected_samples.T, bw_method="scott")
    except np.linalg.LinAlgError as exc:
        raise ValueError("projected Gaussian KDE covariance is singular") from exc

    x_min, y_min = projected_samples.min(axis=0)
    x_max, y_max = projected_samples.max(axis=0)
    x_pad = 0.15 * max(float(x_max - x_min), 1.0e-12)
    y_pad = 0.15 * max(float(y_max - y_min), 1.0e-12)
    x_values = np.linspace(x_min - x_pad, x_max + x_pad, grid_size)
    y_values = np.linspace(y_min - y_pad, y_max + y_pad, grid_size)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(
        grid_x.shape
    )
    if not np.isfinite(density).all() or float(density.sum()) <= 0.0:
        raise ValueError("projected Gaussian KDE produced invalid density values")

    sorted_density = np.sort(density.ravel())[::-1]
    cumulative_mass = np.cumsum(sorted_density)
    cumulative_mass /= cumulative_mass[-1]
    thresholds = []
    for mass_level in mass_levels:
        threshold_index = min(
            int(np.searchsorted(cumulative_mass, mass_level, side="left")),
            sorted_density.size - 1,
        )
        thresholds.append(float(sorted_density[threshold_index]))
    return grid_x, grid_y, density, thresholds


def _resolve_tsne_perplexity(
    num_points: int,
    visualization_config: dict,
) -> tuple[float, float]:
    """Resolve the configured perplexity against one concrete fit set."""
    perplexity_cap = max(5.0, (num_points - 1) / 3.0)
    mode = visualization_config["perplexity_mode"]
    if mode == "config":
        perplexity_used = visualization_config["perplexity"]
    elif mode in {"config_clamped", "clamped_config", "clamp"}:
        perplexity_used = min(
            visualization_config["perplexity"], perplexity_cap
        )
    else:
        perplexity_used = perplexity_cap
    if perplexity_used >= num_points:
        raise ValueError(
            "[gaussian_progress_fitting] effective t-SNE perplexity must be "
            f"< n_fit; got perplexity={perplexity_used} and n_fit={num_points}."
        )
    return float(perplexity_used), float(perplexity_cap)


def _fit_tsne_projection(
    features: np.ndarray,
    visualization_config: dict,
    fit_label: str,
) -> tuple[np.ndarray, float]:
    """Apply the shared preprocessing and t-SNE policy to one feature set."""
    from sklearn.decomposition import PCA  # noqa: PLC0415
    from sklearn.manifold import TSNE  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError(
            "[gaussian_progress_fitting] t-SNE features must have shape [N, D] "
            f"with N>=2 and D>=1; got {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError(
            "[gaussian_progress_fitting] t-SNE features contain NaN or Inf."
        )

    num_points, embedding_dim = features.shape
    processed = (
        StandardScaler().fit_transform(features)
        if visualization_config["standardize"]
        else features.copy()
    )
    if visualization_config["use_pca_before_tsne"]:
        pca_dim = min(
            visualization_config["pca_dim"],
            processed.shape[1],
            processed.shape[0] - 1,
        )
        pca = PCA(n_components=pca_dim)
        processed = pca.fit_transform(processed)
        print(
            f"[gaussian_progress_fitting] tsne: {fit_label} PCA "
            f"{embedding_dim}->{pca_dim} "
            f"var={pca.explained_variance_ratio_.sum():.3f}"
        )

    perplexity_used, perplexity_cap = _resolve_tsne_perplexity(
        num_points, visualization_config
    )
    mode = visualization_config["perplexity_mode"]

    print(
        f"[gaussian_progress_fitting] tsne: running {fit_label} "
        f"n={num_points} mode={mode} "
        f"config={visualization_config['perplexity']:.1f} "
        f"cap={perplexity_cap:.1f} used={perplexity_used:.1f}"
    )
    tsne_init = visualization_config["init"]
    if tsne_init == "pca" and processed.shape[1] < 2:
        tsne_init = "random"
        print(
            "[gaussian_progress_fitting] tsne: falling back from PCA to random "
            "initialization because the processed feature dimension is < 2."
        )
    fit_start = time.perf_counter()
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity_used,
        learning_rate=visualization_config["learning_rate"],
        init=tsne_init,
        random_state=visualization_config["random_seed"],
        max_iter=visualization_config["max_iter"],
        verbose=2,
        method="barnes_hut",
        angle=0.5,
        n_jobs=None,
    ).fit_transform(processed)
    if coordinates.shape != (num_points, 2) or not np.isfinite(coordinates).all():
        raise ValueError(
            "[gaussian_progress_fitting] t-SNE returned invalid coordinates: "
            f"{coordinates.shape}."
        )
    print(
        f"[gaussian_progress_fitting] tsne: {fit_label} finished in "
        f"{time.perf_counter() - fit_start:.1f}s"
    )
    return coordinates, float(perplexity_used)


def _fit_open_tsne_reference_projection(
    real_features: np.ndarray,
    out_of_sample_features: np.ndarray,
    visualization_config: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit openTSNE on real features and transform non-real points afterward."""
    from sklearn.decomposition import PCA  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    try:
        from openTSNE import TSNE as OpenTSNE  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "[gaussian_progress_fitting] tsne_viz.tsne.use_open_tsne=true "
            "requires the 'openTSNE' package. Install the project requirements "
            "or set use_open_tsne=false to use sklearn."
        ) from exc

    real_features = np.asarray(real_features, dtype=np.float64)
    out_of_sample_features = np.asarray(out_of_sample_features, dtype=np.float64)
    if (
        real_features.ndim != 2
        or real_features.shape[0] < 2
        or real_features.shape[1] < 1
    ):
        raise ValueError(
            "[gaussian_progress_fitting] openTSNE real features must have shape "
            f"[N, D] with N>=2 and D>=1; got {real_features.shape}."
        )
    if (
        out_of_sample_features.ndim != 2
        or out_of_sample_features.shape[0] < 1
        or out_of_sample_features.shape[1] != real_features.shape[1]
    ):
        raise ValueError(
            "[gaussian_progress_fitting] openTSNE out-of-sample features must "
            "have shape [M, D] with M>=1 and the real embedding dimension; "
            f"got {out_of_sample_features.shape}."
        )
    if not np.isfinite(real_features).all() or not np.isfinite(
        out_of_sample_features
    ).all():
        raise ValueError(
            "[gaussian_progress_fitting] openTSNE features contain NaN or Inf."
        )

    num_real, embedding_dim = real_features.shape
    if visualization_config["standardize"]:
        scaler = StandardScaler()
        processed_real = scaler.fit_transform(real_features)
        processed_out_of_sample = scaler.transform(out_of_sample_features)
    else:
        processed_real = real_features.copy()
        processed_out_of_sample = out_of_sample_features.copy()

    if visualization_config["use_pca_before_tsne"]:
        pca_dim = min(
            visualization_config["pca_dim"],
            processed_real.shape[1],
            processed_real.shape[0] - 1,
        )
        pca = PCA(n_components=pca_dim)
        processed_real = pca.fit_transform(processed_real)
        processed_out_of_sample = pca.transform(processed_out_of_sample)
        print(
            f"[gaussian_progress_fitting] tsne: openTSNE real-reference PCA "
            f"{embedding_dim}->{pca_dim} "
            f"var={pca.explained_variance_ratio_.sum():.3f}"
        )

    perplexity_used, perplexity_cap = _resolve_tsne_perplexity(
        num_real, visualization_config
    )
    mode = visualization_config["perplexity_mode"]
    print(
        "[gaussian_progress_fitting] tsne: running openTSNE real-reference fit "
        f"n={num_real} mode={mode} "
        f"config={visualization_config['perplexity']:.1f} "
        f"cap={perplexity_cap:.1f} used={perplexity_used:.1f}"
    )
    tsne_init = visualization_config["init"]
    if tsne_init == "pca" and processed_real.shape[1] < 2:
        tsne_init = "random"
        print(
            "[gaussian_progress_fitting] tsne: falling back from PCA to random "
            "initialization because the processed feature dimension is < 2."
        )
    fit_start = time.perf_counter()
    reference_embedding = OpenTSNE(
        n_components=2,
        perplexity=perplexity_used,
        learning_rate=visualization_config["learning_rate"],
        initialization=tsne_init,
        random_state=visualization_config["random_seed"],
        n_iter=visualization_config["max_iter"],
        verbose=True,
        negative_gradient_method="bh",
        theta=0.5,
        n_jobs=1,
    ).fit(processed_real)
    real_coordinates = np.asarray(reference_embedding, dtype=np.float64)
    print(
        "[gaussian_progress_fitting] tsne: openTSNE real-reference fit "
        f"finished in {time.perf_counter() - fit_start:.1f}s"
    )

    transform_start = time.perf_counter()
    out_of_sample_coordinates = np.asarray(
        reference_embedding.transform(processed_out_of_sample),
        dtype=np.float64,
    )
    print(
        "[gaussian_progress_fitting] tsne: openTSNE out-of-sample transform "
        f"n={out_of_sample_features.shape[0]} finished in "
        f"{time.perf_counter() - transform_start:.1f}s"
    )
    if real_coordinates.shape != (num_real, 2) or not np.isfinite(
        real_coordinates
    ).all():
        raise ValueError(
            "[gaussian_progress_fitting] openTSNE returned invalid real "
            f"coordinates: {real_coordinates.shape}."
        )
    expected_out_shape = (out_of_sample_features.shape[0], 2)
    if out_of_sample_coordinates.shape != expected_out_shape or not np.isfinite(
        out_of_sample_coordinates
    ).all():
        raise ValueError(
            "[gaussian_progress_fitting] openTSNE returned invalid out-of-sample "
            f"coordinates: {out_of_sample_coordinates.shape}."
        )
    return real_coordinates, out_of_sample_coordinates, perplexity_used


def _save_real_only_tsne_visualization(
    all_embeddings: np.ndarray,
    real_coordinates: np.ndarray,
    perplexity_used: float,
    backend_label: str,
    bin_indices: np.ndarray,
    bin_counts: np.ndarray,
    bin_progress_values: np.ndarray,
    visualization_config: dict,
    output_dir: Path,
    artifact_suffix: str,
) -> dict:
    """Save a t-SNE containing only real expert fitting features."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    if real_coordinates.shape != (all_embeddings.shape[0], 2) or not np.isfinite(
        real_coordinates
    ).all():
        raise ValueError(
            "[gaussian_progress_fitting] real-only visualization received "
            f"invalid coordinates: {real_coordinates.shape}."
        )
    num_bins = int(bin_progress_values.shape[0])
    figure, axis = plt.subplots(figsize=visualization_config["figsize"])
    colormap = plt.get_cmap(visualization_config["colormap"])
    colors = colormap(bin_progress_values)

    bin_handles = []
    for bin_index in range(num_bins):
        mask = bin_indices == bin_index
        axis.scatter(
            real_coordinates[mask, 0],
            real_coordinates[mask, 1],
            s=visualization_config["point_size"],
            alpha=visualization_config["alpha"],
            color=colors[bin_index],
            marker="o",
            rasterized=True,
        )
        bin_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[bin_index],
                marker="o",
                linewidth=1.3,
                markersize=4,
                label=(
                    f"bin {bin_index:02d}  "
                    f"p={bin_progress_values[bin_index]:.3f} "
                    f"n={int(bin_counts[bin_index])}"
                ),
            )
        )

    axis.legend(
        handles=bin_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=7,
        ncol=1,
        title="Progress bins",
    )
    axis.set_xlabel("t-SNE dim 1")
    axis.set_ylabel("t-SNE dim 2")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(
        f"Expert Fitting Features — Real-only t-SNE ({backend_label})\n"
        f"bins={num_bins}  real={all_embeddings.shape[0]}"
    )
    figure.subplots_adjust(right=0.73, bottom=0.08)

    output_path = output_dir / (
        f"gaussian_progress_tsne_real_only-{artifact_suffix}.png"
    )
    figure.savefig(
        output_path,
        dpi=visualization_config["dpi"],
        bbox_inches="tight",
    )
    plt.close(figure)
    print(
        f"[gaussian_progress_fitting] output_real_visualization_path: {output_path}"
    )
    return {
        "output_real_visualization_path": str(output_path),
        "num_tsne_points": int(all_embeddings.shape[0]),
        "perplexity_used": perplexity_used,
    }


def _save_real_video_idx_tsne_visualization(
    real_coordinates: np.ndarray,
    video_indices: np.ndarray,
    video_ids: list[str],
    backend_label: str,
    visualization_config: dict,
    output_dir: Path,
    artifact_suffix: str,
) -> str:
    """Save the real-only t-SNE colored by sorted expert-video index."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import BoundaryNorm  # noqa: PLC0415

    real_coordinates = np.asarray(real_coordinates, dtype=np.float64)
    video_indices = np.asarray(video_indices, dtype=np.int64)
    num_videos = len(video_ids)
    if num_videos < 1:
        raise ValueError(
            "[gaussian_progress_fitting] video-index visualization requires "
            "at least one video."
        )
    if video_indices.shape != (real_coordinates.shape[0],):
        raise ValueError(
            "[gaussian_progress_fitting] video indices must align with real "
            f"coordinates; got {video_indices.shape} and {real_coordinates.shape}."
        )
    if (
        real_coordinates.ndim != 2
        or real_coordinates.shape[1] != 2
        or not np.isfinite(real_coordinates).all()
        or np.any(video_indices < 0)
        or np.any(video_indices >= num_videos)
    ):
        raise ValueError(
            "[gaussian_progress_fitting] invalid real coordinates or video "
            "indices for video-index visualization."
        )

    figure, axis = plt.subplots(figsize=visualization_config["figsize"])
    colormap = plt.get_cmap("turbo", num_videos)
    boundaries = np.arange(num_videos + 1, dtype=np.float64) - 0.5
    norm = BoundaryNorm(boundaries, colormap.N)
    scatter = axis.scatter(
        real_coordinates[:, 0],
        real_coordinates[:, 1],
        c=video_indices,
        cmap=colormap,
        norm=norm,
        s=visualization_config["point_size"],
        alpha=visualization_config["alpha"],
        marker="o",
        rasterized=True,
    )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
    tick_stride = max(1, int(np.ceil(num_videos / 18.0)))
    ticks = np.arange(0, num_videos, tick_stride, dtype=np.int64)
    if ticks[-1] != num_videos - 1:
        ticks = np.append(ticks, num_videos - 1)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([str(int(index)) for index in ticks])
    colorbar.set_label("Video index (sorted H5 order)")

    axis.set_xlabel("t-SNE dim 1")
    axis.set_ylabel("t-SNE dim 2")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(
        f"Expert Fitting Features — Real-only t-SNE by Video Index "
        f"({backend_label})\n"
        f"videos={num_videos}  real={real_coordinates.shape[0]}"
    )

    output_path = output_dir / (
        f"gaussian_progress_tsne_real_only_video_idx-{artifact_suffix}.png"
    )
    figure.savefig(
        output_path,
        dpi=visualization_config["dpi"],
        bbox_inches="tight",
    )
    plt.close(figure)
    print(
        "[gaussian_progress_fitting] "
        f"output_real_video_idx_visualization_path: {output_path}"
    )
    return str(output_path)


def _save_gaussian_progress_visualization(
    expert_h5_path: str,
    gaussian_model_h5_path: Path,
    visualization_config: dict,
    artifact_suffix: str,
) -> dict:
    """Render all fitting features and saved Gaussian contours in one t-SNE."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    video_ids, embeddings, progresses, _, _ = _read_expert_trajectories(
        expert_h5_path
    )
    model = _read_gaussian_model_for_visualization(
        gaussian_model_h5_path, expert_h5_path
    )
    if model["enable_pca"]:
        embeddings = _apply_saved_model_pca(
            embeddings=embeddings,
            pca_mean=model["pca_mean"],
            pca_components=model["pca_components"],
        )
    assigned = _assign_progress_bins(embeddings, progresses, model["num_bins"])
    all_embeddings = assigned["all_embeddings"]
    bin_indices = assigned["bin_indices"]
    bin_counts = assigned["bin_counts"]
    if assigned["embedding_dim"] != model["embedding_dim"]:
        raise ValueError(
            "[gaussian_progress_fitting] expert/model embedding dimensions do not match."
        )
    if assigned["all_embeddings"].shape[0] != model["num_expert_features"]:
        raise ValueError(
            "[gaussian_progress_fitting] expert/model feature counts do not match."
        )
    if not np.array_equal(bin_counts, model["bin_counts"]):
        raise ValueError(
            "[gaussian_progress_fitting] recomputed bin_counts do not match the model."
        )
    empirical_means = np.stack(
        [
            all_embeddings[bin_indices == bin_index].mean(axis=0)
            for bin_index in range(model["num_bins"])
        ]
    )
    if not np.allclose(
        empirical_means,
        model["bin_means"],
        rtol=1.0e-7,
        atol=1.0e-10,
    ):
        raise ValueError(
            "[gaussian_progress_fitting] expert empirical means do not match "
            "the saved model means."
        )

    gaussian_samples = _sample_saved_gaussians(
        model,
        visualization_config["gaussian_samples_per_bin"],
        visualization_config["random_seed"],
    )
    flat_gaussian_samples = gaussian_samples.reshape(-1, model["embedding_dim"])
    num_real = all_embeddings.shape[0]
    num_synthetic = flat_gaussian_samples.shape[0]
    use_open_tsne = visualization_config["use_open_tsne"]
    enable_real_only_debug = visualization_config["enable_real_only_debug"]

    if use_open_tsne:
        out_of_sample_features = np.concatenate(
            [model["bin_means"], flat_gaussian_samples], axis=0
        )
        print(
            "[gaussian_progress_fitting] tsne: openTSNE reference input "
            f"real={num_real} out_of_sample={out_of_sample_features.shape[0]} "
            f"means={model['num_bins']} synthetic={num_synthetic} "
            f"dim={model['embedding_dim']}"
        )
        (
            real_coordinates,
            out_of_sample_coordinates,
            perplexity_used,
        ) = _fit_open_tsne_reference_projection(
            real_features=all_embeddings,
            out_of_sample_features=out_of_sample_features,
            visualization_config=visualization_config,
        )
        real_only_coordinates = real_coordinates if enable_real_only_debug else None
        mean_coordinates = out_of_sample_coordinates[: model["num_bins"]]
        synthetic_coordinates = out_of_sample_coordinates[
            model["num_bins"] :
        ].reshape(
            model["num_bins"],
            visualization_config["gaussian_samples_per_bin"],
            2,
        )
        num_tsne_points = num_real
        real_perplexity_used = perplexity_used if enable_real_only_debug else None
        backend_label = "openTSNE reference"
    else:
        if enable_real_only_debug:
            real_only_coordinates, real_perplexity_used = _fit_tsne_projection(
                all_embeddings,
                visualization_config,
                fit_label="real-only fit",
            )
        else:
            real_only_coordinates = None
            real_perplexity_used = None
        joint_embeddings = np.concatenate(
            [all_embeddings, flat_gaussian_samples, model["bin_means"]], axis=0
        )
        num_tsne_points = joint_embeddings.shape[0]
        print(
            "[gaussian_progress_fitting] tsne: sklearn joint input "
            f"real={num_real} synthetic={num_synthetic} "
            f"means={model['num_bins']} total={num_tsne_points} "
            f"dim={model['embedding_dim']}"
        )
        coordinates, perplexity_used = _fit_tsne_projection(
            joint_embeddings,
            visualization_config,
            fit_label="joint Gaussian fit",
        )
        real_coordinates = coordinates[:num_real]
        synthetic_coordinates = coordinates[
            num_real : num_real + num_synthetic
        ].reshape(
            model["num_bins"],
            visualization_config["gaussian_samples_per_bin"],
            2,
        )
        mean_coordinates = coordinates[num_real + num_synthetic :]
        backend_label = "sklearn"

    real_visualization_result = {
        "output_real_visualization_path": None,
        "num_tsne_points": 0,
        "perplexity_used": None,
    }
    output_real_video_idx_visualization_path = None
    if enable_real_only_debug:
        real_visualization_result = _save_real_only_tsne_visualization(
            all_embeddings=all_embeddings,
            real_coordinates=real_only_coordinates,
            perplexity_used=real_perplexity_used,
            backend_label=backend_label,
            bin_indices=bin_indices,
            bin_counts=bin_counts,
            bin_progress_values=model["bin_progress_values"],
            visualization_config=visualization_config,
            output_dir=gaussian_model_h5_path.parent,
            artifact_suffix=artifact_suffix,
        )
    if enable_real_only_debug and visualization_config["enable_real_video_idx_plot"]:
        video_indices = np.concatenate(
            [
                np.full(video_embeddings.shape[0], video_index, dtype=np.int64)
                for video_index, video_embeddings in enumerate(embeddings)
            ]
        )
        output_real_video_idx_visualization_path = (
            _save_real_video_idx_tsne_visualization(
                real_coordinates=real_only_coordinates,
                video_indices=video_indices,
                video_ids=video_ids,
                backend_label=backend_label,
                visualization_config=visualization_config,
                output_dir=gaussian_model_h5_path.parent,
                artifact_suffix=artifact_suffix,
            )
        )

    figure, axis = plt.subplots(figsize=visualization_config["figsize"])
    colormap = plt.get_cmap(visualization_config["colormap"])
    colors = colormap(model["bin_progress_values"])
    line_styles = ["solid", "dashed", "dotted"]
    if len(visualization_config["mass_levels"]) != len(line_styles):
        line_styles = ["solid"] * len(visualization_config["mass_levels"])

    for bin_index in range(model["num_bins"]):
        try:
            grid_x, grid_y, density, thresholds = _evaluate_projected_kde(
                synthetic_coordinates[bin_index],
                visualization_config["mass_levels"],
            )
        except ValueError as exc:
            raise ValueError(
                f"[gaussian_progress_fitting] KDE contour failed for bin "
                f"{bin_index}: {exc}"
            ) from exc
        for threshold, line_style in zip(thresholds, line_styles):
            axis.contour(
                grid_x,
                grid_y,
                density,
                levels=[threshold],
                colors=[colors[bin_index]],
                linewidths=1.3,
                linestyles=[line_style],
                zorder=1,
            )

    bin_handles = []
    for bin_index in range(model["num_bins"]):
        mask = bin_indices == bin_index
        axis.scatter(
            real_coordinates[mask, 0],
            real_coordinates[mask, 1],
            s=visualization_config["point_size"],
            alpha=visualization_config["alpha"],
            color=colors[bin_index],
            marker="o",
            rasterized=True,
            zorder=2,
        )
        axis.scatter(
            mean_coordinates[bin_index, 0],
            mean_coordinates[bin_index, 1],
            s=70,
            color=colors[bin_index],
            edgecolor="black",
            linewidth=0.8,
            marker="X",
            zorder=3,
        )
        bin_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[bin_index],
                marker="o",
                linewidth=1.3,
                markersize=4,
                label=(
                    f"bin {bin_index:02d}  p={model['bin_progress_values'][bin_index]:.3f} "
                    f"n={int(bin_counts[bin_index])}"
                ),
            )
        )

    bin_legend = axis.legend(
        handles=bin_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=7,
        ncol=1,
        title="Progress bins",
    )
    axis.add_artist(bin_legend)
    semantic_handles = [
        Line2D([0], [0], marker="o", color="gray", linewidth=0, label="Fitting feature"),
        Line2D([0], [0], marker="X", color="gray", markeredgecolor="black", linewidth=0, label="Saved Gaussian mean"),
    ]
    for mass_level, line_style in zip(
        visualization_config["mass_levels"], line_styles
    ):
        semantic_handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle=line_style,
                label=f"Gaussian KDE {100.0 * mass_level:g}% mass",
            )
        )
    axis.legend(
        handles=semantic_handles,
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0,
        fontsize=7,
        title="Marks",
    )

    axis.set_xlabel("t-SNE dim 1")
    axis.set_ylabel("t-SNE dim 2")
    axis.set_aspect("equal", adjustable="datalim")
    if use_open_tsne:
        title_prefix = "Gaussian Progress Fitting — openTSNE Real Reference"
        footer = (
            "Qualitative only: means and Gaussian draws are out-of-sample "
            "transforms into a real-only openTSNE reference; KDE contours are "
            "not exact latent-space covariance geometry."
        )
    else:
        title_prefix = "Gaussian Progress Fitting — Shared sklearn t-SNE"
        footer = (
            "Qualitative only: contours are KDEs of saved high-dimensional "
            "Gaussian draws after joint t-SNE, not exact latent-space covariance "
            "geometry."
        )
    axis.set_title(
        f"{title_prefix} — Qualitative Validation\n"
        f"mode={model['covariance_mode']}  bins={model['num_bins']}  "
        f"real={num_real}  Gaussian draws={num_synthetic}"
    )
    figure.text(0.5, 0.015, footer, ha="center", fontsize=8)
    figure.subplots_adjust(right=0.73, bottom=0.10)

    output_path = gaussian_model_h5_path.parent / (
        f"gaussian_progress_tsne_contours-{artifact_suffix}.png"
    )
    figure.savefig(
        output_path,
        dpi=visualization_config["dpi"],
        bbox_inches="tight",
    )
    plt.close(figure)
    print(
        f"[gaussian_progress_fitting] output_visualization_path: {output_path}"
    )
    return {
        "output_visualization_path": str(output_path),
        "output_real_visualization_path": real_visualization_result[
            "output_real_visualization_path"
        ],
        "output_real_video_idx_visualization_path": (
            output_real_video_idx_visualization_path
        ),
        "num_tsne_points": int(num_tsne_points),
        "perplexity_used": float(perplexity_used),
        "num_real_points": int(num_real),
        "num_synthetic_points": int(num_synthetic),
        "real_num_tsne_points": real_visualization_result["num_tsne_points"],
        "real_perplexity_used": real_visualization_result["perplexity_used"],
    }


class GaussianProgressFittingTask(BaseTask):
    """Fit a progress-conditioned Gaussian model from expert trajectories."""

    def __init__(self):
        super().__init__(task_name="gaussian_progress_fitting", downstream_task=False)
        self.config: dict = {}

    def configure(self, config: dict) -> None:
        """Store the resolved V2 evaluation config."""
        self.config = dict(config)

    def evaluate(self, embeddings_dataset=None) -> dict:  # noqa: ARG002
        if not self.config:
            raise ValueError(
                "[gaussian_progress_fitting] task must be configured before evaluate()."
            )
        fitting_config = _parse_fitting_config(self.config)
        visualization_config = _parse_visualization_config(self.config)
        expert_h5_value = self.config.get("expert_h5_path")
        if not expert_h5_value:
            raise ValueError(
                "[gaussian_progress_fitting] expert_h5_path is required after ConfigV2 resolution."
            )
        expert_h5_path = str(Path(expert_h5_value).expanduser().resolve())

        print()
        print(f"[gaussian_progress_fitting] expert_h5_path : {expert_h5_path}")
        print(
            "[gaussian_progress_fitting] covariance_mode: "
            f"{fitting_config['covariance_mode']}"
        )
        print(f"[gaussian_progress_fitting] num_bins       : {fitting_config['num_bins']}")
        print(f"[gaussian_progress_fitting] enable_pca     : {fitting_config['enable_pca']}")
        if fitting_config["enable_pca"]:
            print(
                f"[gaussian_progress_fitting] pca_dim        : "
                f"{fitting_config['pca_dim']}"
            )

        (
            video_ids,
            embeddings,
            progresses,
            fallbacks,
            embedding_normalization,
        ) = _read_expert_trajectories(expert_h5_path)
        for video_id, reason in fallbacks:
            print(
                f"[gaussian_progress_fitting] progress fallback for video "
                f"'{video_id}': {reason}"
            )

        input_embedding_dim = int(embeddings[0].shape[1])
        if fitting_config["enable_pca"]:
            pre_pca_model = _fit_gaussian_progress_model(
                embeddings=embeddings,
                progresses=progresses,
                fitting_config=fitting_config,
            )
            bin_pre_pca_final_covariance_ranks = pre_pca_model[
                "bin_final_covariance_ranks"
            ].copy()
            del pre_pca_model
            model_embeddings, pca_state = _fit_model_pca(
                embeddings=embeddings,
                pca_dim=fitting_config["pca_dim"],
            )
            print(
                "[gaussian_progress_fitting] model PCA       : "
                f"{input_embedding_dim}->{fitting_config['pca_dim']}"
            )
        else:
            model_embeddings = embeddings
            bin_pre_pca_final_covariance_ranks = None
            pca_state = {
                "input_embedding_dim": input_embedding_dim,
                "pca_mean": None,
                "pca_components": None,
            }

        model = _fit_gaussian_progress_model(
            embeddings=model_embeddings,
            progresses=progresses,
            fitting_config=fitting_config,
        )
        model.update(
            {
                "enable_pca": fitting_config["enable_pca"],
                "input_embedding_dim": pca_state["input_embedding_dim"],
                "pca_mean": pca_state["pca_mean"],
                "pca_components": pca_state["pca_components"],
                "bin_pre_pca_final_covariance_ranks": (
                    bin_pre_pca_final_covariance_ranks
                ),
            }
        )

        print(
            f"[gaussian_progress_fitting] expert videos   : {len(video_ids)}"
        )
        print(
            "[gaussian_progress_fitting] expert features : "
            f"{model['num_expert_features']}"
        )
        print("[gaussian_progress_fitting] per-bin sample counts:")
        for bin_index, count in enumerate(model["bin_counts"]):
            print(f"  bin {bin_index:02d}: {int(count)}")
        print(
            "[gaussian_progress_fitting] covariance rank tolerance: "
            f"{fitting_config['covariance_rank_tolerance']:.12e}"
        )
        if model["enable_pca"]:
            print(
                "[gaussian_progress_fitting] per-bin final covariance ranks "
                "(pre-PCA -> post-PCA):"
            )
            for bin_index, (pre_pca_rank, post_pca_rank) in enumerate(
                zip(
                    model["bin_pre_pca_final_covariance_ranks"],
                    model["bin_final_covariance_ranks"],
                    strict=True,
                )
            ):
                print(
                    f"  bin {bin_index:02d}: "
                    f"{int(pre_pca_rank)} -> {int(post_pca_rank)}"
                )
        else:
            print(
                "[gaussian_progress_fitting] per-bin final covariance ranks:"
            )
            for bin_index, rank in enumerate(
                model["bin_final_covariance_ranks"]
            ):
                print(f"  bin {bin_index:02d}: {int(rank)}")

        output_dir = Path(
            self.config.get("output_dir")
            or (_PROJ_ROOT / "outputs" / "gaussian_progress_fitting")
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Keep every fitting invocation self-contained so its model and any
        # visualization artifacts cannot be mixed with another invocation.
        run_output_dir = output_dir / timestamp
        run_output_dir.mkdir(parents=True, exist_ok=False)
        covariance_tag = _build_covariance_artifact_tag(fitting_config)
        artifact_suffix = f"{covariance_tag}-{timestamp}"
        output_h5_path = run_output_dir / (
            f"gaussian_progress_model-{artifact_suffix}.h5"
        )
        print(f"[gaussian_progress_fitting] output_dir     : {run_output_dir}")
        _write_gaussian_model_h5(
            output_h5_path=output_h5_path,
            model=model,
            fitting_config=fitting_config,
            expert_h5_path=expert_h5_path,
            num_expert_videos=len(video_ids),
            embedding_normalization=embedding_normalization,
        )
        print(f"[gaussian_progress_fitting] output_h5_path : {output_h5_path}")

        visualization_result = {
            "output_visualization_path": None,
            "output_real_visualization_path": None,
            "output_real_video_idx_visualization_path": None,
            "num_tsne_points": 0,
            "perplexity_used": None,
            "num_real_points": 0,
            "num_synthetic_points": 0,
            "real_num_tsne_points": 0,
            "real_perplexity_used": None,
        }
        if visualization_config["enabled"]:
            visualization_result = _save_gaussian_progress_visualization(
                expert_h5_path=expert_h5_path,
                gaussian_model_h5_path=output_h5_path,
                visualization_config=visualization_config,
                artifact_suffix=artifact_suffix,
            )

        return {
            "task_name": "gaussian_progress_fitting",
            "metric_name": "num_bins",
            "metric_value": float(fitting_config["num_bins"]),
            "num_bins": int(fitting_config["num_bins"]),
            "bin_counts": model["bin_counts"].astype(int).tolist(),
            "bin_final_covariance_ranks": model[
                "bin_final_covariance_ranks"
            ].astype(int).tolist(),
            "bin_pre_pca_final_covariance_ranks": (
                model["bin_pre_pca_final_covariance_ranks"].astype(int).tolist()
                if model["bin_pre_pca_final_covariance_ranks"] is not None
                else None
            ),
            "num_expert_videos": len(video_ids),
            "num_expert_features": int(model["num_expert_features"]),
            "enable_pca": bool(model["enable_pca"]),
            "input_embedding_dim": int(model["input_embedding_dim"]),
            "embedding_dim": int(model["embedding_dim"]),
            "covariance_mode": fitting_config["covariance_mode"],
            "output_dir": str(run_output_dir),
            "output_h5_path": str(output_h5_path),
            "visualization_enabled": visualization_config["enabled"],
            "output_visualization_path": visualization_result[
                "output_visualization_path"
            ],
            "output_real_visualization_path": visualization_result[
                "output_real_visualization_path"
            ],
            "output_real_video_idx_visualization_path": visualization_result[
                "output_real_video_idx_visualization_path"
            ],
            "visualization_num_tsne_points": visualization_result[
                "num_tsne_points"
            ],
            "visualization_perplexity_used": visualization_result[
                "perplexity_used"
            ],
            "visualization_num_real_points": visualization_result[
                "num_real_points"
            ],
            "visualization_num_synthetic_points": visualization_result[
                "num_synthetic_points"
            ],
            "real_visualization_num_tsne_points": visualization_result[
                "real_num_tsne_points"
            ],
            "real_visualization_perplexity_used": visualization_result[
                "real_perplexity_used"
            ],
        }
