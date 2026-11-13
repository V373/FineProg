"""Local-image progress from PCA, temporal-bin Gaussians and conformal OOD."""

from pathlib import Path
import csv
import hashlib
import json

import h5py
import numpy as np
from scipy.linalg import solve_triangular


DEFAULTS = {
    "enable_pca": True,
    "pca_dim": 32,
    "num_bins": 20,
    "variance_floor": 1.0e-8,
    "temperature": 1.0e4,
    "ood_threshold": 0.05,
    "batch_size": 20,
}


def temporal_progress(length, target_steps=None):
    """Temporal labels retain the original first/last-step normalization."""
    if length < 2:
        raise ValueError("Fitting/calibration trajectories need at least two frames")
    steps = None if target_steps is None else np.asarray(target_steps, dtype=np.float64)
    if steps is None or steps.shape != (length,) or steps[-1] == steps[0]:
        progress = np.arange(length, dtype=np.float64) / (length - 1)
    else:
        progress = (steps - steps[0]) / (steps[-1] - steps[0])
    if not np.isfinite(progress).all() or progress.min() < -1e-12 or progress.max() > 1 + 1e-12:
        raise ValueError("Temporal progress must be finite and lie in [0, 1]")
    return progress.clip(0, 1)


def _trajectories(embeddings):
    arrays = [np.asarray(array, dtype=np.float64) for array in embeddings]
    if not arrays or any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Expected finite [T,D] embedding trajectories")
    if len({array.shape[1] for array in arrays}) != 1:
        raise ValueError("Embedding dimensions differ between trajectories")
    return arrays


def _progresses(embeddings, target_steps):
    steps = [None] * len(embeddings) if target_steps is None else target_steps
    if len(steps) != len(embeddings):
        raise ValueError("One target_steps array is required per trajectory")
    return [temporal_progress(len(array), times) for array, times in zip(embeddings, steps)]


def _bin_indices(progress, num_bins):
    return np.minimum(np.floor(num_bins * progress).astype(np.int64), num_bins - 1)


def project_embeddings(model, embeddings):
    array = np.asarray(embeddings, dtype=np.float64)
    if model["enable_pca"]:
        array = (array - model["pca_mean"][None]) @ model["pca_components"].T
    return array


def fit_gaussian(embeddings, config=None, target_steps=None):
    """Fit on expert trajectories; PCA sees every expert frame exactly once."""
    cfg = {**DEFAULTS, **(config or {})}
    trajectories = _trajectories(embeddings)
    progresses = _progresses(trajectories, target_steps)
    joined = np.concatenate(trajectories)
    input_dim = joined.shape[1]
    enable_pca = bool(cfg["enable_pca"])
    pca_mean, pca_components = None, None
    if enable_pca:
        from sklearn.decomposition import PCA

        pca_dim = int(cfg["pca_dim"])
        if not 1 <= pca_dim <= min(len(joined), input_dim - 1):
            raise ValueError("pca_dim must be smaller than the input embedding dimension")
        pca = PCA(n_components=pca_dim, whiten=False, svd_solver="full")
        joined = np.asarray(pca.fit_transform(joined), dtype=np.float64)
        pca_mean = np.asarray(pca.mean_, dtype=np.float64)
        pca_components = np.asarray(pca.components_, dtype=np.float64)
    num_bins = int(cfg["num_bins"])
    if num_bins < 2 or float(cfg["variance_floor"]) <= 0:
        raise ValueError("num_bins must be >=2 and variance_floor must be positive")
    indices = _bin_indices(np.concatenate(progresses), num_bins)
    counts = np.bincount(indices, minlength=num_bins).astype(np.int64)
    if np.any(counts < 2):
        raise ValueError(f"Every progress bin needs >=2 samples; counts={counts.tolist()}")
    dimension = joined.shape[1]
    means = np.empty((num_bins, dimension), dtype=np.float64)
    covariances = np.empty((num_bins, dimension, dimension), dtype=np.float64)
    log_determinants = np.empty(num_bins, dtype=np.float64)
    for index in range(num_bins):
        values = joined[indices == index]
        means[index] = values.mean(0)
        residuals = values - means[index]
        covariance = (residuals.T @ residuals) / float(len(values) - 1)
        values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        covariance = (vectors * np.maximum(values, cfg["variance_floor"])[None]) @ vectors.T
        covariances[index] = 0.5 * (covariance + covariance.T)
        sign, log_determinants[index] = np.linalg.slogdet(covariances[index])
        if sign <= 0:
            raise ValueError("Fitted covariance is not positive definite")
    return {
        "enable_pca": enable_pca,
        "input_embedding_dim": input_dim,
        "embedding_dim": dimension,
        "num_bins": num_bins,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
        "bin_progress_values": np.arange(num_bins, dtype=np.float64) / (num_bins - 1),
        "bin_means": means,
        "bin_final_covariances": covariances,
        "bin_log_determinants": log_determinants,
        "bin_counts": counts,
        "cholesky_factors": np.linalg.cholesky(covariances),
        "variance_floor": float(cfg["variance_floor"]),
        "temperature": float(cfg["temperature"]),
        "ood_threshold": float(cfg["ood_threshold"]),
    }


def squared_mahalanobis(model, embeddings):
    array = np.asarray(embeddings, dtype=np.float64)
    distances = np.empty((len(array), model["num_bins"]), dtype=np.float64)
    for index in range(model["num_bins"]):
        whitened = solve_triangular(model["cholesky_factors"][index],
                                   (array - model["bin_means"][index]).T,
                                   lower=True, check_finite=False)
        distances[:, index] = np.einsum("dt,dt->t", whitened, whitened, optimize=True)
    return distances


def calibrate(model, embeddings, target_steps=None):
    """Collect only the matching temporal-bin Gaussian distance per frame."""
    trajectories = _trajectories(embeddings)
    progresses = _progresses(trajectories, target_steps)
    parts = [[] for _ in range(model["num_bins"])]
    for array, progress in zip(trajectories, progresses):
        distances = squared_mahalanobis(model, project_embeddings(model, array))
        indices = _bin_indices(progress, model["num_bins"])
        for index in range(model["num_bins"]):
            selected = distances[indices == index, index]
            if len(selected):
                parts[index].append(selected)
    if any(not values for values in parts):
        raise ValueError("Calibration has empty temporal bins")
    return [np.concatenate(values) for values in parts]


def conformal_p_values(distances, calibration_bins):
    """Ties are excluded: p=(1 + count(calibration_distance > query))/(n+1)."""
    p_values = np.empty_like(distances, dtype=np.float64)
    for index, values in enumerate(calibration_bins):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not len(values):
            raise ValueError("Each calibration bin needs at least one distance")
        counts = (distances[:, index, None] < values[None]).sum(axis=1)
        p_values[:, index] = (1.0 + counts) / (1.0 + len(values))
    return p_values


def gate_progress(raw_progress, is_ood, last_progress=0.0):
    gated = np.empty_like(raw_progress, dtype=np.float64)
    for index, value in enumerate(raw_progress):
        if not is_ood[index]:
            last_progress = float(value)
        gated[index] = last_progress
    return gated


def predict_embeddings(model, embeddings, temperature=None, ood_threshold=None,
                       last_progress=0.0):
    """Use the same Gaussian and causal gating path for offline and online input."""
    temperature = float(model["temperature"] if temperature is None else temperature)
    threshold = float(model["ood_threshold"] if ood_threshold is None else ood_threshold)
    if not np.isfinite(temperature) or temperature <= 0 or not 0 < threshold < 1:
        raise ValueError("temperature must be positive and ood_threshold must lie in (0,1)")
    distances = squared_mahalanobis(model, project_embeddings(model, embeddings))
    log_likelihood = -0.5 * (distances + model["bin_log_determinants"][None]
                             + model["embedding_dim"] * np.log(2 * np.pi))
    logits = log_likelihood / temperature
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    posterior = weights / weights.sum(axis=1, keepdims=True)
    raw = (posterior @ model["bin_progress_values"]).clip(0, 1)
    p_values = conformal_p_values(distances, model["calibration_bins"])
    is_ood = (p_values < threshold).all(axis=1)
    return {"raw_progress": raw, "gated_progress": gate_progress(raw, is_ood, last_progress),
            "max_p": p_values.max(axis=1), "is_ood": is_ood, "posterior": posterior,
            "p_values": p_values, "squared_mahalanobis": distances,
            "log_likelihood": log_likelihood}


def save_progress_model(model, path):
    """One portable artifact contains fitted PCA, Gaussians and calibration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        for key in ("enable_pca", "input_embedding_dim", "embedding_dim", "num_bins",
                    "temperature", "ood_threshold", "variance_floor"):
            file.attrs[key] = model[key]
        file.attrs["embedding_normalization"] = "l2"
        if "checkpoint_sha256" in model:
            file.attrs["checkpoint_sha256"] = model["checkpoint_sha256"]
        group = file.create_group("model")
        for key in ("bin_progress_values", "bin_means", "bin_final_covariances", "bin_log_determinants",
                    "bin_counts", "pca_mean", "pca_components"):
            if model.get(key) is not None:
                group.create_dataset(key, data=model[key])
        calibration = file.create_group("calibration")
        for index, values in enumerate(model["calibration_bins"]):
            calibration.create_dataset(str(index), data=values)
    return path


def load_progress_model(path):
    with h5py.File(path, "r") as file:
        model = dict(file.attrs)
        model.update({key: value[:] for key, value in file["model"].items()})
        model["calibration_bins"] = [file[f"calibration/{index}"][:]
                                     for index in range(model["num_bins"])]
    model["cholesky_factors"] = np.linalg.cholesky(model["bin_final_covariances"])
    return model


def _resolve(path):
    from .config import resolve_path

    return resolve_path(path)


def _load_encoder(checkpoint, device):
    from .models import load_local

    return load_local(checkpoint, device=device).eval()


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_checkpoint(model, checkpoint):
    if ("checkpoint_sha256" in model
            and _checkpoint_sha256(checkpoint) != model["checkpoint_sha256"]):
        raise ValueError("Progress model and encoder checkpoint SHA256 do not match")


def _encode_rgb(encoder, frames, device):
    import torch

    tensor = torch.as_tensor(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).to(device)
    with torch.inference_mode():
        return encoder(tensor).float().cpu().numpy()


def _extract_h5(encoder, path, batch_size, device):
    with h5py.File(path, "r") as file:
        for video_id in sorted(file["videos"]):
            frames = file[f"videos/{video_id}/frames"]
            chunks = [_encode_rgb(encoder, frames[start:start + batch_size], device)
                      for start in range(0, len(frames), batch_size)]
            if not chunks:
                raise ValueError(f"Empty video: {video_id}")
            yield video_id, np.concatenate(chunks), np.arange(len(frames), dtype=np.int64)


def fit_progress(task, cfg, checkpoint, output, device="cpu"):
    """Extract train/calibration RGB with this checkpoint and fit progress.h5."""
    cfg = {**DEFAULTS, **cfg}
    checkpoint = _resolve(checkpoint)
    encoder = _load_encoder(checkpoint, device)
    expert = list(_extract_h5(encoder, task["data"]["train"], int(cfg["batch_size"]), device))
    calibration = list(_extract_h5(encoder, task["data"]["calibration"],
                                   int(cfg["batch_size"]), device))
    model = fit_gaussian([item[1] for item in expert], cfg, [item[2] for item in expert])
    model["checkpoint_sha256"] = _checkpoint_sha256(checkpoint)
    model["calibration_bins"] = calibrate(model, [item[1] for item in calibration],
                                         [item[2] for item in calibration])
    output = _resolve(output)
    path = output if output.suffix == ".h5" else output / "progress.h5"
    return save_progress_model(model, path)


def _save_plot(path, prediction, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.arange(len(prediction["raw_progress"]))
    figure, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True, constrained_layout=True)
    axes[0].plot(frames, prediction["raw_progress"], label="Raw")
    axes[0].plot(frames, prediction["gated_progress"], label="Gated")
    axes[0].fill_between(frames, 0, 1, where=prediction["is_ood"], alpha=0.15, color="red")
    axes[0].set(ylabel="Progress", ylim=(0, 1))
    axes[0].legend()
    axes[1].plot(frames, prediction["max_p"])
    axes[1].axhline(threshold, color="red", linestyle="--")
    axes[1].set(xlabel="Frame", ylabel="Max p", ylim=(0, 1))
    figure.savefig(path, dpi=120)
    plt.close(figure)


def predict_progress_file(task, cfg, checkpoint, model_path, split, output, device="cpu"):
    """Write per-video CSV/NPZ and a compact PNG; inputs are processed RGB H5."""
    checkpoint = _resolve(checkpoint)
    model = load_progress_model(_resolve(model_path))
    _check_checkpoint(model, checkpoint)
    encoder = _load_encoder(checkpoint, device)
    temperature = float(cfg.get("temperature", model["temperature"]))
    threshold = float(cfg.get("ood_threshold", model["ood_threshold"]))
    output = _resolve(output)
    output.mkdir(parents=True, exist_ok=True)
    for video_id, embeddings, frame_idx in _extract_h5(
            encoder, task["data"][split], int(cfg.get("batch_size", DEFAULTS["batch_size"])), device):
        prediction = predict_embeddings(model, embeddings, temperature, threshold)
        np.savez_compressed(output / f"{video_id}.npz", frame_idx=frame_idx, **prediction)
        with (output / f"{video_id}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_idx", "raw_progress", "gated_progress", "max_p", "is_ood"])
            writer.writerows(zip(frame_idx, prediction["raw_progress"], prediction["gated_progress"],
                                  prediction["max_p"], prediction["is_ood"]))
        _save_plot(output / f"{video_id}.png", prediction, threshold)
    (output / "config.json").write_text(json.dumps({"temperature": temperature,
        "ood_threshold": threshold, "split": split}, indent=2) + "\n")
    return output


class ProgressEstimator:
    """Causal single-frame inference; only the last in-distribution progress is stateful."""

    def __init__(self, checkpoint, model_path, device="cpu", temperature=None, ood_threshold=None):
        self.device = device
        checkpoint = _resolve(checkpoint)
        self.model = load_progress_model(_resolve(model_path))
        _check_checkpoint(self.model, checkpoint)
        self.encoder = _load_encoder(checkpoint, device)
        self.temperature = temperature
        self.ood_threshold = ood_threshold
        self.reset()

    def reset(self):
        self.last_progress = 0.0

    def predict_frame(self, rgb):
        array = np.asarray(rgb)
        if array.shape != (224, 224, 3):
            raise ValueError("Expected an RGB frame with shape [224,224,3]")
        embeddings = _encode_rgb(self.encoder, array[None], self.device)
        prediction = predict_embeddings(self.model, embeddings, self.temperature,
                                          self.ood_threshold, self.last_progress)
        self.last_progress = float(prediction["gated_progress"][0])
        return {"raw_progress": float(prediction["raw_progress"][0]),
                "gated_progress": self.last_progress, "max_p": float(prediction["max_p"][0]),
                "is_ood": bool(prediction["is_ood"][0]), "posterior": prediction["posterior"][0]}
