"""Image-to-progress inference with an offline-fitted Gaussian model."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_FINEPROG_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _FINEPROG_ROOT.parent
if str(_WORKSPACE_ROOT) not in sys.path:
	sys.path.insert(0, str(_WORKSPACE_ROOT))

from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_pred import (  # noqa: E402
	_apply_model_pca,
	_compute_conformal_p_values,
	_compute_gaussian_log_likelihood,
	_compute_progress_posterior,
	_compute_squared_mahalanobis,
	_read_calibration_distance_bins,
	_read_gaussian_model,
)
from fineprog.utils.embedding_normalization import (  # noqa: E402
	validate_embedding_normalization,
	validate_embeddings_for_normalization,
)


class GaussianProgressGatedProvider:
	"""Infer stateful OOD-gated progress from online RGB observations."""

	_CONTEXT_SIZE = 2
	_CONTEXT_STRIDE = 15
	_IMAGE_HEIGHT = 224
	_IMAGE_WIDTH = 224
	_HISTORY_LEN = (_CONTEXT_SIZE - 1) * _CONTEXT_STRIDE + 1

	def __init__(
		self,
		checkpoint_path: str | Path,
		gaussian_model_h5_path: str | Path,
		calibration_h5_path: str | Path,
		device: str | torch.device,
		*,
		ood_p_value_threshold: float,
		posterior_temperature: float = 1.0e4,
		image_key: str = "agentview_image",
	):
		self.device = torch.device(device)
		self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
		self.gaussian_model_h5_path = str(
			Path(gaussian_model_h5_path).expanduser().resolve()
		)
		self.calibration_h5_path = str(
			Path(calibration_h5_path).expanduser().resolve()
		)
		self.posterior_temperature = float(posterior_temperature)
		self.ood_p_value_threshold = float(ood_p_value_threshold)
		self.image_key = str(image_key)

		if (
			not np.isfinite(self.posterior_temperature)
			or self.posterior_temperature <= 0.0
		):
			raise ValueError("posterior_temperature must be finite and > 0.")
		if (
			not np.isfinite(self.ood_p_value_threshold)
			or self.ood_p_value_threshold <= 0.0
			or self.ood_p_value_threshold >= 1.0
		):
			raise ValueError("ood_p_value_threshold must be finite and in (0, 1).")

		self.gaussian_model = _read_gaussian_model(self.gaussian_model_h5_path)
		self.calibration_distance_bins = _read_calibration_distance_bins(
			calibration_h5_path=self.calibration_h5_path,
			model=self.gaussian_model,
		)
		self.encoder = self._build_encoder()

		self.frame_history: deque[np.ndarray] = deque(maxlen=self._HISTORY_LEN)
		self.progress_current: float | None = None
		self._last_in_distribution_progress = 0.0

	def _build_encoder(self) -> torch.nn.Module:
		checkpoint_path = Path(self.checkpoint_path)
		if not checkpoint_path.is_file():
			raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")

		checkpoint = torch.load(
			checkpoint_path,
			map_location=self.device,
			weights_only=True,
		)
		if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
			raise ValueError(
				"Encoder checkpoint must contain a 'model_state_dict' mapping."
			)
		if "embedding_normalization" not in checkpoint:
			raise ValueError(
				"Encoder checkpoint must contain 'embedding_normalization' metadata."
			)
		checkpoint_normalization = validate_embedding_normalization(
			checkpoint["embedding_normalization"],
			self.checkpoint_path,
		)
		model_normalization = self.gaussian_model["embedding_normalization"]
		if checkpoint_normalization != model_normalization:
			raise ValueError(
				"embedding_normalization mismatch: checkpoint is "
				f"{checkpoint_normalization!r}, Gaussian model is {model_normalization!r}."
			)

		from fineprog.models.encoder import TCCEncoder  # noqa: PLC0415

		encoder = TCCEncoder(
			clip_len=1,
			context_size=self._CONTEXT_SIZE,
			context_stride=self._CONTEXT_STRIDE,
			pretrained=False,
			embedding_dim=int(self.gaussian_model["input_embedding_dim"]),
			embedding_normalization=model_normalization,
		)
		encoder.to(self.device)
		encoder.load_state_dict(checkpoint["model_state_dict"])
		encoder.eval()
		return encoder

	def _prepare_frame_array(self, frame: Any) -> np.ndarray:
		array = np.asarray(frame)
		if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
			array = np.transpose(array, (1, 2, 0))
		if array.ndim != 3 or array.shape[-1] != 3:
			raise ValueError(
				f"Expected an RGB image for {self.image_key!r}, got shape {array.shape}."
			)
		if array.shape[0] < 1 or array.shape[1] < 1:
			raise ValueError(f"RGB image has an empty spatial dimension: {array.shape}.")

		integer_pixels = np.issubdtype(array.dtype, np.integer)
		array = array.astype(np.float32, copy=False)
		if not np.isfinite(array).all():
			raise ValueError(f"RGB image for {self.image_key!r} contains NaN or Inf.")
		if array.shape[:2] != (self._IMAGE_HEIGHT, self._IMAGE_WIDTH):
			frame_t = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
			frame_t = F.interpolate(
				frame_t,
				size=(self._IMAGE_HEIGHT, self._IMAGE_WIDTH),
				mode="bilinear",
				align_corners=False,
			)
			array = frame_t.squeeze(0).permute(1, 2, 0).numpy()
		if integer_pixels or float(np.max(array)) > 1.5:
			array = array / 255.0
		return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)

	def _prepare_frame(self, obs_dict: dict[str, Any]) -> np.ndarray:
		if self.image_key not in obs_dict:
			raise KeyError(f"Observation key {self.image_key!r} is missing.")
		return self._prepare_frame_array(obs_dict[self.image_key])

	def _build_current_context(self) -> torch.Tensor:
		if not self.frame_history:
			raise RuntimeError("Frame history is empty; call reset(obs_dict) first.")
		context = np.stack(
			(self.frame_history[0], self.frame_history[-1]),
			axis=0,
		)
		tensor = torch.from_numpy(context).permute(0, 3, 1, 2)
		return tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)

	def _infer_progress_mean_and_ood(self) -> tuple[float, bool]:
		context = self._build_current_context()
		with torch.inference_mode():
			embeddings = self.encoder(context)
		if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (
			1,
			1,
			int(self.gaussian_model["input_embedding_dim"]),
		):
			raise ValueError(
				"Encoder output must have shape "
				f"(1, 1, {self.gaussian_model['input_embedding_dim']}), got "
				f"{getattr(embeddings, 'shape', None)}."
			)

		raw_embedding = np.asarray(
			embeddings[0, 0].detach().cpu().numpy(),
			dtype=np.float64,
		).reshape(1, -1)
		validate_embeddings_for_normalization(
			raw_embedding,
			self.gaussian_model["embedding_normalization"],
			"online encoder output",
		)
		query_embedding = _apply_model_pca(raw_embedding, self.gaussian_model)
		squared_mahalanobis = _compute_squared_mahalanobis(
			query_embeddings=query_embedding,
			bin_means=self.gaussian_model["bin_means"],
			cholesky_factors=self.gaussian_model["cholesky_factors"],
		)
		log_likelihood = _compute_gaussian_log_likelihood(
			squared_mahalanobis=squared_mahalanobis,
			bin_log_determinants=self.gaussian_model["bin_log_determinants"],
			embedding_dim=int(self.gaussian_model["embedding_dim"]),
		)
		posterior = _compute_progress_posterior(
			log_likelihood=log_likelihood,
			posterior_temperature=self.posterior_temperature,
		)
		progress_mean = float(
			np.clip(
				posterior[0] @ self.gaussian_model["bin_progress_values"],
				0.0,
				1.0,
			)
		)
		conformal_p_value = _compute_conformal_p_values(
			squared_mahalanobis=squared_mahalanobis,
			calibration_distance_bins=self.calibration_distance_bins,
		)
		is_ood = bool(
			np.all(conformal_p_value[0] < self.ood_p_value_threshold)
		)
		return progress_mean, is_ood

	def _gate_progress(self, progress_mean: float, is_ood: bool) -> float:
		"""Hold the last in-distribution progress while the query is OOD."""
		if not is_ood:
			self._last_in_distribution_progress = progress_mean
		return float(self._last_in_distribution_progress)

	def _update_progress(self) -> float:
		progress_mean, is_ood = self._infer_progress_mean_and_ood()
		self.progress_current = self._gate_progress(progress_mean, is_ood)
		return self.progress_current

	def reset(self, obs_dict: dict[str, Any]) -> float:
		"""Start a new episode and infer its first gated progress value."""
		self.frame_history.clear()
		self._last_in_distribution_progress = 0.0
		self.progress_current = None
		self.frame_history.append(self._prepare_frame(obs_dict))
		return self._update_progress()

	def advance(self, obs_dict: dict[str, Any]) -> float:
		"""Append one observation and infer the next gated progress value."""
		if not self.frame_history:
			raise RuntimeError("Provider has not been reset for the current episode.")
		self.frame_history.append(self._prepare_frame(obs_dict))
		return self._update_progress()

	def infer_progress_trace_from_frames(self, frames: np.ndarray) -> np.ndarray:
		"""Infer a trace without changing the provider's current online state."""
		"""This is only for test-time debugging."""
		array = np.asarray(frames)
		if array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] < 1:
			raise ValueError(
				"Expected video frames with shape [T, H, W, 3] and T>=1, "
				f"got {array.shape}."
			)
		# save the current online state and restore it after the trace inference
		previous_history = deque(self.frame_history, maxlen=self._HISTORY_LEN)
		previous_progress = self.progress_current
		previous_last_in_distribution = self._last_in_distribution_progress
		trace: list[float] = []
		try:
			trace.append(self.reset({self.image_key: array[0]}))
			for frame in array[1:]:
				trace.append(self.advance({self.image_key: frame}))
		finally:
			# restore the previous online state
			self.frame_history = previous_history
			self.progress_current = previous_progress
			self._last_in_distribution_progress = previous_last_in_distribution

		return np.asarray(trace, dtype=np.float64)
