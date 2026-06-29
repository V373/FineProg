"""TCC expert-projection dense reward provider for online SAC."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F


_FINEPROG_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _FINEPROG_ROOT.parent
if str(_WORKSPACE_ROOT) not in sys.path:
	sys.path.insert(0, str(_WORKSPACE_ROOT))


@dataclass
class DenseRewardResult:
	reward: float
	progress_current: float
	sparse_reward: float
	progress_next: float | None = None
	shaping_reward: float | None = None


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
	shifted = logits - logits.max(axis=axis, keepdims=True)
	exp_logits = np.exp(shifted)
	return exp_logits / exp_logits.sum(axis=axis, keepdims=True)


def project_embedding_to_progress(
	query_embedding: np.ndarray,
	expert_embeddings: np.ndarray,
	temperature: float,
) -> float:
	query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
	expert = np.asarray(expert_embeddings, dtype=np.float32)
	if expert.ndim != 2:
		raise ValueError(f"expert_embeddings must be 2-D, got shape {expert.shape}.")
	if query.shape[1] != expert.shape[1]:
		raise ValueError(
			f"Embedding dim mismatch: query dim {query.shape[1]} vs expert dim {expert.shape[1]}."
		)
	if float(temperature) <= 0.0:
		raise ValueError("projection_temperature must be > 0.")

	q_sq = (query ** 2).sum(axis=1, keepdims=True)
	e_sq = (expert ** 2).sum(axis=1, keepdims=True).T
	dists = q_sq + e_sq - 2.0 * query @ expert.T
	np.clip(dists, 0.0, None, out=dists)
	alpha = stable_softmax(-dists / float(temperature), axis=1)
	expert_indices = np.arange(expert.shape[0], dtype=np.float64)
	soft_expert_index = float((alpha @ expert_indices)[0])
	denom = max(float(expert.shape[0] - 1), 1.0)
	return float(np.clip(soft_expert_index / denom, 0.0, 1.0))


def project_embedding_to_progress_torch(
	query_embedding: torch.Tensor,
	expert_embeddings: torch.Tensor,
	temperature: float,
) -> torch.Tensor:
	"""Torch version of project_embedding_to_progress.

	Returns one progress value per query row and keeps the computation on the
	query tensor's device.
	"""
	if float(temperature) <= 0.0:
		raise ValueError("projection_temperature must be > 0.")
	query = query_embedding.to(dtype=torch.float32)
	if query.ndim == 1:
		query = query.reshape(1, -1)
	expert = expert_embeddings.to(device=query.device, dtype=torch.float32)
	if expert.ndim != 2:
		raise ValueError(f"expert_embeddings must be 2-D, got shape {tuple(expert.shape)}.")
	if query.shape[1] != expert.shape[1]:
		raise ValueError(
			f"Embedding dim mismatch: query dim {query.shape[1]} vs expert dim {expert.shape[1]}."
		)

	q_sq = (query ** 2).sum(dim=1, keepdim=True)
	e_sq = (expert ** 2).sum(dim=1, keepdim=True).transpose(0, 1)
	dists = torch.clamp(q_sq + e_sq - 2.0 * query @ expert.transpose(0, 1), min=0.0)
	alpha = torch.softmax(-dists / float(temperature), dim=1)
	expert_indices = torch.arange(expert.shape[0], device=query.device, dtype=torch.float32)
	soft_expert_index = alpha @ expert_indices
	denom = max(float(expert.shape[0] - 1), 1.0)
	return torch.clamp(soft_expert_index / denom, 0.0, 1.0)


def load_expert_mean_embeddings(path: str | Path, expert_group: str = "videos/mean") -> np.ndarray:
	with h5py.File(str(path), "r") as h5_file:
		group_path = str(expert_group).strip("/")
		if group_path not in h5_file:
			raise KeyError(f"Expert group '{expert_group}' not found in {path}.")
		group = h5_file[group_path]
		if "embeddings" not in group:
			raise KeyError(f"Dataset '{expert_group}/embeddings' not found in {path}.")
		embeddings = np.asarray(group["embeddings"], dtype=np.float32)
	if embeddings.ndim != 2:
		raise ValueError(f"Expert embeddings must be 2-D, got shape {embeddings.shape}.")
	return embeddings


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
	checkpoint = torch.load(str(checkpoint_path), map_location=device)
	if isinstance(checkpoint, dict):
		if "model_state_dict" in checkpoint:
			state_dict = checkpoint["model_state_dict"]
		elif "state_dict" in checkpoint:
			state_dict = checkpoint["state_dict"]
		else:
			state_dict = checkpoint
	else:
		state_dict = checkpoint
	model.load_state_dict(state_dict)


class TCCExpertProjectionDenseRewardProvider:
	"""Infer online progress with a TCC encoder and convert it to dense reward."""

	def __init__(
		self,
		checkpoint_path: str | Path | None,
		expert_path_h5: str | Path | None,
		device: str | torch.device,
		*,
		expert_group: str = "videos/mean",
		projection_temperature: float = 0.1,
		clip_len: int = 20,
		context_size: int = 2,
		context_stride: int = 15,
		pretrained: bool = True,
		train_config_path: str | Path | None = None,
		image_key: str = "agentview_image",
		image_height: int = 224,
		image_width: int = 224,
		sparse_scale: float = 1.0,
		encoder: Optional[torch.nn.Module] = None,
		expert_embeddings: Optional[np.ndarray] = None,
		load_model: bool = True,
		use_torch_projection: bool = True,
	):
		self.device = torch.device(device)
		self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
		self.expert_path_h5 = str(expert_path_h5) if expert_path_h5 else None
		self.expert_group = str(expert_group)
		self.projection_temperature = float(projection_temperature)
		self.clip_len = int(clip_len)
		self.context_size = int(context_size)
		self.context_stride = int(context_stride)
		self.pretrained = bool(pretrained)
		self.train_config_path = str(train_config_path) if train_config_path else None
		self.image_key = str(image_key)
		self.image_height = int(image_height)
		self.image_width = int(image_width)
		self.sparse_scale = float(sparse_scale)
		self.use_torch_projection = bool(use_torch_projection)
		self.timing_callback = None
		self.profile_cuda = False
		if self.clip_len <= 0 or self.context_size <= 0 or self.context_stride <= 0:
			raise ValueError("clip_len, context_size, and context_stride must be positive.")

		self.history_len = self.clip_len - 1 + (self.context_size - 1) * self.context_stride + 1
		self.frame_history: deque[np.ndarray] = deque(maxlen=self.history_len)
		self.progress_current: Optional[float] = None

		self.expert_embeddings = (
			np.asarray(expert_embeddings, dtype=np.float32)
			if expert_embeddings is not None
			else load_expert_mean_embeddings(self.expert_path_h5, self.expert_group)
		)
		self.expert_embeddings_t: Optional[torch.Tensor] = None
		if self.use_torch_projection:
			self.expert_embeddings_t = torch.as_tensor(
				self.expert_embeddings,
				device=self.device,
				dtype=torch.float32,
			)
		self.encoder = encoder if encoder is not None else None
		if self.encoder is None and load_model:
			self.encoder = self._build_encoder()

	def _sync_for_profile(self) -> None:
		if self.profile_cuda and self.device.type == "cuda" and torch.cuda.is_available():
			torch.cuda.synchronize(self.device)

	@contextmanager
	def _profile_stage(self, name: str):
		if self.timing_callback is None:
			yield
			return
		self._sync_for_profile()
		start = time.perf_counter()
		try:
			yield
		finally:
			self._sync_for_profile()
			self.timing_callback(name, time.perf_counter() - start)

	def _build_encoder(self) -> torch.nn.Module:
		if not self.checkpoint_path:
			raise ValueError("checkpoint_path is required when load_model=True.")
		from fineprog.models.encoder import TCCEncoder

		encoder = TCCEncoder(
			clip_len=self.clip_len,
			context_size=self.context_size,
			context_stride=self.context_stride,
			pretrained=self.pretrained,
			train_config_path=self.train_config_path,
		)
		encoder.to(self.device)
		load_checkpoint(encoder, self.checkpoint_path, self.device)
		encoder.eval()
		return encoder

	def _prepare_frame_array(self, frame: Any) -> np.ndarray:
		with self._profile_stage("provider_prepare_frame"):
			frame = np.asarray(frame)
			if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
				frame = np.transpose(frame, (1, 2, 0))
			if frame.ndim != 3 or frame.shape[-1] != 3:
				raise ValueError(f"Expected RGB image for '{self.image_key}', got shape {frame.shape}.")
			if frame.shape[0] != self.image_height or frame.shape[1] != self.image_width:
				frame_t = torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
				frame_t = F.interpolate(
					frame_t,
					size=(self.image_height, self.image_width),
					mode="bilinear",
					align_corners=False,
				)
				frame = frame_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
			if frame.dtype != np.float32:
				frame = frame.astype(np.float32)
			if frame.max(initial=0.0) > 1.5:
				frame = frame / 255.0
			return np.clip(frame, 0.0, 1.0).astype(np.float32)

	def _prepare_frame(self, obs_dict: dict[str, Any]) -> np.ndarray:
		if self.image_key not in obs_dict:
			raise KeyError(f"Observation key '{self.image_key}' missing from env observation.")
		return self._prepare_frame_array(obs_dict[self.image_key])

	def _history_frame(self, index_from_start: int) -> np.ndarray:
		if not self.frame_history:
			raise RuntimeError("Frame history is empty; call reset(obs_dict) first.")
		idx = min(max(int(index_from_start), 0), len(self.frame_history) - 1)
		return self.frame_history[idx]

	def build_rolling_clip(self) -> torch.Tensor:
		with self._profile_stage("provider_build_clip"):
			frames: list[np.ndarray] = []
			current_index = len(self.frame_history) - 1
			first_target_index = current_index - (self.clip_len - 1)
			for target_offset in range(self.clip_len):
				target_index = first_target_index + target_offset
				for ctx_offset in range(self.context_size):
					past_offset = (self.context_size - 1 - ctx_offset) * self.context_stride
					frames.append(self._history_frame(target_index - past_offset))
			array = np.stack(frames, axis=0).reshape(
				self.clip_len,
				self.context_size,
				self.image_height,
				self.image_width,
				3,
			)
			tensor = torch.from_numpy(array).permute(0, 1, 4, 2, 3).unsqueeze(0)
			return tensor.to(self.device, dtype=torch.float32)

	def _project_embeddings_to_progress_torch(self, embeddings: torch.Tensor) -> torch.Tensor:
		if self.expert_embeddings_t is None or self.expert_embeddings_t.device != embeddings.device:
			self.expert_embeddings_t = torch.as_tensor(
				self.expert_embeddings,
				device=embeddings.device,
				dtype=torch.float32,
			)
		return project_embedding_to_progress_torch(
			embeddings,
			self.expert_embeddings_t,
			self.projection_temperature,
		)

	def infer_progress_current(self) -> float:
		if self.encoder is None:
			raise RuntimeError("Reward provider has no encoder.")
		frames = self.build_rolling_clip()
		with torch.inference_mode():
			with self._profile_stage("provider_encoder_forward"):
				embeddings = self.encoder(frames)
			current_embedding = embeddings[0, -1]
			if self.use_torch_projection:
				with self._profile_stage("provider_projection_torch"):
					progress_t = self._project_embeddings_to_progress_torch(current_embedding)
				return float(progress_t.detach().cpu().item())
			with self._profile_stage("provider_projection_numpy"):
				current_embedding_np = current_embedding.detach().cpu().numpy().astype(np.float32)
				return project_embedding_to_progress(
					current_embedding_np,
					self.expert_embeddings,
					self.projection_temperature,
				)

	def infer_progress_trace_from_frames(self, frames: np.ndarray) -> np.ndarray:
		if self.encoder is None:
			raise RuntimeError("Reward provider has no encoder.")
		array = np.asarray(frames)
		if array.ndim != 4 or array.shape[-1] != 3:
			raise ValueError(f"Expected video frames with shape [T, H, W, 3], got {array.shape}.")

		previous_history = deque(self.frame_history, maxlen=self.history_len)
		previous_progress = self.progress_current
		trace: list[float] = []
		try:
			self.frame_history.clear()
			for frame in array:
				self.frame_history.append(self._prepare_frame_array(frame))
				trace.append(float(self.infer_progress_current()))
		finally:
			self.frame_history = previous_history
			self.progress_current = previous_progress

		return np.asarray(trace, dtype=np.float32)

	def infer_progress_trace_from_video(
		self,
		video_path: str | Path,
		*,
		camera_names: list[str] | None = None,
		camera_index: int = 0,
	) -> np.ndarray:
		import cv2  # noqa: PLC0415

		video_path = Path(video_path)
		if not video_path.is_file():
			raise FileNotFoundError(f"Video file not found: {video_path}")

		cap = cv2.VideoCapture(str(video_path))
		if not cap.isOpened():
			raise ValueError(f"Could not open video file: {video_path}")

		frames: list[np.ndarray] = []
		try:
			while True:
				ok, bgr = cap.read()
				if not ok:
					break
				rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
				if camera_names and len(camera_names) > 1:
					num_cameras = max(1, int(len(camera_names)))
					cam_idx = min(max(int(camera_index), 0), num_cameras - 1)
					camera_width = rgb.shape[1] // num_cameras
					if camera_width > 0:
						left = cam_idx * camera_width
						rgb = rgb[:, left:left + camera_width, :]
				frames.append(rgb)
		finally:
			cap.release()

		if not frames:
			raise ValueError(f"No frames decoded from video: {video_path}")

		return self.infer_progress_trace_from_frames(np.stack(frames, axis=0))

	def reset(self, obs_dict: dict[str, Any]) -> float:
		self.frame_history.clear()
		self.frame_history.append(self._prepare_frame(obs_dict))
		self.progress_current = self.infer_progress_current()
		return self.progress_current

	def compute_dense_reward(self, sparse_reward: float) -> DenseRewardResult:
		if self.progress_current is None:
			raise RuntimeError("Reward provider has no cached progress; call reset(obs_dict) first.")
		reward = self.sparse_scale * float(sparse_reward) + float(self.progress_current)
		return DenseRewardResult(
			reward=float(reward),
			progress_current=float(self.progress_current),
			sparse_reward=float(sparse_reward),
		)

	def compute_pbrs_reward(
		self,
		sparse_reward: float,
		*,
		pbrs_gamma: float,
		progress_next: float,
		progress_current: float | None = None,
	) -> DenseRewardResult:
		if progress_current is None:
			if self.progress_current is None:
				raise RuntimeError("Reward provider has no cached progress; call reset(obs_dict) first.")
			progress_current = float(self.progress_current)
		else:
			progress_current = float(progress_current)
		progress_next = float(progress_next)
		shaping = float(pbrs_gamma) * progress_next - progress_current
		reward = self.sparse_scale * float(sparse_reward) + shaping
		return DenseRewardResult(
			reward=float(reward),
			progress_current=progress_current,
			progress_next=progress_next,
			shaping_reward=shaping,
			sparse_reward=float(sparse_reward),
		)

	def advance(self, obs_dict: dict[str, Any]) -> float:
		self.frame_history.append(self._prepare_frame(obs_dict))
		self.progress_current = self.infer_progress_current()
		return self.progress_current
