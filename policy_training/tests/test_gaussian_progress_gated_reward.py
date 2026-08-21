from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from reward_model.gaussian_progress_gated_reward import (  # noqa: E402
	GaussianProgressGatedProvider,
)
from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_pred import (  # noqa: E402
	_infer_one_trajectory,
)


class _RecordingEncoder(torch.nn.Module):
	def __init__(self, embedding: tuple[float, float] = (0.0, 0.0)):
		super().__init__()
		self.embedding = embedding
		self.inputs: list[torch.Tensor] = []

	def forward(self, frames: torch.Tensor) -> torch.Tensor:
		self.inputs.append(frames.detach().cpu())
		output = frames.new_zeros((1, 1, 2))
		output[0, 0] = output.new_tensor(self.embedding)
		return output


def _write_gaussian_model(path: Path, normalization: str = "none") -> None:
	means = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
	covariances = np.stack([np.eye(2), np.eye(2)])
	with h5py.File(path, "w") as h5_file:
		h5_file.attrs["embedding_normalization"] = normalization
		h5_file.attrs["enable_pca"] = False
		h5_file.attrs["input_embedding_dim"] = 2
		h5_file.attrs["embedding_dim"] = 2
		model = h5_file.create_group("model")
		model.create_dataset("bin_progress_values", data=[0.0, 1.0])
		model.create_dataset("bin_means", data=means)
		model.create_dataset("bin_independent_covariances", data=covariances)
		model.create_dataset("shared_covariance", data=np.eye(2))
		model.create_dataset("bin_final_covariances", data=covariances)
		model.create_dataset("bin_log_determinants", data=np.zeros(2))
		model.create_dataset("bin_counts", data=np.full(2, 2, dtype=np.int64))


def _write_calibration(path: Path, normalization: str = "none") -> None:
	with h5py.File(path, "w") as h5_file:
		h5_file.attrs["embedding_normalization"] = normalization
		video = h5_file.create_group("videos").create_group("000001")
		video.create_dataset(
			"embeddings",
			data=np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64),
		)
		video.create_dataset("target_steps", data=np.array([0, 1]))


def _make_provider(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	*,
	encoder: _RecordingEncoder | None = None,
	threshold: float = 0.5,
) -> GaussianProgressGatedProvider:
	model_path = tmp_path / "model.h5"
	calibration_path = tmp_path / "calibration.h5"
	_write_gaussian_model(model_path)
	_write_calibration(calibration_path)
	encoder = encoder or _RecordingEncoder()
	monkeypatch.setattr(
		GaussianProgressGatedProvider,
		"_build_encoder",
		lambda self: encoder,
	)
	return GaussianProgressGatedProvider(
		checkpoint_path=tmp_path / "unused.pt",
		gaussian_model_h5_path=model_path,
		calibration_h5_path=calibration_path,
		device="cpu",
		ood_p_value_threshold=threshold,
		posterior_temperature=1.0,
	)


def _obs(value: int) -> dict[str, np.ndarray]:
	return {
		"agentview_image": np.full((4, 5, 3), value, dtype=np.uint8),
	}


def test_current_context_matches_t_minus_15_and_t(tmp_path, monkeypatch):
	encoder = _RecordingEncoder()
	provider = _make_provider(tmp_path, monkeypatch, encoder=encoder)

	provider.reset(_obs(0))
	first_context = encoder.inputs[-1]
	assert first_context.shape == (1, 1, 2, 3, 224, 224)
	assert first_context[0, 0, 0].mean() == pytest.approx(0.0)
	assert first_context[0, 0, 1].mean() == pytest.approx(0.0)

	for value in range(1, 16):
		provider.advance(_obs(value * 10))
	context_at_15 = encoder.inputs[-1]
	assert context_at_15[0, 0, 0].mean() == pytest.approx(0.0)
	assert context_at_15[0, 0, 1].mean() == pytest.approx(150.0 / 255.0)

	provider.advance(_obs(160))
	context_at_16 = encoder.inputs[-1]
	assert context_at_16[0, 0, 0].mean() == pytest.approx(10.0 / 255.0)
	assert context_at_16[0, 0, 1].mean() == pytest.approx(160.0 / 255.0)

	prepared_dark_uint8 = provider._prepare_frame_array(_obs(1)["agentview_image"])
	assert prepared_dark_uint8.mean() == pytest.approx(1.0 / 255.0)
	prepared_chw = provider._prepare_frame_array(
		np.ones((3, 4, 5), dtype=np.float32)
	)
	assert prepared_chw.shape == (224, 224, 3)
	assert prepared_chw.mean() == pytest.approx(1.0)


def test_online_math_matches_existing_gaussian_trajectory_helper(tmp_path, monkeypatch):
	provider = _make_provider(tmp_path, monkeypatch)
	actual = provider.reset(_obs(0))

	expected = _infer_one_trajectory(
		query_embeddings=np.array([[0.0, 0.0]], dtype=np.float64),
		model=provider.gaussian_model,
		posterior_temperature=1.0,
		entropy_epsilon=1.0e-12,
		calibration_distance_bins=provider.calibration_distance_bins,
		ood_p_value_threshold=0.5,
	)

	assert not bool(expected["is_ood"][0])
	assert actual == pytest.approx(float(expected["progress_gated"][0]))


def test_gating_holds_last_in_distribution_progress(tmp_path, monkeypatch):
	provider = _make_provider(tmp_path, monkeypatch)
	sequence = iter(
		[
			(0.1, True),
			(0.2, True),
			(0.3, False),
			(0.4, True),
			(0.6, False),
		]
	)
	monkeypatch.setattr(provider, "_infer_progress_mean_and_ood", lambda: next(sequence))

	trace = [provider.reset(_obs(0))]
	for value in range(1, 5):
		trace.append(provider.advance(_obs(value * 10)))

	np.testing.assert_allclose(trace, [0.0, 0.0, 0.3, 0.3, 0.6])

	monkeypatch.setattr(
		provider,
		"_infer_progress_mean_and_ood",
		lambda: (0.9, True),
	)
	assert provider.reset(_obs(0)) == 0.0


def test_frame_trace_restores_online_state(tmp_path, monkeypatch):
	provider = _make_provider(tmp_path, monkeypatch)
	provider.reset(_obs(0))
	provider.advance(_obs(10))
	previous_frames = [frame.copy() for frame in provider.frame_history]
	previous_progress = provider.progress_current
	previous_last = provider._last_in_distribution_progress

	trace = provider.infer_progress_trace_from_frames(
		np.stack([_obs(20)["agentview_image"], _obs(30)["agentview_image"]])
	)

	assert trace.shape == (2,)
	assert provider.progress_current == previous_progress
	assert provider._last_in_distribution_progress == previous_last
	assert len(provider.frame_history) == len(previous_frames)
	for actual, expected in zip(provider.frame_history, previous_frames):
		np.testing.assert_array_equal(actual, expected)


def test_checkpoint_and_gaussian_normalization_must_match(tmp_path):
	model_path = tmp_path / "model.h5"
	calibration_path = tmp_path / "calibration.h5"
	checkpoint_path = tmp_path / "encoder.pt"
	_write_gaussian_model(model_path, normalization="none")
	_write_calibration(calibration_path, normalization="none")
	torch.save(
		{"model_state_dict": {}, "embedding_normalization": "l2"},
		checkpoint_path,
	)

	with pytest.raises(ValueError, match="embedding_normalization mismatch"):
		GaussianProgressGatedProvider(
			checkpoint_path=checkpoint_path,
			gaussian_model_h5_path=model_path,
			calibration_h5_path=calibration_path,
			device="cpu",
			ood_p_value_threshold=0.5,
		)


@pytest.mark.parametrize(
	("name", "value"),
	[
		("posterior_temperature", 0.0),
		("posterior_temperature", np.inf),
		("ood_p_value_threshold", 0.0),
		("ood_p_value_threshold", 1.0),
	],
)
def test_invalid_inference_parameter_is_rejected(tmp_path, name, value):
	kwargs = {
		"checkpoint_path": tmp_path / "missing.pt",
		"gaussian_model_h5_path": tmp_path / "missing-model.h5",
		"calibration_h5_path": tmp_path / "missing-calibration.h5",
		"device": "cpu",
		"ood_p_value_threshold": 0.5,
		"posterior_temperature": 1.0,
	}
	kwargs[name] = value
	with pytest.raises(ValueError, match=name):
		GaussianProgressGatedProvider(**kwargs)
