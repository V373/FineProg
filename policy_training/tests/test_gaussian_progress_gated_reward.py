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
	BatchedGaussianProgressGatedProvider,
	GaussianProgressGatedProvider,
)
from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_pred import (  # noqa: E402
	_infer_one_trajectory,
)
from fineprog.models.encoder import TCCEncoder  # noqa: E402
from fineprog.utils.encoder_checkpoint import checkpoint_dict  # noqa: E402
from fineprog.tests.test_streaming_mamba2 import cpu_memory, tiny_local  # noqa: E402, F401


class _RecordingEncoder(torch.nn.Module):
	memory = None
	context_stride = 15

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
	frame_history_stride: int = 15,
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
		frame_history_stride=frame_history_stride,
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


def test_configurable_frame_history_stride_changes_context_only(tmp_path, monkeypatch):
	encoder = _RecordingEncoder()
	provider = _make_provider(
		tmp_path,
		monkeypatch,
		encoder=encoder,
		frame_history_stride=4,
	)

	provider.reset(_obs(0))
	assert provider.frame_history.maxlen == 5
	for value in range(1, 5):
		provider.advance(_obs(value * 10))
	context_at_4 = encoder.inputs[-1]
	assert context_at_4.shape == (1, 1, 2, 3, 224, 224)
	assert context_at_4[0, 0, 0].mean() == pytest.approx(0.0)
	assert context_at_4[0, 0, 1].mean() == pytest.approx(40.0 / 255.0)

	provider.advance(_obs(50))
	context_at_5 = encoder.inputs[-1]
	assert context_at_5[0, 0, 0].mean() == pytest.approx(10.0 / 255.0)
	assert context_at_5[0, 0, 1].mean() == pytest.approx(50.0 / 255.0)


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
		("frame_history_stride", 0),
		("frame_history_stride", -1),
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


class _MeanEncoder(torch.nn.Module):
	"""Per-sample embedding, so batching may not change any single env's result."""
	memory = None
	context_stride = 15

	def forward(self, frames: torch.Tensor) -> torch.Tensor:
		old_mean = frames[:, 0, 0].mean(dim=(1, 2, 3))
		new_mean = frames[:, 0, 1].mean(dim=(1, 2, 3))
		return torch.stack((new_mean * 5.0, old_mean * 0.5), dim=-1).unsqueeze(1)


def _write_spread_calibration(path: Path) -> None:
	"""Calibration samples spread along both bins, so p-values are not degenerate."""
	num_frames = 20
	positions = np.linspace(0.0, 4.0, num_frames)
	embeddings = np.stack((positions, np.zeros(num_frames)), axis=1)
	with h5py.File(path, "w") as h5_file:
		h5_file.attrs["embedding_normalization"] = "none"
		video = h5_file.create_group("videos").create_group("000001")
		video.create_dataset("embeddings", data=embeddings.astype(np.float64))
		video.create_dataset("target_steps", data=np.arange(num_frames))


def test_batched_provider_matches_single_env_providers(tmp_path, monkeypatch):
	num_envs = 3
	num_steps = 40
	encoder = _MeanEncoder()
	monkeypatch.setattr(
		GaussianProgressGatedProvider,
		"_build_encoder",
		lambda self: encoder,
	)
	model_path = tmp_path / "model.h5"
	calibration_path = tmp_path / "calibration.h5"
	_write_gaussian_model(model_path)
	_write_spread_calibration(calibration_path)
	kwargs = dict(
		checkpoint_path=tmp_path / "unused.pt",
		gaussian_model_h5_path=model_path,
		calibration_h5_path=calibration_path,
		device="cpu",
		ood_p_value_threshold=0.5,
		posterior_temperature=1.0,
	)
	batched = BatchedGaussianProgressGatedProvider(num_envs, **kwargs)
	singles = [GaussianProgressGatedProvider(**kwargs) for _ in range(num_envs)]

	rng = np.random.default_rng(0)
	pixels = rng.integers(0, 256, size=(num_steps, num_envs), dtype=np.uint8)
	resets = rng.random((num_steps, num_envs)) < 0.1
	resets[0] = True

	trace = []
	for step in range(num_steps):
		frames = [
			np.full((224, 224, 3), pixels[step, env], dtype=np.uint8)
			for env in range(num_envs)
		]
		if step == 0:
			batched_progress = batched.reset_all(frames)
		else:
			batched_progress = batched.advance_all(frames, reset_mask=resets[step])
		single_progress = [
			provider.reset({provider.image_key: frames[env]})
			if resets[step, env]
			else provider.advance({provider.image_key: frames[env]})
			for env, provider in enumerate(singles)
		]
		np.testing.assert_allclose(
			batched_progress,
			np.asarray(single_progress),
			rtol=0.0,
			atol=1e-6,
			err_msg=f"progress mismatch at step {step}",
		)
		trace.append(np.asarray(batched_progress))

	trace = np.asarray(trace)
	# guard against a degenerate fixture where every query is OOD and progress stays 0
	assert len(np.unique(np.round(trace, 6))) > 3
	assert np.any(trace[1:] == trace[:-1])


def _checkpoint_assets(tmp_path, encoder, *, legacy=False):
	payload = checkpoint_dict(encoder)
	if legacy:
		payload.pop("encoder_config")
	path = tmp_path / "encoder.pt"
	torch.save(payload, path)
	model_path = tmp_path / "model.h5"
	calibration_path = tmp_path / "calibration.h5"
	_write_gaussian_model(model_path)
	_write_calibration(calibration_path)
	return dict(
		checkpoint_path=path, gaussian_model_h5_path=model_path,
		calibration_h5_path=calibration_path, device="cpu",
		ood_p_value_threshold=0.1, posterior_temperature=1.0,
	)


def _small_encoder(memory=True):
	return TCCEncoder(
		pretrained=False, clip_len=7, context_stride=2, embedding_dim=2,
		memory_config={"enabled": memory, "alpha_init": 0.5},
	).eval()


@pytest.mark.parametrize("memory,legacy,stride,expected_stride", [
	(False, True, None, 15), (False, True, 4, 4),
	(False, False, None, 2), (False, False, 4, 4),
	(True, False, None, 2), (True, False, 2, 2),
])
def test_checkpoint_selects_inference_mode(
	tmp_path, cpu_memory, memory, legacy, stride, expected_stride,
):
	encoder = _small_encoder(memory)
	kwargs = _checkpoint_assets(tmp_path, encoder, legacy=legacy)
	provider = GaussianProgressGatedProvider(**kwargs, frame_history_stride=stride)
	assert provider._use_memory is memory
	assert provider.frame_history_stride == expected_stride
	assert not provider.encoder.training
	assert not any(p.requires_grad for p in provider.encoder.parameters())
	for key, value in encoder.state_dict().items():
		torch.testing.assert_close(provider.encoder.state_dict()[key], value)
	assert np.isfinite(provider.reset(_obs(30)))
	assert np.isfinite(provider.advance(_obs(60)))
	if memory:
		assert provider._stream_state["steps"].tolist() == [2]
	else:
		assert provider.encoder.clip_len == 1
		assert provider._stream_state is None


def test_memory_checkpoint_rejects_missing_structure_weights_and_stride(tmp_path, cpu_memory):
	kwargs = _checkpoint_assets(tmp_path, _small_encoder())
	with pytest.raises(ValueError, match="frame_history_stride.*context_stride"):
		GaussianProgressGatedProvider(**kwargs, frame_history_stride=4)
	payload = torch.load(kwargs["checkpoint_path"], weights_only=True)
	payload["model_state_dict"].pop("memory.alpha")
	torch.save(payload, kwargs["checkpoint_path"])
	with pytest.raises(RuntimeError, match="memory.alpha"):
		GaussianProgressGatedProvider(**kwargs)
	payload.pop("encoder_config")
	torch.save(payload, kwargs["checkpoint_path"])
	with pytest.raises(ValueError, match="encoder_config"):
		GaussianProgressGatedProvider(**kwargs)


def test_checkpoint_dimension_must_match_gaussian(tmp_path, cpu_memory):
	encoder = TCCEncoder(pretrained=False, embedding_dim=3)
	kwargs = _checkpoint_assets(tmp_path, encoder)
	with pytest.raises(ValueError, match="embedding dimension"):
		GaussianProgressGatedProvider(**kwargs)


@pytest.mark.parametrize("fail", [False, True])
def test_memory_debug_trace_preserves_live_state(tmp_path, cpu_memory, monkeypatch, fail):
	kwargs = _checkpoint_assets(tmp_path, _small_encoder())
	kwargs.update(enable_ood_filter=True, ood_filter_max_gap=1, ood_filter_min_ood_run=2)
	provider = GaussianProgressGatedProvider(**kwargs)
	control = GaussianProgressGatedProvider(**kwargs)
	for p in (provider, control):
		with pytest.raises(RuntimeError, match="reset"):
			p.advance(_obs(1))
		p.reset(_obs(20))
		p.advance(_obs(80))
	live_state = provider._stream_state
	before = {k: v.clone() for k, v in live_state.items()}
	progress_before = provider.progress_current
	trace_before = list(provider._episode_progress_raw)
	frames = np.stack([_obs(value)["agentview_image"] for value in (30, 100, 200)])
	with monkeypatch.context() as patch:
		if fail:
			def broken_inference():
				raise RuntimeError("debug inference failed")
			patch.setattr(provider, "_infer_progress_mean_and_ood", broken_inference)
			with pytest.raises(RuntimeError, match="debug inference failed"):
				provider.infer_progress_trace_from_frames(frames)
		else:
			assert provider.infer_progress_trace_from_frames(frames).shape == (3,)
	assert provider._stream_state is live_state
	provider.encoder.get_stream_embedding(live_state)
	provider.finalize_episode()
	provider._infer_progress_mean_and_ood()
	for k, value in before.items():
		torch.testing.assert_close(live_state[k], value)
	assert provider.progress_current == progress_before
	assert provider._episode_progress_raw == trace_before
	assert provider.advance(_obs(120)) == pytest.approx(control.advance(_obs(120)))


def test_memory_batch_matches_independent_episodes(tmp_path, cpu_memory):
	kwargs = _checkpoint_assets(tmp_path, _small_encoder())
	kwargs.update(enable_ood_filter=True, ood_filter_max_gap=1, ood_filter_min_ood_run=2)
	batch = BatchedGaussianProgressGatedProvider(2, **kwargs)
	singles = [GaussianProgressGatedProvider(**kwargs) for _ in range(2)]
	assert batch._frame_ring is None
	with pytest.raises(RuntimeError, match="reset"):
		batch.advance_all([np.zeros((224, 224, 3), dtype=np.uint8)] * 2)
	counts = np.zeros(2, dtype=int)
	for step, resets in enumerate(([True, True], [False, False], [True, False], [False, False])):
		frames = [np.full((224, 224, 3), 25 + 30 * step + 10 * i, dtype=np.uint8) for i in range(2)]
		actual = batch.reset_all(frames) if step == 0 else batch.advance_all(frames, resets)
		expected = [
			p.reset({p.image_key: frame}) if reset else p.advance({p.image_key: frame})
			for p, frame, reset in zip(singles, frames, resets)
		]
		np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
		counts = np.where(resets, 1, counts + 1)
		assert batch._stream_state["steps"].tolist() == counts.tolist()
		for i, p in enumerate(singles):
			for key, value in p._stream_state.items():
				torch.testing.assert_close(batch._stream_state[key][i:i + 1], value)
			np.testing.assert_allclose(batch.finalize_episode(i), p.finalize_episode(), atol=1e-6)
	before = {k: v.clone() for k, v in batch._stream_state.items()}
	with torch.inference_mode():
		batch._infer_batch()
	for k, value in before.items():
		torch.testing.assert_close(batch._stream_state[k], value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Official Mamba requires CUDA")
def test_official_cuda_provider_matches_sequence(tmp_path, tiny_local):
	pytest.importorskip("mamba_ssm")
	encoder = TCCEncoder(
		pretrained=False, context_stride=2, embedding_dim=128,
		memory_config={"enabled": True, "alpha_init": 0.5},
	).cuda().eval()
	kwargs = _checkpoint_assets(tmp_path, encoder)
	kwargs["device"] = "cuda"
	# A saved 128 -> 2 PCA keeps Gaussian arithmetic well-conditioned in this smoke test.
	with h5py.File(kwargs["gaussian_model_h5_path"], "a") as f:
		f.attrs["enable_pca"] = True
		f.attrs["input_embedding_dim"] = 128
		f["model"].create_dataset("pca_mean", data=np.zeros(128))
		f["model"].create_dataset("pca_components", data=np.eye(128)[:2])
	with h5py.File(kwargs["calibration_h5_path"], "a") as f:
		del f["videos/000001/embeddings"]
		calibration = np.zeros((2, 128))
		calibration[1, 0] = 4.0
		f["videos/000001"].create_dataset("embeddings", data=calibration)
	batch = BatchedGaussianProgressGatedProvider(2, **kwargs)
	singles = [GaussianProgressGatedProvider(**kwargs) for _ in range(2)]
	frames = np.random.default_rng(7).integers(0, 256, (6, 2, 224, 224, 3), dtype=np.uint8)
	frames_t = torch.from_numpy(frames).permute(1, 0, 4, 2, 3).cuda()
	with torch.inference_mode():
		raw = [encoder.encode_local_episode(video) for video in frames_t]
		dense = encoder.forward_sequence(raw, [6, 6]).cpu().numpy()
	for t, images in enumerate(frames):
		progress = batch.reset_all(images) if t == 0 else batch.advance_all(images)
		z = batch.encoder.get_stream_embedding(batch._stream_state).cpu().numpy()
		np.testing.assert_allclose(z, dense[:, t], rtol=2e-3, atol=2e-4)
		for i, p in enumerate(singles):
			p.reset({p.image_key: images[i]}) if t == 0 else p.advance({p.image_key: images[i]})
			np.testing.assert_allclose(p.encoder.get_stream_embedding(p._stream_state).cpu().numpy()[0], dense[i, t], rtol=2e-3, atol=2e-4)
			assert p.progress_current == pytest.approx(progress[i], abs=1e-4)
	for i in range(2):
		expected = _infer_one_trajectory(
			query_embeddings=dense[i, :, :2].astype(np.float64), model=batch.gaussian_model,
			posterior_temperature=1.0, entropy_epsilon=1e-12,
			calibration_distance_bins=batch.calibration_distance_bins, ood_p_value_threshold=0.1,
		)
		assert batch.progress_current[i] == pytest.approx(expected["progress_gated"][-1], abs=1e-4)
