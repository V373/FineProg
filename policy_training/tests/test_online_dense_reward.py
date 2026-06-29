from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.robomimic.online_replay_buffer import RobomimicOnlineDataCollector, RobomimicOnlineReplayBuffer
from reward_model.tcc_expert_proj_reward import (
    TCCExpertProjectionDenseRewardProvider,
    project_embedding_to_progress,
    project_embedding_to_progress_torch,
)
from train_policy import _validate_online_sac_config
from utils.logger import derive_run_metadata


class _FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_frames = None

    def forward(self, frames):
        self.last_frames = frames.detach().cpu()
        batch, clip_len = frames.shape[:2]
        embeddings = frames.new_zeros((batch, clip_len, 2))
        embeddings[:, :, 0] = frames.mean(dim=(2, 3, 4, 5))
        return embeddings


def _make_provider(**kwargs):
    return TCCExpertProjectionDenseRewardProvider(
        checkpoint_path=None,
        expert_path_h5=None,
        device="cpu",
        encoder=kwargs.pop("encoder", _FakeEncoder()),
        expert_embeddings=kwargs.pop("expert_embeddings", np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)),
        load_model=False,
        **kwargs,
    )


def _obs(value: float, size: int = 2):
    image = np.full((size, size, 3), float(value), dtype=np.float32)
    return {"agentview_image": image, "state": np.asarray([value], dtype=np.float32)}


def test_project_embedding_to_progress_soft_nn():
    expert = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    progress = project_embedding_to_progress(np.asarray([2.0], dtype=np.float32), expert, temperature=0.01)
    assert progress == pytest.approx(1.0, abs=1e-4)


def test_project_embedding_to_progress_torch_matches_numpy():
    expert = np.asarray([[0.0, 0.0], [1.0, 0.1], [2.0, 0.0]], dtype=np.float32)
    queries = np.asarray([[0.2, 0.0], [1.8, 0.05]], dtype=np.float32)
    expected = np.asarray([
        project_embedding_to_progress(q, expert, temperature=0.25) for q in queries
    ], dtype=np.float32)

    actual = project_embedding_to_progress_torch(
        torch.as_tensor(queries),
        torch.as_tensor(expert),
        temperature=0.25,
    ).detach().cpu().numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_rolling_clip_shape_and_causal_indices():
    encoder = _FakeEncoder()
    provider = _make_provider(
        encoder=encoder,
        clip_len=20,
        context_size=2,
        context_stride=15,
        image_height=2,
        image_width=2,
    )
    for value in range(provider.history_len):
        provider.frame_history.append(np.full((2, 2, 3), float(value), dtype=np.float32))

    clip = provider.build_rolling_clip().cpu().numpy()

    assert clip.shape == (1, 20, 2, 3, 2, 2)
    assert clip[0, 0, 0].mean() == pytest.approx(0.0)
    assert clip[0, 0, 1].mean() == pytest.approx(15.0)
    assert clip[0, -1, 0].mean() == pytest.approx(19.0)
    assert clip[0, -1, 1].mean() == pytest.approx(34.0)


def test_dense_reward_uses_cached_current_progress():
    provider = _make_provider(clip_len=2, context_size=1, context_stride=1, image_height=2, image_width=2, sparse_scale=2.0)
    provider.reset(_obs(1.0))

    result = provider.compute_dense_reward(0.5)

    assert result.progress_current == pytest.approx(1.0, abs=1e-4)
    assert result.reward == pytest.approx(2.0, abs=1e-4)


def test_pbrs_reward_formula_non_terminal():
    provider = _make_provider(clip_len=2, context_size=1, context_stride=1, image_height=2, image_width=2, sparse_scale=2.0)
    provider.reset(_obs(0.2))

    result = provider.compute_pbrs_reward(
        sparse_reward=0.5,
        pbrs_gamma=0.99,
        progress_next=0.8,
        progress_current=0.2,
    )

    expected = 2.0 * 0.5 + 0.99 * 0.8 - 0.2
    assert result.reward == pytest.approx(expected, abs=1e-6)
    assert result.progress_current == pytest.approx(0.2, abs=1e-6)
    assert result.progress_next == pytest.approx(0.8, abs=1e-6)
    assert result.shaping_reward == pytest.approx(0.99 * 0.8 - 0.2, abs=1e-6)


def test_progress_trace_from_frames_tracks_latest_video_frame_progress():
    provider = _make_provider(clip_len=2, context_size=1, context_stride=1, image_height=2, image_width=2)
    frames = np.stack([
        np.full((2, 2, 3), 0.0, dtype=np.float32),
        np.full((2, 2, 3), 0.5, dtype=np.float32),
        np.full((2, 2, 3), 1.0, dtype=np.float32),
    ], axis=0)

    progress = provider.infer_progress_trace_from_frames(frames)

    assert progress.shape == (3,)
    assert progress[0] == pytest.approx(0.0, abs=1e-4)
    assert progress[1] == pytest.approx(0.5, abs=1e-4)
    assert progress[2] == pytest.approx(1.0, abs=1e-4)


class _FakeObsAdapter:
    def flatten(self, obs):
        return np.asarray(obs["state"], dtype=np.float32).reshape(-1)


class _FakeActionDist:
    def actions_from_params(self, mean_actions, log_std, deterministic=False, **kwargs):
        del log_std, deterministic, kwargs
        return mean_actions


class _FakeActor:
    def __init__(self):
        self.action_dist = _FakeActionDist()

    def get_action_dist_params(self, obs_tensor):
        return obs_tensor.new_zeros((obs_tensor.shape[0], 1)), obs_tensor.new_zeros((obs_tensor.shape[0], 1)), {}


class _FakeAlgo:
    def __init__(self):
        self.device = torch.device("cpu")
        self.actor = _FakeActor()


class _FakeEnv:
    def __init__(self):
        self.step_idx = 0

    def reset(self):
        self.step_idx = 0
        return _obs(0.0)

    def step(self, action):
        del action
        self.step_idx += 1
        return _obs(float(self.step_idx)), 0.0, False, {}


class _FakeProvider:
    def __init__(self):
        self.progress_current = 0.25
        self.reset_calls = 0
        self.advance_calls = 0

    def reset(self, obs):
        del obs
        self.reset_calls += 1
        self.progress_current = 0.25
        return self.progress_current

    def compute_dense_reward(self, sparse_reward):
        from reward_model.tcc_expert_proj_reward import DenseRewardResult

        return DenseRewardResult(reward=float(sparse_reward) + self.progress_current, progress_current=self.progress_current, sparse_reward=float(sparse_reward))

    def compute_pbrs_reward(self, sparse_reward, *, pbrs_gamma, progress_next, progress_current=None):
        from reward_model.tcc_expert_proj_reward import DenseRewardResult

        if progress_current is None:
            progress_current = self.progress_current
        shaping = float(pbrs_gamma) * float(progress_next) - float(progress_current)
        reward = float(sparse_reward) + shaping
        return DenseRewardResult(
            reward=reward,
            progress_current=float(progress_current),
            progress_next=float(progress_next),
            shaping_reward=float(shaping),
            sparse_reward=float(sparse_reward),
        )

    def advance(self, obs):
        del obs
        self.advance_calls += 1
        self.progress_current += 0.25
        return self.progress_current


def _make_collector(reward_type: str, provider=None):
    observation_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    replay_buffer = RobomimicOnlineReplayBuffer(
        buffer_size=10,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
    )
    collector = RobomimicOnlineDataCollector(
        env=_FakeEnv(),
        algo=_FakeAlgo(),
        obs_adapter=_FakeObsAdapter(),
        replay_buffer=replay_buffer,
        reward_type=reward_type,
        reward_provider=provider,
        horizon=10,
    )
    return collector, replay_buffer


def test_collector_dense_stores_provider_reward():
    provider = _FakeProvider()
    collector, replay_buffer = _make_collector("dense", provider=provider)
    collector.reset()

    metrics = collector.collect_step(learning_starts=99)

    assert replay_buffer.rewards[0, 0] == pytest.approx(0.25)
    assert metrics["online/reward_selected"] == pytest.approx(0.25)
    assert metrics["online/progress_current"] == pytest.approx(0.25)
    assert provider.advance_calls == 1


def test_collector_sparse_with_provider_records_trace_but_stores_sparse():
    provider = _FakeProvider()
    collector, replay_buffer = _make_collector("sparse_done", provider=provider)
    collector.reset()

    collector.collect_step(learning_starts=99)

    assert replay_buffer.rewards[0, 0] == pytest.approx(0.0)
    assert collector.dense_reward_trace[0]["dense_reward"] == pytest.approx(0.25)


def test_collector_pbrs_stores_pbrs_reward_and_logs_progress_next():
    provider = _FakeProvider()
    collector, replay_buffer = _make_collector("pbrs", provider=provider)
    collector.pbrs_gamma = 0.99
    collector.reset()

    metrics = collector.collect_step(learning_starts=99)

    expected = 0.99 * 0.5 - 0.25
    assert replay_buffer.rewards[0, 0] == pytest.approx(expected)
    assert metrics["online/reward_pbrs_predicted"] == pytest.approx(expected)
    assert metrics["online/progress_current"] == pytest.approx(0.25)
    assert metrics["online/progress_next"] == pytest.approx(0.5)
    assert provider.advance_calls == 1


def test_online_sac_validation_accepts_pbrs():
    cfg = SimpleNamespace(
        action_chunk_size=1,
        offline_warmstart=SimpleNamespace(enabled=False),
        mixed_buffer=SimpleNamespace(enabled=False),
        reward=SimpleNamespace(type="pbrs"),
        reward_model=SimpleNamespace(enabled=True, checkpoint_path="x.pt", expert_path_h5="x.h5", kind="tcc_expert_projection"),
    )
    assert _validate_online_sac_config(cfg) == 1


def test_online_sac_metadata_uses_config_reward_type():
    cfg = SimpleNamespace(
        algo_name="online_sac",
        seed=7,
        dataset=SimpleNamespace(
            h5_path="/tmp/datasets/robomimic/can/mh/reward_labeled/foo_reward_labeled_PBRS_resnet18feats.hdf5",
            filter_key="IQL_expert_worse",
        ),
        online=SimpleNamespace(reward=SimpleNamespace(type="dense")),
    )

    meta = derive_run_metadata(cfg)

    assert meta["reward_type"] == "dense"
