from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algos.offline_rl.base_offline_rl import OfflineRLBase
from algos.online_rl.base_online_rl import OnlineRLBase
from utils.config import PolicyTrainingConfig
from utils.eval_utils import TrainingRolloutEvaluator


class _FakeActionDist:
    def actions_from_params(self, mean_actions, log_std, deterministic=True):
        del log_std, deterministic
        return mean_actions


class _FakeActor:
    def __init__(self, action_dim: int):
        self._action_dim = int(action_dim)
        self.action_dist = _FakeActionDist()
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def get_action_dist_params(self, obs_tensor):
        batch = int(obs_tensor.shape[0])
        mean = obs_tensor.new_zeros((batch, self._action_dim))
        log_std = obs_tensor.new_zeros((batch, self._action_dim))
        return mean, log_std, {}


class _FakePolicy:
    def __init__(self):
        self.training = True

    def set_training_mode(self, mode: bool):
        self.training = bool(mode)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _FakeAlgoForEval:
    def __init__(self, action_dim: int):
        import torch

        self.device = torch.device("cpu")
        self.policy = _FakePolicy()
        self.actor = _FakeActor(action_dim=action_dim)


class _FakeRobomimicEnv:
    rollout_exceptions = (RuntimeError,)

    def __init__(self, episodes):
        self._episodes = episodes
        self._episode_idx = -1
        self._step_idx = 0
        self._success = False

    def _obs(self):
        return {
            "state": np.asarray([self._episode_idx, self._step_idx], dtype=np.float32),
        }

    def reset(self):
        self._episode_idx = (self._episode_idx + 1) % len(self._episodes)
        self._step_idx = 0
        self._success = False
        return self._obs()

    def get_state(self):
        return {
            "episode_idx": int(self._episode_idx),
            "step_idx": int(self._step_idx),
            "success": bool(self._success),
        }

    def reset_to(self, state_dict):
        self._episode_idx = int(state_dict["episode_idx"])
        self._step_idx = int(state_dict["step_idx"])
        self._success = bool(state_dict["success"])
        return self._obs()

    def step(self, action):
        del action
        ep = self._episodes[self._episode_idx]
        reward = float(ep["rewards"][self._step_idx])
        success_step = ep.get("success_step")
        if success_step is not None and self._step_idx >= int(success_step):
            self._success = True
        done = self._step_idx >= (len(ep["rewards"]) - 1)
        self._step_idx += 1
        return self._obs(), reward, done, {}

    def is_success(self):
        return {"task": bool(self._success)}

    def render(self, mode="rgb_array", height=64, width=64, camera_name="agentview"):
        del mode, camera_name
        return np.zeros((height, width, 3), dtype=np.uint8)


class _SimpleBatch:
    observations = None


class _SimpleReplayBuffer:
    def sample(self, batch_size):
        del batch_size
        return _SimpleBatch()


class _TestAlgo(OfflineRLBase):
    def _setup_model(self) -> None:
        self._saved_tags = []

    def train_step(self, batch):
        del batch
        return {"train/loss": 1.0}

    def _module_state_dict(self):
        return {}

    def _optimizer_state_dict(self):
        return {}

    def _load_module_state_dict(self, modules):
        del modules

    def _load_optimizer_state_dict(self, optimizers):
        del optimizers

    def save(self, save_dir: str, tag: str) -> str:
        del save_dir
        self._saved_tags.append(str(tag))
        return str(tag)


class _EvalRecorder:
    def __init__(self, every_n_steps: int, warmstart_steps: int):
        self.eval_cfg = {
            "enabled": True,
            "every_n_steps": int(every_n_steps),
            "warmstart_steps": int(warmstart_steps),
        }
        self.called_steps = []
        self.last_video_paths = []

    def run(self, global_step: int):
        self.called_steps.append(int(global_step))
        return {
            "eval/return": 0.0,
            "eval/horizon": 0.0,
            "eval/success_rate": 0.0,
            "eval/num_success": 0.0,
            "eval/num_rollouts": 0.0,
            "eval/time_minutes": 0.0,
        }


class _NoopLogger:
    def __init__(self):
        self.logged = []
        self.records = []
        self.videos = []
        self.images = []
        self.dumps = []

    def record(self, key: str, value):
        self.records.append((str(key), value))

    def record_dict(self, metrics):
        self.logged.append(dict(metrics))

    def dump(self, step: int):
        self.dumps.append(int(step))

    def record_video(self, key: str, path: str, step: int, fps: int = 20):
        self.videos.append((str(key), str(path), int(step), int(fps)))

    def record_image(self, key: str, path: str, step: int):
        self.images.append((str(key), str(path), int(step)))


class _FakeTrainableModule:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _FakeOnlinePolicy(_FakeTrainableModule):
    def set_training_mode(self, mode: bool):
        self.training = bool(mode)


class _OnlineTestAlgo(OnlineRLBase):
    def _setup_model(self) -> None:
        self._saved_tags = []
        self.train_batches = 0
        self.policy = _FakeOnlinePolicy()
        self.actor = _FakeTrainableModule()
        self.critic = _FakeTrainableModule()
        self.critic_target = _FakeTrainableModule()

    def train_step(self, batch):
        del batch
        self.train_batches += 1
        return {"train/loss": float(self.train_batches)}

    def _module_state_dict(self):
        return {}

    def _optimizer_state_dict(self):
        return {}

    def _load_module_state_dict(self, modules):
        del modules

    def _load_optimizer_state_dict(self, optimizers):
        del optimizers

    def save(self, save_dir: str, tag: str) -> str:
        del save_dir
        self._saved_tags.append(str(tag))
        return str(tag)


class _OnlineReplayBuffer:
    def __init__(self, size: int = 100):
        self._size = int(size)
        self.sample_calls = 0

    def can_sample(self, batch_size: int) -> bool:
        del batch_size
        return self._size > 0

    def size(self) -> int:
        return self._size

    def sample(self, batch_size: int):
        del batch_size
        self.sample_calls += 1
        return _SimpleBatch()


class _OnlineCollector:
    def __init__(self):
        self.reset_calls = 0
        self.collect_calls = 0

    def reset(self):
        self.reset_calls += 1

    def collect_step(self, learning_starts: int = 0):
        del learning_starts
        self.collect_calls += 1
        return {"online/reward_sparse": float(self.collect_calls)}


def _make_online_test_algo():
    import gymnasium as gym
    import torch

    return _OnlineTestAlgo(
        observation_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        cfg=SimpleNamespace(),
        device=torch.device("cpu"),
    )


def test_policy_training_config_has_eval_defaults(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("seed: 1\n", encoding="utf-8")

    cfg = PolicyTrainingConfig.load(str(cfg_path))

    assert cfg.eval.enabled is False
    assert cfg.eval.every_n_steps == 5000
    assert cfg.eval.warmstart_steps == 0
    assert cfg.eval.n_rollouts == 20
    assert cfg.eval.horizon == 400
    assert cfg.eval.stochastic is False
    assert cfg.eval.terminate_on_success is True
    assert cfg.eval.video.enabled is False
    assert cfg.eval.video.max_episodes == 1
    assert cfg.eval.output.json_dir is None


def test_training_rollout_evaluator_run_aggregates_metrics(monkeypatch, tmp_path):
    import utils.eval_utils as eval_utils

    episodes = [
        {"rewards": [1.0, 1.0], "success_step": 1},
        {"rewards": [0.0, 0.0, 1.0], "success_step": None},
        {"rewards": [2.0], "success_step": 0},
    ]
    fake_env = _FakeRobomimicEnv(episodes=episodes)
    monkeypatch.setattr(eval_utils, "create_robomimic_env", lambda **kwargs: fake_env)

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(obs_keys=["state"]),
        iql=SimpleNamespace(features_extractor_type="flat_range"),
    )
    eval_cfg = SimpleNamespace(
        enabled=True,
        every_n_steps=10,
        warmstart_steps=0,
        n_rollouts=3,
        horizon=5,
        stochastic=False,
        terminate_on_success=True,
        env_name_override=None,
        video=SimpleNamespace(
            enabled=False,
            max_episodes=1,
            dir=None,
            skip=5,
            fps=20,
            frame_height=64,
            frame_width=64,
            camera_names=["agentview"],
        ),
        output=SimpleNamespace(json_dir=None),
    )

    evaluator = TrainingRolloutEvaluator(
        algo=_FakeAlgoForEval(action_dim=2),
        cfg=cfg,
        env_metadata={"type": 1},
        shape_metadata={
            "observation_dim": 2,
            "action_dim": 2,
            "visual_obs_keys": [],
        },
        obs_slices={"state": [0, 2]},
        obs_normalization_stats=None,
        action_normalization_stats=None,
        eval_cfg=eval_cfg,
        save_dir=str(tmp_path),
    )

    metrics = evaluator.run(global_step=30)

    assert metrics["eval/num_rollouts"] == 3.0
    assert metrics["eval/num_success"] == 2.0
    assert metrics["eval/horizon"] == 2.0
    assert np.isclose(metrics["eval/success_rate"], 2.0 / 3.0)
    assert np.isclose(metrics["eval/return"], (2.0 + 1.0 + 2.0) / 3.0)
    assert metrics["eval/time_minutes"] >= 0.0


def test_learn_offline_eval_trigger_schedule_and_saves(tmp_path):
    import gymnasium as gym
    import torch

    algo = _TestAlgo(
        observation_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        cfg=SimpleNamespace(),
        device=torch.device("cpu"),
    )

    evaluator = _EvalRecorder(every_n_steps=3, warmstart_steps=2)
    logger = _NoopLogger()

    algo.learn_offline(
        replay_buffer=_SimpleReplayBuffer(),
        n_steps=7,
        batch_size=4,
        log_every=100,
        save_every=2,
        save_dir=str(tmp_path),
        logger=logger,
        rollout_evaluator=evaluator,
    )

    assert evaluator.called_steps == [3, 6]
    assert algo._saved_tags == ["step_2", "step_4", "step_6", "final"]


def test_learn_online_eval_uses_env_step_and_logs_video(tmp_path):
    algo = _make_online_test_algo()
    collector = _OnlineCollector()
    replay_buffer = _OnlineReplayBuffer(size=100)
    evaluator = _EvalRecorder(every_n_steps=3, warmstart_steps=2)
    evaluator.last_video_paths = [str(tmp_path / "eval.mp4")]
    logger = _NoopLogger()

    algo.learn_online(
        collector=collector,
        replay_buffer=replay_buffer,
        n_steps=7,
        batch_size=4,
        learning_starts=0,
        train_freq=2,
        gradient_steps=1,
        log_every=100,
        save_every=3,
        save_dir=str(tmp_path),
        logger=logger,
        rollout_evaluator=evaluator,
    )

    assert collector.reset_calls == 1
    assert collector.collect_calls == 7
    assert replay_buffer.sample_calls == 4
    assert algo.global_step == 4
    assert evaluator.called_steps == [3, 6]
    assert logger.dumps == [3, 6]
    assert logger.videos == [
        ("eval/video", str(tmp_path / "eval.mp4"), 3, 20),
        ("eval/video", str(tmp_path / "eval.mp4"), 6, 20),
    ]
    assert algo._saved_tags == ["step_3", "step_6", "final"]
    assert algo.policy.training is True
    assert algo.actor.training is True
    assert algo.critic.training is True
    assert algo.critic_target.training is True


def test_learn_online_logs_progress_image_when_eval_video_is_available(tmp_path):
    algo = _make_online_test_algo()
    collector = _OnlineCollector()
    replay_buffer = _OnlineReplayBuffer(size=100)
    evaluator = _EvalRecorder(every_n_steps=1, warmstart_steps=0)
    evaluator.last_video_paths = [str(tmp_path / "eval.mp4")]
    evaluator.last_progress_path = str(tmp_path / "progress.png")
    logger = _NoopLogger()

    algo.learn_online(
        collector=collector,
        replay_buffer=replay_buffer,
        n_steps=1,
        batch_size=4,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
        log_every=100,
        save_every=10,
        save_dir=str(tmp_path),
        logger=logger,
        rollout_evaluator=evaluator,
    )

    assert logger.videos == [("eval/video", str(tmp_path / "eval.mp4"), 1, 20)]
    assert logger.images == [("eval/progress", str(tmp_path / "progress.png"), 1)]


# ---------------------------------------------------------------------------
# New tests: num_workers config defaults and parallel dispatch
# ---------------------------------------------------------------------------

class _FakeAlgoWithModules(_FakeAlgoForEval):
    """_FakeAlgoForEval extended with _module_state_dict for parallel worker packaging."""

    def _module_state_dict(self):
        return {}

    def _load_module_state_dict(self, modules):
        pass  # noqa: WPS428


def _make_eval_cfg_ns(num_workers: int = 1, video_enabled: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        every_n_steps=10,
        warmstart_steps=0,
        n_rollouts=3,
        horizon=5,
        stochastic=False,
        terminate_on_success=True,
        env_name_override=None,
        num_workers=num_workers,
        worker_device="cpu",
        video=SimpleNamespace(
            enabled=video_enabled,
            max_episodes=2,
            dir=None,
            skip=5,
            fps=20,
            frame_height=64,
            frame_width=64,
            camera_names=["agentview"],
        ),
        output=SimpleNamespace(json_dir=None),
    )


def _make_evaluator(monkeypatch, eval_utils, eval_cfg_ns, tmp_path, fake_env=None):
    if fake_env is None:
        episodes = [
            {"rewards": [1.0, 1.0], "success_step": 1},
            {"rewards": [0.0, 0.0, 0.0], "success_step": None},
            {"rewards": [2.0], "success_step": 0},
        ]
        fake_env = _FakeRobomimicEnv(episodes=episodes)
    monkeypatch.setattr(eval_utils, "create_robomimic_env", lambda **kwargs: fake_env)

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(obs_keys=["state"]),
        iql=SimpleNamespace(features_extractor_type="flat_range"),
    )
    return TrainingRolloutEvaluator(
        algo=_FakeAlgoWithModules(action_dim=2),
        cfg=cfg,
        env_metadata={"type": 1},
        shape_metadata={"observation_dim": 2, "action_dim": 2, "visual_obs_keys": []},
        obs_slices={"state": [0, 2]},
        obs_normalization_stats=None,
        action_normalization_stats=None,
        eval_cfg=eval_cfg_ns,
        save_dir=str(tmp_path),
    )


def test_eval_config_has_num_workers_and_worker_device_defaults(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("seed: 1\n", encoding="utf-8")
    cfg = PolicyTrainingConfig.load(str(cfg_path))
    assert cfg.eval.num_workers == 1
    assert cfg.eval.worker_device == "cpu"


def test_parallel_evaluator_dispatches_run_parallel_when_num_workers_gt_1(monkeypatch, tmp_path):
    import utils.eval_utils as eval_utils

    eval_cfg_ns = _make_eval_cfg_ns(num_workers=2)
    evaluator = _make_evaluator(monkeypatch, eval_utils, eval_cfg_ns, tmp_path)

    dispatched = []

    def fake_run_parallel(self, global_step, n_rollouts, horizon, terminate_on_success,
                          video_enabled, max_video_episodes, video_cfg):
        dispatched.append(global_step)
        # Return stats that match n_rollouts=3
        return [
            {"Return": 1.0, "Horizon": 2.0, "Success_Rate": 1.0},
            {"Return": 0.0, "Horizon": 3.0, "Success_Rate": 0.0},
            {"Return": 2.0, "Horizon": 1.0, "Success_Rate": 1.0},
        ], []

    monkeypatch.setattr(eval_utils.TrainingRolloutEvaluator, "_run_parallel", fake_run_parallel)

    metrics = evaluator.run(global_step=50)

    assert dispatched == [50], "Expected _run_parallel to be called exactly once"
    assert metrics["eval/num_rollouts"] == 3.0


def test_parallel_aggregates_return_horizon_success(monkeypatch, tmp_path):
    import utils.eval_utils as eval_utils

    eval_cfg_ns = _make_eval_cfg_ns(num_workers=2)
    evaluator = _make_evaluator(monkeypatch, eval_utils, eval_cfg_ns, tmp_path)

    known_stats = [
        {"Return": 3.0, "Horizon": 4.0, "Success_Rate": 1.0},
        {"Return": 1.0, "Horizon": 2.0, "Success_Rate": 0.0},
        {"Return": 2.0, "Horizon": 3.0, "Success_Rate": 1.0},
    ]

    def fake_run_parallel(self, global_step, n_rollouts, horizon, terminate_on_success,
                          video_enabled, max_video_episodes, video_cfg):
        return known_stats, []

    monkeypatch.setattr(eval_utils.TrainingRolloutEvaluator, "_run_parallel", fake_run_parallel)

    metrics = evaluator.run(global_step=10)

    assert metrics["eval/num_rollouts"] == 3.0
    assert metrics["eval/num_success"] == 2.0
    assert np.isclose(metrics["eval/return"], (3.0 + 1.0 + 2.0) / 3.0)
    assert np.isclose(metrics["eval/horizon"], (4.0 + 2.0 + 3.0) / 3.0)
    assert np.isclose(metrics["eval/success_rate"], 2.0 / 3.0)


def test_parallel_video_paths_stored_in_last_video_paths(monkeypatch, tmp_path):
    """Video paths returned from workers are captured in last_video_paths."""
    import utils.eval_utils as eval_utils

    eval_cfg_ns = _make_eval_cfg_ns(num_workers=2, video_enabled=True)
    evaluator = _make_evaluator(monkeypatch, eval_utils, eval_cfg_ns, tmp_path)

    fake_video_paths = ["/tmp/v0.mp4", "/tmp/v1.mp4"]

    def fake_run_parallel(self, global_step, n_rollouts, horizon, terminate_on_success,
                          video_enabled, max_video_episodes, video_cfg):
        stats = [{"Return": 1.0, "Horizon": 1.0, "Success_Rate": 1.0}] * 3
        return stats, list(fake_video_paths)

    monkeypatch.setattr(eval_utils.TrainingRolloutEvaluator, "_run_parallel", fake_run_parallel)

    evaluator.run(global_step=20)

    assert evaluator.last_video_paths == fake_video_paths


def test_serial_path_used_when_num_workers_equals_1(monkeypatch, tmp_path):
    """num_workers=1 must call _run_serial, not _run_parallel."""
    import utils.eval_utils as eval_utils

    eval_cfg_ns = _make_eval_cfg_ns(num_workers=1)
    evaluator = _make_evaluator(monkeypatch, eval_utils, eval_cfg_ns, tmp_path)

    parallel_called = []

    def fake_run_parallel(self, *args, **kwargs):
        parallel_called.append(True)
        return [], []

    monkeypatch.setattr(eval_utils.TrainingRolloutEvaluator, "_run_parallel", fake_run_parallel)

    metrics = evaluator.run(global_step=5)

    assert not parallel_called, "_run_parallel must not be called when num_workers=1"
    # Serial path produces real results from the fake env
    assert metrics["eval/num_rollouts"] == 3.0


def test_serial_video_path_appears_in_last_video_paths_when_file_exists(monkeypatch, tmp_path):
    """When video.enabled=True (serial), videos that exist on disk are recorded."""
    import utils.eval_utils as eval_utils

    # Patch video writer to create a real (empty) file so os.path.isfile passes.
    def fake_make_video_writer(video_path, video_cfg):
        if video_path:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            Path(video_path).touch()
            return video_path, _FakeVideoWriter()
        return None, None

    class _FakeVideoWriter:
        def send(self, frame):
            pass
        def close(self):
            pass

    monkeypatch.setattr(eval_utils, "_make_video_writer", fake_make_video_writer)

    eval_cfg_ns = _make_eval_cfg_ns(num_workers=1, video_enabled=True)
    evaluator = _make_evaluator(monkeypatch, eval_utils, eval_cfg_ns, tmp_path)

    evaluator.run(global_step=100)

    # max_episodes=2, so at most 2 videos should be recorded
    assert len(evaluator.last_video_paths) <= 2
    assert all(p.endswith(".mp4") for p in evaluator.last_video_paths)

