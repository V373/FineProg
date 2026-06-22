#!/usr/bin/env python3
"""Unified entrypoint for offline policy training."""

from __future__ import annotations

import csv
import numpy as np
from pathlib import Path
from gymnasium import spaces
from types import SimpleNamespace

from algos import build_algo
from datasets.robomimic import (
	RobomimicOnlineDataCollector,
	RobomimicOnlineReplayBuffer,
	RobomimicReplayBuffer,
)
from envs.robomimic import create_online_robomimic_env
from reward_model import TCCExpertProjectionDenseRewardProvider
from utils.algo_config import get_algo_cfg
from utils.config import PolicyTrainingConfig
from utils.eval_utils import DatasetValueEvaluator, ObservationAdapter, TrainingRolloutEvaluator
from utils.logger import (
	WandBLogger,
	build_wandb_run_id,
	derive_run_metadata,
	namespace_to_dict,
	parse_policy_train_args,
	resolve_device,
	resolve_save_dir,
	seed_everything,
)
from utils.train_utils import build_checkpoint_metadata, extract_dataset_metadata


def _inject_runtime_obs_slices(cfg: SimpleNamespace, algo_name: str, replay_buffer: RobomimicReplayBuffer) -> None:
	algo_cfg = get_algo_cfg(cfg, algo_name=algo_name)
	if getattr(algo_cfg, "features_extractor_type", "flat_range") != "resnet18conv":
		return
	obs_slices_tuples = {k: (s.start, s.stop) for k, s in replay_buffer.obs_slices.items()}
	fe_kwargs = getattr(algo_cfg, "features_extractor_kwargs", SimpleNamespace())
	fe_kwargs.obs_slices = obs_slices_tuples
	algo_cfg.features_extractor_kwargs = fe_kwargs


def _run_iql_training(
	cfg: SimpleNamespace,
	args,
	device,
	logger: WandBLogger,
	save_dir: str,
	checkpoint_metadata: dict,
	replay_buffer: RobomimicReplayBuffer,
) -> None:
	_inject_runtime_obs_slices(cfg, algo_name="iql", replay_buffer=replay_buffer)
	algo = build_algo(
		algo_name="iql",
		observation_space=replay_buffer.observation_space,
		action_space=replay_buffer.action_space,
		cfg=cfg.iql,
		device=device,
	)
	rollout_evaluator = None
	if bool(getattr(getattr(cfg, "eval", None), "enabled", False)):
		rollout_evaluator = TrainingRolloutEvaluator(
			algo=algo,
			cfg=cfg,
			env_metadata=checkpoint_metadata["env_metadata"],
			shape_metadata=checkpoint_metadata["shape_metadata"],
			obs_slices=checkpoint_metadata["obs_slices"],
			obs_normalization_stats=replay_buffer.obs_normalization_stats,
			action_normalization_stats=replay_buffer.action_normalization_stats,
			eval_cfg=cfg.eval,
			save_dir=save_dir,
		)

	value_evaluator = None
	eval_ns = getattr(cfg, "eval", None)
	if bool(getattr(eval_ns, "enabled", False)):
		value_cfg_ns = getattr(eval_ns, "value", None)
		if value_cfg_ns is not None and bool(getattr(value_cfg_ns, "enabled", True)):
			value_evaluator = DatasetValueEvaluator(
				algo=algo,
				cfg=cfg,
				obs_normalization_stats=replay_buffer.obs_normalization_stats,
				action_normalization_stats=replay_buffer.action_normalization_stats,
				eval_cfg=cfg.eval,
				save_dir=save_dir,
			)

	if args.resume:
		algo.load(args.resume)
		print(f"[train_policy] resumed checkpoint: {args.resume}")

	print(
		"[train_policy] start offline training "
		f"algo=iql buffer_size={replay_buffer.size()} steps={cfg.train.n_steps} "
		f"batch_size={cfg.train.batch_size} device={device}"
	)

	algo.learn_offline(
		replay_buffer=replay_buffer,
		n_steps=int(cfg.train.n_steps),
		batch_size=int(cfg.train.batch_size),
		log_every=int(cfg.train.log_every),
		save_every=int(cfg.train.save_every),
		save_dir=save_dir,
		logger=logger,
		checkpoint_metadata=checkpoint_metadata,
		rollout_evaluator=rollout_evaluator,
		value_evaluator=value_evaluator,
	)


def _validate_online_sac_config(online_cfg: SimpleNamespace) -> int:
	action_chunk_size = int(getattr(online_cfg, "action_chunk_size", 1))
	if action_chunk_size != 1:
		raise NotImplementedError("online_sac action chunking is not implemented; set online.action_chunk_size=1.")
	if bool(getattr(getattr(online_cfg, "offline_warmstart", SimpleNamespace(enabled=False)), "enabled", False)):
		raise NotImplementedError("online_sac offline_warmstart is reserved for a later port.")
	if bool(getattr(getattr(online_cfg, "mixed_buffer", SimpleNamespace(enabled=False)), "enabled", False)):
		raise NotImplementedError("online_sac mixed offline/online replay is reserved for a later port.")
	reward_cfg = getattr(online_cfg, "reward", SimpleNamespace())
	reward_type = str(getattr(reward_cfg, "type", "sparse_done"))
	if reward_type not in {"sparse_done", "dense", "pbrs"}:
		raise ValueError("online_sac supports online.reward.type values: sparse_done, dense, pbrs.")
	reward_model_cfg = getattr(online_cfg, "reward_model", SimpleNamespace(enabled=False))
	reward_model_enabled = bool(getattr(reward_model_cfg, "enabled", False))
	if reward_type in {"dense", "pbrs"} and not reward_model_enabled:
		raise ValueError("online.reward.type=dense or pbrs requires online.reward_model.enabled=true.")
	if reward_type == "pbrs":
		_ = float(getattr(reward_cfg, "pbrs_gamma", 0.99))
	if reward_model_enabled:
		kind = str(getattr(reward_model_cfg, "kind", "tcc_expert_projection"))
		if kind != "tcc_expert_projection":
			raise ValueError("online.reward_model.kind must be 'tcc_expert_projection'.")
		if not getattr(reward_model_cfg, "checkpoint_path", None):
			raise ValueError("online.reward_model.checkpoint_path is required when reward_model.enabled=true.")
		if not getattr(reward_model_cfg, "expert_path_h5", None):
			raise ValueError("online.reward_model.expert_path_h5 is required when reward_model.enabled=true.")
	return action_chunk_size


def _resolve_policy_training_path(path_value: object, project_root: str) -> str | None:
	if path_value is None:
		return None
	path = Path(str(path_value))
	if not path.is_absolute():
		path = Path(project_root) / path
	return str(path.resolve())


def _build_online_reward_provider(cfg: SimpleNamespace, device) -> TCCExpertProjectionDenseRewardProvider | None:
	online_cfg = getattr(cfg, "online", SimpleNamespace())
	reward_model_cfg = getattr(online_cfg, "reward_model", SimpleNamespace(enabled=False))
	if not bool(getattr(reward_model_cfg, "enabled", False)):
		return None
	reward_cfg = getattr(online_cfg, "reward", SimpleNamespace())
	project_root = str(getattr(cfg, "project_root", Path(__file__).resolve().parent))
	checkpoint_path = _resolve_policy_training_path(getattr(reward_model_cfg, "checkpoint_path", None), project_root)
	expert_path_h5 = _resolve_policy_training_path(getattr(reward_model_cfg, "expert_path_h5", None), project_root)
	train_config_path = _resolve_policy_training_path(getattr(reward_model_cfg, "train_config_path", None), project_root)
	return TCCExpertProjectionDenseRewardProvider(
		checkpoint_path=checkpoint_path,
		expert_path_h5=expert_path_h5,
		device=device,
		expert_group=str(getattr(reward_model_cfg, "expert_group", "videos/mean")),
		projection_temperature=float(getattr(reward_model_cfg, "projection_temperature", 0.1)),
		clip_len=int(getattr(reward_model_cfg, "clip_len", 20)),
		context_size=int(getattr(reward_model_cfg, "context_size", 2)),
		context_stride=int(getattr(reward_model_cfg, "context_stride", 15)),
		pretrained=bool(getattr(reward_model_cfg, "pretrained", True)),
		train_config_path=train_config_path,
		image_key=str(getattr(reward_model_cfg, "image_key", "agentview_image")),
		image_height=int(getattr(reward_model_cfg, "image_height", 224)),
		image_width=int(getattr(reward_model_cfg, "image_width", 224)),
		sparse_scale=float(getattr(reward_cfg, "sparse_scale", 1.0)),
	)


def _write_online_dense_reward_trace(
	cfg: SimpleNamespace,
	meta: dict,
	collector: RobomimicOnlineDataCollector,
	online_buffer: RobomimicOnlineReplayBuffer,
) -> None:
	project_root = Path(str(getattr(cfg, "project_root", Path(__file__).resolve().parent)))
	out_dir = project_root / "outputs" / "online_reward_inference" / str(meta["run_tag"])
	out_dir.mkdir(parents=True, exist_ok=True)
	trace = list(getattr(collector, "dense_reward_trace", []))
	if trace:
		indices = np.asarray([row["buffer_index"] for row in trace], dtype=np.float32)
		reward_selected = np.asarray([row["selected_reward"] for row in trace], dtype=np.float32)
		dense_rewards = np.asarray([row["dense_reward"] for row in trace], dtype=np.float32)
		pbrs_rewards = np.asarray([row["pbrs_reward"] for row in trace], dtype=np.float32)
		progress = np.asarray([row["progress_current"] for row in trace], dtype=np.float32)
		progress_next = np.asarray([row["progress_next"] for row in trace], dtype=np.float32)
		shaping_reward = np.asarray([row["shaping_reward"] for row in trace], dtype=np.float32)
		csv_path = out_dir / "dense_reward_trace.csv"
		with csv_path.open("w", newline="", encoding="utf-8") as handle:
			writer = csv.DictWriter(
				handle,
				fieldnames=[
					"buffer_index",
					"selected_reward",
					"dense_reward",
					"pbrs_reward",
					"progress_current",
					"progress_next",
					"shaping_reward",
					"sparse_reward",
				],
			)
			writer.writeheader()
			writer.writerows(trace)
		npy_path = out_dir / "dense_reward_trace.npy"
		np.save(npy_path, np.stack([indices, reward_selected, dense_rewards, pbrs_rewards, progress, progress_next, shaping_reward], axis=1))
	else:
		size = online_buffer.size()
		if size <= 0:
			return
		indices = np.arange(size, dtype=np.float32)
		reward_selected = online_buffer.rewards[:size, 0].astype(np.float32)
		dense_rewards = reward_selected

	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(figsize=(8, 4))
	ax.plot(indices, reward_selected, linewidth=1.0)
	ax.set_xlabel("transition buffer index")
	ax.set_ylabel("selected reward")
	ax.set_title("Online reward trace")
	ax.grid(True, alpha=0.3)
	fig.tight_layout()
	plot_path = out_dir / "dense_reward_trace.png"
	fig.savefig(plot_path, dpi=150)
	plt.close(fig)
	print(f"[train_policy] dense reward trace saved: {plot_path}")


def _build_online_sac_spaces(checkpoint_metadata: dict) -> tuple[spaces.Box, spaces.Box]:
	obs_dim = int(checkpoint_metadata["shape_metadata"]["observation_dim"])
	action_dim = int(checkpoint_metadata["shape_metadata"]["action_dim"])
	observation_space = spaces.Box(
		low=-float("inf"),
		high=float("inf"),
		shape=(obs_dim,),
		dtype=np.float32,
	)
	action_space = spaces.Box(
		low=-1.0,
		high=1.0,
		shape=(action_dim,),
		dtype=np.float32,
	)
	return observation_space, action_space


def _build_online_sac_collector(
	cfg: SimpleNamespace,
	device,
	algo,
	observation_space: spaces.Box,
	action_space: spaces.Box,
	checkpoint_metadata: dict,
	action_chunk_size: int,
) -> tuple[RobomimicOnlineDataCollector, RobomimicOnlineReplayBuffer]:
	online_cfg = getattr(cfg, "online", SimpleNamespace())
	reward_cfg = getattr(online_cfg, "reward", SimpleNamespace())
	reward_type = str(getattr(reward_cfg, "type", "sparse_done"))
	reward_provider = _build_online_reward_provider(cfg=cfg, device=device)

	obs_adapter = ObservationAdapter(
		cfg=cfg,
		shape_metadata=checkpoint_metadata["shape_metadata"],
		obs_slices=checkpoint_metadata["obs_slices"],
		device=device,
		obs_normalization_stats=None,
	)
	visual_keys = list(checkpoint_metadata["shape_metadata"].get("visual_obs_keys", []))
	env = create_online_robomimic_env(
		env_meta=checkpoint_metadata["env_metadata"],
		obs_keys=list(cfg.dataset.obs_keys),
		visual_keys=visual_keys,
		env_name=getattr(getattr(cfg, "eval", SimpleNamespace()), "env_name_override", None),
		render=False,
		render_offscreen=False,
	)

	online_buffer = RobomimicOnlineReplayBuffer(
		buffer_size=int(getattr(online_cfg, "buffer_size", 100000)),
		observation_space=observation_space,
		action_space=action_space,
		device=str(device),
		action_chunk_size=action_chunk_size,
	)
	collector = RobomimicOnlineDataCollector(
		env=env,
		algo=algo,
		obs_adapter=obs_adapter,
		replay_buffer=online_buffer,
		action_normalization_stats=None,
		reward_type=reward_type,
		reward_provider=reward_provider,
		pbrs_gamma=float(getattr(reward_cfg, "pbrs_gamma", 0.99)),
		horizon=int(getattr(online_cfg, "horizon", 400)),
	)
	return collector, online_buffer


def _run_online_sac_training(
	cfg: SimpleNamespace,
	args,
	device,
	logger: WandBLogger,
	save_dir: str,
	checkpoint_metadata: dict,
	meta: dict,
) -> None:
	online_cfg = getattr(cfg, "online", SimpleNamespace())
	action_chunk_size = _validate_online_sac_config(online_cfg)

	_inject_runtime_obs_slices(
		cfg,
		algo_name="online_sac",
		replay_buffer=SimpleNamespace(
			obs_slices={k: slice(v[0], v[1]) for k, v in checkpoint_metadata["obs_slices"].items()}
		),
	)
	algo_cfg = get_algo_cfg(cfg, algo_name="online_sac")
	observation_space, action_space = _build_online_sac_spaces(checkpoint_metadata)
	algo = build_algo(
		algo_name="online_sac",
		observation_space=observation_space,
		action_space=action_space,
		cfg=algo_cfg,
		device=device,
	)
	if args.resume:
		algo.load(args.resume)
		print(f"[train_policy] resumed checkpoint: {args.resume}")

	rollout_evaluator = None
	if bool(getattr(getattr(cfg, "eval", None), "enabled", False)):
		rollout_evaluator = TrainingRolloutEvaluator(
			algo=algo,
			cfg=cfg,
			env_metadata=checkpoint_metadata["env_metadata"],
			shape_metadata=checkpoint_metadata["shape_metadata"],
			obs_slices=checkpoint_metadata["obs_slices"],
			obs_normalization_stats=None,
			action_normalization_stats=None,
			eval_cfg=cfg.eval,
			save_dir=save_dir,
		)

	collector, online_buffer = _build_online_sac_collector(
		cfg=cfg,
		device=device,
		algo=algo,
		observation_space=observation_space,
		action_space=action_space,
		checkpoint_metadata=checkpoint_metadata,
		action_chunk_size=action_chunk_size,
	)

	learning_starts = int(getattr(online_cfg, "learning_starts", 1000))
	train_freq = int(getattr(online_cfg, "train_freq", 1))
	gradient_steps = int(getattr(online_cfg, "gradient_steps", 1))
	batch_size = int(cfg.train.batch_size)
	log_every = max(1, int(cfg.train.log_every))
	save_every = max(1, int(cfg.train.save_every))
	n_steps = int(cfg.train.n_steps)

	print(
		"[train_policy] start online training "
		f"algo=online_sac target_steps={n_steps} learning_starts={learning_starts} "
		f"train_freq={train_freq} grad_steps={gradient_steps} batch_size={batch_size} device={device}"
	)

	algo.learn_online(
		collector=collector,
		replay_buffer=online_buffer,
		n_steps=n_steps,
		batch_size=batch_size,
		learning_starts=learning_starts,
		train_freq=train_freq,
		gradient_steps=gradient_steps,
		log_every=log_every,
		save_every=save_every,
		save_dir=save_dir,
		logger=logger,
		checkpoint_metadata=checkpoint_metadata,
		rollout_evaluator=rollout_evaluator,
	)

	# _write_online_dense_reward_trace(
	# 	cfg=cfg,
	# 	meta=meta,
	# 	collector=collector,
	# 	online_buffer=online_buffer,
	# )


def _build_offline_replay_buffer(cfg: SimpleNamespace, device) -> RobomimicReplayBuffer:
	return RobomimicReplayBuffer(
		h5_path=str(cfg.dataset.h5_path),
		obs_keys=list(cfg.dataset.obs_keys),
		filter_key=cfg.dataset.filter_key,
		device=str(device),
		action_keys=getattr(cfg.dataset, "action_keys", None),
		strict_next_obs=bool(getattr(cfg.dataset, "strict_next_obs", True)),
		normalize_obs=bool(getattr(cfg.dataset, "normalize_obs", False)),
		normalize_actions=bool(getattr(cfg.dataset, "normalize_actions", False)),
		normalization_epsilon=float(getattr(cfg.dataset, "normalization_epsilon", 1e-3)),
		normalization_clip=getattr(cfg.dataset, "normalization_clip", None),
		return_dict_obs=bool(getattr(cfg.dataset, "return_dict_obs", False)),
	)


def main() -> None:
	args = parse_policy_train_args()
	requested_algo = str(args.algo).lower() if args.algo else None
	config_path = args.config or f"configs/{requested_algo or 'iql'}.yaml"

	cfg_ns = PolicyTrainingConfig.load(config_path)
	cfg = cfg_ns
	algo_name = str(getattr(cfg, "algo_name", "")).lower()
	if args.algo:
		cfg.algo_name = str(args.algo).lower()
		algo_name = str(cfg.algo_name).lower()

	# Fail early with a direct message when config and --algo mismatch.
	_ = get_algo_cfg(cfg, algo_name=algo_name)

	if args.smoke:
		cfg.train.n_steps = 50
		if algo_name == "online_sac":
			online_cfg = getattr(cfg, "online", None)
			if online_cfg is not None:
				online_cfg.learning_starts = min(int(getattr(online_cfg, "learning_starts", 1000)), 10)

	seed_everything(int(cfg.seed))
	device = resolve_device(str(cfg.device), args.device)

	meta = derive_run_metadata(cfg)
	save_dir = resolve_save_dir(cfg, meta, str(cfg.project_root))
	wandb_group, wandb_name = build_wandb_run_id(meta)

	full_cfg_dict = namespace_to_dict(cfg)
	logger = WandBLogger(
		enabled=bool(cfg.wandb.enabled),
		project=str(cfg.wandb.project),
		group=wandb_group,
		name=wandb_name,
		config=full_cfg_dict,
	)

	if algo_name == "iql":
		replay_buffer = _build_offline_replay_buffer(cfg=cfg, device=device)
		checkpoint_metadata = build_checkpoint_metadata(cfg, replay_buffer)
		dataset_buffer_size = replay_buffer.size()
	else:
		dataset_metadata = extract_dataset_metadata(cfg)
		checkpoint_metadata = build_checkpoint_metadata(cfg, replay_buffer=None, dataset_metadata=dataset_metadata)
		dataset_buffer_size = int(checkpoint_metadata["shape_metadata"]["observation_dim"])

	print(
		"[train_policy] start training "
		f"algo={cfg.algo_name} env={meta['env_name']} task={meta['task_name']} "
		f"mask={meta['mask_name']} seed={meta['seed']} "
		f"dataset_buffer_size={dataset_buffer_size} steps={cfg.train.n_steps} "
		f"batch_size={cfg.train.batch_size} device={device}"
	)
	print(f"[train_policy] save_dir = {save_dir}")
	print(f"[train_policy] wandb group = {wandb_group} | name = {wandb_name}")
	if algo_name == "iql":
		_run_iql_training(
			cfg=cfg,
			args=args,
			device=device,
			logger=logger,
			save_dir=save_dir,
			checkpoint_metadata=checkpoint_metadata,
			replay_buffer=replay_buffer,
		)
	elif algo_name == "online_sac":
		_run_online_sac_training(
			cfg=cfg,
			args=args,
			device=device,
			logger=logger,
			save_dir=save_dir,
			checkpoint_metadata=checkpoint_metadata,
			meta=meta,
		)
	else:
		raise ValueError(f"Unsupported algo_name for train_policy.py: {cfg.algo_name!r}")

	logger.finish()
	print("[train_policy] done")


if __name__ == "__main__":
	main()
