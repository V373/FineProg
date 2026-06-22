"""Thin robomimic environment adapter helpers for policy_training."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from . import ensure_vendored_robomimic

ensure_vendored_robomimic()

from envs.robomimic_runtime.robomimic.envs.env_base import EnvType
import envs.robomimic_runtime.robomimic.utils.env_utils as EnvUtils
import envs.robomimic_runtime.robomimic.utils.file_utils as FileUtils
import envs.robomimic_runtime.robomimic.utils.obs_utils as ObsUtils


def _unique_in_order(keys: Iterable[str]) -> list[str]:
	seen: set[str] = set()
	ordered: list[str] = []
	for key in keys:
		if key not in seen:
			seen.add(key)
			ordered.append(key)
	return ordered


def load_env_metadata_from_dataset(dataset_path: str) -> dict:
	"""Load robomimic environment metadata from an HDF5 dataset."""
	env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
	if env_meta.get("type") != EnvType.ROBOSUITE_TYPE:
		raise NotImplementedError(
			"V1 evaluation only supports robosuite datasets. "
			f"Got env type {env_meta.get('type')} from {dataset_path}."
		)
	return env_meta


def create_robomimic_env(
	env_meta: dict,
	obs_keys: Sequence[str],
	visual_keys: Sequence[str],
	env_name: str | None = None,
	render: bool = False,
	render_offscreen: bool = False,
) -> object:
	"""Initialize obs modality mapping and create a robomimic env from metadata."""
	if env_meta.get("type") != EnvType.ROBOSUITE_TYPE:
		raise NotImplementedError(
			"V1 evaluation only supports robosuite env creation. "
			f"Got env type {env_meta.get('type')}."
		)

	visual_set = set(visual_keys)
	low_dim_keys = [key for key in obs_keys if key not in visual_set]
	obs_spec = {
		"obs": {
			"low_dim": _unique_in_order(low_dim_keys),
			"rgb": _unique_in_order(visual_keys),
		}
	}
	ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_spec)
	# EnvRobosuite imports robomimic.utils.obs_utils under the top-level package name,
	# so initialize that module too if it is distinct from the vendored import path.
	import robomimic.utils.obs_utils as RuntimeObsUtils
	if RuntimeObsUtils is not ObsUtils:
		RuntimeObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_spec)
	return EnvUtils.create_env_from_metadata(
		env_meta=env_meta,
		env_name=env_name,
		render=render,
		render_offscreen=(render_offscreen or bool(visual_keys)),
		use_image_obs=bool(visual_keys),
		use_depth_obs=False,
	)


def create_online_robomimic_env(
	env_meta: dict,
	obs_keys: Sequence[str],
	visual_keys: Sequence[str],
	env_name: str | None = None,
	render: bool = False,
	render_offscreen: bool = False,
) -> object:
	"""Create a robomimic env for online training data collection."""
	return create_robomimic_env(
		env_meta=env_meta,
		obs_keys=obs_keys,
		visual_keys=visual_keys,
		env_name=env_name,
		render=render,
		render_offscreen=render_offscreen,
	)


def reset_online_env(env: object) -> dict[str, Any]:
	"""Reset a robomimic env and normalize the return value to an obs dict."""
	obs = env.reset()
	if isinstance(obs, tuple):
		obs = obs[0]
	if not isinstance(obs, dict):
		raise TypeError(f"Expected robomimic reset() to return dict obs, got {type(obs)!r}")
	return obs


def step_online_env(env: object, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
	"""Step a robomimic env and normalize common gym/robomimic return formats."""
	result = env.step(action)
	if not isinstance(result, tuple):
		raise TypeError(f"Expected env.step() to return a tuple, got {type(result)!r}")
	if len(result) == 4:
		next_obs, reward, done, info = result
	elif len(result) == 5:
		next_obs, reward, terminated, truncated, info = result
		done = bool(terminated) or bool(truncated)
	else:
		raise ValueError(f"Unsupported env.step() return length: {len(result)}")
	if isinstance(next_obs, tuple):
		next_obs = next_obs[0]
	if not isinstance(next_obs, dict):
		raise TypeError(f"Expected robomimic step() to return dict obs, got {type(next_obs)!r}")
	return next_obs, float(reward), bool(done), dict(info or {})


def extract_success_done(done: bool, info: dict[str, Any] | None = None, env: object | None = None) -> bool:
	"""Interpret robomimic native done as the first-pass sparse success signal."""
	# First-port online_sac intentionally uses native done as requested.  If an
	# environment exposes richer success diagnostics, collectors may log them, but
	# they must not change the reward label in this helper.
	return bool(done)


def compute_sparse_done_reward(
	done: bool,
	success_reward: float = 1.0,
	failure_reward: float = 0.0,
) -> float:
	"""Map native robomimic done to a sparse success reward."""
	return float(success_reward if bool(done) else failure_reward)