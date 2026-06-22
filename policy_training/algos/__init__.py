"""Algorithm factory and package exports."""

from .offline_rl.iql import IQL
from .online_rl import OnlineSAC


def build_algo(algo_name, observation_space, action_space, cfg, device):
    """Build a policy-training algorithm instance from config name."""
    name = algo_name.lower()
    if name == "iql":
        return IQL(observation_space, action_space, cfg, device)
    if name == "online_sac":
        return OnlineSAC(observation_space, action_space, cfg, device)
    raise ValueError(f"Unknown algorithm: {algo_name}")
