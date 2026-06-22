"""Helpers for selecting algorithm-specific config blocks."""

from __future__ import annotations

from typing import Any


def get_algo_cfg(cfg: Any, algo_name: str | None = None) -> Any:
    """Return the config namespace for the active policy-training algorithm."""
    name = str(algo_name or getattr(cfg, "algo_name", "iql")).lower()
    if name == "iql":
        return cfg.iql
    if name == "online_sac":
        if hasattr(cfg, "online_sac"):
            return cfg.online_sac
        raise AttributeError("online_sac config block not found; expected cfg.online_sac")
    raise ValueError(f"Unknown algorithm config block for algo_name={name!r}")


def uses_resnet18conv(cfg: Any, algo_name: str | None = None) -> bool:
    """Return whether the active algorithm uses the ResNet18Conv feature extractor."""
    algo_cfg = get_algo_cfg(cfg, algo_name=algo_name)
    return str(getattr(algo_cfg, "features_extractor_type", "flat_range")) == "resnet18conv"