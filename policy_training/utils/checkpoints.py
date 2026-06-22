"""Checkpoint schema helpers for policy_training."""

from __future__ import annotations

from typing import Any


CHECKPOINT_SCHEMA_VERSION = 1

REQUIRED_CHECKPOINT_FIELDS = (
    "checkpoint_schema_version",
    "global_step",
    "algo_name",
    "config",
    "env_metadata",
    "shape_metadata",
    "obs_slices",
    "modules",
    "optimizers",
)


def normalize_obs_slices(obs_slices: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Normalize serialized obs_slices into integer tuple pairs."""
    normalized: dict[str, tuple[int, int]] = {}
    for key, value in obs_slices.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"obs_slices[{key!r}] must be a length-2 list or tuple, got {value!r}")
        start, stop = int(value[0]), int(value[1])
        if stop < start:
            raise ValueError(f"obs_slices[{key!r}] has invalid range ({start}, {stop})")
        normalized[str(key)] = (start, stop)
    return normalized


def validate_checkpoint_payload(payload: dict[str, Any], *, context: str) -> None:
    """Validate that a checkpoint payload matches the strict v1 schema."""
    missing = [field for field in REQUIRED_CHECKPOINT_FIELDS if field not in payload]
    if missing:
        raise ValueError(
            f"{context}: checkpoint is missing required fields: {', '.join(missing)}. "
            "Legacy checkpoints are not supported by Eval V1; regenerate them with the new trainer."
        )

    version = payload.get("checkpoint_schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"{context}: unsupported checkpoint_schema_version={version!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )

    if not isinstance(payload.get("config"), dict):
        raise ValueError(f"{context}: checkpoint field 'config' must be a dict.")
    if not isinstance(payload.get("env_metadata"), dict):
        raise ValueError(f"{context}: checkpoint field 'env_metadata' must be a dict.")
    if not isinstance(payload.get("shape_metadata"), dict):
        raise ValueError(f"{context}: checkpoint field 'shape_metadata' must be a dict.")
    if not isinstance(payload.get("modules"), dict):
        raise ValueError(f"{context}: checkpoint field 'modules' must be a dict.")
    if not isinstance(payload.get("optimizers"), dict):
        raise ValueError(f"{context}: checkpoint field 'optimizers' must be a dict.")

    normalize_obs_slices(payload.get("obs_slices", {}))