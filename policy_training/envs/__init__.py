"""Environment adapters and vendored runtime setup for policy evaluation."""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_vendored_robomimic() -> Path:
	"""Put the vendored robomimic runtime on sys.path and return its root."""
	vendor_root = Path(__file__).resolve().parent / "robomimic_runtime"
	if str(vendor_root) not in sys.path:
		sys.path.insert(0, str(vendor_root))
	return vendor_root


ensure_vendored_robomimic()

from .robomimic import create_robomimic_env, load_env_metadata_from_dataset

__all__ = ["ensure_vendored_robomimic", "create_robomimic_env", "load_env_metadata_from_dataset"]