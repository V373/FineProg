"""V2 Runtime Registry — append-only writer for configs_v2/registry/*.yaml.

Appends new run/embedding/dataset entries to the registry YAML files without
touching existing entries or comments.  Only activated when the calling script
passes ``--register``; without that flag nothing is written.

Usage
-----
    from utils.registry_v2 import RegistryV2

    reg = RegistryV2()

    # After training:
    reg.register_run(
        alias       = "can_ph_180_50k",          # or pass None → auto-suggested
        run_name    = "TCC-robomimic_can_ph-...",
        train_dataset = "robomimic_can_ph_180vid_train",
        backbone    = "resnet50_conv4c",
        train_base  = "only_bn",
        checkpoint_epoch = 50000,
        description = "Can PH 180-vid, epoch 50000",
    )

    # After extract_embeddings.py:
    reg.register_embedding(
        alias       = "can_ph_valid_ep50k",
        run_ref     = "can_ph_180_50k",
        dataset_ref = "robomimic_can_ph_20vid_valid",
        variant     = "standard",
        description = "Can PH valid embeddings (20 vids)",
    )

    # After mp4vid_to_h5data.py:
    reg.register_dataset(
        alias        = "robomimic_can_ph_180vid_train",
        processed_h5 = "robomimic_can_ph-180vid_train.h5",
        display_name = "Robomimic Can PH Train (180 videos)",
        raw_dir      = "robomimic_can_ph",
        robomimic_hdf5 = "datasets/raw/robomimic_can_ph/demo_v15.hdf5",
        mask_key     = "20_percent",
    )
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Optional

import yaml

_PROJ_ROOT    = Path(__file__).resolve().parent.parent
_REGISTRY_DIR = _PROJ_ROOT / "configs_v2" / "registry"

_RUNS_YAML     = _REGISTRY_DIR / "runs.yaml"
_DATASETS_YAML = _REGISTRY_DIR / "datasets.yaml"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Load YAML; return empty dict on failure."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _alias_exists(path: Path, section: str, alias: str) -> bool:
    """Return True if *alias* is already a key under *section* in *path*."""
    doc = _load_yaml(path)
    return alias in doc.get(section, {})


def _indent_block(text: str, spaces: int = 2) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def _yaml_scalar(value) -> str:
    """Format a scalar value for inline YAML (strings with colons quoted)."""
    if isinstance(value, str) and any(c in value for c in (':', '#', '{', '}')):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "~"
    return str(value)


def _build_entry_block(alias: str, fields: dict[str, object]) -> str:
    """Build a 2-space-indented YAML block for one registry entry.

    Example output::

        can_ph_180_ep50k:
          description: "Can PH 180-vid, epoch 50000"
          run_name: TCC-robomimic_can_ph-...
          ...
    """
    lines = [f"  {alias}:"]
    for k, v in fields.items():
        if v is None or v == "":
            continue
        lines.append(f"    {k}: {_yaml_scalar(v)}")
    return "\n".join(lines)


def _append_entry(yaml_path: Path, section: str, alias: str,
                  fields: dict, overwrite: bool = False) -> bool:
    """Append *alias* entry under *section* in *yaml_path*.

    Reads the current file content as plain text and appends a new block at the
    end of the *section* (detected by the bare ``<section>:`` line).  This
    preserves all existing entries and comments.

    Returns True if the entry was written, False if skipped (duplicate).
    """
    if _alias_exists(yaml_path, section, alias):
        if not overwrite:
            print(
                f"[RegistryV2] SKIP: alias '{alias}' already exists in "
                f"{yaml_path.name}::{section}.  "
                f"Use overwrite=True to replace it."
            )
            return False
        # overwrite: remove the existing entry block before re-appending
        _remove_entry(yaml_path, section, alias)

    entry_block = _build_entry_block(alias, fields)
    with open(yaml_path, "a") as f:
        f.write(f"\n{entry_block}\n")

    print(f"[RegistryV2] Registered '{alias}' → {yaml_path.relative_to(_PROJ_ROOT)}")
    return True


def _remove_entry(yaml_path: Path, section: str, alias: str) -> None:
    """Remove an existing alias block from the YAML file (best-effort, text-based)."""
    text = yaml_path.read_text()
    # Match the alias line and all following indented lines
    pattern = re.compile(
        rf"^\s{{2}}{re.escape(alias)}:.*?(?=\n\S|\n  \S|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    new_text = pattern.sub("", text)
    yaml_path.write_text(new_text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RegistryV2:
    """Append-only writer for configs_v2/registry/*.yaml.

    Parameters
    ----------
    registry_dir:
        Path to the configs_v2/registry/ directory.
        Defaults to ``<project_root>/configs_v2/registry/``.
    """

    def __init__(self, registry_dir: Optional[str | Path] = None) -> None:
        self._dir = Path(registry_dir) if registry_dir else _REGISTRY_DIR
        self._runs_yaml     = self._dir / "runs.yaml"
        self._datasets_yaml = self._dir / "datasets.yaml"

    # ------------------------------------------------------------------ #
    # Register a training run
    # ------------------------------------------------------------------ #

    def register_run(
        self,
        alias:            str,
        run_name:         str,
        train_dataset:    str,
        backbone:         str,
        train_base:       str,
        checkpoint_epoch: int,
        description:      str = "",
        overwrite:        bool = False,
    ) -> bool:
        """Append a run entry to registry/runs.yaml under ``runs:``."""
        fields = {
            "description":      description or f"{train_dataset}, epoch {checkpoint_epoch}",
            "run_name":         run_name,
            "train_dataset":    train_dataset,
            "backbone":         backbone,
            "train_base":       train_base,
            "checkpoint_epoch": checkpoint_epoch,
        }
        return _append_entry(self._runs_yaml, "runs", alias, fields, overwrite)

    # ------------------------------------------------------------------ #
    # Register an embedding artifact
    # ------------------------------------------------------------------ #

    def register_embedding(
        self,
        alias:       str,
        run_ref:     str,
        dataset_ref: str,
        variant:     str = "standard",
        description: str = "",
        overwrite:   bool = False,
    ) -> bool:
        """Append an embedding entry to registry/runs.yaml under ``embeddings:``."""
        fields = {
            "description": description or f"{dataset_ref} ({variant})",
            "run_ref":     run_ref,
            "dataset_ref": dataset_ref,
            "variant":     variant,
        }
        return _append_entry(self._runs_yaml, "embeddings", alias, fields, overwrite)

    # ------------------------------------------------------------------ #
    # Register a processed dataset
    # ------------------------------------------------------------------ #

    def register_dataset(
        self,
        alias:          str,
        processed_h5:   str,
        display_name:   str = "",
        raw_dir:        Optional[str] = None,
        robomimic_hdf5: Optional[str] = None,
        mask_key:       Optional[str] = None,
        phase_labels_csv: Optional[str] = None,
        overwrite:      bool = False,
    ) -> bool:
        """Append a dataset entry to registry/datasets.yaml under ``datasets:``."""
        fields: dict = {"display_name": display_name or alias, "processed_h5": processed_h5}
        if raw_dir:
            fields["raw_dir"] = raw_dir
        if robomimic_hdf5:
            fields["robomimic_hdf5"] = robomimic_hdf5
        if mask_key:
            fields["mask_key"] = mask_key
        if phase_labels_csv:
            fields["phase_labels_csv"] = phase_labels_csv
        return _append_entry(self._datasets_yaml, "datasets", alias, fields, overwrite)

    # ------------------------------------------------------------------ #
    # Auto-suggest aliases
    # ------------------------------------------------------------------ #

    @staticmethod
    def suggest_run_alias(train_dataset: str, checkpoint_epoch: int) -> str:
        """Generate a short run alias, e.g. 'can_ph_180_ep50k'."""
        # Shorten known prefixes
        stem = train_dataset.replace("robomimic_", "")
        epoch_k = checkpoint_epoch // 1000
        return f"{stem}_ep{epoch_k}k"

    @staticmethod
    def suggest_embedding_alias(run_ref: str, dataset_ref: str, variant: str) -> str:
        """Generate a short embedding alias."""
        ds = dataset_ref.replace("robomimic_", "")
        suffix = {"standard": "", "mean_path": "_mean_path", "labeled": "_labeled"}.get(variant, f"_{variant}")
        # Strip epoch suffix from run_ref to avoid double-stamping
        run_short = re.sub(r"_ep\d+k$", "", run_ref)
        epoch_m = re.search(r"_ep(\d+k)$", run_ref)
        epoch_sfx = f"_{epoch_m.group(1)}" if epoch_m else ""
        return f"{ds}{suffix}{epoch_sfx}"

    @staticmethod
    def suggest_dataset_alias(
        processed_h5_filename: str,
        mask_key: Optional[str] = None,
        two_split: bool = False,
    ) -> str | list[str]:
        """Generate a dataset alias from the output H5 filename.

        Returns a single string normally, or a list of two strings when
        ``two_split=True`` (train + valid aliases).
        """
        stem = Path(processed_h5_filename).stem  # e.g. robomimic_can_ph-180vid_train
        # Normalise: replace hyphens and spaces with underscores
        alias = stem.replace("-", "_").replace(" ", "_").lower()
        if two_split and mask_key:
            return [f"{alias}_train", f"{alias}_valid"]
        return alias
