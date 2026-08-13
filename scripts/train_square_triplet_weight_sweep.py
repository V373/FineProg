#!/usr/bin/env python3
"""Train the current encoder config serially with two triplet weights.

The script snapshots ``configs_v2`` at startup, creates one isolated config
copy per weight, and overrides only the composite component whose alias is
``temporal_triplet``.  The repository's original YAML files are never edited.

Run from the fineprog conda environment, for example:

    conda run -n fineprog python scripts/train_square_triplet_weight_sweep.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


TRIPLET_WEIGHTS = (0.01, 0.5)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = PROJECT_ROOT / "configs_v2"


def set_temporal_triplet_weight(config_root: Path, weight: float) -> Path:
    """Override temporal_triplet.weight in the selected composite loss YAML."""
    train_path = config_root / "train.yaml"
    train_config = yaml.safe_load(train_path.read_text(encoding="utf-8")) or {}

    if train_config.get("loss_name") != "composite":
        raise ValueError(
            f"Expected loss_name='composite' in {train_path}, "
            f"got {train_config.get('loss_name')!r}"
        )

    loss_config_rel = train_config.get("loss_config")
    if not loss_config_rel:
        raise ValueError(f"Missing loss_config in {train_path}")

    loss_path = config_root / str(loss_config_rel)
    loss_config = yaml.safe_load(loss_path.read_text(encoding="utf-8")) or {}
    components = loss_config.get("components", [])
    matches = [item for item in components if item.get("alias") == "temporal_triplet"]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one components entry with alias='temporal_triplet' "
            f"in {loss_path}, found {len(matches)}"
        )

    matches[0]["weight"] = float(weight)
    loss_path.write_text(
        yaml.safe_dump(loss_config, sort_keys=False),
        encoding="utf-8",
    )
    return loss_path


def run_training_worker(config_root: Path) -> None:
    """Run train_encoder with the supplied isolated ConfigV2 directory."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import train_encoder
    from utils.config_v2 import ConfigV2

    config_v2 = ConfigV2(config_root)
    train_config = config_v2.load_train()
    loss_config_rel = train_config.get("loss_config", "loss/loss_tcc.yaml")

    # train_encoder resolves these module globals on import. Point them at the
    # isolated snapshot before invoking the same train() call as its CLI.
    train_encoder._CFG_V2 = config_v2
    train_encoder._TRAIN_V2 = train_config
    train_encoder._V2_TRAIN_YAML = str(config_root / "train.yaml")
    train_encoder._loss_name_v2 = train_config.get("loss_name", "tcc")
    train_encoder._loss_cfg_file = loss_config_rel
    train_encoder._V2_LOSS_YAML = str(config_root / loss_config_rel)

    train_encoder.train(
        num_epochs=train_config.get("num_epochs", 5),
        batch_size=train_config.get("batch_size", 2),
        learning_rate=train_config.get("learning_rate", 1e-4),
        log_every=train_config.get("log_every", 10),
        num_workers=train_config.get("num_workers", 0),
        checkpoint_every=train_config.get("checkpoint_every", 1000),
        checkpoint_dir=train_config.get("checkpoint_dir", "checkpoints/encoder"),
        h5_path=None,
        register=False,
        register_alias=None,
    )


def prepare_snapshots(temp_root: Path) -> list[tuple[float, Path, Path]]:
    """Create both configs up front so the two runs use one source snapshot."""
    source_snapshot = temp_root / "configs_source_snapshot"
    shutil.copytree(SOURCE_CONFIG_ROOT, source_snapshot)

    prepared = []
    for weight in TRIPLET_WEIGHTS:
        label = str(weight).replace(".", "p")
        config_root = temp_root / f"configs_triplet_weight_{label}"
        shutil.copytree(source_snapshot, config_root)
        loss_path = set_temporal_triplet_weight(config_root, weight)
        prepared.append((weight, config_root, loss_path))
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current encoder training config serially with "
            "temporal_triplet weights 0.01 and 0.5."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate both config overrides without starting training.",
    )
    parser.add_argument(
        "--worker-config-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker_config_root is not None:
        run_training_worker(args.worker_config_root.resolve())
        return 0

    if not SOURCE_CONFIG_ROOT.is_dir():
        raise FileNotFoundError(f"Missing config directory: {SOURCE_CONFIG_ROOT}")

    with tempfile.TemporaryDirectory(prefix="fineprog_triplet_weight_sweep_") as tmp:
        prepared = prepare_snapshots(Path(tmp))
        for index, (weight, config_root, loss_path) in enumerate(prepared, start=1):
            print(
                f"[triplet-weight-sweep] Prepared run {index}/{len(prepared)}: "
                f"temporal_triplet.weight={weight}\n"
                f"[triplet-weight-sweep] Temporary loss config: {loss_path}",
                flush=True,
            )

        if args.dry_run:
            print("[triplet-weight-sweep] Dry run complete; no training started.")
            return 0

        for index, (weight, config_root, _) in enumerate(prepared, start=1):
            print(
                f"\n[triplet-weight-sweep] Starting run {index}/{len(prepared)} "
                f"with temporal_triplet.weight={weight}",
                flush=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-config-root",
                    str(config_root),
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
            print(
                f"[triplet-weight-sweep] Finished run {index}/{len(prepared)} "
                f"with temporal_triplet.weight={weight}",
                flush=True,
            )

    print("[triplet-weight-sweep] All training runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
