#!/usr/bin/env python3
"""Run online_sac training three times with different reward types.

This script updates policy_training/configs/online_sac.yaml at key
online.reward.type in order: pbrs -> dense -> sparse, then runs:
python train_policy.py --algo online_sac
for each type serially.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REWARD_TYPES = ("pbrs", "dense", "sparse")
REWARD_TYPE_TO_CONFIG_VALUE = {
    "pbrs": "pbrs",
    "dense": "dense",
    # Current online_sac implementation uses sparse_done as sparse reward keyword.
    "sparse": "sparse_done",
}


def set_online_reward_type(config_path: Path, reward_type: str) -> None:
    """Replace only online.reward.type while keeping file text unchanged otherwise."""
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)

    in_online = False
    in_reward = False
    replaced = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        # Top-level section handling.
        if indent == 0 and stripped.endswith(":"):
            in_online = stripped == "online:"
            in_reward = False
            continue

        if not in_online:
            continue

        # Leave online section if we hit next top-level key.
        if indent == 0 and stripped and not stripped.startswith("#"):
            in_online = False
            in_reward = False
            continue

        # Enter reward subsection under online (2-space indent).
        if indent == 2 and stripped == "reward:":
            in_reward = True
            continue

        # Leave reward subsection if indentation comes back to <= 2 on a key line.
        if in_reward and indent <= 2 and stripped and not stripped.startswith("#"):
            in_reward = False

        if not in_reward:
            continue

        # Replace the exact key online.reward.type (expected 4-space indent).
        if indent == 4 and stripped.startswith("type:"):
            line_ending = "\n" if line.endswith("\n") else ""
            lines[idx] = f"    type: {reward_type}{line_ending}"
            replaced = True
            break

    if not replaced:
        raise RuntimeError(
            f"Could not find online.reward.type in config: {config_path}"
        )

    config_path.write_text("".join(lines), encoding="utf-8")


def run_training(policy_training_dir: Path, python_cmd: str) -> None:
    cmd = [python_cmd, "train_policy.py", "--algo", "online_sac"]
    subprocess.run(cmd, cwd=str(policy_training_dir), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run online_sac training serially for pbrs/dense/sparse rewards."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch train_policy.py (default: current interpreter).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    policy_training_dir = repo_root / "policy_training"
    config_path = policy_training_dir / "configs" / "online_sac.yaml"

    if not policy_training_dir.exists():
        raise FileNotFoundError(f"Missing directory: {policy_training_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    for reward_type in REWARD_TYPES:
        config_value = REWARD_TYPE_TO_CONFIG_VALUE[reward_type]
        print(
            f"\n[reward-sweep] Setting online.reward.type={config_value} "
            f"(requested: {reward_type})"
        )
        set_online_reward_type(config_path, config_value)
        print("[reward-sweep] Running: python train_policy.py --algo online_sac")
        run_training(policy_training_dir=policy_training_dir, python_cmd=args.python)

    print("\n[reward-sweep] All runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
