"""Explicit project-relative configuration for the six supported tasks."""

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_settings(config_dir=None):
    directory = resolve_path(config_dir) if config_dir else ROOT / "configs"
    return {name: yaml.safe_load((directory / f"{name}.yaml").read_text())
            for name in ("tasks", "training", "progress")}


def get_task(settings, name):
    task = deepcopy(settings["tasks"][name])
    task["name"] = name
    for section in ("data", "labels", "weights"):
        task[section] = {key: resolve_path(path)
                         for key, path in task.get(section, {}).items()}
    return task
