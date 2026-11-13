"""CPU-only data/config checks; never start an optimizer or create checkpoints."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import h5py
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "real_launcher", PROJECT / "scripts/train_real_single_frame_baselines_serial.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def data_manifest(tmp_path):
    manifest = {"schema_version": 1, "tasks": {}}
    for task in launcher.TASKS:
        splits = {}
        for split, count in (("train", 36), ("valid", 4), ("calibration", 2)):
            path = tmp_path / f"{task}-{split}.h5"
            with h5py.File(path, "w") as handle:
                for index in range(count):
                    video = handle.create_group(f"videos/{index + 1:06d}")
                    video.create_dataset("frames", shape=(2, 224, 224, 3), dtype="uint8")
                    video.attrs["source_episode"] = f"{split}_{index}"
            splits[split] = {"path": str(path), "count": count,
                             "sha256": launcher.sha256(path),
                             "alias": f"{task}_{count}vid_{split}"}
        manifest["tasks"][task] = {"splits": splits}
    path = tmp_path / "data_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_dry_run_is_torch_free_and_configs_resolve(data_manifest, tmp_path):
    # Run through the public CLI in a fresh interpreter, then verify no PyTorch
    # import and no run_queue call occurred. This does not mock config generation.
    root = tmp_path / "dry_run"
    code = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('real_launcher', sys.argv[1])
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
def forbidden(*args):
    raise AssertionError('dry-run started training')
launcher.run_queue = forbidden
assert launcher.main(['--dataset-manifest', sys.argv[2], '--output-root', sys.argv[3], '--dry-run']) == 0
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", code,
                             str(PROJECT / "scripts/train_real_single_frame_baselines_serial.py"),
                             str(data_manifest), str(root)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["stage_order"] == ["drawer_place/local", "drawer_place/hf1",
                                        "towel/local", "towel/hf1"]
    assert not list(root.rglob("*.pt"))
    sys.path.insert(0, str(PROJECT))
    from utils.config_v2 import ConfigV2
    for name, job in manifest["stages"].items():
        config_root = Path(job["config_root"])
        resolver = ConfigV2(configs_v2_dir=config_root)
        config = resolver.load_train()
        task, stage = name.split("/")
        assert config["num_epochs"] == 20000 and config["single_frame_context"]
        assert config["h5_path"] == str(tmp_path / f"{task}-train.h5")
        assert config["in_training_eval"]["eval_dataset_ref"] == f"{task}_4vid_valid"
        assert config["in_training_eval"]["selected_eval_tasks_in_training"] == ["kendalls_tau"]
        assert config["in_training_eval"]["log_images_to_wandb"] is False
        assert config["in_training_eval"]["kendalls_tau"]["save_heatmap"] is False
        if stage == "hf1":
            assert job["source_stage"] == f"{task}/local"
            assert config["memory"]["local_checkpoint_path"] is None
            assert config["memory"]["freeze_bn_stats"]
            assert config["train_embedding"] is False


def test_changed_dataset_hash_rejected(data_manifest):
    manifest = json.loads(data_manifest.read_text())
    manifest["tasks"]["towel"]["splits"]["train"]["sha256"] = "wrong"
    data_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        launcher.validate_sources(data_manifest)


@pytest.mark.parametrize("split", ["valid", "calibration"])
def test_source_overlap_rejected_even_when_file_hash_matches(data_manifest, split):
    manifest = json.loads(data_manifest.read_text())
    entry = manifest["tasks"]["drawer_place"]["splits"][split]
    with h5py.File(entry["path"], "r+") as handle:
        handle["videos/000001"].attrs["source_episode"] = "train_0"
    entry["sha256"] = launcher.sha256(Path(entry["path"]))
    data_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlapping source episode"):
        launcher.validate_sources(data_manifest)


def test_kendall_numeric_monitor_does_not_render(monkeypatch):
    import numpy as np
    sys.path.insert(0, str(PROJECT))
    from algos.eval_task.tcc_eval_tasks import task_kendall

    def forbidden(*args, **kwargs):
        raise AssertionError("Numeric Kendall monitoring must not render a heatmap")

    monkeypatch.setattr(task_kendall, "_save_kendall_heatmap", forbidden)
    task = task_kendall.KendallsTauTask()
    assert task.save_heatmap is True  # Preserve old callers' behavior.
    task.configure({"kendall_stride": 1, "save_heatmap": False})
    reference = np.arange(6, dtype=np.float32).reshape(-1, 1)
    result = task.evaluate({"embeddings": [reference, reference.copy()],
                            "video_id": ["first", "second"]})
    assert result["metric_value"] == pytest.approx(1.0)
    assert "output_heatmap_path" not in result
