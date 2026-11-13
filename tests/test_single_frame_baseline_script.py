"""The serial launcher hands one validated local model to both HF1 baselines."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import h5py
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "single_frame_baselines", ROOT / "scripts/train_pick_mango_single_frame_baselines_serial.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def source_config(tmp_path, monkeypatch):
    source = tmp_path / "source_configs"
    shutil.copytree(ROOT / "configs_v2", source)
    config = yaml.safe_load((source / "train.yaml").read_text())
    config["data_seed"] = 42  # Must not leak from another ablation into local/warm.
    (source / "train.yaml").write_text(yaml.safe_dump(config))
    dataset = tmp_path / "pick_mango_vid-36vid_train.h5"
    with h5py.File(dataset, "w") as handle:
        videos = handle.create_group("videos")
        for index in range(36):
            videos.create_group(f"{index + 1:06d}")
    monkeypatch.setattr(launcher, "SOURCE_CONFIG_ROOT", source)
    monkeypatch.setattr(launcher, "EXPECTED_H5", dataset)
    monkeypatch.setattr(launcher, "_require_cuda", lambda: None)
    return source


def _payload(config=None):
    config = config or {
        **deepcopy(launcher.COMMON_CONFIG), "train_base": "only_bn",
        "train_embedding": True, "memory": {"enabled": False},
    }
    encoder_config = {key: config[key] for key in (
        "single_frame_context", "clip_len", "context_size", "context_stride",
        "embedding_dim", "embedding_normalization", "train_base", "train_embedding",
    )}
    memory = deepcopy(config["memory"])
    if memory.get("local_checkpoint_path"):
        memory["local_checkpoint_sha256"] = launcher.sha256(Path(memory["local_checkpoint_path"]))
    encoder_config["memory_config"] = memory
    return {
        "model_state_dict": {"backbone.test_weight": torch.ones(1)},
        "embedding_normalization": "l2", "encoder_config": encoder_config,
        "training_provenance": {"epoch": launcher.EPOCHS, "run_id": "fake-test-run"},
    }


def _write_checkpoint(path, payload=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_payload() if payload is None else payload, path)
    return path


def _read_manifest(root):
    return json.loads((root / "manifest.json").read_text())


def test_dry_run_creates_isolated_three_stage_recipe(source_config, tmp_path, monkeypatch):
    before = (source_config / "train.yaml").read_bytes()
    root = tmp_path / "dry_run"
    monkeypatch.setattr(launcher, "_require_cuda", lambda: pytest.fail("CUDA check in dry run"))
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: pytest.fail("training in dry run"))
    assert launcher.main(["--dry-run", "--output-root", str(root)]) == 0
    manifest = _read_manifest(root)
    assert manifest["stage_order"] == list(launcher.STAGES)
    assert manifest["state"] == "prepared"
    for stage, job in manifest["stages"].items():
        config = yaml.safe_load((Path(job["config_root"]) / "train.yaml").read_text())
        assert config["num_epochs"] == 20000
        assert config["single_frame_context"] is True
        assert (config["context_size"], config["context_stride"], config["clip_len"]) == (2, 20, 20)
        assert config["seed"] == 42 and "data_seed" not in config
        assert config["embedding_normalization"] == "l2" and config["embedding_dim"] == 128
        assert config["in_training_eval"] == launcher.COMMON_CONFIG["in_training_eval"]
        assert config["in_training_eval"]["enabled"] is True
        assert config["checkpoint_every"] == 2000
        assert config["train_base"] == ("frozen" if stage == "hf1" else "only_bn")
        assert config["train_embedding"] is (stage != "hf1")
        if stage == "local":
            assert config["memory"] == {"enabled": False}
            assert config["pretrained"] is True
        else:
            assert config["memory"]["local_checkpoint_path"] is None
            assert config["memory"]["mode"] is None and config["memory"]["dual_branch"] is False
            assert config["memory"]["use_residual"] is False
            assert config["memory"]["normalize_input"] is True
            assert config["memory"]["history_frames"] == 1
            assert config["memory"]["freeze_bn_stats"] is (stage == "hf1")
            assert config["memory"]["cache_local_embeddings"] is (stage == "hf1")
            assert config["memory"]["checkpoint_visual"] is True
    assert (source_config / "train.yaml").read_bytes() == before


def _mock_training(monkeypatch, *, failure_stage=None, final_count=1):
    calls = []

    def train(command, **kwargs):
        config = yaml.safe_load((Path(kwargs["env"]["FINEPROG_CONFIGS_V2_DIR"]) / "train.yaml").read_text())
        stage = kwargs["env"]["WANDB_JOB_TYPE"]
        calls.append((stage, config, kwargs["env"]))
        kwargs["stdout"].write(f"Mock training: {stage}\n")
        if stage == failure_stage:
            return subprocess.CompletedProcess(command, 7)
        for index in range(final_count):
            path = Path(config["checkpoint_dir"]) / f"mock_run_{index}" / "encoder_epoch020000.pt"
            _write_checkpoint(path, _payload(config))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", train)
    return calls


def test_serial_handoff_uses_same_local_source_for_both_hf1(source_config, tmp_path, monkeypatch):
    calls = _mock_training(monkeypatch)
    root = tmp_path / "train"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("WANDB_RUN_ID", "unrelated-run")
    monkeypatch.setenv("WANDB_RESUME", "must")
    monkeypatch.setenv("WANDB_NAME", "unrelated-name")
    assert launcher.main(["--output-root", str(root), "--gpu", "3"]) == 0
    assert [stage for stage, _, _ in calls] == list(launcher.STAGES)
    manifest = _read_manifest(root)
    assert manifest["state"] == "complete"
    local = manifest["stages"]["local"]
    for stage, config, env in calls:
        assert env["CUDA_VISIBLE_DEVICES"] == "3"
        assert not {"WANDB_RUN_ID", "WANDB_RESUME", "WANDB_NAME"}.intersection(env)
        job = manifest["stages"][stage]
        assert job["state"] == "complete"
        assert Path(job["log_path"]).read_text() == f"Mock training: {stage}\n"
        assert job["checkpoint_sha256"] == launcher.sha256(Path(job["checkpoint_path"]))
        if stage != "local":
            assert config["memory"]["local_checkpoint_path"] == local["checkpoint_path"]
            assert job["source_checkpoint"] == local["checkpoint_path"]
            assert job["source_checkpoint_sha256"] == local["checkpoint_sha256"]


def test_completed_local_can_be_reused_without_retraining(source_config, tmp_path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path / "original_local.pt")
    original_hash = launcher.sha256(checkpoint)
    calls = _mock_training(monkeypatch)
    root = tmp_path / "reuse"
    assert launcher.main(["--output-root", str(root), "--local-checkpoint", str(checkpoint)]) == 0
    assert [stage for stage, _, _ in calls] == ["hf1", "hf1_warm_only_bn"]
    assert _read_manifest(root)["stages"]["local"]["state"] == "reused"
    assert all(config["memory"]["local_checkpoint_path"] == str(checkpoint) for _, config, _ in calls)
    assert launcher.sha256(checkpoint) == original_hash


@pytest.mark.parametrize("field,value", [
    ("single_frame_context", False), ("embedding_dim", 64),
    ("embedding_normalization", "none"), ("context_size", 1), ("context_stride", 1),
])
def test_reuse_rejects_wrong_local_metadata(source_config, tmp_path, field, value):
    payload = _payload()
    payload["encoder_config"][field] = value
    checkpoint = _write_checkpoint(tmp_path / "wrong.pt", payload)
    with pytest.raises(ValueError, match=field):
        launcher.prepare(tmp_path / "invalid", checkpoint)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("invalid", ["epoch", "memory"])
def test_reuse_rejects_incomplete_or_memory_checkpoint(source_config, tmp_path, invalid):
    payload = _payload()
    if invalid == "epoch":
        payload["training_provenance"]["epoch"] = 18000
    else:
        payload["encoder_config"]["memory_config"]["enabled"] = True
    checkpoint = _write_checkpoint(tmp_path / "invalid.pt", payload)
    with pytest.raises(ValueError, match=invalid):
        launcher.validate_checkpoint(checkpoint, "local")


@pytest.mark.parametrize("failure_stage", ["local", "hf1"])
def test_stage_failure_stops_later_training(source_config, tmp_path, monkeypatch, failure_stage):
    calls = _mock_training(monkeypatch, failure_stage=failure_stage)
    root = tmp_path / "failure"
    assert launcher.main(["--output-root", str(root)]) == 7
    assert [name for name, _, _ in calls] == list(launcher.STAGES[:launcher.STAGES.index(failure_stage) + 1])
    manifest = _read_manifest(root)
    assert manifest["state"] == "failed"
    assert manifest["stages"][failure_stage]["state"] == "failed"
    assert manifest["stages"]["hf1_warm_only_bn"]["state"] == "pending"


@pytest.mark.parametrize("final_count", [0, 2])
def test_missing_or_ambiguous_final_checkpoint_stops_handoff(source_config, tmp_path, monkeypatch, final_count):
    calls = _mock_training(monkeypatch, final_count=final_count)
    root = tmp_path / "bad_final"
    assert launcher.main(["--output-root", str(root)]) == 1
    assert [name for name, _, _ in calls] == ["local"]
    assert _read_manifest(root)["stages"]["local"]["state"] == "failed"


def test_existing_output_is_never_overwritten(source_config, tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    sentinel = root / "manifest.json"
    sentinel.write_text("existing user output")
    with pytest.raises(FileExistsError):
        launcher.prepare(root, None)
    assert sentinel.read_text() == "existing user output"


def test_reference_loss_change_is_reported_before_snapshot(source_config, tmp_path):
    path = source_config / "loss/loss_temporal_triplet.yaml"
    config = yaml.safe_load(path.read_text())
    config["margin"] = 1.0
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="margin"):
        launcher.prepare(tmp_path / "changed_loss", None)
