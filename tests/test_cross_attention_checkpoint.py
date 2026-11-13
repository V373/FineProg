"""Source migration, self-contained checkpoints and lightweight train logging."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch
import yaml

from test_streaming_mamba2 import (  # Reuse CPU glue fixtures, not a Mamba numerical substitute.
    cpu_memory, tiny_local, training_module, make_encoder,
)
from fineprog.utils import encoder_checkpoint as checkpoints


def source_checkpoint(tmp_path, *, normalize_input=True, normalization="l2", **memory):
    source = make_encoder(
        embedding_dim=4, embedding_normalization=normalization,
        train_base="frozen", train_embedding=False,
        memory_config={"history_frames": 1, "history_stride": 7,
                       "normalize_input": normalize_input, "use_residual": False, **memory},
    ).eval()
    path = tmp_path / "source.pt"
    torch.save(checkpoints.checkpoint_dict(source), path)
    return source, path


def migration_config(path, **memory):
    return {
        "clip_len": 5, "context_size": 2, "context_stride": 99,
        "embedding_dim": 128, "embedding_normalization": "none",
        "train_base": "frozen", "train_embedding": False,
        "memory": {"enabled": True, "mode": "cross_attention",
                   "source_checkpoint_path": str(path), "d_model": 128, "d_state": 8,
                   "normalize_input": False, "use_residual": True,
                   "freeze_pretrained_memory": True, **memory},
    }


@pytest.mark.parametrize("normalize_input", [False, True])
@pytest.mark.parametrize("normalization", ["none", "l2"])
def test_source_short_path_and_architecture_are_preserved(
        cpu_memory, tmp_path, normalize_input, normalization):
    source, path = source_checkpoint(
        tmp_path, normalize_input=normalize_input, normalization=normalization)
    config = migration_config(path, fusion_mode="short_only")
    before = copy.deepcopy(config)
    migrated, resolved = checkpoints.build_training_encoder(config, "cpu")
    migrated.eval()
    assert config == before
    assert migrated.clip_len == 5
    assert migrated.context_stride == source.context_stride == 2
    assert migrated.embedding_dim == source.embedding_dim == 4
    assert migrated.embedding_normalization == normalization
    assert migrated.memory.normalize_input is normalize_input
    assert migrated.memory.branch_output_l2 is (normalization == "l2")
    assert migrated.memory.config["d_state"] == source.memory.config["d_state"]
    assert resolved["memory"]["source_checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert resolved["memory"]["source_encoder_config"] == checkpoints.encoder_config(source)
    for name in ("backbone", "temporal_embedder"):
        for key, value in getattr(source, name).state_dict().items():
            torch.testing.assert_close(getattr(migrated, name).state_dict()[key], value, rtol=0, atol=0)
    raw = [torch.randn(n, 4) for n in (7, 3)]
    with torch.no_grad():
        torch.testing.assert_close(migrated.forward_sequence(raw, [7, 3]),
                                   source.forward_sequence(raw, [7, 3]), rtol=1e-6, atol=1e-6)


def test_checkpoint_roundtrip_does_not_read_source_and_preserves_provenance(cpu_memory, tmp_path):
    source, path = source_checkpoint(tmp_path)
    source_provenance = {"run_id": "source-id", "epoch": 20000}
    torch.save(checkpoints.checkpoint_dict(source, source_provenance), path)
    model, _ = checkpoints.build_training_encoder(migration_config(path, branch_output_l2=False), "cpu")
    model.eval()
    assert not model.memory.branch_output_l2
    assert model.memory_config["source_training_provenance"] == source_provenance
    provenance = {"run_id": "current-id", "run_name": "migration", "project": "unit-test", "epoch": 7}
    checkpoint = checkpoints.checkpoint_dict(model, provenance)
    restored_path = tmp_path / "migrated.pt"
    torch.save(checkpoint, restored_path)
    path.unlink()
    restored = checkpoints.build_encoder_from_checkpoint(restored_path, "cpu").eval()
    assert restored.training_provenance == provenance
    assert checkpoints.checkpoint_dict(restored)["training_provenance"] == provenance
    raw = [torch.randn(n, 4) for n in (7, 2)]
    with torch.no_grad():
        torch.testing.assert_close(model.forward_sequence(raw, [7, 2]),
                                   restored.forward_sequence(raw, [7, 2]), rtol=0, atol=0)


@pytest.mark.parametrize("memory", [
    {"enabled": False}, {"history_frames": 2}, {"use_residual": True}, {"dual_branch": True},
])
def test_migration_rejects_incompatible_sources(cpu_memory, tmp_path, memory):
    _, path = source_checkpoint(tmp_path, **memory)
    with pytest.raises(ValueError, match="single-branch"):
        checkpoints.build_training_encoder(migration_config(path), "cpu")


def test_migration_requires_explicit_source_and_does_not_accept_local_alias(cpu_memory, tmp_path):
    _, path = source_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="requires memory.source_checkpoint_path"):
        checkpoints.build_training_encoder(migration_config(None, source_checkpoint_path=None), "cpu")
    with pytest.raises(ValueError, match="source_checkpoint_path alone"):
        checkpoints.build_training_encoder(migration_config(path, local_checkpoint_path=str(path)), "cpu")


def test_frozen_source_survives_train_and_optimizer_updates(cpu_memory, training_module, tmp_path):
    _, path = source_checkpoint(tmp_path)
    model, _ = checkpoints.build_training_encoder(migration_config(path), "cpu")
    model.eval().train()
    assert not model.backbone.bn.training
    source_prefixes = ("backbone.", "temporal_embedder.", "memory.block.", "memory.proj.")
    frozen = {name: value.detach().clone() for name, value in model.state_dict().items()
              if name.startswith(source_prefixes)}
    assert all(not param.requires_grad for name, param in model.named_parameters()
               if name.startswith(source_prefixes))
    parameters = training_module._trainable_parameters(model)
    before = [param.detach().clone() for param in parameters]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    raw = [torch.randn(7, 4), torch.randn(5, 4)]
    targets = torch.tensor([[2, 3, 6], [2, 3, 4]])
    desired = torch.randn(2, 3, 4)
    for _ in range(2):
        optimizer.zero_grad()
        output = model.forward_episode(raw, [7, 5], targets)
        (output - desired).square().mean().backward()
        assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in parameters)
        optimizer.step()
    assert any(not torch.equal(param, initial) for param, initial in zip(parameters, before))
    for name, initial in frozen.items():
        torch.testing.assert_close(model.state_dict()[name], initial, rtol=0, atol=0)


def test_short_only_frozen_training_has_clear_error(cpu_memory, training_module, tmp_path):
    _, path = source_checkpoint(tmp_path)
    model, _ = checkpoints.build_training_encoder(migration_config(path, fusion_mode="short_only"), "cpu")
    with pytest.raises(ValueError, match="evaluation baseline"):
        training_module._trainable_parameters(model)


def test_memory_logs_use_real_frames_and_nonempty_history_counts(training_module):
    encoder = SimpleNamespace(memory=object())
    stats = {
        "num_frames": 7, "num_history_frames": 5,
        "input_norm_sum": torch.tensor(14.), "pre_normalization_norm_sum": torch.tensor(21.),
        "branch_weight_sum": torch.tensor([1.4, 5.6]), "branch_cosine_distance_sum": torch.tensor(3.5),
        "current_component_norm_sum": torch.tensor(20.), "history_component_norm_sum": torch.tensor(10.),
        "history_injection_ratio_sum": torch.tensor(3.),
    }
    metrics = training_module._memory_log(encoder, stats)
    assert metrics["memory/real_frame_count"] == 7
    assert metrics["memory/nonempty_history_count"] == 5
    assert metrics["memory/long_weight_mean"] == pytest.approx(.2)
    assert metrics["memory/branch_cosine_distance_mean"] == pytest.approx(.5)
    assert metrics["memory/current_component_norm_mean"] == pytest.approx(4.)
    assert metrics["memory/history_component_norm_mean"] == pytest.approx(2.)
    assert metrics["memory/history_injection_ratio_mean"] == pytest.approx(.6)
    stats["num_history_frames"] = 0
    metrics = training_module._memory_log(encoder, stats)
    assert metrics["memory/nonempty_history_count"] == 0
    for name in ("current_component_norm", "history_component_norm", "history_injection_ratio"):
        assert f"memory/{name}_mean" not in metrics


def test_save_helper_records_current_run_without_network(cpu_memory, training_module, monkeypatch, tmp_path):
    model = make_encoder(embedding_dim=4)
    monkeypatch.setattr(training_module, "checkpoint_dict", checkpoints.checkpoint_dict)
    monkeypatch.setattr(training_module.wandb, "run", SimpleNamespace(
        id="run-42", name="unit-test", project="project", entity="entity"))
    path = tmp_path / "saved.pt"
    training_module._save_encoder_checkpoint(model, path, epoch=2000)
    payload = torch.load(path, weights_only=True)
    assert payload["training_provenance"] == {
        "run_id": "run-42", "run_name": "unit-test", "project": "project", "entity": "entity", "epoch": 2000,
    }


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("mamba_ssm") is None,
                    reason="Official Mamba2 and CUDA required")
def test_official_cuda_real_hf1_migration_and_video_stream():
    """Numerical acceptance on the fixed HF1 source, without training or uploads."""
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "outputs/mamba2_normfix_v1_20261012/manifest.json"
    if not manifest_path.is_file():
        pytest.skip("The recorded HF1 source manifest is unavailable")
    entries = json.loads(manifest_path.read_text())
    entry = next(item for item in entries if item["name"] == "on_hf1_hs1")
    config_path = Path(entry["configs"]) / "train.yaml"
    if not config_path.is_file():
        pytest.skip("The HF1 source configuration snapshot is unavailable")
    source_config = yaml.safe_load(config_path.read_text())
    candidates = sorted(Path(source_config["checkpoint_dir"]).glob("*/encoder_epoch020000.pt"))
    if not candidates:
        pytest.skip("The HF1 checkpoint named by the manifest is unavailable")
    assert len(candidates) == 1, "HF1 source must be unambiguous"
    path = candidates[0]
    source = checkpoints.build_encoder_from_checkpoint(path, "cuda").eval()
    config = migration_config(path, fusion_mode="short_only",
                              visual_chunk_size=source.memory_config["visual_chunk_size"])
    migrated, _ = checkpoints.build_training_encoder(config, "cuda")
    migrated.eval()
    assert not any(param.requires_grad for param in migrated.parameters())
    torch.manual_seed(123)
    raw = [1000 * torch.randn(n, source.embedding_dim, device="cuda") for n in (257, 17)]
    errors = {}
    with torch.no_grad():
        expected = source.forward_sequence(raw, [257, 17])
        actual = migrated.forward_sequence(raw, [257, 17])
        errors["source_short_max_abs"] = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)

        # Exercise real local features across the source context-stride boundary,
        # then compare the newly fused model's full and online paths.
        video_path = root / "datasets/processed/pick_mango_vid-4vid_valid.h5"
        if video_path.is_file():
            with h5py.File(video_path, "r") as file:
                video = file["videos"][sorted(file["videos"])[0]]["frames"]
                frames = torch.from_numpy(video[:source.context_stride + 3]).permute(0, 3, 1, 2).cuda()
            source_local = source.encode_local_episode(frames)
            migrated_local = migrated.encode_local_episode(frames)
            errors["local_max_abs"] = (migrated_local - source_local).abs().max().item()
            torch.testing.assert_close(migrated_local, source_local, rtol=1e-6, atol=1e-5)
            migrated.memory.fusion_mode = "learned"
            migrated.memory.configure_trainability()
            full = migrated.forward_sequence([migrated_local], [len(frames)])[0]
            state = migrated.init_stream_state(1)
            online = torch.stack([migrated.encode_step(frame.unsqueeze(0), state)[0][0]
                                  for frame in frames])
            errors["full_stream_max_abs"] = (full - online).abs().max().item()
            errors["video_frames"] = len(frames)
            torch.testing.assert_close(online, full, rtol=2e-3, atol=5e-4)
        else:
            errors["video_check"] = "dataset unavailable"
    print("HF1_MIGRATION_NUMERICS", json.dumps(errors, sort_keys=True))
