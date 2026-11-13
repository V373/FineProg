"""Offline replay tests: short unchanged, long replayed, no runtime-mode leaks."""

import copy
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from test_streaming_mamba2 import cpu_memory, tiny_local  # noqa: F401
from test_cross_attention_integration import encoder
from extract_embeddings import save_embeddings_h5
from utils.encoder_checkpoint import _checkpoint_sha256
from utils.temporal_intervention import make_history_mask, replay_intervention


def test_replay_preserves_short_and_original_modes(cpu_memory):
    model = encoder().eval()
    raw = torch.randn(8, 4)
    short_outputs = []
    hook = model.memory.proj.register_forward_hook(
        lambda module, inputs, output: short_outputs.append(output.detach().clone()))
    baseline = replay_intervention(model, raw)
    allowed, metadata = make_history_mask(baseline["diagnostics"], "top_mass", interval_frames=2)
    changed = replay_intervention(model, raw, history_mask=allowed)
    uniform = replay_intervention(model, raw, attention_mode="uniform", fusion_mode="fixed")
    hook.remove()
    torch.testing.assert_close(short_outputs[0], short_outputs[2], rtol=0, atol=0)
    torch.testing.assert_close(short_outputs[0], short_outputs[4], rtol=0, atol=0)
    first_affected = metadata["start"] + 1
    np.testing.assert_array_equal(changed["embeddings"][:first_affected], baseline["embeddings"][:first_affected])
    assert not np.array_equal(changed["embeddings"][first_affected:], baseline["embeddings"][first_affected:])
    assert not changed["diagnostics"]["attention_valid"][:, metadata["start"]:metadata["end"]].any()
    np.testing.assert_allclose(uniform["diagnostics"]["branch_weights"], np.tile([.1, .9], (8, 1)))
    assert model.memory.attention_mode == model.memory_config["attention_mode"] == "learned"
    assert model.memory.fusion_mode == model.memory_config["fusion_mode"] == "learned"
    # Replaying a new run has not contaminated a persistent online state.
    again = replay_intervention(model, raw)
    np.testing.assert_array_equal(again["embeddings"], baseline["embeddings"])


def test_random_interval_matches_length_and_seed_and_empty_history(cpu_memory):
    result = replay_intervention(encoder().eval(), torch.randn(9, 4))
    top_mask, top = make_history_mask(result["diagnostics"], "top_mass", interval_frames=2)
    random_mask, random = make_history_mask(result["diagnostics"], "random_matched", interval_frames=2, seed=17)
    repeated, _ = make_history_mask(result["diagnostics"], "random_matched", interval_frames=2, seed=17)
    np.testing.assert_array_equal(random_mask, repeated)
    assert (~top_mask).sum() == (~random_mask).sum()
    assert top["start"] != random["start"]
    mask, info = make_history_mask(result["diagnostics"], "top_mass", query_step=0)
    assert mask is None and not info["available"]
    mask, info = make_history_mask(result["diagnostics"], "random_matched", query_step=1)
    assert mask is None and not info["available"]


def test_invalid_replay_restores_modes(cpu_memory):
    model = encoder().eval()
    with pytest.raises(ValueError, match="inputs"):
        replay_intervention(model, torch.randn(4, 5), attention_mode="none", fusion_mode="long_only")
    assert model.memory.attention_mode == model.memory.fusion_mode == "learned"
    with pytest.raises(ValueError, match="boolean"):
        replay_intervention(model, torch.randn(4, 4), history_mask=[1, 1, 0, 1])


def test_intervention_entrypoint_writes_fixed_gaussian_results(cpu_memory, tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_temporal_retrieval.py"
    spec = importlib.util.spec_from_file_location("intervention_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    model = encoder().eval()
    frames = torch.randint(0, 256, (6, 3, 224, 224), dtype=torch.uint8)
    raw = model.encode_local_episode(frames)
    reference = replay_intervention(model, raw)
    source = tmp_path / "frames.h5"
    with h5py.File(source, "w") as file:
        group = file.create_group("videos/a")
        group.create_dataset("frames", data=frames.permute(0, 2, 3, 1).numpy())
        group.create_dataset("keyframe_labels", data=[-1, -1, 0, -1, -1, 1])
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    query = tmp_path / "query.h5"
    entry = {"video_id": "a", "embeddings": reference["embeddings"],
             "target_steps": np.arange(6), "seq_len": 6, "action_id": 0}
    save_embeddings_h5([entry], str(query), "l2", checkpoint_path=checkpoint, processed_h5_path=source)
    with h5py.File(query, "r") as file:
        np.testing.assert_array_equal(file["videos/a/keyframe_labels"], [-1, -1, 0, -1, -1, 1])
    gaussian = tmp_path / "gaussian.h5"
    with h5py.File(gaussian, "w") as file:
        file.attrs["checkpoint_sha256"] = _checkpoint_sha256(checkpoint)
    fitted = {"num_bins": 2, "embedding_dim": 4, "input_embedding_dim": 4, "enable_pca": False,
              "bin_means": np.array([[0.] * 4, [1.] * 4]),
              "cholesky_factors": np.stack([np.eye(4)] * 2),
              "bin_log_determinants": np.zeros(2), "bin_progress_values": np.array([0., 1.])}
    monkeypatch.setattr(cli, "build_encoder_from_checkpoint", lambda *a, **k: model)
    monkeypatch.setattr(cli, "_read_gaussian_model", lambda *a: fitted)
    cfg = {"gaussian_model_h5_path": str(gaussian), "nonexpert_h5_path": str(query),
           "enable_calibration": False, "save_posterior": True, "posterior_temperature": 1.,
           "temporal_diagnostics": {"enabled": False, "video_ids": ["a"]},
           "temporal_intervention": {"enabled": True, "attention_mode": "none"}}
    destination = tmp_path / "output"
    result = cli.run_intervention(cfg, destination)
    assert result["local_status"] == "complete"
    assert (destination / "a.png").is_file()
    with h5py.File(destination / "interventions.h5", "r") as file:
        assert file["videos/a/intervention/posterior"].shape == (6, 2)
        assert not file["videos/a/intervention/temporal_diagnostics/attention"][:].any()
        np.testing.assert_allclose(file["videos/a/original/posterior"][:].sum(-1), 1)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["upload_status"] == "not_requested"
    assert model.memory.attention_mode == "learned"
