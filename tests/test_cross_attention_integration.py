"""Encoder/extraction contracts; real Mamba numerics are tested separately on CUDA."""

import copy
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from test_streaming_mamba2 import cpu_memory, tiny_local, make_encoder  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_embeddings import extract_embeddings, save_embeddings_h5
from fineprog.utils import encoder_checkpoint as checkpoints


def encoder(**memory):
    return make_encoder(
        embedding_dim=4, embedding_normalization="l2", train_base="frozen",
        train_embedding=False, memory_config={"mode": "cross_attention",
                                             "normalize_input": True,
                                             "use_residual": False, **memory})


def test_disabled_new_mode_remains_a_local_encoder(cpu_memory):
    model = encoder(enabled=False).eval()
    assert model.memory is None
    result = model(torch.randn(1, 3, 2, 3, 224, 224))
    assert result.shape == (1, 3, 4)
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(1, 3))


def test_full_fuse_once_then_gather_and_stats(cpu_memory):
    model = encoder().eval()
    raw = [torch.randn(3, 4), torch.randn(1, 4)]
    targets = torch.tensor([[2, 0, 2], [0, 0, 0]])
    full = model.forward_episode(raw, [3, 1])
    calls = {"block": 0, "proj": 0, "gate": 0}
    hooks = []
    for key in calls:
        def count(module, args, output, key=key):
            calls[key] += 1
        hooks.append(getattr(model.memory, key).register_forward_hook(count))
    stats, diagnostic = {}, {}
    gathered = model.forward_episode(raw, [3, 1], targets,
                                     memory_norm_stats=stats, diagnostics=diagnostic)
    for hook in hooks:
        hook.remove()
    assert calls == {"block": 2, "proj": 2, "gate": 1}
    torch.testing.assert_close(gathered, full[torch.arange(2)[:, None], targets], rtol=0, atol=0)
    assert stats["num_frames"] == 4
    assert stats["num_history_frames"] == 2
    assert diagnostic["attention"].shape == (2, 3, 3)
    assert torch.count_nonzero(full[1, 1:]) == 0
    for value in stats.values():
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad and value.numel() <= 2


def test_logging_preserves_gradients_and_no_future_dependency(cpu_memory):
    model = encoder().train()
    twin = copy.deepcopy(model)
    raw = [torch.randn(7, 4), torch.randn(4, 4)]
    targets = torch.tensor([[2, 6], [1, 3]])
    direction = torch.randn(2, 2, 4)
    baseline = model.forward_episode(raw, [7, 4], targets)
    observed = twin.forward_episode(raw, [7, 4], targets,
                                    memory_norm_stats={}, diagnostics={})
    torch.testing.assert_close(baseline, observed, rtol=0, atol=0)
    (baseline * direction).sum().backward()
    (observed * direction).sum().backward()
    for (name, param), (_, other) in zip(model.named_parameters(), twin.named_parameters()):
        if param.grad is None:
            assert other.grad is None, name
        else:
            torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0, msg=name)
    changed = [v.clone() for v in raw]
    changed[0][3:] += 100
    with torch.no_grad():
        before = model.forward_episode(raw, [7, 4])
        after = model.forward_episode(changed, [7, 4])
    torch.testing.assert_close(before[0, :3], after[0, :3], rtol=0, atol=0)


def test_encoder_streaming_diagnostics_and_partial_reset(cpu_memory):
    model = encoder().eval()
    frames = torch.randn(2, 7, 3, 224, 224)
    raw = [model.encode_local_episode(video) for video in frames]
    full = model.forward_episode(raw, [7, 7])
    state, observed_state = model.init_stream_state(2), model.init_stream_state(2)
    outputs = []
    for t in range(7):
        value, state = model.encode_step(frames[:, t], state)
        observed, observed_state = model.encode_step(
            frames[:, t], observed_state, memory_norm_stats={}, diagnostics={})
        torch.testing.assert_close(value, observed, rtol=0, atol=0)
        outputs.append(value.clone())
    torch.testing.assert_close(torch.stack(outputs, 1), full, rtol=1e-5, atol=1e-6)
    for key in state:
        if state[key] is not None:
            torch.testing.assert_close(state[key], observed_state[key], rtol=0, atol=0)
    continuing = copy.deepcopy(state)
    model.reset_stream_state(state, [True, False])
    obs = torch.randn(2, 3, 224, 224)
    reset_result, state = model.encode_step(obs, state)
    fresh_result, _ = model.encode_step(obs[:1], model.init_stream_state(1))
    continued, _ = model.encode_step(obs, continuing)
    torch.testing.assert_close(reset_result[:1], fresh_result, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reset_result[1:], continued[1:], rtol=0, atol=0)


def test_selected_extraction_and_h5_roundtrip(cpu_memory, tmp_path):
    model = encoder().eval()
    batch = {"video_id": ["a"], "frames_full": [torch.randn(4, 3, 224, 224)],
             "target_steps": torch.arange(4)[None], "seq_len": torch.tensor([4]),
             "action_id": torch.tensor([0])}
    plain = extract_embeddings(model, [batch], torch.device("cpu"))
    observed = extract_embeddings(model, [batch], torch.device("cpu"),
                                  temporal_diagnostics={"enabled": True, "video_ids": ["a"]})
    np.testing.assert_array_equal(plain[0]["embeddings"], observed[0]["embeddings"])
    path = tmp_path / "embeddings.h5"
    save_embeddings_h5(observed, str(path), "l2", checkpoints.encoder_config(model),
                       training_provenance={"run_id": "test-run", "epoch": 2})
    with h5py.File(path) as handle:
        group = handle["videos/a/temporal_diagnostics"]
        np.testing.assert_array_equal(group["attention"], observed[0]["temporal_diagnostics"]["attention"])
        np.testing.assert_array_equal(group["frame_indices"], np.arange(4))
        assert json.loads(handle.attrs["training_provenance"])["run_id"] == "test-run"
    with pytest.raises(ValueError, match="Unknown"):
        extract_embeddings(model, [batch], torch.device("cpu"),
                           temporal_diagnostics={"enabled": True, "video_ids": ["missing"]})
    with pytest.raises(ValueError, match="nonempty"):
        extract_embeddings(model, [batch], torch.device("cpu"),
                           temporal_diagnostics={"enabled": True, "video_ids": []})
    skipped = {**batch, "target_steps": torch.tensor([[0, 2]])}
    with pytest.raises(ValueError, match="sample_all_stride"):
        extract_embeddings(model, [skipped], torch.device("cpu"),
                           temporal_diagnostics={"enabled": True, "video_ids": ["a"]})
    # Validation happens before opening/replacing an existing output artifact.
    damaged = copy.deepcopy(observed)
    damaged[0]["temporal_diagnostics"]["attention"][0, 0] = 1
    before = path.read_bytes()
    with pytest.raises(ValueError, match="masked"):
        save_embeddings_h5(damaged, str(path), "l2")
    assert path.read_bytes() == before


@pytest.mark.parametrize("fusion,weights", [
    ("fixed", [.1, .9]), ("short_only", [0, 1]), ("long_only", [1, 0]),
])
def test_legacy_video_helper_uses_actual_ablation_weights(cpu_memory, fusion, weights):
    from fineprog.algos.eval_task.tcc_eval_tasks.task_gaussian_progress_pred import _compute_video_branch_weights
    model = encoder(fusion_mode=fusion).eval()
    frames = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8)
    raw = model.encode_local_episode(frames)
    with torch.no_grad():
        expected = model.forward_episode([raw], [4])[0].numpy()
    actual = _compute_video_branch_weights(model, frames.permute(0, 2, 3, 1).numpy(), expected, np.arange(4))
    np.testing.assert_allclose(actual, np.tile(weights, (4, 1)))


def test_existing_video_renderer_reads_saved_weights_without_encoder(tmp_path, monkeypatch):
    from fineprog.algos.eval_task.tcc_eval_tasks import task_gaussian_progress_pred as pred
    frames_path, embeddings_path, prediction_path = [tmp_path / name for name in ("frames.h5", "emb.h5", "pred.h5")]
    with h5py.File(frames_path, "w") as file:
        file.create_dataset("videos/a/frames", data=np.zeros((3, 2, 2, 3), dtype=np.uint8))
    weights = np.tile([1., 0.], (3, 1))
    with h5py.File(embeddings_path, "w") as file:
        # No checkpoint path: a saved-weight rendering must not load an encoder.
        file.attrs["processed_h5_path"] = str(frames_path)
        file.create_dataset("videos/a/target_steps", data=np.arange(3))
        file.create_dataset("videos/a/temporal_diagnostics/branch_weights", data=weights)
    with h5py.File(prediction_path, "w") as file:
        file.attrs["nonexpert_h5_path"] = str(embeddings_path)
        file.create_dataset("model/bin_progress_values", data=[0., 1.])
        group = file.create_group("nonexperts/a")
        for key, value in {"target_steps": np.arange(3), "progress_gated": np.arange(3) / 2,
                           "posterior": np.full((3, 2), .5), "is_ood": np.zeros(3, dtype=bool),
                           "conformal_p_value": np.ones((3, 2)),
                           "gaussian_log_likelihood": np.zeros((3, 2))}.items():
            group.create_dataset(key, data=value)
    captured = []
    monkeypatch.setattr(pred, "_resolve_raw_video_path", lambda *a: (tmp_path / "dummy.mp4", 3))
    monkeypatch.setattr(pred, "_compute_video_branch_weights", lambda *a: pytest.fail("No encoder replay for saved weights"))
    monkeypatch.setattr(pred, "_save_prediction_video", lambda **kwargs: captured.append(kwargs))
    pred._save_prediction_videos(prediction_path, branch_weights=True, output_dir=tmp_path / "video")
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0]["branch_weights"], weights)
