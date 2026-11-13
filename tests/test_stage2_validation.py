"""Fixed semantic validation must not perturb training or omit rare G frames."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_streaming_mamba2 import cpu_memory, tiny_local
from fineprog.models.encoder import TCCEncoder
from fineprog.utils.stage2_validation import Stage2Validator, _summarize


def dataset(labels_by_id):
    rng = np.random.default_rng(18)
    return SimpleNamespace(
        video_ids=list(labels_by_id),
        labels={vid: np.asarray(labels, dtype=np.int64) for vid, labels in labels_by_id.items()},
        _frames_cache={vid: {"frames": rng.integers(0, 256, (len(labels), 224, 224, 3), dtype=np.uint8)}
                       for vid, labels in labels_by_id.items()},
    )


def model(memory):
    encoder = TCCEncoder(pretrained=False, clip_len=3, context_size=2, context_stride=20,
                         embedding_dim=4, embedding_normalization="l2", single_frame_context=True,
                         train_base="only_bn", train_embedding=True,
                         memory_config={"enabled": memory, "history_frames": 1,
                                        "use_residual": False, "normalize_input": True,
                                        "freeze_bn_stats": False, "visual_chunk_size": 20})
    encoder.train()
    encoder.configure_trainability()
    return encoder


@pytest.mark.parametrize("memory", [False, True])
def test_validation_is_fixed_full_episode_and_preserves_training(cpu_memory, tmp_path, memory):
    experts = dataset({"expert_a": [0] * 7 + [1] * 7 + [2] * 9,
                       "expert_b": [0] * 9 + [1] * 8 + [2] * 7})
    queries = dataset({"heldout_a": [0] * 6 + [1] * 8 + [2] * 9,
                       "heldout_b": [0] * 21})
    encoder = model(memory)
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    buffers = {k: v.clone() for k, v in encoder.named_buffers()}
    flags = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    validator = Stage2Validator(experts, queries)
    first = validator.evaluate(encoder, tmp_path / "one", 0)
    second = validator.evaluate(encoder, tmp_path / "two", 25)
    assert first["g_count"] == 9 and first["r_count"] == 27
    assert first["valid_queries"] == 36 and first["reference_count"] == 24
    assert first["reference_video_ids"] == ["expert_a", "expert_b"]
    assert first["validation_video_ids"] == ["heldout_a", "heldout_b"]
    assert first["per_video"]["heldout_b"]["g_loss"] is None
    assert first["set_loss"] == second["set_loss"]
    assert encoder.training and encoder.backbone.bn.training
    assert flags == (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    current_numpy = np.random.get_state()
    assert numpy_state[0] == current_numpy[0]
    np.testing.assert_array_equal(numpy_state[1], current_numpy[1])
    assert numpy_state[2:] == current_numpy[2:]
    for name, value in encoder.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    assert all(p.grad is None for p in encoder.parameters())
    if memory:
        assert encoder.memory.block.lengths == [23, 24, 23, 21] * 2
    with np.load(tmp_path / "one" / "frames.npz") as arrays:
        assert arrays["heldout_a_probs"].shape == (23, 3)
        np.testing.assert_allclose(arrays["heldout_a_probs"].sum(1), 1, rtol=1e-5)
        assert np.isnan(arrays["heldout_a_set_nll"][6:14]).all()
        assert np.isfinite(arrays["heldout_a_set_nll"][-9:]).all()
        assert set(arrays["ref_video_ids"]) == {"expert_a", "expert_b"}


def test_frame_weighted_summary_separates_rare_goal():
    labels = np.array([0, 0, 1, 2])
    probs = np.array([[0.8, 0.1, 0.1], [0.6, 0.2, 0.2], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    nll = np.array([-np.log(0.8), -np.log(0.6), np.nan, -np.log(0.8)])
    result = _summarize(labels, probs, nll)
    assert result["set_loss"] == pytest.approx(np.nanmean(nll))
    assert result["positive_mass"] == pytest.approx((0.8 + 0.6 + 0.8) / 3)
    assert result["r_loss"] == pytest.approx((-np.log(0.8) - np.log(0.6)) / 2)
    assert result["g_loss"] == pytest.approx(-np.log(0.8))
    assert result["r_count"] == 2 and result["g_count"] == 1


def test_eval_mode_and_precision_restored_after_failure(cpu_memory, tmp_path, monkeypatch):
    experts = dataset({"expert": [0, 1, 2]})
    queries = dataset({"heldout": [0, 1, 2]})
    encoder = model(False).eval()
    validator = Stage2Validator(experts, queries)
    flags = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setattr(validator, "_encode", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("test")))
    with pytest.raises(RuntimeError, match="test"):
        validator.evaluate(encoder, tmp_path, 0)
    assert not encoder.training and not encoder.backbone.bn.training
    assert flags == (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
