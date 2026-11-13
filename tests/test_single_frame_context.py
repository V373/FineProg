"""Single-frame local inputs across training, caches and checkpoint reloads."""

import copy

import pytest
import torch
from torch.nn import functional as F
import yaml

from test_streaming_mamba2 import cpu_memory, tiny_local  # Reuse cheap CPU fixtures.
from fineprog.models.encoder import TCCEncoder
from fineprog.utils import encoder_checkpoint as checkpoints


def local_encoder(**kwargs):
    return TCCEncoder(**{
        "pretrained": False, "clip_len": 3, "context_stride": 2,
        "embedding_dim": 4, **kwargs,
    })


def memory_encoder(**kwargs):
    memory = {"enabled": True, "history_frames": 1, "use_residual": False,
              "normalize_input": True, "visual_chunk_size": 2,
              **kwargs.pop("memory_config", {})}
    return local_encoder(memory_config=memory, **kwargs)


def video(length=7):
    # Distinct frame content makes history mistakes visible beyond the first frame.
    return torch.rand(length, 3, 224, 224) + torch.arange(length)[:, None, None, None] / 10


def contexts(frames, stride=2):
    times = torch.arange(len(frames))
    return frames[torch.stack(((times - stride).clamp_min(0), times), dim=-1)].unsqueeze(0)


def test_flag_defaults_and_explicit_override_train_yaml(tiny_local, tmp_path):
    assert local_encoder().single_frame_context is False
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump({"single_frame_context": True, "pretrained": False}))
    assert local_encoder(train_config_path=path).single_frame_context is True
    assert local_encoder(train_config_path=path, single_frame_context=False).single_frame_context is False
    path.write_text(yaml.safe_dump({"single_frame_context": False, "pretrained": False}))
    assert local_encoder(train_config_path=path, single_frame_context=True).single_frame_context is True


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
@pytest.mark.parametrize("from_yaml", [False, True])
def test_flag_requires_boolean(tiny_local, tmp_path, value, from_yaml):
    kwargs = {"single_frame_context": value}
    if from_yaml:
        path = tmp_path / "train.yaml"
        path.write_text(yaml.safe_dump(kwargs))
        kwargs = {"train_config_path": path}
    with pytest.raises(ValueError, match="single_frame_context"):
        local_encoder(**kwargs)


@pytest.mark.parametrize("enabled", [False, True])
def test_context_replacement_precedes_backbone_and_preserves_input(tiny_local, enabled):
    model = local_encoder(single_frame_context=enabled, train_base="only_bn").train()
    frames = torch.rand(1, 3, 2, 3, 224, 224)
    original = frames.clone()
    observed = []
    hook = model.backbone.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone()))
    model(frames)
    hook.remove()
    expected = frames[:, :, -1:].expand_as(frames) if enabled else frames
    torch.testing.assert_close(observed[0], expected.reshape(-1, 3, 224, 224), rtol=0, atol=0)
    torch.testing.assert_close(frames, original, rtol=0, atol=0)
    assert model.backbone.bn.num_batches_tracked.item() == 1
    assert model.context_size == 2 and model.context_stride == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_past_context_invariance_and_gradients(tiny_local, enabled):
    model = local_encoder(single_frame_context=enabled).eval()
    frames = torch.rand(1, 3, 2, 3, 224, 224, requires_grad=True)
    altered = frames.detach().clone()
    altered[:, :, 0].add_(10)
    actual = model(frames)
    changed = model(altered)
    if enabled:
        torch.testing.assert_close(actual, changed, rtol=0, atol=0)
    else:
        assert not torch.allclose(actual, changed)
    actual.square().sum().backward()
    assert frames.grad[:, :, -1].abs().sum() > 0
    if enabled:
        assert frames.grad[:, :, 0].count_nonzero() == 0
    else:
        assert frames.grad[:, :, 0].abs().sum() > 0


@pytest.mark.parametrize("normalization", ["none", "l2"])
def test_rgb_cached_features_and_full_episode_agree(tiny_local, normalization):
    model = local_encoder(single_frame_context=True, embedding_normalization=normalization,
                          memory_config={"visual_chunk_size": 2}).eval()
    frames = video()
    grouped = contexts(frames)
    with torch.no_grad():
        raw = model.encode_local_raw(grouped)[0]
        episode = model.encode_local_episode(frames)
        frame_feats = model.backbone(frames)
        cached_episode = model.encode_local_episode(frame_feats, from_feats=True)
        grouped_feats = contexts(frame_feats)
        before = grouped_feats.clone()
        cached = model.forward_from_feats(grouped_feats)[0]
        result = model(grouped[:, :model.clip_len], return_backbone_feats=True)
    torch.testing.assert_close(raw, episode)
    torch.testing.assert_close(raw, cached_episode)
    torch.testing.assert_close(cached, model._apply_embedding_normalization(raw))
    torch.testing.assert_close(result["embeddings"][0], cached[:model.clip_len])
    torch.testing.assert_close(result["grouped_backbone_feats"][:, :, 0],
                               result["grouped_backbone_feats"][:, :, 1])
    torch.testing.assert_close(grouped_feats, before, rtol=0, atol=0)


def test_stream_matches_full_episode_and_selective_reset(cpu_memory):
    model = memory_encoder(single_frame_context=True, embedding_normalization="l2").eval()
    frames = torch.stack((video(), video()))
    with torch.no_grad():
        raw = [model.encode_local_episode(x) for x in frames]
        full = model.forward_sequence(raw, [7, 7])
        state = model.init_stream_state(2)
        online = torch.stack([model.encode_step(frames[:, t], state)[0] for t in range(7)], dim=1)
        torch.testing.assert_close(online, full)
        continuing = copy.deepcopy(state)
        model.reset_stream_state(state, [True, False])
        next_frames = torch.rand(2, 3, 224, 224)
        expected_continuing = model.encode_step(next_frames, continuing)[0]
        expected_fresh = model.encode_step(next_frames[:1], model.init_stream_state(1))[0]
        actual = model.encode_step(next_frames, state)[0]
    torch.testing.assert_close(actual[:1], expected_fresh)
    torch.testing.assert_close(actual[1:], expected_continuing[1:])
    assert state["steps"].tolist() == [1, 8]


@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_roundtrip_preserves_flag_and_weights(tiny_local, tmp_path, enabled):
    source = local_encoder(single_frame_context=enabled).eval()
    payload = checkpoints.checkpoint_dict(source)
    assert payload["encoder_config"]["single_frame_context"] is enabled
    # The flag is metadata: the model's parameter/buffer schema stays compatible.
    assert source.state_dict().keys() == local_encoder().state_dict().keys()
    path = tmp_path / "local.pt"
    torch.save(payload, path)
    restored = checkpoints.build_encoder_from_checkpoint(
        path, "cpu", {"single_frame_context": not enabled}).eval()
    existing = local_encoder(single_frame_context=not enabled).eval()
    checkpoints.load_checkpoint(existing, path, "cpu")
    assert restored.single_frame_context is enabled
    assert existing.single_frame_context is enabled
    frames = contexts(video(3))
    with torch.no_grad():
        torch.testing.assert_close(source(frames), restored(frames), rtol=0, atol=0)
        torch.testing.assert_close(source(frames), existing(frames), rtol=0, atol=0)


@pytest.mark.parametrize("metadata", ["old", "missing", "none"])
def test_old_metadata_defaults_false_when_loading_existing_encoder(tiny_local, tmp_path, metadata):
    source = local_encoder()
    payload = checkpoints.checkpoint_dict(source)
    payload["encoder_config"].pop("single_frame_context")
    fallback = dict(payload["encoder_config"], single_frame_context=True)
    if metadata == "missing":
        payload.pop("encoder_config")
    elif metadata == "none":
        payload["encoder_config"] = None
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    restored = checkpoints.build_encoder_from_checkpoint(path, "cpu", fallback)
    existing = local_encoder(single_frame_context=True)
    checkpoints.load_checkpoint(existing, path, "cpu")
    assert restored.single_frame_context is False
    assert existing.single_frame_context is False


@pytest.mark.parametrize("source_flag", [False, True])
@pytest.mark.parametrize("train_base,train_embedding", [("frozen", False), ("only_bn", True)])
def test_frozen_and_joint_stages_inherit_source_context(
        cpu_memory, tmp_path, source_flag, train_base, train_embedding):
    source = local_encoder(single_frame_context=source_flag, embedding_normalization="l2").eval()
    path = tmp_path / "source.pt"
    torch.save(checkpoints.checkpoint_dict(source), path)
    model, resolved = checkpoints.build_training_encoder({
        "single_frame_context": not source_flag, "context_stride": 99,
        "train_base": train_base, "train_embedding": train_embedding,
        "memory": {"enabled": True, "local_checkpoint_path": str(path),
                   "history_frames": 1, "use_residual": False, "normalize_input": True},
    }, "cpu")
    assert model.single_frame_context is resolved["single_frame_context"] is source_flag
    assert model.context_stride == source.context_stride
    model.eval()
    frames = video(3)
    with torch.no_grad():
        torch.testing.assert_close(model.encode_local_episode(frames),
                                   source.encode_local_episode(frames), rtol=0, atol=0)
    final_path = tmp_path / "memory.pt"
    torch.save(checkpoints.checkpoint_dict(model), final_path)
    path.unlink()
    restored = checkpoints.build_encoder_from_checkpoint(final_path, "cpu").eval()
    assert restored.single_frame_context is source_flag
    assert restored.memory.history_frames == 1 and not restored.memory.use_residual
    with torch.no_grad():
        original_raw = model.encode_local_episode(frames)
        restored_raw = restored.encode_local_episode(frames)
        torch.testing.assert_close(model.forward_sequence([original_raw], [3]),
                                   restored.forward_sequence([restored_raw], [3]), rtol=0, atol=0)


@pytest.mark.parametrize("train_base,train_embedding", [("frozen", False), ("only_bn", True)])
def test_two_updates_freezing_and_visual_checkpoint_equivalence(cpu_memory, train_base, train_embedding):
    model = memory_encoder(single_frame_context=True, embedding_normalization="l2",
                           train_base=train_base, train_embedding=train_embedding,
                           memory_config={"freeze_bn_stats": False}).train()
    plain = copy.deepcopy(model)
    plain.memory_config["checkpoint_visual"] = False
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizers = [torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
                  for m in (model, plain)]
    frames = video(5)
    target = torch.linspace(-0.8, 0.9, 20).reshape(1, 5, 4)
    losses = []
    for step in range(2):
        outputs = []
        for current, optimizer in zip((model, plain), optimizers):
            optimizer.zero_grad(set_to_none=True)
            raw = current.encode_local_episode(frames)
            output = current.forward_sequence([raw], [5])
            loss = F.mse_loss(output, target)
            assert torch.isfinite(loss)
            loss.backward()
            outputs.append(output.detach())
            if current is model:
                losses.append(loss.item())
            for parameter in current.parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                else:
                    assert parameter.grad is None
        torch.testing.assert_close(*outputs)
        for p, q in zip(model.parameters(), plain.parameters()):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad)
        for optimizer in optimizers:
            optimizer.step()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, plain.state_dict()[key])
        assert model.backbone.bn.num_batches_tracked.item() == (
            3 * (step + 1) if train_base == "only_bn" else 0)
    assert losses[0] != losses[1]
    after = model.state_dict()
    torch.testing.assert_close(after["backbone.conv.weight"], before["backbone.conv.weight"], rtol=0, atol=0)
    assert not torch.equal(after["memory.proj.weight"], before["memory.proj.weight"])
    if train_base == "frozen":
        for key in before:
            if not key.startswith("memory."):
                torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    else:
        assert not torch.equal(after["backbone.bn.weight"], before["backbone.bn.weight"])
        assert not torch.equal(after["temporal_embedder.proj.weight"], before["temporal_embedder.proj.weight"])
        assert not torch.equal(after["backbone.bn.running_mean"], before["backbone.bn.running_mean"])
